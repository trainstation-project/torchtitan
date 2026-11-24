# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GptTrainstation: one GQA decoder mixing dense MLP and MoE layers, for test coverage.

Every layer uses GQA (CosSin RoPE, QK norm). The debug flavor repeats a
four-layer FFN pattern so that every pipeline stage of a 2-stage split sees
every FFN type:

    layer % 4 == 0: dense MLP
    layer % 4 == 1: MoE, softmax router, routed experts only
    layer % 4 == 2: MoE, sigmoid router, shared experts, aux loss
    layer % 4 == 3: dense MLP
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.config.transform import (
    ModelConfigConverter,
    validate_converter_compatibility,
)
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    CosSinRoPE,
    Embedding,
    Linear,
    RMSNorm,
    Sigmoid,
    Softmax,
)
from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.qwen3.parallelize import parallelize_qwen3
from torchtitan.protocols.model_spec import ModelSpec

from .model import GptTrainstationModel, GptTrainstationTransformerBlock

__all__ = [
    "GptTrainstationModel",
    "MuPConfig",
    "GptTrainstationTransformerBlock",
    "gpt_trainstation_configs",
    "model_registry",
]


_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}
_EPS = 1e-6


@dataclass(kw_only=True, slots=True)
class MuPConfig:
    """muP multipliers, following lm-engine.

    Token embeddings are scaled by ``m_emb``, each residual branch by
    ``m_residual``, and logits by ``1 / m_width`` (``m_width = dim / base_dim``).
    Hidden matrices init with std ``0.02 / sqrt(m_width)`` and train at
    ``lr / m_width`` (see ``config_registry.mup_adamw_optimizer``).
    """

    m_width: float
    m_emb: float | None = None
    m_residual: float | None = None


# Arbitrary (untuned) values; the debug model only needs muP exercised.
_DEBUG_MUP = MuPConfig(m_width=2.0, m_emb=12.0, m_residual=0.22)

# Hidden matrices; the embedding and LM head keep std 0.02.
_HIDDEN_STD = 0.02 / math.sqrt(_DEBUG_MUP.m_width)
_LINEAR_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.trunc_normal_, std=_HIDDEN_STD),
    "bias": nn.init.zeros_,
}
_LM_HEAD_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(
            nn.init.trunc_normal_, std=depth_scaled_std(_HIDDEN_STD, layer_id)
        ),
        "bias": nn.init.zeros_,
    }


def _experts_init(layer_id: int) -> dict[str, Callable]:
    depth_init = _depth_init(layer_id)["weight"]
    return {
        "w1_EFD": _LINEAR_INIT["weight"],
        "w2_EDF": depth_init,
        "w3_EFD": depth_init,
    }


def _norm(dim: int) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


def _moe(
    *,
    layer_id: int,
    dim: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    with_shared_experts: bool,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
) -> MoE.Config:
    """MoE in one of two styles.

    ``with_shared_experts=False``: softmax top-k router, routed experts only.
    ``with_shared_experts=True``: sigmoid top-k router, shared experts, and a
    microbatch-wise auxiliary load-balancing loss.
    """
    router = make_router_config(
        dim=dim,
        num_experts=num_experts,
        gate_param_init=_LINEAR_INIT,
        top_k=top_k,
        score_func=Sigmoid.Config() if with_shared_experts else Softmax.Config(),
        route_norm=True,
    )
    return make_moe_config(
        num_experts=num_experts,
        router=router,
        routed_experts=make_routed_experts_config(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            param_init=_experts_init(layer_id),
            comm_backend=moe_comm_backend,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
        ),
        shared_experts=(
            make_ffn_config(
                dim=dim,
                hidden_dim=hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            )
            if with_shared_experts
            else None
        ),
        aux_loss_coeff=1e-3 if with_shared_experts else None,
    )


def _debugmodel(
    attn_backend: str,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    *,
    seq_len: int,
) -> GptTrainstationModel.Config:
    dim = 256
    n_layers = 8
    vocab_size = 2048
    n_heads = 16
    head_dim = 64
    dense_hidden_dim = 768
    moe_hidden_dim = 256
    num_experts = 8
    top_k = 2

    inner_attention = get_attention_config(attn_backend)
    rope = CosSinRoPE.Config(dim=head_dim, max_context_length=seq_len, theta=1000000.0)

    layers = []
    for layer_id in range(n_layers):
        kind = layer_id % 4
        attention = make_gqa_config(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=8,
            head_dim=head_dim,
            wqkv_param_init=_LINEAR_INIT,
            wo_param_init=_depth_init(layer_id),
            inner_attention=inner_attention,
            rope=rope,
            qk_norm=_norm(head_dim),
        )

        if kind in (0, 3):
            feed_forward = make_ffn_config(
                dim=dim,
                hidden_dim=dense_hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            )
            moe = None
        else:
            feed_forward = None
            moe = _moe(
                layer_id=layer_id,
                dim=dim,
                hidden_dim=moe_hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                with_shared_experts=kind == 2,
                moe_comm_backend=moe_comm_backend,
                non_blocking_capacity_factor=non_blocking_capacity_factor,
            )

        layers.append(
            GptTrainstationTransformerBlock.Config(
                attention=attention,
                attention_norm=_norm(dim),
                ffn_norm=_norm(dim),
                feed_forward=feed_forward,
                moe=moe,
                m_residual=_DEBUG_MUP.m_residual,
            )
        )

    return GptTrainstationModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=_norm(dim),
        lm_head=Linear.Config(
            in_features=dim, out_features=vocab_size, param_init=_LM_HEAD_INIT
        ),
        layers=layers,
        m_emb=_DEBUG_MUP.m_emb,
        m_width=_DEBUG_MUP.m_width,
    )


gpt_trainstation_configs = {
    "debugmodel": (_debugmodel, 4096),
}


def _post_optimizer_build_fn(optimizers, model_parts, parallel_dims):
    """Register step pre-hooks for load balancing and aux-loss accumulators."""
    register_moe_load_balancing_hook(optimizers, model_parts, parallel_dims)
    register_aux_loss_zero_hook(optimizers, model_parts, parallel_dims)


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    get_config, max_context_len = gpt_trainstation_configs[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        seq_len=context_len,
    )
    if converters is not None:
        validate_converter_compatibility(converters)
        for c in converters:
            config = c.build().convert(config)
    return ModelSpec(
        name="gpt_trainstation",
        flavor=flavor,
        model=config,
        max_context_length=context_len,
        # Nothing model-specific happens at parallelize time: per-layer
        # sharding is set in ``update_from_config``, so Qwen3's generic
        # Decoder parallelize (AC, compile, FSDP) applies unchanged.
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=_post_optimizer_build_fn,
        # No released checkpoint, so there is no HF format to convert.
        state_dict_adapter=None,
    )
