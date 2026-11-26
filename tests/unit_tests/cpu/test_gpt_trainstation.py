# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.gpt_trainstation import gpt_trainstation_configs
from torchtitan.models.gpt_trainstation.config_registry import (
    dist_muon_optimizer,
    gpt_trainstation_debugmodel,
    mup_adamw_optimizer,
)

_SEQ_LEN = 64


class _CausalAttention(nn.Module):
    """Eager causal attention on (T, H, K) inputs.

    FlexAttention has no CPU backward, so the CPU test swaps it out.
    """

    def forward(self, q_THK, k_THK, v_THV, *, scale=None, **kwargs):
        q_HTK, k_HTK, v_HTV = (t.transpose(0, 1) for t in (q_THK, k_THK, v_THV))
        out_HTV = F.scaled_dot_product_attention(
            q_HTK,
            k_HTK,
            v_HTV,
            is_causal=True,
            scale=scale,
            enable_gqa=q_HTK.shape[0] != k_HTK.shape[0],
        )
        return out_HTV.transpose(0, 1)


def _debug_config(
    initializer_range: float = 0.02, attention_multiplier: float | None = None
):
    build_config, _ = gpt_trainstation_configs["debugmodel"]
    return build_config(
        "flex",
        seq_len=_SEQ_LEN,
        initializer_range=initializer_range,
        attention_multiplier=attention_multiplier,
    )


def _build_for_cpu(config):
    model = config.build()
    model.init_states()
    for layer in model.layers.values():
        layer.attention.inner_attention = _CausalAttention()
    return model


def test_debugmodel_mixes_every_layer_type() -> None:
    layers = _debug_config().layers
    assert len(layers) % 4 == 0
    for layer_id, layer in enumerate(layers):
        kind = layer_id % 4
        assert isinstance(layer.attention, GQAttention.Config)
        assert (layer.feed_forward is not None) == (kind in (0, 3))
        assert (layer.moe is not None) == (kind in (1, 2))
        if kind == 1:
            assert layer.moe.shared_experts is None
            assert layer.moe.router.aux_loss is None
        if kind == 2:
            assert layer.moe.shared_experts is not None
            assert layer.moe.router.aux_loss is not None


@pytest.fixture(autouse=True)
def _clean_aux_loss_state():
    """Reset AuxLoss class-level state so it does not leak into other tests."""
    yield
    AuxLoss._group_counts.clear()
    AuxLoss.group_acc.clear()
    AuxLoss._step_denominator = None


def test_forward_backward() -> None:
    torch.manual_seed(0)
    config = _debug_config()
    model = config.build()
    model.init_states()
    for layer in model.layers.values():
        layer.attention.inner_attention = _CausalAttention()

    tokens = torch.randint(0, config.vocab_size, (_SEQ_LEN,))
    AuxLoss.set_step_denominator(torch.tensor(float(_SEQ_LEN), dtype=torch.float64))
    logits = model(tokens, torch.arange(_SEQ_LEN), None)
    assert logits.shape == (_SEQ_LEN, config.vocab_size)

    logits.float().logsumexp(-1).mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name

    _, flops = config.get_nparams_and_flops(model, _SEQ_LEN)
    assert flops > 0


@pytest.mark.parametrize(
    "tp,ep,enable_sp", [(1, 1, False), (2, 2, True), (2, 4, False)]
)
def test_sharding_config_covers_every_layer(tp: int, ep: int, enable_sp: bool) -> None:
    trainer_config = gpt_trainstation_debugmodel(seq_len=_SEQ_LEN)
    trainer_config.parallelism.tensor_parallel_degree = tp
    trainer_config.parallelism.expert_parallel_degree = ep
    trainer_config.parallelism.enable_sequence_parallel = enable_sp
    model_config = trainer_config.model_spec.model
    model_config.update_from_config(config=trainer_config)
    for layer in model_config.layers:
        assert layer.attention_norm.sharding_config is not None
        assert layer.attention.wo.sharding_config is not None
        if layer.moe is not None:
            assert layer.moe.router.gate.sharding_config is not None


def test_dist_muon_covers_every_matrix() -> None:
    trainer_config = gpt_trainstation_debugmodel(seq_len=_SEQ_LEN)
    trainer_config.parallelism.expert_parallel_degree = 2
    optimizer = dist_muon_optimizer(
        trainer_config.model_spec, lr=1e-3, parallelism=trainer_config.parallelism
    )
    compute_sharding_by_fqn = optimizer.optimizer_factory_kwargs_by_name["DistMuon"][
        "compute_sharding_by_fqn"
    ]
    model = trainer_config.model_spec.model.build()
    # Every DistMuon FQN is a real parameter, and every non-norm matrix is
    # claimed. Embedding and LM head stay on AdamW.
    matrices = {
        name
        for name, param in model.named_parameters()
        if param.ndim >= 2 and not name.startswith(("tok_embeddings", "lm_head"))
    }
    assert set(compute_sharding_by_fqn) == matrices

    trainer_config.parallelism.tensor_parallel_degree = 2
    with pytest.raises(ValueError, match="tensor_parallel_degree"):
        dist_muon_optimizer(
            trainer_config.model_spec, lr=1e-3, parallelism=trainer_config.parallelism
        )


def test_mup_multipliers() -> None:
    """Embedding, residual and logit multipliers match an explicit rescaling of
    the same weights without muP."""
    mup_config = _debug_config()
    m_residual = mup_config.layers[0].m_residual
    plain_config = _debug_config()
    plain_config.m_emb = plain_config.m_width = None
    for layer in plain_config.layers:
        layer.m_residual = None

    torch.manual_seed(0)
    mup_model = _build_for_cpu(mup_config)
    plain_model = _build_for_cpu(plain_config)
    plain_model.load_state_dict(mup_model.state_dict())
    mup_model.eval()
    plain_model.eval()

    # Fold m_emb into the embedding table and m_residual into every branch's
    # output projection, so both models carry the same residual stream.
    with torch.no_grad():
        plain_model.tok_embeddings.weight.mul_(mup_config.m_emb)
        for layer in plain_model.layers.values():
            layer.attention.wo.weight.mul_(m_residual)
            ffn = layer.moe if layer.moe_enabled else layer.feed_forward
            for name, param in ffn.named_parameters():
                if name.endswith(("w2.weight", "w2_EDF")):
                    param.mul_(m_residual)
    # RMSNorm before lm_head removes the residual-stream scale, so the
    # remaining difference is exactly the 1 / m_width logit multiplier.
    tokens = torch.randint(0, mup_config.vocab_size, (_SEQ_LEN,))
    positions = torch.arange(_SEQ_LEN)
    AuxLoss.set_step_denominator(torch.tensor(float(_SEQ_LEN), dtype=torch.float64))
    with torch.no_grad():
        mup_logits = mup_model(tokens, positions, None)
        plain_logits = plain_model(tokens, positions, None)
    torch.testing.assert_close(mup_logits * mup_config.m_width, plain_logits)


@pytest.mark.parametrize("initializer_range", [0.02, 0.05])
def test_mup_init_std(initializer_range: float) -> None:
    """lm-engine init: hidden std initializer_range / sqrt(m_width), output
    projections further divided by sqrt(2 * num_layers), embedding and LM head
    initializer_range."""
    config = _debug_config(initializer_range)
    torch.manual_seed(0)
    model = config.build()
    model.init_states()
    hidden_std = initializer_range / config.m_width**0.5
    out_std = hidden_std / (2 * len(config.layers)) ** 0.5
    expected = {
        "attention.qkv_linear.wqkv.weight": hidden_std,
        "attention.wo.weight": out_std,
        "feed_forward.w13.weight": hidden_std,
        "feed_forward.w2.weight": out_std,
        "moe.router.gate.weight": hidden_std,
        "moe.routed_experts.inner_experts.w1_EFD": hidden_std,
        "moe.routed_experts.inner_experts.w3_EFD": hidden_std,
        "moe.routed_experts.inner_experts.w2_EDF": out_std,
        "moe.shared_experts.w13.weight": hidden_std,
        "moe.shared_experts.w2.weight": out_std,
    }
    params = dict(model.named_parameters())
    for suffix, std in expected.items():
        names = [n for n in params if n.endswith(suffix)]
        assert names, suffix
        for name in names:
            assert torch.std(params[name]).item() == pytest.approx(std, rel=0.1), name
    assert torch.std(model.tok_embeddings.weight).item() == pytest.approx(
        initializer_range, rel=0.05
    )
    assert torch.std(model.lm_head.weight).item() == pytest.approx(
        initializer_range, rel=0.05
    )


def test_mup_forward_backward() -> None:
    torch.manual_seed(0)
    model_spec = gpt_trainstation_debugmodel(seq_len=_SEQ_LEN).model_spec
    config = model_spec.model
    model = _build_for_cpu(config)
    tokens = torch.randint(0, config.vocab_size, (_SEQ_LEN,))
    AuxLoss.set_step_denominator(torch.tensor(float(_SEQ_LEN), dtype=torch.float64))
    logits = model(tokens, torch.arange(_SEQ_LEN), None)
    logits.float().logsumexp(-1).mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name


def test_mup_param_groups() -> None:
    """Matches lm-engine's mup.yml: norms at lr without weight decay, embedding
    and LM head at lr, everything else at lr / m_width."""
    import re

    model_spec = gpt_trainstation_debugmodel(seq_len=_SEQ_LEN).model_spec
    m_width = model_spec.model.m_width
    optimizer = mup_adamw_optimizer(model_spec, lr=1.0, weight_decay=0.1)
    no_decay, full_lr, hidden_lr = optimizer.param_groups
    assert no_decay.optimizer_kwargs["lr"] == 1.0
    assert no_decay.optimizer_kwargs["weight_decay"] == 0.0
    assert full_lr.optimizer_kwargs["lr"] == 1.0
    assert full_lr.optimizer_kwargs["weight_decay"] == 0.1
    assert full_lr.allow_empty
    assert hidden_lr.optimizer_kwargs["lr"] == 1.0 / m_width
    assert hidden_lr.optimizer_kwargs["weight_decay"] == 0.1

    model = model_spec.model.build()
    names = [name for name, _ in model.named_parameters()]
    no_decay_names = {n for n in names if re.search(no_decay.pattern, n)}
    assert no_decay_names == {
        name
        for name, param in model.named_parameters()
        if param.ndim == 1 and "norm" in name
    }
    full_lr_names = {
        n for n in names if n not in no_decay_names and re.search(full_lr.pattern, n)
    }
    assert full_lr_names == {"tok_embeddings.weight", "lm_head.weight"}

    model_spec.model.m_width = None
    with pytest.raises(ValueError, match="muP"):
        mup_adamw_optimizer(model_spec, lr=1.0)


@pytest.mark.parametrize("attention_multiplier", [None, 1 / 64])
def test_attention_multiplier_reaches_inner_attention(
    attention_multiplier: float | None,
) -> None:
    """``attention_multiplier`` overrides the default ``1 / sqrt(head_dim)``."""
    config = _debug_config(attention_multiplier=attention_multiplier)
    model = _build_for_cpu(config)
    expected = 64**-0.5 if attention_multiplier is None else attention_multiplier
    seen = []
    for layer in model.layers.values():
        layer.attention.inner_attention.register_forward_hook(
            lambda mod, args, kwargs, out: seen.append(kwargs["scale"]),
            with_kwargs=True,
        )
    tokens = torch.randint(0, config.vocab_size, (_SEQ_LEN,))
    AuxLoss.set_step_denominator(torch.tensor(float(_SEQ_LEN), dtype=torch.float64))
    with torch.no_grad():
        model(tokens, torch.arange(_SEQ_LEN), None)
    assert seen == [expected] * len(config.layers)
