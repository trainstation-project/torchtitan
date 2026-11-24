# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.models.common.attention import AttentionMasksType, GQAttention
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)


class GptTrainstationTransformerBlock(TransformerBlock):
    """GQA transformer block whose FFN type is chosen per layer.

    Exactly one of ``feed_forward`` (dense MLP) and ``moe`` is set.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        m_residual: float | None = None
        """muP residual multiplier: scales each branch output before the
        residual add. ``None`` disables it."""

    def __init__(self, config: Config):
        super().__init__()
        if (config.feed_forward is None) == (config.moe is None):
            raise ValueError(
                "GptTrainstationTransformerBlock needs exactly one of feed_forward and moe."
            )

        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

        self.m_residual = config.m_residual

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ):
        h = self.attention(self.attention_norm(x), attention_masks, positions)
        if self.m_residual is not None:
            h = h * self.m_residual
        x = x + h

        if self.moe_enabled:
            h = self.moe(self.ffn_norm(x), padding_mask_T=padding_mask)
        else:
            h = self.feed_forward(self.ffn_norm(x))
        if self.m_residual is not None:
            h = h * self.m_residual
        return x + h


class GptTrainstationModel(Decoder):
    """GQA decoder that mixes dense MLP and MoE layers in one model.

    It has no released checkpoint. Its purpose is test coverage: a single
    model whose GQA layers interleave dense MLP and MoE (with and without
    shared experts, softmax and sigmoid routers), so every
    parallelism combination can be exercised against all of them at once.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        m_emb: float | None = None
        """muP embedding multiplier applied to the token embedding output.
        ``None`` disables it."""

        m_width: float | None = None
        """muP width multiplier (``dim / base_dim``). Logits are scaled by
        ``1 / m_width``. ``None`` disables it."""

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)

            from torchtitan.models.gpt_trainstation.sharding import (
                set_gpt_trainstation_sharding_config,
            )

            parallelism = config.parallelism
            set_gpt_trainstation_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            attention_op_flops = 0
            for layer in self.layers:
                attention = layer.attention
                assert isinstance(attention, GQAttention.Config)
                head_dim = (
                    attention.head_dim
                    if attention.head_dim is not None
                    else attention.dim // attention.n_heads
                )
                attention_op_flops += quadratic_attention_flops_per_token(
                    num_heads=attention.n_heads,
                    qk_head_dim=head_dim,
                    v_head_dim=head_dim,
                    seq_len=seq_len,
                )
            return nparams, 6 * active_nparams + attention_op_flops

    def __init__(self, config: Config):
        super().__init__(config)
        self.m_emb = config.m_emb
        self.m_width = config.m_width
        if self.m_width is not None and config.lm_head.bias:
            # The logit multiplier is applied to the lm_head input (see
            # forward), which only equals scaling the logits without a bias.
            raise ValueError("muP m_width requires an lm_head without bias.")

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ):
        # Same as Decoder.forward plus the muP embedding and logit multipliers.
        if self.tok_embeddings is not None:
            h = self.tok_embeddings(tokens)
            if self.m_emb is not None:
                h = h * self.m_emb
        else:
            h = tokens

        for layer in self.layers.values():
            h = layer(h, attention_masks, positions, padding_mask=padding_mask)

        h = self.norm(h) if self.norm is not None else h

        # Scale the lm_head input rather than the logits so the multiplier also
        # applies when ChunkedLossWrapper runs lm_head itself (_skip_lm_head).
        if self.m_width is not None and self.lm_head is not None:
            h = h * (1 / self.m_width)

        if self._skip_lm_head:
            return h
        return self.lm_head(h) if self.lm_head is not None else h
