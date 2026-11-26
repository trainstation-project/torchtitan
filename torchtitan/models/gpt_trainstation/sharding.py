# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.decoder_sharding import set_decoder_sharding_config
from torchtitan.models.qwen3.sharding import set_qwen3_layer_sharding

if TYPE_CHECKING:
    from torchtitan.models.gpt_trainstation.model import GptTrainstationModel


def set_gpt_trainstation_sharding_config(
    config: "GptTrainstationModel.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on all GptTrainstation sub-configs.

    Every layer reuses Qwen3's per-layer sharding, which handles GQA, dense FFN
    and MoE (with or without shared experts).
    """
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        attention = layer_cfg.attention
        if not isinstance(attention, GQAttention.Config):
            raise TypeError(
                f"Unsupported attention type {type(attention).__name__} in GptTrainstation."
            )
        set_qwen3_layer_sharding(layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep)
