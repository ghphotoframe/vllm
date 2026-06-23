# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Bailing MoE v3 MTP model."""

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.models.bailing_moe_mtp import (
    BailingMTPModelBase, BailingMTPVariant)
from vllm.model_executor.models.bailing_moe_v3 import (
    BailingMoeV3MLAAttention, BailingMoeV3MoE)


def _make_bailing_v3_expert_mapping(
        model: BailingMTPModelBase) -> list[tuple[str, str, int, str]]:
    return FusedMoE.make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=model.config.num_experts,
        num_redundant_experts=0,
    )


_BAILING_V3_MTP_VARIANT = BailingMTPVariant(
    error_name="Bailing V3",
    attention_cls=BailingMoeV3MLAAttention,
    mlp_cls=BailingMoeV3MoE,
    expert_mapping_fn=_make_bailing_v3_expert_mapping,
)


@support_torch_compile
class BailingMoeV3MTPModel(BailingMTPModelBase):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            variant=_BAILING_V3_MTP_VARIANT,
        )
