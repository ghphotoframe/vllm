# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import bailing_moe_v3 as bailing


class _GateLinear(nn.Module):
    """CPU stand-in for the device-dispatched BF16-to-FP32 gate."""

    def __init__(
        self, input_size, output_size, *, bias, out_dtype, params_dtype, prefix
    ):
        super().__init__()
        assert not bias
        self.out_dtype = out_dtype
        self.weight = nn.Parameter(
            torch.zeros(output_size, input_size, dtype=params_dtype)
        )
        self.last_input = None

    def forward(self, hidden_states):
        self.last_input = hidden_states
        logits = hidden_states.float() @ self.weight.float().t()
        return logits.to(self.out_dtype), None


class _Experts(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.factory_args = kwargs
        self.last_router_logits = None

    def forward(self, hidden_states, router_logits):
        self.last_router_logits = router_logits
        return hidden_states


@pytest.fixture
def make_moe(monkeypatch):
    monkeypatch.setattr(bailing, "GateLinear", _GateLinear)
    monkeypatch.setattr(bailing, "BailingMoeV3MLP", lambda **kwargs: nn.Identity())
    monkeypatch.setattr(bailing, "FusedMoEFactory", _Experts)

    def make(router_weight_dtype=None):
        config = SimpleNamespace(
            num_experts=4,
            num_experts_per_tok=2,
            hidden_size=8,
            moe_intermediate_size=4,
            moe_shared_expert_intermediate_size=4,
            num_shared_experts=1,
            n_group=2,
            topk_group=1,
        )
        if router_weight_dtype is not None:
            config.router_weight_dtype = router_weight_dtype
        return bailing.BailingMoeV3MoE(config, prefix="model.layers.0.mlp")

    return make


@pytest.mark.parametrize(
    ("setting", "expected_dtype"),
    [(None, torch.float32), ("float32", torch.float32), ("bfloat16", torch.bfloat16)],
)
def test_router_preserves_parameter_names_and_fp32_logits(
    make_moe, setting, expected_dtype
):
    moe = make_moe(setting)
    assert moe.gate.weight.dtype == expected_dtype
    assert moe.gate.expert_bias.dtype == torch.float32
    assert {"gate.weight", "gate.expert_bias"} <= set(dict(moe.named_parameters()))
    assert moe.experts.factory_args["e_score_correction_bias"] is moe.gate.expert_bias
    hidden_states = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
    output = moe(hidden_states)
    assert moe.gate.last_input.dtype == torch.bfloat16
    assert moe.experts.last_router_logits.dtype == torch.float32
    assert moe.experts.last_router_logits.shape == (2, 4)
    torch.testing.assert_close(output, hidden_states, rtol=0, atol=0)


def test_router_observes_in_place_weight_updates(make_moe):
    moe = make_moe("bfloat16")
    weight = moe.gate.weight
    hidden_states = torch.ones(2, 8, dtype=torch.bfloat16)
    for value in (1, 3):
        loaded = torch.full((4, 8), value, dtype=torch.bfloat16)
        default_weight_loader(weight, loaded)
        moe(hidden_states)
        assert moe.gate.weight is weight
        torch.testing.assert_close(
            moe.experts.last_router_logits,
            torch.full((2, 4), 8 * value, dtype=torch.float32),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("setting", ["float16", "fp32", "invalid"])
def test_router_rejects_unsupported_weight_dtype(make_moe, setting):
    with pytest.raises(ValueError, match="router_weight_dtype"):
        make_moe(setting)
