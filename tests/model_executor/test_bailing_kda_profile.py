# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import DeviceConfig, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.bailing_moe_v3 import BailingMoeV3KimiDeltaAttention


def test_kda_profile_without_attention_metadata_preserves_output():
    """The startup memory probe has no KV metadata and must not run KDA."""
    attention = object.__new__(BailingMoeV3KimiDeltaAttention)
    torch.nn.Module.__init__(attention)
    output = torch.full((1, 2, 1, 4), 3.0)
    original = output.clone()
    states = torch.zeros(2, 4)
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    with set_forward_context(None, config):
        attention._forward(states, states, states, states, states, output)
    torch.testing.assert_close(output, original, rtol=0, atol=0)
