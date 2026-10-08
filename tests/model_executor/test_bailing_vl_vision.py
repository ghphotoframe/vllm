# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small CPU and CUDA checks against the Bailing HF vision equations."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.model_executor.layers.attention import mm_encoder_attention
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.models import qwen3_vl, vision
from vllm.model_executor.models.bailing_moe_v3_vl import (
    BailingMoeV3VisionTransformer,
    BailingMoeV3VLForConditionalGeneration,
    BailingMoeV3VLProjector,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.bailing_moe_v3_vl import (
    BailingMoeV3VisionConfig,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.ops.vit_attn_wrappers import torch_sdpa_wrapper


@pytest.fixture
def cpu_vision_context(monkeypatch):
    # No distributed communication is needed for this one-rank tower.
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(world_size=1, rank_in_group=0)
    )
    monkeypatch.setattr(
        vision,
        "_get_vit_attn_backend",
        lambda *args, **kwargs: AttentionBackendEnum.TORCH_SDPA,
    )
    # ApplyRotaryEmb is force-enabled by the tower; use its CPU equations
    # without requiring the CUDA extension on the test host.
    monkeypatch.setattr(ApplyRotaryEmb, "forward_cuda", ApplyRotaryEmb.forward_native)
    # CUDA hosts have Triton installed, but these fixtures keep all tensors
    # on CPU. Use the identical native interpolation equations there.
    monkeypatch.setattr(
        qwen3_vl,
        "triton_pos_embed_interpolate",
        qwen3_vl.pos_embed_interpolate_native,
        raising=False,
    )
    # On CUDA hosts this custom op is registered only for CUDA tensors.
    # Call its Python implementation to keep the real per-image CPU SDPA.
    monkeypatch.setattr(
        mm_encoder_attention, "vit_torch_sdpa_wrapper", torch_sdpa_wrapper
    )
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        yield


def _vision_config(disable_merger_proj):
    return BailingMoeV3VisionConfig(
        depth=1,
        hidden_size=8,
        intermediate_size=16,
        num_heads=1,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=12,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
        disable_merger_proj=disable_merger_proj,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("disable_merger_proj", [False, True])
@torch.inference_mode()
def test_merger_matches_hf_projection_switch(cpu_vision_context, disable_merger_proj):
    config = _vision_config(disable_merger_proj)
    tower = BailingMoeV3VisionTransformer(config)
    torch.manual_seed(0)
    for parameter in tower.parameters():
        parameter.copy_(torch.randn_like(parameter) * 0.1)
    patches = torch.randn(8, 1, 8)
    actual = tower.merger(patches)
    assert actual.shape == (2, 32 if disable_merger_proj else 12)

    merger = tower.merger
    expected = F.layer_norm(
        patches, (8,), merger.norm.weight, merger.norm.bias, 1e-6
    ).reshape(2, 32)
    if not disable_merger_proj:
        expected = F.linear(
            F.gelu(
                F.linear(expected, merger.linear_fc1.weight, merger.linear_fc1.bias)
            ),
            merger.linear_fc2.weight,
            merger.linear_fc2.bias,
        )
    torch.testing.assert_close(actual, expected)
    assert config.deepstack_visual_indexes == [0]
    assert config.out_hidden_size == 12


@pytest.mark.cpu_test
@torch.inference_mode()
def test_vision_rope_keeps_fp32_frequencies_and_rotation(cpu_vision_context):
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        tower = BailingMoeV3VisionTransformer(_vision_config(True))
    finally:
        torch.set_default_dtype(old_dtype)

    cos, sin = tower.rot_pos_emb([[1, 2, 4]])
    # Patch order is the four pixels of each 2x2 spatial group.
    coords = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]],
        dtype=torch.float32,
    )
    frequencies = (coords[..., None] * torch.tensor([1.0, 0.01])).flatten(1)
    torch.testing.assert_close(cos, frequencies.cos(), rtol=0, atol=0)
    torch.testing.assert_close(sin, frequencies.sin(), rtol=0, atol=0)

    torch.manual_seed(1)
    query = torch.randn(8, 2, 8, dtype=torch.bfloat16)
    actual = tower.blocks[0].attn.apply_rotary_emb.forward_native(query, cos, sin)
    cos_full = torch.cat([frequencies.cos()] * 2, dim=-1).unsqueeze(1)
    sin_full = torch.cat([frequencies.sin()] * 2, dim=-1).unsqueeze(1)
    q_float = query.float()
    rotated = torch.cat([-q_float[..., 4:], q_float[..., :4]], dim=-1)
    expected = (q_float * cos_full + rotated * sin_full).to(query.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cpu_test
@pytest.mark.parametrize("disable_merger_proj", [False, True])
@torch.inference_mode()
def test_image_projector_consumes_main_vision_stream(
    cpu_vision_context, disable_merger_proj
):
    tower = BailingMoeV3VisionTransformer(_vision_config(disable_merger_proj))
    vision_out_dim = 32 if disable_merger_proj else 12
    projector = BailingMoeV3VLProjector(vision_out_dim, 8)
    torch.manual_seed(2)
    for parameter in [*tower.parameters(), *projector.parameters()]:
        parameter.copy_(torch.randn_like(parameter) * 0.1)

    model = object.__new__(BailingMoeV3VLForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.visual = tower
    model.linear_proj = projector
    model.vision_out_dim = vision_out_dim
    pixels = torch.randn(12, 12)
    grids = torch.tensor([[1, 2, 2], [1, 2, 4]])
    features = tower(pixels, grids)
    assert features.shape == (3, 32 if disable_merger_proj else 24)
    expected = projector(features[:, :vision_out_dim])

    actual = model._process_image_input(
        {"type": "pixel_values", "pixel_values": pixels, "image_grid_thw": grids}
    )
    assert [part.shape for part in actual] == [torch.Size([1, 8]), torch.Size([2, 8])]
    torch.testing.assert_close(torch.cat(actual), expected)


@pytest.fixture
def cuda_vision_context(monkeypatch):
    # Only remove distributed setup and select the attention backend. In
    # particular, keep the actual CUDA interpolation dispatch untouched.
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(world_size=1, rank_in_group=0)
    )
    monkeypatch.setattr(
        vision,
        "_get_vit_attn_backend",
        lambda *args, **kwargs: AttentionBackendEnum.TORCH_SDPA,
    )
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cuda"))):
        yield


def _hf_position_interpolation(weight, height, width, merge):
    """HF bilinear equations, including model-dtype product rounding."""
    side = int(weight.shape[0] ** 0.5)
    rows = torch.linspace(
        0, side - 1, height, device=weight.device, dtype=torch.float32
    )
    columns = torch.linspace(
        0, side - 1, width, device=weight.device, dtype=torch.float32
    )
    row_low, col_low = rows.long(), columns.long()
    row_high = (row_low + 1).clamp_max(side - 1)
    col_high = (col_low + 1).clamp_max(side - 1)
    dh = (rows - row_low)[:, None]
    dw = (columns - col_low)[None, :]
    bottom_right = dh * dw
    bottom_left = dh - bottom_right
    top_right = dw - bottom_right
    top_left = 1 - dh - top_right
    table = weight.reshape(side, side, -1)
    products = [
        table[ri[:, None], ci[None, :]] * coefficient.to(weight.dtype)[..., None]
        for ri, ci, coefficient in (
            (row_low, col_low, top_left),
            (row_low, col_high, top_right),
            (row_high, col_low, bottom_left),
            (row_high, col_high, bottom_right),
        )
    ]
    interpolated = torch.stack(products).sum(dim=0).reshape(height * width, -1)
    # Enumerate image patches in spatial-merge order independently of the
    # implementation's reshape/permute sequence.
    patch_order = [
        (block_row + local_row) * width + block_col + local_col
        for block_row in range(0, height, merge)
        for block_col in range(0, width, merge)
        for local_row in range(merge)
        for local_col in range(merge)
    ]
    return interpolated[patch_order]


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Requires CUDA vision interpolation"
)
@pytest.mark.parametrize("height,width", [(32, 48), (6, 10)])
@torch.inference_mode()
def test_vision_position_interpolation_cuda_bf16_matches_hf(
    cuda_vision_context, height, width
):
    # These nontrivial grids occur in the native image-input probe. Preserve
    # HF's BF16 product/reduction rounding, rather than the inherited Triton
    # multiply-add contraction.
    config = _vision_config(True)
    config.hidden_size = 1152
    config.num_heads = 16
    config.num_position_embeddings = 48 * 48
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            tower = BailingMoeV3VisionTransformer(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    assert tower.hidden_size // tower.num_heads == 72
    generator = torch.Generator(device="cpu").manual_seed(20261004)
    weight = torch.randn(
        tower.pos_embed.weight.shape, generator=generator, dtype=torch.float32
    )
    tower.pos_embed.weight.copy_(weight.to(device="cuda", dtype=torch.bfloat16))
    expected = _hf_position_interpolation(
        tower.pos_embed.weight, height, width, config.spatial_merge_size
    )

    actual = tower.fast_pos_embed_interpolate([[1, height, width]])

    assert actual.shape == (height * width, config.hidden_size)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Requires CUDA rotary kernel"
)
@torch.inference_mode()
def test_vision_rope_cuda_bf16_head_dim_72_matches_hf(monkeypatch):
    # Only distributed metadata and attention-backend selection are stubbed.
    # The Bailing cache and ApplyRotaryEmb CUDA dispatch remain real.
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(world_size=1, rank_in_group=0)
    )
    monkeypatch.setattr(
        vision,
        "_get_vit_attn_backend",
        lambda *args, **kwargs: AttentionBackendEnum.TORCH_SDPA,
    )
    config = _vision_config(True)
    config.hidden_size = 72
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with (
            set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cuda"))),
            torch.device("cuda"),
        ):
            tower = BailingMoeV3VisionTransformer(config).to(device="cuda")
    finally:
        torch.set_default_dtype(previous_dtype)

    cos, sin = tower.rot_pos_emb([[1, 4, 6]])
    coords = torch.tensor(
        [
            [block_h + local_h, block_w + local_w]
            for block_h in range(0, 4, 2)
            for block_w in range(0, 6, 2)
            for local_h in range(2)
            for local_w in range(2)
        ],
        device="cuda",
        dtype=torch.float32,
    )
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, 36, 2, device="cuda", dtype=torch.float32) / 36)
    )
    frequencies = (coords[..., None] * inv_freq).flatten(1)
    torch.testing.assert_close(cos, frequencies.cos(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(sin, frequencies.sin(), rtol=1e-6, atol=1e-7)
    assert cos.dtype == sin.dtype == torch.float32

    torch.manual_seed(3)
    # The first dimension packs Q and K as in Qwen2_5_VisionAttention.
    query_key = torch.randn(2, 24, 1, 72, device="cuda", dtype=torch.bfloat16)
    rotary = tower.blocks[0].attn.apply_rotary_emb
    assert rotary.enable_fp32_compute
    actual = rotary(query_key, cos, sin)
    qk_float = query_key.float()
    rotated = torch.cat([-qk_float[..., 36:], qk_float[..., :36]], dim=-1)
    cos_full = torch.cat([frequencies.cos()] * 2, dim=-1)[None, :, None, :]
    sin_full = torch.cat([frequencies.sin()] * 2, dim=-1)[None, :, None, :]
    expected = (qk_float * cos_full + rotated * sin_full).to(query_key.dtype)
    assert actual.dtype == torch.bfloat16
    # A fused FP32 multiply-add can cross a final BF16 rounding midpoint.
    torch.testing.assert_close(
        actual, expected, rtol=torch.finfo(torch.bfloat16).eps, atol=1e-5
    )
