# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.bailing_mrope import (
    BailingMRotaryEmbedding,
)
from vllm.platforms import current_platform


def _direct_rotary(
    x: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    """Compute signed angles directly, independently of the RoPE cache."""
    frequency_axes = [1, 2, 1, 2, 0, 0] if rotary_dim == 12 else [1, 2] * 12 + [0] * 8
    if positions.ndim == 1:
        selected_positions = positions[:, None]
    else:
        selected_positions = positions.T[:, frequency_axes]
    inv_freq = 1.0 / (
        10_000 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    angles = selected_positions.float() * inv_freq
    cos = angles.cos().to(x.dtype).unsqueeze(1)
    sin = angles.sin().to(x.dtype).unsqueeze(1)
    even, odd = x[..., :rotary_dim:2], x[..., 1:rotary_dim:2]
    result = x.clone()
    # Retain the native path's dtype rounding after each multiplication.
    result[..., :rotary_dim:2] = even * cos - odd * sin
    result[..., 1:rotary_dim:2] = odd * cos + even * sin
    return result


def _signed_positions(dtype: torch.dtype = torch.int64) -> torch.Tensor:
    # Include zero, both signs on every axis, and the cache's final valid row.
    return torch.tensor(
        [[0, 1, -2, 63, -63], [-3, 0, 4, -63, 63], [5, -6, 0, 63, -63]],
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("signed", [False, True])
@torch.inference_mode()
def test_bailing_mrope_native_matches_signed_angles(
    dtype: torch.dtype,
    position_dtype: torch.dtype,
    multimodal: bool,
    signed: bool,
):
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        rotary_emb = BailingMRotaryEmbedding(
            head_size=16,
            rotary_dim=12,
            max_position_embeddings=64,
            base=10_000,
            is_neox_style=False,
            dtype=dtype,
            mrope_section=[2, 2, 2],
        )
    positions = _signed_positions(position_dtype)
    if not signed:
        positions = positions.abs()
    if not multimodal:
        positions = positions[0]
    torch.manual_seed(0)
    query = torch.randn(5, 3, 16, dtype=dtype)
    key = torch.randn(5, 1, 16, dtype=dtype)
    expected_query = _direct_rotary(query, positions, 12)
    expected_key = _direct_rotary(key, positions, 12)

    actual_query, actual_key = rotary_emb.forward_native(positions, query, key)

    # Positive positions must retain the same cache and intermediate rounding.
    torch.testing.assert_close(actual_query, expected_query, atol=0, rtol=0)
    torch.testing.assert_close(actual_key, expected_key, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Skipping CUDA/ROCm-only test.",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rotary_dim", [12, 64])
@torch.inference_mode()
def test_bailing_mrope_cuda_signed_strided_views_preserve_nope(
    dtype: torch.dtype,
    rotary_dim: int,
):
    head_size = rotary_dim + 4
    with set_current_vllm_config(VllmConfig()):
        rotary_emb = BailingMRotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=64,
            base=10_000,
            is_neox_style=False,
            dtype=dtype,
            mrope_section=[2, 2, 2] if rotary_dim == 12 else [8, 12, 12],
        ).cuda()
    position_buffer = torch.zeros(3, 10, dtype=torch.int32, device="cuda")
    position_buffer[:, ::2] = _signed_positions(torch.int32).cuda()
    positions = position_buffer[:, ::2]
    torch.manual_seed(0)
    query_buffer = torch.randn(5, 3, 2 * head_size + 8, dtype=dtype, device="cuda")
    key_buffer = torch.randn(5, 1, 2 * head_size + 8, dtype=dtype, device="cuda")
    query = query_buffer[..., 4 : 4 + 2 * head_size : 2]
    key = key_buffer[..., 4 : 4 + 2 * head_size : 2]
    assert not positions.is_contiguous()
    assert not query.is_contiguous()
    assert not key.is_contiguous()
    expected_query = _direct_rotary(query.cpu(), positions.cpu(), rotary_dim).cuda()
    expected_key = _direct_rotary(key.cpu(), positions.cpu(), rotary_dim).cuda()
    saved_query_buffer = query_buffer.clone()
    saved_key_buffer = key_buffer.clone()

    actual_query, actual_key = rotary_emb.forward_cuda(positions, query, key)

    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_key, expected_key)
    assert actual_query.data_ptr() == query.data_ptr()
    assert actual_key is not None
    assert actual_key.data_ptr() == key.data_ptr()
    # Only the rotary elements of each strided head may be modified.
    untouched = torch.ones(query_buffer.shape[-1], dtype=torch.bool, device="cuda")
    untouched[4 : 4 + 2 * rotary_dim : 2] = False
    assert torch.equal(query_buffer[..., untouched], saved_query_buffer[..., untouched])
    assert torch.equal(key_buffer[..., untouched], saved_key_buffer[..., untouched])


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Skipping CUDA/ROCm-only test.",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("signed", [False, True])
@torch.inference_mode()
def test_bailing_mrope_cuda_preserves_native_product_rounding(
    dtype: torch.dtype,
    signed: bool,
):
    with set_current_vllm_config(VllmConfig()):
        rotary_emb = BailingMRotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=128,
            base=6_000_000,
            is_neox_style=False,
            dtype=dtype,
            mrope_section=[8, 12, 12],
        ).cuda()
    torch.manual_seed(0)
    positions = torch.randint(-127, 128, (3, 17), device="cuda")
    positions[:, 0] = 0
    positions[:, 1] = torch.tensor([-127, 127, -127], device="cuda")
    if not signed:
        positions = positions.abs()
    query = torch.randn(17, 3, 64, dtype=dtype, device="cuda")
    key = torch.randn(17, 1, 64, dtype=dtype, device="cuda")

    # Use the same cached coefficients to isolate the rounding of each
    # multiplication. Fusing a product into the addition skips this rounding.
    expected_query, expected_key = rotary_emb.forward_native(positions, query, key)
    actual_query, actual_key = rotary_emb.forward_cuda(positions, query, key)
    assert expected_key is not None
    assert actual_key is not None

    assert torch.equal(
        actual_query.contiguous().view(torch.int16),
        expected_query.contiguous().view(torch.int16),
    )
    assert torch.equal(
        actual_key.contiguous().view(torch.int16),
        expected_key.contiguous().view(torch.int16),
    )
