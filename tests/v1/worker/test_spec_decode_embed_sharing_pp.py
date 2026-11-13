# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.utils import (
    PPMissingLayer,
    spec_decode_needs_target_embed,
)
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

VOCAB, HIDDEN = 32, 8


def _fake_pp(world_size: int, is_last_rank: bool = True):
    return lambda: SimpleNamespace(
        world_size=world_size,
        is_last_rank=is_last_rank,
        is_first_rank=world_size == 1,
    )


def _inner(embed: nn.Module | None) -> nn.Module:
    inner = nn.Module()
    if embed is not None:
        inner.embed_tokens = embed
    return inner


def _embed(fill: float | None = None) -> nn.Embedding:
    embed = nn.Embedding(VOCAB, HIDDEN)
    if fill is not None:
        with torch.no_grad():
            embed.weight.fill_(fill)
    return embed


@pytest.mark.parametrize("draft_embed", ["loaded", "unset"])
def test_drafter_without_own_embedding_gets_the_targets(monkeypatch, draft_embed):
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp(2))
    target_embed = _embed()
    draft_inner = _inner(_embed() if draft_embed == "loaded" else None)
    if draft_embed == "unset":
        draft_inner.embed_tokens = None
    draft = SimpleNamespace(has_own_embed_tokens=False)

    eagle_utils.maybe_share_target_embed(draft, draft_inner, _inner(target_embed))

    assert draft_inner.embed_tokens is target_embed


def test_missing_target_embedding_raises_instead_of_running_on_garbage(monkeypatch):
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp(2))
    draft_inner = _inner(_embed())
    draft = SimpleNamespace(has_own_embed_tokens=False)

    with pytest.raises(RuntimeError, match="needs the target input embedding"):
        eagle_utils.maybe_share_target_embed(
            draft, draft_inner, _inner(PPMissingLayer())
        )


def test_drafter_with_distinct_weights_keeps_them(monkeypatch):
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp(2))
    draft_embed = _embed(fill=1.0)
    draft_inner = _inner(draft_embed)
    draft = SimpleNamespace(has_own_embed_tokens=True)

    eagle_utils.maybe_share_target_embed(draft, draft_inner, _inner(_embed(fill=2.0)))

    assert draft_inner.embed_tokens is draft_embed


def test_mtp_style_drafter_is_left_alone_under_pp(monkeypatch):
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp(2))
    draft_embed = _embed()
    draft_inner = _inner(draft_embed)

    eagle_utils.maybe_share_target_embed(nn.Module(), draft_inner, _inner(_embed()))

    assert draft_inner.embed_tokens is draft_embed


@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (4, 0), (4, 1), (4, 2), (4, 3)])
def test_bailing_mtp_last_pp_stage_loads_and_uses_own_embedding(
    monkeypatch, pp_size, tp_size, tp_rank
):
    """A text MTP forward has no external embeddings on the last target stage."""
    from vllm.model_executor.layers import vocab_parallel_embedding as vocab_module
    from vllm.model_executor.models import bailing_moe_v3_mtp as mtp_module

    config = SimpleNamespace(
        vocab_size=128,
        hidden_size=HIDDEN,
        num_hidden_layers=72,
        num_nextn_predict_layers=1,
    )
    pp_group = _fake_pp(pp_size)
    monkeypatch.setattr(eagle_utils, "get_pp_group", pp_group)
    # Also exercise the pre-fix constructor, which used this local import.
    monkeypatch.setattr(mtp_module, "get_pp_group", pp_group, raising=False)
    monkeypatch.setattr(vocab_module, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(
        vocab_module, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(
        vocab_module, "tensor_model_parallel_all_reduce", lambda value: value
    )
    monkeypatch.setattr(mtp_module, "_get_draft_hf_config", lambda _: config)
    monkeypatch.setattr(mtp_module, "LogitsProcessor", lambda _: nn.Identity())

    class DraftLayer(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.marker = nn.Parameter(torch.zeros(1))

        def forward(self, ids, positions, hidden, embeds, step):
            return hidden + embeds

    monkeypatch.setattr(mtp_module, "BailingMoeV3MultiTokenPredictorLayer", DraftLayer)
    predictor = mtp_module.BailingMoeV3MultiTokenPredictor(vllm_config=None)
    model = object.__new__(mtp_module.BailingMoeV3MTPModel)
    nn.Module.__init__(model)
    model.config, model.quant_config, model.model = config, None, predictor
    monkeypatch.setattr(model, "get_expert_mapping", lambda: [])
    weight = torch.arange(config.vocab_size * HIDDEN, dtype=torch.float32).reshape(
        config.vocab_size, HIDDEN
    )
    loaded = model.load_weights(
        [
            ("model.word_embeddings.weight", weight),
            ("model.layers.72.marker", torch.ones(1)),
        ]
    )
    assert "model.embed_tokens.weight" in loaded
    embedding = predictor.embed_tokens
    start = tp_rank * (config.vocab_size // tp_size)
    end = start + config.vocab_size // tp_size
    torch.testing.assert_close(embedding.weight, weight[start:end])
    target = _inner(PPMissingLayer() if pp_size > 1 else embedding)
    eagle_utils.maybe_share_target_embed(model, predictor, target)
    assert predictor.embed_tokens is embedding
    ids = torch.tensor([start, end - 1])
    hidden = torch.ones(2, HIDDEN)
    for step in range(3):
        output = predictor(ids, torch.tensor([0, 1]), hidden, spec_step_idx=step)
        torch.testing.assert_close(output, hidden + weight[ids])


@pytest.mark.parametrize(
    "method,pp_size,is_last_rank,expected",
    [
        ("eagle", 2, True, True),
        ("eagle3", 2, True, True),
        ("dflash", 2, True, True),
        ("dspark", 2, True, True),
        ("eagle3", 1, True, False),
        ("eagle3", 2, False, False),
        ("mtp", 2, True, False),
        (None, 2, True, False),
    ],
)
def test_target_embedding_provisioning(
    monkeypatch, method, pp_size, is_last_rank, expected
):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group",
        _fake_pp(pp_size, is_last_rank),
        raising=True,
    )
    speculative_config = None if method is None else SimpleNamespace(method=method)
    config = SimpleNamespace(speculative_config=speculative_config)
    assert spec_decode_needs_target_embed(config) is expected
