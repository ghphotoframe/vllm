# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning parser for InclusionAI Ling 3 models."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike


class Ling3ReasoningParser(Qwen3ReasoningParser):
    """Parse Ling3 ``<think>`` output, with thinking enabled by default.

    Ling3 uses the GLM-style ``<tool_call>`` marker, which also acts as an
    implicit reasoning terminator when the model omits ``</think>``.
    """

    def __init__(self, tokenizer: "TokenizerLike", *args, **kwargs):
        chat_kwargs = dict(kwargs.get("chat_template_kwargs", {}) or {})
        thinking = chat_kwargs.get("thinking")
        enable_thinking = chat_kwargs.get("enable_thinking")
        chat_kwargs["enable_thinking"] = (
            True
            if thinking is None and enable_thinking is None
            else bool(thinking) or bool(enable_thinking)
        )
        kwargs["chat_template_kwargs"] = chat_kwargs
        super().__init__(tokenizer, *args, **kwargs)

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        """Count reasoning tokens when ``<think>`` is in the prompt."""
        if self.end_token_id in token_ids:
            return token_ids.index(self.end_token_id) + 1
        return len(token_ids) if self.thinking_enabled else 0

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output

        return super().extract_reasoning(model_output, request)
