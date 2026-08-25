# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.ling3_reasoning_parser import Ling3ReasoningParser


class MockTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {
            "<think>": 1,
            "</think>": 2,
            "<tool_call>": 3,
            "</tool_call>": 4,
        }


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="test-model", messages=[])


def test_ling3_registered_and_defaults_thinking_on():
    parser_cls = ReasoningParserManager.get_reasoning_parser("ling3")
    assert parser_cls is Ling3ReasoningParser

    parser = parser_cls(MockTokenizer())
    reasoning, content = parser.extract_reasoning(
        "<think>reason</think>answer", _request()
    )

    assert parser.thinking_enabled
    assert reasoning == "reason"
    assert content == "answer"


def test_ling3_can_disable_thinking():
    parser = Ling3ReasoningParser(
        MockTokenizer(), chat_template_kwargs={"enable_thinking": False}
    )

    reasoning, content = parser.extract_reasoning(
        "<think>reason</think>answer", _request()
    )

    assert not parser.thinking_enabled
    assert reasoning is None
    assert content == "<think>reason</think>answer"


def test_ling3_truncated_reasoning_stays_reasoning():
    parser = Ling3ReasoningParser(MockTokenizer())

    reasoning, content = parser.extract_reasoning("only reasoning", _request())

    assert reasoning == "only reasoning"
    assert content is None


def test_ling3_tool_call_implicitly_ends_reasoning():
    parser = Ling3ReasoningParser(MockTokenizer())
    output = "<think>need a tool<tool_call>ping</tool_call>"

    reasoning, content = parser.extract_reasoning(output, _request())

    assert reasoning == "need a tool"
    assert content == "<tool_call>ping</tool_call>"


def test_ling3_tool_call_implicitly_ends_reasoning_streaming():
    parser = Ling3ReasoningParser(MockTokenizer())

    reasoning_delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="<think>need a tool",
        delta_text="<think>need a tool",
        previous_token_ids=[],
        current_token_ids=[1, 10],
        delta_token_ids=[1, 10],
    )
    tool_delta = parser.extract_reasoning_streaming(
        previous_text="<think>need a tool",
        current_text="<think>need a tool<tool_call>",
        delta_text="<tool_call>",
        previous_token_ids=[1, 10],
        current_token_ids=[1, 10, 3],
        delta_token_ids=[3],
    )

    assert reasoning_delta is not None
    assert reasoning_delta.reasoning == "need a tool"
    assert reasoning_delta.content is None
    assert tool_delta is not None
    assert tool_delta.reasoning is None
    assert tool_delta.content == "<tool_call>"
