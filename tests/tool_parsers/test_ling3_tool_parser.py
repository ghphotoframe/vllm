# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import Mock

import pytest

from tests.tool_parsers.utils import run_tool_extraction_streaming
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.ling3_tool_parser import Ling3ToolParser


class MockTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {
            "<tool_call>": 1,
            "</tool_call>": 2,
            "<arg_key>": 3,
            "</arg_key>": 4,
            "<arg_value>": 5,
            "</arg_value>": 6,
        }

    def tokenize(self, text: str) -> list[str]:
        return [text] if text in self.get_vocab() else []


@pytest.fixture
def sample_tools():
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(name="ping", parameters={}),
        ),
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            ),
        ),
    ]


@pytest.fixture
def ling3_tool_parser(sample_tools):
    return Ling3ToolParser(MockTokenizer(), tools=sample_tools)


@pytest.fixture
def mock_request(sample_tools) -> ChatCompletionRequest:
    request = Mock(spec=ChatCompletionRequest)
    request.tools = sample_tools
    request.tool_choice = "auto"
    return request


def test_ling3_registered():
    assert ToolParserManager.get_tool_parser("ling3") is Ling3ToolParser


@pytest.mark.parametrize("separator", ["\n", r"\n", ""])
def test_ling3_tool_call_name_separator(separator, ling3_tool_parser, mock_request):
    output = (
        f"<tool_call>get_weather{separator}<arg_key>city</arg_key>"
        "<arg_value>Beijing</arg_value></tool_call>"
    )

    result = ling3_tool_parser.extract_tool_calls(output, request=mock_request)

    assert result.tools_called
    assert result.tool_calls[0].function.name == "get_weather"
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Beijing"}


@pytest.mark.parametrize(
    "separator_chunks",
    [
        pytest.param(["\n"], id="newline"),
        pytest.param([r"\n"], id="literal-newline"),
        pytest.param(["\\", "n"], id="split-literal-newline"),
        pytest.param([], id="compact"),
    ],
)
def test_ling3_tool_call_streaming(separator_chunks, ling3_tool_parser, mock_request):
    chunks = (
        ["<tool_call>", "get_weather"]
        + separator_chunks
        + [
            "<arg_key>city</arg_key>",
            "<arg_value>",
            "Beijing",
            "</arg_value>",
            "</tool_call>",
        ]
    )

    result = run_tool_extraction_streaming(
        ling3_tool_parser, chunks, request=mock_request
    )

    assert result.other_content == ""
    assert result.tool_calls[0].function.name == "get_weather"
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Beijing"}


def test_ling3_zero_arg_tool_call_streaming(ling3_tool_parser, mock_request):
    result = run_tool_extraction_streaming(
        ling3_tool_parser,
        ["<tool_call>", "ping", "</tool_call>"],
        request=mock_request,
        assert_one_tool_per_delta=False,
    )

    assert result.tool_calls[0].function.name == "ping"
    assert json.loads(result.tool_calls[0].function.arguments) == {}
