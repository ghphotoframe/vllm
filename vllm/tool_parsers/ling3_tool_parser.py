# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tool parser adapter for InclusionAI Ling 3 models."""

import regex as re

from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser


class Ling3ToolParser(Glm47MoeModelToolParser):
    """Parse Ling3's GLM-style XML tool calls."""

    supports_required_and_named = False

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self.func_detail_regex = re.compile(
            r"<tool_call>\s*(.*?)"
            r"(?:(?:\\n|\n)\s*|(?=<arg_key>)|(?=</tool_call>))"
            r"(.*?)</tool_call>",
            re.DOTALL,
        )

    def _extract_tool_call_regions(self, text: str) -> list[tuple[str, bool]]:
        regions = super()._extract_tool_call_regions(text)
        normalized_regions: list[tuple[str, bool]] = []
        for inner_text, is_complete in regions:
            arg_key_index = inner_text.find(self.arg_key_start)
            literal_newline_index = inner_text.find(r"\n")
            if literal_newline_index != -1 and (
                arg_key_index == -1 or literal_newline_index < arg_key_index
            ):
                inner_text = (
                    inner_text[:literal_newline_index]
                    + "\n"
                    + inner_text[literal_newline_index + 2 :]
                )
            elif is_complete and "\n" not in inner_text and arg_key_index == -1:
                # The parent streaming parser waits for a name delimiter. Add
                # one only to the completed, internal zero-argument region.
                inner_text += "\n"
            normalized_regions.append((inner_text, is_complete))
        return normalized_regions
