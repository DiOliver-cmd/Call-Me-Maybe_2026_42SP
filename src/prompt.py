"""Deterministic function-calling prompt construction (T3.2).

The prompt only *presents* the catalog and the request; it never guarantees the
output structure. The constrained decoder (DFA) is what makes the JSON valid,
so the prompt can stay plain text.

**Qwen chat template: not used.** The public SDK does not expose
``apply_chat_template``, and ``<|im_start|>``/``<|im_end|>`` are excluded control
tokens in this project anyway. The prompt therefore uses a plain-text template
with a stable serialization of the functions.
"""

from __future__ import annotations

import json

from src.llm_client import LLMClient
from src.schemas import FunctionCatalog

__all__ = [
    "FUNCTIONS_BEGIN",
    "FUNCTIONS_END",
    "build_prompt",
    "count_prompt_tokens",
]

# Markers make the embedded JSON block easy to extract in tests and reports.
FUNCTIONS_BEGIN = "<functions>"
FUNCTIONS_END = "</functions>"


def build_prompt(catalog: FunctionCatalog, request: str) -> str:
    """Build a deterministic plain-text prompt for ``request``.

    Functions are sorted by name and serialized with sorted keys and compact
    separators, so the same catalog always produces the same bytes regardless
    of its declaration order. The request is only stripped; its content is
    never rewritten.
    """
    functions_json = json.dumps(
        [
            function.model_dump(mode="json")
            for function in sorted(
                catalog.functions,
                key=lambda function: function.name,
            )
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    lines = [
        "You are a function-calling assistant.",
        "Choose exactly one function from the catalog below and return ONE "
        "JSON object.",
        "Do not write prose, Markdown fences, or any text outside that object.",
        "",
        "Available functions (JSON):",
        FUNCTIONS_BEGIN,
        functions_json,
        FUNCTIONS_END,
        "",
        "User request:",
        request.strip(),
        "",
        'Response format (keys "name" and "arguments"):',
        '{"name": "<function_name>", "arguments": '
        '{"<parameter_name>": <value>}}',
        "",
        "Return exactly one JSON object and nothing else.",
    ]
    return "\n".join(lines)


def count_prompt_tokens(client: LLMClient, prompt: str) -> int:
    """Return the number of tokens ``client.encode(prompt)`` produces."""
    return len(client.encode(prompt))
