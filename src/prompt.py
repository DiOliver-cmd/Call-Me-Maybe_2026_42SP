"""Deterministic prompt and output-template construction (T3.2/T4.5).

The prompt only *presents* the catalog and the request; it never guarantees the
output structure. The constrained decoder (DFA) is what makes the JSON valid,
so the prompt can stay plain text.

The output template is assembled separately. ``build_template`` produces the
exact fixed-key JSON, while ``build_segments`` splits it into tokenized fixed
text and model-filled slots so the decoder can inject fixed token ids directly.

**Qwen chat template: not used.** The public SDK does not expose
``apply_chat_template``, and ``<|im_start|>``/``<|im_end|>`` are excluded control
tokens in this project anyway. The prompt therefore uses a plain-text template
with a stable serialization of the functions.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from src.dfa import DFA
from src.llm_client import LLMClient
from src.schemas import FunctionCatalog, FunctionDefinition

__all__ = [
    "FUNCTIONS_BEGIN",
    "FUNCTIONS_END",
    "TemplateSegment",
    "apply_fixed",
    "build_prompt",
    "build_segments",
    "build_template",
    "count_prompt_tokens",
]

# Markers make the embedded JSON block easy to extract in tests and reports.
FUNCTIONS_BEGIN = "<functions>"
FUNCTIONS_END = "</functions>"


@dataclass(frozen=True)
class TemplateSegment:
    """One fixed-text or model-filled part of the output template.

    Fixed segments are injected by the decoder and carry their token ids. Slot
    segments are filled by the model: the first slot is always the function
    name and contains only its content, because the fixed segments provide the
    quotes. Later slots map to parameter keys; a parameter may itself be named
    ``name``, so consumers must use the slot position to distinguish them.
    """

    kind: Literal["fixed", "slot"]
    text: str
    token_ids: tuple[int, ...] = ()
    slot: str | None = None


def _resolve_function(
    catalog: FunctionCatalog,
    name: str,
) -> FunctionDefinition:
    """Return the declared function or raise ``ValueError``."""
    try:
        return catalog.by_name(name)
    except KeyError as exc:
        raise ValueError(f"unknown function name {name!r}") from exc


def _validated_parameters(
    catalog: FunctionCatalog,
    name: str,
    parameters: dict[str, Any],
) -> tuple[FunctionDefinition, dict[str, Any]]:
    """Resolve ``name`` and return parameters in declaration order."""
    function = _resolve_function(catalog, name)

    declared = function.parameters
    missing = [key for key in declared if key not in parameters]
    unexpected = sorted(set(parameters) - set(declared))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing parameter(s): {', '.join(missing)}")
        if unexpected:
            details.append(
                f"unexpected parameter(s): {', '.join(unexpected)}"
            )
        raise ValueError(
            f"parameters for function {name!r} must match the declaration; "
            + "; ".join(details)
        )
    return function, {key: parameters[key] for key in declared}


def build_template(
    catalog: FunctionCatalog,
    prompt: str,
    name: str,
    parameters: dict[str, Any],
) -> str:
    """Build the exact compact JSON output template.

    The object keys are always ``prompt``, ``name`` and ``parameters``, and the
    parameters follow their declaration order, so the same inputs always produce
    the same bytes. ``prompt`` is serialized with :func:`json.dumps`, preserving
    its content (including quotes and Unicode) when parsed back.
    """
    _, ordered = _validated_parameters(catalog, name, parameters)
    obj = {"prompt": prompt, "name": name, "parameters": ordered}
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def build_segments(
    catalog: FunctionCatalog,
    prompt: str,
    name: str,
    tokenize: Callable[[str], Sequence[int]],
) -> tuple[TemplateSegment, ...]:
    """Split the output template into fixed text and model-filled slots.

    Every fixed segment is passed through ``tokenize`` once and stores the
    resulting ids. Slots carry no tokens: the decoder generates the name content
    (without quotes) and each parameter value as JSON text. Only the function
    declaration is needed here, because the decoder does not know the slot
    values yet. Joining the fixed texts with the slot fills reproduces
    :func:`build_template` exactly.
    """
    function = _resolve_function(catalog, name)
    keys = tuple(function.parameters)
    segments: list[TemplateSegment] = []

    def fixed(text: str) -> None:
        segments.append(
            TemplateSegment(
                kind="fixed",
                text=text,
                token_ids=tuple(int(token_id) for token_id in tokenize(text)),
            )
        )

    fixed('{"prompt":')
    fixed(json.dumps(prompt, ensure_ascii=False))
    fixed(',"name":"')
    segments.append(TemplateSegment(kind="slot", text="", slot="name"))
    fixed('","parameters":{')

    for index, key in enumerate(keys):
        if index:
            fixed(",")
        fixed(json.dumps(key, ensure_ascii=False) + ":")
        segments.append(TemplateSegment(kind="slot", text="", slot=key))

    fixed("}}")
    return tuple(segments)


def apply_fixed(dfa: DFA, segment: TemplateSegment) -> bool:
    """Inject one fixed segment into ``dfa`` through ``consume_token``."""
    if segment.kind != "fixed":
        raise ValueError(
            f"apply_fixed requires a fixed segment, got {segment.kind!r}"
        )
    return dfa.consume_token(segment.text)


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
