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
    "build_prefix_segments",
    "build_prompt",
    "build_segments",
    "build_tail_segments",
    "build_template",
    "build_verbose_prompt",
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


def _fixed_segment(
    text: str,
    tokenize: Callable[[str], Sequence[int]],
) -> TemplateSegment:
    """Return one fixed segment with ``text`` tokenized once."""
    return TemplateSegment(
        kind="fixed",
        text=text,
        token_ids=tuple(int(token_id) for token_id in tokenize(text)),
    )


def build_prefix_segments(
    prompt: str,
    tokenize: Callable[[str], Sequence[int]],
) -> tuple[TemplateSegment, ...]:
    """Return the fixed segments before the function-name slot.

    The prefix is ``{"prompt":``, the JSON-encoded prompt, and ``,"name":"``.
    The name slot that follows is created by :func:`build_segments`; it is not
    part of this prefix so the decoder can inject the prefix independently of
    the function chosen by the model later.
    """
    return (
        _fixed_segment('{"prompt":', tokenize),
        _fixed_segment(json.dumps(prompt, ensure_ascii=False), tokenize),
        _fixed_segment(',"name":"', tokenize),
    )


def build_tail_segments(
    catalog: FunctionCatalog,
    name: str,
    tokenize: Callable[[str], Sequence[int]],
) -> tuple[TemplateSegment, ...]:
    """Return the fixed/tail segments after the function-name slot.

    The tail starts with the fixed ``","parameters":{`` opener. Each declared
    parameter contributes an optional fixed comma, its JSON-encoded key plus
    ``:``, and a model-filled value slot in declaration order. The final fixed
    ``}}`` closes the parameters object and the root object.
    """
    function = _resolve_function(catalog, name)
    keys = tuple(function.parameters)
    segments: list[TemplateSegment] = [
        _fixed_segment('","parameters":{', tokenize)
    ]

    for index, key in enumerate(keys):
        if index:
            segments.append(_fixed_segment(",", tokenize))
        segments.append(
            _fixed_segment(json.dumps(key, ensure_ascii=False) + ":", tokenize)
        )
        segments.append(TemplateSegment(kind="slot", text="", slot=key))

    segments.append(_fixed_segment("}}", tokenize))
    return tuple(segments)


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
    _resolve_function(catalog, name)
    return (
        build_prefix_segments(prompt, tokenize)
        + (TemplateSegment(kind="slot", text="", slot="name"),)
        + build_tail_segments(catalog, name, tokenize)
    )


def apply_fixed(dfa: DFA, segment: TemplateSegment) -> bool:
    """Inject one fixed segment into ``dfa`` through ``consume_token``."""
    if segment.kind != "fixed":
        raise ValueError(
            f"apply_fixed requires a fixed segment, got {segment.kind!r}"
        )
    return dfa.consume_token(segment.text)


def _compact_signature(function: FunctionDefinition) -> str:
    """Render one function as ``name(param:type, ...): description``.

    The signature carries everything the model needs to choose a function and
    fill its parameters: the name, the parameter names and their types, and the
    natural-language description. Return types are omitted because the model
    never emits them: the decoder injects the fixed output template and the
    model only fills the name and the parameter values.
    """
    params = ", ".join(
        f"{name}:{spec.type}" for name, spec in function.parameters.items()
    )
    head = f"{function.name}({params})"
    if function.description:
        return f"{head}: {function.description}"
    return head


def build_prompt(catalog: FunctionCatalog, request: str) -> str:
    """Build the compact production prompt for ``request`` (T8.4).

    Every model step re-reads the whole prompt because the public SDK exposes no
    KV cache, so prompt size is a direct multiplier on the run time. This format
    therefore replaces the verbose JSON catalog of T3.2 with one signature line
    per function: same names, parameters, types and descriptions, roughly a
    third of the tokens (see ``experiments/benchmark.md``).

    Functions are sorted by name, so the same catalog always produces the same
    bytes regardless of its declaration order. The request is only stripped;
    its content is never rewritten.
    """
    lines = [
        FUNCTIONS_BEGIN,
        *(_compact_signature(function) for function in _sorted_functions(catalog)),
        FUNCTIONS_END,
        "",
        f"Request: {request.strip()}",
        "",
        "Return one JSON object for the request using exactly one function.",
    ]
    return "\n".join(lines)


def _sorted_functions(catalog: FunctionCatalog) -> list[FunctionDefinition]:
    """Return the catalog functions ordered by name."""
    return sorted(catalog.functions, key=lambda function: function.name)


def build_verbose_prompt(catalog: FunctionCatalog, request: str) -> str:
    """Build the verbose T3.2 prompt used as the T8.4 comparison baseline.

    The production pipeline uses :func:`build_prompt` instead. This variant is
    kept verbatim so the benchmark can measure both formats on the same machine
    and so the size report can show the reduction.
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
