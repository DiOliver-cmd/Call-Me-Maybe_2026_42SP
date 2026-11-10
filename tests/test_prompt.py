"""Unit tests for prompt and output-template construction (T3.2/T4.5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.data_loader import load_functions
from src.dfa import DFA
from src.llm_client import LLMClient
from src.prompt import (
    FUNCTIONS_BEGIN,
    FUNCTIONS_END,
    TemplateSegment,
    apply_fixed,
    build_prompt,
    build_segments,
    build_template,
    count_prompt_tokens,
)
from src.schemas import (
    FunctionCatalog,
    FunctionDefinition,
    ParameterSpec,
)
from tests.fakes import FakeLLM

REPO_ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS_PATH = REPO_ROOT / "data" / "input" / "functions_definition.json"


@pytest.fixture(scope="module")
def catalog() -> FunctionCatalog:
    """Load the real function catalog once for all prompt tests."""
    return load_functions(FUNCTIONS_PATH)


@pytest.fixture
def template_catalog() -> FunctionCatalog:
    """Small catalog with ordered multi-parameter and empty functions."""
    return FunctionCatalog.from_list(
        [
            {
                "name": "fn_greet",
                "description": "Greet someone.",
                "parameters": {"name": {"type": "string"}},
                "returns": {"type": "string"},
            },
            {
                "name": "fn_add_numbers",
                "description": "Add two numbers.",
                "parameters": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "returns": {"type": "number"},
            },
            {
                "name": "fn_ping",
                "description": "Ping.",
                "parameters": {},
                "returns": {"type": "string"},
            },
        ]
    )


def _fake_tokenize(text: str) -> list[int]:
    """Return one deterministic token id per character."""
    return [ord(char) for char in text]


def _fill_segments(
    segments: tuple[TemplateSegment, ...],
    name: str,
    parameters: dict[str, Any],
) -> str:
    """Join fixed text and JSON-compatible slot fills."""
    pieces: list[str] = []
    name_slot_seen = False
    for segment in segments:
        if segment.kind == "fixed":
            pieces.append(segment.text)
            continue
        slot = segment.slot
        assert slot is not None
        if not name_slot_seen:
            assert slot == "name"
            pieces.append(name)
            name_slot_seen = True
        else:
            pieces.append(
                json.dumps(parameters[slot], ensure_ascii=False)
            )
    return "".join(pieces)


def _inject_segments(
    segments: tuple[TemplateSegment, ...],
    name: str,
    parameters: dict[str, Any],
) -> DFA:
    """Inject fixed segments and slot fills into a fresh DFA."""
    dfa = DFA.start([name])
    name_slot_seen = False
    for segment in segments:
        if segment.kind == "fixed":
            assert apply_fixed(dfa, segment) is True
            continue

        slot = segment.slot
        assert slot is not None
        if not name_slot_seen:
            assert slot == "name"
            text = name
            name_slot_seen = True
        else:
            text = json.dumps(parameters[slot], ensure_ascii=False)
        assert dfa.consume_token(text) is True
    return dfa


def _functions_json_block(prompt: str) -> str:
    start = prompt.index(FUNCTIONS_BEGIN) + len(FUNCTIONS_BEGIN)
    end = prompt.index(FUNCTIONS_END)
    return prompt[start:end].strip()


def test_prompt_includes_every_function_name_and_request(
    catalog: FunctionCatalog,
) -> None:
    request = "Greet shrek"
    prompt = build_prompt(catalog, request)

    for function in catalog:
        assert function.name in prompt
    assert request in prompt


def test_prompt_is_byte_identical_for_same_input(
    catalog: FunctionCatalog,
) -> None:
    assert build_prompt(catalog, "Greet shrek") == build_prompt(
        catalog, "Greet shrek"
    )


def test_prompt_ignores_catalog_order(catalog: FunctionCatalog) -> None:
    reversed_catalog = FunctionCatalog(
        functions=list(reversed(catalog.functions))
    )

    assert build_prompt(catalog, "Greet shrek") == build_prompt(
        reversed_catalog, "Greet shrek"
    )


@pytest.mark.parametrize("count", [1, 3, None])
def test_prompt_works_with_function_subsets(
    catalog: FunctionCatalog,
    count: int | None,
) -> None:
    functions = (
        list(catalog.functions)
        if count is None
        else list(catalog.functions[:count])
    )
    subset = FunctionCatalog(functions=functions)
    prompt = build_prompt(subset, "Do the useful thing with 42 and 'banana'.")

    included = {function.name for function in functions}
    for function in functions:
        assert function.name in prompt
    for function in catalog.functions:
        if function.name not in included:
            assert function.name not in prompt


def test_embedded_json_block_round_trips(catalog: FunctionCatalog) -> None:
    prompt = build_prompt(catalog, "Greet shrek")
    expected = [
        function.model_dump(mode="json")
        for function in sorted(
            catalog.functions,
            key=lambda function: function.name,
        )
    ]

    payload = json.loads(_functions_json_block(prompt))

    assert payload == expected
    assert [item["name"] for item in payload] == sorted(
        function.name for function in catalog.functions
    )
    assert _functions_json_block(prompt) == json.dumps(
        expected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_prompt_preserves_non_ascii_and_strips_only() -> None:
    function = FunctionDefinition(
        name="fn_acento",
        description="Ação rápida — não remova",
        parameters={"texto": ParameterSpec(type="string")},
        returns=ParameterSpec(type="string"),
    )
    catalog = FunctionCatalog(functions=[function])
    request = 'olá, mundo — "42"'

    prompt = build_prompt(catalog, f"  {request}  ")

    assert "Ação rápida — não remova" in prompt
    assert request in prompt
    assert "\\u00e7" not in prompt
    assert "\\u2014" not in prompt
    assert prompt == build_prompt(catalog, request)


def test_prompt_asks_for_one_json_object(catalog: FunctionCatalog) -> None:
    prompt = build_prompt(catalog, "Greet shrek")

    assert '"name"' in prompt
    assert '"arguments"' in prompt
    assert "ONE JSON object" in prompt


def test_count_prompt_tokens_delegates_to_encode() -> None:
    fake = FakeLLM(
        encode_result=[10, 20, 30],
        decode_result="",
        logits_result=[0.0],
        vocab_path_text="vocab.json",
    )

    assert count_prompt_tokens(LLMClient(backend=fake), "hello") == 3
    assert fake.encode_calls == ["hello"]


def test_build_template_has_exact_known_string(
    template_catalog: FunctionCatalog,
) -> None:
    result = build_template(
        template_catalog,
        "Greet bob",
        "fn_greet",
        {"name": "bob"},
    )

    assert result == (
        '{"prompt":"Greet bob","name":"fn_greet",'
        '"parameters":{"name":"bob"}}'
    )


def test_build_template_round_trips_with_fixed_key_order(
    template_catalog: FunctionCatalog,
) -> None:
    result = build_template(
        template_catalog,
        "Greet bob",
        "fn_greet",
        {"name": "bob"},
    )
    parsed = json.loads(result)

    assert parsed == {
        "prompt": "Greet bob",
        "name": "fn_greet",
        "parameters": {"name": "bob"},
    }
    assert list(parsed) == ["prompt", "name", "parameters"]


def test_build_template_preserves_prompt_quotes_and_unicode(
    template_catalog: FunctionCatalog,
) -> None:
    prompt = 'She said "olá" — \\ path'
    result = build_template(
        template_catalog,
        prompt,
        "fn_greet",
        {"name": "bob"},
    )

    assert json.loads(result)["prompt"] == prompt
    assert "\\u00e1" not in result
    assert "\\u2014" not in result


def test_build_template_with_empty_parameters(
    template_catalog: FunctionCatalog,
) -> None:
    result = build_template(template_catalog, "Ping", "fn_ping", {})

    assert result == '{"prompt":"Ping","name":"fn_ping","parameters":{}}'


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        ("fn_missing", {}),
        ("fn_greet", {}),
        ("fn_greet", {"name": "bob", "extra": 1}),
    ],
)
def test_build_template_rejects_bad_name_or_parameter_keys(
    template_catalog: FunctionCatalog,
    name: str,
    parameters: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        build_template(template_catalog, "request", name, parameters)


def test_build_template_uses_declaration_order_and_is_deterministic(
    template_catalog: FunctionCatalog,
) -> None:
    first = build_template(
        template_catalog,
        "Add",
        "fn_add_numbers",
        {"b": 2, "a": 1},
    )
    second = build_template(
        template_catalog,
        "Add",
        "fn_add_numbers",
        {"a": 1, "b": 2},
    )

    assert first == second
    assert first == (
        '{"prompt":"Add","name":"fn_add_numbers",'
        '"parameters":{"a":1,"b":2}}'
    )


def test_build_segments_fill_reproduces_template(
    template_catalog: FunctionCatalog,
) -> None:
    parameters = {"a": 1, "b": 2}
    segments = build_segments(
        template_catalog,
        "Add",
        "fn_add_numbers",
        _fake_tokenize,
    )

    assert _fill_segments(
        segments,
        "fn_add_numbers",
        parameters,
    ) == build_template(
        template_catalog,
        "Add",
        "fn_add_numbers",
        parameters,
    )


def test_build_segments_fill_reproduces_empty_parameter_template(
    template_catalog: FunctionCatalog,
) -> None:
    segments = build_segments(
        template_catalog,
        "Ping",
        "fn_ping",
        _fake_tokenize,
    )

    assert _fill_segments(segments, "fn_ping", {}) == build_template(
        template_catalog,
        "Ping",
        "fn_ping",
        {},
    )


def test_build_segments_tokenizes_only_fixed_text(
    template_catalog: FunctionCatalog,
) -> None:
    calls: list[str] = []

    def tokenize(text: str) -> list[int]:
        calls.append(text)
        return [ord(char) for char in text]

    segments = build_segments(
        template_catalog,
        "Add",
        "fn_add_numbers",
        tokenize,
    )
    fixed = [segment for segment in segments if segment.kind == "fixed"]

    assert calls == [segment.text for segment in fixed]
    assert all(segment.text for segment in fixed)
    for segment in fixed:
        assert segment.token_ids == tuple(
            ord(char) for char in segment.text
        )


def test_build_segments_slots_follow_declaration_order(
    template_catalog: FunctionCatalog,
) -> None:
    segments = build_segments(
        template_catalog,
        "Add",
        "fn_add_numbers",
        _fake_tokenize,
    )
    slots = [segment for segment in segments if segment.kind == "slot"]

    assert [segment.slot for segment in slots] == ["name", "a", "b"]
    assert all(segment.text == "" for segment in slots)
    assert all(segment.token_ids == () for segment in slots)


def test_build_segments_fixed_sequence_is_stable(
    template_catalog: FunctionCatalog,
) -> None:
    segments = build_segments(
        template_catalog,
        "Add",
        "fn_add_numbers",
        _fake_tokenize,
    )

    assert [segment.text for segment in segments if segment.kind == "fixed"] == [
        '{"prompt":',
        '"Add"',
        ',"name":"',
        '","parameters":{',
        '"a":',
        ",",
        '"b":',
        "}}",
    ]


def test_build_segments_are_deterministic(
    template_catalog: FunctionCatalog,
) -> None:
    first = build_segments(
        template_catalog,
        "Add",
        "fn_add_numbers",
        _fake_tokenize,
    )
    second = build_segments(
        template_catalog,
        "Add",
        "fn_add_numbers",
        _fake_tokenize,
    )

    assert first == second


def test_template_segments_inject_into_dfa(
    template_catalog: FunctionCatalog,
) -> None:
    parameters = {"name": "bob"}
    segments = build_segments(
        template_catalog,
        "Greet bob",
        "fn_greet",
        _fake_tokenize,
    )

    dfa = _inject_segments(segments, "fn_greet", parameters)

    assert dfa.is_accepting() is True


def test_empty_parameter_template_segments_inject_into_dfa(
    template_catalog: FunctionCatalog,
) -> None:
    segments = build_segments(
        template_catalog,
        "Ping",
        "fn_ping",
        _fake_tokenize,
    )

    dfa = _inject_segments(segments, "fn_ping", {})

    assert dfa.is_accepting() is True


def test_apply_fixed_rejects_a_slot_segment() -> None:
    dfa = DFA.start()

    with pytest.raises(ValueError):
        apply_fixed(
            dfa,
            TemplateSegment(kind="slot", text="", slot="name"),
        )
