"""Unit tests for deterministic function-calling prompt construction (T3.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data_loader import load_functions
from src.llm_client import LLMClient
from src.prompt import (
    FUNCTIONS_BEGIN,
    FUNCTIONS_END,
    build_prompt,
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
