"""Edge cases for strings, numbers, booleans and ambiguous prompts (T9.3).

The fake tests drive the real decoder and DFA with character-token scripts, so
they are deterministic and model-free. One real-model case is marked ``slow``
and uses the production prompt builder and number-EOS policy.
"""

from __future__ import annotations

import pytest

from src.data_loader import load_functions
from src.decoder import NUMBER_EOS_BONUS, decode_call
from src.llm_client import LLMClient
from src.pipeline import build_decoder_index
from src.schemas import FunctionCall, FunctionCatalog
from src.vocab import TokenizerIndex
from tests.fakes import ScriptedLLM

_EOS_ID = 1000
_EOS_TEXT = "<|im_end|>"
_FUNCTIONS = "data/input/functions_definition.json"

# Characters used by the edge values and the synthetic function names.
_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    '{}[]:,"\\ _+-.eE'
    "áé"
)


def _tokens(text: str) -> list[int]:
    """Return one character token id per character."""
    return [ord(char) for char in text]


def _index() -> TokenizerIndex:
    texts = {ord(char): char for char in sorted(set(_CHARS))}
    texts[_EOS_ID] = _EOS_TEXT
    allowed_ids = tuple(
        token_id for token_id in sorted(texts) if token_id != _EOS_ID
    )
    buckets: dict[str, list[int]] = {}
    for token_id in allowed_ids:
        text = texts[token_id]
        if text:
            buckets.setdefault(text[0], []).append(token_id)
    return TokenizerIndex(
        texts=texts,
        allowed_ids=allowed_ids,
        by_first_char={
            char: tuple(token_ids) for char, token_ids in buckets.items()
        },
    )


def _catalog() -> FunctionCatalog:
    return FunctionCatalog.from_list(
        [
            {
                "name": "fn_text",
                "description": "Return text.",
                "parameters": {"text": {"type": "string"}},
                "returns": {"type": "string"},
            },
            {
                "name": "fn_number",
                "description": "Return a number.",
                "parameters": {"value": {"type": "number"}},
                "returns": {"type": "number"},
            },
            {
                "name": "fn_integer",
                "description": "Return an integer.",
                "parameters": {"count": {"type": "integer"}},
                "returns": {"type": "integer"},
            },
            {
                "name": "fn_flag",
                "description": "Return a boolean.",
                "parameters": {"enabled": {"type": "boolean"}},
                "returns": {"type": "boolean"},
            },
            {
                "name": "fn_mixed",
                "description": "One value of each scalar type.",
                "parameters": {
                    "label": {"type": "string"},
                    "amount": {"type": "number"},
                    "active": {"type": "boolean"},
                },
                "returns": {"type": "string"},
            },
            {
                "name": "fn_three",
                "description": "Three strings.",
                "parameters": {
                    "first": {"type": "string"},
                    "second": {"type": "string"},
                    "third": {"type": "string"},
                },
                "returns": {"type": "string"},
            },
            {
                "name": "fn_greet",
                "description": "Greet someone.",
                "parameters": {"name": {"type": "string"}},
                "returns": {"type": "string"},
            },
            {
                "name": "fn_reverse",
                "description": "Reverse a string.",
                "parameters": {"s": {"type": "string"}},
                "returns": {"type": "string"},
            },
            {
                "name": "fn_zero",
                "description": "No parameters.",
                "parameters": {},
                "returns": {"type": "string"},
            },
        ]
    )


def _decode(script: list[int], prompt: str = "prompt") -> FunctionCall:
    model = LLMClient(backend=ScriptedLLM(script))
    return decode_call(model, _catalog(), prompt, index=_index())


STRING_CASES = [
    ("empty", '""', ""),
    ("escaped quote", '"a\\"b"', 'a"b'),
    ("escaped backslash", '"a\\\\b"', "a\\b"),
    ("escaped newline", '"a\\nb"', "a\nb"),
    ("literal unicode", '"olá"', "olá"),
    ("unicode escape", '"\\u00e9"', "é"),
]


@pytest.mark.parametrize(("label", "encoded", "expected"), STRING_CASES)
def test_string_edge_values(
    label: str,
    encoded: str,
    expected: str,
) -> None:
    script = _tokens("fn_text") + [_EOS_ID] + _tokens(encoded)

    call = _decode(script)

    assert call.parameters == {"text": expected}
    assert type(call.parameters["text"]) is str


NUMBER_CASES = [
    ("int", "42", 42),
    ("negative", "-7", -7),
    ("float", "2.5", 2.5),
    ("negative exponent", "-1.5e+2", -150.0),
    ("exponent", "6.02e23", 6.02e23),
    ("large int", "12345678901234567890", 12345678901234567890),
]


@pytest.mark.parametrize(("label", "encoded", "expected"), NUMBER_CASES)
def test_number_edge_values(
    label: str,
    encoded: str,
    expected: int | float,
) -> None:
    script = (
        _tokens("fn_number")
        + [_EOS_ID]
        + _tokens(encoded)
        + [_EOS_ID]
    )

    call = _decode(script)

    assert call.parameters == {"value": expected}
    assert type(call.parameters["value"]) is type(expected)


@pytest.mark.parametrize(("encoded", "expected"), [("true", True), ("false", False)])
def test_boolean_edge_values(encoded: str, expected: bool) -> None:
    script = (
        _tokens("fn_flag")
        + [_EOS_ID]
        + _tokens(encoded)
        + [_EOS_ID]
    )

    call = _decode(script)

    assert call.parameters == {"enabled": expected}
    assert type(call.parameters["enabled"]) is bool


def test_three_parameters_keep_declaration_order() -> None:
    script = (
        _tokens("fn_three")
        + [_EOS_ID]
        + _tokens('"one"')
        + _tokens('"two"')
        + _tokens('"three"')
    )

    call = _decode(script)

    assert call.parameters == {
        "first": "one",
        "second": "two",
        "third": "three",
    }


def test_mixed_scalar_parameters_are_typed() -> None:
    script = (
        _tokens("fn_mixed")
        + [_EOS_ID]
        + _tokens('"x"')
        + _tokens("2")
        + [_EOS_ID]
        + _tokens("true")
        + [_EOS_ID]
    )

    call = _decode(script)

    assert call.parameters == {"label": "x", "amount": 2, "active": True}
    assert type(call.parameters["label"]) is str
    assert type(call.parameters["amount"]) is int
    assert type(call.parameters["active"]) is bool


def test_ambiguous_prompt_follows_the_model_not_keywords() -> None:
    """The prompt says "greet", but the model chose reverse; no heuristic wins."""
    script = _tokens("fn_reverse") + [_EOS_ID] + _tokens('"shrek"')

    call = _decode(script, prompt="Greet shrek")

    assert call.name == "fn_reverse"
    assert call.parameters == {"s": "shrek"}


def test_prompt_without_a_matching_function_still_produces_a_valid_call() -> None:
    """The catalog has no poem function; the decoder must still emit one of ours."""
    script = _tokens("fn_zero") + [_EOS_ID]

    call = _decode(script, prompt="Write a haiku about the sea.")

    assert call.name == "fn_zero"
    assert call.parameters == {}
    assert _catalog().validate_call(call) is call


def test_integer_parameter_accepts_a_large_int() -> None:
    script = (
        _tokens("fn_integer")
        + [_EOS_ID]
        + _tokens("12345678901234567890")
        + [_EOS_ID]
    )

    call = _decode(script)

    assert call.parameters == {"count": 12345678901234567890}
    assert type(call.parameters["count"]) is int


@pytest.mark.slow
def test_real_model_one_edge_prompt_is_schema_valid() -> None:
    """One real-model case for T9.3; the rest of the module stays model-free."""
    client = LLMClient()
    index = build_decoder_index(client)
    catalog = load_functions(_FUNCTIONS)

    call = decode_call(
        client,
        catalog,
        "What is the sum of 2 and 3?",
        index=index,
        number_eos_bonus=NUMBER_EOS_BONUS,
    )

    assert call.name == "fn_add_numbers"
    assert catalog.validate_call(call) is call
    assert type(call.parameters["a"]) in (int, float)
    assert type(call.parameters["b"]) in (int, float)
