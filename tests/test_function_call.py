"""Unit tests for the output-side schema models (T1.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.schemas import FunctionCall, FunctionCallBatch, FunctionDefinition

DATA_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "input"
    / "functions_definition.json"
)


def _load_real_catalog() -> list[dict[str, Any]]:
    with DATA_FILE.open(encoding="utf-8") as handle:
        data: list[dict[str, Any]] = json.load(handle)
    return data


def _find_function(name: str) -> FunctionDefinition:
    for raw in _load_real_catalog():
        function = FunctionDefinition.model_validate(raw)
        if function.name == name:
            return function
    raise AssertionError(f"function {name!r} not found in the catalog")


def _inline_function(parameters: dict[str, dict[str, str]]) -> FunctionDefinition:
    return FunctionDefinition.model_validate(
        {
            "name": "fn_inline",
            "description": "Inline function used only by these tests.",
            "parameters": parameters,
            "returns": {"type": "string"},
        }
    )


def _call(
    parameters: dict[str, Any],
    name: str = "fn_add_numbers",
) -> FunctionCall:
    return FunctionCall(prompt="a prompt", name=name, parameters=parameters)


def test_model_validate_round_trip_preserves_scalar_types() -> None:
    raw = {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {"s": "text", "i": 1, "f": 2.5, "b": True},
    }

    dumped = FunctionCall.model_validate(raw).model_dump()

    assert dumped == raw
    assert type(dumped["parameters"]["s"]) is str
    assert type(dumped["parameters"]["i"]) is int
    assert type(dumped["parameters"]["f"]) is float
    assert type(dumped["parameters"]["b"]) is bool


def test_extra_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        FunctionCall.model_validate(
            {
                "prompt": "a prompt",
                "name": "fn_add_numbers",
                "parameters": {},
                "unexpected": True,
            }
        )


def test_blank_name_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        FunctionCall.model_validate({"prompt": "a prompt", "name": "   "})


def test_batch_accepts_root_list_and_dumps_it_back() -> None:
    raw = [
        {
            "prompt": "What is the sum of 2 and 3?",
            "name": "fn_add_numbers",
            "parameters": {"a": 2, "b": 3},
        },
        {
            "prompt": "Greet shrek",
            "name": "fn_greet",
            "parameters": {"name": "shrek"},
        },
    ]

    batch = FunctionCallBatch.model_validate(raw)

    assert len(batch) == 2
    assert [call.name for call in batch] == ["fn_add_numbers", "fn_greet"]
    assert batch.calls is batch.root
    assert batch.model_dump() == raw


def test_batch_from_calls_wraps_validated_calls() -> None:
    call = _call({"a": 2, "b": 3})

    batch = FunctionCallBatch.from_calls([call])

    assert len(batch) == 1
    assert batch.calls == [call]


def test_validate_against_accepts_matching_number_call() -> None:
    function = _find_function("fn_add_numbers")
    call = _call({"a": 2, "b": 3.0})

    assert call.validate_against(function) is call


def test_validate_against_reports_missing_parameter() -> None:
    function = _find_function("fn_add_numbers")
    call = _call({"a": 2})

    with pytest.raises(ValueError, match=r"missing.*\bb\b"):
        call.validate_against(function)


def test_validate_against_reports_unexpected_parameter() -> None:
    function = _find_function("fn_add_numbers")
    call = _call({"a": 2, "b": 3, "c": 4})

    with pytest.raises(ValueError, match=r"unexpected.*\bc\b"):
        call.validate_against(function)


def test_validate_against_rejects_bool_for_number() -> None:
    function = _find_function("fn_add_numbers")
    call = _call({"a": True, "b": 3})

    with pytest.raises(ValueError, match="number") as excinfo:
        call.validate_against(function)

    message = str(excinfo.value)
    assert "bool" in message
    assert "'a'" in message


def test_validate_against_integer_rejects_float_and_bool() -> None:
    # The real catalog has no integer parameter, so declare one inline.
    function = _inline_function({"count": {"type": "integer"}})

    assert _call({"count": 2}, name="fn_inline").validate_against(function)

    with pytest.raises(ValueError, match="integer"):
        _call({"count": 2.5}, name="fn_inline").validate_against(function)

    with pytest.raises(ValueError, match="integer"):
        _call({"count": True}, name="fn_inline").validate_against(function)


def test_validate_against_boolean_parameter() -> None:
    function = _inline_function({"flag": {"type": "boolean"}})

    assert _call({"flag": True}, name="fn_inline").validate_against(function)

    with pytest.raises(ValueError, match="boolean"):
        _call({"flag": "true"}, name="fn_inline").validate_against(function)


def test_validate_against_string_rejects_int() -> None:
    function = _inline_function({"text": {"type": "string"}})

    with pytest.raises(ValueError, match="string"):
        _call({"text": 1}, name="fn_inline").validate_against(function)
