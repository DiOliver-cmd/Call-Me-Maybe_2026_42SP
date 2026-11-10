"""Unit tests for the schema models and call validation (T1.1, T6.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.errors import SchemaError
from src.schemas import FunctionCall, FunctionCatalog

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


def _minimal_function() -> dict[str, Any]:
    return {
        "name": "fn_echo",
        "description": "Echo a value back to the caller.",
        "returns": {"type": "string"},
    }


def _function_with(
    name: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    raw = _minimal_function()
    raw["name"] = name
    raw["parameters"] = parameters
    return raw


def _validation_catalog() -> FunctionCatalog:
    return FunctionCatalog.model_validate(
        [
            _function_with("fn_text", {"text": {"type": "string"}}),
            _function_with("fn_number", {"value": {"type": "number"}}),
            _function_with("fn_flag", {"enabled": {"type": "boolean"}}),
            _function_with("fn_zero", {}),
        ]
    )


def test_real_catalog_validates() -> None:
    catalog = FunctionCatalog.model_validate(_load_real_catalog())

    assert len(catalog) == 6
    names = [function.name for function in catalog]
    assert "fn_add_numbers" in names
    assert "fn_greet" in names
    assert "fn_set_alarm" in names
    assert catalog.by_name("fn_greet").parameters["name"].type == "string"


def test_minimal_function_schema_validates() -> None:
    catalog = FunctionCatalog.model_validate([_minimal_function()])

    function = catalog.by_name("fn_echo")
    assert function.parameters == {}
    assert function.returns.type == "string"


def test_duplicate_function_name_is_rejected() -> None:
    with pytest.raises(ValidationError, match="fn_echo"):
        FunctionCatalog.model_validate([_minimal_function(), _minimal_function()])


@pytest.mark.parametrize("missing", ["name", "description", "returns"])
def test_missing_required_field_is_rejected(missing: str) -> None:
    raw = _minimal_function()
    raw.pop(missing)

    with pytest.raises(ValidationError):
        FunctionCatalog.model_validate([raw])


def test_unknown_parameter_type_is_rejected() -> None:
    raw = _minimal_function()
    raw["parameters"] = {"items": {"type": "array"}}

    with pytest.raises(ValidationError, match="supported types"):
        FunctionCatalog.model_validate([raw])


def test_extra_unknown_field_is_rejected() -> None:
    raw = _minimal_function()
    raw["unexpected"] = True

    with pytest.raises(ValidationError):
        FunctionCatalog.model_validate([raw])


def test_blank_function_name_is_rejected() -> None:
    raw = _minimal_function()
    raw["name"] = "   "

    with pytest.raises(ValidationError, match="must not be empty"):
        FunctionCatalog.model_validate([raw])


def test_function_name_is_stripped() -> None:
    raw = _minimal_function()
    raw["name"] = "  fn_echo  "

    catalog = FunctionCatalog.model_validate([raw])

    assert catalog.by_name("fn_echo").name == "fn_echo"


def test_unknown_function_lookup_raises_key_error() -> None:
    catalog = FunctionCatalog.model_validate([_minimal_function()])

    with pytest.raises(KeyError, match="fn_missing"):
        catalog.by_name("fn_missing")


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        ("fn_text", {"text": "hello"}),
        ("fn_number", {"value": 2}),
        ("fn_number", {"value": 2.5}),
        ("fn_flag", {"enabled": True}),
        ("fn_zero", {}),
    ],
)
def test_validate_call_accepts_declared_signature(
    name: str,
    parameters: dict[str, Any],
) -> None:
    catalog = _validation_catalog()
    call = FunctionCall(prompt="prompt", name=name, parameters=parameters)

    assert catalog.validate_call(call) is call


def test_validate_call_rejects_unknown_name() -> None:
    catalog = _validation_catalog()
    call = FunctionCall(prompt="prompt", name="fn_missing", parameters={})

    with pytest.raises(
        SchemaError,
        match=r"function 'fn_missing' is not declared in the catalog",
    ):
        catalog.validate_call(call)


def test_validate_call_rejects_missing_parameter() -> None:
    catalog = _validation_catalog()
    call = FunctionCall(prompt="prompt", name="fn_text", parameters={})

    with pytest.raises(
        SchemaError,
        match=r"function 'fn_text' is missing required parameter\(s\): text",
    ):
        catalog.validate_call(call)


def test_validate_call_rejects_extra_parameter() -> None:
    catalog = _validation_catalog()
    call = FunctionCall(
        prompt="prompt",
        name="fn_zero",
        parameters={"extra": 1},
    )

    with pytest.raises(
        SchemaError,
        match=r"function 'fn_zero' received unexpected parameter\(s\): extra",
    ):
        catalog.validate_call(call)


def test_validate_call_rejects_bool_for_number() -> None:
    catalog = _validation_catalog()
    call = FunctionCall(
        prompt="prompt",
        name="fn_number",
        parameters={"value": True},
    )

    with pytest.raises(
        SchemaError,
        match=(
            r"function 'fn_number' parameter 'value' must be a number "
            r"\(int or float, not bool\), got bool"
        ),
    ):
        catalog.validate_call(call)


def test_validate_call_rejects_int_for_string() -> None:
    catalog = _validation_catalog()
    call = FunctionCall(
        prompt="prompt",
        name="fn_text",
        parameters={"text": 1},
    )

    with pytest.raises(
        SchemaError,
        match=(
            r"function 'fn_text' parameter 'text' must be a string, got int"
        ),
    ):
        catalog.validate_call(call)


def test_validate_call_rejects_str_for_boolean() -> None:
    catalog = _validation_catalog()
    call = FunctionCall(
        prompt="prompt",
        name="fn_flag",
        parameters={"enabled": "true"},
    )

    with pytest.raises(
        SchemaError,
        match=(
            r"function 'fn_flag' parameter 'enabled' must be a boolean, "
            r"got str"
        ),
    ):
        catalog.validate_call(call)
