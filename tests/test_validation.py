"""Focused schema and call-validation matrix (T9.1).

The base models are exercised in ``test_schemas.py`` and
``test_function_call.py``. This module covers the validation edges that the
decoder's output relies on: catalog shape, strict scalar types, error
precedence and the batch boundary. Every test is deterministic and model-free.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from src.errors import SchemaError
from src.schemas import (
    FunctionCall,
    FunctionCallBatch,
    FunctionCatalog,
    FunctionDefinition,
)


def _function(
    name: str = "fn",
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "name": name,
        "description": "Synthetic function used by validation tests.",
        "returns": {"type": "string"},
    }
    if parameters is not None:
        raw["parameters"] = parameters
    else:
        raw["parameters"] = {}
    return raw


def _catalog(*functions: dict[str, Any]) -> FunctionCatalog:
    return FunctionCatalog.model_validate(list(functions))


def _call(
    name: str,
    parameters: dict[str, Any],
    prompt: str = "prompt",
) -> FunctionCall:
    return FunctionCall(prompt=prompt, name=name, parameters=parameters)


def test_catalog_accepts_wrapped_object_and_root_list() -> None:
    raw = [_function("fn_a"), _function("fn_b")]

    from_list = FunctionCatalog.from_list(raw)
    wrapped = FunctionCatalog.model_validate({"functions": raw})

    assert [function.name for function in from_list] == ["fn_a", "fn_b"]
    assert [function.name for function in wrapped] == ["fn_a", "fn_b"]


def test_catalog_rejects_extra_top_level_key() -> None:
    with pytest.raises(ValidationError):
        FunctionCatalog.model_validate(
            {"functions": [_function()], "unexpected": True}
        )


def test_duplicate_names_are_detected_after_stripping() -> None:
    with pytest.raises(ValidationError, match="fn_echo"):
        _catalog(_function("fn_echo"), _function("  fn_echo  "))


def test_parameter_type_is_stripped() -> None:
    catalog = _catalog(
        _function(parameters={"text": {"type": "  string  "}})
    )

    assert catalog.by_name("fn").parameters["text"].type == "string"


def test_blank_parameter_type_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        _catalog(_function(parameters={"text": {"type": "   "}}))


def test_parameter_description_must_be_a_string_or_none() -> None:
    catalog = _catalog(
        _function(
            parameters={
                "text": {"type": "string", "description": "human readable"}
            }
        )
    )

    assert (
        catalog.by_name("fn").parameters["text"].description
        == "human readable"
    )

    with pytest.raises(ValidationError):
        _catalog(
            _function(
                parameters={"text": {"type": "string", "description": 1}}
            )
        )


def test_return_type_uses_the_same_whitelist() -> None:
    raw = _function()
    raw["returns"] = {"type": "array"}

    with pytest.raises(ValidationError, match="supported types"):
        _catalog(raw)


def test_parameters_must_be_a_mapping() -> None:
    raw = _function()
    raw["parameters"] = []

    with pytest.raises(ValidationError):
        _catalog(raw)


@pytest.mark.parametrize("value", [2, 2.0])
def test_validate_call_number_accepts_int_and_float(value: Any) -> None:
    catalog = _catalog(_function(parameters={"n": {"type": "number"}}))
    call = _call("fn", {"n": value})

    assert catalog.validate_call(call) is call


@pytest.mark.parametrize("value", ["2", True])
def test_validate_call_number_rejects_string_and_bool(value: Any) -> None:
    catalog = _catalog(_function(parameters={"n": {"type": "number"}}))

    with pytest.raises(SchemaError, match="number"):
        catalog.validate_call(_call("fn", {"n": value}))


def test_validate_call_integer_accepts_int_only() -> None:
    catalog = _catalog(_function(parameters={"n": {"type": "integer"}}))

    assert catalog.validate_call(_call("fn", {"n": 2}))

    for value in (2.5, True):
        with pytest.raises(SchemaError, match="integer"):
            catalog.validate_call(_call("fn", {"n": value}))


def test_validate_call_lists_all_missing_parameters_sorted() -> None:
    catalog = _catalog(
        _function(
            parameters={
                "z": {"type": "string"},
                "a": {"type": "string"},
                "m": {"type": "string"},
            }
        )
    )

    with pytest.raises(SchemaError, match=r"a, m, z"):
        catalog.validate_call(_call("fn", {}))


def test_validate_call_lists_all_unexpected_parameters_sorted() -> None:
    catalog = _catalog(_function())

    with pytest.raises(SchemaError, match=r"a, z"):
        catalog.validate_call(_call("fn", {"z": 1, "a": 2}))


def test_validate_call_checks_name_before_parameters() -> None:
    catalog = _catalog(_function())

    with pytest.raises(SchemaError, match="not declared"):
        catalog.validate_call(_call("missing", {"unexpected": 1}))


def test_function_call_requires_prompt_and_defaults_parameters() -> None:
    with pytest.raises(ValidationError):
        FunctionCall.model_validate({"name": "fn"})

    call = FunctionCall(prompt="prompt", name="fn")

    assert call.parameters == {}


def test_function_call_rejects_non_string_prompt() -> None:
    with pytest.raises(ValidationError):
        FunctionCall.model_validate({"prompt": 1, "name": "fn"})


def test_validate_against_reports_missing_and_unexpected_together() -> None:
    function = FunctionDefinition.model_validate(
        _function(
            parameters={
                "a": {"type": "string"},
                "b": {"type": "string"},
            }
        )
    )
    call = _call("fn", {"a": "x", "c": "y"})

    with pytest.raises(ValueError) as excinfo:
        call.validate_against(function)

    message = str(excinfo.value)
    assert "missing parameter(s): b" in message
    assert "unexpected parameter(s): c" in message


def test_batch_rejects_extra_key_in_any_call() -> None:
    raw = [
        {"prompt": "p", "name": "fn", "parameters": {}},
        {
            "prompt": "p",
            "name": "fn",
            "parameters": {},
            "unexpected": True,
        },
    ]

    with pytest.raises(ValidationError):
        FunctionCallBatch.model_validate(raw)


def test_batch_accepts_empty_root_list() -> None:
    batch = FunctionCallBatch.model_validate([])

    assert batch.calls == []
    assert len(batch) == 0
