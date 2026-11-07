"""Fixture-driven tests for the JSON loaders (T1.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data_loader import DataLoadError, load_functions, load_prompts

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _assert_data_load_error(
    excinfo: pytest.ExceptionInfo[DataLoadError],
    *fragments: str,
) -> None:
    """Assert the error is exactly ``DataLoadError`` and contains ``fragments``."""
    assert type(excinfo.value) is DataLoadError
    message = str(excinfo.value)
    for fragment in fragments:
        assert fragment in message


def test_valid_functions_fixture_covers_all_parameter_types() -> None:
    catalog = load_functions(FIXTURES_DIR / "functions_valid_edge.json")

    assert len(catalog) == 5
    parameter_types = {
        spec.type
        for function in catalog
        for spec in function.parameters.values()
    }
    assert {"string", "number", "boolean", "integer"} <= parameter_types
    assert any(len(function.parameters) == 3 for function in catalog)


def test_valid_prompts_fixture_preserves_order_and_unicode() -> None:
    prompts = load_prompts(FIXTURES_DIR / "prompts_valid_edge.json")

    assert prompts == [
        "What is the sum of -12 and -7?",
        "Translate 'olá, mundo — \"42\"' into English.",
        "Disable the feature flag for dark mode.",
        "Handle the thing with 42 and 'banana'.",
    ]


@pytest.mark.parametrize(
    ("filename", "fragments"),
    [
        ("functions_empty.json", ("at least one function",)),
        ("functions_unknown_type.json", ("unsupported parameter type",)),
        ("functions_duplicate_name.json", ("duplicate",)),
        ("functions_missing_field.json", ("invalid function catalog",)),
        ("functions_root_object.json", ("invalid function catalog",)),
        ("functions_broken.json", ("invalid JSON", "line")),
    ],
)
def test_invalid_functions_fixtures_raise_data_load_error(
    filename: str,
    fragments: tuple[str, ...],
) -> None:
    path = FIXTURES_DIR / filename

    with pytest.raises(DataLoadError) as excinfo:
        load_functions(path)

    _assert_data_load_error(excinfo, *fragments)


@pytest.mark.parametrize(
    ("filename", "fragments"),
    [
        ("prompts_empty.json", ("at least one prompt",)),
        ("prompts_not_list.json", ("expected a JSON list",)),
        ("prompts_missing_key.json", ("index 1", "missing")),
        ("prompts_blank.json", ("blank",)),
        ("prompts_not_object.json", ("index 0", "must be an object")),
        ("prompts_broken.json", ("invalid JSON",)),
    ],
)
def test_invalid_prompts_fixtures_raise_data_load_error(
    filename: str,
    fragments: tuple[str, ...],
) -> None:
    path = FIXTURES_DIR / filename

    with pytest.raises(DataLoadError) as excinfo:
        load_prompts(path)

    _assert_data_load_error(excinfo, *fragments)
