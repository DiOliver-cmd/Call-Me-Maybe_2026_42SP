"""Unit tests for the JSON data loaders (T1.3, T1.5)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.data_loader import load_functions, load_prompts
from src.errors import InputError, ProjectError, SchemaError
from src.schemas import FunctionCatalog

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "input"
FUNCTIONS_FILE = DATA_DIR / "functions_definition.json"
PROMPTS_FILE = DATA_DIR / "function_calling_tests.json"


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _assert_project_error(
    excinfo: pytest.ExceptionInfo[ProjectError],
    expected_type: type[ProjectError],
    *fragments: str,
) -> None:
    """Assert the exact error type and that the message contains ``fragments``."""
    assert issubclass(expected_type, ProjectError)
    assert type(excinfo.value) is expected_type
    message = str(excinfo.value)
    for fragment in fragments:
        assert fragment in message


def test_load_real_functions() -> None:
    catalog = load_functions(FUNCTIONS_FILE)

    assert isinstance(catalog, FunctionCatalog)
    assert len(catalog) == 6


def test_load_real_prompts() -> None:
    prompts = load_prompts(PROMPTS_FILE)

    assert len(prompts) == 17
    assert prompts[0] == "What is the sum of 2 and 3?"


@pytest.mark.parametrize("loader", [load_functions, load_prompts])
def test_missing_file_raises_input_error(
    tmp_path: Path,
    loader: Callable[[str | Path], object],
) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(ProjectError) as excinfo:
        loader(missing)

    _assert_project_error(excinfo, InputError, str(missing), "not found")


def test_broken_json_raises_input_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ProjectError) as excinfo:
        load_functions(broken)

    _assert_project_error(excinfo, InputError, str(broken), "invalid JSON", "line")


@pytest.mark.parametrize("loader", [load_functions, load_prompts])
def test_directory_instead_of_file_raises_input_error(
    tmp_path: Path,
    loader: Callable[[str | Path], object],
) -> None:
    directory = tmp_path / "input"
    directory.mkdir()

    with pytest.raises(ProjectError) as excinfo:
        loader(directory)

    _assert_project_error(excinfo, InputError, str(directory), "directory")


def test_empty_functions_list_raises_input_error(tmp_path: Path) -> None:
    empty = _write_json(tmp_path / "functions.json", [])

    with pytest.raises(ProjectError) as excinfo:
        load_functions(empty)

    _assert_project_error(excinfo, InputError, str(empty), "at least one function")


def test_unknown_parameter_type_raises_schema_error(tmp_path: Path) -> None:
    payload = [
        {
            "name": "fn_bad",
            "description": "Uses a type the grammar cannot represent.",
            "parameters": {"items": {"type": "array"}},
            "returns": {"type": "string"},
        }
    ]
    bad = _write_json(tmp_path / "functions.json", payload)

    with pytest.raises(ProjectError) as excinfo:
        load_functions(bad)

    _assert_project_error(excinfo, SchemaError, str(bad), "supported types")


def test_functions_root_object_raises_schema_error(tmp_path: Path) -> None:
    bad = _write_json(tmp_path / "functions.json", {})

    with pytest.raises(ProjectError) as excinfo:
        load_functions(bad)

    _assert_project_error(excinfo, SchemaError, str(bad), "invalid function catalog")


def test_empty_prompts_list_raises_input_error(tmp_path: Path) -> None:
    empty = _write_json(tmp_path / "prompts.json", [])

    with pytest.raises(ProjectError) as excinfo:
        load_prompts(empty)

    _assert_project_error(excinfo, InputError, str(empty), "at least one prompt")


def test_prompts_root_not_list_raises_schema_error(tmp_path: Path) -> None:
    not_a_list = _write_json(tmp_path / "prompts.json", {})

    with pytest.raises(ProjectError) as excinfo:
        load_prompts(not_a_list)

    _assert_project_error(excinfo, SchemaError, str(not_a_list), "list", "dict")


def test_prompt_entry_not_object_raises_schema_error(tmp_path: Path) -> None:
    bad = _write_json(tmp_path / "prompts.json", ["hi"])

    with pytest.raises(ProjectError) as excinfo:
        load_prompts(bad)

    _assert_project_error(excinfo, SchemaError, "index 0", "object")


def test_prompt_missing_key_raises_schema_error(tmp_path: Path) -> None:
    payload = [{"prompt": "first prompt"}, {}]
    bad = _write_json(tmp_path / "prompts.json", payload)

    with pytest.raises(ProjectError) as excinfo:
        load_prompts(bad)

    _assert_project_error(excinfo, SchemaError, "index 1", "prompt")


def test_non_string_prompt_raises_schema_error(tmp_path: Path) -> None:
    bad = _write_json(tmp_path / "prompts.json", [{"prompt": 42}])

    with pytest.raises(ProjectError) as excinfo:
        load_prompts(bad)

    _assert_project_error(excinfo, SchemaError, "index 0", "string")


def test_blank_prompt_raises_input_error(tmp_path: Path) -> None:
    bad = _write_json(tmp_path / "prompts.json", [{"prompt": "   "}])

    with pytest.raises(ProjectError) as excinfo:
        load_prompts(bad)

    _assert_project_error(excinfo, InputError, "index 0", "blank")


def test_prompts_preserve_order_and_unicode(tmp_path: Path) -> None:
    payload = [
        {"prompt": "Olá, mundo!"},
        {"prompt": "Second prompt"},
        {"prompt": "第三"},
    ]
    path = _write_json(tmp_path / "prompts.json", payload)

    prompts = load_prompts(path)

    assert prompts == ["Olá, mundo!", "Second prompt", "第三"]
