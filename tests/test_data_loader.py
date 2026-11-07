"""Unit tests for the JSON data loaders (T1.3)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.data_loader import DataLoadError, load_functions, load_prompts
from src.schemas import FunctionCatalog

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "input"
FUNCTIONS_FILE = DATA_DIR / "functions_definition.json"
PROMPTS_FILE = DATA_DIR / "function_calling_tests.json"


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _assert_data_load_error(
    excinfo: pytest.ExceptionInfo[DataLoadError],
    *fragments: str,
) -> None:
    """Assert the error is exactly ``DataLoadError`` and contains ``fragments``."""
    assert type(excinfo.value) is DataLoadError
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
def test_missing_file_raises_data_load_error(
    tmp_path: Path,
    loader: Callable[[str | Path], object],
) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(DataLoadError) as excinfo:
        loader(missing)

    _assert_data_load_error(excinfo, str(missing), "not found")


def test_broken_json_raises_data_load_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(DataLoadError) as excinfo:
        load_functions(broken)

    _assert_data_load_error(excinfo, str(broken), "invalid JSON", "line")


@pytest.mark.parametrize("loader", [load_functions, load_prompts])
def test_directory_instead_of_file_raises_data_load_error(
    tmp_path: Path,
    loader: Callable[[str | Path], object],
) -> None:
    directory = tmp_path / "input"
    directory.mkdir()

    with pytest.raises(DataLoadError) as excinfo:
        loader(directory)

    _assert_data_load_error(excinfo, str(directory), "directory")


def test_empty_functions_list_is_rejected(tmp_path: Path) -> None:
    empty = _write_json(tmp_path / "functions.json", [])

    with pytest.raises(DataLoadError) as excinfo:
        load_functions(empty)

    _assert_data_load_error(excinfo, str(empty), "at least one function")


def test_unknown_parameter_type_mentions_path(tmp_path: Path) -> None:
    payload = [
        {
            "name": "fn_bad",
            "description": "Uses a type the grammar cannot represent.",
            "parameters": {"items": {"type": "array"}},
            "returns": {"type": "string"},
        }
    ]
    bad = _write_json(tmp_path / "functions.json", payload)

    with pytest.raises(DataLoadError) as excinfo:
        load_functions(bad)

    _assert_data_load_error(excinfo, str(bad), "supported types")


def test_functions_root_must_validate(tmp_path: Path) -> None:
    bad = _write_json(tmp_path / "functions.json", {})

    with pytest.raises(DataLoadError) as excinfo:
        load_functions(bad)

    _assert_data_load_error(excinfo, str(bad), "invalid function catalog")


def test_empty_prompts_list_is_rejected(tmp_path: Path) -> None:
    empty = _write_json(tmp_path / "prompts.json", [])

    with pytest.raises(DataLoadError) as excinfo:
        load_prompts(empty)

    _assert_data_load_error(excinfo, str(empty), "at least one prompt")


def test_prompts_root_must_be_a_list(tmp_path: Path) -> None:
    not_a_list = _write_json(tmp_path / "prompts.json", {})

    with pytest.raises(DataLoadError) as excinfo:
        load_prompts(not_a_list)

    _assert_data_load_error(excinfo, str(not_a_list), "list", "dict")


def test_prompt_entry_must_be_an_object(tmp_path: Path) -> None:
    bad = _write_json(tmp_path / "prompts.json", ["hi"])

    with pytest.raises(DataLoadError) as excinfo:
        load_prompts(bad)

    _assert_data_load_error(excinfo, "index 0", "object")


def test_prompt_entry_missing_key_names_index(tmp_path: Path) -> None:
    payload = [{"prompt": "first prompt"}, {}]
    bad = _write_json(tmp_path / "prompts.json", payload)

    with pytest.raises(DataLoadError) as excinfo:
        load_prompts(bad)

    _assert_data_load_error(excinfo, "index 1", "prompt")


def test_prompt_entry_must_be_a_string(tmp_path: Path) -> None:
    bad = _write_json(tmp_path / "prompts.json", [{"prompt": 42}])

    with pytest.raises(DataLoadError) as excinfo:
        load_prompts(bad)

    _assert_data_load_error(excinfo, "index 0", "string")


def test_blank_prompt_is_rejected(tmp_path: Path) -> None:
    bad = _write_json(tmp_path / "prompts.json", [{"prompt": "   "}])

    with pytest.raises(DataLoadError) as excinfo:
        load_prompts(bad)

    _assert_data_load_error(excinfo, "index 0", "blank")


def test_prompts_preserve_order_and_unicode(tmp_path: Path) -> None:
    payload = [
        {"prompt": "Olá, mundo!"},
        {"prompt": "Second prompt"},
        {"prompt": "第三"},
    ]
    path = _write_json(tmp_path / "prompts.json", payload)

    prompts = load_prompts(path)

    assert prompts == ["Olá, mundo!", "Second prompt", "第三"]
