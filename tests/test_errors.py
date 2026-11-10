"""Unit tests for the error hierarchy and the CLI input-error matrix (T1.5, T8.1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.__main__ import main
from src.data_loader import load_functions
from src.errors import (
    EXIT_ERROR,
    EXIT_INTERRUPTED,
    EXIT_OK,
    DecodingError,
    InputError,
    ModelError,
    ProjectError,
    SchemaError,
    exit_code_for,
    format_error,
)

PROJECT_ERROR_TYPES: list[type[ProjectError]] = [
    InputError,
    SchemaError,
    DecodingError,
    ModelError,
]


def test_project_error_is_an_exception() -> None:
    assert issubclass(ProjectError, Exception)


@pytest.mark.parametrize("error_type", PROJECT_ERROR_TYPES)
def test_subclasses_inherit_from_project_error(
    error_type: type[ProjectError],
) -> None:
    assert issubclass(error_type, ProjectError)


def test_exit_code_constants() -> None:
    assert EXIT_OK == 0
    assert EXIT_ERROR == 1
    assert EXIT_INTERRUPTED == 130


@pytest.mark.parametrize("error_type", PROJECT_ERROR_TYPES)
def test_project_errors_map_to_error_exit_code(
    error_type: type[ProjectError],
) -> None:
    assert error_type.exit_code == EXIT_ERROR
    assert exit_code_for(error_type("boom")) == EXIT_ERROR


def test_keyboard_interrupt_maps_to_interrupted() -> None:
    assert exit_code_for(KeyboardInterrupt()) == EXIT_INTERRUPTED


def test_unexpected_exception_maps_to_error_code() -> None:
    assert exit_code_for(ValueError("boom")) == EXIT_ERROR


def test_format_error_prefixes_message() -> None:
    assert format_error(InputError("bad input")) == "error: bad input"
    assert format_error(ValueError("unexpected")) == "error: unexpected"


@pytest.mark.parametrize("error_type", PROJECT_ERROR_TYPES)
def test_messages_are_preserved_through_hierarchy(
    error_type: type[ProjectError],
) -> None:
    assert str(error_type("payload is invalid")) == "payload is invalid"


@pytest.mark.parametrize("error_type", PROJECT_ERROR_TYPES)
def test_catching_project_error_catches_subclasses(
    error_type: type[ProjectError],
) -> None:
    with pytest.raises(ProjectError):
        raise error_type("boom")


# --- T8.1: every bad input is a clear ``error:`` line, never a traceback. ---

_MINIMAL_FUNCTION: dict[str, Any] = {
    "name": "fn_echo",
    "description": "Echo a value back to the caller.",
    "parameters": {},
    "returns": {"type": "string"},
}


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_functions_file(tmp_path: Path) -> Path:
    return _write_json(tmp_path / "functions.json", [_MINIMAL_FUNCTION])


def _valid_prompts_file(tmp_path: Path) -> Path:
    return _write_json(tmp_path / "prompts.json", [{"prompt": "Ping"}])


def _run_cli_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    *fragments: str,
) -> str:
    """Run ``main`` expecting a CLI error and assert the shared contract.

    The model constructor is patched to fail loudly so a test that accidentally
    reaches model setup fails instead of silently loading weights or touching
    the network: every case here must fail in the loaders.
    """

    def fail_model(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "the model must not be constructed when input loading fails"
        )

    monkeypatch.setattr("src.pipeline.LLMClient", fail_model)

    assert main(argv) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err
    for fragment in fragments:
        assert fragment in captured.err
    return captured.err


def test_cli_missing_functions_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "nope.json"

    _run_cli_error(
        monkeypatch,
        capsys,
        ["--functions_definition", str(missing), "--input", str(_valid_prompts_file(tmp_path))],
        str(missing),
        "file not found",
    )


def test_cli_invalid_json_in_functions_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    broken = tmp_path / "functions.json"
    broken.write_text("{not valid json", encoding="utf-8")

    _run_cli_error(
        monkeypatch,
        capsys,
        ["--functions_definition", str(broken), "--input", str(_valid_prompts_file(tmp_path))],
        str(broken),
        "invalid JSON",
        "line",
    )


@pytest.mark.parametrize("payload", [[], {"functions": []}])
def test_cli_empty_functions_catalog(
    payload: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    empty = _write_json(tmp_path / "functions.json", payload)

    _run_cli_error(
        monkeypatch,
        capsys,
        ["--functions_definition", str(empty), "--input", str(_valid_prompts_file(tmp_path))],
        "must contain at least one function",
    )


def test_cli_function_missing_parameters_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    function = dict(_MINIMAL_FUNCTION)
    del function["parameters"]
    bad = _write_json(tmp_path / "functions.json", [function])

    _run_cli_error(
        monkeypatch,
        capsys,
        ["--functions_definition", str(bad), "--input", str(_valid_prompts_file(tmp_path))],
        "invalid function catalog",
        "parameters",
        "Field required",
    )


def test_cli_unknown_parameter_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    function = dict(_MINIMAL_FUNCTION)
    function["parameters"] = {"items": {"type": "object"}}
    bad = _write_json(tmp_path / "functions.json", [function])

    _run_cli_error(
        monkeypatch,
        capsys,
        ["--functions_definition", str(bad), "--input", str(_valid_prompts_file(tmp_path))],
        "unsupported parameter type",
        "object",
    )


def test_cli_empty_prompt_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    empty = _write_json(tmp_path / "prompts.json", [])

    _run_cli_error(
        monkeypatch,
        capsys,
        ["--functions_definition", str(_valid_functions_file(tmp_path)), "--input", str(empty)],
        "must contain at least one prompt",
    )


def test_cli_blank_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    blank = _write_json(tmp_path / "prompts.json", [{"prompt": "   "}])

    _run_cli_error(
        monkeypatch,
        capsys,
        ["--functions_definition", str(_valid_functions_file(tmp_path)), "--input", str(blank)],
        "must not be blank",
    )


def test_cli_input_path_is_a_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    directory = tmp_path / "prompts_dir"
    directory.mkdir()

    _run_cli_error(
        monkeypatch,
        capsys,
        ["--functions_definition", str(_valid_functions_file(tmp_path)), "--input", str(directory)],
        "expected a file but found a directory",
    )


def test_loader_names_the_missing_parameters_field(tmp_path: Path) -> None:
    function = dict(_MINIMAL_FUNCTION)
    del function["parameters"]
    bad = _write_json(tmp_path / "functions.json", [function])

    with pytest.raises(SchemaError, match=r"functions\.0\.parameters: Field required"):
        load_functions(bad)


def test_loader_accepts_explicit_empty_parameters(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "functions.json",
        [{**_MINIMAL_FUNCTION, "parameters": {}}],
    )

    catalog = load_functions(path)

    assert catalog.by_name("fn_echo").parameters == {}
