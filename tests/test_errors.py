"""Unit tests for the error hierarchy and the CLI error matrix (T1.5, T8.1, T8.2)."""

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
from src.llm_client import LLMClient
from src.schemas import FunctionCall
from src.vocab import TokenizerIndex
from tests.fakes import ScriptedLLM

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


# --- T8.2: SDK/model failures are controlled ProjectErrors, never tracebacks. ---

_EOS_ID = 1000
_EOS_TEXT = "<|im_end|>"
_CHARS = "abcdefghijklmnopqrstuvwxyz_\" 0123456789.-"

_GREET_FUNCTION: dict[str, Any] = {
    "name": "fn_greet",
    "description": "Greet someone.",
    "parameters": {"name": {"type": "string"}},
    "returns": {"type": "string"},
}


def _tokens(text: str) -> list[int]:
    """Return one token id per character, matching ``_char_index``."""
    return [ord(char) for char in text]


def _char_index() -> TokenizerIndex:
    """Build a small char-token index that maps the EOS marker."""
    texts = {ord(char): char for char in sorted(set(_CHARS))}
    allowed_ids = tuple(sorted(texts))
    texts[_EOS_ID] = _EOS_TEXT
    buckets: dict[str, list[int]] = {}
    for token_id in allowed_ids:
        buckets.setdefault(texts[token_id][0], []).append(token_id)
    return TokenizerIndex(
        texts=texts,
        allowed_ids=allowed_ids,
        by_first_char={
            char: tuple(token_ids) for char, token_ids in buckets.items()
        },
    )


def _decoding_files(tmp_path: Path) -> tuple[Path, Path]:
    """Write a valid two-function catalog and a single prompt."""
    functions = _write_json(
        tmp_path / "functions.json",
        [{**_MINIMAL_FUNCTION, "name": "fn_ping"}, _GREET_FUNCTION],
    )
    prompts = _write_json(tmp_path / "prompts.json", [{"prompt": "Ping"}])
    return functions, prompts


def _install_scripted_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    backend: Any,
) -> None:
    """Inject ``backend`` and a decoded index without touching assets."""
    index = _char_index()
    monkeypatch.setattr(
        "src.pipeline.LLMClient",
        lambda: LLMClient(backend=backend),
    )
    monkeypatch.setattr(
        "src.pipeline.build_decoder_index",
        lambda *args, **kwargs: index,
    )


class _FailingScripted(ScriptedLLM):
    """Scripted backend whose configured public method raises ``RuntimeError``."""

    def __init__(self, script: list[int], *, method: str) -> None:
        super().__init__(script)
        self.method = method

    def encode(self, text: str) -> list[list[int]]:
        if self.method == "encode":
            raise RuntimeError("backend encode exploded")
        return super().encode(text)

    def decode(self, ids: list[int]) -> str:
        if self.method == "decode":
            raise RuntimeError("backend decode exploded")
        return super().decode(ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        if self.method == "logits":
            raise RuntimeError("backend logits exploded")
        return super().get_logits_from_input_ids(input_ids)


_METHOD_CALLS: dict[str, Any] = {
    "encode": lambda client: client.encode("x"),
    "logits": lambda client: client.logits([1]),
    "decode": lambda client: client.decode([1]),
}


@pytest.mark.parametrize("method", ["encode", "logits", "decode"])
def test_backend_failure_is_wrapped_as_model_error(method: str) -> None:
    client = LLMClient(backend=_FailingScripted([], method=method))

    with pytest.raises(ModelError) as excinfo:
        _METHOD_CALLS[method](client)

    assert type(excinfo.value) is ModelError
    assert f"{method} failed" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


@pytest.mark.parametrize("method", ["encode", "logits", "decode"])
def test_backend_failure_via_main_exits_one_without_traceback(
    method: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    functions, prompts = _decoding_files(tmp_path)
    output = tmp_path / "out.json"
    script = _tokens("fn_ping") + [_EOS_ID]
    backend = _FailingScripted(script, method=method)
    _install_scripted_pipeline(monkeypatch, backend)

    code = main(
        [
            "--functions_definition",
            str(functions),
            "--input",
            str(prompts),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert f"{method} failed" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()


def test_lazy_model_load_failure_is_a_model_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_construct(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no weights available")

    monkeypatch.setattr("llm_sdk.Small_LLM_Model", fail_construct)

    with pytest.raises(ModelError) as excinfo:
        LLMClient().encode("x")

    assert type(excinfo.value) is ModelError
    assert str(excinfo.value) == (
        "failed to load the model: no weights available"
    )
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_lazy_model_load_failure_via_main_exits_one_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    functions, prompts = _decoding_files(tmp_path)
    output = tmp_path / "out.json"

    def fail_construct(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no weights available")

    monkeypatch.setattr("llm_sdk.Small_LLM_Model", fail_construct)

    code = main(
        [
            "--functions_definition",
            str(functions),
            "--input",
            str(prompts),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert "failed to load the model" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()


def test_all_masked_slot_via_main_exits_one_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    functions, prompts = _decoding_files(tmp_path)
    output = tmp_path / "out.json"
    backend = ScriptedLLM(
        _tokens("fn_ping") + [_EOS_ID],
        all_masked_at=1,
    )
    _install_scripted_pipeline(monkeypatch, backend)

    code = main(
        [
            "--functions_definition",
            str(functions),
            "--input",
            str(prompts),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "no valid token" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()


def test_max_tokens_via_main_exits_one_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    functions = _write_json(tmp_path / "functions.json", [_GREET_FUNCTION])
    prompts = _write_json(
        tmp_path / "prompts.json",
        [{"prompt": "Greet someone"}],
    )
    output = tmp_path / "out.json"
    script = (
        _tokens("fn_greet")
        + [_EOS_ID]
        + _tokens('"')
        + _tokens("a" * 50)
    )
    backend = ScriptedLLM(script)
    _install_scripted_pipeline(monkeypatch, backend)

    code = main(
        [
            "--functions_definition",
            str(functions),
            "--input",
            str(prompts),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "MAX_TOKENS" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()


def test_invalid_final_validation_via_main_exits_one_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    functions, prompts = _decoding_files(tmp_path)
    output = tmp_path / "out.json"
    _install_scripted_pipeline(monkeypatch, ScriptedLLM([]))

    def invalid_decode_call(*args: object, **kwargs: object) -> FunctionCall:
        return FunctionCall(prompt="Ping", name="fn_undeclared", parameters={})

    monkeypatch.setattr("src.pipeline.decode_call", invalid_decode_call)

    code = main(
        [
            "--functions_definition",
            str(functions),
            "--input",
            str(prompts),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "is not declared in the catalog" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()


def test_mid_batch_failure_leaves_existing_output_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "out.json"
    sentinel = '{"sentinel": true}\n'
    output.write_text(sentinel, encoding="utf-8")

    def fail_run(*args: object, **kwargs: object) -> list[FunctionCall]:
        raise DecodingError("prompt 1 failed mid-batch")

    monkeypatch.setattr("src.__main__.run", fail_run)

    code = main(
        [
            "--functions_definition",
            str(tmp_path / "functions.json"),
            "--input",
            str(tmp_path / "prompts.json"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "prompt 1 failed mid-batch" in captured.err
    assert "Traceback" not in captured.err
    assert output.read_text(encoding="utf-8") == sentinel
