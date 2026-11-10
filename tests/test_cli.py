"""CLI tests for ``python -m src`` without loading a model (T7.1)."""

from __future__ import annotations

import pytest

from src.__main__ import build_parser, main
from src.errors import DecodingError
from src.schemas import FunctionCall

_DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
_DEFAULT_INPUT = "data/input/function_calling_tests.json"
_DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def _calls() -> list[FunctionCall]:
    return [
        FunctionCall(
            prompt="What is the sum of 2 and 3?",
            name="fn_add_numbers",
            parameters={"a": 2, "b": 3},
        ),
        FunctionCall(prompt="Ping", name="fn_ping", parameters={}),
    ]


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[FunctionCall],
) -> dict[str, tuple[object, ...]]:
    seen: dict[str, tuple[object, ...]] = {}

    def fake_run(functions_path: str, input_path: str) -> list[FunctionCall]:
        seen["run"] = (functions_path, input_path)
        return calls

    def fake_write_output(path: str, written: list[FunctionCall]) -> None:
        seen["write_output"] = (path, written)

    monkeypatch.setattr("src.__main__.run", fake_run)
    monkeypatch.setattr("src.__main__.write_output", fake_write_output)
    return seen


def test_help_lists_the_three_flags_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    stdout = capsys.readouterr().out
    assert "--functions_definition" in stdout
    assert "--input" in stdout
    assert "--output" in stdout


def test_help_does_not_touch_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> list[FunctionCall]:
        raise AssertionError("run must not be called for --help")

    monkeypatch.setattr("src.__main__.run", fail)

    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0


def test_parser_uses_default_paths() -> None:
    args = build_parser().parse_args([])

    assert args.functions_definition == _DEFAULT_FUNCTIONS
    assert args.input == _DEFAULT_INPUT
    assert args.output == _DEFAULT_OUTPUT


def test_parser_accepts_custom_paths() -> None:
    args = build_parser().parse_args(
        [
            "--functions_definition",
            "custom/functions.json",
            "--input",
            "custom/prompts.json",
            "--output",
            "custom/out.json",
        ]
    )

    assert args.functions_definition == "custom/functions.json"
    assert args.input == "custom/prompts.json"
    assert args.output == "custom/out.json"


def test_main_uses_defaults_and_forwards_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _calls()
    seen = _install_fakes(monkeypatch, calls)

    assert main([]) == 0
    assert seen["run"] == (_DEFAULT_FUNCTIONS, _DEFAULT_INPUT)
    write_args = seen["write_output"]
    assert write_args[0] == _DEFAULT_OUTPUT
    assert write_args[1] is calls


def test_main_forwards_custom_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _calls()
    seen = _install_fakes(monkeypatch, calls)

    assert (
        main(
            [
                "--functions_definition",
                "custom/functions.json",
                "--input",
                "custom/prompts.json",
                "--output",
                "custom/out.json",
            ]
        )
        == 0
    )
    assert seen["run"] == ("custom/functions.json", "custom/prompts.json")
    write_args = seen["write_output"]
    assert write_args[0] == "custom/out.json"
    assert write_args[1] is calls


def test_main_reports_project_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_run(
        functions_path: str,
        input_path: str,
    ) -> list[FunctionCall]:
        raise DecodingError("boom")

    def fail_write(path: str, calls: list[FunctionCall]) -> None:
        raise AssertionError("write_output must not run after a failure")

    monkeypatch.setattr("src.__main__.run", fail_run)
    monkeypatch.setattr("src.__main__.write_output", fail_write)

    assert main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: boom\n"
    assert "Traceback" not in captured.err
