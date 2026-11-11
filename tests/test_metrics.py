"""Tests for the T9.4 metrics report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.metrics import build_fake_calls, collect_metrics, main


def test_fake_metrics_are_perfect() -> None:
    metrics = collect_metrics(build_fake_calls(), runtime_total_s=1.0)

    assert metrics.total == 17
    assert metrics.json_valid == 17
    assert metrics.schema_valid == 17
    assert metrics.function_ok == 16
    assert metrics.defined == 16
    assert metrics.arguments_ok == 16
    assert metrics.function_rate == 1.0
    assert metrics.arguments_rate == 1.0


def test_fake_report_is_written(tmp_path: Path) -> None:
    report = tmp_path / "fake.md"

    assert main(["--fake", "--report", str(report)]) == 0

    text = report.read_text(encoding="utf-8")
    assert "| JSON valid | 17/17 |" in text
    assert "| Schema valid | 17/17 |" in text
    assert "| Function accuracy | 16/16 (100.0%) |" in text
    assert "| Argument accuracy | 16/16 (100.0%) |" in text


def test_time_report_supplies_runtime(tmp_path: Path) -> None:
    output = tmp_path / "output.json"
    output.write_text(
        json.dumps(
            [call.model_dump(mode="json") for call in build_fake_calls()]
        ),
        encoding="utf-8",
    )
    time_report = tmp_path / "time.json"
    time_report.write_text('{"wall_total_s": 12.5}', encoding="utf-8")
    report = tmp_path / "report.md"

    assert (
        main(
            [
                "--output",
                str(output),
                "--time-report",
                str(time_report),
                "--report",
                str(report),
            ]
        )
        == 0
    )

    assert "| Runtime total | 12.50 s |" in report.read_text(encoding="utf-8")


def test_invalid_output_is_reported_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "bad.json"
    output.write_text("{not json", encoding="utf-8")

    code = main(["--output", str(output)])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err
