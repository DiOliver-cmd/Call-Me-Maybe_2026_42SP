"""Reproducible metrics report from a saved output (T9.4).

The report separates the guarantees:

* ``json_valid`` and ``schema_valid`` — structural validity of the written file;
* ``function_accuracy`` — the model picked the declared function;
* ``arguments_accuracy`` — every declared argument matches the ground truth;
* ``runtime_total`` — wall time of the measured run.

The ground truth and the per-prompt scoring live in
:mod:`experiments.accuracy`; this module only loads a saved output, validates
it and renders a Markdown report. The real output is produced by the pipeline
(``uv run python -m src``) or by ``--run-real``; ``--fake`` builds a perfect
deterministic batch for CI and for checking the report itself.

Examples::

    # Report from the saved production output (T8.4) and its measured runtime.
    uv run python -m experiments.metrics \\
        --output data/output/function_calls.json \\
        --runtime-seconds 180.24 \\
        --report experiments/report.md

    # Re-run the real pipeline, write a fresh output and measure the runtime.
    uv run python -m experiments.metrics --run-real \\
        --output data/output/function_calls.json \\
        --report experiments/report.md

    # Deterministic fake batch, no model.
    uv run python -m experiments.metrics --fake --report /tmp/fake_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.accuracy import EXPECTED, score
from src.data_loader import load_prompts
from src.errors import InputError, ProjectError, SchemaError
from src.pipeline import run
from src.schemas import FunctionCall, FunctionCallBatch

__all__ = [
    "Metrics",
    "build_fake_calls",
    "collect_metrics",
    "main",
    "render_report",
]

FUNCTIONS = Path("data/input/functions_definition.json")
PROMPTS = Path("data/input/function_calling_tests.json")
OUTPUT = Path("data/output/function_calls.json")


@dataclass
class Metrics:
    """The five numbers required by the task plus the per-prompt rows."""

    json_valid: int
    schema_valid: int
    total: int
    function_ok: int
    defined: int
    arguments_ok: int
    runtime_total_s: float | None
    rows: list[dict[str, Any]]

    @property
    def function_rate(self) -> float:
        return self.function_ok / self.defined if self.defined else 0.0

    @property
    def arguments_rate(self) -> float:
        return self.arguments_ok / self.defined if self.defined else 0.0


def build_fake_calls() -> list[FunctionCall]:
    """Build a deterministic perfect batch from :data:`EXPECTED`."""
    prompts = load_prompts(PROMPTS)
    calls: list[FunctionCall] = []
    for index, expected in enumerate(EXPECTED):
        if expected is None:
            calls.append(
                FunctionCall(
                    prompt=prompts[index],
                    name="fn_reverse_string",
                    parameters={"s": "banana 42"},
                )
            )
            continue
        parameters: dict[str, Any] = {}
        for key, value in expected.parameters.items():
            if key in expected.regex_alternatives:
                value = sorted(expected.regex_alternatives[key])[0]
            parameters[key] = value
        calls.append(
            FunctionCall(
                prompt=prompts[index],
                name=expected.name,
                parameters=parameters,
            )
        )
    return calls


def collect_metrics(
    calls: Sequence[FunctionCall],
    *,
    runtime_total_s: float | None,
) -> Metrics:
    """Score ``calls`` and package the five report numbers."""
    result = score(calls)
    return Metrics(
        json_valid=len(calls),
        schema_valid=len(calls),
        total=len(calls),
        function_ok=result["function_ok"],
        defined=result["defined"],
        arguments_ok=result["arguments_ok"],
        runtime_total_s=runtime_total_s,
        rows=result["rows"],
    )


def _load_output(path: Path) -> list[FunctionCall]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(f"cannot read output {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"output {path} is not valid JSON: {exc}") from exc

    try:
        batch = FunctionCallBatch.model_validate(raw)
    except Exception as exc:
        raise SchemaError(f"output {path} is not a valid call batch: {exc}") from exc

    calls = batch.calls
    if len(calls) != len(EXPECTED):
        raise InputError(
            f"output {path} has {len(calls)} call(s); "
            f"the fixture has {len(EXPECTED)} prompt(s)"
        )
    return calls


def _runtime_from_report(path: Path) -> float:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot read time report {path}: {exc}") from exc
    for key in ("wall_total_s", "runtime_total_s", "decode_total_s"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    raise InputError(
        f"time report {path} has none of wall_total_s, "
        "runtime_total_s or decode_total_s"
    )


def _write_output(path: Path, calls: Sequence[FunctionCall]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [call.model_dump(mode="json") for call in calls],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def render_report(
    metrics: Metrics,
    *,
    source: str,
    runtime_note: str = "",
) -> str:
    """Render the Markdown report for ``metrics`` and ``source``."""
    runtime = (
        f"{metrics.runtime_total_s:.2f} s"
        if metrics.runtime_total_s is not None
        else "not measured"
    )
    lines = [
        "# Metrics report (T9.4)",
        "",
        f"Source: `{source}`.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| JSON valid | {metrics.json_valid}/{metrics.total} |",
        f"| Schema valid | {metrics.schema_valid}/{metrics.total} |",
        f"| Function accuracy | {metrics.function_ok}/{metrics.defined} "
        f"({metrics.function_rate:.1%}) |",
        f"| Argument accuracy | {metrics.arguments_ok}/{metrics.defined} "
        f"({metrics.arguments_rate:.1%}) |",
        f"| Runtime total | {runtime} |",
        "",
    ]
    if runtime_note:
        lines.extend([runtime_note, ""])

    lines.extend(
        [
            "## Per-prompt",
            "",
            "| # | prompt | expected | predicted | function | arguments | detail |",
            "| ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in metrics.rows:
        prompt = str(row["prompt"]).replace("|", "\\|")
        detail = str(row["detail"]).replace("|", "\\|")
        lines.append(
            f"| {row['index']} | {prompt} | {row['expected']} | "
            f"{row['predicted']} | {row['function']} | "
            f"{row['arguments']} | {detail} |"
        )

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "# Score the saved production output without loading the model.",
            "uv run python -m experiments.metrics \\",
            "  --output data/output/function_calls.json \\",
            "  --runtime-seconds 180.24 \\",
            "  --report experiments/report.md",
            "",
            "# Re-run the real pipeline and measure the runtime.",
            "uv run python -m experiments.metrics --run-real \\",
            "  --output data/output/function_calls.json \\",
            "  --report experiments/report.md",
            "",
            "# Deterministic fake batch (no model).",
            "uv run python -m experiments.metrics --fake \\",
            "  --report /tmp/fake_report.md",
            "```",
            "",
            "The ground truth and the prompt-variant study are in",
            "`experiments/accuracy.py` / `experiments/accuracy.md`; the",
            "performance budget is in `experiments/benchmark.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="saved output JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="write the Markdown report to this path",
    )
    parser.add_argument(
        "--runtime-seconds",
        type=float,
        default=None,
        help="measured wall time for the run that produced the output",
    )
    parser.add_argument(
        "--time-report",
        type=Path,
        default=None,
        help="JSON from experiments.time_budget (wall_total_s is used)",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="score a deterministic perfect batch instead of the output file",
    )
    parser.add_argument(
        "--run-real",
        action="store_true",
        help="run the real pipeline, write --output and measure the runtime",
    )
    parser.add_argument(
        "--functions",
        type=Path,
        default=FUNCTIONS,
        help="function catalog used by --run-real",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROMPTS,
        help="prompt fixture used by --run-real",
    )
    args = parser.parse_args(argv)

    try:
        if args.fake:
            calls = build_fake_calls()
            runtime: float | None = args.runtime_seconds
            source = "deterministic fake batch"
            runtime_note = (
                "The fake batch is a report check, not a model measurement."
            )
        elif args.run_real:
            started = time.perf_counter()
            calls = run(args.functions, args.input)
            runtime = time.perf_counter() - started
            _write_output(args.output, calls)
            source = f"real pipeline run written to {args.output}"
            runtime_note = "Runtime measured by this invocation."
        else:
            calls = _load_output(args.output)
            runtime = args.runtime_seconds
            if runtime is None and args.time_report is not None:
                runtime = _runtime_from_report(args.time_report)
            source = f"saved output {args.output}"
            runtime_note = (
                "Runtime is the T8.4 CLI measurement (180.24 s); use "
                "`--run-real` or `--time-report` to measure a fresh run."
                if args.runtime_seconds is not None
                else ""
            )

        metrics = collect_metrics(calls, runtime_total_s=runtime)
        report = render_report(
            metrics,
            source=source,
            runtime_note=runtime_note,
        )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report, encoding="utf-8")
        print(report)
        return 0
    except ProjectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
