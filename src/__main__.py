"""Command-line entry point for ``python -m src`` (T7.1)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from src.errors import ProjectError, format_error
from src.pipeline import run, write_output

__all__ = ["build_parser", "main"]

DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser with the subject's default paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Run constrained function calling over the configured prompt file."
        )
    )
    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS,
        help="function catalog JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="prompt list JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="output JSON path (default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline and return a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        calls = run(args.functions_definition, args.input)
        write_output(args.output, calls)
    except ProjectError as exc:
        print(format_error(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
