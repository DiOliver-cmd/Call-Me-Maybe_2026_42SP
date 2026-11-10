"""Unconstrained greedy baseline for function calling (T3.3).

This experiment measures what ``Qwen/Qwen3-0.6B`` does when every vocabulary
position is allowed: the next token is always ``argmax(logits)`` and generation
stops only at a stop id or at ``MAX_NEW_TOKENS``. These numbers are the
comparison line for the constrained decoder (T8.5).

The input files contain no ground-truth labels, so :data:`CASES` is a manual,
documented labeling of the first 10 prompts. Cases 8 and 9 use ``keys-only``
argument checking because several regex spellings are equivalent (``\\d+`` vs
``[0-9]+``): the required keys must exist, ``source_string`` and
``replacement`` must match exactly, and ``regex`` must be a non-empty string.

The prompt ids are kept in ``ids`` for the forward pass, but only the generated
suffix is decoded. Decoding the whole sequence would include the catalog JSON
and make the model output impossible to evaluate.

Run with::

    uv run python -m experiments.baseline
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.data_loader import load_functions, load_prompts
from src.llm_client import LLMClient
from src.prompt import build_verbose_prompt

FUNCTIONS_PATH = Path("data/input/functions_definition.json")
PROMPTS_PATH = Path("data/input/function_calling_tests.json")
OUTPUTS_PATH = Path("experiments/baseline_outputs.json")
REPORT_PATH = Path("experiments/baseline_report.md")

MAX_NEW_TOKENS = 128
DEFAULT_LIMIT = 10

# From T2.2: 151643 <|endoftext|>, 151644 <|im_start|>, 151645 <|im_end|>.
STOP_IDS: frozenset[int] = frozenset({151643, 151644, 151645})

ArgumentsMode = Literal["exact", "keys-only"]


@dataclass(frozen=True)
class BaselineCase:
    """One manually labeled prompt (the JSON files have no labels)."""

    prompt: str
    expected_function: str
    expected_arguments: dict[str, Any]
    arguments_mode: ArgumentsMode = "exact"


# First 10 prompts of ``function_calling_tests.json`` with manual labels.
CASES: tuple[BaselineCase, ...] = (
    BaselineCase(
        "What is the sum of 2 and 3?",
        "fn_add_numbers",
        {"a": 2, "b": 3},
    ),
    BaselineCase(
        "What is the sum of 265 and 345?",
        "fn_add_numbers",
        {"a": 265, "b": 345},
    ),
    BaselineCase(
        "Greet shrek",
        "fn_greet",
        {"name": "shrek"},
    ),
    BaselineCase(
        "Greet john",
        "fn_greet",
        {"name": "john"},
    ),
    BaselineCase(
        "Reverse the string 'hello'",
        "fn_reverse_string",
        {"s": "hello"},
    ),
    BaselineCase(
        "Reverse the string 'world'",
        "fn_reverse_string",
        {"s": "world"},
    ),
    BaselineCase(
        "What is the square root of 16?",
        "fn_get_square_root",
        {"a": 16},
    ),
    BaselineCase(
        "Calculate the square root of 144",
        "fn_get_square_root",
        {"a": 144},
    ),
    # Regex equivalence is ambiguous: any non-empty regex is accepted.
    BaselineCase(
        'Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS',
        "fn_substitute_string_with_regex",
        {
            "source_string": "Hello 34 I'm 233 years old",
            "regex": r"\d+",
            "replacement": "NUMBERS",
        },
        "keys-only",
    ),
    BaselineCase(
        "Replace all vowels in 'Programming is fun' with asterisks",
        "fn_substitute_string_with_regex",
        {
            "source_string": "Programming is fun",
            "regex": r"[aeiouAEIOU]",
            "replacement": "*",
        },
        "keys-only",
    ),
)


@dataclass(frozen=True)
class GenerationResult:
    """Raw greedy output and how generation ended."""

    generated_ids: list[int]
    generated_text: str
    stop_reason: str
    decode_error: str | None
    seconds: float


def _generate(
    client: LLMClient,
    prompt: str,
    max_new_tokens: int,
) -> GenerationResult:
    """Run the greedy loop until a stop id or ``max_new_tokens``."""
    started = time.perf_counter()
    ids = client.encode(prompt)
    generated_ids: list[int] = []
    stop_reason = "max_tokens"

    for _ in range(max_new_tokens):
        logits = client.logits(ids)
        next_id = int(np.argmax(logits))
        generated_ids.append(next_id)
        ids.append(next_id)
        if next_id in STOP_IDS:
            stop_reason = "eos"
            break

    seconds = time.perf_counter() - started

    try:
        generated_text = client.decode(generated_ids)
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        return GenerationResult(
            generated_ids=generated_ids,
            generated_text="",
            stop_reason="decode_error",
            decode_error=f"{type(exc).__name__}: {exc}",
            seconds=seconds,
        )

    return GenerationResult(
        generated_ids=generated_ids,
        generated_text=generated_text,
        stop_reason=stop_reason,
        decode_error=None,
        seconds=seconds,
    )


def _extract_object(text: str) -> dict[str, Any] | None:
    """Return the first JSON object with ``name`` and ``arguments``.

    Tries the whole text first, then the substring from the first ``{`` to the
    last ``}``. This keeps the check simple while tolerating short prose or
    markdown fences around an otherwise valid object.
    """
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if (
            isinstance(parsed, dict)
            and "name" in parsed
            and "arguments" in parsed
        ):
            return parsed
    return None


def _arguments_match(case: BaselineCase, arguments: Any) -> bool:
    """Compare arguments exactly, or keys-only for the regex cases."""
    if not isinstance(arguments, dict):
        return False

    if case.arguments_mode == "exact":
        return arguments == case.expected_arguments

    expected = case.expected_arguments
    if not all(key in arguments for key in expected):
        return False
    for key in ("source_string", "replacement"):
        if arguments[key] != expected[key]:
            return False
    regex = arguments["regex"]
    return isinstance(regex, str) and bool(regex)


def _run_case(
    client: LLMClient,
    catalog: Any,
    case: BaselineCase,
    index: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Generate and score one labeled case."""
    prompt = build_verbose_prompt(catalog, case.prompt)
    generation = _generate(client, prompt, max_new_tokens)
    parsed = _extract_object(generation.generated_text)

    function_correct = (
        parsed is not None
        and parsed.get("name") == case.expected_function
    )
    arguments_correct = (
        parsed is not None
        and _arguments_match(case, parsed.get("arguments"))
    )

    return {
        "index": index,
        "prompt": case.prompt,
        "expected_function": case.expected_function,
        "expected_arguments": case.expected_arguments,
        "arguments_mode": case.arguments_mode,
        "generated_ids": generation.generated_ids,
        "generated_text": generation.generated_text,
        "json_valid": parsed is not None,
        "function_correct": function_correct,
        "arguments_correct": arguments_correct,
        "stop_reason": generation.stop_reason,
        "decode_error": generation.decode_error,
        "tokens": len(generation.generated_ids),
        "seconds": round(generation.seconds, 3),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-case metrics."""
    count = len(rows)
    total_seconds = sum(float(row["seconds"]) for row in rows)
    return {
        "cases": count,
        "json_valid": sum(1 for row in rows if row["json_valid"]),
        "function_correct": sum(
            1 for row in rows if row["function_correct"]
        ),
        "arguments_correct": sum(
            1 for row in rows if row["arguments_correct"]
        ),
        "total_tokens": sum(int(row["tokens"]) for row in rows),
        "total_seconds": round(total_seconds, 3),
        "average_seconds": (
            round(total_seconds / count, 3) if count else 0.0
        ),
    }


def _cell(text: str, limit: int = 100) -> str:
    """Format arbitrary text as a compact, pipe-safe markdown code cell."""
    compact = text.replace("\n", "\\n").replace("\r", "\\r")
    if len(compact) > limit:
        compact = compact[: limit - 3] + "..."
    compact = compact.replace("|", "\\|").replace("`", "'")
    if not compact:
        return "`(empty)`"
    return f"`{compact}`"


def _write_report(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Write the committed markdown report."""
    lines = [
        "# Unconstrained greedy baseline (T3.3)",
        "",
        "Generated by `uv run python -m experiments.baseline`.",
        "",
        "## Configuration",
        "",
        "- Model: `Qwen/Qwen3-0.6B` via the project `LLMClient`",
        "- Decoding: greedy (`argmax`), no mask",
        f"- Max new tokens: `{config['max_new_tokens']}`",
        f"- Cases: `{config['limit']}` "
        f"(first prompts of `{PROMPTS_PATH}`)",
        "- Stop ids: `151643` (`<|endoftext|>`), `151644` (`<|im_start|>`), "
        "`151645` (`<|im_end|>`)",
        "- `seconds` covers `encode` plus all forward passes; decoding is "
        "excluded",
        "",
        "## Per-prompt results",
        "",
        "| # | prompt | JSON | function | args | tokens | seconds | stop | "
        "generated |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["index"]),
                    _cell(str(row["prompt"])),
                    "yes" if row["json_valid"] else "no",
                    "yes" if row["function_correct"] else "no",
                    "yes" if row["arguments_correct"] else "no",
                    str(row["tokens"]),
                    f"{row['seconds']:.2f}",
                    str(row["stop_reason"]),
                    _cell(str(row["generated_text"])),
                ]
            )
            + " |"
        )

    count = summary["cases"]
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- JSON valid: **{summary['json_valid']}/{count}**",
            f"- Function correct: **{summary['function_correct']}/{count}**",
            f"- Arguments correct: "
            f"**{summary['arguments_correct']}/{count}**",
            f"- Total generation time: **{summary['total_seconds']:.2f} s**",
            f"- Average per prompt: **{summary['average_seconds']:.2f} s**",
            f"- Total generated tokens: **{summary['total_tokens']}**",
            "",
            "## Constrained decoder comparison",
            "",
            "Pending (T8.5): the same prompts will be rerun with the DFA mask, "
            "and the constrained numbers will be compared with this "
            "unconstrained baseline.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the optional CLI overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="maximum number of tokens to generate per prompt",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="number of labeled cases to run",
    )
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.limit < 1 or args.limit > len(CASES):
        parser.error(f"--limit must be between 1 and {len(CASES)}")
    return args


def main(argv: list[str] | None = None) -> None:
    """Run the baseline and write the raw outputs and the report."""
    args = _parse_args(argv)
    catalog = load_functions(FUNCTIONS_PATH)
    prompts = load_prompts(PROMPTS_PATH)

    for index, case in enumerate(CASES):
        if prompts[index] != case.prompt:
            raise SystemExit(
                f"CASES[{index}] does not match prompt {index} in "
                f"{PROMPTS_PATH}: {case.prompt!r} != {prompts[index]!r}"
            )

    selected = CASES[: args.limit]
    client = LLMClient()
    print(
        f"Running {len(selected)} cases "
        f"(max_new_tokens={args.max_new_tokens})",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(selected):
        row = _run_case(
            client,
            catalog,
            case,
            index,
            args.max_new_tokens,
        )
        rows.append(row)
        print(
            f"[{index}] json={row['json_valid']} "
            f"fn={row['function_correct']} "
            f"args={row['arguments_correct']} "
            f"tokens={row['tokens']} "
            f"seconds={row['seconds']:.2f} "
            f"stop={row['stop_reason']}",
            flush=True,
        )

    summary = _summarize(rows)
    config = {
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "stop_ids": sorted(STOP_IDS),
    }
    payload = {"config": config, "cases": rows, "summary": summary}
    OUTPUTS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(rows, summary, config)

    print(
        f"JSON valid {summary['json_valid']}/{summary['cases']}, "
        f"function {summary['function_correct']}/{summary['cases']}, "
        f"arguments {summary['arguments_correct']}/{summary['cases']}, "
        f"total {summary['total_seconds']:.2f} s",
        flush=True,
    )
    print(f"Wrote {OUTPUTS_PATH} and {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
