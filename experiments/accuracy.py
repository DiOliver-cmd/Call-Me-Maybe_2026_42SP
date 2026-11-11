"""Measure function-selection and argument accuracy on the prompt fixture (T8.5).

Constrained decoding guarantees a valid JSON shape, but not that the model
picked the right function or copied the right argument values. This report
scores the real pipeline against a hand-written ground truth and compares
prompt formats, so any prompt change is justified by numbers instead of
intuition.

Ground truth decisions:

* 16 of the 17 fixture prompts have a defined function and arguments.
* Prompt 16 (``Do the useful thing with 42 and 'banana'.``) is intentionally
  ambiguous: there is no correct function, and the decoder must emit one. It is
  audited separately and excluded from both denominators.
* Regex prompts accept equivalent patterns (``\\d+``, ``[0-9]+``,
  ``([0-9]+)``); the replacement is compared exactly.
* Strings are compared exactly. A trailing ``}}`` or prose copied from the
  prompt is a failure, because it changes the value the caller receives.

Run with::

    uv run python -m experiments.accuracy --score-only data/output/function_calling_results.json
    uv run python -m experiments.accuracy --prompt values
    uv run python -m experiments.accuracy --prompt values --indices 5,8,9,10,14,15

The last form is the fast feedback loop: only the string-heavy prompts are
decoded, which keeps an iteration near one minute instead of three.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data_loader import load_prompts
from src.pipeline import run
from src.prompt import build_prompt, build_verbose_prompt
from src.schemas import FunctionCall, FunctionCatalog

__all__ = [
    "EXPECTED",
    "PROMPT_BUILDERS",
    "build_minimal_prompt",
    "build_roles_prompt",
    "build_values_prompt",
    "main",
    "score",
]

FUNCTIONS = Path("data/input/functions_definition.json")
PROMPTS = Path("data/input/function_calling_tests.json")


@dataclass(frozen=True)
class Expected:
    """The accepted answer for one prompt.

    ``regex_alternatives`` lists equivalent patterns for the ``regex``
    parameter; every other value is compared exactly (numbers by value).
    """

    name: str
    parameters: dict[str, Any]
    regex_alternatives: dict[str, frozenset[str]] = field(default_factory=dict)


# Index 16 is the ambiguous prompt; ``None`` means "no defined answer".
EXPECTED: tuple[Expected | None, ...] = (
    Expected("fn_add_numbers", {"a": 2, "b": 3}),
    Expected("fn_add_numbers", {"a": 265, "b": 345}),
    Expected("fn_greet", {"name": "shrek"}),
    Expected("fn_greet", {"name": "john"}),
    Expected("fn_reverse_string", {"s": "hello"}),
    Expected("fn_reverse_string", {"s": "world"}),
    Expected("fn_get_square_root", {"a": 16}),
    Expected("fn_get_square_root", {"a": 144}),
    Expected(
        "fn_substitute_string_with_regex",
        {
            "source_string": "Hello 34 I'm 233 years old",
            "regex": r"\d+",
            "replacement": "NUMBERS",
        },
        {"regex": frozenset({r"\d+", r"[0-9]+", r"([0-9]+)", r"\d"})},
    ),
    Expected(
        "fn_substitute_string_with_regex",
        {
            "source_string": "Programming is fun",
            "regex": "[aeiou]",
            "replacement": "*",
        },
        {"regex": frozenset({"[aeiou]", "aeiou", "[aeiouAEIOU]"})},
    ),
    Expected(
        "fn_substitute_string_with_regex",
        {
            "source_string": "The cat sat on the mat with another cat",
            "regex": "cat",
            "replacement": "dog",
        },
        {"regex": frozenset({"cat"})},
    ),
    Expected("fn_set_alarm", {"enabled": True}),
    Expected("fn_set_alarm", {"enabled": False}),
    Expected("fn_add_numbers", {"a": -12, "b": 7}),
    Expected("fn_reverse_string", {"s": 'olá, mundo — "42"'}),
    Expected(
        "fn_substitute_string_with_regex",
        {
            "source_string": "Room 101, floor 2",
            "regex": r"\d+",
            "replacement": "#",
        },
        {"regex": frozenset({r"\d+", r"[0-9]+", r"([0-9]+)", r"\d"})},
    ),
    None,
)


def _signature(function: Any) -> str:
    """Render ``name(param:type, ...): description`` like the production prompt."""
    params = ", ".join(
        f"{name}:{spec.type}" for name, spec in function.parameters.items()
    )
    head = f"{function.name}({params})"
    return f"{head}: {function.description}" if function.description else head


def build_values_prompt(catalog: FunctionCatalog, request: str) -> str:
    """Prompt the model for slot values only, not for a whole JSON object.

    The decoder injects ``{"prompt":...,"name":"...","parameters":{...}}``;
    asking the model to "return one JSON object" made it close the object inside
    string slots (``world}}``) and copy prose into values. This variant states
    that the structure is already written.
    """
    lines = [
        "<functions>",
        *(_signature(function) for function in sorted(
            catalog.functions, key=lambda function: function.name
        )),
        "</functions>",
        "",
        f"Request: {request.strip()}",
        "",
        "Choose exactly one function from the list.",
        "The output JSON structure is already written. Fill only the function "
        "name and the parameter values.",
        "Do not add surrounding quotes, braces, keys, or explanations.",
        "String values are raw text: emit the text itself and keep any quotes "
        "that belong to it.",
        "For regex, emit only the pattern, never the replacement.",
    ]
    return "\n".join(lines)


def build_roles_prompt(catalog: FunctionCatalog, request: str) -> str:
    """Variant with explicit parameter roles and a stop cue for strings."""
    lines = [
        "<functions>",
        *(_signature(function) for function in sorted(
            catalog.functions, key=lambda function: function.name
        )),
        "</functions>",
        "",
        f"Request: {request.strip()}",
        "",
        "The JSON is already written for you. Output only the function name and "
        "the parameter values.",
        "For a string value, copy the text exactly as it appears in the request, "
        "then stop; do not add quotes, braces, keys, or explanations.",
        "Parameter roles: source_string is the text to modify; regex is the "
        "pattern to match; replacement is the text that replaces each match.",
    ]
    return "\n".join(lines)


def build_minimal_prompt(catalog: FunctionCatalog, request: str) -> str:
    """Shortest variant: one sentence about the pre-written JSON skeleton."""
    lines = [
        "<functions>",
        *(_signature(function) for function in sorted(
            catalog.functions, key=lambda function: function.name
        )),
        "</functions>",
        "",
        f"Request: {request.strip()}",
        "",
        "The JSON is already written for you. Output only the function name and "
        "the parameter values, with no quotes, braces, keys, or extra text.",
    ]
    return "\n".join(lines)


PROMPT_BUILDERS: dict[str, Callable[[FunctionCatalog, str], str]] = {
    "compact": build_prompt,
    "minimal": build_minimal_prompt,
    "roles": build_roles_prompt,
    "values": build_values_prompt,
    "verbose": build_verbose_prompt,
}


def _value_ok(
    key: str,
    actual: Any,
    expected: Any,
    alternatives: dict[str, frozenset[str]],
) -> bool:
    if key in alternatives:
        return isinstance(actual, str) and actual in alternatives[key]
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and float(actual) == float(expected)
        )
    return bool(actual == expected)


def _parameters_ok(actual: Any, expected: Expected) -> tuple[bool, str]:
    if not isinstance(actual, dict):
        return False, f"parameters is {type(actual).__name__}, not dict"
    if set(actual) != set(expected.parameters):
        return False, f"keys {sorted(actual)} != {sorted(expected.parameters)}"
    for key, wanted in expected.parameters.items():
        if not _value_ok(
            key,
            actual[key],
            wanted,
            expected.regex_alternatives,
        ):
            return False, f"{key}={actual[key]!r} != {wanted!r}"
    return True, ""


def score(
    calls: Sequence[FunctionCall],
    indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Score ``calls`` against :data:`EXPECTED` and return the table + totals."""
    selected = list(indices) if indices is not None else list(range(len(calls)))
    rows: list[dict[str, Any]] = []
    function_ok = 0
    arguments_ok = 0
    defined = 0

    for call, index in zip(calls, selected):
        expected = EXPECTED[index]
        if expected is None:
            rows.append(
                {
                    "index": index,
                    "prompt": call.prompt,
                    "expected": "excluded (ambiguous)",
                    "predicted": call.name,
                    "function": "excluded",
                    "arguments": "excluded",
                    "detail": "",
                }
            )
            continue
        defined += 1
        name_ok = call.name == expected.name
        params_ok, detail = _parameters_ok(call.parameters, expected)
        function_ok += int(name_ok)
        arguments_ok += int(params_ok)
        rows.append(
            {
                "index": index,
                "prompt": call.prompt,
                "expected": expected.name,
                "predicted": call.name,
                "function": "ok" if name_ok else "FAIL",
                "arguments": "ok" if params_ok else "FAIL",
                "detail": detail,
            }
        )

    return {
        "rows": rows,
        "defined": defined,
        "function_ok": function_ok,
        "arguments_ok": arguments_ok,
        "function_rate": function_ok / defined if defined else 0.0,
        "arguments_rate": arguments_ok / defined if defined else 0.0,
    }


def _print_report(result: dict[str, Any], style: str) -> None:
    print(f"\n## Prompt: `{style}`\n")
    print("| # | prompt | expected | predicted | function | arguments | detail |")
    print("| ---: | --- | --- | --- | --- | --- | --- |")
    for row in result["rows"]:
        prompt = row["prompt"].replace("|", "\\|")
        detail = row["detail"].replace("|", "\\|")
        print(
            f"| {row['index']} | {prompt} | {row['expected']} | "
            f"{row['predicted']} | {row['function']} | "
            f"{row['arguments']} | {detail} |"
        )
    print()
    print(
        f"- Function: **{result['function_ok']}/{result['defined']}** "
        f"({result['function_rate']:.1%})"
    )
    print(
        f"- Arguments: **{result['arguments_ok']}/{result['defined']}** "
        f"({result['arguments_rate']:.1%})"
    )


def _parse_indices(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(part) for part in raw.split(",")]


def _run_selected(
    style: str,
    indices: Sequence[int] | None,
) -> list[FunctionCall]:
    builder = PROMPT_BUILDERS[style]
    prompts = load_prompts(PROMPTS)
    if indices is None:
        return run(FUNCTIONS, PROMPTS, prompt_builder=builder)
    selected = [{"prompt": prompts[index]} for index in indices]
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(selected, handle, ensure_ascii=False)
        temp_path = Path(handle.name)
    try:
        return run(FUNCTIONS, temp_path, prompt_builder=builder)
    finally:
        temp_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        choices=sorted(PROMPT_BUILDERS),
        default="values",
        help="prompt format to run (default: %(default)s)",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="comma-separated prompt indices (default: all)",
    )
    parser.add_argument(
        "--score-only",
        type=Path,
        default=None,
        help="score an existing output JSON instead of running the model",
    )
    parser.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="write the decoded calls to this JSON path",
    )
    args = parser.parse_args(argv)

    indices = _parse_indices(args.indices)

    if args.score_only is not None:
        payload = json.loads(args.score_only.read_text(encoding="utf-8"))
        calls = [FunctionCall.model_validate(item) for item in payload]
    else:
        calls = _run_selected(args.prompt, indices)
        if args.dump is not None:
            args.dump.write_text(
                json.dumps(
                    [call.model_dump(mode="json") for call in calls],
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

    result = score(calls, indices)
    _print_report(result, args.prompt if args.score_only is None else "saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
