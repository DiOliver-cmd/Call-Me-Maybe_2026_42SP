"""Deterministic tests for the T8.5 accuracy scorer."""

from __future__ import annotations

from typing import Any

from experiments.accuracy import score
from src.schemas import FunctionCall


def _call(name: str, parameters: dict[str, Any], prompt: str = "") -> FunctionCall:
    return FunctionCall(prompt=prompt, name=name, parameters=parameters)


def test_score_exact_and_numeric_values() -> None:
    result = score([_call("fn_add_numbers", {"a": 2.0, "b": 3})], [0])

    assert result["function_ok"] == 1
    assert result["arguments_ok"] == 1
    assert result["arguments_rate"] == 1.0


def test_score_rejects_string_contamination() -> None:
    result = score([_call("fn_reverse_string", {"s": "world}}"})], [5])

    assert result["function_ok"] == 1
    assert result["arguments_ok"] == 0
    assert "world}}" in result["rows"][0]["detail"]


def test_score_accepts_regex_alternatives() -> None:
    result = score(
        [
            _call(
                "fn_substitute_string_with_regex",
                {
                    "source_string": "Hello 34 I'm 233 years old",
                    "regex": "[0-9]+",
                    "replacement": "NUMBERS",
                },
            )
        ],
        [8],
    )

    assert result["arguments_ok"] == 1


def test_score_rejects_extra_keys_and_bool_as_number() -> None:
    extra = score([_call("fn_add_numbers", {"a": 2, "b": 3, "c": 4})], [0])
    boolean = score([_call("fn_add_numbers", {"a": True, "b": 3})], [0])

    assert extra["arguments_ok"] == 0
    assert boolean["arguments_ok"] == 0


def test_score_excludes_the_ambiguous_prompt() -> None:
    result = score([_call("fn_reverse_string", {"s": "banana 42"})], [16])

    assert result["defined"] == 0
    assert result["function_ok"] == 0
    assert result["arguments_ok"] == 0
    assert result["rows"][0]["function"] == "excluded"
