"""Tests for the small-model string artifact normalizer."""

from __future__ import annotations

from src.normalize import normalize_parameter_value


def test_removes_template_closing_artifact() -> None:
    assert normalize_parameter_value("s", "world}}") == "world"


def test_removes_backspace_and_trailing_word_boundary() -> None:
    raw = "(\\x08[0-9]+)\\b"

    assert normalize_parameter_value("regex", raw) == "([0-9]+)"


def test_wraps_bare_vowel_set() -> None:
    assert normalize_parameter_value("regex", "aeiouAEIOU") == "[aeiouAEIOU]"


def test_collapses_repeated_literal_regex() -> None:
    assert normalize_parameter_value("regex", "cat.*cat.*") == "cat"


def test_removes_replacement_tail() -> None:
    assert normalize_parameter_value("replacement", "*.*") == "*"


def test_removes_unmatched_delimiter_quotes() -> None:
    assert (
        normalize_parameter_value("s", 'olá, mundo — "42"\'')
        == 'olá, mundo — "42"'
    )
    assert normalize_parameter_value("replacement", '#"') == "#"


def test_leaves_valid_values_and_non_strings_unchanged() -> None:
    assert normalize_parameter_value("s", "hello") == "hello"
    assert normalize_parameter_value("number", 2) == 2
    assert normalize_parameter_value("enabled", True) is True
