"""Unit tests for the output-template DFA state model (T4.1)."""

from __future__ import annotations

import pytest

from src.dfa import (
    DFA,
    DFAState,
    ESCAPE_CHARS,
    LITERAL_CHARS,
    NUMBER_CHARS,
    STRUCTURAL_CHARS,
    TEMPLATE_KEYS,
    WHITESPACE_CHARS,
    is_string_char,
)

EXPECTED_STATES = {
    "START",
    "ROOT_OPEN",
    "PROMPT_KEY_DONE",
    "PROMPT_VALUE",
    "PROMPT_VALUE_DONE",
    "NAME_KEY",
    "NAME_KEY_DONE",
    "NAME_VALUE",
    "NAME_VALUE_DONE",
    "PARAMETERS_KEY",
    "PARAMETERS_KEY_DONE",
    "PARAMETERS_OPEN",
    "PARAMETERS_EMPTY_OR_KEY",
    "PARAM_KEY",
    "PARAM_KEY_DONE",
    "PARAM_VALUE_STRING",
    "PARAM_VALUE_NUMBER",
    "PARAM_VALUE_BOOLEAN",
    "PARAM_VALUE_DONE",
    "PARAM_KEY_START",
    "PARAMETERS_CLOSED",
    "ACCEPT",
}


def test_enum_contains_the_documented_states() -> None:
    names = {state.name for state in DFAState}

    assert len(names) >= 20
    assert EXPECTED_STATES <= names


def test_fresh_machine_starts_at_start_and_is_not_accepting() -> None:
    dfa = DFA()

    assert dfa.state is DFAState.START
    assert dfa.is_accepting() is False


def test_accept_is_the_only_accepting_state() -> None:
    for state in DFAState:
        dfa = DFA(state=state)
        assert dfa.is_accepting() == (state is DFAState.ACCEPT)


def test_accept_state_is_accepting() -> None:
    assert DFA(state=DFAState.ACCEPT).is_accepting() is True


def test_non_dfa_state_raises_clear_error() -> None:
    with pytest.raises(TypeError) as excinfo:
        DFA(state="ACCEPT")  # type: ignore[arg-type]

    message = str(excinfo.value)
    assert "DFAState" in message
    assert "str" in message


def test_start_returns_a_fresh_independent_machine() -> None:
    first = DFA.start()
    second = DFA.start()

    assert first is not second
    assert first.state is DFAState.START
    assert second.state is DFAState.START

    first.state = DFAState.ACCEPT
    assert first.is_accepting() is True
    assert second.state is DFAState.START


def test_template_keys_are_in_fixed_order() -> None:
    assert TEMPLATE_KEYS == ("prompt", "name", "parameters")


def test_structural_chars_cover_all_template_symbols() -> None:
    assert STRUCTURAL_CHARS == frozenset('{}[]:,"')


def test_alphabet_constants_have_the_documented_members() -> None:
    assert NUMBER_CHARS == frozenset("-+0123456789.eE")
    assert LITERAL_CHARS == frozenset("truefals")
    assert ESCAPE_CHARS == frozenset('"\\/bfnrtu')
    assert WHITESPACE_CHARS == frozenset(" \t\r\n")


def test_is_string_char_accepts_regular_and_unicode_content() -> None:
    assert is_string_char("a") is True
    assert is_string_char("á") is True


def test_is_string_char_rejects_quote_and_control_chars() -> None:
    assert is_string_char('"') is False
    assert is_string_char("\n") is False
    assert is_string_char("\t") is False


def test_is_string_char_requires_one_character() -> None:
    with pytest.raises(ValueError):
        is_string_char("ab")
