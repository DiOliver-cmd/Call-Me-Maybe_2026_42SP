"""Unit tests for the output-template DFA states, transitions and slots (T4.1-T4.3)."""

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
    NameMatcher,
    is_complete_number,
    is_number_prefix,
    is_string_char,
    transition,
)

EXPECTED_STATES = {
    "START",
    "ROOT_OPEN",
    "PROMPT_KEY_STRING",
    "PROMPT_KEY_DONE",
    "PROMPT_VALUE_START",
    "PROMPT_VALUE",
    "PROMPT_VALUE_ESCAPE",
    "PROMPT_VALUE_DONE",
    "NAME_KEY",
    "NAME_KEY_STRING",
    "NAME_KEY_DONE",
    "NAME_VALUE_START",
    "NAME_VALUE",
    "NAME_VALUE_DONE",
    "PARAMETERS_KEY",
    "PARAMETERS_KEY_STRING",
    "PARAMETERS_KEY_DONE",
    "PARAMETERS_OPEN",
    "PARAMETERS_EMPTY_OR_KEY",
    "PARAM_KEY",
    "PARAM_KEY_DONE",
    "PARAM_VALUE_START",
    "PARAM_VALUE_STRING",
    "PARAM_VALUE_STRING_ESCAPE",
    "PARAM_VALUE_NUMBER",
    "PARAM_VALUE_BOOLEAN",
    "PARAM_VALUE_DONE",
    "PARAM_KEY_START",
    "PARAMETERS_CLOSED",
    "ACCEPT",
}


def _walk(text: str, names: list[str] | None = None) -> DFA:
    """Consume every character of ``text`` and return the final machine."""
    dfa = DFA.start(names)
    for index, char in enumerate(text):
        if not dfa.consume(char):
            raise AssertionError(
                f"rejected {char!r} at index {index} from {dfa.state.name}"
            )
    return dfa


def _call(name: str, parameters: str = "{}") -> str:
    """Build a valid template around ``name`` and ``parameters``."""
    return (
        '{"prompt":"x","name":"'
        + name
        + '","parameters":'
        + parameters
        + "}"
    )


def _rejects(text: str, names: list[str] | None = None) -> bool:
    """Return whether consuming ``text`` hits a rejected character."""
    dfa = DFA.start(names)
    return not all(dfa.consume(char) for char in text)


def _snapshot(dfa: DFA) -> tuple[object, str, str, str, int]:
    """Return the mutable fields a failed consume must preserve."""
    return (
        dfa.state,
        dfa.name_buffer,
        dfa.number_buffer,
        dfa.literal_buffer,
        dfa.unicode_remaining,
    )


def _assert_atomic_rejection(
    prefix: str,
    bad: str,
    names: list[str] | None = None,
) -> None:
    """Assert that ``bad`` fails without changing any machine field."""
    dfa = DFA.start(names)
    for char in prefix:
        assert dfa.consume(char), f"prefix rejected at {char!r}"

    before = _snapshot(dfa)
    assert dfa.consume(bad) is False
    assert _snapshot(dfa) == before


def _assert_token_atomic_rejection(dfa: DFA, text: str) -> None:
    """Assert that token rejection leaves every mutable field unchanged."""
    before = _snapshot(dfa)
    assert dfa.try_consume_token(text) is None
    assert _snapshot(dfa) == before
    assert dfa.consume_token(text) is False
    assert _snapshot(dfa) == before


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


@pytest.mark.parametrize(
    "text",
    [
        '{"prompt":"hi","name":"fn_greet","parameters":{}}',
        '{"prompt":"hi","name":"fn_greet","parameters":{"name":"shrek"}}',
        '{"prompt":"sum","name":"fn_add_numbers","parameters":{"a":2,"b":3}}',
        '{"prompt":"alarm","name":"fn_set_alarm","parameters":'
        '{"enabled":true}}',
    ],
)
def test_full_valid_paths_reach_accept(text: str) -> None:
    dfa = _walk(text)

    assert dfa.is_accepting() is True


def test_whitespace_between_structural_tokens_is_accepted() -> None:
    text = (
        '{ \n\t"prompt" \r: \t"hi" \n, "name" :\n"fn_greet" ,'
        ' "parameters" : { } }'
    )

    dfa = _walk(text)

    assert dfa.is_accepting() is True


def test_escaped_quote_does_not_close_string_value() -> None:
    dfa = DFA.start()
    for char in '{"prompt":"a\\"b':
        assert dfa.consume(char)

    assert dfa.state is DFAState.PROMPT_VALUE

    assert dfa.consume('"') is True
    assert dfa.state is DFAState.PROMPT_VALUE_DONE


@pytest.mark.parametrize(
    ("state", "char"),
    [
        (DFAState.START, " "),
        (DFAState.START, "x"),
        (DFAState.ROOT_OPEN, "x"),
        (DFAState.PROMPT_KEY_DONE, "x"),
        (DFAState.PROMPT_VALUE, "\n"),
        (DFAState.PROMPT_VALUE_ESCAPE, "x"),
        (DFAState.NAME_VALUE, "\\"),
        (DFAState.PARAM_KEY_START, "}"),
        (DFAState.ACCEPT, "}"),
    ],
)
def test_invalid_transitions_return_none(
    state: DFAState,
    char: str,
) -> None:
    assert transition(state, char) is None


def test_transition_is_pure_and_deterministic() -> None:
    dfa = DFA(state=DFAState.PROMPT_VALUE)

    first = transition(dfa.state, "a")
    second = transition(dfa.state, "a")

    assert first is DFAState.PROMPT_VALUE
    assert second is DFAState.PROMPT_VALUE
    assert dfa.state is DFAState.PROMPT_VALUE


def test_consume_advances_on_valid_char() -> None:
    dfa = DFA.start()

    assert dfa.consume("{") is True
    assert dfa.state is DFAState.ROOT_OPEN


def test_consume_keeps_state_on_invalid_char() -> None:
    dfa = DFA.start()

    assert dfa.consume("x") is False
    assert dfa.state is DFAState.START


def test_accept_is_reached_only_after_root_close() -> None:
    text = '{"prompt":"hi","name":"fn_greet","parameters":{}}'
    dfa = DFA.start()

    for index, char in enumerate(text):
        assert dfa.consume(char)
        if index < len(text) - 1:
            assert dfa.state is not DFAState.ACCEPT

    assert dfa.state is DFAState.ACCEPT
    assert dfa.is_accepting() is True


def test_name_matcher_accepts_prefixes_and_exact_names() -> None:
    matcher = NameMatcher(["fn_add_numbers", "fn_add", "fn_set_alarm"])

    assert matcher.is_prefix("") is True
    assert matcher.is_prefix("fn_a") is True
    assert matcher.is_prefix("fn_add") is True
    assert matcher.is_prefix("fn_add_n") is True
    assert matcher.is_prefix("fn_missing") is False
    assert matcher.is_complete("fn_add") is True
    assert matcher.is_complete("fn_add_n") is False
    assert matcher.candidates("fn_add") == ("fn_add", "fn_add_numbers")


@pytest.mark.parametrize("names", [[], [""], [" "], ["fn", ""]])
def test_name_matcher_rejects_empty_names(names: list[str]) -> None:
    with pytest.raises(ValueError):
        NameMatcher(names)


@pytest.mark.parametrize("name", ['fn"add', "fn\\add", "fn\nadd"])
def test_name_matcher_rejects_unrepresentable_characters(name: str) -> None:
    with pytest.raises(ValueError, match="cannot represent"):
        NameMatcher(["fn_add", name])


def test_name_matcher_rejects_bare_string() -> None:
    with pytest.raises(TypeError, match="iterable of names"):
        NameMatcher("fn_add")


def test_dfa_start_rejects_bare_string() -> None:
    with pytest.raises(TypeError, match="pass a list"):
        DFA.start("fn_add")


def test_reset_restores_start_and_clears_buffers_but_keeps_names() -> None:
    matcher = NameMatcher(["fn_add"])
    dfa = DFA(
        state=DFAState.PARAM_VALUE_NUMBER,
        names=matcher,
        name_buffer="fn_add",
        number_buffer="12",
        literal_buffer="tru",
        unicode_remaining=2,
    )

    dfa.reset()

    assert dfa.state is DFAState.START
    assert dfa.name_buffer == ""
    assert dfa.number_buffer == ""
    assert dfa.literal_buffer == ""
    assert dfa.unicode_remaining == 0
    assert dfa.names is matcher


def test_name_slot_accepts_a_valid_prefix_path() -> None:
    names = ["fn_add", "fn_add_numbers"]
    dfa = _walk(_call("fn_add_numbers"), names)

    assert dfa.is_accepting() is True


def test_name_slot_without_matcher_accepts_any_name() -> None:
    assert _walk(_call("anything")).is_accepting() is True


def test_name_slot_without_matcher_accepts_an_empty_name() -> None:
    assert _walk(_call("")).is_accepting() is True


@pytest.mark.parametrize(
    "name",
    ["fn_missing", "fn_add_n", "fn_add_extra", ""],
)
def test_name_slot_rejects_unknown_or_empty_names(name: str) -> None:
    assert _rejects(_call(name), ["fn_add", "fn_add_numbers"]) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "-",
        "0",
        "1",
        "12",
        "1.",
        "1.5",
        "1e",
        "1e-",
        "1e+",
        "1e1",
        "-1.5e-2",
        "0.0",
        "10",
    ],
)
def test_number_prefix_accepts_partial_forms(text: str) -> None:
    assert is_number_prefix(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "+1",
        ".5",
        "01",
        "00",
        "-01",
        "1..",
        "1.e",
        "--1",
        "1.-",
        "1e--",
        "1e+.",
        "1e1.0",
        "abc",
    ],
)
def test_number_prefix_rejects_invalid_forms(text: str) -> None:
    assert is_number_prefix(text) is False


@pytest.mark.parametrize(
    "text",
    ["0", "-0", "12", "-12", "1.5", "0.5", "1e3", "1E+3", "-1.5e-2"],
)
def test_complete_number_accepts_valid_forms(text: str) -> None:
    assert is_complete_number(text) is True


@pytest.mark.parametrize(
    "text",
    ["", "-", "1.", "01", "+1", ".5", "1.e", "1e", "1e+", "1..", "--1"],
)
def test_complete_number_rejects_invalid_forms(text: str) -> None:
    assert is_complete_number(text) is False


@pytest.mark.parametrize(
    "value",
    ["0", "-0", "12", "-12", "1.5", "1e3", "1E+3", "-1.5e-2"],
)
def test_number_slot_accepts_valid_numbers(value: str) -> None:
    dfa = _walk(_call("fn", '{"n":' + value + "}"), ["fn"])

    assert dfa.is_accepting() is True


@pytest.mark.parametrize(
    "value",
    ["01", "+1", ".5", "1.", "1e", "1e+", "1..", "--1"],
)
def test_number_slot_rejects_invalid_numbers(value: str) -> None:
    assert _rejects(_call("fn", '{"n":' + value + "}"), ["fn"]) is True


@pytest.mark.parametrize("value", ["true", "false"])
def test_boolean_slot_accepts_true_and_false(value: str) -> None:
    dfa = _walk(_call("fn", '{"ok":' + value + "}"), ["fn"])

    assert dfa.is_accepting() is True


@pytest.mark.parametrize(
    "value",
    ["True", "False", "tru", "fals", "truex", "falsey", "yes"],
)
def test_boolean_slot_rejects_invalid_literals(value: str) -> None:
    assert _rejects(_call("fn", '{"ok":' + value + "}"), ["fn"]) is True


@pytest.mark.parametrize("prefix", ["t", "tr", "fals"])
def test_boolean_slot_accepts_incomplete_prefixes(prefix: str) -> None:
    dfa = DFA.start(["fn"])
    for char in '{"prompt":"x","name":"fn","parameters":{"ok":' + prefix:
        assert dfa.consume(char) is True

    assert dfa.state is DFAState.PARAM_VALUE_BOOLEAN
    assert dfa.literal_buffer == prefix


def test_boolean_slot_rejects_diverging_prefix() -> None:
    assert _rejects(_call("fn", '{"ok":trx}'), ["fn"]) is True


@pytest.mark.parametrize(
    "state",
    [
        DFAState.PROMPT_VALUE_START,
        DFAState.NAME_VALUE_START,
        DFAState.PARAM_VALUE_START,
    ],
)
def test_opening_string_value_resets_unicode_remaining(
    state: DFAState,
) -> None:
    dfa = DFA(state=state, unicode_remaining=3)

    assert dfa.consume('"') is True
    assert dfa.unicode_remaining == 0


def test_string_slot_accepts_escapes_and_unicode() -> None:
    parameters = '{"s":"a\\"b\\\\c\\n\\t\\u00e9"}'
    dfa = _walk(_call("fn", parameters), ["fn"])

    assert dfa.is_accepting() is True


def test_prompt_string_slot_accepts_unicode_escape() -> None:
    text = '{"prompt":"caf\\u00e9","name":"fn","parameters":{}}'

    assert _walk(text, ["fn"]).is_accepting() is True


@pytest.mark.parametrize("escaped", ["\\uZZZZ", "\\u12", "\\q"])
def test_string_slot_rejects_invalid_escapes(escaped: str) -> None:
    assert _rejects(_call("fn", '{"s":"' + escaped + '"}'), ["fn"]) is True


def test_string_slot_rejects_raw_control_character() -> None:
    assert _rejects(_call("fn", '{"s":"a\nb"}'), ["fn"]) is True


def test_all_slots_together_reach_accept() -> None:
    parameters = '{"a":-1.5e+2,"b":true,"c":"x\\"y"}'
    names = ["fn_add", "fn_add_numbers"]
    dfa = _walk(_call("fn_add", parameters), names)

    assert dfa.is_accepting() is True


@pytest.mark.parametrize(
    ("prefix", "bad", "names"),
    [
        ('{"prompt":"x","name":"fn_a', "x", ["fn_add"]),
        ('{"prompt":"x","name":"fn","parameters":{"n":0', "1", ["fn"]),
        ('{"prompt":"x","name":"fn","parameters":{"ok":tru', "x", ["fn"]),
        ('{"prompt":"x","name":"fn","parameters":{"s":"\\u12', "Z", ["fn"]),
        ('{"prompt":"x","name":"fn","parameters":{"s":"\\u12', '"', ["fn"]),
        ('{"prompt":"x","name":"fn","parameters":{"s":"\\u12', "\\", ["fn"]),
    ],
)
def test_failed_consume_is_atomic(
    prefix: str,
    bad: str,
    names: list[str],
) -> None:
    _assert_atomic_rejection(prefix, bad, names)


def test_try_consume_token_returns_a_new_advanced_machine() -> None:
    dfa = _walk('{"prompt":"x","name":"fn","parameters":{"n":')

    advanced = dfa.try_consume_token("2.0")

    assert advanced is not None
    assert advanced is not dfa
    assert advanced.state is DFAState.PARAM_VALUE_NUMBER
    assert advanced.number_buffer == "2.0"
    assert dfa.state is DFAState.PARAM_VALUE_START
    assert dfa.number_buffer == ""


def test_try_consume_token_returns_none_and_keeps_original() -> None:
    dfa = _walk('{"prompt":"x","name":"fn","parameters":{"n":')
    before = _snapshot(dfa)

    assert dfa.try_consume_token("2x") is None
    assert _snapshot(dfa) == before


def test_consume_token_accepts_comma_after_prompt_value() -> None:
    dfa = _walk('{"prompt":"x"')

    assert dfa.consume_token(",") is True
    assert dfa.state is DFAState.NAME_KEY


def test_consume_token_accepts_quote_comma() -> None:
    dfa = _walk('{"prompt":"x')

    assert dfa.consume_token('",') is True
    assert dfa.state is DFAState.NAME_KEY


def test_consume_token_accepts_quote_brace() -> None:
    prefix = '{"prompt":"x","name":"fn","parameters":{"s":"x'
    dfa = _walk(prefix)

    assert dfa.consume_token('"}') is True
    assert dfa.state is DFAState.PARAMETERS_CLOSED


def test_consume_token_accepts_single_brace() -> None:
    dfa = _walk('{"prompt":"x","name":"fn","parameters":{')

    assert dfa.consume_token("}") is True
    assert dfa.state is DFAState.PARAMETERS_CLOSED


def test_consume_token_accepts_number_text() -> None:
    dfa = _walk('{"prompt":"x","name":"fn","parameters":{"n":')

    assert dfa.consume_token("2.0") is True
    assert dfa.state is DFAState.PARAM_VALUE_NUMBER
    assert dfa.number_buffer == "2.0"


def test_consume_token_accepts_boolean_text() -> None:
    dfa = _walk('{"prompt":"x","name":"fn","parameters":{"ok":')

    assert dfa.consume_token("true") is True
    assert dfa.state is DFAState.PARAM_VALUE_BOOLEAN
    assert dfa.literal_buffer == "true"


def test_consume_token_accepts_quoted_function_name() -> None:
    dfa = _walk(
        '{"prompt":"x","name":',
        ["fn_add", "fn_add_numbers"],
    )

    assert dfa.consume_token('"fn_add_numbers"') is True
    assert dfa.state is DFAState.NAME_VALUE_DONE
    assert dfa.name_buffer == "fn_add_numbers"


def test_consume_token_accepts_unicode_escape_inside_string() -> None:
    dfa = _walk('{"prompt":"x')

    assert dfa.consume_token("\\u00e9") is True
    assert dfa.state is DFAState.PROMPT_VALUE
    assert dfa.unicode_remaining == 0


def test_unicode_remaining_is_maintained_across_tokens() -> None:
    dfa = _walk('{"prompt":"x')

    assert dfa.consume_token("\\u00") is True
    assert dfa.unicode_remaining == 2

    assert dfa.consume_token("e9") is True
    assert dfa.unicode_remaining == 0
    assert dfa.state is DFAState.PROMPT_VALUE


@pytest.mark.parametrize(
    ("prefix", "text", "names"),
    [
        ("", "x", None),
        ('{"prompt":"x","name":', '"x', ["fn"]),
        ('{"prompt":"x","name":"fn","parameters":{"n":', "2x", ["fn"]),
        ('{"prompt":"x","name":"fn","parameters":{"ok":', "trX", ["fn"]),
    ],
)
def test_consume_token_failure_is_atomic(
    prefix: str,
    text: str,
    names: list[str] | None,
) -> None:
    dfa = _walk(prefix, names)

    _assert_token_atomic_rejection(dfa, text)


def test_partial_token_success_survives_a_later_failure() -> None:
    dfa = _walk('{"prompt":"x","name":"fn","parameters":{"n":')

    assert dfa.consume_token("2.0") is True
    after_success = _snapshot(dfa)

    assert dfa.consume_token("x") is False
    assert _snapshot(dfa) == after_success


def test_try_consume_token_empty_returns_identical_copy() -> None:
    dfa = _walk('{"prompt":"x')
    before = _snapshot(dfa)

    advanced = dfa.try_consume_token("")

    assert advanced is not None
    assert advanced is not dfa
    assert _snapshot(advanced) == before
    assert _snapshot(dfa) == before


def test_consume_token_empty_is_a_noop_success() -> None:
    dfa = _walk('{"prompt":"x')
    before = _snapshot(dfa)

    assert dfa.consume_token("") is True
    assert _snapshot(dfa) == before


def test_full_template_one_char_per_token_reaches_accept() -> None:
    text = _call("fn_greet")
    dfa = DFA.start(["fn_greet"])

    for index, char in enumerate(text):
        assert dfa.consume_token(char) is True
        if index < len(text) - 1:
            assert dfa.is_accepting() is False

    assert dfa.state is DFAState.ACCEPT
    assert dfa.is_accepting() is True
