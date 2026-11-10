"""Unit tests for slot-scoped constrained-decoding token masks (T5.1)."""

from __future__ import annotations

import copy

from src.decoder import allowed_token_ids, slot_can_stop
from src.dfa import DFA, DFAState
from src.vocab import TokenizerIndex, TokenizerVocab, build_token_index

_NAMES = [
    "fn_add",
    "fn_add_numbers",
    "fn_greet",
    "fn_set_alarm",
]
_PROMPT_STRING_PREFIX = '{"prompt":"'
_NAME_PREFIX = '{"prompt":"x","name":"'
_PARAM_START_PREFIX = (
    '{"prompt":"x","name":"fn_add_numbers","parameters":{"a":'
)
_PARAM_STRING_PREFIX = (
    '{"prompt":"x","name":"fn_greet","parameters":{"name":"'
)
_PARAM_BOOLEAN_PREFIX = (
    '{"prompt":"x","name":"fn_set_alarm","parameters":{"enabled":'
)
_PARAM_DONE_PREFIX = _PARAM_STRING_PREFIX + 'bob"'
_ACCEPT = '{"prompt":"x","name":"fn_greet","parameters":{}}'

# Independent copy of the slot contract, so the brute-force oracle does not
# agree with the implementation by construction.
_REFERENCE_TARGETS: dict[DFAState, frozenset[DFAState]] = {
    DFAState.NAME_VALUE: frozenset({DFAState.NAME_VALUE}),
    DFAState.PARAM_VALUE_START: frozenset(
        {
            DFAState.PARAM_VALUE_START,
            DFAState.PARAM_VALUE_STRING,
            DFAState.PARAM_VALUE_STRING_ESCAPE,
            DFAState.PARAM_VALUE_NUMBER,
            DFAState.PARAM_VALUE_BOOLEAN,
            DFAState.PARAM_VALUE_DONE,
        }
    ),
    DFAState.PARAM_VALUE_STRING: frozenset(
        {
            DFAState.PARAM_VALUE_STRING,
            DFAState.PARAM_VALUE_STRING_ESCAPE,
            DFAState.PARAM_VALUE_DONE,
        }
    ),
    DFAState.PARAM_VALUE_STRING_ESCAPE: frozenset(
        {
            DFAState.PARAM_VALUE_STRING,
            DFAState.PARAM_VALUE_STRING_ESCAPE,
            DFAState.PARAM_VALUE_DONE,
        }
    ),
    DFAState.PARAM_VALUE_NUMBER: frozenset({DFAState.PARAM_VALUE_NUMBER}),
    DFAState.PARAM_VALUE_BOOLEAN: frozenset({DFAState.PARAM_VALUE_BOOLEAN}),
}


def _walk(text: str, names: list[str] | None = _NAMES) -> DFA:
    """Return the DFA state after consuming every character of ``text``."""
    dfa = DFA.start(names)
    for index, char in enumerate(text):
        assert dfa.consume(char) is True, (
            f"prefix rejected at index {index} ({char!r}): {text!r}"
        )
    return dfa


def _snapshot(dfa: DFA) -> tuple[object, ...]:
    """Capture every mutable DFA field for an unchanged-state assertion."""
    return (
        dfa.state,
        dfa.names,
        dfa.name_buffer,
        dfa.number_buffer,
        dfa.literal_buffer,
        dfa.unicode_remaining,
    )


def _make_index(texts: dict[int, str]) -> TokenizerIndex:
    """Build a small consistent index directly from id -> text."""
    allowed_ids = tuple(sorted(texts))
    buckets: dict[str, list[int]] = {}
    for token_id in allowed_ids:
        text = texts[token_id]
        if text:
            buckets.setdefault(text[0], []).append(token_id)
    return TokenizerIndex(
        texts=dict(texts),
        allowed_ids=allowed_ids,
        by_first_char={
            char: tuple(token_ids) for char, token_ids in buckets.items()
        },
    )


def _brute_allowed(dfa: DFA, index: TokenizerIndex) -> tuple[int, ...]:
    """Reference implementation: every character must stay in the slot."""
    targets = _REFERENCE_TARGETS.get(dfa.state)
    if targets is None:
        return ()

    allowed: list[int] = []
    for token_id in index.allowed_ids:
        text = index.texts.get(token_id)
        if not text:
            continue
        candidate = copy.copy(dfa)
        for char in text:
            if not candidate.consume(char):
                break
            if candidate.state not in targets:
                break
        else:
            if candidate.state in targets:
                allowed.append(token_id)
    return tuple(sorted(allowed))


def test_fixed_injected_states_return_empty_mask() -> None:
    index = _make_index({0: "{", 1: '"', 2: "a", 3: "}", 4: ","})
    states = [
        DFA.start(_NAMES),
        _walk("{"),
        _walk(_PROMPT_STRING_PREFIX),
        _walk(_NAME_PREFIX + 'fn_greet"'),
        _walk(_PARAM_DONE_PREFIX),
        _walk('{"prompt":"x","name":"fn_greet","parameters":{}'),
        _walk(_ACCEPT),
    ]

    for dfa in states:
        assert allowed_token_ids(dfa, index) == ()


def test_start_state_is_injected_not_generated() -> None:
    index = _make_index({0: "{", 1: '{"', 2: "x"})

    assert allowed_token_ids(DFA.start(_NAMES), index) == ()


def test_name_state_never_consumes_the_fixed_closing_quote() -> None:
    texts = {
        0: "f",
        1: "fn_",
        2: "fn_add",
        3: "fn_add_",
        4: "fn_add_number",
        5: "fn_add_numbers",
        6: "zzz",
        7: '"',
        8: 'fn_add"',
        9: 'fn_add_numbers"',
        10: "fn_add_namesX",
    }
    index = _make_index(texts)
    dfa = _walk(_NAME_PREFIX)
    before = _snapshot(dfa)

    allowed = allowed_token_ids(dfa, index)

    assert allowed == (0, 1, 2, 3, 4, 5)
    assert {6, 7, 8, 9, 10}.isdisjoint(allowed)
    assert _snapshot(dfa) == before

    complete = _walk(_NAME_PREFIX + "fn_add_numbers")
    assert complete.state is DFAState.NAME_VALUE
    assert allowed_token_ids(complete, index) == ()
    assert slot_can_stop(complete) is True


def test_name_slot_can_stop_only_for_complete_names() -> None:
    assert slot_can_stop(_walk(_NAME_PREFIX)) is False
    assert slot_can_stop(_walk(_NAME_PREFIX + "fn_add_")) is False
    assert slot_can_stop(_walk(_NAME_PREFIX + "fn_add")) is True
    assert slot_can_stop(_walk(_NAME_PREFIX, None)) is False


def test_number_start_allows_prefixes_but_not_delimiters() -> None:
    texts = {
        0: "1",
        1: "1.",
        2: "1e",
        3: "1e+",
        4: "}",
        5: "1}",
        6: ",",
        7: "1,",
        8: "-",
        9: "0.5",
        10: "a",
    }
    index = _make_index(texts)
    dfa = _walk(_PARAM_START_PREFIX)

    allowed = allowed_token_ids(dfa, index)

    assert allowed == (0, 1, 2, 3, 8, 9)
    assert {4, 5, 6, 7, 10}.isdisjoint(allowed)


def test_number_continuation_rejects_fixed_delimiters() -> None:
    texts = {
        0: "2",
        1: ".",
        2: "e",
        3: "E",
        4: "+",
        5: "-",
        6: "}",
        7: "1}",
        8: ",",
        9: "1,",
        10: "5",
    }
    index = _make_index(texts)
    dfa = _walk(_PARAM_START_PREFIX + "1")

    allowed = allowed_token_ids(dfa, index)

    assert allowed == (0, 1, 2, 3, 10)
    assert {4, 5, 6, 7, 8, 9}.isdisjoint(allowed)


def test_number_slot_can_stop_only_for_complete_numbers() -> None:
    assert slot_can_stop(_walk(_PARAM_START_PREFIX)) is False
    assert slot_can_stop(_walk(_PARAM_START_PREFIX + "1")) is True
    assert slot_can_stop(_walk(_PARAM_START_PREFIX + "1.")) is False
    assert slot_can_stop(_walk(_PARAM_START_PREFIX + "1e")) is False
    assert slot_can_stop(_walk(_PARAM_START_PREFIX + "1e+")) is False
    assert slot_can_stop(_walk(_PARAM_START_PREFIX + "1.5")) is True
    assert slot_can_stop(_walk(_PARAM_START_PREFIX + "1e3")) is True


def test_string_state_allows_content_and_rejects_fixed_crossing() -> None:
    texts = {
        0: "a",
        1: '\\"',
        2: "\\\\",
        3: "\\n",
        4: '"',
        5: '"}}',
        6: '"}',
        7: '",',
        8: " ",
        9: "\t",
        10: "\\u00",
        11: "\x01",
        12: "\\x",
        13: "b",
        14: 'a"',
    }
    index = _make_index(texts)
    dfa = _walk(_PARAM_STRING_PREFIX)

    allowed = allowed_token_ids(dfa, index)

    assert allowed == (0, 1, 2, 3, 4, 8, 10, 13, 14)
    assert {5, 6, 7, 9, 11, 12}.isdisjoint(allowed)


def test_string_unicode_and_escape_states_are_scoped() -> None:
    unicode_index = _make_index(
        {
            0: "0",
            1: "a",
            2: "F",
            3: "e9",
            4: "g",
            5: '"',
            6: "\x01",
            7: "f",
        }
    )
    unicode_dfa = _walk(_PARAM_STRING_PREFIX + "\\u00")

    assert allowed_token_ids(unicode_dfa, unicode_index) == (
        0,
        1,
        2,
        3,
        7,
    )

    escape_index = _make_index(
        {
            0: '"',
            1: "\\",
            2: "/",
            3: "b",
            4: "f",
            5: "n",
            6: "r",
            7: "t",
            8: "u",
            9: "x",
            10: "0",
        }
    )
    escape_dfa = _walk(_PARAM_STRING_PREFIX + "\\")

    assert allowed_token_ids(escape_dfa, escape_index) == tuple(range(9))


def test_param_value_start_allows_whitespace_prefixes() -> None:
    texts = {
        0: " ",
        1: "\t",
        2: " 1",
        3: ' "a"',
        4: " ,",
        5: " }",
    }
    index = _make_index(texts)
    dfa = _walk(_PARAM_START_PREFIX)

    assert dfa.state is DFAState.PARAM_VALUE_START
    assert allowed_token_ids(dfa, index) == (0, 1, 2, 3)


def test_boolean_value_start_rejects_fixed_delimiters() -> None:
    texts = {
        0: "t",
        1: "true",
        2: "true}",
        3: "true,",
        4: "f",
        5: "false",
        6: "false}",
        7: "false,",
        8: "tru",
        9: "yes",
    }
    index = _make_index(texts)
    dfa = _walk(_PARAM_BOOLEAN_PREFIX)

    allowed = allowed_token_ids(dfa, index)

    assert allowed == (0, 1, 4, 5, 8)
    assert {2, 3, 6, 7, 9}.isdisjoint(allowed)


def test_boolean_state_rejects_literal_overshoot() -> None:
    texts = {
        0: "rue",
        1: "rue}",
        2: "rue,",
        3: "ru",
        4: "}",
        5: "x",
    }
    index = _make_index(texts)
    dfa = _walk(_PARAM_BOOLEAN_PREFIX + "t")

    assert dfa.state is DFAState.PARAM_VALUE_BOOLEAN
    assert allowed_token_ids(dfa, index) == (0, 3)
    assert slot_can_stop(_walk(_PARAM_BOOLEAN_PREFIX + "true")) is True


def test_boolean_slot_can_stop_only_for_complete_literals() -> None:
    assert slot_can_stop(_walk(_PARAM_BOOLEAN_PREFIX)) is False
    assert slot_can_stop(_walk(_PARAM_BOOLEAN_PREFIX + "t")) is False
    assert slot_can_stop(_walk(_PARAM_BOOLEAN_PREFIX + "tru")) is False
    assert slot_can_stop(_walk(_PARAM_BOOLEAN_PREFIX + "true")) is True
    assert slot_can_stop(_walk(_PARAM_BOOLEAN_PREFIX + "false")) is True


def test_param_value_start_allows_whole_values_but_not_commas() -> None:
    texts = {
        0: '"bob"',
        1: "1",
        2: "true",
        3: '"bob",',
        4: "}",
        5: '"bob"}}',
    }
    index = _make_index(texts)
    dfa = _walk(_PARAM_START_PREFIX)

    allowed = allowed_token_ids(dfa, index)

    assert allowed == (0, 1, 2)
    assert {3, 4, 5}.isdisjoint(allowed)


def test_param_value_done_and_accept_return_empty_and_can_stop() -> None:
    index = _make_index({0: "}", 1: ",", 2: "a", 3: " ", 4: "true"})
    done = _walk(_PARAM_DONE_PREFIX)
    accepted = _walk(_ACCEPT)

    assert done.state is DFAState.PARAM_VALUE_DONE
    assert allowed_token_ids(done, index) == ()
    assert slot_can_stop(done) is True
    assert accepted.state is DFAState.ACCEPT
    assert allowed_token_ids(accepted, index) == ()
    assert slot_can_stop(accepted) is True


def test_result_is_sorted_unique_deterministic_and_does_not_mutate_dfa() -> None:
    index = TokenizerIndex(
        texts={0: "a", 1: "b", 2: "a", 3: "", 4: "<|special|>"},
        allowed_ids=(2, 0, 1),
        by_first_char={"b": (1,), "a": (2, 0)},
    )
    dfa = _walk(_PARAM_STRING_PREFIX)
    before = _snapshot(dfa)

    first = allowed_token_ids(dfa, index)
    second = allowed_token_ids(dfa, index)

    assert first == (0, 1, 2)
    assert first == second
    assert len(first) == len(set(first))
    assert {3, 4}.isdisjoint(first)
    assert _snapshot(dfa) == before


def test_build_token_index_excludes_special_and_empty_ids() -> None:
    vocab = TokenizerVocab(
        token_text={
            0: "a",
            1: "b",
            2: "<|special|>",
            3: "",
        },
        special_ids=frozenset({2}),
        added_ids=frozenset({2}),
        source="test",
    )
    index = build_token_index(vocab)

    allowed = allowed_token_ids(_walk(_PARAM_STRING_PREFIX), index)

    assert allowed == (0, 1)
    assert 2 not in allowed
    assert 3 not in allowed


def test_defensive_missing_and_empty_text_is_skipped() -> None:
    index = TokenizerIndex(
        texts={0: "a", 1: ""},
        allowed_ids=(0, 1),
        by_first_char={"a": (1, 0)},
    )

    assert allowed_token_ids(_walk(_PARAM_STRING_PREFIX), index) == (0,)


def test_oracle_matches_brute_force_for_six_slot_states() -> None:
    texts = {
        0: "f",
        1: "fn_",
        2: "fn_add",
        3: "fn_add_numbers",
        4: 'fn_add"',
        5: '"',
        6: '"bob"',
        7: '"bob",',
        8: '"bob"}}',
        9: "a",
        10: 'a"',
        11: '\\"',
        12: "\\n",
        13: "\\u00",
        14: "\\",
        15: "0",
        16: "1",
        17: "1.",
        18: "1e",
        19: "1e+",
        20: "1}",
        21: "1,",
        22: "true",
        23: "true}",
        24: "true,",
        25: "t",
        26: "}",
        27: ",",
        28: " ",
        29: "{",
        30: "x",
        31: "\x01",
    }
    index = _make_index(texts)
    states = [
        _walk(_NAME_PREFIX),
        _walk(_PARAM_START_PREFIX),
        _walk(_PARAM_STRING_PREFIX),
        _walk(_PARAM_STRING_PREFIX + "\\"),
        _walk(_PARAM_START_PREFIX + "1"),
        _walk(_PARAM_BOOLEAN_PREFIX + "t"),
    ]

    for dfa in states:
        assert allowed_token_ids(dfa, index) == _brute_allowed(dfa, index)
