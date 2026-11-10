"""Unit tests for constrained decoding masks and the T5.3 generation loop."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from src.decoder import (
    _resolve_eos_token_id,
    allowed_token_ids,
    decode_call,
    masked_logits,
    select_next_token,
    slot_can_stop,
)
from src.dfa import DFA, DFAState
from src.errors import DecodingError
from src.llm_client import LLMClient
from src.prompt import build_template
from src.schemas import FunctionCall, FunctionCatalog
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


# T5.2: -inf mask, guarded argmax and all-masked detection.


def test_invalid_high_logit_is_never_selected() -> None:
    logits = [0.0, 100.0, 1.0, 2.0, 3.0]

    assert select_next_token(logits, [0, 2]) == 2


def test_only_allowed_logits_are_compared() -> None:
    logits = [100.0, 1.0, 50.0, 2.0, 75.0]

    assert select_next_token(logits, [1, 3]) == 3


def test_all_masked_allowed_ids_raise_decoding_error() -> None:
    logits = [0.0, -np.inf, -np.inf, 2.0]

    with pytest.raises(DecodingError) as excinfo:
        select_next_token(logits, [1, 2])

    message = str(excinfo.value)
    assert "no valid token" in message
    assert "2 allowed" in message
    assert "size 4" in message


def test_empty_allowed_ids_raise_decoding_error() -> None:
    with pytest.raises(DecodingError) as excinfo:
        select_next_token([0.0, 1.0, 2.0], [])

    message = str(excinfo.value)
    assert "no valid token" in message
    assert "0 allowed" in message
    assert "size 3" in message


def test_non_finite_allowed_scores_are_sanitized() -> None:
    logits = [0.0, np.nan, 1.0, np.inf]

    assert select_next_token(logits, [1, 2, 3]) == 2
    assert masked_logits(logits, [3])[3] == -np.inf

    with pytest.raises(DecodingError):
        select_next_token(logits, [3])


def test_all_nan_and_infinite_scores_raise_decoding_error() -> None:
    with pytest.raises(DecodingError):
        select_next_token([np.nan, np.inf, -np.inf], [0, 1, 2])


def test_masked_logits_copies_only_allowed_finite_scores() -> None:
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    original = logits.copy()

    masked = masked_logits(logits, [0, 2, 4])

    assert masked.dtype == np.float64
    assert masked.shape == logits.shape
    assert masked[0] == 1.0
    assert masked[2] == 3.0
    assert masked[4] == 5.0
    assert np.isneginf(masked[1])
    assert np.isneginf(masked[3])
    assert np.array_equal(logits, original)


def test_masked_logits_sanitizes_non_finite_and_never_mutates() -> None:
    logits = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
    original = logits.copy()

    masked = masked_logits(logits, [0, 2, 3, 4])

    assert masked[0] == 1.0
    assert masked[4] == 2.0
    assert np.isneginf(masked[1])
    assert np.isneginf(masked[2])
    assert np.isneginf(masked[3])
    assert np.array_equal(logits, original, equal_nan=True)


def test_masked_logits_empty_allowed_is_all_negative_infinity() -> None:
    masked = masked_logits([1.0, 2.0, 3.0], [])

    assert masked.dtype == np.float64
    assert np.isneginf(masked).all()


@pytest.mark.parametrize("bad_id", [-1, 5, 6, 1.0, "1", None, True])
def test_masked_logits_rejects_bad_allowed_ids(bad_id: Any) -> None:
    with pytest.raises(ValueError) as excinfo:
        masked_logits([0.0] * 5, [bad_id])

    message = str(excinfo.value)
    assert "allowed token id" in message
    assert repr(bad_id) in message
    assert "5" in message


def test_masked_logits_rejects_non_1d_logits() -> None:
    bad_logits: Any = [[1.0, 2.0], [3.0, 4.0]]

    with pytest.raises(ValueError, match="one-dimensional"):
        masked_logits(bad_logits, [0])


def test_tied_scores_select_the_lowest_id() -> None:
    assert select_next_token([0.0, 5.0, 5.0, 0.0], [1, 2, 3]) == 1


def test_duplicate_allowed_ids_do_not_change_the_result() -> None:
    logits = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    assert select_next_token(logits, [2, 2, 4]) == 4
    assert np.array_equal(
        masked_logits(logits, [2, 2, 4]),
        masked_logits(logits, [2, 4]),
    )


def test_realistic_vocab_size_mask_has_only_two_finite_scores() -> None:
    logits = np.full(151936, -5.0)
    logits[1234] = 1.0
    logits[150000] = 2.0
    allowed = [1234, 150000]

    masked = masked_logits(logits, allowed)

    assert masked.shape == (151936,)
    assert masked.dtype == np.float64
    assert int(np.isfinite(masked).sum()) == 2
    assert select_next_token(logits, allowed) == 150000


# T5.3: generation loop with fixed injection, EOS and MAX_TOKENS.

_DECODER_CHARS = 'abcdefghijklmnopqrstuvwxyz_" 0123456789.-'
_EOS_ID = 1000
_EOS_TEXT = "<|im_end|>"


def _decoder_catalog() -> FunctionCatalog:
    """Catalog containing name-prefix and parameter-shape edge cases."""
    return FunctionCatalog.from_list(
        [
            {
                "name": "fn_add",
                "description": "Add nothing.",
                "parameters": {},
                "returns": {"type": "string"},
            },
            {
                "name": "fn_add_numbers",
                "description": "Add two numbers.",
                "parameters": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "returns": {"type": "number"},
            },
            {
                "name": "fn_greet",
                "description": "Greet someone.",
                "parameters": {"name": {"type": "string"}},
                "returns": {"type": "string"},
            },
            {
                "name": "fn_ping",
                "description": "Ping.",
                "parameters": {},
                "returns": {"type": "string"},
            },
        ]
    )


def _decoder_index(
    *,
    chars: str = _DECODER_CHARS,
    eos_id: int | None = _EOS_ID,
    eos_text: str = _EOS_TEXT,
) -> TokenizerIndex:
    """Build a char-token index; the EOS text is mapped but never allowed."""
    texts = {ord(char): char for char in sorted(set(chars))}
    allowed_ids = tuple(sorted(texts))
    if eos_id is not None:
        texts[eos_id] = eos_text

    buckets: dict[str, list[int]] = {}
    for token_id in allowed_ids:
        text = texts[token_id]
        if text:
            buckets.setdefault(text[0], []).append(token_id)
    return TokenizerIndex(
        texts=texts,
        allowed_ids=allowed_ids,
        by_first_char={
            char: tuple(token_ids) for char, token_ids in buckets.items()
        },
    )


def _tokens(text: str) -> list[int]:
    """Return one token id per character, matching the char-token index."""
    return [ord(char) for char in text]


class _ScriptedBackend:
    """Model fake whose logits always favour the next scripted token id."""

    def __init__(
        self,
        script: list[int],
        *,
        size: int = 2048,
        all_masked_at: int | None = None,
    ) -> None:
        self.script = list(script)
        self.size = size
        self.all_masked_at = all_masked_at
        self.calls = 0
        self.encode_calls: list[str] = []
        self.decode_calls: list[list[int]] = []

    def encode(self, text: str) -> list[list[int]]:
        self.encode_calls.append(text)
        return [_tokens(text)]

    def decode(self, ids: list[int]) -> str:
        self.decode_calls.append(list(ids))
        return "".join(chr(token_id) for token_id in ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        self.calls += 1
        if self.all_masked_at == self.calls:
            return [float("-inf")] * self.size
        if not self.script:
            raise AssertionError("logits called after the script was exhausted")

        token_id = self.script.pop(0)
        logits = [-1.0] * self.size
        logits[token_id] = 10.0
        return logits

    def get_path_to_vocab_file(self) -> str:
        return "unused"


def _scripted_model(
    script: list[int],
    **kwargs: Any,
) -> tuple[LLMClient, _ScriptedBackend]:
    backend = _ScriptedBackend(script, **kwargs)
    return LLMClient(backend=backend), backend


def test_decode_call_controlled_string_case() -> None:
    catalog = _decoder_catalog()
    script = _tokens("fn_greet") + [_EOS_ID] + _tokens('"bob"')
    model, backend = _scripted_model(script)

    call = decode_call(model, catalog, "Greet bob", index=_decoder_index())

    expected = build_template(
        catalog,
        "Greet bob",
        "fn_greet",
        {"name": "bob"},
    )
    assert backend.calls == len(script)
    assert backend.script == []
    assert "".join(chr(i) for i in backend.decode_calls[-1]) == expected
    assert call == FunctionCall(
        prompt="Greet bob",
        name="fn_greet",
        parameters={"name": "bob"},
    )


def test_decode_call_empty_parameter_case() -> None:
    catalog = _decoder_catalog()
    script = _tokens("fn_ping") + [_EOS_ID]
    model, backend = _scripted_model(script)

    call = decode_call(model, catalog, "Ping", index=_decoder_index())

    assert call == FunctionCall(prompt="Ping", name="fn_ping", parameters={})
    assert backend.calls == len(script)


def test_decode_call_does_not_extend_complete_prefix_name() -> None:
    catalog = _decoder_catalog()
    script = _tokens("fn_add") + [_EOS_ID]
    model, backend = _scripted_model(script)

    call = decode_call(model, catalog, "Add", index=_decoder_index())

    assert call.name == "fn_add"
    assert call.parameters == {}
    assert backend.calls == len(script)


def test_decode_call_preserves_prompt_quotes_and_unicode() -> None:
    catalog = _decoder_catalog()
    prompt = 'say "hi" José'
    script = _tokens("fn_greet") + [_EOS_ID] + _tokens('"bob"')
    model, backend = _scripted_model(script)

    call = decode_call(model, catalog, prompt, index=_decoder_index())

    expected = build_template(
        catalog,
        prompt,
        "fn_greet",
        {"name": "bob"},
    )
    assert call.prompt == prompt
    assert "".join(chr(i) for i in backend.decode_calls[-1]) == expected
    assert backend.calls == len(script)


def test_decode_call_aborts_at_max_tokens() -> None:
    catalog = _decoder_catalog()
    script = _tokens("fn_greet") + [_EOS_ID] + _tokens('"bob"')
    model, backend = _scripted_model(script)

    with pytest.raises(DecodingError, match="MAX_TOKENS"):
        decode_call(
            model,
            catalog,
            "Greet bob",
            index=_decoder_index(),
            max_tokens=2,
        )

    assert backend.calls == 2


def test_decode_call_counts_eos_against_max_tokens() -> None:
    catalog = _decoder_catalog()
    script = _tokens("fn_greet") + [_EOS_ID] + _tokens('"bob"')
    model, backend = _scripted_model(script)

    with pytest.raises(DecodingError, match="MAX_TOKENS"):
        decode_call(
            model,
            catalog,
            "Greet bob",
            index=_decoder_index(),
            max_tokens=len(_tokens("fn_greet")),
        )

    assert backend.calls == len(_tokens("fn_greet"))


@pytest.mark.parametrize("marker", ["<|im_end|>", "<|endoftext|>"])
def test_decode_call_resolves_eos_marker(marker: str) -> None:
    catalog = _decoder_catalog()
    eos_id = 1001
    index = _decoder_index(eos_id=eos_id, eos_text=marker)
    script = _tokens("fn_greet") + [eos_id] + _tokens('"bob"')
    model, backend = _scripted_model(script)

    call = decode_call(model, catalog, "Greet bob", index=index)

    assert call.name == "fn_greet"
    assert backend.calls == len(script)


def test_build_token_index_preserves_added_text_for_eos() -> None:
    vocab = TokenizerVocab(
        token_text={ord("a"): "a", _EOS_ID: _EOS_TEXT},
        special_ids=frozenset({_EOS_ID}),
        added_ids=frozenset({_EOS_ID}),
        source="test",
    )
    index = build_token_index(
        vocab,
        texts={ord("a"): "a", _EOS_ID: ""},
    )

    assert index.texts[_EOS_ID] == _EOS_TEXT
    assert _EOS_ID not in index.allowed_ids
    assert _resolve_eos_token_id(index, None) == _EOS_ID


def test_decode_call_wraps_invalid_schema_as_decoding_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _decoder_catalog()
    script = _tokens("fn_ping") + [_EOS_ID]
    model, backend = _scripted_model(script)
    monkeypatch.setattr(backend, "decode", lambda ids: "{not json")

    with pytest.raises(DecodingError, match="not a valid FunctionCall"):
        decode_call(model, catalog, "Ping", index=_decoder_index())


def test_decode_call_requires_a_resolvable_eos() -> None:
    catalog = _decoder_catalog()
    model, backend = _scripted_model([])

    with pytest.raises(DecodingError, match="EOS"):
        decode_call(
            model,
            catalog,
            "Greet bob",
            index=_decoder_index(eos_id=None),
        )

    assert backend.calls == 0


def test_decode_call_all_masked_slot_raises() -> None:
    catalog = _decoder_catalog()
    script = _tokens("fn_greet") + [_EOS_ID] + _tokens('"bob"')
    model, backend = _scripted_model(script, all_masked_at=1)

    with pytest.raises(DecodingError, match="no valid token"):
        decode_call(model, catalog, "Greet bob", index=_decoder_index())

    assert backend.calls == 1


def test_decode_call_empty_slot_mask_raises_before_logits() -> None:
    catalog = _decoder_catalog()
    model, backend = _scripted_model([])

    with pytest.raises(DecodingError, match="no allowed tokens"):
        decode_call(
            model,
            catalog,
            "Greet bob",
            index=_decoder_index(chars="x"),
        )

    assert backend.calls == 0


def test_decode_call_fills_multiple_number_slots_with_eos() -> None:
    catalog = _decoder_catalog()
    script = (
        _tokens("fn_add_numbers")
        + [_EOS_ID]
        + _tokens("1")
        + [_EOS_ID]
        + _tokens("2")
        + [_EOS_ID]
    )
    model, backend = _scripted_model(script)

    call = decode_call(model, catalog, "Add", index=_decoder_index())

    assert call == FunctionCall(
        prompt="Add",
        name="fn_add_numbers",
        parameters={"a": 1, "b": 2},
    )
    assert backend.calls == len(script)
