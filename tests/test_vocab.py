"""Unit tests for vocabulary loading and token_text construction (T2.1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.errors import ModelError, ProjectError
from src.vocab import (
    TokenizerVocab,
    build_token_index,
    build_token_text_via_decode,
    excluded_token_ids,
    load_token_text_cache,
    parse_tokenizer_json,
    parse_vocab_json,
    save_token_text_cache,
    special_token_ids,
    unmapped_token_ids,
    validate_token_text,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TOKENIZER_FIXTURE = FIXTURES_DIR / "tokenizer_mini.json"
TOKENIZER_CHARS_FIXTURE = FIXTURES_DIR / "tokenizer_chars.json"
VOCAB_FIXTURE = FIXTURES_DIR / "vocab_mini.json"

# Raw one-character vocabulary used by tokenizer_chars.json.
CHARS_RAW_TO_ID = {
    "{": 0,
    "}": 1,
    "[": 2,
    "]": 3,
    '"': 4,
    ":": 5,
    ",": 6,
    "0": 7,
    "1": 8,
    "a": 9,
    "b": 10,
    "Ġ": 11,
    "Ċ": 12,
}
CHARS_ADDED_IDS = frozenset({13, 14})


def _chars_id_to_decoded() -> dict[int, str]:
    """Map each artificial 1-char token id to the text the decoder sees."""
    mapping: dict[int, str] = {}
    for raw, token_id in CHARS_RAW_TO_ID.items():
        if raw == "Ġ":
            mapping[token_id] = " "
        elif raw == "Ċ":
            mapping[token_id] = "\n"
        else:
            mapping[token_id] = raw
    return mapping


def _assert_model_error(
    excinfo: pytest.ExceptionInfo[ModelError],
    *fragments: str,
) -> None:
    """Assert the error is exactly ModelError (a ProjectError) and has fragments."""
    assert type(excinfo.value) is ModelError
    assert isinstance(excinfo.value, ProjectError)
    message = str(excinfo.value)
    for fragment in fragments:
        assert fragment in message


class FakeModel:
    """Minimal duck-typed model exposing only the public ``decode`` method."""

    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def decode(self, ids: list[int]) -> str:
        self.calls.append(list(ids))
        return f"<{ids[0]}>"


def test_parse_tokenizer_json_reads_vocab_and_added_tokens() -> None:
    result = parse_tokenizer_json(TOKENIZER_FIXTURE)

    assert isinstance(result, TokenizerVocab)
    assert result.token_text == {
        0: "hello",
        1: "Ġhello",
        2: ":",
        3: "world",
        4: "<|special|>",
        5: "<tool_call>",
    }
    assert result.special_ids == frozenset({2, 4})
    assert result.added_ids == frozenset({2, 4, 5})
    assert special_token_ids(result) == frozenset({2, 4})
    assert result.vocab_size == 6
    assert str(TOKENIZER_FIXTURE) in result.source
    assert "tokenizer.json" in result.source


def test_added_token_same_text_collision_is_marked_special() -> None:
    result = parse_tokenizer_json(TOKENIZER_FIXTURE)

    assert result.token_text[2] == ":"
    assert 2 in result.special_ids


def test_added_token_different_text_collision_raises(tmp_path: Path) -> None:
    data = json.loads(TOKENIZER_FIXTURE.read_text(encoding="utf-8"))
    data["added_tokens"][0]["content"] = "different"
    path = tmp_path / "tokenizer_collision.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ModelError) as excinfo:
        parse_tokenizer_json(path)

    _assert_model_error(excinfo, str(path), "2", ":", "different")


def test_parse_vocab_json_fallback() -> None:
    result = parse_vocab_json(VOCAB_FIXTURE)

    assert result.token_text == {0: "hello", 1: "Ġworld", 2: ":"}
    assert result.special_ids == frozenset()
    assert result.added_ids == frozenset()
    assert special_token_ids(result) == frozenset()
    assert str(VOCAB_FIXTURE) in result.source
    assert "vocab.json" in result.source


def test_missing_tokenizer_file_raises_model_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(ModelError) as excinfo:
        parse_tokenizer_json(missing)

    _assert_model_error(excinfo, str(missing), "not found")


def test_invalid_json_raises_model_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ModelError) as excinfo:
        parse_tokenizer_json(bad)

    _assert_model_error(excinfo, str(bad), "invalid JSON")


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"model": {}},
        {"model": {"vocab": []}},
        {"model": {"vocab": {"hello": 0}}, "added_tokens": {}},
        {"model": {"vocab": {"hello": "zero"}}},
    ],
)
def test_tokenizer_wrong_shape_raises_model_error(
    tmp_path: Path,
    payload: Any,
) -> None:
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelError) as excinfo:
        parse_tokenizer_json(path)

    _assert_model_error(excinfo, str(path))


def test_vocab_wrong_shape_raises_model_error(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(["hello"]), encoding="utf-8")

    with pytest.raises(ModelError) as excinfo:
        parse_vocab_json(path)

    _assert_model_error(excinfo, str(path))


def test_build_token_text_via_decode_uses_decode_per_id() -> None:
    model = FakeModel()

    result = build_token_text_via_decode(model, [3, 1, 2])

    assert result == {3: "<3>", 1: "<1>", 2: "<2>"}
    assert model.calls == [[3], [1], [2]]


def test_validate_token_text_accepts_complete_mapping() -> None:
    # Empty strings are allowed; only coverage and range are enforced.
    validate_token_text({0: "a", 1: "b", 2: ""}, 3)


def test_validate_token_text_rejects_out_of_range_id() -> None:
    with pytest.raises(ModelError) as excinfo:
        validate_token_text({0: "a", 3: "d"}, 3)

    _assert_model_error(excinfo, "outside", "3")


def test_validate_token_text_rejects_missing_ids() -> None:
    with pytest.raises(ModelError) as excinfo:
        validate_token_text({0: "a", 2: "c"}, 3)

    _assert_model_error(excinfo, "missing", "1")


def test_validate_token_text_rejects_empty_mapping() -> None:
    with pytest.raises(ModelError) as excinfo:
        validate_token_text({}, 3)

    _assert_model_error(excinfo, "empty")


def test_cache_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "token_text.json"
    original = {0: "hello", 1: "Ġhello", 2: "<|special|>"}

    save_token_text_cache(path, original)
    loaded = load_token_text_cache(path)

    assert loaded == original
    assert path.exists()


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        {"abc": "text"},
        {"0": 1},
    ],
)
def test_load_token_text_cache_wrong_shape_raises(
    tmp_path: Path,
    payload: Any,
) -> None:
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelError) as excinfo:
        load_token_text_cache(path)

    _assert_model_error(excinfo, str(path))


def test_excluded_token_ids_covers_all_added_tokens() -> None:
    tokenizer = parse_tokenizer_json(TOKENIZER_FIXTURE)

    assert excluded_token_ids(tokenizer) == tokenizer.added_ids
    assert excluded_token_ids(tokenizer) == frozenset({2, 4, 5})


def test_excluded_token_ids_includes_non_special_added_marker() -> None:
    tokenizer = parse_tokenizer_json(TOKENIZER_FIXTURE)

    # Id 5 is added with special=false, but it is still a control marker.
    assert 5 in tokenizer.added_ids
    assert 5 not in tokenizer.special_ids
    assert 5 in excluded_token_ids(tokenizer)


def test_unmapped_token_ids_reports_padding_positions() -> None:
    tokenizer = parse_tokenizer_json(TOKENIZER_FIXTURE)

    # The fixture covers ids 0..5, so 6..7 are logits-only padding positions.
    assert unmapped_token_ids(tokenizer, 8) == frozenset({6, 7})


def test_unmapped_token_ids_empty_when_logits_match_vocab() -> None:
    tokenizer = parse_tokenizer_json(TOKENIZER_FIXTURE)

    assert unmapped_token_ids(tokenizer, 6) == frozenset()


@pytest.mark.parametrize("logits_size", [0, -1])
def test_unmapped_token_ids_rejects_non_positive_size(logits_size: int) -> None:
    tokenizer = parse_tokenizer_json(TOKENIZER_FIXTURE)

    with pytest.raises(ModelError) as excinfo:
        unmapped_token_ids(tokenizer, logits_size)

    _assert_model_error(excinfo, "logits_size")


def test_build_token_index_excludes_all_added_tokens() -> None:
    tokenizer = parse_tokenizer_json(TOKENIZER_FIXTURE)
    index = build_token_index(tokenizer)

    mapped = set(tokenizer.token_text)
    assert set(index.allowed_ids) == mapped - tokenizer.added_ids
    assert index.allowed_ids == (0, 1, 3)
    for added_id in tokenizer.added_ids:
        assert added_id not in index.allowed_ids


def test_tokens_starting_with_empty_prefix_returns_allowed_ids() -> None:
    index = build_token_index(parse_tokenizer_json(TOKENIZER_FIXTURE))

    assert index.tokens_starting_with("") == index.allowed_ids


def test_tokens_starting_with_multi_char_prefix_filters_by_text() -> None:
    index = build_token_index(parse_tokenizer_json(TOKENIZER_FIXTURE))

    assert index.tokens_starting_with("hel") == (0,)
    assert index.tokens_starting_with("Ġh") == (1,)


def test_tokens_starting_with_unknown_prefix_returns_empty() -> None:
    index = build_token_index(parse_tokenizer_json(TOKENIZER_FIXTURE))

    assert index.tokens_starting_with("zzz") == ()
    assert index.tokens_starting_with("\ufffd") == ()


def test_by_first_char_keys_match_allowed_tokens() -> None:
    tokenizer = parse_tokenizer_json(TOKENIZER_FIXTURE)
    index = build_token_index(tokenizer)

    expected = {
        tokenizer.token_text[token_id][0]
        for token_id in index.allowed_ids
        if tokenizer.token_text[token_id]
    }
    assert set(index.by_first_char) == expected
    for char, token_ids in index.by_first_char.items():
        assert token_ids == tuple(sorted(token_ids))
        assert all(
            tokenizer.token_text[token_id].startswith(char)
            for token_id in token_ids
        )


def test_build_token_index_keys_byte_fragments_on_replacement_char() -> None:
    tokenizer = parse_tokenizer_json(TOKENIZER_FIXTURE)
    texts = dict(tokenizer.token_text)
    texts[3] = "\ufffd"

    index = build_token_index(tokenizer, texts=texts)

    assert index.texts[3] == "\ufffd"
    assert index.by_first_char["\ufffd"] == (3,)


def test_artificial_single_char_tokenizer_and_index() -> None:
    tokenizer = parse_tokenizer_json(TOKENIZER_CHARS_FIXTURE)

    expected_vocab = {
        token_id: raw for raw, token_id in CHARS_RAW_TO_ID.items()
    }
    expected_vocab[13] = "<|special|>"
    expected_vocab[14] = "<tool_call>"
    assert tokenizer.token_text == expected_vocab
    assert tokenizer.special_ids == frozenset({13})
    assert tokenizer.added_ids == CHARS_ADDED_IDS

    index = build_token_index(tokenizer)
    assert index.allowed_ids == tuple(range(13))
    assert index.by_first_char == {
        raw: (token_id,) for raw, token_id in CHARS_RAW_TO_ID.items()
    }
    for char in ("{", "}", "[", "]", '"', ":", ",", "0", "a", "Ġ", "Ċ"):
        assert index.tokens_starting_with(char) == (CHARS_RAW_TO_ID[char],)

    allowed = set(index.allowed_ids)
    assert CHARS_ADDED_IDS.isdisjoint(allowed)
    assert CHARS_ADDED_IDS.isdisjoint(index.tokens_starting_with(""))
    assert index.tokens_starting_with("") == index.allowed_ids


def test_artificial_index_decoded_space_and_newline() -> None:
    tokenizer = parse_tokenizer_json(TOKENIZER_CHARS_FIXTURE)
    raw_index = build_token_index(tokenizer)
    # Raw BPE text uses placeholders, so a literal space/newline finds nothing.
    assert raw_index.tokens_starting_with(" ") == ()
    assert raw_index.tokens_starting_with("\n") == ()

    decoded = dict(tokenizer.token_text)
    decoded[CHARS_RAW_TO_ID["Ġ"]] = " "
    decoded[CHARS_RAW_TO_ID["Ċ"]] = "\n"
    index = build_token_index(tokenizer, texts=decoded)

    assert index.tokens_starting_with(" ") == (CHARS_RAW_TO_ID["Ġ"],)
    assert index.tokens_starting_with("\n") == (CHARS_RAW_TO_ID["Ċ"],)
    assert index.by_first_char[" "] == (CHARS_RAW_TO_ID["Ġ"],)
    assert index.by_first_char["\n"] == (CHARS_RAW_TO_ID["Ċ"],)


def test_artificial_encode_decode_round_trip_for_json() -> None:
    id_to_decoded = _chars_id_to_decoded()
    decoded_to_id = {
        text: token_id for token_id, text in id_to_decoded.items()
    }
    assert len(decoded_to_id) == len(CHARS_RAW_TO_ID)

    text = '{"a": 1}'
    ids = [decoded_to_id[char] for char in text]

    assert "".join(id_to_decoded[token_id] for token_id in ids) == text


@pytest.fixture(scope="module")
def real_model() -> Any:
    """Instantiate the real SDK model once for the slow round-trip tests."""
    from llm_sdk import Small_LLM_Model

    return Small_LLM_Model()


@pytest.mark.slow
@pytest.mark.parametrize("text", ['{"a": 1}', '"hello"', '"olá"'])
def test_real_model_encode_decode_round_trip(
    real_model: Any,
    text: str,
) -> None:
    # encode() returns a 2-D tensor and decode() calls .tolist() on it, which
    # yields a nested list; flatten the batch dimension before decoding.
    ids = real_model.encode(text)[0]
    assert real_model.decode(ids) == text


def test_load_token_text_cache_rejects_negative_id(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"-1": "x"}), encoding="utf-8")

    with pytest.raises(ModelError) as excinfo:
        load_token_text_cache(path)

    _assert_model_error(excinfo, str(path), "non-negative integer string")


def test_load_token_text_cache_with_expected_size_rejects_out_of_range(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"0": "a", "3": "d"}), encoding="utf-8")

    with pytest.raises(ModelError) as excinfo:
        load_token_text_cache(path, expected_size=3)

    _assert_model_error(excinfo, "outside", "3")


def test_load_token_text_cache_with_expected_size_accepts_complete_cache(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cache.json"
    path.write_text(
        json.dumps({"0": "a", "1": "b", "2": "c"}),
        encoding="utf-8",
    )

    assert load_token_text_cache(path, expected_size=3) == {
        0: "a",
        1: "b",
        2: "c",
    }


def test_out_of_range_cache_id_is_rejected_by_validate_token_text(
    tmp_path: Path,
) -> None:
    # The cache loader checks key shape only; validate_token_text enforces range.
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"0": "a", "3": "d"}), encoding="utf-8")
    loaded = load_token_text_cache(path)

    with pytest.raises(ModelError) as excinfo:
        validate_token_text(loaded, 3)

    _assert_model_error(excinfo, "outside", "3")


def test_validate_token_text_rejects_max_id_equal_to_size() -> None:
    # Off-by-one: id 3 is the first id outside range(3).
    with pytest.raises(ModelError) as excinfo:
        validate_token_text({0: "a", 3: "d"}, 3)

    _assert_model_error(excinfo, "outside", "range(3)", "3")


@pytest.mark.parametrize(
    "fixture",
    [TOKENIZER_FIXTURE, TOKENIZER_CHARS_FIXTURE],
)
def test_index_allowed_ids_exclude_added_tokens(fixture: Path) -> None:
    tokenizer = parse_tokenizer_json(fixture)
    index = build_token_index(tokenizer)
    excluded = set(excluded_token_ids(tokenizer))

    assert set(index.allowed_ids) & excluded == set()
    assert set(index.tokens_starting_with("")) & excluded == set()
