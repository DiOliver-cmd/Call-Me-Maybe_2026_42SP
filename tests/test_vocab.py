"""Unit tests for vocabulary loading and token_text construction (T2.1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.errors import ModelError, ProjectError
from src.vocab import (
    TokenizerVocab,
    build_token_text_via_decode,
    load_token_text_cache,
    parse_tokenizer_json,
    parse_vocab_json,
    save_token_text_cache,
    special_token_ids,
    validate_token_text,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TOKENIZER_FIXTURE = FIXTURES_DIR / "tokenizer_mini.json"
VOCAB_FIXTURE = FIXTURES_DIR / "vocab_mini.json"


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
    }
    assert result.special_ids == frozenset({2, 4})
    assert special_token_ids(result) == frozenset({2, 4})
    assert result.vocab_size == 5
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
