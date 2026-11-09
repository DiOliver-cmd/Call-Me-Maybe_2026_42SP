"""Unit tests for the LLM client wrapper (T3.1)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.errors import ModelError
from src.llm_client import LLMClient
from tests.fakes import FakeLLM, _TensorLike


def _fake(
    encode_result: Any = None,
    decode_result: str = "text",
    logits_result: Any = None,
    vocab_path_text: str = "vocab.json",
) -> FakeLLM:
    """Build a fake with convenient defaults for one focused assertion."""
    return FakeLLM(
        encode_result=[1] if encode_result is None else encode_result,
        decode_result=decode_result,
        logits_result=[0.0] if logits_result is None else logits_result,
        vocab_path_text=vocab_path_text,
    )


@pytest.mark.parametrize(
    ("encode_result", "expected"),
    [
        ([1, 2, 3], [1, 2, 3]),
        ([[4, 5]], [4, 5]),
        (_TensorLike([6, 7]), [6, 7]),
        (_TensorLike([[8, 9]]), [8, 9]),
        ([], []),
    ],
)
def test_encode_normalizes_to_flat_list_of_ints(
    encode_result: Any,
    expected: list[int],
) -> None:
    fake = _fake(encode_result=encode_result)

    assert LLMClient(backend=fake).encode("hello") == expected
    assert fake.encode_calls == ["hello"]


def test_decode_forwards_ids_and_returns_text() -> None:
    fake = _fake(decode_result='"hello"')

    assert LLMClient(backend=fake).decode([1, 2, 3]) == '"hello"'
    assert fake.decode_calls == [[1, 2, 3]]


@pytest.mark.parametrize(
    ("logits_result", "expected"),
    [
        ([0.5, 1.5], [0.5, 1.5]),
        ([[2.5, 3.5]], [2.5, 3.5]),
    ],
)
def test_logits_normalizes_to_float64_1d(
    logits_result: Any,
    expected: list[float],
) -> None:
    fake = _fake(logits_result=logits_result)

    result = LLMClient(backend=fake).logits([9, 10])

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == (2,)
    np.testing.assert_array_equal(
        result,
        np.array(expected, dtype=np.float64),
    )
    assert fake.logits_calls == [[9, 10]]


def test_vocab_path_delegates() -> None:
    fake = _fake(vocab_path_text="/tmp/vocab.json")

    assert LLMClient(backend=fake).vocab_path() == "/tmp/vocab.json"
    assert fake.vocab_path_calls == 1


def test_encode_scalar_raises_model_error() -> None:
    fake = _fake(encode_result=7)

    with pytest.raises(ModelError) as excinfo:
        LLMClient(backend=fake).encode("x")

    assert type(excinfo.value) is ModelError
    assert "scalar" in str(excinfo.value)


def test_encode_multi_row_batch_raises_model_error() -> None:
    fake = _fake(encode_result=[[1], [2]])

    with pytest.raises(ModelError) as excinfo:
        LLMClient(backend=fake).encode("x")

    assert type(excinfo.value) is ModelError
    assert "2-row batch" in str(excinfo.value)


def test_logits_empty_raises_model_error() -> None:
    fake = _fake(logits_result=[])

    with pytest.raises(ModelError) as excinfo:
        LLMClient(backend=fake).logits([1])

    assert type(excinfo.value) is ModelError
    assert "empty" in str(excinfo.value)


def test_logits_scalar_raises_model_error() -> None:
    fake = _fake(logits_result=1.5)

    with pytest.raises(ModelError) as excinfo:
        LLMClient(backend=fake).logits([1])

    assert type(excinfo.value) is ModelError
    assert "scalar" in str(excinfo.value)


def test_injected_backend_is_used_without_real_model() -> None:
    fake = _fake()
    client = LLMClient(backend=fake)

    assert client.encode("x") == [1]
    assert client.decode([1]) == "text"
    assert client.logits([1]).shape == (1,)
    assert client.vocab_path() == "vocab.json"

    assert fake.encode_calls == ["x"]
    assert fake.decode_calls == [[1]]
    assert fake.logits_calls == [[1]]
    assert fake.vocab_path_calls == 1
