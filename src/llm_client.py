"""Small typed wrapper over the LLM SDK's public API (T3.1).

The client isolates the rest of the project from SDK details: it normalizes the
SDK's 2-D ``encode`` tensor to a flat ``list[int]`` and its logits to a 1-D
``float64`` numpy array. The real model is imported and constructed lazily on
first use, so importing this module is cheap; injecting a fake backend skips the
real model entirely.

No ``torch`` or ``transformers`` imports live here.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np

from src.errors import ModelError

__all__ = ["LLMBackend", "LLMClient"]


class LLMBackend(Protocol):
    """Minimal backend surface ``LLMClient`` depends on.

    Both the real ``llm_sdk.Small_LLM_Model`` and the test fake satisfy this
    protocol structurally, so the client never needs to know which one it has.
    """

    def encode(self, text: str) -> Any:
        """Tokenize ``text``; may return a tensor-like object or a list."""
        ...

    def decode(self, ids: list[int]) -> str:
        """Decode flat token ids to text."""
        ...

    def get_logits_from_input_ids(self, input_ids: list[int]) -> Any:
        """Return the next-token logits for ``input_ids``."""
        ...

    def get_path_to_vocab_file(self) -> str:
        """Return the path to the vocabulary asset."""
        ...


class LLMClient:
    """Normalize the SDK's public methods behind a small, testable interface."""

    def __init__(self, backend: LLMBackend | None = None) -> None:
        """Store ``backend``; the real SDK is loaded only on first use."""
        self._backend = backend

    def _get_backend(self) -> LLMBackend:
        """Return the injected backend or lazily construct the real model."""
        backend = self._backend
        if backend is None:
            from llm_sdk import Small_LLM_Model

            backend = cast(LLMBackend, Small_LLM_Model())
            self._backend = backend
        return backend

    def encode(self, text: str) -> list[int]:
        """Return flat token ids for ``text``."""
        return _normalize_ids(self._get_backend().encode(text))

    def decode(self, ids: list[int]) -> str:
        """Decode flat token ids to text."""
        result = self._get_backend().decode(list(ids))
        if not isinstance(result, str):
            raise ModelError(
                f"decode returned {type(result).__name__}; expected str"
            )
        return result

    def logits(self, ids: list[int]) -> np.ndarray:
        """Return the last-position logits as a 1-D ``float64`` array.

        With the real model the vector has one value per vocabulary position
        (151,936, including the 267 padding ids).
        """
        raw = self._get_backend().get_logits_from_input_ids(list(ids))
        return _normalize_logits(raw)

    def vocab_path(self) -> str:
        """Return the path to the vocabulary asset."""
        return self._get_backend().get_path_to_vocab_file()


def _normalize_ids(value: Any) -> list[int]:
    """Normalize a tensor-like or nested list of ids to a flat list."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value is None:
        raise ModelError("encode returned None; expected token ids")
    if isinstance(value, (int, float)):
        raise ModelError(
            f"encode returned a scalar {type(value).__name__}; "
            f"expected a list of ids"
        )
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ModelError(
            f"encode returned {type(value).__name__}; "
            f"expected a list of ids"
        )
    return _flatten_ids(value)


def _flatten_ids(value: list[Any]) -> list[int]:
    """Flatten a 1-D list or a single-row 2-D batch, validating every id."""
    if not value:
        return []
    if isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ModelError(
                f"encode returned a {len(value)}-row batch; expected one row"
            )
        return _flatten_ids(list(value[0]))

    ids: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ModelError(
                f"encode returned non-integer id {item!r}; expected ints"
            )
        ids.append(item)
    return ids


def _normalize_logits(value: Any) -> np.ndarray:
    """Normalize logits to a non-empty 1-D ``float64`` array."""
    if value is None:
        raise ModelError("logits returned None; expected a numeric vector")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelError(f"logits are not numeric: {exc}") from exc
    if array.ndim == 0:
        raise ModelError(
            f"logits returned a scalar {type(value).__name__}; "
            f"expected a vector"
        )
    flat = array.reshape(-1)
    if flat.size == 0:
        raise ModelError("logits vector is empty; expected at least one value")
    return flat
