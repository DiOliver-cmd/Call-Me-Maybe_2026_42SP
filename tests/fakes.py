"""Dependency-free test doubles for the LLM backend (T3.1)."""

from __future__ import annotations

from typing import Any


class _TensorLike:
    """Minimal object exposing ``tolist`` like a torch tensor."""

    def __init__(self, data: Any) -> None:
        self._data = data

    def tolist(self) -> Any:
        """Return the wrapped data unchanged."""
        return self._data


class FakeLLM:
    """Controllable backend that records every call made by ``LLMClient``."""

    def __init__(
        self,
        encode_result: Any,
        decode_result: str,
        logits_result: Any,
        vocab_path_text: str,
    ) -> None:
        self.encode_result = encode_result
        self.decode_result = decode_result
        self.logits_result = logits_result
        self.vocab_path_text = vocab_path_text

        self.encode_calls: list[str] = []
        self.decode_calls: list[list[int]] = []
        self.logits_calls: list[list[int]] = []
        self.vocab_path_calls = 0

    def encode(self, text: str) -> Any:
        """Record ``text`` and return the configured result."""
        self.encode_calls.append(text)
        return self.encode_result

    def decode(self, ids: list[int]) -> str:
        """Record ``ids`` and return the configured text."""
        self.decode_calls.append(list(ids))
        return self.decode_result

    def get_logits_from_input_ids(self, input_ids: list[int]) -> Any:
        """Record ``input_ids`` and return the configured logits."""
        self.logits_calls.append(list(input_ids))
        return self.logits_result

    def get_path_to_vocab_file(self) -> str:
        """Record the lookup and return the configured vocabulary path."""
        self.vocab_path_calls += 1
        return self.vocab_path_text
