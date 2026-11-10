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

    def get_path_to_tokenizer_file(self) -> str:
        """Return an unused tokenizer path; pipeline tests inject an index."""
        return "unused"


class ScriptedLLM:
    """Deterministic backend that plays a fixed script of next-token ids.

    ``encode`` returns a one-row batch (like the SDK), ``decode`` joins character
    ids, and ``get_logits_from_input_ids`` pops the next scripted id, sets it to
    10.0, and records the call. ``distractors`` maps a 1-based call number to
    extra ids set to 100.0, so masking can be tested against a higher invalid
    logit. No NumPy dependency is used here.
    """

    def __init__(
        self,
        script: list[int],
        *,
        size: int = 2048,
        all_masked_at: int | None = None,
        distractors: dict[int, list[int]] | None = None,
    ) -> None:
        self.script = list(script)
        self.size = size
        self.all_masked_at = all_masked_at
        self.distractors = {
            call: tuple(ids) for call, ids in (distractors or {}).items()
        }
        self.calls = 0
        self.encode_calls: list[str] = []
        self.decode_calls: list[list[int]] = []

    def encode(self, text: str) -> list[list[int]]:
        """Record ``text`` and return one character-id row."""
        self.encode_calls.append(text)
        return [[ord(char) for char in text]]

    def decode(self, ids: list[int]) -> str:
        """Record ``ids`` and decode them as characters."""
        self.decode_calls.append(list(ids))
        return "".join(chr(token_id) for token_id in ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Record a model step and return logits favouring the script."""
        self.calls += 1
        if self.all_masked_at == self.calls:
            return [float("-inf")] * self.size
        if not self.script:
            raise AssertionError("logits called after the script was exhausted")

        token_id = self.script.pop(0)
        logits = [-1.0] * self.size
        logits[token_id] = 10.0
        for distractor_id in self.distractors.get(self.calls, ()):
            logits[distractor_id] = 100.0
        return logits

    def get_path_to_vocab_file(self) -> str:
        """Return an unused path; decoding tests never load the real vocab."""
        return "unused"

    def get_path_to_tokenizer_file(self) -> str:
        """Return an unused tokenizer path; decoding tests never load it."""
        return "unused"
