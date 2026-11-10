"""Deterministic tests for the T6.1 pipeline orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.errors import DecodingError, SchemaError
from src.llm_client import LLMClient
from src.pipeline import build_decoder_index, run
from src.schemas import FunctionCall
from src.vocab import TokenizerIndex
from tests.fakes import ScriptedLLM

_EOS_ID = 1000
_DECODER_CHARS = 'abcdefghijklmnopqrstuvwxyz_" '


def _tokens(text: str) -> list[int]:
    """Return one character token id per character."""
    return [ord(char) for char in text]


def _pipeline_index() -> TokenizerIndex:
    """Small char-token index matching the temporary catalog names."""
    texts = {ord(char): char for char in sorted(set(_DECODER_CHARS))}
    texts[_EOS_ID] = "<|im_end|>"
    allowed_ids = tuple(
        token_id for token_id in sorted(texts) if token_id != _EOS_ID
    )
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


def _greet_script(name: str) -> list[int]:
    return _tokens("fn_greet") + [_EOS_ID] + _tokens(f'"{name}"')


def _ping_script() -> list[int]:
    return _tokens("fn_ping") + [_EOS_ID]


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    functions_path = tmp_path / "functions.json"
    prompts_path = tmp_path / "prompts.json"
    prompts = ["Greet bob", 'say "hi" José', "Ping"]

    functions_path.write_text(
        json.dumps(
            [
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
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    prompts_path.write_text(
        json.dumps(
            [{"prompt": prompt} for prompt in prompts],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return functions_path, prompts_path, prompts


class _IndexBackend:
    """Backend exposing only the tokenizer/decode surface needed by the index."""

    def __init__(
        self,
        tokenizer_path: Path,
        decoded: dict[int, str],
    ) -> None:
        self.tokenizer_path = tokenizer_path
        self.decoded = decoded
        self.decode_calls: list[list[int]] = []

    def encode(self, text: str) -> list[list[int]]:
        return [_tokens(text)]

    def decode(self, ids: list[int]) -> str:
        self.decode_calls.append(list(ids))
        return self.decoded[ids[0]]

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        raise AssertionError("logits are not used while building an index")

    def get_path_to_vocab_file(self) -> str:
        return "unused"

    def get_path_to_tokenizer_file(self) -> str:
        return str(self.tokenizer_path)


def _write_tokenizer(tmp_path: Path) -> Path:
    path = tmp_path / "tokenizer.json"
    path.write_text(
        json.dumps(
            {
                "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}},
                "added_tokens": [
                    {
                        "id": 2,
                        "content": "<|im_end|>",
                        "special": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_run_preserves_order_prompts_and_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    functions_path, prompts_path, prompts = _write_inputs(tmp_path)
    script = _greet_script("bob") + _greet_script("bob") + _ping_script()
    backend = ScriptedLLM(script)

    def fail_tokenizer_path() -> str:
        raise AssertionError("run with an injected index must not build one")

    monkeypatch.setattr(
        backend,
        "get_path_to_tokenizer_file",
        fail_tokenizer_path,
    )

    calls = run(
        functions_path,
        prompts_path,
        model=LLMClient(backend=backend),
        index=_pipeline_index(),
    )

    assert [call.prompt for call in calls] == prompts
    assert [call.name for call in calls] == ["fn_greet", "fn_greet", "fn_ping"]
    assert calls[0].parameters == {"name": "bob"}
    assert calls[1].parameters == {"name": "bob"}
    assert calls[2].parameters == {}
    assert backend.calls == len(script)


def test_run_wraps_middle_prompt_error_with_position_and_text(
    tmp_path: Path,
) -> None:
    functions_path, prompts_path, prompts = _write_inputs(tmp_path)
    first_call = _greet_script("bob")
    script = first_call + _greet_script("bob") + _ping_script()
    backend = ScriptedLLM(script, all_masked_at=len(first_call) + 1)

    with pytest.raises(DecodingError) as excinfo:
        run(
            functions_path,
            prompts_path,
            model=LLMClient(backend=backend),
            index=_pipeline_index(),
        )

    message = str(excinfo.value)
    assert "prompt 1" in message
    assert prompts[1] in message
    assert backend.calls == len(first_call) + 1


def test_run_wraps_validation_failure_with_prompt_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    functions_path, prompts_path, prompts = _write_inputs(tmp_path)

    def invalid_decode(*args: object, **kwargs: object) -> FunctionCall:
        return FunctionCall(
            prompt=prompts[0],
            name="fn_missing",
            parameters={},
        )

    monkeypatch.setattr("src.pipeline.decode_call", invalid_decode)

    with pytest.raises(DecodingError) as excinfo:
        run(
            functions_path,
            prompts_path,
            model=LLMClient(backend=ScriptedLLM([])),
            index=_pipeline_index(),
        )

    message = str(excinfo.value)
    assert "prompt 0" in message
    assert prompts[0] in message
    assert "fn_missing" in message
    assert isinstance(excinfo.value.__cause__, SchemaError)


def test_build_decoder_index_uses_and_reuses_cache(tmp_path: Path) -> None:
    tokenizer_path = _write_tokenizer(tmp_path)
    backend = _IndexBackend(
        tokenizer_path,
        {0: "a", 1: "b", 2: ""},
    )
    model = LLMClient(backend=backend)
    cache_path = tmp_path / "nested" / "token_text.json"

    first = build_decoder_index(model, cache_path=cache_path)

    assert first.texts == {0: "a", 1: "b", 2: "<|im_end|>"}
    assert first.allowed_ids == (0, 1)
    assert cache_path.exists()
    assert len(backend.decode_calls) == 3

    second = build_decoder_index(model, cache_path=cache_path)

    assert second == first
    assert len(backend.decode_calls) == 3


def test_build_decoder_index_rebuilds_a_stale_cache(tmp_path: Path) -> None:
    tokenizer_path = _write_tokenizer(tmp_path)
    backend = _IndexBackend(
        tokenizer_path,
        {0: "a", 1: "b", 2: ""},
    )
    model = LLMClient(backend=backend)
    cache_path = tmp_path / "token_text.json"
    cache_path.write_text('{"0": "stale"}', encoding="utf-8")

    index = build_decoder_index(model, cache_path=cache_path)

    assert index.texts == {0: "a", 1: "b", 2: "<|im_end|>"}
    assert len(backend.decode_calls) == 3
