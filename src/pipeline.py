"""Pipeline orchestration: prompt batch to validated function calls (T6.1).

The pipeline owns the reusable tokenizer index and the prompt loop, while
:func:`src.decoder.decode_call` owns one constrained generation. Calls are
processed in input order and the original prompt text is preserved by the
decoder. If any prompt fails, the run raises instead of returning a partial
list, so a caller cannot accidentally write incomplete output (T6.3).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from src.data_loader import PathLike, load_functions, load_prompts
from src.decoder import NUMBER_EOS_BONUS, decode_call
from src.errors import DecodingError, ProjectError, SchemaError
from src.llm_client import LLMClient
from src.schemas import FunctionCall, FunctionCallBatch
from src.vocab import (
    TokenizerIndex,
    build_token_index,
    build_token_text_via_decode,
    load_token_text_cache,
    parse_tokenizer_json,
    save_token_text_cache,
)

__all__ = [
    "DEFAULT_TOKEN_TEXT_CACHE",
    "build_decoder_index",
    "run",
    "write_output",
]

DEFAULT_TOKEN_TEXT_CACHE = Path(".cache/token_text.json")


def build_decoder_index(
    model: LLMClient | None = None,
    *,
    cache_path: PathLike = DEFAULT_TOKEN_TEXT_CACHE,
) -> TokenizerIndex:
    """Build the decoder index using the tokenizer asset and a text cache.

    The tokenizer metadata always comes from ``model.tokenizer_path()``. A cache
    that exists but is stale or partial is discarded after any
    :class:`ProjectError` from loading and rebuilt from the current tokenizer.
    A successful rebuild writes the cache through
    :func:`save_token_text_cache`, creating its parent directory first.
    """
    client = model or LLMClient()
    vocab = parse_tokenizer_json(client.tokenizer_path())
    cache_file = Path(cache_path)

    texts: dict[int, str] | None = None
    if cache_file.exists():
        try:
            texts = load_token_text_cache(
                cache_file,
                expected_size=vocab.vocab_size,
            )
        except ProjectError:
            texts = None

    if texts is None:
        texts = build_token_text_via_decode(
            client,
            range(vocab.vocab_size),
        )
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        save_token_text_cache(cache_file, texts)

    return build_token_index(vocab, texts)


def run(
    functions_path: PathLike,
    input_path: PathLike,
    *,
    model: LLMClient | None = None,
    index: TokenizerIndex | None = None,
    eos_token_id: int | None = None,
) -> list[FunctionCall]:
    """Decode one function call per prompt, preserving input order.

    ``index`` is injectable for deterministic tests and avoids touching the
    token-text cache. Any :class:`ProjectError` from a prompt is wrapped in a
    :class:`DecodingError` naming the zero-based prompt position and its text;
    the exception aborts the run, so no partial list is returned.
    """
    catalog = load_functions(functions_path)
    prompts = load_prompts(input_path)
    client = model or LLMClient()
    decoder_index = index or build_decoder_index(client)

    calls: list[FunctionCall] = []
    for prompt_index, prompt in enumerate(prompts):
        try:
            call = decode_call(
                client,
                catalog,
                prompt,
                index=decoder_index,
                eos_token_id=eos_token_id,
                number_eos_bonus=NUMBER_EOS_BONUS,
            )
            calls.append(catalog.validate_call(call))
        except ProjectError as exc:
            raise DecodingError(
                f"prompt {prompt_index} ({prompt!r}) failed: {exc}"
            ) from exc

    return calls


def write_output(path: PathLike, calls: Sequence[FunctionCall]) -> None:
    """Validate and atomically write the function-call batch as JSON.

    Validation happens before any filesystem change, so an invalid call leaves
    both the target and its parent untouched. The payload is written to a
    temporary file in the target directory, flushed and fsynced, then published
    with :func:`os.replace`; any failure removes the temporary file and
    re-raises, so a reader never observes a partial output file.
    """
    try:
        batch = FunctionCallBatch.model_validate(list(calls))
    except ValidationError as exc:
        raise SchemaError(f"invalid function call batch: {exc}") from exc

    payload = [call.model_dump(mode="json") for call in batch.root]
    target = Path(path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        assert temp_path is not None
        os.replace(temp_path, target)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
