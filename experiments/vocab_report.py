"""Vocabulary and token_text report for the real model (T2.1).

Run manually with: ``uv run python -m experiments.vocab_report``

This loads the real model, decodes every vocabulary id exactly once, validates
coverage, writes ``.cache/token_text.json`` and prints a short summary. It is
not collected by pytest because pytest only scans ``tests/``.
"""

from __future__ import annotations

import time
from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.vocab import (
    build_token_text_via_decode,
    parse_tokenizer_json,
    save_token_text_cache,
    validate_token_text,
)

CACHE_PATH = Path(".cache/token_text.json")


def main() -> int:
    model = Small_LLM_Model()
    tokenizer_path = model.get_path_to_tokenizer_file()
    tokenizer = parse_tokenizer_json(tokenizer_path)

    ids = sorted(tokenizer.token_text)
    # The id range declared by the asset; validation also catches gaps.
    expected_size = ids[-1] + 1 if ids else 0

    start = time.perf_counter()
    token_text = build_token_text_via_decode(model, ids)
    elapsed = time.perf_counter() - start

    validate_token_text(token_text, expected_size)
    save_token_text_cache(CACHE_PATH, token_text)

    raw_diff_samples: list[tuple[int, str, str]] = []
    for token_id in ids:
        raw = tokenizer.token_text[token_id]
        decoded = token_text[token_id]
        if raw != decoded and len(raw_diff_samples) < 5:
            raw_diff_samples.append((token_id, raw, decoded))

    ascii_letters_equal = all(
        tokenizer.token_text[token_id] == token_text[token_id]
        for token_id in ids
        if len(tokenizer.token_text[token_id]) == 1
        and tokenizer.token_text[token_id].isascii()
        and tokenizer.token_text[token_id].isalpha()
    )

    print(f"tokenizer.json: {tokenizer_path}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"id range size: {expected_size}")
    print(f"special tokens: {len(tokenizer.special_ids)}")
    print(f"decode all ids: {elapsed:.3f}s")
    print("raw != decoded samples (id, raw, decoded):")
    for token_id, raw, decoded in raw_diff_samples:
        print(f"  {token_id}: {raw!r} -> {decoded!r}")
    print(f"raw == decoded for single ASCII letters: {ascii_letters_equal}")
    print(f"cache written: {CACHE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
