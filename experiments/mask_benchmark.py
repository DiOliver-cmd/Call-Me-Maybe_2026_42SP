"""Per-slot token-mask benchmark for T5.1.

Run manually with: ``uv run python -m experiments.mask_benchmark``

The benchmark requires the decoded token cache at ``.cache/token_text.json`` and
fails loudly when it is missing; raw tokenizer strings are not a valid
substitute. The tokenizer asset is read from ``data/input/tokenizer.json`` when
present, otherwise from the pinned Qwen Hugging Face cache without loading model
weights. No logits or model forward passes are used.
"""

from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

from src.decoder import _SLOT_TARGETS, allowed_token_ids
from src.dfa import DFA
from src.errors import ModelError
from src.vocab import (
    TokenizerIndex,
    build_token_index,
    load_token_text_cache,
    parse_tokenizer_json,
)

TOKENIZER_PATH = Path("data/input/tokenizer.json")
TOKEN_TEXT_CACHE = Path(".cache/token_text.json")
RUNS = 5

NAMES = [
    "fn_add_numbers",
    "fn_greet",
    "fn_reverse_string",
    "fn_get_square_root",
    "fn_substitute_string_with_regex",
    "fn_set_alarm",
]

STATES = (
    ("NAME_VALUE", '{"prompt":"x","name":"'),
    (
        "PARAM_VALUE_START",
        '{"prompt":"x","name":"fn_add_numbers","parameters":{"a":',
    ),
    (
        "PARAM_VALUE_STRING",
        '{"prompt":"x","name":"fn_greet","parameters":{"name":"',
    ),
    (
        "PARAM_VALUE_NUMBER",
        '{"prompt":"x","name":"fn_add_numbers","parameters":{"a":1',
    ),
    (
        "PARAM_VALUE_BOOLEAN",
        '{"prompt":"x","name":"fn_set_alarm","parameters":{"enabled":t',
    ),
    (
        "PARAM_VALUE_DONE",
        '{"prompt":"x","name":"fn_greet","parameters":{"name":"bob"',
    ),
    ("ACCEPT", '{"prompt":"x","name":"fn_greet","parameters":{}}'),
)


def _resolve_tokenizer_path() -> Path:
    """Return the local tokenizer asset, using the pinned HF cache if needed."""
    if TOKENIZER_PATH.exists():
        return TOKENIZER_PATH

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download("Qwen/Qwen3-0.6B", "tokenizer.json"))


def _load_decoded_texts(vocab_size: int) -> dict[int, str]:
    """Load the decoded cache or fail with an actionable message."""
    if not TOKEN_TEXT_CACHE.exists():
        raise ModelError(
            f"required decoded token cache missing: {TOKEN_TEXT_CACHE}; "
            "build it with experiments.index_benchmark or the T2.2 cache step"
        )
    return load_token_text_cache(
        TOKEN_TEXT_CACHE,
        expected_size=vocab_size,
    )


def _walk(text: str, names: list[str]) -> DFA:
    """Return a DFA after consuming every character of ``text``."""
    dfa = DFA.start(names)
    for index, char in enumerate(text):
        if not dfa.consume(char):
            raise RuntimeError(
                f"benchmark prefix rejected at index {index} "
                f"({char!r}): {text!r}"
            )
    return dfa


def _median_seconds(fn: Callable[[], object], runs: int = RUNS) -> float:
    """Return the median wall time of ``runs`` calls to ``fn``."""
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _probe_counts(
    dfa: DFA,
    index: TokenizerIndex,
) -> tuple[int, int, int]:
    """Return validated token count and surviving/total bucket counts."""
    if dfa.state not in _SLOT_TARGETS:
        return 0, 0, 0
    surviving = 0
    validated = 0
    for first_char, token_ids in index.by_first_char.items():
        if dfa.try_consume_token(first_char) is not None:
            surviving += 1
            validated += len(token_ids)
    return validated, surviving, len(index.by_first_char)


def main() -> int:
    tokenizer_path = _resolve_tokenizer_path()
    tokenizer = parse_tokenizer_json(tokenizer_path)
    try:
        texts = _load_decoded_texts(tokenizer.vocab_size)
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    start = time.perf_counter()
    index = build_token_index(tokenizer, texts)
    build_seconds = time.perf_counter() - start

    print(f"tokenizer={tokenizer_path}")
    print(
        f"mapped={tokenizer.vocab_size} "
        f"allowed={len(index.allowed_ids)} "
        f"text=decoded cache "
        f"index_build_ms={build_seconds * 1000:.1f}"
    )
    print(
        f"{'state':<21} {'candidates':>10} {'validated':>10} "
        f"{'buckets':>12} {'ms/step':>10}"
    )

    for label, prefix in STATES:
        dfa = _walk(prefix, NAMES)
        candidates = allowed_token_ids(dfa, index)
        allowed_token_ids(dfa, index)  # warm up once
        validated, surviving, total = _probe_counts(dfa, index)
        milliseconds = _median_seconds(
            lambda: allowed_token_ids(dfa, index)
        ) * 1000
        print(
            f"{label:<21} {len(candidates):>10} {validated:>10} "
            f"{surviving:>5}/{total:<6} {milliseconds:>10.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
