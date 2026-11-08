"""Vocabulary analysis spike: special, byte-level and multi-char tokens (T2.2).

Run manually with: ``uv run python -m experiments.vocab_analysis``

Loads the real model through the public SDK API, reuses
``.cache/token_text.json`` (building it if absent), and writes the deterministic
report ``experiments/vocab_report.md``. Not collected by pytest because pytest
only scans ``tests/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_sdk import Small_LLM_Model

from src.vocab import (
    TokenizerVocab,
    build_token_text_via_decode,
    excluded_token_ids,
    load_token_text_cache,
    parse_tokenizer_json,
    save_token_text_cache,
    unmapped_token_ids,
)

CACHE_PATH = Path(".cache/token_text.json")
REPORT_PATH = Path("experiments/vocab_report.md")

# The model exposes padding rows beyond the tokenizer id range.
FALLBACK_LOGITS_SIZE = 151936

SPECIAL_SEARCH = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<tool_call>",
    "</tool_call>",
)
BOUNDARY_CHARS = ('"', "}", ",", ":", "\\")
LOSSY_MARKER = "\ufffd"


def _escape(text: str) -> str:
    """Escape a table-cell value so a literal pipe cannot break Markdown."""
    return text.replace("|", "\\|")


def _code(text: str) -> str:
    """Wrap ``text`` in a code span whose fence cannot collide with content."""
    escaped = _escape(text)
    longest_run = 0
    current_run = 0
    for char in escaped:
        if char == "`":
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    fence = "`" * (longest_run + 1)
    return f"{fence}{escaped}{fence}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a deterministic Markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _load_or_build_token_text(
    model: Any,
    tokenizer: TokenizerVocab,
) -> dict[int, str]:
    """Load a fully covered cache, building and saving it when absent."""
    ids = sorted(tokenizer.token_text)
    # Id-range size: validation rejects missing or out-of-range ids.
    expected_size = ids[-1] + 1 if ids else None
    if CACHE_PATH.exists():
        return load_token_text_cache(
            CACHE_PATH,
            expected_size=expected_size,
        )
    token_text = build_token_text_via_decode(model, ids)
    save_token_text_cache(CACHE_PATH, token_text)
    return token_text


def _added_tokens(path: Any) -> list[tuple[int, str, bool]]:
    """Read raw ``added_tokens`` metadata (id, content, special) for the report.

    ``TokenizerVocab`` only keeps the ids whose ``special`` flag is true, so the
    analysis reads the asset once more to distinguish added-special tokens from
    added tokens that are ordinary text (for example ``<tool_call>``).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    result: list[tuple[int, str, bool]] = []
    for item in raw.get("added_tokens", []):
        result.append(
            (
                int(item["id"]),
                str(item["content"]),
                bool(item.get("special", False)),
            )
        )
    return result


def _examples_with(
    tokenizer: TokenizerVocab,
    token_text: dict[int, str],
    marker: str,
    limit: int = 3,
) -> list[tuple[int, str, str]]:
    """Pick tokens whose raw text contains ``marker``; prefer exact matches."""
    ids = sorted(tokenizer.token_text)
    exact = [i for i in ids if tokenizer.token_text[i] == marker]
    containing = [
        i for i in ids if marker in tokenizer.token_text[i] and i not in exact
    ]
    chosen = (exact + containing)[:limit]
    return [(i, tokenizer.token_text[i], token_text[i]) for i in chosen]


def _boundary_examples(
    tokenizer: TokenizerVocab,
    token_text: dict[int, str],
    excluded: frozenset[int],
    limit: int = 8,
) -> list[tuple[int, str, str]]:
    """Pick multi-character tokens that can cross a JSON/string boundary."""
    result: list[tuple[int, str, str]] = []
    for token_id in sorted(tokenizer.token_text):
        if token_id in excluded:
            continue
        raw = tokenizer.token_text[token_id]
        decoded = token_text.get(token_id, "")
        has_boundary = any(
            char in raw or char in decoded for char in BOUNDARY_CHARS
        )
        if (len(raw) > 1 or len(decoded) > 1) and has_boundary:
            result.append((token_id, raw, decoded))
        if len(result) >= limit:
            break
    return result


def _comparison_examples(
    tokenizer: TokenizerVocab,
    token_text: dict[int, str],
) -> tuple[list[int], list[int]]:
    """Select 2 lossy byte-fragment ids and 3 normal raw-vs-decoded ids."""
    ids = sorted(tokenizer.token_text)
    lossy = [
        token_id
        for token_id in range(94, 99)
        if token_id in token_text and LOSSY_MARKER in token_text[token_id]
    ]
    if len(lossy) < 2:
        lossy.extend(
            token_id
            for token_id in ids
            if LOSSY_MARKER in token_text[token_id] and token_id not in lossy
        )
    lossy_ids = lossy[:2]

    normal: list[int] = []
    for marker in ("Ġ", "Ċ"):
        for token_id in ids:
            if (
                tokenizer.token_text[token_id] == marker
                and token_id not in lossy_ids
                and token_id not in normal
            ):
                normal.append(token_id)
                break
    for token_id in ids:
        if len(normal) >= 3:
            break
        if token_id in lossy_ids or token_id in normal:
            continue
        raw = tokenizer.token_text[token_id]
        decoded = token_text[token_id]
        if raw != decoded and LOSSY_MARKER not in decoded:
            normal.append(token_id)
    if len(normal) < 3:
        for token_id in ids:
            if len(normal) >= 3:
                break
            if token_id in lossy_ids or token_id in normal:
                continue
            if LOSSY_MARKER not in token_text[token_id]:
                normal.append(token_id)
    return lossy_ids, normal[:3]


def main() -> int:
    model = Small_LLM_Model()
    tokenizer_path = model.get_path_to_tokenizer_file()
    tokenizer = parse_tokenizer_json(tokenizer_path)
    token_text = _load_or_build_token_text(model, tokenizer)

    ids = sorted(tokenizer.token_text)
    id_range_size = ids[-1] + 1 if ids else 0

    logits_size: int
    logits_source = "measured via get_logits_from_input_ids([0])"
    try:
        logits_size = len(model.get_logits_from_input_ids([0]))
    except Exception:
        logits_size = FALLBACK_LOGITS_SIZE
        logits_source = (
            f"documented constant {FALLBACK_LOGITS_SIZE} "
            f"(forward pass failed)"
        )

    excluded = excluded_token_ids(tokenizer)
    unmapped = unmapped_token_ids(tokenizer, logits_size)

    added_tokens = _added_tokens(tokenizer_path)
    added_sorted = sorted(added_tokens, key=lambda item: item[0])
    added_by_id = {token_id: special for token_id, _, special in added_tokens}
    special_added = [item for item in added_tokens if item[2]]
    non_special_added = [item for item in added_tokens if not item[2]]

    reverse: dict[str, list[int]] = {}
    for token_id, raw in tokenizer.token_text.items():
        reverse.setdefault(raw, []).append(token_id)

    added_rows = [
        [str(token_id), "yes" if special else "no", _code(repr(content))]
        for token_id, content, special in added_sorted
    ]
    search_rows: list[list[str]] = []
    for token in SPECIAL_SEARCH:
        found = sorted(reverse.get(token, []))
        if not found:
            search_rows.append(
                [_code(token), "NOT FOUND", "-", "-", "-"]
            )
            continue
        for token_id in found:
            search_rows.append(
                [
                    _code(token),
                    str(token_id),
                    "yes" if token_id in added_by_id else "no",
                    "yes" if added_by_id.get(token_id) else "no",
                    "yes" if token_id in excluded else "no",
                ]
            )

    space_examples = _examples_with(tokenizer, token_text, "Ġ")
    newline_examples = _examples_with(tokenizer, token_text, "Ċ")
    boundary_examples = _boundary_examples(tokenizer, token_text, excluded)
    lossy_ids, normal_ids = _comparison_examples(tokenizer, token_text)

    comparison_rows: list[list[str]] = []
    for token_id in lossy_ids:
        comparison_rows.append(
            [
                str(token_id),
                _code(repr(tokenizer.token_text[token_id])),
                _code(repr(token_text[token_id])),
                "lossy `U+FFFD`; byte fragment - keep allowed",
            ]
        )
    for token_id in normal_ids:
        comparison_rows.append(
            [
                str(token_id),
                _code(repr(tokenizer.token_text[token_id])),
                _code(repr(token_text[token_id])),
                "normal",
            ]
        )

    lines = [
        "# Vocabulary analysis (T2.2)",
        "",
        "Deterministic report generated by "
        "`uv run python -m experiments.vocab_analysis`.",
        "",
        "## 1. Sizes",
        "",
        _table(
            ["Quantity", "Value"],
            [
                ["`tokenizer.json` entries", str(tokenizer.vocab_size)],
                ["id range size (max id + 1)", str(id_range_size)],
                ["model logits size", f"{logits_size} ({logits_source})"],
                ["unmapped/padding ids", str(len(unmapped))],
            ],
        ),
        "",
        "## 2. Special / added tokens",
        "",
        f"`added_tokens`: **{len(added_tokens)}** entries, "
        f"{len(special_added)} with `special=true` and "
        f"{len(non_special_added)} with `special=false`.",
        "",
        f"Excluded added ids: **{len(excluded)}** "
        "(all added tokens, special or not).",
        "",
        _table(["id", "special", "content"], added_rows),
        "",
        "## 3. Special token search",
        "",
        _table(
            ["token", "id", "in added_tokens", "special flag", "excluded"],
            search_rows,
        ),
        "",
        "Conclusion: `<|im_start|>`, `<|im_end|>` and `<|endoftext|>` are "
        "added-special tokens and are excluded. `<tool_call>` (151657) and "
        "`</tool_call>` (151658) are added tokens with `special=false`, and so "
        "are `<tool_response>`, `</tool_response>`, ` thinking` and "
        "` response`; like every added token they are **excluded** from the "
        "decoder's allowed set because they are control markers, never JSON "
        "content.",
        "",
        "## 4. Space and newline (`Ġ`, `Ċ`)",
        "",
        "Examples containing `Ġ` (space):",
        "",
        _table(
            ["id", "raw", "decoded"],
            [
                [str(i), _code(repr(raw)), _code(repr(decoded))]
                for i, raw, decoded in space_examples
            ],
        ),
        "",
        "Examples containing `Ċ` (newline):",
        "",
        _table(
            ["id", "raw", "decoded"],
            [
                [str(i), _code(repr(raw)), _code(repr(decoded))]
                for i, raw, decoded in newline_examples
            ],
        ),
        "",
        "## 5. Multi-character tokens crossing JSON boundaries",
        "",
        "Tokens whose raw or decoded text contains any of these characters:",
        "",
        "```text",
        '" } , : \\',
        "```",
        "",
        _table(
            ["id", "raw", "decoded"],
            [
                [str(i), _code(repr(raw)), _code(repr(decoded))]
                for i, raw, decoded in boundary_examples
            ],
        ),
        "",
        "## 6. Raw vs decoded (5 samples)",
        "",
        _table(["id", "raw", "decoded", "note"], comparison_rows),
        "",
        "`U+FFFD` (`\ufffd`) means the per-id decode is lossy. These byte "
        "fragments must **not** be excluded; they are needed to rebuild "
        "multi-byte characters.",
        "",
        "## 7. Exclusion rule and byte-level caveat",
        "",
        "- **Exclude** every id returned by `excluded_token_ids(tokenizer)`: "
        f"all {len(excluded)} added tokens ({len(special_added)} special + "
        f"{len(non_special_added)} non-special control markers such as "
        "`<tool_call>`, `</tool_call>` and ` thinking`). None of them is "
        "normal JSON content, so all are excluded.",
        "- **Never allow** any id returned by "
        "`unmapped_token_ids(tokenizer, logits_size)`: padding positions the "
        "model exposes but the tokenizer cannot decode. There are "
        f"{len(unmapped)} such ids.",
        "- **Keep** every other id, including byte-level fragments whose "
        "per-id `decode([id])` is lossy (`U+FFFD`, e.g. ids 94-98). These "
        "tokens are needed to assemble multi-byte UTF-8 characters; "
        "reconstruct text at the byte level (T4.4/T5.1) instead of excluding "
        "them.",
        "",
        "## 8. Run",
        "",
        "```bash",
        "uv run python -m experiments.vocab_analysis",
        "```",
        "",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {REPORT_PATH}")
    print(
        f"vocab_size={tokenizer.vocab_size} id_range={id_range_size} "
        f"logits={logits_size} unmapped={len(unmapped)} "
        f"added={len(added_tokens)} special={len(special_added)} "
        f"excluded={len(excluded)}"
    )
    print(
        "special search: "
        + ", ".join(
            f"{token}={'found' if reverse.get(token) else 'NOT FOUND'}"
            for token in SPECIAL_SEARCH
        )
    )
    print(
        f"boundary examples={len(boundary_examples)} "
        f"lossy ids={lossy_ids}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
