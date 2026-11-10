"""Vocabulary loading and ``token_text`` construction for the decoder.

Qwen uses **byte-level BPE**: the raw vocabulary string is not the text the
model actually emits. For example ``"Ġhello"`` represents ``" hello"`` and
``"Ċ"`` represents a newline. The constrained decoder must validate tokens
against the DFA using real decoded text, so the raw representation is only kept
as a documented fallback; prefer :func:`build_token_text_via_decode`, which
asks the SDK's public ``decode`` method for the final text.

This module never loads the model at import time. All functions are pure except
the two cache helpers, which only touch the filesystem path they are given.

Exclusion rule: the decoder's allowed set is ``tokenizer.token_text`` minus
:func:`excluded_token_ids`, which covers **every** id declared in
``added_tokens`` (special or not). Non-special added tokens such as
``<tool_call>`` or `` thinking`` are control markers, never JSON content. Ids
exposed by the logits vector but absent from the tokenizer are padding positions
and are reported by :func:`unmapped_token_ids`. Byte-level fragment tokens must
stay allowed even when a per-id ``decode`` is lossy (returns ``U+FFFD``): they
are needed to assemble multi-byte characters.

Lookup strategy: :class:`TokenizerIndex` groups allowed tokens by their first
decoded character, narrowing ~151k ids to a few thousand before the full
``str.startswith`` check. Byte-level fragments stay in the index (keyed on
``U+FFFD`` when the per-id decode is lossy) because the decoder needs them to
assemble UTF-8. The ``(state_id, token_id) -> next_state_id`` transition cache is
deferred: it needs DFA state ids, which do not exist until T4; revisit it in
T8.3 profiling.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.errors import ModelError

__all__ = [
    "PathLike",
    "TokenizerIndex",
    "TokenizerVocab",
    "build_token_index",
    "build_token_text_via_decode",
    "excluded_token_ids",
    "load_token_text_cache",
    "parse_tokenizer_json",
    "parse_vocab_json",
    "save_token_text_cache",
    "special_token_ids",
    "unmapped_token_ids",
    "validate_token_text",
]

# Public helpers accept either a string or a ``pathlib.Path``.
PathLike = str | Path


@dataclass(frozen=True)
class TokenizerVocab:
    """A parsed tokenizer asset.

    Attributes:
        token_text: Raw vocabulary strings keyed by token id (e.g. ``"Ġhello"``).
        special_ids: Ids whose ``added_tokens`` entry has ``special=true``.
        added_ids: Every id declared in ``added_tokens``, special or not.
        source: Which file and format the mapping came from, for diagnostics.
    """

    token_text: dict[int, str]
    special_ids: frozenset[int]
    added_ids: frozenset[int]
    source: str

    @property
    def vocab_size(self) -> int:
        """Number of token entries in the mapping."""
        return len(self.token_text)


@dataclass(frozen=True)
class TokenizerIndex:
    """First-character index over the tokens allowed in decoded JSON text.

    Attributes:
        texts: Id -> token text for every mapped id (including control tokens
            that ``allowed_ids`` then removes).
        allowed_ids: Sorted ids that may be emitted: mapped ids minus every
            added control token.
        by_first_char: First character -> sorted allowed ids. Tokens whose text
            is empty are skipped; lossy byte-level fragments are keyed on
            ``U+FFFD`` because that is the first character of their per-id
            ``decode``.
    """

    texts: dict[int, str]
    allowed_ids: tuple[int, ...]
    by_first_char: dict[str, tuple[int, ...]]

    def tokens_starting_with(self, prefix: str) -> tuple[int, ...]:
        """Return sorted allowed ids whose text starts with ``prefix``.

        An empty prefix returns every allowed id. Otherwise the first-character
        bucket narrows the search before the full ``startswith`` check.
        """
        if prefix == "":
            return self.allowed_ids
        candidates = self.by_first_char.get(prefix[0], ())
        return tuple(
            token_id
            for token_id in candidates
            if self.texts[token_id].startswith(prefix)
        )


def _load_json(path: Path) -> Any:
    """Read a UTF-8 JSON model asset, wrapping every failure as ModelError."""
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ModelError(f"{path}: file not found") from exc
    except IsADirectoryError as exc:
        raise ModelError(
            f"{path}: expected a file but found a directory"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ModelError(
            f"{path}: invalid JSON at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise ModelError(f"{path}: could not read file: {exc}") from exc


def _insert_token(
    token_text: dict[int, str],
    token_id: int,
    text: str,
    path: Path,
) -> None:
    """Insert one id -> text pair, rejecting a conflicting duplicate id."""
    existing = token_text.get(token_id)
    if existing is not None and existing != text:
        raise ModelError(
            f"{path}: conflicting text for token id {token_id}: "
            f"{existing!r} vs {text!r}"
        )
    token_text[token_id] = text


def _parse_plain_vocab(
    path: Path,
    raw: Mapping[Any, Any],
    source: str,
) -> TokenizerVocab:
    """Validate a plain token -> id mapping and wrap it as TokenizerVocab."""
    token_text: dict[int, str] = {}
    for token, raw_id in raw.items():
        if not isinstance(token, str):
            raise ModelError(
                f"{path}: vocabulary key must be a string, "
                f"got {type(token).__name__}"
            )
        if type(raw_id) is not int:
            raise ModelError(
                f"{path}: id for vocabulary token {token!r} must be an integer"
            )
        _insert_token(token_text, raw_id, token, path)
    return TokenizerVocab(
        token_text=token_text,
        special_ids=frozenset(),
        added_ids=frozenset(),
        source=source,
    )


def parse_tokenizer_json(path: PathLike) -> TokenizerVocab:
    """Parse a Hugging Face ``tokenizer.json`` file.

    Reads both ``model.vocab`` (token string -> id) and ``added_tokens`` (a
    list of objects with ``id``, ``content`` and optional ``special``). Every
    added token id is stored in ``added_ids``; ``special_ids`` keeps only the
    entries whose ``special`` flag is true. Text is stored raw; use
    :func:`build_token_text_via_decode` for decoded text.

    Raises:
        ModelError: if the file is missing/unreadable, not valid JSON, has the
            wrong shape, or an added token collides with a different text.
    """
    file_path = Path(path)
    raw = _load_json(file_path)
    if not isinstance(raw, dict):
        raise ModelError(
            f"{file_path}: expected a JSON object, got {type(raw).__name__}"
        )

    model = raw.get("model")
    if not isinstance(model, dict):
        raise ModelError(f"{file_path}: missing or invalid 'model' object")
    vocab = model.get("vocab")
    if not isinstance(vocab, dict):
        raise ModelError(
            f"{file_path}: missing or invalid 'model.vocab' object"
        )

    token_text: dict[int, str] = {}
    for token, raw_id in vocab.items():
        if not isinstance(token, str):
            raise ModelError(
                f"{file_path}: model.vocab key must be a string, "
                f"got {type(token).__name__}"
            )
        if type(raw_id) is not int:
            raise ModelError(
                f"{file_path}: id for vocab token {token!r} must be an integer"
            )
        _insert_token(token_text, raw_id, token, file_path)

    added_tokens = raw.get("added_tokens", [])
    if not isinstance(added_tokens, list):
        raise ModelError(f"{file_path}: 'added_tokens' must be a list")

    added_ids: set[int] = set()
    special_ids: set[int] = set()
    for index, item in enumerate(added_tokens):
        if not isinstance(item, dict):
            raise ModelError(
                f"{file_path}: added_tokens[{index}] must be an object"
            )
        raw_id = item.get("id")
        if type(raw_id) is not int:
            raise ModelError(
                f"{file_path}: added_tokens[{index}] is missing an integer 'id'"
            )
        content = item.get("content")
        if not isinstance(content, str):
            raise ModelError(
                f"{file_path}: added_tokens[{index}] is missing a string "
                f"'content'"
            )
        special = item.get("special", False)
        if not isinstance(special, bool):
            raise ModelError(
                f"{file_path}: added_tokens[{index}].special must be a boolean"
            )
        _insert_token(token_text, raw_id, content, file_path)
        added_ids.add(raw_id)
        if special:
            special_ids.add(raw_id)

    return TokenizerVocab(
        token_text=token_text,
        special_ids=frozenset(special_ids),
        added_ids=frozenset(added_ids),
        source=f"{file_path}:tokenizer.json",
    )


def parse_vocab_json(path: PathLike) -> TokenizerVocab:
    """Parse a plain ``vocab.json`` (token -> id) as a documented fallback.

    This older format carries no ``added_tokens`` information, so
    ``special_ids`` and ``added_ids`` are always empty and the stored text is
    the raw BPE string. Prefer :func:`parse_tokenizer_json` when a
    ``tokenizer.json`` is available.

    Raises:
        ModelError: if the file is missing/unreadable, not valid JSON, or its
            shape is not a string -> integer mapping.
    """
    file_path = Path(path)
    raw = _load_json(file_path)
    if not isinstance(raw, dict):
        raise ModelError(
            f"{file_path}: expected a JSON object, got {type(raw).__name__}"
        )
    return _parse_plain_vocab(
        file_path,
        raw,
        source=f"{file_path}:vocab.json",
    )


def build_token_text_via_decode(
    model: Any,
    ids: Sequence[int],
) -> dict[int, str]:
    """Decode every id once through the SDK's public ``decode`` method.

    ``model`` is duck-typed: any object with a ``decode(list[int]) -> str``
    method works, so tests can pass a fake without importing the SDK. The
    result is the real text the decoder should validate against the DFA.

    Raises:
        ModelError: if ``decode`` fails for an id.
    """
    token_text: dict[int, str] = {}
    for token_id in ids:
        try:
            token_text[token_id] = model.decode([token_id])
        except Exception as exc:
            raise ModelError(
                f"failed to decode token id {token_id}: {exc}"
            ) from exc
    return token_text


def validate_token_text(
    token_text: Mapping[int, str],
    expected_size: int,
) -> None:
    """Check that ``token_text`` covers every id in ``range(expected_size)``.

    Empty strings are allowed (some tokens legitimately decode to nothing); an
    empty *mapping* is not. Duplicate keys cannot occur in a mapping, but the
    count is checked defensively.

    Raises:
        ModelError: if the mapping is empty, contains ids outside the expected
            range, or is missing ids.
    """
    if not token_text:
        raise ModelError(
            f"token_text is empty; expected ids in range({expected_size})"
        )

    ids = set(token_text)
    if len(token_text) != len(ids):
        raise ModelError("token_text contains duplicate ids")

    out_of_range = sorted(
        token_id for token_id in ids if not 0 <= token_id < expected_size
    )
    if out_of_range:
        preview = ", ".join(str(token_id) for token_id in out_of_range[:5])
        suffix = "..." if len(out_of_range) > 5 else ""
        raise ModelError(
            f"{len(out_of_range)} token id(s) outside "
            f"range({expected_size}): {preview}{suffix}"
        )

    missing = sorted(set(range(expected_size)) - ids)
    if missing:
        preview = ", ".join(str(token_id) for token_id in missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ModelError(
            f"token_text is missing {len(missing)} id(s) in "
            f"range({expected_size}): {preview}{suffix}"
        )


def save_token_text_cache(
    path: PathLike,
    token_text: Mapping[int, str],
) -> None:
    """Write ``token_text`` as a JSON object mapping ``str(id)`` -> text."""
    file_path = Path(path)
    payload = {str(token_id): text for token_id, text in token_text.items()}
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise ModelError(
            f"{file_path}: could not write token_text cache: {exc}"
        ) from exc


def load_token_text_cache(
    path: PathLike,
    expected_size: int | None = None,
) -> dict[int, str]:
    """Load a cache written by :func:`save_token_text_cache`.

    Callers that know the vocabulary size should pass ``expected_size`` so
    :func:`validate_token_text` also checks full coverage and rejects
    out-of-range ids. When omitted, only key format (non-negative integer
    strings) is checked.

    Raises:
        ModelError: if the file is missing/unreadable, not valid JSON, is not
            an object, a key/value does not have the expected shape, or
            ``expected_size`` is given and the mapping does not cover
            ``range(expected_size)`` exactly.
    """
    file_path = Path(path)
    raw = _load_json(file_path)
    if not isinstance(raw, dict):
        raise ModelError(
            f"{file_path}: cache must be a JSON object, "
            f"got {type(raw).__name__}"
        )

    token_text: dict[int, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not (key.isascii() and key.isdigit()):
            raise ModelError(
                f"{file_path}: cache key {key!r} is not a non-negative "
                f"integer string"
            )
        if not isinstance(value, str):
            raise ModelError(
                f"{file_path}: cache value for id {key} must be a string, "
                f"got {type(value).__name__}"
            )
        token_text[int(key)] = value

    if expected_size is not None:
        validate_token_text(token_text, expected_size)
    return token_text


def special_token_ids(tokenizer: TokenizerVocab) -> frozenset[int]:
    """Return the ids whose ``added_tokens`` entry has ``special=true``."""
    return tokenizer.special_ids


def excluded_token_ids(tokenizer: TokenizerVocab) -> frozenset[int]:
    """Return ids excluded from the decoder's allowed token set.

    Exclusion rule: every tokenizer-added control token must be excluded, so
    this returns ``tokenizer.added_ids`` (all ``added_tokens`` entries, 26 for
    Qwen3-0.6B). ``special=false`` markers such as ``<tool_call>`` or
    `` thinking`` are still control tokens, never JSON content, and are
    therefore excluded too.

    Byte-level caveat: *only* added tokens are excluded. Byte-level fragment
    tokens (for example ids 94-98, whose per-id ``decode`` is lossy and yields
    ``U+FFFD``) stay allowed; the decoder needs them to assemble multi-byte
    characters.
    """
    return tokenizer.added_ids


def build_token_index(
    tokenizer: TokenizerVocab,
    texts: Mapping[int, str] | None = None,
) -> TokenizerIndex:
    """Build the first-character index for the decoder's allowed token set.

    ``texts`` defaults to ``tokenizer.token_text`` (raw BPE strings). Pass the
    decoded ``token_text`` produced by :func:`build_token_text_via_decode` when
    indexing real model output, so ``"Ġ"`` entries become spaces and lossy
    byte-level fragments key on ``U+FFFD``. Only ids present in ``texts`` are
    indexed; every added control token is removed from ``allowed_ids``.

    Complexity is O(vocab) to build. The strategy is to narrow ~151k ids by
    first character before running the string prefix check, keeping byte-level
    fragments available for UTF-8 assembly. The DFA transition cache is
    deferred until state ids exist (T4/T8.3).
    """
    selected = tokenizer.token_text if texts is None else texts
    merged = dict(selected)
    # Added tokens are excluded from the mask, but decoding them yields an
    # empty string. Keep their raw content so EOS discovery still works with a
    # decoded index.
    for added_id in tokenizer.added_ids:
        raw_text = tokenizer.token_text.get(added_id)
        if raw_text:
            merged[added_id] = raw_text
    index_texts = {
        token_id: text
        for token_id, text in merged.items()
        if token_id in tokenizer.token_text
    }
    excluded = excluded_token_ids(tokenizer)
    allowed_ids = tuple(
        token_id
        for token_id in sorted(index_texts)
        if token_id not in excluded
    )

    buckets: dict[str, list[int]] = {}
    for token_id in allowed_ids:
        text = index_texts[token_id]
        if not text:
            continue
        buckets.setdefault(text[0], []).append(token_id)

    by_first_char = {
        char: tuple(token_ids) for char, token_ids in buckets.items()
    }
    return TokenizerIndex(
        texts=index_texts,
        allowed_ids=allowed_ids,
        by_first_char=by_first_char,
    )


def unmapped_token_ids(
    tokenizer: TokenizerVocab,
    logits_size: int,
) -> frozenset[int]:
    """Return ids exposed by the logits vector but absent from the tokenizer.

    These are padding positions: the model can emit them, but ``tokenizer.json``
    has no text for them, so the constrained decoder must never select them.

    Raises:
        ModelError: if ``logits_size`` is not positive.
    """
    if logits_size <= 0:
        raise ModelError(f"logits_size must be positive, got {logits_size}")
    known = tokenizer.token_text
    return frozenset(
        token_id
        for token_id in range(logits_size)
        if token_id not in known
    )
