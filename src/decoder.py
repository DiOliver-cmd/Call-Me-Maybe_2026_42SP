"""Slot-scoped constrained-decoding masks and logits selection (T5.1-T5.2).

The decoder only asks for tokens while the DFA is inside a model-filled slot.
Fixed template text is injected by the decoder and must never be emitted by the
model, so :func:`allowed_token_ids` returns ``()`` in every injected state.
:func:`masked_logits` applies ``-inf`` outside the allowed set and
:func:`select_next_token` performs the guarded argmax; the generation loop and
EOS handling remain in T5.3.

No transition cache is used here. A cache key would need the whole DFA state,
including ``name_buffer``, ``number_buffer``, ``literal_buffer`` and
``unicode_remaining``, not just the enum member. That design, together with
profiling-driven cache sizing, is deferred to T8.3.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike

from src.dfa import DFA, DFAState, is_complete_number
from src.errors import DecodingError
from src.vocab import TokenizerIndex

__all__ = [
    "allowed_token_ids",
    "masked_logits",
    "select_next_token",
    "slot_can_stop",
]

# States reachable while the model is still allowed to generate inside one slot.
# The closing name quote, comma/brace delimiters and all key/structural text are
# fixed and injected by T4.5 segments, so they are deliberately absent.
_SLOT_TARGETS: dict[DFAState, frozenset[DFAState]] = {
    DFAState.NAME_VALUE: frozenset({DFAState.NAME_VALUE}),
    DFAState.PARAM_VALUE_START: frozenset(
        {
            DFAState.PARAM_VALUE_START,
            DFAState.PARAM_VALUE_STRING,
            DFAState.PARAM_VALUE_STRING_ESCAPE,
            DFAState.PARAM_VALUE_NUMBER,
            DFAState.PARAM_VALUE_BOOLEAN,
            DFAState.PARAM_VALUE_DONE,
        }
    ),
    DFAState.PARAM_VALUE_STRING: frozenset(
        {
            DFAState.PARAM_VALUE_STRING,
            DFAState.PARAM_VALUE_STRING_ESCAPE,
            DFAState.PARAM_VALUE_DONE,
        }
    ),
    DFAState.PARAM_VALUE_STRING_ESCAPE: frozenset(
        {
            DFAState.PARAM_VALUE_STRING,
            DFAState.PARAM_VALUE_STRING_ESCAPE,
            DFAState.PARAM_VALUE_DONE,
        }
    ),
    DFAState.PARAM_VALUE_NUMBER: frozenset({DFAState.PARAM_VALUE_NUMBER}),
    DFAState.PARAM_VALUE_BOOLEAN: frozenset({DFAState.PARAM_VALUE_BOOLEAN}),
}

# Multi-character tokens containing any of these characters must not cross a
# fixed-text boundary and then return to an allowed state.
_PATH_STRUCTURAL_CHARS: frozenset[str] = frozenset('"{}:,')


def _path_stays_in_targets(
    dfa: DFA,
    text: str,
    targets: frozenset[DFAState],
) -> bool:
    """Return whether every character stays inside a slot target state."""
    candidate = copy.copy(dfa)
    for char in text:
        if not candidate.consume(char):
            return False
        if candidate.state not in targets:
            return False
    return True


def allowed_token_ids(dfa: DFA, index: TokenizerIndex) -> tuple[int, ...]:
    """Return sorted ids that keep the machine inside the current slot.

    If ``dfa.state`` is not a model-filled slot, the decoder must inject the
    next fixed segment and this function returns ``()``. That includes
    ``START``, the injected prompt/key states, ``PARAM_VALUE_DONE`` and
    ``ACCEPT``.

    A bucket is skipped when its first character is rejected. One-character
    texts reuse that probe; longer texts are validated atomically. A longer text
    containing ``"{}:,`` must additionally stay inside ``targets`` after every
    character, so it cannot consume fixed text and end in an allowed state.

    The input machine is never mutated. Empty and special ids are absent from
    ``by_first_char``; missing/empty texts are skipped defensively and are never
    re-opened.
    """
    targets = _SLOT_TARGETS.get(dfa.state)
    if targets is None:
        return ()

    allowed: set[int] = set()

    for first_char, token_ids in index.by_first_char.items():
        probe = dfa.try_consume_token(first_char)
        if probe is None:
            continue

        for token_id in token_ids:
            text = index.texts.get(token_id)
            if not text:
                continue

            advanced: DFA | None
            if len(text) == 1:
                advanced = probe
            else:
                advanced = dfa.try_consume_token(text)
                if advanced is None:
                    continue
                if _PATH_STRUCTURAL_CHARS.intersection(text):
                    if not _path_stays_in_targets(dfa, text, targets):
                        continue

            if advanced is not None and advanced.state in targets:
                allowed.add(token_id)

    return tuple(sorted(allowed))


def masked_logits(
    logits: ArrayLike,
    allowed_ids: Iterable[int],
) -> np.ndarray:
    """Return float64 scores with only ``allowed_ids`` left selectable.

    Every non-finite input score (NaN, ``+inf`` or ``-inf``) is sanitized to
    ``-np.inf`` so it can never be selected. The output is always a new
    all-``-inf`` array of the same length; allowed finite positions copy their
    sanitized scores. The input sequence is never mutated.

    Raises:
        ValueError: if ``logits`` is not one-dimensional or an allowed id is
            not an ``int`` in ``[0, len(logits))``.
    """
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(
            f"logits must be one-dimensional, got shape {scores.shape}"
        )

    sanitized = scores.copy()
    sanitized[~np.isfinite(sanitized)] = -np.inf

    masked = np.full(sanitized.shape, -np.inf, dtype=np.float64)
    size = sanitized.shape[0]
    for token_id in allowed_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(
                f"allowed token id {token_id!r} is not an int "
                f"(logits size {size})"
            )
        if token_id < 0 or token_id >= size:
            raise ValueError(
                f"allowed token id {token_id} outside [0, {size})"
            )
        masked[token_id] = sanitized[token_id]
    return masked


def select_next_token(
    logits: ArrayLike,
    allowed_ids: Iterable[int],
) -> int:
    """Select the highest-scoring allowed token id with a guarded argmax.

    ``allowed_ids`` may be any iterable, including a caller-provided EOS id; this
    function does not add EOS itself. Ties resolve to the lowest id because
    ``np.argmax`` returns the first maximum.

    Raises:
        DecodingError: if no allowed score is finite after masking.
        ValueError: if masking rejects the logits shape or an allowed id.
    """
    ids = tuple(allowed_ids)
    masked = masked_logits(logits, ids)
    if not np.isfinite(masked).any():
        raise DecodingError(
            f"no valid token: {len(ids)} allowed id(s) but all masked "
            f"scores are -inf (logits size {masked.size})"
        )

    token_id = int(np.argmax(masked))
    assert np.isfinite(masked[token_id])
    return token_id


def slot_can_stop(dfa: DFA) -> bool:
    """Return whether the current slot has a complete value.

    String values become complete when their closing quote moves the machine to
    ``PARAM_VALUE_DONE``; a finished parameter object is also complete. For a
    name, number or boolean, T5.3 adds EOS when this is true so the decoder can
    inject the fixed closing quote or delimiter.
    """
    if dfa.is_accepting() or dfa.state is DFAState.PARAM_VALUE_DONE:
        return True

    if dfa.state is DFAState.NAME_VALUE:
        return dfa.names is not None and dfa.names.is_complete(dfa.name_buffer)

    if dfa.state is DFAState.PARAM_VALUE_NUMBER:
        return is_complete_number(dfa.number_buffer)

    if dfa.state is DFAState.PARAM_VALUE_BOOLEAN:
        return dfa.literal_buffer in {"true", "false"}

    return False
