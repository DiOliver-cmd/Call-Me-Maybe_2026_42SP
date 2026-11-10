"""Slot-scoped constrained-decoding token masks (T5.1).

The decoder only asks for tokens while the DFA is inside a model-filled slot.
Fixed template text is injected by the decoder and must never be emitted by the
model, so :func:`allowed_token_ids` returns ``()`` in every injected state.

No transition cache is used here. A cache key would need the whole DFA state,
including ``name_buffer``, ``number_buffer``, ``literal_buffer`` and
``unicode_remaining``, not just the enum member. That design, together with
profiling-driven cache sizing, is deferred to T8.3.
"""

from __future__ import annotations

import copy

from src.dfa import DFA, DFAState, is_complete_number
from src.vocab import TokenizerIndex

__all__ = ["allowed_token_ids", "slot_can_stop"]

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
