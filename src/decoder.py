"""Constrained JSON generation for function calls (T5.1-T5.3).

The decoder only asks the model for tokens while the DFA is inside a
model-filled slot. Fixed template text is injected by the decoder and must never
be emitted by the model, so :func:`allowed_token_ids` returns ``()`` in every
injected state. :func:`masked_logits` applies ``-inf`` outside the allowed set,
:func:`select_next_token` performs the guarded argmax, and :func:`decode_call`
alternates fixed injection with slot generation until ``ACCEPT``.

T8.3 memoizes :func:`allowed_token_ids` per :class:`TokenizerIndex`. The cache
key is the DFA state plus the only buffer that can change the mask: the name
prefix in ``NAME_VALUE``, the number buffer in ``PARAM_VALUE_NUMBER``, the
literal buffer in ``PARAM_VALUE_BOOLEAN`` and the remaining Unicode escape
digits in ``PARAM_VALUE_STRING``. All other states are buffer-independent, so
the expensive string mask is computed once per index and reused across prompts.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike
from pydantic import ValidationError

from src.config import MAX_TOKENS
from src.dfa import DFA, DFAState, is_complete_number
from src.errors import DecodingError
from src.llm_client import LLMClient
from src.prompt import (
    TemplateSegment,
    build_prefix_segments,
    build_prompt,
    build_tail_segments,
)
from src.schemas import FunctionCall, FunctionCatalog
from src.vocab import TokenizerIndex

__all__ = [
    "NUMBER_EOS_BONUS",
    "allowed_token_ids",
    "decode_call",
    "masked_logits",
    "select_next_token",
    "slot_can_stop",
]

# Logit bonus added to EOS once a number is already complete. Greedy decoding
# otherwise keeps extending a valid number (for example "2" -> "2.0000...")
# and never reaches EOS. The bonus only biases the choice; every token,
# including EOS, is still validated by the DFA.
#
# Production calibration against the real Qwen/Qwen3-0.6B greedy model:
# the sum prompt has EOS at -2.09 and the dot token at 19.54, so a raw EOS
# loses by 21.63; in the "265" case the next digit "6" reaches 28.42, so EOS
# must stay below it once EOS is boosted. A +25 bonus puts EOS at 22.91, which
# beats "." (19.54) but not "6" (28.42). Probes: sum -> a=2, b=3; 265+345 ->
# a=265, b=345; sqrt 2 -> a=2. Callers keep the raw default of 0.0; the
# production pipeline opts in by passing this constant explicitly.
NUMBER_EOS_BONUS = 25.0

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


def _allowed_cache_key(dfa: DFA) -> tuple[object, str]:
    """Return the transparency-preserving cache key for the current mask.

    A mask depends on the DFA state and, in a few states, on the value already
    consumed. Every other state ignores the remaining buffers, so it uses an
    empty discriminator and shares one cache entry per state.
    """
    state = dfa.state
    if state is DFAState.NAME_VALUE:
        return state, dfa.name_buffer
    if state is DFAState.PARAM_VALUE_NUMBER:
        return state, dfa.number_buffer
    if state is DFAState.PARAM_VALUE_BOOLEAN:
        return state, dfa.literal_buffer
    if state is DFAState.PARAM_VALUE_STRING and dfa.unicode_remaining:
        return state, f"u{dfa.unicode_remaining}"
    return state, ""


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

    Results are memoized in ``index.allowed_cache`` under the state plus the
    buffer that can change the outcome. The cache is transparent: a hit returns
    exactly the tuple the uncached scan would produce, and the mask is reused
    across prompts because :func:`src.pipeline.run` reuses one index.

    The input machine is never mutated. Empty and special ids are absent from
    ``by_first_char``; missing/empty texts are skipped defensively and are never
    re-opened.
    """
    targets = _SLOT_TARGETS.get(dfa.state)
    if targets is None:
        return ()

    key = _allowed_cache_key(dfa)
    cached = index.allowed_cache.get(key)
    if cached is not None:
        return cached

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

    result = tuple(sorted(allowed))
    index.allowed_cache[key] = result
    return result


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


def _resolve_eos_token_id(
    index: TokenizerIndex,
    eos_token_id: int | None,
) -> int:
    """Return the caller's EOS id or discover the project's preferred marker."""
    if eos_token_id is not None:
        return eos_token_id

    for marker in ("<|im_end|>", "<|endoftext|>"):
        for token_id, text in index.texts.items():
            if text == marker:
                return token_id

    raise DecodingError(
        "cannot resolve EOS token: index contains neither "
        "<|im_end|> nor <|endoftext|>"
    )


def decode_call(
    model: LLMClient,
    functions: FunctionCatalog,
    prompt: str,
    *,
    index: TokenizerIndex,
    eos_token_id: int | None = None,
    max_tokens: int = MAX_TOKENS,
    number_eos_bonus: float = 0.0,
    prompt_builder: Callable[[FunctionCatalog, str], str] = build_prompt,
) -> FunctionCall:
    """Decode one complete function call without asking for fixed text.

    Fixed template segments are injected with ``model.encode`` and never call
    ``model.logits``. Only function-name and parameter-value slots consume model
    steps. A step is counted for every ``model.logits`` call, including a call
    that selects EOS; when the next step would exceed ``max_tokens`` the decoder
    raises :class:`DecodingError` before calling the model again.

    ``number_eos_bonus`` is added to the EOS logit once a number slot already
    holds a complete value. It defaults to ``0.0`` so raw decoding keeps the
    model's scores unchanged; the production pipeline opts in by passing
    :data:`NUMBER_EOS_BONUS`.

    ``prompt_builder`` defaults to :func:`src.prompt.build_prompt`; it exists so
    the T8.4 benchmark can compare prompt formats without a second decoder.

    The returned :class:`FunctionCall` is parsed from the complete generated
    template. Any DFA, budget or schema failure raises an exception instead of
    returning a partial call.
    """
    eos = _resolve_eos_token_id(index, eos_token_id)
    dfa = DFA.start([function.name for function in functions])

    prompt_ids = model.encode(prompt_builder(functions, prompt))
    input_ids: list[int] = list(prompt_ids)
    output_ids: list[int] = []
    steps = 0

    def inject(segment: TemplateSegment) -> None:
        """Append one fixed segment without a logits call."""
        if not dfa.consume_token(segment.text):
            raise DecodingError(
                f"fixed template segment rejected by the DFA: "
                f"{segment.text!r}"
            )
        input_ids.extend(segment.token_ids)
        output_ids.extend(segment.token_ids)

    def next_token(candidates: tuple[int, ...]) -> int:
        """Call the model once and select one candidate within the budget."""
        nonlocal steps
        if steps >= max_tokens:
            raise DecodingError(
                f"MAX_TOKENS exceeded: {steps} model step(s) already used "
                f"(limit {max_tokens})"
            )
        logits = model.logits(input_ids)
        steps += 1
        if (
            number_eos_bonus
            and dfa.state is DFAState.PARAM_VALUE_NUMBER
            and slot_can_stop(dfa)
        ):
            logits = logits.copy()
            logits[eos] = logits[eos] + number_eos_bonus
        return select_next_token(logits, candidates)

    for segment in build_prefix_segments(prompt, model.encode):
        inject(segment)

    while dfa.state is DFAState.NAME_VALUE:
        allowed = allowed_token_ids(dfa, index)
        candidates = allowed + ((eos,) if slot_can_stop(dfa) else ())
        if not candidates:
            raise DecodingError(
                f"no allowed tokens in {dfa.state.name}; cannot choose a name"
            )

        token = next_token(candidates)
        if token == eos:
            break

        if not dfa.consume_token(index.texts[token]):
            raise DecodingError(
                f"selected name token {token} rejected by the DFA"
            )
        input_ids.append(token)
        output_ids.append(token)

    if dfa.names is None or not dfa.names.is_complete(dfa.name_buffer):
        raise DecodingError(
            f"incomplete function name {dfa.name_buffer!r} at EOS"
        )
    name = dfa.name_buffer

    for segment in build_tail_segments(functions, name, model.encode):
        if segment.kind == "fixed":
            inject(segment)
            continue

        while dfa.state is not DFAState.PARAM_VALUE_DONE:
            allowed = allowed_token_ids(dfa, index)
            candidates = allowed + ((eos,) if slot_can_stop(dfa) else ())
            if not candidates:
                raise DecodingError(
                    f"no allowed tokens in {dfa.state.name}; "
                    f"cannot fill parameter {segment.slot!r}"
                )

            token = next_token(candidates)
            if token == eos:
                break

            if not dfa.consume_token(index.texts[token]):
                raise DecodingError(
                    f"selected value token {token} rejected by the DFA"
                )
            input_ids.append(token)
            output_ids.append(token)

        if not (dfa.state is DFAState.PARAM_VALUE_DONE or slot_can_stop(dfa)):
            raise DecodingError(
                f"incomplete value for parameter {segment.slot!r} at EOS"
            )

    if not dfa.is_accepting():
        raise DecodingError(
            f"decoder stopped in {dfa.state.name}; expected ACCEPT"
        )

    decoded = model.decode(output_ids)
    try:
        return FunctionCall.model_validate_json(decoded)
    except ValidationError as exc:
        raise DecodingError(
            f"decoded call is not a valid FunctionCall: {exc}"
        ) from exc
