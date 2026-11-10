"""Token and wall-clock budget for the decoding pipeline (T3.4/T8.4).

The subject requires the whole run (all prompts) to finish in under 5 minutes.
T3.4 estimated the cost from the verbose-prompt baseline (2.712 s/token, so
17 x 48 tokens was ~36.9 min) and marked the target as at risk. T8.4 measured
the real pipeline and closed the gap by shrinking the prompt, because every
model step re-reads the whole context and the public SDK exposes no KV cache.

Measured on the development machine (CPU, 20 cores, Python 3.13) with the
compact prompt of :func:`src.prompt.build_prompt`:

* 17 prompts, 235 model steps (13.82 per prompt), 183.5 s of decoding;
* 8.2 s to load the cached model and 0.5 s to build the decoder index;
* wall clock **192.2 s** (3m12s), against a 300 s target.

A first run on a fresh machine also downloads the model (~40 s), which still
fits: 192.2 + 40 = 232 s. The measurement is in ``experiments/benchmark.md``.

Abort behavior: :func:`ensure_within_deadline` raises
:class:`~src.errors.BudgetExceededError` (a ``DecodingError``, hence a
``ProjectError``) when the budget is exceeded. The pipeline writes its output
only after all calls complete (T6.3), so an abort never leaves a partial output
file. This module only provides the helpers; the decoding loop does not call
them yet.
"""

from __future__ import annotations

from src.errors import BudgetExceededError

__all__ = [
    "MAX_TOKENS",
    "MEASURED_SECONDS_PER_STEP",
    "MEASURED_SETUP_SECONDS",
    "MEASURED_STEPS_PER_PROMPT",
    "PROMPT_TIMEOUT_SECONDS",
    "TOTAL_TIMEOUT_SECONDS",
    "ensure_within_deadline",
    "estimate_seconds",
    "exceeds_time_target",
]

# A compliant JSON answer costs about 20 tokens. The T3.3 baseline used 48 and
# only rambling outputs were truncated, so 48 keeps roughly 2.4x headroom.
# T8.4 kept it: with the compact prompt the budget is met without lowering it.
MAX_TOKENS = 48

# Subject target: the whole run (all prompts) must finish in under 5 minutes.
TOTAL_TIMEOUT_SECONDS = 300

# Safety valve for one prompt. The T3.3 verbose baseline averaged 130.16 s per
# prompt; the compact prompt now averages 10.8 s, so 180 s is a wide margin.
PROMPT_TIMEOUT_SECONDS = 180

# T8.4 measurement of the compact prompt (see experiments/benchmark.md):
# 235 model steps over 17 prompts and 171.4 s of forward passes, i.e.
# 0.729 s per step at ~140 prompt tokens (the verbose prompt cost 1.818 s per
# step at ~390 tokens).
MEASURED_SECONDS_PER_STEP = 0.729
MEASURED_STEPS_PER_PROMPT = 13.82

# One-time setup before the prompt loop: model load (cached weights, no
# download) plus decoder index build (token-text cache present).
MEASURED_SETUP_SECONDS = 8.751


def estimate_seconds(
    n_prompts: int,
    max_tokens: int = MAX_TOKENS,
    seconds_per_step: float = MEASURED_SECONDS_PER_STEP,
    steps_per_prompt: float = MEASURED_STEPS_PER_PROMPT,
    setup_seconds: float = MEASURED_SETUP_SECONDS,
) -> float:
    """Estimate total wall-clock time for ``n_prompts`` prompts.

    The estimate uses the measured constant cost per model step and the
    measured number of steps per prompt, capped by ``max_tokens`` because a
    prompt can never exceed the decoder's per-prompt limit. It is a projection
    from real measurements, not an upper bound: a prompt with a long string
    value can need more steps than the average.

    Raises:
        ValueError: if ``n_prompts``, ``max_tokens``, ``seconds_per_step`` or
            ``steps_per_prompt`` is not positive, or ``setup_seconds`` is
            negative.
    """
    if n_prompts <= 0:
        raise ValueError(f"n_prompts must be positive, got {n_prompts}")
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if seconds_per_step <= 0:
        raise ValueError(
            f"seconds_per_step must be positive, got {seconds_per_step}"
        )
    if steps_per_prompt <= 0:
        raise ValueError(
            f"steps_per_prompt must be positive, got {steps_per_prompt}"
        )
    if setup_seconds < 0:
        raise ValueError(
            f"setup_seconds must not be negative, got {setup_seconds}"
        )
    steps = min(steps_per_prompt, float(max_tokens))
    return setup_seconds + n_prompts * steps * seconds_per_step


def exceeds_time_target(
    n_prompts: int,
    max_tokens: int = MAX_TOKENS,
) -> bool:
    """Return whether the estimate for ``n_prompts`` exceeds 300 seconds."""
    return (
        estimate_seconds(n_prompts, max_tokens) > TOTAL_TIMEOUT_SECONDS
    )


def ensure_within_deadline(
    elapsed_seconds: float,
    limit_seconds: float = TOTAL_TIMEOUT_SECONDS,
) -> None:
    """Raise :class:`BudgetExceededError` if ``elapsed_seconds`` is over.

    The check is strict: equality with ``limit_seconds`` is still allowed.
    It is pure and side-effect free, so the decoding loop can call it before
    every step or at the end of a prompt without touching the model.
    """
    if elapsed_seconds > limit_seconds:
        raise BudgetExceededError(
            f"time budget exceeded: {elapsed_seconds:.1f}s > "
            f"{limit_seconds:.1f}s"
        )
