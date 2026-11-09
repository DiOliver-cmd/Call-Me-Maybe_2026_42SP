"""Token and wall-clock budget for the decoding pipeline (T3.4).

The subject requires the whole run (all prompts) to finish in under 5 minutes.
On the current CPU that target cannot be met with the 6-function prompt: the
T3.3 baseline measured 2.712 s/token overall, so 17 prompts x 48 tokens is
about 2213.0 s (36.9 min). The shortfall is documented in
``experiments/benchmark.md`` and is deferred to T8.4, where mitigation options
(shorter prompt, fewer tokens, faster hardware, batching) will be validated.

Abort behavior: :func:`ensure_within_deadline` raises
:class:`~src.errors.BudgetExceededError` (a ``DecodingError``, hence a
``ProjectError``) when the budget is exceeded. The pipeline writes its output
only after all calls complete (T6.3), so an abort never leaves a partial output
file. T5.3 will call these helpers from the decoding loop; this module does not
implement the loop itself.
"""

from __future__ import annotations

from src.errors import BudgetExceededError

__all__ = [
    "MAX_TOKENS",
    "MEASURED_SECONDS_PER_TOKEN",
    "PROMPT_TIMEOUT_SECONDS",
    "TOTAL_TIMEOUT_SECONDS",
    "ensure_within_deadline",
    "estimate_seconds",
    "exceeds_time_target",
]

# A compliant JSON answer costs about 20 tokens. The T3.3 baseline used 48 and
# only rambling outputs were truncated, so 48 keeps roughly 2.4x headroom.
MAX_TOKENS = 48

# Subject target: the whole run (all prompts) must finish in under 5 minutes.
TOTAL_TIMEOUT_SECONDS = 300

# Safety valve for one prompt. The T3.3 baseline averaged 130.16 s per prompt.
PROMPT_TIMEOUT_SECONDS = 180

# Derived from the T3.3 baseline: 1301.576 s / 480 generated tokens = 2.7116,
# rounded to 2.712 s/token. It includes context growth and model initialization.
MEASURED_SECONDS_PER_TOKEN = 2.712


def estimate_seconds(
    n_prompts: int,
    max_tokens: int = MAX_TOKENS,
    seconds_per_token: float = MEASURED_SECONDS_PER_TOKEN,
) -> float:
    """Estimate total generation time for ``n_prompts`` prompts.

    The estimate assumes every prompt generates ``max_tokens`` tokens at a
    constant ``seconds_per_token``, so it is a lower bound in practice because
    the per-step cost grows with context length.

    Raises:
        ValueError: if any argument is not positive.
    """
    if n_prompts <= 0:
        raise ValueError(f"n_prompts must be positive, got {n_prompts}")
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if seconds_per_token <= 0:
        raise ValueError(
            f"seconds_per_token must be positive, got {seconds_per_token}"
        )
    return n_prompts * max_tokens * seconds_per_token


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
    It is pure and side-effect free, so T5.3 can call it before every step or
    at the end of a prompt without touching the model.
    """
    if elapsed_seconds > limit_seconds:
        raise BudgetExceededError(
            f"time budget exceeded: {elapsed_seconds:.1f}s > "
            f"{limit_seconds:.1f}s"
        )
