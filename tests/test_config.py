"""Unit tests for the token/time budget and abort behavior (T3.4/T8.4)."""

from __future__ import annotations

import pytest

from src.config import (
    MAX_TOKENS,
    MEASURED_SECONDS_PER_STEP,
    MEASURED_SETUP_SECONDS,
    MEASURED_STEPS_PER_PROMPT,
    PROMPT_TIMEOUT_SECONDS,
    TOTAL_TIMEOUT_SECONDS,
    ensure_within_deadline,
    estimate_seconds,
    exceeds_time_target,
)
from src.errors import BudgetExceededError, DecodingError, ProjectError


def test_constants_are_positive_and_max_tokens_is_48() -> None:
    assert MAX_TOKENS == 48
    assert MAX_TOKENS > 0
    assert TOTAL_TIMEOUT_SECONDS > 0
    assert PROMPT_TIMEOUT_SECONDS > 0
    assert MEASURED_SECONDS_PER_STEP > 0
    assert MEASURED_STEPS_PER_PROMPT > 0
    assert MEASURED_SETUP_SECONDS >= 0


def test_measured_budget_fits_the_default_run() -> None:
    """The T8.4 measurement keeps the 17-prompt run inside 300 s."""
    measured_wall = MEASURED_SETUP_SECONDS + 17 * 13.82 * 0.729

    assert measured_wall < TOTAL_TIMEOUT_SECONDS
    assert estimate_seconds(17) == pytest.approx(measured_wall)


def test_estimate_seconds_exact_math() -> None:
    setup = MEASURED_SETUP_SECONDS
    per_prompt = MEASURED_STEPS_PER_PROMPT * MEASURED_SECONDS_PER_STEP

    assert estimate_seconds(17) == pytest.approx(setup + 17 * per_prompt)
    assert estimate_seconds(1) == pytest.approx(setup + per_prompt)
    assert estimate_seconds(2, max_tokens=3, seconds_per_step=0.5) == (
        pytest.approx(setup + 3.0)
    )


def test_max_tokens_caps_the_steps_per_prompt() -> None:
    capped = estimate_seconds(1, max_tokens=2, steps_per_prompt=10.0)
    uncapped = estimate_seconds(1, max_tokens=10, steps_per_prompt=10.0)

    assert capped < uncapped


def test_exceeds_time_target_matches_the_measurement() -> None:
    assert exceeds_time_target(17) is False
    assert exceeds_time_target(1) is False
    assert exceeds_time_target(40) is True


def test_ensure_within_deadline_is_noop_at_299_seconds() -> None:
    ensure_within_deadline(299.0)


def test_ensure_within_deadline_raises_after_limit() -> None:
    with pytest.raises(BudgetExceededError) as excinfo:
        ensure_within_deadline(300.001)

    message = str(excinfo.value).lower()
    assert "budget" in message
    assert "300" in message


def test_budget_exceeded_error_is_decoding_and_project_error() -> None:
    assert issubclass(BudgetExceededError, DecodingError)
    assert issubclass(BudgetExceededError, ProjectError)
    assert BudgetExceededError.exit_code == 1


@pytest.mark.parametrize(
    ("n_prompts", "max_tokens"),
    [(0, 48), (-1, 48), (17, 0), (17, -1)],
)
def test_non_positive_counts_raise_value_error(
    n_prompts: int,
    max_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        estimate_seconds(n_prompts, max_tokens)


@pytest.mark.parametrize("seconds_per_step", [0, -1.0])
def test_non_positive_seconds_per_step_raises(
    seconds_per_step: float,
) -> None:
    with pytest.raises(ValueError):
        estimate_seconds(1, seconds_per_step=seconds_per_step)


@pytest.mark.parametrize("steps_per_prompt", [0, -1.0])
def test_non_positive_steps_per_prompt_raises(
    steps_per_prompt: float,
) -> None:
    with pytest.raises(ValueError):
        estimate_seconds(1, steps_per_prompt=steps_per_prompt)


def test_negative_setup_seconds_raises() -> None:
    with pytest.raises(ValueError):
        estimate_seconds(1, setup_seconds=-0.1)
