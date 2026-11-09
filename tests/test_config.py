"""Unit tests for the token/time budget and abort behavior (T3.4)."""

from __future__ import annotations

import pytest

from src.config import (
    MAX_TOKENS,
    MEASURED_SECONDS_PER_TOKEN,
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
    assert MEASURED_SECONDS_PER_TOKEN > 0


def test_estimate_seconds_exact_math() -> None:
    assert estimate_seconds(17) == pytest.approx(2212.992)
    assert estimate_seconds(1) == pytest.approx(130.176)
    assert estimate_seconds(10) == pytest.approx(1301.76)
    assert estimate_seconds(2, max_tokens=3, seconds_per_token=0.5) == (
        pytest.approx(3.0)
    )


def test_exceeds_time_target_for_seventeen_prompts() -> None:
    assert exceeds_time_target(17) is True
    assert exceeds_time_target(1) is False


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


@pytest.mark.parametrize("seconds_per_token", [0, -1.0])
def test_non_positive_seconds_per_token_raises(
    seconds_per_token: float,
) -> None:
    with pytest.raises(ValueError):
        estimate_seconds(1, seconds_per_token=seconds_per_token)
