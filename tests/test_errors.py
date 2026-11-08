"""Unit tests for the project-wide error hierarchy and policy (T1.5)."""

from __future__ import annotations

import pytest

from src.errors import (
    EXIT_ERROR,
    EXIT_INTERRUPTED,
    EXIT_OK,
    DecodingError,
    InputError,
    ModelError,
    ProjectError,
    SchemaError,
    exit_code_for,
    format_error,
)

PROJECT_ERROR_TYPES: list[type[ProjectError]] = [
    InputError,
    SchemaError,
    DecodingError,
    ModelError,
]


def test_project_error_is_an_exception() -> None:
    assert issubclass(ProjectError, Exception)


@pytest.mark.parametrize("error_type", PROJECT_ERROR_TYPES)
def test_subclasses_inherit_from_project_error(
    error_type: type[ProjectError],
) -> None:
    assert issubclass(error_type, ProjectError)


def test_exit_code_constants() -> None:
    assert EXIT_OK == 0
    assert EXIT_ERROR == 1
    assert EXIT_INTERRUPTED == 130


@pytest.mark.parametrize("error_type", PROJECT_ERROR_TYPES)
def test_project_errors_map_to_error_exit_code(
    error_type: type[ProjectError],
) -> None:
    assert error_type.exit_code == EXIT_ERROR
    assert exit_code_for(error_type("boom")) == EXIT_ERROR


def test_keyboard_interrupt_maps_to_interrupted() -> None:
    assert exit_code_for(KeyboardInterrupt()) == EXIT_INTERRUPTED


def test_unexpected_exception_maps_to_error_code() -> None:
    assert exit_code_for(ValueError("boom")) == EXIT_ERROR


def test_format_error_prefixes_message() -> None:
    assert format_error(InputError("bad input")) == "error: bad input"
    assert format_error(ValueError("unexpected")) == "error: unexpected"


@pytest.mark.parametrize("error_type", PROJECT_ERROR_TYPES)
def test_messages_are_preserved_through_hierarchy(
    error_type: type[ProjectError],
) -> None:
    assert str(error_type("payload is invalid")) == "payload is invalid"


@pytest.mark.parametrize("error_type", PROJECT_ERROR_TYPES)
def test_catching_project_error_catches_subclasses(
    error_type: type[ProjectError],
) -> None:
    with pytest.raises(ProjectError):
        raise error_type("boom")
