"""Project-wide error hierarchy and CLI exit-code policy.

Policy
------
* Modules raise the most specific :class:`ProjectError` subclass for an expected
  failure and never swallow exceptions with ``except Exception: pass``.
* Third-party exceptions are wrapped at the module boundary with
  ``raise ... from exc`` so the original cause is preserved.
* Only the CLI boundary (``src/__main__.py``, T7.1) catches
  :class:`ProjectError` and :class:`KeyboardInterrupt`, prints
  :func:`format_error` to stderr and exits with :func:`exit_code_for`.

Error categories
----------------
* :class:`InputError`: missing, unreadable or malformed input, or empty input.
* :class:`SchemaError`: data does not match the declared contract.
* :class:`DecodingError`: constrained decoding failure.
* :class:`ModelError`: model or SDK failure.
"""

from __future__ import annotations

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130


class ProjectError(Exception):
    """Base class for every expected project failure."""

    exit_code: int = EXIT_ERROR


class InputError(ProjectError):
    """Missing, unreadable or malformed input, or empty input."""


class SchemaError(ProjectError):
    """Data does not match the declared contract."""


class DecodingError(ProjectError):
    """Constrained decoding failed to produce a valid output."""


class ModelError(ProjectError):
    """The model or the SDK failed."""


def exit_code_for(error: BaseException) -> int:
    """Map ``error`` to the process exit code the CLI should use.

    Only :class:`KeyboardInterrupt` maps to :data:`EXIT_INTERRUPTED`; every
    :class:`ProjectError` uses its own ``exit_code``. Any other exception maps
    to :data:`EXIT_ERROR`: unexpected failures are still reported with exit
    code 1 and must never be silently swallowed.
    """
    if isinstance(error, KeyboardInterrupt):
        return EXIT_INTERRUPTED
    if isinstance(error, ProjectError):
        return error.exit_code
    return EXIT_ERROR


def format_error(error: BaseException) -> str:
    """Format ``error`` for the CLI boundary to print to stderr."""
    return f"error: {error}"
