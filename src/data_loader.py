"""Load and validate the JSON input files used by the pipeline.

This module is the only place allowed to touch ``data/input/*.json``. It turns
every expected input problem (missing file, bad JSON, schema mismatch, empty
input) into :class:`DataLoadError` so that callers never have to handle a raw
``FileNotFoundError``, ``json.JSONDecodeError`` or Pydantic ``ValidationError``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.schemas import FunctionCatalog

# Public loaders accept either a string or a ``pathlib.Path``.
PathLike = str | Path


class DataLoadError(Exception):
    """Raised for any expected problem while loading an input file.

    T1.5 will consolidate the project-wide exception policy; until then this
    loader-local exception keeps input failures distinguishable from real bugs
    in the rest of the code.
    """


def _read_json(path: PathLike) -> Any:
    """Read and parse a UTF-8 JSON file, wrapping I/O and parse errors.

    Returns the decoded JSON value, which may be any JSON type. Every expected
    failure is reported as :class:`DataLoadError` naming the offending file.
    """
    file_path = Path(path)
    try:
        with file_path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise DataLoadError(f"{file_path}: file not found") from exc
    except IsADirectoryError as exc:
        raise DataLoadError(
            f"{file_path}: expected a file but found a directory"
        ) from exc
    except json.JSONDecodeError as exc:
        raise DataLoadError(
            f"{file_path}: invalid JSON at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise DataLoadError(f"{file_path}: could not read file: {exc}") from exc


def _summarize_validation_error(exc: ValidationError, limit: int = 3) -> str:
    """Return a compact, readable summary of a Pydantic validation error."""
    errors = exc.errors()
    parts: list[str] = []
    for error in errors[:limit]:
        location = ".".join(str(item) for item in error["loc"]) or "<root>"
        parts.append(f"{location}: {error['msg']}")
    remaining = len(errors) - limit
    if remaining > 0:
        parts.append(f"... and {remaining} more error(s)")
    return "; ".join(parts)


def load_functions(path: PathLike) -> FunctionCatalog:
    """Load and validate the function catalog from ``path``.

    ``path`` must contain a UTF-8 JSON root list of function definitions; the
    object form with a ``functions`` key is also accepted because
    :class:`FunctionCatalog` handles both. The catalog must declare at least
    one function: an empty list is rejected, since the decoder needs something
    to choose from.

    Raises:
        DataLoadError: if the file is missing, unreadable, not valid JSON,
            fails schema validation, or declares no functions.
    """
    source = Path(path)
    raw = _read_json(source)
    try:
        catalog = FunctionCatalog.model_validate(raw)
    except ValidationError as exc:
        raise DataLoadError(
            f"{source}: invalid function catalog: "
            f"{_summarize_validation_error(exc)}"
        ) from exc

    if not catalog.functions:
        raise DataLoadError(
            f"{source}: function catalog must contain at least one function"
        )
    return catalog


def _extract_prompt(source: Path, index: int, item: Any) -> str:
    """Validate one prompt entry and return its text unchanged."""
    if not isinstance(item, dict):
        raise DataLoadError(
            f"{source}: prompt entry at index {index} must be an object, "
            f"got {type(item).__name__}"
        )
    if "prompt" not in item:
        raise DataLoadError(
            f"{source}: prompt entry at index {index} is missing 'prompt'"
        )
    prompt = item["prompt"]
    if not isinstance(prompt, str):
        raise DataLoadError(
            f"{source}: prompt entry at index {index} must be a string, "
            f"got {type(prompt).__name__}"
        )
    if not prompt.strip():
        raise DataLoadError(
            f"{source}: prompt entry at index {index} must not be blank"
        )
    return prompt


def load_prompts(path: PathLike) -> list[str]:
    """Load the ordered list of prompts from ``path``.

    ``path`` must contain a UTF-8 JSON root list of objects with a string
    ``prompt`` key. The list must be non-empty and every prompt must be
    non-blank; both rules are enforced because a run with nothing to ask (or a
    blank question) is almost always a data mistake.

    Raises:
        DataLoadError: if the file is missing, unreadable, not valid JSON, the
            root is not a list, the list is empty, or an entry is not an object
            with a non-blank string ``prompt``.
    """
    source = Path(path)
    raw = _read_json(source)

    if not isinstance(raw, list):
        raise DataLoadError(
            f"{source}: expected a JSON list of prompts, got {type(raw).__name__}"
        )
    if not raw:
        raise DataLoadError(
            f"{source}: prompt list must contain at least one prompt"
        )

    prompts: list[str] = []
    for index, item in enumerate(raw):
        prompts.append(_extract_prompt(source, index, item))
    return prompts
