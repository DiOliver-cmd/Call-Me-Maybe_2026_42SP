"""Input-side Pydantic models for the dynamic function schema.

The function catalog is data-driven: function names, descriptions, parameter
names and return types all come from ``data/input/functions_definition.json``
and must never be hardcoded. These models only validate the *input* contract;
they do not describe model output (see the output models for that).

The constrained decoder (DFA) only supports four parameter types: ``string``,
``number``, ``boolean`` and ``integer``. Any other type (for example ``array``
or ``object``) is rejected with a clear error at validation time, because the
grammar cannot represent it and silently accepting it would let an
unsupported function slip into the catalog.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

KNOWN_PARAMETER_TYPES: frozenset[str] = frozenset(
    {"string", "number", "boolean", "integer"}
)

_SUPPORTED_TYPES = ", ".join(sorted(KNOWN_PARAMETER_TYPES))


class ParameterSpec(BaseModel):
    """A single function parameter (or return value) declaration."""

    model_config = ConfigDict(extra="forbid")

    type: str
    description: str | None = None

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("parameter type must not be empty")
        if normalized not in KNOWN_PARAMETER_TYPES:
            raise ValueError(
                f"unsupported parameter type {normalized!r}; "
                f"supported types: {_SUPPORTED_TYPES}"
            )
        return normalized


class FunctionDefinition(BaseModel):
    """One callable function as declared in the input catalog."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterSpec] = {}
    returns: ParameterSpec

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("function name must not be empty")
        return normalized


class FunctionCatalog(BaseModel):
    """The full set of functions available to the model.

    The catalog accepts either the JSON root list directly or an object with a
    ``functions`` key, so callers can pass ``json.load(...)`` straight in.
    """

    model_config = ConfigDict(extra="forbid")

    functions: list[FunctionDefinition]

    @model_validator(mode="before")
    @classmethod
    def _wrap_root_list(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"functions": data}
        return data

    @model_validator(mode="after")
    def _reject_duplicate_names(self) -> FunctionCatalog:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for function in self.functions:
            if function.name in seen:
                duplicates.add(function.name)
            seen.add(function.name)
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise ValueError(f"duplicate function name(s): {names}")
        return self

    def by_name(self, name: str) -> FunctionDefinition:
        """Return the function called ``name`` or raise ``KeyError``."""
        for function in self.functions:
            if function.name == name:
                return function
        raise KeyError(f"function {name!r} is not present in the catalog")

    def __iter__(self) -> Iterator[FunctionDefinition]:  # type: ignore[override]
        return iter(self.functions)

    def __len__(self) -> int:
        return len(self.functions)

    @classmethod
    def from_list(cls, items: list[dict[str, Any]]) -> FunctionCatalog:
        """Build a catalog from the raw JSON root list."""
        return cls.model_validate(items)
