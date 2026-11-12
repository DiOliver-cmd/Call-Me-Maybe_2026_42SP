"""Pydantic models for the dynamic function schema and the model output.

The function catalog is data-driven: function names, descriptions, parameter
names and return types all come from ``data/input/functions_definition.json``
and must never be hardcoded. The input models validate that catalog; the
output models (:class:`FunctionCall` and :class:`FunctionCallBatch`) validate
what the model produced before it is written to disk.

The constrained decoder (DFA) only supports four parameter types: ``string``,
``number``, ``boolean`` and ``integer``. Any other type (for example ``array``
or ``object``) is rejected with a clear error at validation time, because the
grammar cannot represent it and silently accepting it would let an
unsupported function slip into the catalog.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    RootModel,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from src.errors import SchemaError
from src.normalize import normalize_parameter_value

# Strict JSON scalars. The strict variants keep the runtime type intact instead
# of coercing: ``1`` stays ``int``, ``2.0`` stays ``float`` and ``True`` stays
# ``bool`` even though ``bool`` is a subclass of ``int`` in Python.
ParameterValue = StrictStr | StrictInt | StrictFloat | StrictBool

# Human-readable labels used when reporting a type mismatch in a call.
_EXPECTED_TYPE_LABELS: dict[str, str] = {
    "string": "a string",
    "number": "a number (int or float, not bool)",
    "integer": "an integer (int, not bool)",
    "boolean": "a boolean",
}

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
    parameters: dict[str, ParameterSpec]
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

    def validate_call(self, call: FunctionCall) -> FunctionCall:
        """Validate a decoded call against this catalog as a final safety net.

        The function name must be declared, the parameter keys must match the
        declaration exactly (every declared parameter is required), and each
        value must have the declared type. ``bool`` never satisfies ``number``
        or ``integer`` because :func:`_value_matches_spec` uses strict type
        identity. A declared ``number`` parameter given a Python ``int`` is
        normalized to ``float`` in place, so the JSON output keeps the decimal
        form (``2`` becomes ``2.0``).

        Raises:
            SchemaError: if the name is unknown, a parameter is missing or
                unexpected, or a value has the wrong type.

        Returns:
            The same ``call`` object, with integer ``number`` values
            normalized to ``float``, so the pipeline can append it.
        """
        try:
            function = self.by_name(call.name)
        except KeyError as exc:
            raise SchemaError(
                f"function {call.name!r} is not declared in the catalog"
            ) from exc

        declared = function.parameters
        provided = call.parameters
        unexpected = sorted(set(provided) - set(declared))
        if unexpected:
            raise SchemaError(
                f"function {function.name!r} received unexpected "
                f"parameter(s): {', '.join(unexpected)}"
            )

        missing = sorted(set(declared) - set(provided))
        if missing:
            raise SchemaError(
                f"function {function.name!r} is missing required "
                f"parameter(s): {', '.join(missing)}"
            )

        for parameter_name, spec in declared.items():
            value = provided[parameter_name]
            if not _value_matches_spec(value, spec.type):
                expected_label = _EXPECTED_TYPE_LABELS.get(spec.type, spec.type)
                raise SchemaError(
                    f"function {function.name!r} parameter "
                    f"{parameter_name!r} must be {expected_label}, "
                    f"got {type(value).__name__}"
                )
            if spec.type == "number" and type(value) is int:
                provided[parameter_name] = float(value)
            elif spec.type == "string":
                provided[parameter_name] = normalize_parameter_value(
                    parameter_name, value
                )
        return call

    def __iter__(self) -> Iterator[FunctionDefinition]:  # type: ignore[override]
        return iter(self.functions)

    def __len__(self) -> int:
        return len(self.functions)

    @classmethod
    def from_list(cls, items: list[dict[str, Any]]) -> FunctionCatalog:
        """Build a catalog from the raw JSON root list."""
        return cls.model_validate(items)


def _value_matches_spec(value: ParameterValue, expected: str) -> bool:
    """Return whether ``value`` matches the declared parameter ``expected``.

    ``type(...) is`` is used instead of ``isinstance`` on purpose: ``bool`` is a
    subclass of ``int`` in Python, so ``isinstance(True, int)`` is ``True`` and
    would let a boolean masquerade as a number or integer.
    """
    if expected == "string":
        return type(value) is str
    if expected == "number":
        return type(value) is int or type(value) is float
    if expected == "integer":
        return type(value) is int
    if expected == "boolean":
        return type(value) is bool
    # ``ParameterSpec`` already rejects unknown types; reject defensively here.
    return False


class FunctionCall(BaseModel):
    """A single function call as produced by the model.

    This is the output-side counterpart of :class:`FunctionDefinition`: the
    ``prompt`` is copied verbatim from the request, while ``name`` and
    ``parameters`` describe the model's decision.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, ParameterValue] = {}

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("function name must not be empty")
        return normalized

    def validate_against(self, function: FunctionDefinition) -> FunctionCall:
        """Check that this call agrees with the declared ``function`` signature.

        Parameter names must match the declaration exactly (no missing and no
        unexpected keys), and each value must match its declared type. T6.2
        wires this primitive into the pipeline so that a structurally valid
        :class:`FunctionCall` cannot be emitted with names or types that
        disagree with the catalog. A declared ``number`` parameter given a
        Python ``int`` is normalized to ``float`` in place, so the JSON output
        keeps the decimal form (``2`` becomes ``2.0``).

        Raises:
            ValueError: if parameter keys differ from the declaration or a
                value does not match its declared type. The message names the
                function, the parameter and the expected/actual types.

        Returns:
            ``self`` on success, with integer ``number`` values normalized to
            ``float``, so calls can be chained.
        """
        declared = function.parameters
        provided = self.parameters

        missing = sorted(set(declared) - set(provided))
        unexpected = sorted(set(provided) - set(declared))
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(f"missing parameter(s): {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected parameter(s): {', '.join(unexpected)}")
            raise ValueError(
                f"function {function.name!r} call has {'; '.join(details)}"
            )

        for key, spec in declared.items():
            value = provided[key]
            if not _value_matches_spec(value, spec.type):
                expected_label = _EXPECTED_TYPE_LABELS.get(spec.type, spec.type)
                raise ValueError(
                    f"parameter {key!r} of function {function.name!r} "
                    f"expects {expected_label}, got {type(value).__name__}"
                )
            if spec.type == "number" and type(value) is int:
                provided[key] = float(value)
            elif spec.type == "string":
                provided[key] = normalize_parameter_value(key, value)
        return self


class FunctionCallBatch(RootModel[list[FunctionCall]]):
    """The full output file: a JSON root list of function calls.

    Subclassing :class:`~pydantic.RootModel` makes ``model_dump()`` return the
    plain list directly, which is exactly the shape required by the output
    file (no wrapping object). Extra keys are still rejected per call by
    :class:`FunctionCall`.
    """

    def __iter__(self) -> Iterator[FunctionCall]:  # type: ignore[override]
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    @property
    def calls(self) -> list[FunctionCall]:
        """The wrapped list of calls."""
        return self.root

    @classmethod
    def from_calls(cls, items: list[FunctionCall]) -> FunctionCallBatch:
        """Build a batch from already-validated calls."""
        return cls(items)
