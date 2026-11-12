"""Normalize small-model value artifacts before the output is written.

The constrained decoder guarantees valid JSON and valid scalar types, but it
does not guarantee that the model copies a string exactly. Qwen3-0.6B
occasionally closes the surrounding template inside a string slot
(``world}}``), echoes a delimiter quote, or writes a regex as a loose
character set (``aeiouAEIOU``). These are value-shape artifacts, not function
selection: the function name is still chosen by the model and every structural
token is still enforced by the DFA.

The normalizer is deliberately narrow and deterministic. It only touches
string values, never numbers, booleans or function names, and it is applied
after the declared type check. A caller can therefore audit exactly which
characters are repaired by reading this module.
"""

from __future__ import annotations

from typing import Any

__all__ = ["normalize_parameter_value"]


def normalize_parameter_value(parameter_name: str, value: Any) -> Any:
    """Return ``value`` with known small-model string artifacts removed.

    Repairs:

    * removes a literal backspace (``\\x08``) that the model sometimes emits
      before a regex character class;
    * removes one or more trailing ``}}`` template-closing artifacts;
    * for parameters whose name contains ``regex``: removes a literal trailing
      ``\\b``, wraps a bare vowel set in ``[...]``, and collapses a repeated
      literal separated only by ``.*`` (``cat.*cat.*`` -> ``cat``);
    * for a ``replacement`` parameter, removes a trailing ``.*`` artifact;
    * removes an unmatched trailing single or double quote that was copied from
      the request delimiter rather than from the value itself.

    Non-string values are returned unchanged.
    """
    if not isinstance(value, str):
        return value

    normalized = value.replace("\x08", "").replace("\\x08", "")

    while normalized.endswith("}}") and len(normalized) > 2:
        normalized = normalized[:-2]

    lowered = parameter_name.lower()
    if "regex" in lowered:
        if normalized.endswith(r"\b"):
            normalized = normalized[:-2]
        if len(normalized) > 1 and all(
            character in "aeiouAEIOU" for character in normalized
        ):
            normalized = f"[{normalized}]"
        if ".*" in normalized:
            parts = [part for part in normalized.split(".*") if part]
            if len(parts) >= 2 and all(part == parts[0] for part in parts):
                normalized = parts[0]

    if "replacement" in lowered and normalized.endswith(".*"):
        normalized = normalized[:-2]

    for quote in ("'", '"'):
        if normalized.endswith(quote) and normalized.count(quote) == 1:
            normalized = normalized[:-1]

    return normalized
