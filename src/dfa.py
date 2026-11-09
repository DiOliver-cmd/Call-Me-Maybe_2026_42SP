"""State model and alphabet for the fixed function-call output template (T4.1).

The decoder emits exactly one JSON object, with a fixed key order:

    {"prompt":<json string>,"name":"<function name>","parameters":{...}}

The structural text is injected by the decoder later (T4.5); the model only
chooses the function name and the parameter values. ``ACCEPT`` is reachable
only after the root ``}`` that follows the parameters object. Transitions and
``consume`` are deliberately not implemented here; they belong to T4.2–T4.4.

State table
-----------
=====================  ==========================================  =========================
State                  Expects                                     On success
=====================  ==========================================  =========================
``START``              root ``{``                                  ``ROOT_OPEN``
``ROOT_OPEN``          optional space, fixed ``"prompt"``          ``PROMPT_KEY_DONE``
``PROMPT_KEY_DONE``    optional space, ``:``                       ``PROMPT_VALUE``
``PROMPT_VALUE``       prompt string content, escapes allowed      ``PROMPT_VALUE_DONE``
``PROMPT_VALUE_DONE``  ``,``                                       ``NAME_KEY``
``NAME_KEY``           optional space, fixed ``"name"``            ``NAME_KEY_DONE``
``NAME_KEY_DONE``      optional space, ``:``                       ``NAME_VALUE``
``NAME_VALUE``         name string content                         ``NAME_VALUE_DONE``
``NAME_VALUE_DONE``    ``,``                                       ``PARAMETERS_KEY``
``PARAMETERS_KEY``     optional space, fixed ``"parameters"``      ``PARAMETERS_KEY_DONE``
``PARAMETERS_KEY_DONE``  optional space, ``:``                      ``PARAMETERS_OPEN``
``PARAMETERS_OPEN``    ``{``                                       ``PARAMETERS_EMPTY_OR_KEY``
``PARAMETERS_EMPTY_OR_KEY``  ``}`` for empty, or ``"`` for a key    closed, or ``PARAM_KEY``
``PARAM_KEY``          parameter key string content                ``PARAM_KEY_DONE``
``PARAM_KEY_DONE``     ``:`` then a value                          a value state
``PARAM_VALUE_STRING``  string content                             ``PARAM_VALUE_DONE``
``PARAM_VALUE_NUMBER``  JSON number                                ``PARAM_VALUE_DONE``
``PARAM_VALUE_BOOLEAN``  ``true`` or ``false``                      ``PARAM_VALUE_DONE``
``PARAM_VALUE_DONE``   ``,`` or ``}``                              next key, or closed
``PARAM_KEY_START``    ``"`` opening the next key                  ``PARAM_KEY``
``PARAMETERS_CLOSED``  root ``}``                                  ``ACCEPT``
``ACCEPT``             nothing (terminal)                          —
=====================  ==========================================  =========================

``PARAM_KEY_START`` is an addition to the minimum state list: after a comma a
key opening quote is required, so ``PARAM_VALUE_DONE`` cannot jump straight to
``PARAM_KEY`` (that would accept a trailing comma) or to
``PARAMETERS_EMPTY_OR_KEY`` (which is only valid right after ``{``).

Alphabet
--------
The alphabet is the union of ``STRUCTURAL_CHARS``, ``NUMBER_CHARS``,
``LITERAL_CHARS``, ``ESCAPE_CHARS``, ``WHITESPACE_CHARS``, and any Unicode
content inside a JSON string. Text is supplied as characters for now; T4.4
maps whole token texts onto these characters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

__all__ = [
    "DFA",
    "DFAState",
    "ESCAPE_CHARS",
    "LITERAL_CHARS",
    "NUMBER_CHARS",
    "STRUCTURAL_CHARS",
    "TEMPLATE_KEYS",
    "WHITESPACE_CHARS",
    "is_string_char",
]

# Fixed key order of the output template (T4.5).
TEMPLATE_KEYS: tuple[str, ...] = ("prompt", "name", "parameters")

# Structural JSON characters used by the template.
STRUCTURAL_CHARS: frozenset[str] = frozenset('{}[]:,"')

# Characters used by JSON numbers: sign, digits, decimal point and exponent.
NUMBER_CHARS: frozenset[str] = frozenset("-+0123456789.eE")

# Letters used by the JSON literals ``true`` and ``false``.
LITERAL_CHARS: frozenset[str] = frozenset("truefals")

# Valid characters after a backslash inside a JSON string.
ESCAPE_CHARS: frozenset[str] = frozenset('"\\/bfnrtu')

# Whitespace allowed between structural tokens.
WHITESPACE_CHARS: frozenset[str] = frozenset(" \t\r\n")


class DFAState(Enum):
    """Every state of the fixed output-template machine."""

    START = auto()
    ROOT_OPEN = auto()
    PROMPT_KEY_DONE = auto()
    PROMPT_VALUE = auto()
    PROMPT_VALUE_DONE = auto()
    NAME_KEY = auto()
    NAME_KEY_DONE = auto()
    NAME_VALUE = auto()
    NAME_VALUE_DONE = auto()
    PARAMETERS_KEY = auto()
    PARAMETERS_KEY_DONE = auto()
    PARAMETERS_OPEN = auto()
    PARAMETERS_EMPTY_OR_KEY = auto()
    PARAM_KEY = auto()
    PARAM_KEY_DONE = auto()
    PARAM_VALUE_STRING = auto()
    PARAM_VALUE_NUMBER = auto()
    PARAM_VALUE_BOOLEAN = auto()
    PARAM_VALUE_DONE = auto()
    PARAM_KEY_START = auto()
    PARAMETERS_CLOSED = auto()
    ACCEPT = auto()


@dataclass
class DFA:
    """The state of one in-progress output-template parse."""

    state: DFAState = DFAState.START

    def __post_init__(self) -> None:
        """Reject states that are not :class:`DFAState` members."""
        if not isinstance(self.state, DFAState):
            raise TypeError(
                f"state must be a DFAState, got {type(self.state).__name__}"
            )

    def is_accepting(self) -> bool:
        """Return whether the machine has closed the root object."""
        return self.state is DFAState.ACCEPT

    @classmethod
    def start(cls) -> DFA:
        """Return a fresh machine in ``START``."""
        return cls(state=DFAState.START)


def is_string_char(char: str) -> bool:
    """Return whether ``char`` may appear literally inside a JSON string.

    A closing quote and control characters (``ord(char) < 0x20``) are rejected;
    every other Unicode character is accepted. Escape handling is separate: the
    caller recognises a backslash and then validates the next character against
    :data:`ESCAPE_CHARS`. A single ``"`` is always a closing quote here, so the
    caller must not use this helper for an escaped quote.
    """
    if len(char) != 1:
        raise ValueError(f"expected a single character, got {char!r}")
    if char == '"':
        return False
    return ord(char) >= 0x20
