"""DFA states, alphabet and pure transitions for the output template (T4.1/T4.2).

The decoder emits exactly one JSON object, with a fixed key order:

    {"prompt":<json string>,"name":"<function name>","parameters":{...}}

The structural text is injected by the decoder later (T4.5); the model only
chooses the function name and the parameter values. :func:`transition` is the
pure char-by-char core; :meth:`DFA.consume` updates one machine. T4.3 refines
slot validation (catalog names, JSON numbers, true/false, unicode escapes) and
T4.4 adds ``consume_token``. ``ACCEPT`` is reachable only after the root ``}``
that follows the parameters object.

State table
-----------
``ws`` means any character in :data:`WHITESPACE_CHARS`.

| State | Expects | On success |
| --- | --- | --- |
| ``START`` | root ``{`` | ``ROOT_OPEN`` |
| ``ROOT_OPEN`` | ws, then ``"`` | ``PROMPT_KEY_STRING`` |
| ``PROMPT_KEY_STRING`` | key content; ``"`` closes | ``PROMPT_KEY_DONE`` |
| ``PROMPT_KEY_DONE`` | ws, then ``:`` | ``PROMPT_VALUE_START`` |
| ``PROMPT_VALUE_START`` | ws, then opening ``"`` | ``PROMPT_VALUE`` |
| ``PROMPT_VALUE`` | string content; quote closes, backslash escapes | done / escape |
| ``PROMPT_VALUE_ESCAPE`` | one char in ``ESCAPE_CHARS`` | ``PROMPT_VALUE`` |
| ``PROMPT_VALUE_DONE`` | ws, then ``,`` | ``NAME_KEY`` |
| ``NAME_KEY`` | ws, then ``"`` | ``NAME_KEY_STRING`` |
| ``NAME_KEY_STRING`` | key content; ``"`` closes | ``NAME_KEY_DONE`` |
| ``NAME_KEY_DONE`` | ws, then ``:`` | ``NAME_VALUE_START`` |
| ``NAME_VALUE_START`` | ws, then opening ``"`` | ``NAME_VALUE`` |
| ``NAME_VALUE`` | name content; quote closes, backslash invalid | ``NAME_VALUE_DONE`` |
| ``NAME_VALUE_DONE`` | ws, then ``,`` | ``PARAMETERS_KEY`` |
| ``PARAMETERS_KEY`` | ws, then ``"`` | ``PARAMETERS_KEY_STRING`` |
| ``PARAMETERS_KEY_STRING`` | key content; ``"`` closes | ``PARAMETERS_KEY_DONE`` |
| ``PARAMETERS_KEY_DONE`` | ws, then ``:`` | ``PARAMETERS_OPEN`` |
| ``PARAMETERS_OPEN`` | ws, then ``{`` | ``PARAMETERS_EMPTY_OR_KEY`` |
| ``PARAMETERS_EMPTY_OR_KEY`` | ws; ``}`` empty; ``"`` key | closed / key state |
| ``PARAM_KEY`` | key content; quote closes, backslash invalid | ``PARAM_KEY_DONE`` |
| ``PARAM_KEY_DONE`` | ws, then ``:`` | ``PARAM_VALUE_START`` |
| ``PARAM_VALUE_START`` | ws; quote, ``t``/``f``, ``-``/digit | string / boolean / number |
| ``PARAM_VALUE_STRING`` | string content; quote closes, backslash escapes | done / escape |
| ``PARAM_VALUE_STRING_ESCAPE`` | one char in ``ESCAPE_CHARS`` | ``PARAM_VALUE_STRING`` |
| ``PARAM_VALUE_NUMBER`` | ``NUMBER_CHARS``; ws/``,``/``}`` end | done / next key / closed |
| ``PARAM_VALUE_BOOLEAN`` | ``LITERAL_CHARS``; ws/``,``/``}`` end | done / next key / closed |
| ``PARAM_VALUE_DONE`` | ws; ``,`` or ``}`` | next key / closed |
| ``PARAM_KEY_START`` | ws, then ``"`` | ``PARAM_KEY`` |
| ``PARAMETERS_CLOSED`` | ws, then ``}`` | ``ACCEPT`` |
| ``ACCEPT`` | nothing (terminal) | — |

Three states extend the minimum T4.1 list. ``PROMPT_VALUE_START`` and
``NAME_VALUE_START`` consume the opening quote after the colon; without them the
first quote would close the value immediately. ``PARAM_VALUE_START`` separates
the colon from the value dispatch, because one pure call consumes one character.
Whitespace is accepted in all three start states, matching JSON. The earlier
``PARAM_KEY_START`` remains necessary: after a comma a key opening quote is
required, so a ``}`` cannot be accepted as a trailing comma.

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
    "transition",
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
    PROMPT_KEY_STRING = auto()
    PROMPT_KEY_DONE = auto()
    PROMPT_VALUE_START = auto()
    PROMPT_VALUE = auto()
    PROMPT_VALUE_ESCAPE = auto()
    PROMPT_VALUE_DONE = auto()
    NAME_KEY = auto()
    NAME_KEY_STRING = auto()
    NAME_KEY_DONE = auto()
    NAME_VALUE_START = auto()
    NAME_VALUE = auto()
    NAME_VALUE_DONE = auto()
    PARAMETERS_KEY = auto()
    PARAMETERS_KEY_STRING = auto()
    PARAMETERS_KEY_DONE = auto()
    PARAMETERS_OPEN = auto()
    PARAMETERS_EMPTY_OR_KEY = auto()
    PARAM_KEY = auto()
    PARAM_KEY_DONE = auto()
    PARAM_VALUE_START = auto()
    PARAM_VALUE_STRING = auto()
    PARAM_VALUE_STRING_ESCAPE = auto()
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

    def consume(self, char: str) -> bool:
        """Consume one character, updating the state on success.

        This is the stateful convenience used by the decoder. It returns
        ``True`` when :func:`transition` accepts ``char``; on rejection it
        leaves the state unchanged and returns ``False``.
        """
        next_state = transition(self.state, char)
        if next_state is None:
            return False
        self.state = next_state
        return True

    @classmethod
    def start(cls) -> DFA:
        """Return a fresh machine in ``START``."""
        return cls(state=DFAState.START)


def is_string_char(char: str) -> bool:
    """Return whether ``char`` may appear literally inside a JSON string.

    A closing quote and control characters (``ord(char) < 0x20``) are rejected;
    every other Unicode character is accepted. Escape handling is separate: the
    caller recognises a backslash and then validates the next character against
    :data:`ESCAPE_CHARS`. A single quote is always a closing quote here, so the
    caller must not use this helper for an escaped quote.
    """
    if len(char) != 1:
        raise ValueError(f"expected a single character, got {char!r}")
    if char == '"':
        return False
    return ord(char) >= 0x20


def transition(state: DFAState, char: str) -> DFAState | None:
    """Return the next state after consuming ``char``, or ``None``.

    The function is pure and deterministic: it reads no global state, mutates
    nothing, and always returns the same state for the same arguments. Invalid
    characters and dead ends return ``None``. T4.3 will refine the slot states
    (catalog name, JSON number grammar, true/false, unicode escapes) without
    changing this structural dispatch.
    """
    if not isinstance(state, DFAState):
        return None
    if not isinstance(char, str) or len(char) != 1:
        return None

    whitespace = char in WHITESPACE_CHARS
    control = ord(char) < 0x20

    if state is DFAState.START:
        return DFAState.ROOT_OPEN if char == "{" else None

    if state is DFAState.ROOT_OPEN:
        if whitespace:
            return state
        if char == '"':
            return DFAState.PROMPT_KEY_STRING
        return None

    if state is DFAState.PROMPT_KEY_STRING:
        if char == '"':
            return DFAState.PROMPT_KEY_DONE
        if char == "\\" or control:
            return None
        return state

    if state is DFAState.PROMPT_KEY_DONE:
        if whitespace:
            return state
        if char == ":":
            return DFAState.PROMPT_VALUE_START
        return None

    if state is DFAState.PROMPT_VALUE_START:
        if whitespace:
            return state
        if char == '"':
            return DFAState.PROMPT_VALUE
        return None

    if state is DFAState.PROMPT_VALUE:
        if char == '"':
            return DFAState.PROMPT_VALUE_DONE
        if char == "\\":
            return DFAState.PROMPT_VALUE_ESCAPE
        if control:
            return None
        return state

    if state is DFAState.PROMPT_VALUE_ESCAPE:
        if char in ESCAPE_CHARS:
            return DFAState.PROMPT_VALUE
        return None

    if state is DFAState.PROMPT_VALUE_DONE:
        if whitespace:
            return state
        if char == ",":
            return DFAState.NAME_KEY
        return None

    if state is DFAState.NAME_KEY:
        if whitespace:
            return state
        if char == '"':
            return DFAState.NAME_KEY_STRING
        return None

    if state is DFAState.NAME_KEY_STRING:
        if char == '"':
            return DFAState.NAME_KEY_DONE
        if char == "\\" or control:
            return None
        return state

    if state is DFAState.NAME_KEY_DONE:
        if whitespace:
            return state
        if char == ":":
            return DFAState.NAME_VALUE_START
        return None

    if state is DFAState.NAME_VALUE_START:
        if whitespace:
            return state
        if char == '"':
            return DFAState.NAME_VALUE
        return None

    if state is DFAState.NAME_VALUE:
        if char == '"':
            return DFAState.NAME_VALUE_DONE
        if char == "\\" or control:
            return None
        return state

    if state is DFAState.NAME_VALUE_DONE:
        if whitespace:
            return state
        if char == ",":
            return DFAState.PARAMETERS_KEY
        return None

    if state is DFAState.PARAMETERS_KEY:
        if whitespace:
            return state
        if char == '"':
            return DFAState.PARAMETERS_KEY_STRING
        return None

    if state is DFAState.PARAMETERS_KEY_STRING:
        if char == '"':
            return DFAState.PARAMETERS_KEY_DONE
        if char == "\\" or control:
            return None
        return state

    if state is DFAState.PARAMETERS_KEY_DONE:
        if whitespace:
            return state
        if char == ":":
            return DFAState.PARAMETERS_OPEN
        return None

    if state is DFAState.PARAMETERS_OPEN:
        if whitespace:
            return state
        if char == "{":
            return DFAState.PARAMETERS_EMPTY_OR_KEY
        return None

    if state is DFAState.PARAMETERS_EMPTY_OR_KEY:
        if whitespace:
            return state
        if char == "}":
            return DFAState.PARAMETERS_CLOSED
        if char == '"':
            return DFAState.PARAM_KEY
        return None

    if state is DFAState.PARAM_KEY:
        if char == '"':
            return DFAState.PARAM_KEY_DONE
        if char == "\\" or control:
            return None
        return state

    if state is DFAState.PARAM_KEY_DONE:
        if whitespace:
            return state
        if char == ":":
            return DFAState.PARAM_VALUE_START
        return None

    if state is DFAState.PARAM_VALUE_START:
        if whitespace:
            return state
        if char == '"':
            return DFAState.PARAM_VALUE_STRING
        if char in {"t", "f"}:
            return DFAState.PARAM_VALUE_BOOLEAN
        if char == "-" or char in "0123456789":
            return DFAState.PARAM_VALUE_NUMBER
        return None

    if state is DFAState.PARAM_VALUE_STRING:
        if char == '"':
            return DFAState.PARAM_VALUE_DONE
        if char == "\\":
            return DFAState.PARAM_VALUE_STRING_ESCAPE
        if control:
            return None
        return state

    if state is DFAState.PARAM_VALUE_STRING_ESCAPE:
        if char in ESCAPE_CHARS:
            return DFAState.PARAM_VALUE_STRING
        return None

    if state is DFAState.PARAM_VALUE_NUMBER:
        if char in NUMBER_CHARS:
            return state
        if whitespace:
            return DFAState.PARAM_VALUE_DONE
        if char == ",":
            return DFAState.PARAM_KEY_START
        if char == "}":
            return DFAState.PARAMETERS_CLOSED
        return None

    if state is DFAState.PARAM_VALUE_BOOLEAN:
        if char in LITERAL_CHARS:
            return state
        if whitespace:
            return DFAState.PARAM_VALUE_DONE
        if char == ",":
            return DFAState.PARAM_KEY_START
        if char == "}":
            return DFAState.PARAMETERS_CLOSED
        return None

    if state is DFAState.PARAM_VALUE_DONE:
        if whitespace:
            return state
        if char == ",":
            return DFAState.PARAM_KEY_START
        if char == "}":
            return DFAState.PARAMETERS_CLOSED
        return None

    if state is DFAState.PARAM_KEY_START:
        if whitespace:
            return state
        if char == '"':
            return DFAState.PARAM_KEY
        return None

    if state is DFAState.PARAMETERS_CLOSED:
        if whitespace:
            return state
        if char == "}":
            return DFAState.ACCEPT
        return None

    if state is DFAState.ACCEPT:
        return None

    return None
