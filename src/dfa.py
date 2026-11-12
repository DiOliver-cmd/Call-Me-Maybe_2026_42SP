"""DFA states, alphabet, pure transitions and slot validation (T4.1-T4.3).

The decoder emits exactly one JSON object, with a fixed key order:

    {"prompt":<json string>,"name":"<function name>","parameters":{...}}

The structural text is injected by the decoder later (T4.5); the model only
chooses the function name and the parameter values. :func:`transition` is the
pure, structural char-by-char core; :meth:`DFA.consume` adds the four slot
validators on top and updates one machine atomically. ``ACCEPT`` is reachable
only after the root ``}`` that follows the parameters object.

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

Slot validation
---------------
``transition`` stays structural and pure. ``DFA.consume`` additionally checks:

* function names against an optional :class:`NameMatcher`: prefixes while the
  name is being typed, exact membership when the closing quote arrives. With
  ``names=None`` the T4.2 behavior is preserved;
* numbers with :func:`is_number_prefix` while typing and
  :func:`is_complete_number` on the delimiter;
* booleans as prefixes of ``true``/``false``, complete on the delimiter;
* string escapes, including exactly four hex digits after a ``u`` escape.

A rejected character leaves the state and every buffer unchanged (atomicity),
which is what T4.4 will rely on for token-level rollback. T5.1 can use
:meth:`NameMatcher.candidates` to mask the next name token.

Reuse and whitespace
--------------------
The decoder should create one machine per prompt with
:meth:`DFA.start` or call :meth:`DFA.reset` before reuse; assigning only
``state`` leaks slot buffers and the escape counter. Whitespace is accepted only
between structural tokens. Outer whitespace before the root ``{`` and after the
final ``}`` is rejected, so a multi-character token such as ``"}\\n"`` will be
rejected by T4.4's ``consume_token``. The decoder must handle EOS and any outer
whitespace policy explicitly.

Alphabet
--------
The alphabet is the union of ``STRUCTURAL_CHARS``, ``NUMBER_CHARS``,
``LITERAL_CHARS``, ``ESCAPE_CHARS``, ``WHITESPACE_CHARS``, and any Unicode
content inside a JSON string. Text is supplied as characters for now; T4.4
maps whole token texts onto these characters.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from enum import Enum, auto
from typing import Annotated

from pydantic import BeforeValidator, ConfigDict
from pydantic.dataclasses import dataclass

__all__ = [
    "DFA",
    "DFAState",
    "ESCAPE_CHARS",
    "LITERAL_CHARS",
    "NUMBER_CHARS",
    "NameMatcher",
    "STRUCTURAL_CHARS",
    "TEMPLATE_KEYS",
    "WHITESPACE_CHARS",
    "is_complete_number",
    "is_number_prefix",
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

_HEX_DIGITS: frozenset[str] = frozenset("0123456789abcdefABCDEF")

_BOOLEAN_LITERALS: tuple[str, ...] = ("false", "true")

_STR_NAMES_ERROR = (
    "names must be an iterable of names, not a str; "
    "pass a list such as ['fn_add']"
)

# Complete JSON number grammar from T4.3.
_COMPLETE_NUMBER_RE = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
)

# Prefix grammar: a fraction dot may be the last character, but an exponent
# cannot follow a dot that has no digit yet ("1." is a prefix, "1.e" is not).
_NUMBER_PREFIX_RE = re.compile(
    r"-?(?:0|[1-9][0-9]*)"
    r"(?:"
    r"\.[0-9]+(?:[eE][+-]?[0-9]*)?"
    r"|\."
    r"|[eE][+-]?[0-9]*"
    r")?"
)


class NameMatcher:
    """Prefix/exact matcher for the function names allowed in a call.

    Every name must be representable inside the ``NAME_VALUE`` string slot:
    quotes, backslashes and control characters are rejected because that slot
    would never accept them as literal content.
    """

    def __init__(self, names: Iterable[str]) -> None:
        """Store ``names`` as a frozenset, rejecting invalid entries."""
        if isinstance(names, str):
            raise TypeError(_STR_NAMES_ERROR)
        items = tuple(names)
        if not items:
            raise ValueError("names must not be empty")
        for name in items:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"name must be a non-blank string, got {name!r}"
                )
            if any(
                char in {'"', "\\"} or ord(char) < 0x20 for char in name
            ):
                raise ValueError(
                    f"name {name!r} contains a character the name slot "
                    "cannot represent"
                )
        self._names = frozenset(items)

    def is_prefix(self, text: str) -> bool:
        """Return whether some allowed name starts with ``text``.

        The empty string is a valid prefix. This is what lets the decoder keep
        typing a name that is not complete yet.
        """
        return any(name.startswith(text) for name in self._names)

    def is_complete(self, text: str) -> bool:
        """Return whether ``text`` is exactly one of the allowed names."""
        return text in self._names

    def candidates(self, prefix: str) -> tuple[str, ...]:
        """Return the allowed names starting with ``prefix``, sorted."""
        return tuple(
            sorted(
                name for name in self._names if name.startswith(prefix)
            )
        )


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


def _validate_state(value: object) -> DFAState:
    """Keep the historical ``TypeError`` for a non-:class:`DFAState` value.

    Pydantic would otherwise wrap this in a ``ValidationError`` before
    ``__post_init__`` runs, which would change the public error contract.
    """
    if not isinstance(value, DFAState):
        raise TypeError(
            f"state must be a DFAState, got {type(value).__name__}"
        )
    return value


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class DFA:
    """The state of one in-progress output-template parse."""

    state: Annotated[DFAState, BeforeValidator(_validate_state)] = (
        DFAState.START
    )
    names: NameMatcher | None = None
    name_buffer: str = ""
    number_buffer: str = ""
    literal_buffer: str = ""
    unicode_remaining: int = 0

    def __post_init__(self) -> None:
        """Reject states that are not :class:`DFAState` members."""
        if not isinstance(self.state, DFAState):
            raise TypeError(
                f"state must be a DFAState, got {type(self.state).__name__}"
            )

    def is_accepting(self) -> bool:
        """Return whether the machine has closed the root object."""
        return self.state is DFAState.ACCEPT

    def reset(self) -> None:
        """Return to ``START`` and clear every slot buffer, keeping ``names``.

        The decoder should create one machine per prompt with
        :meth:`DFA.start` or call this method before reuse. Assigning only
        ``state`` leaks the name/number/literal buffers and the escape counter
        into the next prompt.
        """
        self.state = DFAState.START
        self.name_buffer = ""
        self.number_buffer = ""
        self.literal_buffer = ""
        self.unicode_remaining = 0

    def try_consume_token(self, text: str) -> DFA | None:
        """Try to consume one complete token text atomically.

        ``text`` is token text and may contain one or several characters, such
        as ``","``, ``"}"``, ``"2.0"``, ``"true"`` or ``"\\u00e9"``. The machine
        is shallow-copied and each character is passed to :meth:`consume` on the
        copy. If any character fails, ``None`` is returned and ``self`` is
        untouched. An empty string succeeds and returns an identical copy.

        On success the returned :class:`DFA` is a new object, not ``self``.
        """
        candidate = copy.copy(self)
        for char in text:
            if not candidate.consume(char):
                return None
        return candidate

    def consume_token(self, text: str) -> bool:
        """Consume one complete token text, committing only on full success.

        This is the atomic token-level wrapper used by the decoder. ``text`` is
        token text and may span several characters; multi-character tokens such
        as ``","``, ``"}"``, ``"2.0"``, ``"true"`` and ``"\\u00e9"`` are common.
        Empty text is a successful no-op.

        On failure the method returns ``False`` and leaves the state and every
        slot buffer unchanged. On success it commits the advanced state and
        buffers from :meth:`try_consume_token` into ``self`` and returns
        ``True``.
        """
        advanced = self.try_consume_token(text)
        if advanced is None:
            return False
        self.state = advanced.state
        self.name_buffer = advanced.name_buffer
        self.number_buffer = advanced.number_buffer
        self.literal_buffer = advanced.literal_buffer
        self.unicode_remaining = advanced.unicode_remaining
        return True

    def consume(self, char: str) -> bool:
        """Consume one character, validating the active slot atomically.

        The structural move comes from :func:`transition`; the slot checks are
        layered on top. On rejection neither the state nor any buffer changes,
        so a failed character can be dropped without corrupting the machine.
        """
        if not isinstance(char, str) or len(char) != 1:
            return False

        state = self.state
        next_state = transition(state, char)
        if next_state is None:
            return False

        name_buffer = self.name_buffer
        number_buffer = self.number_buffer
        literal_buffer = self.literal_buffer
        unicode_remaining = self.unicode_remaining
        names = self.names

        if state is DFAState.PROMPT_VALUE_START and char == '"':
            unicode_remaining = 0

        if state is DFAState.NAME_VALUE_START and char == '"':
            name_buffer = ""
            unicode_remaining = 0

        if state is DFAState.PARAM_KEY_DONE and char == ":":
            number_buffer = ""
            literal_buffer = ""
            unicode_remaining = 0

        if (
            state
            in {
                DFAState.PARAMETERS_EMPTY_OR_KEY,
                DFAState.PARAM_KEY_START,
            }
            and char == '"'
        ):
            number_buffer = ""
            literal_buffer = ""
            unicode_remaining = 0

        if state is DFAState.PARAM_VALUE_START:
            if char == '"':
                number_buffer = ""
                literal_buffer = ""
                unicode_remaining = 0
            elif char in {"t", "f"}:
                number_buffer = ""
                literal_buffer = char
                unicode_remaining = 0
            elif char == "-" or char in "0123456789":
                number_buffer = char
                literal_buffer = ""
                unicode_remaining = 0

        if state is DFAState.NAME_VALUE:
            if char == '"':
                if names is not None and not names.is_complete(name_buffer):
                    return False
            else:
                candidate = name_buffer + char
                if names is not None and not names.is_prefix(candidate):
                    return False
                name_buffer = candidate

        if state is DFAState.PARAM_VALUE_NUMBER:
            if char in NUMBER_CHARS:
                candidate = number_buffer + char
                if not is_number_prefix(candidate):
                    return False
                number_buffer = candidate
            elif char in WHITESPACE_CHARS or char in {",", "}"}:
                if not is_complete_number(number_buffer):
                    return False
                number_buffer = ""
                literal_buffer = ""
            else:
                return False

        if state is DFAState.PARAM_VALUE_BOOLEAN:
            if char in LITERAL_CHARS:
                candidate = literal_buffer + char
                if not _is_boolean_prefix(candidate):
                    return False
                literal_buffer = candidate
            elif char in WHITESPACE_CHARS or char in {",", "}"}:
                if not _is_complete_boolean(literal_buffer):
                    return False
                literal_buffer = ""
                number_buffer = ""
            else:
                return False

        if state in {
            DFAState.PROMPT_VALUE_ESCAPE,
            DFAState.PARAM_VALUE_STRING_ESCAPE,
        }:
            unicode_remaining = 4 if char == "u" else 0

        if state in {
            DFAState.PROMPT_VALUE,
            DFAState.PARAM_VALUE_STRING,
        } and unicode_remaining > 0:
            if char not in _HEX_DIGITS:
                return False
            unicode_remaining -= 1

        self.state = next_state
        self.name_buffer = name_buffer
        self.number_buffer = number_buffer
        self.literal_buffer = literal_buffer
        self.unicode_remaining = unicode_remaining
        return True

    @classmethod
    def start(cls, names: Iterable[str] | None = None) -> DFA:
        """Return a fresh machine in ``START``.

        ``names`` enables function-name validation; passing ``None`` keeps the
        T4.2 behavior of accepting any name string. A bare string is rejected
        instead of being silently split into characters.
        """
        if isinstance(names, str):
            raise TypeError(_STR_NAMES_ERROR)
        matcher = NameMatcher(names) if names is not None else None
        return cls(state=DFAState.START, names=matcher)


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


def is_number_prefix(text: str) -> bool:
    """Return whether ``text`` can still become a complete JSON number.

    Partial forms such as ``""``, ``"-"``, ``"1."``, ``"1e"`` and ``"1e+"``
    are accepted. Invalid starts and irrecoverable forms (``"01"``, ``"+1"``,
    ``".5"``, ``"1.."``, ``"1.e"``, ``"--1"``, ``"1.-"``) are rejected.
    """
    if text in {"", "-"}:
        return True
    return _NUMBER_PREFIX_RE.fullmatch(text) is not None


def is_complete_number(text: str) -> bool:
    """Return whether ``text`` matches the full T4.3 JSON number grammar."""
    return _COMPLETE_NUMBER_RE.fullmatch(text) is not None


def _is_boolean_prefix(text: str) -> bool:
    return any(literal.startswith(text) for literal in _BOOLEAN_LITERALS)


def _is_complete_boolean(text: str) -> bool:
    return text in _BOOLEAN_LITERALS


def transition(state: DFAState, char: str) -> DFAState | None:
    """Return the next state after consuming ``char``, or ``None``.

    The function is pure and deterministic: it reads no global state, mutates
    nothing, and always returns the same state for the same arguments. Invalid
    characters and dead ends return ``None``. Slot validation lives in
    :meth:`DFA.consume`, not here.
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
