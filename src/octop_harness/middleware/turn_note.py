"""Request-only note placed after the user message for this turn.

The note is not stored in graph state. Each model call writes it at the same
index. Later injections append to that one message, so the system prompt and
the earlier messages stay a cacheable prefix.

Identity is ``additional_kwargs['_octop_turn_note']``. The visible ``[turn note]``
prefix is display text; a user message may start with the same words.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import HumanMessage

_MARKERS = ("[本轮说明]", "[turn note]")
_TURN_NOTE_KEY = "_octop_turn_note"


def is_turn_note(message: Any) -> bool:
    kwargs = getattr(message, "additional_kwargs", None)
    return isinstance(kwargs, dict) and kwargs.get(_TURN_NOTE_KEY) is True


def _turn_note_message(content: str) -> HumanMessage:
    return HumanMessage(content=content, additional_kwargs={_TURN_NOTE_KEY: True})


def _note_body(content: str) -> str:
    text = content.lstrip()
    for marker in _MARKERS:
        if text.startswith(marker):
            rest = text[len(marker) :]
            return rest[1:] if rest.startswith("\n") else rest
    return content


def _append_note(messages: list[Any], index: int, content: str) -> list[Any]:
    body = _note_body(content).strip()
    if not body:
        return messages
    existing = messages[index]
    merged = str(existing.content).rstrip() + "\n\n" + body
    kwargs = dict(getattr(existing, "additional_kwargs", None) or {})
    kwargs[_TURN_NOTE_KEY] = True
    messages[index] = existing.model_copy(update={"content": merged, "additional_kwargs": kwargs})
    return messages


def insert_turn_note(messages: Sequence[Any], content: str) -> list[Any]:
    """Place ``content`` in the single note after this turn's user message."""
    out = list(messages)
    user_at: int | None = None
    for index in range(len(out) - 1, -1, -1):
        msg = out[index]
        if isinstance(msg, HumanMessage) and not is_turn_note(msg):
            user_at = index
            break
    if user_at is None:
        if out and is_turn_note(out[-1]):
            return _append_note(out, len(out) - 1, content)
        out.append(_turn_note_message(content))
        return out
    insert_at = user_at + 1
    if insert_at < len(out) and is_turn_note(out[insert_at]):
        return _append_note(out, insert_at, content)
    out.insert(insert_at, _turn_note_message(content))
    return out
