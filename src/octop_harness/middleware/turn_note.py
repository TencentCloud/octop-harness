"""Request-only note placed after the user message for this turn.

The note is not stored in graph state. Each model call writes it at the same
index. Later injections append to that one message, so the system prompt and
the earlier messages stay a cacheable prefix.

The note role follows the model: ``system`` where the transport accepts a
non-leading system message in place, ``user`` where it does not. Anthropic
rejects a second non-consecutive system message unless the model profile opts
in, and several local models drop the tool call when the note arrives as a
second user message. A model without the capability stamp is assumed to accept
an in-place system message, which is what the OpenAI protocol needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from octop_harness.messages import TURN_NOTE_KEY, is_turn_note

__all__ = ["insert_turn_note", "is_turn_note"]

_MARKERS = ("[本轮说明]", "[turn note]")
_MID_CONVERSATION_SYSTEM = "_harness_mid_conversation_system_messages"

_NoteMessage = HumanMessage | SystemMessage


def _note_role(model: Any) -> type[_NoteMessage]:
    capability = getattr(model, _MID_CONVERSATION_SYSTEM, None)
    if isinstance(capability, bool) and not capability:
        return HumanMessage
    return SystemMessage


def _turn_note_message(content: str, model: Any, *, role: type[_NoteMessage] | None = None) -> _NoteMessage:
    return (role or _note_role(model))(content=content, additional_kwargs={TURN_NOTE_KEY: True})


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
    kwargs[TURN_NOTE_KEY] = True
    messages[index] = existing.model_copy(update={"content": merged, "additional_kwargs": kwargs})
    return messages


def insert_turn_note(messages: Sequence[Any], content: str, *, model: Any = None) -> list[Any]:
    """Place ``content`` in the single note after this turn's user message.

    ``model`` only selects the note role; omit it to keep the default ``system``.
    """
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
        # No user message to follow. A trailing system note would be
        # non-consecutive for transports that hoist it, so keep it a user
        # message; an empty list has nothing to conflict with.
        role = None if not out else HumanMessage
        out.append(_turn_note_message(content, model, role=role))
        return out
    insert_at = user_at + 1
    if insert_at < len(out) and is_turn_note(out[insert_at]):
        return _append_note(out, insert_at, content)
    out.insert(insert_at, _turn_note_message(content, model))
    return out
