"""Reject tool arguments that repeat their own field name.

A small model sometimes copies the schema's field name into the value —
``grep({"pattern": "pattern"})`` — and then stops when the call finds nothing.
The guard names the offending parameter so the model fills a real value instead
of giving up. A required string field whose value equals the field name is the
signal; anything else runs unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)

Language = Literal["en", "zh"]

_MESSAGES: dict[Language, str] = {
    "zh": "参数 {fields} 的值与字段名相同，看起来是照抄了 schema。请填入真实内容后重试。",  # noqa: RUF001
    "en": (
        "The value of {fields} repeats its field name, which looks copied from the schema. "
        "Fill in the real content and try again."
    ),
}


def _string_fields(schema: dict[str, Any]) -> tuple[set[str], set[str]]:
    required = {str(item) for item in (schema.get("required") or [])}
    strings: set[str] = set()
    for name, prop in (schema.get("properties") or {}).items():
        if not isinstance(prop, dict):
            continue
        variants = [prop, *(item for item in (prop.get("anyOf") or []) if isinstance(item, dict))]
        if any(variant.get("type") == "string" for variant in variants):
            strings.add(str(name))
    return required, strings


def placeholder_arg_fields(tool: Any, args: dict[str, Any]) -> list[str]:
    """Required string fields whose value repeats the field name."""
    schema = getattr(tool, "args_schema", None)
    model_json_schema = getattr(schema, "model_json_schema", None)
    if not callable(model_json_schema):
        return []
    try:
        raw = model_json_schema()
    except (TypeError, ValueError):
        return []
    if not isinstance(raw, dict):
        return []
    required, strings = _string_fields(raw)
    return [
        name
        for name in sorted(required & strings)
        if isinstance(args.get(name), str) and args[name].strip().lower() == name.lower()
    ]


class ToolArgsMiddleware(AgentMiddleware[Any, Any]):
    """Block placeholder tool arguments before the tool runs."""

    def __init__(self, *, language: Language = "zh", enabled: bool = True) -> None:
        super().__init__()
        self._language = language if language in _MESSAGES else "en"
        self._enabled = enabled

    def configure(self, *, enabled: bool | None = None, language: Language | None = None) -> None:
        if enabled is not None:
            self._enabled = enabled
        if language in _MESSAGES:
            self._language = language

    def _block(self, request: ToolCallRequest) -> ToolMessage | None:
        if not self._enabled:
            return None
        tool_call = request.tool_call
        tool = getattr(request, "tool", None)
        args = tool_call.get("args") or {}
        if tool is None or not isinstance(args, dict):
            return None
        fields = placeholder_arg_fields(tool, args)
        if not fields:
            return None
        name = str(tool_call.get("name") or "")
        logger.info("ToolArgsGuard rejected placeholder arguments for %s: %s", name, fields)
        return ToolMessage(
            content=_MESSAGES[self._language].format(fields=", ".join(fields)),
            name=name or "tool",
            tool_call_id=str(tool_call.get("id") or ""),
            status="error",
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        blocked = self._block(request)
        if blocked is not None:
            return blocked
        return await handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        blocked = self._block(request)
        if blocked is not None:
            return blocked
        return handler(request)


__all__ = ["ToolArgsMiddleware", "placeholder_arg_fields"]
