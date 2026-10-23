"""Middleware that breaks retry loops: same tool + same args failing repeatedly."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

logger = logging.getLogger(__name__)

_HINT = (
    "\n\n[loop-guard] This exact call to `{tool}` has now failed {n} times in a row "
    "with identical arguments. Do NOT retry it with the same arguments; change the "
    "arguments, use a different tool, or tell the user what is blocking you."
)
_STOP = (
    "Stopped: `{tool}` was called again with the same arguments after failing {n} "
    "times in a row. I am ending this turn instead of retrying; please check the "
    "error above or tell me how to proceed."
)


def _signature(name: str, args: Any) -> str:
    try:
        return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
    except (TypeError, ValueError):
        return f"{name}:{args!r}"


def _is_error(msg: ToolMessage) -> bool:
    if msg.status == "error":
        return True
    content = msg.content
    return isinstance(content, str) and content.lstrip().startswith("Error")


class LoopGuardMiddleware(AgentMiddleware[Any, Any]):
    """Stateless guard derived from message history.

    After ``warn_after`` consecutive identical failed calls the next failure
    result gets an instruction not to retry; if the model still issues the same
    call after ``block_after`` failures, the turn ends without running it.
    """

    def __init__(self, *, warn_after: int = 3, block_after: int = 5) -> None:
        super().__init__()
        self._warn_after = max(1, warn_after)
        self._block_after = max(self._warn_after, block_after)

    @staticmethod
    def _failed_streak(messages: list[BaseMessage], sig: str) -> int:
        results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
        streak = 0
        for msg in reversed(messages):
            if not isinstance(msg, AIMessage):
                continue
            for call in reversed(msg.tool_calls):
                result = results.get(call.get("id") or "")
                if result is None:
                    continue  # sibling call of the current step, not answered yet
                if _signature(call["name"], call.get("args")) != sig or not _is_error(result):
                    return streak
                streak += 1
        return streak

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        name = str(call.get("name", ""))
        messages = list((request.state or {}).get("messages") or [])
        streak = self._failed_streak(messages, _signature(name, call.get("args")))

        result = await handler(request)
        if (
            streak + 1 >= self._warn_after
            and isinstance(result, ToolMessage)
            and isinstance(result.content, str)
            and _is_error(result)
        ):
            result = result.model_copy(update={"content": result.content + _HINT.format(tool=name, n=streak + 1)})
        return result

    @hook_config(can_jump_to=["end"])
    async def aafter_model(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:
        messages = list(state.get("messages") or [])
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage):
            return None
        for call in last.tool_calls:
            streak = self._failed_streak(messages, _signature(call["name"], call.get("args")))
            if streak >= self._block_after:
                logger.warning("Loop guard ended turn: %s failed %d times identically", call["name"], streak)
                # Answer every pending call so the history stays valid for the next turn.
                replies: list[BaseMessage] = [
                    ToolMessage(
                        content="Error: [loop-guard] not executed; turn ended.",
                        tool_call_id=str(c.get("id", "")),
                        name=c["name"],
                        status="error",
                    )
                    for c in last.tool_calls
                ]
                replies.append(AIMessage(_STOP.format(tool=call["name"], n=streak)))
                return {"messages": replies, "jump_to": "end"}
        return None
