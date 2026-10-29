"""Middleware that breaks tool-call loops by detecting consecutive duplicates."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from collections import deque
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)


def _args_signature(tool_call: dict[str, Any]) -> str:
    """Return a stable short hash over the (tool_name, normalized_args) pair."""
    name = str(tool_call.get("name", ""))
    raw_args = tool_call.get("args") or {}
    if not isinstance(raw_args, dict):
        raw_args = {}
    canonical = json.dumps(raw_args, sort_keys=True, default=str, ensure_ascii=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{name}|{digest}"


class LoopGuardMiddleware(AgentMiddleware[Any, Any]):
    """Detect and break infinite tool-call loops.

    Every time the agent invokes a tool we compute a stable signature over
    ``(tool_name, args)``. If the same signature has been observed more than
    ``max_consecutive`` times in the recent history (default window = 6 calls),
    the middleware short-circuits and returns a ``ToolMessage`` with an
    explicit error instructing the model to stop retrying with identical
    parameters and pick a different strategy instead.

    This does not abort the run; the model still gets a reply it can reason
    about. The guard is intentionally stateful per middleware instance so a
    fresh agent run starts with a clean slate.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_consecutive: int = 2,
        window: int = 6,
    ) -> None:
        super().__init__()
        if max_consecutive < 1:
            raise ValueError("max_consecutive must be >= 1")
        if window < max_consecutive:
            raise ValueError("window must be >= max_consecutive")
        self._enabled = enabled
        self._max_consecutive = max_consecutive
        self._window = window
        self._history: deque[str] = deque(maxlen=window)

    # ------------------------------------------------------------------
    # Public config
    # ------------------------------------------------------------------

    def configure(
        self,
        *,
        enabled: bool | None = None,
        max_consecutive: int | None = None,
    ) -> None:
        if enabled is not None:
            self._enabled = enabled
        if max_consecutive is not None:
            self._max_consecutive = max_consecutive

    def reset(self) -> None:
        """Clear the signature history between runs."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Hook: intercept tool calls
    # ------------------------------------------------------------------

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if not self._enabled:
            return await handler(request)

        tool_call = request.tool_call
        signature = _args_signature(tool_call)
        self._history.append(signature)

        recent = list(self._history)
        repeat = self._count_consecutive_tail(recent, signature)
        if repeat <= self._max_consecutive:
            return await handler(request)

        tool_name = str(tool_call.get("name", ""))
        tool_call_id = str(tool_call.get("id", ""))
        message = (
            f"Loop guard: tool '{tool_name}' has been invoked {repeat} times in a row "
            f"with identical arguments. Do NOT retry this tool with the same parameters. "
            f"Pick a different tool, change the arguments, or answer the user directly."
        )
        logger.warning(
            "Loop guard triggered for %s (repeat=%d, window=%d)",
            tool_name,
            repeat,
            len(recent),
        )
        return ToolMessage(
            content=message,
            tool_call_id=tool_call_id,
            status="error",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_consecutive_tail(history: list[str], signature: str) -> int:
        count = 0
        for entry in reversed(history):
            if entry == signature:
                count += 1
            else:
                break
        return count

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Reset history at the end of each turn so the next turn starts fresh."""
        self._history.clear()
        return None

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)


__all__ = ["LoopGuardMiddleware"]
