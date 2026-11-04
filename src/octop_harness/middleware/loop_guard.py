"""Middleware that breaks tool-call loops before they burn the recursion budget."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

logger = logging.getLogger(__name__)


def _current_thread_id() -> str:
    """Return the LangGraph ``thread_id`` of the currently running graph, or ``"unknown"``."""
    try:
        from langgraph.config import get_config

        cfg = get_config()
        if isinstance(cfg, dict):
            configurable = cfg.get("configurable")
            if isinstance(configurable, dict):
                tid = configurable.get("thread_id")
                if tid:
                    return str(tid)
    except (ImportError, RuntimeError):
        # ImportError: langgraph.config not available.
        # RuntimeError: not inside a graph run (e.g. middleware unit tests).
        pass
    return "unknown"


def _args_signature(tool_name: str, args: dict[str, Any]) -> str:
    """Return a stable hash of ``(tool_name, args)`` for loop detection."""
    try:
        payload = json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        # Fallback for args that contain non-JSON-serializable objects.
        payload = f"tool={tool_name}|args={args!r}"
    raw = f"{tool_name}|{payload}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class LoopGuardMiddleware(AgentMiddleware[Any, Any]):
    """Detects repeated identical tool calls and short-circuits them with an error.

    When an agent keeps invoking the same tool with the same arguments because
    the model cannot break out of the loop, nothing in the normal middleware
    chain intervenes — the run only ends once LangGraph's ``recursion_limit``
    fires a :class:`GraphRecursionError` and the whole turn fails.

    This middleware tracks a per-thread consecutive streak of identical tool
    calls. Once the streak exceeds *max_repeats*, the next identical call is
    short-circuited with a ``ToolMessage.status="error"`` that tells the model
    to stop retrying and change strategy (issue octop-harness#61).

    Parameters
    ----------
    enabled:
        Master switch — set ``False`` to disable without removing the middleware.
    max_repeats:
        How many consecutive identical ``(tool_name, args)`` pairs the model
        may issue before the guard fires. Default ``3`` means the model gets
        three attempts (one fresh call + two retries) before the fourth is
        blocked.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_repeats: int = 3,
    ) -> None:
        super().__init__()
        self._enabled = enabled
        self._max_repeats = max(1, int(max_repeats))
        # thread_id -> (signature, consecutive_count)
        self._streak: dict[str, tuple[str, int]] = defaultdict(lambda: ("", 0))
        # thread_id -> total calls seen (for cleanup)
        self._total_seen: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def configure(self, *, enabled: bool | None = None, max_repeats: int | None = None) -> None:
        if enabled is not None:
            self._enabled = enabled
        if max_repeats is not None:
            self._max_repeats = max(1, int(max_repeats))

    def _reset_streak(self, thread_id: str) -> None:
        self._streak.pop(thread_id, None)
        self._total_seen.pop(thread_id, None)
        # Occasional cleanup of stale thread entries.
        if len(self._streak) > 1024:
            stale = [
                tid for tid in list(self._streak) if self._total_seen.get(tid, 0) <= 1
            ]
            for tid in stale[:256]:
                self._streak.pop(tid, None)
                self._total_seen.pop(tid, None)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        if not self._enabled:
            return await handler(request)

        tool_call = request.tool_call
        tool_name = str(tool_call.get("name", "") or "")
        if not tool_name:
            return await handler(request)

        raw_args = tool_call.get("args") or {}
        args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}

        thread_id = _current_thread_id()
        signature = _args_signature(tool_name, args)

        with self._lock:
            prev_sig, prev_count = self._streak.get(thread_id, ("", 0))
            self._total_seen[thread_id] = self._total_seen.get(thread_id, 0) + 1

            if signature == prev_sig:
                streak = prev_count + 1
            else:
                streak = 1
            self._streak[thread_id] = (signature, streak)

            should_block = streak > self._max_repeats
            tool_call_id = str(tool_call.get("id", "") or "")

            if should_block:
                logger.warning(
                    "LoopGuard: thread=%s tool=%s repeated %d+ times with identical args — short-circuiting",
                    thread_id,
                    tool_name,
                    self._max_repeats,
                )
                # Reset streak so the next *different* call is counted fresh.
                self._reset_streak(thread_id)
                return ToolMessage(
                    content=(
                        f"You have called tool `{tool_name}` with the same arguments "
                        f"{self._max_repeats + 1} times in a row without success. "
                        f"Do NOT call the same tool with the same arguments again. "
                        f"Pick a different tool, change your arguments, or answer "
                        f"the user with what you have learned so far."
                    ),
                    tool_call_id=tool_call_id,
                    status="error",
                )

        return await handler(request)
