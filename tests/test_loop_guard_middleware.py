"""Tests for ``LoopGuardMiddleware`` on a real ``create_agent`` graph."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from octop_harness.middleware.loop_guard import LoopGuardMiddleware


class _Model(GenericFakeChatModel):
    def bind_tools(self, *_: Any, **__: Any) -> Any:
        return self


def _run(args_per_call: list[dict[str, Any]]) -> tuple[list[Any], list[dict[str, Any]]]:
    executed: list[dict[str, Any]] = []

    @tool
    def flaky(a: int) -> str:
        """Always fails."""
        executed.append({"a": a})
        return "Error: boom"

    replies = iter(
        [AIMessage("", tool_calls=[{"name": "flaky", "args": a, "id": f"c{i}"}]) for i, a in enumerate(args_per_call)]
        + [AIMessage("gave up")]
    )
    agent = create_agent(_Model(messages=replies), [flaky], middleware=[LoopGuardMiddleware()])
    out = asyncio.run(agent.ainvoke({"messages": [HumanMessage("go")]}, {"recursion_limit": 50}))
    return out["messages"], executed


def test_identical_failures_end_the_turn() -> None:
    msgs, executed = _run([{"a": 1}] * 10)
    assert len(executed) == 5  # 6th identical call is never run
    assert "Stopped" in msgs[-1].content
    hints = [m for m in msgs if isinstance(m, ToolMessage) and "Do NOT retry" in str(m.content)]
    assert hints  # warned before ending
    pending = {c["id"] for m in msgs if isinstance(m, AIMessage) for c in m.tool_calls}
    assert pending == {m.tool_call_id for m in msgs if isinstance(m, ToolMessage)}


def test_changed_args_reset_streak() -> None:
    msgs, executed = _run([{"a": i} for i in range(6)] + [{"a": 0}])
    assert len(executed) == 7
    assert "Stopped" not in str(msgs[-1].content)
