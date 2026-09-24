# tests/test_inbox.py
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from octop_harness.request import ChatRequest
from octop_harness.teams.inbox import HarnessAgentInboxManager, InboxMessage
from octop_harness.teams.processor import ReplyEvent, default_compose_followup


async def _wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise TimeoutError("condition not met")
        await asyncio.sleep(0.005)


class _Caller:
    """Records (agent_id, text) per call and returns a canned assistant reply."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.requests: list[tuple[str, ChatRequest]] = []

    async def call(self, agent_id: str, request: ChatRequest) -> dict[str, Any]:
        await asyncio.sleep(0.005)
        self.calls.append((agent_id, str(request.messages)))
        self.requests.append((agent_id, request))
        return {"messages": [{"role": "assistant", "content": f"{agent_id}:{request.messages}"}]}


class _Processor:
    def __init__(self) -> None:
        self.events: list[ReplyEvent] = []

    def compose_followup(self, msg: InboxMessage, *, result_text: str | None, error_text: str | None) -> str:
        return default_compose_followup(msg, result_text=result_text, error_text=error_text)

    async def on_reply(self, event: ReplyEvent) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_inbox_runs_target_then_source_then_callback() -> None:
    caller = _Caller()
    proc = _Processor()
    inbox = HarnessAgentInboxManager(call_agent=caller.call, processor=proc)

    mid = inbox.enqueue(
        target_agent_id="child",
        source_agent_id="main",
        source_thread_id="T-main",
        message="do research",
        user_id=1,
    )
    await asyncio.sleep(0.05)

    # target.call then source.call, in order.
    assert [c[0] for c in caller.calls] == ["child", "main"]
    # source reply lands on the source thread (checkpoint continuity).
    assert len(proc.events) == 1
    event = proc.events[0]
    assert event.inbox_id == mid
    assert event.status == "done"
    assert event.source_thread_id == "T-main"
    assert event.reply_text and event.reply_text.startswith("main:")
    # Terminal messages are pruned after on_reply to limit memory growth.
    assert inbox.get(mid) is None

    await inbox.shutdown()


@pytest.mark.asyncio
async def test_inbox_source_followup_forwards_session_key() -> None:
    caller = _Caller()
    proc = _Processor()
    inbox = HarnessAgentInboxManager(call_agent=caller.call, processor=proc)
    inbox.enqueue(
        target_agent_id="child",
        source_agent_id="main",
        source_thread_id="T-dm",
        message="do research",
        user_id=1,
        metadata={"session_key": "main:dashboard:1:dm"},
    )
    await _wait_until(lambda: len(proc.events) == 1)
    source_req = next(req for agent_id, req in caller.requests if agent_id == "main")
    assert source_req.thread_id == "T-dm"
    assert source_req.configurable is not None
    assert source_req.configurable["session_key"] == "main:dashboard:1:dm"
    await inbox.shutdown()


@pytest.mark.asyncio
async def test_inbox_target_failure_still_replies_failed() -> None:
    proc = _Processor()

    class _FailTargetCaller:
        async def call(self, agent_id: str, request: ChatRequest) -> dict[str, Any]:
            if agent_id == "child":
                raise RuntimeError("boom")
            return {"messages": [{"role": "assistant", "content": "sorry, it failed"}]}

    inbox = HarnessAgentInboxManager(call_agent=_FailTargetCaller().call, processor=proc)
    inbox.enqueue(
        target_agent_id="child",
        source_agent_id="main",
        source_thread_id="T-main",
        message="task",
        user_id=1,
    )
    await asyncio.sleep(0.05)

    assert len(proc.events) == 1
    assert proc.events[0].status == "failed"
    assert proc.events[0].error_text == "boom"
    # source agent still produced a user-facing fallback reply.
    assert proc.events[0].reply_text == "sorry, it failed"

    await inbox.shutdown()


@pytest.mark.asyncio
async def test_inbox_runs_different_targets_in_parallel() -> None:
    started: dict[str, float] = {}
    gate = asyncio.Event()

    class _SlowCaller:
        async def call(self, agent_id: str, request: ChatRequest) -> dict[str, Any]:
            if agent_id in {"a", "b"}:
                started[agent_id] = asyncio.get_running_loop().time()
                if len(started) == 2:
                    gate.set()
                await asyncio.wait_for(gate.wait(), timeout=1.0)
                await asyncio.sleep(0.01)
            return {"messages": [{"role": "assistant", "content": agent_id}]}

    proc = _Processor()
    inbox = HarnessAgentInboxManager(call_agent=_SlowCaller().call, processor=proc)
    inbox.enqueue(
        target_agent_id="a",
        source_agent_id="host",
        source_thread_id="T-main",
        message="one",
        user_id=1,
    )
    inbox.enqueue(
        target_agent_id="b",
        source_agent_id="host",
        source_thread_id="T-main",
        message="two",
        user_id=1,
    )
    await asyncio.wait_for(gate.wait(), timeout=1.0)
    assert set(started) == {"a", "b"}
    await asyncio.sleep(0.1)
    assert {event.target_agent_id for event in proc.events} == {"a", "b"}
    await inbox.shutdown()


@pytest.mark.asyncio
async def test_inbox_serializes_same_target() -> None:
    running = 0
    max_running = 0

    class _SerialCaller:
        async def call(self, agent_id: str, request: ChatRequest) -> dict[str, Any]:
            nonlocal running, max_running
            if agent_id == "child":
                running += 1
                max_running = max(max_running, running)
                await asyncio.sleep(0.03)
                running -= 1
            return {"messages": [{"role": "assistant", "content": agent_id}]}

    proc = _Processor()
    inbox = HarnessAgentInboxManager(call_agent=_SerialCaller().call, processor=proc)
    inbox.enqueue(
        target_agent_id="child",
        source_agent_id="host",
        source_thread_id="T-main",
        message="one",
        user_id=1,
    )
    inbox.enqueue(
        target_agent_id="child",
        source_agent_id="host",
        source_thread_id="T-main",
        message="two",
        user_id=1,
    )
    await _wait_until(lambda: len(proc.events) == 2)
    assert max_running == 1
    assert len(proc.events) == 2
    await inbox.shutdown()


@pytest.mark.asyncio
async def test_inbox_serializes_source_followup_per_thread() -> None:
    source_running = 0
    max_source = 0

    class _Caller:
        async def call(self, agent_id: str, request: ChatRequest) -> dict[str, Any]:
            nonlocal source_running, max_source
            if agent_id == "host":
                source_running += 1
                max_source = max(max_source, source_running)
                await asyncio.sleep(0.03)
                source_running -= 1
            else:
                await asyncio.sleep(0.01)
            return {"messages": [{"role": "assistant", "content": agent_id}]}

    proc = _Processor()
    inbox = HarnessAgentInboxManager(call_agent=_Caller().call, processor=proc)
    inbox.enqueue(
        target_agent_id="a",
        source_agent_id="host",
        source_thread_id="T-main",
        message="one",
        user_id=1,
    )
    inbox.enqueue(
        target_agent_id="b",
        source_agent_id="host",
        source_thread_id="T-main",
        message="two",
        user_id=1,
    )
    await _wait_until(lambda: len(proc.events) == 2)
    assert max_source == 1
    await inbox.shutdown()


@pytest.mark.asyncio
async def test_inbox_caps_global_concurrency() -> None:
    started = 0
    active = 0
    max_active = 0
    gate = asyncio.Event()

    class _Caller:
        async def call(self, agent_id: str, request: ChatRequest) -> dict[str, Any]:
            nonlocal started, active, max_active
            if agent_id.startswith("t"):
                started += 1
                active += 1
                max_active = max(max_active, active)
                await gate.wait()
                active -= 1
            return {"messages": [{"role": "assistant", "content": agent_id}]}

    proc = _Processor()
    inbox = HarnessAgentInboxManager(call_agent=_Caller().call, processor=proc, max_concurrency=2)
    for name in ("t1", "t2", "t3", "t4"):
        inbox.enqueue(
            target_agent_id=name,
            source_agent_id="host",
            source_thread_id="T-main",
            message=name,
            user_id=1,
        )
    await _wait_until(lambda: started == 2)
    await asyncio.sleep(0.03)
    assert started == 2
    assert max_active == 2
    gate.set()
    await _wait_until(lambda: len(proc.events) == 4)
    await inbox.shutdown()


@pytest.mark.asyncio
async def test_inbox_waiting_job_stays_queued_until_target_lock() -> None:
    release_first = asyncio.Event()
    first_started = asyncio.Event()

    class _Caller:
        async def call(self, agent_id: str, request: ChatRequest) -> dict[str, Any]:
            if agent_id == "child" and not first_started.is_set():
                first_started.set()
                await release_first.wait()
            return {"messages": [{"role": "assistant", "content": agent_id}]}

    proc = _Processor()
    inbox = HarnessAgentInboxManager(call_agent=_Caller().call, processor=proc)
    first = inbox.enqueue(
        target_agent_id="child",
        source_agent_id="host",
        source_thread_id="T-main",
        message="one",
        user_id=1,
    )
    second = inbox.enqueue(
        target_agent_id="child",
        source_agent_id="host",
        source_thread_id="T-main",
        message="two",
        user_id=1,
    )
    await first_started.wait()
    waiting = inbox.get(second)
    assert waiting is not None
    assert waiting.status == "queued"
    inbox.cancel(second)
    release_first.set()
    await _wait_until(lambda: len(proc.events) == 2)
    statuses = {event.inbox_id: event.status for event in proc.events}
    assert statuses[first] == "done"
    assert statuses[second] == "cancelled"
    await inbox.shutdown()


def test_inbox_rejects_non_positive_concurrency() -> None:
    proc = _Processor()

    async def _call(agent_id: str, request: ChatRequest) -> dict[str, Any]:
        return {"messages": [{"role": "assistant", "content": agent_id}]}

    with pytest.raises(ValueError, match="max_concurrency"):
        HarnessAgentInboxManager(call_agent=_call, processor=proc, max_concurrency=0)
