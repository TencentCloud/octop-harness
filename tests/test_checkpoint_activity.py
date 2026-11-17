"""Activity leases must span graph execution and its final checkpoint flush."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from octop_harness.memory import checkpoint_activity
from octop_harness.protocols.langgraph import LangGraphProtocol

pytestmark = pytest.mark.filterwarnings("ignore:The v3 streaming protocol on Pregel is experimental")


class ActivitySaver(InMemorySaver):
    def __init__(self) -> None:
        super().__init__()
        self.active: list[str] = []
        self.released = 0

    @contextmanager
    def checkpoint_activity(self, thread_id: str) -> Iterator[None]:
        self.active.append(thread_id)
        try:
            yield
        finally:
            self.active.remove(thread_id)
            self.released += 1

    async def aput(self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any) -> Any:
        assert config["configurable"]["thread_id"] in self.active
        return await super().aput(config, checkpoint, metadata, new_versions)


def _graph(saver: ActivitySaver, node: Any) -> Any:
    builder = StateGraph(MessagesState)
    builder.add_node("reply", node)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    return builder.compile(checkpointer=saver)


@pytest.mark.parametrize("entry", ["call", "stream", "resume", "v3"])
async def test_graph_activity_covers_final_checkpoint(entry: str) -> None:
    saver = ActivitySaver()

    async def node(state: MessagesState) -> dict[str, Any]:
        assert saver.active == ["thread"]
        return {"messages": [AIMessage(content="finished")]}

    graph = _graph(saver, node)
    protocol = LangGraphProtocol(graph=graph)
    config = {"configurable": {"thread_id": "thread"}}
    messages = [HumanMessage(content="hello")]
    if entry == "call":
        await protocol.call(messages, config)
    elif entry == "stream":
        _ = [chunk async for chunk in protocol.stream(messages, config)]
    elif entry == "v3":
        run = await protocol.stream_events(messages, config)
        assert saver.active == []  # Pull-driven: an unused stream holds no lease.
        async with run:
            await run.output()
    else:
        # An empty resume is still a graph operation and must release its guard.
        await protocol.call(messages, config)
        saver.released = 0
        _ = [chunk async for chunk in protocol.resume_stream("thread", [], {})]
    assert saver.active == []
    assert saver.released == 1
    assert graph.get_state(config).values["messages"][-1].content == "finished"


@pytest.mark.parametrize("entry", ["call", "stream", "v3"])
async def test_cancellation_and_abort_release_activity_after_graph_closes(entry: str) -> None:
    saver = ActivitySaver()
    started = asyncio.Event()
    closed = asyncio.Event()

    async def node(state: MessagesState) -> dict[str, Any]:
        started.set()
        try:
            await asyncio.Future()
        finally:
            assert saver.active == ["thread"]
            closed.set()
        return {}

    protocol = LangGraphProtocol(graph=_graph(saver, node))
    config = {"configurable": {"thread_id": "thread"}}
    messages = [HumanMessage(content="hello")]
    run = None
    if entry == "call":
        task = asyncio.create_task(protocol.call(messages, config))
    elif entry == "stream":

        async def drain() -> None:
            async for _ in protocol.stream(messages, config):
                pass

        task = asyncio.create_task(drain())
    else:
        run = await protocol.stream_events(messages, config)
        task = asyncio.create_task(run.output())
    await asyncio.wait_for(started.wait(), 5)
    assert saver.active == ["thread"]
    if run is None:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await run.abort()
        await task
    assert closed.is_set()
    assert saver.active == []
    assert saver.released == 1


async def test_cancellation_during_acquisition_joins_worker_and_releases() -> None:
    entered = threading.Event()
    proceed = threading.Event()
    active: list[str] = []

    class BlockingStore:
        @contextmanager
        def checkpoint_activity(self, thread_id: str) -> Iterator[None]:
            entered.set()
            assert proceed.wait(5)
            active.append(thread_id)
            try:
                yield
            finally:
                active.remove(thread_id)

    async def operation() -> None:
        async with checkpoint_activity(BlockingStore(), "thread"):
            pytest.fail("cancelled request must not enter the graph")

    task = asyncio.create_task(operation())
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert active == []


async def test_legacy_checkpointer_is_compatible() -> None:
    async with checkpoint_activity(InMemorySaver(), "thread"):
        pass


async def test_host_guard_and_history_state_operations() -> None:
    from octop_harness.agent import HarnessAgent

    saver = ActivitySaver()
    graph = _graph(saver, lambda state: {"messages": [AIMessage(content="finished")]})
    agent = object.__new__(HarnessAgent)
    agent._checkpointer_instance = saver
    agent._graph = graph
    async with agent.checkpoint_activity("thread"):
        await graph.aupdate_state(
            {"configurable": {"thread_id": "thread"}}, {"messages": [HumanMessage(content="seed")]}
        )
    await agent.aappend_messages("thread", [AIMessage(content="appended")])
    assert (await agent.aget_history("thread"))[-1].content == "appended"
    assert await agent.adelete_thread("thread")
    assert saver.active == []
    assert saver.released == 4


@pytest.mark.parametrize("entry", ["stream", "v3"])
async def test_stream_pulls_keep_the_same_async_activity_context(entry: str) -> None:
    scope: ContextVar[str] = ContextVar("test_checkpoint_scope", default="")

    class ScopedSaver(ActivitySaver):
        @asynccontextmanager
        async def acheckpoint_activity(self, thread_id: str):
            token = scope.set(thread_id)
            try:
                with self.checkpoint_activity(thread_id):
                    yield
            finally:
                scope.reset(token)

    saver = ScopedSaver()

    async def reply(state: MessagesState) -> dict[str, Any]:
        assert scope.get() == "thread"
        return {"messages": [AIMessage(content="finished")]}

    protocol = LangGraphProtocol(graph=_graph(saver, reply))
    config = {"configurable": {"thread_id": "thread"}}
    if entry == "v3":
        async with await protocol.stream_events([HumanMessage(content="hello")], config) as run:
            await run.output()
    else:
        stream = protocol.stream([HumanMessage(content="hello")], config)

        async def pull():
            return await anext(stream)

        try:
            while True:
                # Harness' cancellation poller, like v3, creates a fresh task
                # for each pull rather than draining in one consumer task.
                await asyncio.create_task(pull())
        except StopAsyncIteration:
            pass
        finally:
            await stream.aclose()
    assert scope.get() == ""
    assert saver.active == []
    assert saver.released == 1


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_cancel_during_final_flush_keeps_activity_until_memory_io_finishes(tmp_path, monkeypatch, backend):
    from octop_memory import Memory

    if not callable(getattr(Memory, "acheckpoint_activity", None)):
        pytest.skip("requires octop-memory's async checkpoint activity API")
    if backend == "postgres":
        dsn = os.environ.get("TEST_POSTGRES_DSN")
        if not dsn:
            pytest.skip("set TEST_POSTGRES_DSN to an isolated PostgreSQL test database")
        backend_config = {"dsn": dsn}
    else:
        backend_config = {"db_path": str(tmp_path / "memory.sqlite")}
    memory = Memory("cancel_flush", backend=backend, backend_config=backend_config)
    thread_id = f"cancel-flush-{uuid4().hex}"
    entered = threading.Event()
    proceed = threading.Event()
    original = memory.put

    def blocked_put(*args, **kwargs):
        entered.set()
        assert proceed.wait(5), "test did not release the final checkpoint write"
        return original(*args, **kwargs)

    monkeypatch.setattr(memory, "put", blocked_put)
    graph = _graph(memory, lambda state: {"messages": [AIMessage(content="finished")]})
    config = {"configurable": {"thread_id": thread_id}}
    task = asyncio.create_task(LangGraphProtocol(graph=graph).call([HumanMessage(content="hello")], config))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0.02)
        task.cancel()  # Repeated cancellation must not abandon Memory's drain.
        await asyncio.sleep(0.02)
        assert not task.done()
        if backend == "postgres":

            def can_maintain():
                with memory.checkpoint_maintenance(thread_id) as acquired:
                    return acquired

            assert not await asyncio.to_thread(can_maintain)
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await graph.aget_state(config)).values["messages"][-1].content == "finished"
    finally:
        proceed.set()
        with suppress(asyncio.CancelledError):
            await task
        await asyncio.to_thread(memory.delete_thread, thread_id)
        pool = getattr(memory, "_checkpointer_pool", None)
        if pool is not None:
            pool.close()
        elif memory._checkpointer is not None:
            memory._checkpointer.conn.close()
        memory.backend.close()


async def test_postgres_retention_defers_an_active_historical_fork() -> None:
    from octop_memory import Memory
    from octop_memory.pipeline.lifecycle.checkpoint_gc import prune_checkpoints

    dsn = os.environ.get("TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set TEST_POSTGRES_DSN to an isolated PostgreSQL test database")
    if not callable(getattr(Memory, "checkpoint_maintenance", None)):
        pytest.skip("requires octop-memory's coordinated checkpoint API")
    thread_id = f"activity-{uuid4().hex}"
    memory = Memory("activity_runtime", backend="postgres", backend_config={"dsn": dsn})
    maintainer = Memory("activity_maintenance", backend="postgres", backend_config={"dsn": dsn})
    started = asyncio.Event()
    proceed = asyncio.Event()
    task: asyncio.Task[Any] | None = None

    async def reply(state: MessagesState) -> dict[str, Any]:
        if state["messages"][-1].content == "block":
            started.set()
            await proceed.wait()
        return {"messages": [AIMessage(content="finished")]}

    def prune_if_idle() -> bool:
        with maintainer.checkpoint_maintenance(thread_id) as acquired:
            if acquired:
                prune_checkpoints(maintainer, thread_id=thread_id, offline=True, keep_last=1, skip_pending=True)
            return acquired

    graph = _graph(memory, reply)
    protocol = LangGraphProtocol(graph=graph)
    config = {"configurable": {"thread_id": thread_id}}
    try:
        first = None
        for text in ("one", "two", "three"):
            await protocol.call([HumanMessage(content=text)], config)
            if first is None:
                first = (await graph.aget_state(config)).config
        assert first is not None
        task = asyncio.create_task(protocol.call([HumanMessage(content="block")], first))
        await asyncio.wait_for(started.wait(), 10)
        assert not await asyncio.to_thread(prune_if_idle)
        proceed.set()
        await task
        before = (await graph.aget_state(config)).values
        assert await asyncio.to_thread(prune_if_idle)
        assert (await graph.aget_state(config)).values == before
    finally:
        proceed.set()
        if task is not None:
            await task
        await asyncio.to_thread(memory.delete_thread, thread_id)
        for store in (memory, maintainer):
            pool = getattr(store, "_checkpointer_pool", None)
            if pool is not None:
                pool.close()
            store.backend.close()
