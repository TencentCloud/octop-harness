"""Compare persisted state using a real Harness and CompactSqliteSaver."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, message_to_dict
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from octop_memory import Memory
from pydantic import PrivateAttr

from octop_harness.agent import HarnessAgent
from octop_harness.config import HarnessAgentConfig, ModelConfig, ProviderConfig
from octop_harness.llm.factory import ChatModelFactory
from octop_harness.request import ChatRequest

pytestmark = pytest.mark.filterwarnings("ignore:Direct attribute assignment to ModelRequest.model:DeprecationWarning")


def _message(message: BaseMessage) -> dict[str, Any]:
    result = message_to_dict(message)
    result["data"] = {**result["data"], "id": None}
    additional = result["data"]["additional_kwargs"]
    result["data"]["additional_kwargs"] = {k: v for k, v in additional.items() if k != "checkpoint_ts"}
    return result


class ProbeModel(BaseChatModel):
    """Deterministic local model with explicit synchronization for cancellation."""

    _inputs: list[list[dict[str, Any]]] = PrivateAttr(default_factory=list)
    _model_started: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)

    @property
    def _llm_type(self) -> str:
        return "checkpoint-probe"

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        return self

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError

    async def _agenerate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self._inputs.append([_message(message) for message in messages])
        last = messages[-1]
        text = str(last.text)
        if isinstance(last, ToolMessage):
            parts = last.tool_call_id.split("-")
            if parts[0] in ("chain", "crash") and int(parts[1]) < int(parts[2]):
                step, total = int(parts[1]) + 1, int(parts[2])
                call_id = f"{parts[0]}-{step}-{total}"
                value = "block" if parts[0] == "crash" else str(step)
                reply = self._tool_call(call_id, value)
            else:
                reply = AIMessage(content=f"finished: {text}")
        elif text.startswith("CHAIN "):
            reply = self._tool_call(f"chain-1-{int(text.split()[1])}", "1")
        elif text == "CRASH":
            reply = self._tool_call("crash-1-2", "external")
        elif text == "BLOCK_TOOL":
            reply = self._tool_call("blocked-tool", "block")
        elif text == "BLOCK_MODEL":
            self._model_started.set()
            await asyncio.Future()
            raise AssertionError("the test should cancel this call")
        elif text == "MODEL_ERROR":
            raise RuntimeError("probe model failed")
        elif text == "TOOL_ERROR":
            reply = self._tool_call("failed-tool", "fail")
        elif text in ("APPROVAL", "CHILD_APPROVAL"):
            reply = self._tool_call("approval-tool", "approve")
        elif text in ("SUBAGENT", "SUBAGENT_APPROVAL"):
            reply = AIMessage(
                content="delegating",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "CHILD_APPROVAL" if text.endswith("APPROVAL") else "CHILD",
                            "subagent_type": "worker",
                        },
                        "id": "delegation",
                    }
                ],
            )
        else:
            reply = AIMessage(content=f"reply: {text}")
        return ChatResult(generations=[ChatGeneration(message=reply)])

    @staticmethod
    def _tool_call(call_id: str, value: str) -> AIMessage:
        return AIMessage(
            content="calling tool",
            tool_calls=[
                {
                    "name": "approval_tool" if value == "approve" else "probe_tool",
                    "args": {"value": value},
                    "id": call_id,
                }
            ],
        )


@asynccontextmanager
async def open_probe(
    db: Path,
    workspace: Path,
    *,
    approval: str | None = None,
    subagent: bool = False,
    postgres_dsn: str | None = None,
) -> AsyncIterator[tuple[HarnessAgent, ProbeModel, Memory, asyncio.Event]]:
    db.parent.mkdir(parents=True, exist_ok=True)
    for folder in ("_builtin_skills", "skills"):
        (workspace / folder).mkdir(parents=True, exist_ok=True)
    memory = (
        Memory("durability_probe", backend="postgres", backend_config={"dsn": postgres_dsn})
        if postgres_dsn
        else Memory("durability_probe", backend_config={"db_path": str(db)})
    )
    model = ProbeModel()
    tool_started = asyncio.Event()

    @tool
    async def probe_tool(value: str) -> str:
        """Return a test value, or block until the invocation is cancelled."""
        if value == "fail":
            raise RuntimeError("probe tool failed")
        if value == "block":
            tool_started.set()
            await asyncio.Future()
        if value == "external":
            (db.parent / "tool-effect").write_text("applied", encoding="utf-8")
        return f"tool: {value}"

    @tool
    def approval_tool(value: str) -> str:
        """Return a test value after approval."""
        return f"tool: {value}"

    tools = [probe_tool, approval_tool]
    interrupts: dict[str, Any] = {"approval_tool": True}
    child = {
        "name": "worker",
        "description": "Perform a test task",
        "system_prompt": "Handle the test task.",
        "tools": tools,
        **({"interrupt_on": interrupts} if approval == "child" else {}),
    }
    config = HarnessAgentConfig(
        name="durability-probe",
        workspace_dir=workspace,
        backend={"type": "filesystem", "root_dir": str(workspace), "virtual_mode": True},
        providers=[
            ProviderConfig(
                id="p", base_url="http://127.0.0.1:1", api_key="test", models=[ModelConfig(id="m", input=["text"])]
            )
        ],
        default_model="p/m",
        checkpointer=memory,
        memory_enabled=False,
        session_log_enabled=False,
        bootstrap_enabled=False,
        model_retry_enabled=False,
        web_search_tools=False,
        tools=tools,
        subagents=[child] if subagent else None,  # type: ignore[list-item]
        subagents_auto_load=False,
        interrupt_on=interrupts if approval == "parent" else None,
    )
    factory = ChatModelFactory(config.providers, agent_config=config)
    with patch.object(factory, "get_chat_model", return_value=model):
        agent = HarnessAgent(config, model_factory=factory)
        try:
            yield agent, model, memory, tool_started
        finally:
            await agent.aclose()
            await factory.aclose()
            if memory._checkpointer is not None:
                memory._checkpointer.conn.close()
            memory.backend.close()


def request(thread_id: str, text: str) -> ChatRequest:
    return ChatRequest(messages=text, thread_id=thread_id, user="1", source="dashboard", recursion_limit=512)


async def state(agent: HarnessAgent, thread_id: str) -> dict[str, Any]:
    saved = await agent.graph.aget_state({"configurable": {"thread_id": thread_id}})
    values = dict(saved.values)
    values["messages"] = [_message(message) for message in values.get("messages", [])]
    return {"values": values, "next": saved.next, "interrupts": [item.value for item in saved.interrupts]}


def rows(memory: Memory, thread_id: str) -> tuple[int, int, int]:
    conn = memory._checkpointer.conn
    checkpoints, subgraphs = conn.execute(
        "SELECT count(*), coalesce(sum(checkpoint_ns LIKE 'tools:%'), 0) FROM checkpoints WHERE thread_id=?",
        (thread_id,),
    ).fetchone()
    writes = conn.execute("SELECT count(*) FROM writes WHERE thread_id=?", (thread_id,)).fetchone()[0]
    return checkpoints, writes, subgraphs


async def _stream(agent: HarnessAgent, thread_id: str, text: str) -> list[dict[str, Any]]:
    return [event async for event in agent.stream(request(thread_id, text))]


async def test_multiturn_tool_and_delta_snapshot_replay(tmp_path, monkeypatch):
    results = {}
    for mode in ("async", "exit"):
        monkeypatch.setenv("OCTOP_CHECKPOINT_DURABILITY", mode)
        async with open_probe(tmp_path / mode / "memory.sqlite", tmp_path / "workspace") as (agent, model, memory, _):
            counts = []
            states = []
            for turn in ("first", "CHAIN 3", "CHAIN 28", "follow-up"):
                before = rows(memory, "thread")
                await agent.call(request("thread", turn))
                after = rows(memory, "thread")
                counts.append(tuple(a - b for a, b in zip(after, before, strict=True)))
                states.append(await state(agent, "thread"))
            history = [_message(message) for message in await agent.aget_history("thread", limit=200)]
            assert history == states[-1]["values"]["messages"]
            assert len(history) > 50
            results[mode] = (states, model._inputs, history, counts)
    print("multiturn deltas (checkpoint, writes, tools namespace):", {k: v[-1] for k, v in results.items()})
    assert results["exit"][:3] == results["async"][:3]
    # The first exit can also create LangGraph's delta anchor for a new thread.
    assert [turn[0] for turn in results["exit"][-1]] == [2, 1, 1, 1]


@pytest.mark.parametrize("turn", ["BLOCK_TOOL", "BLOCK_MODEL", "interject"])
async def test_cancel_and_follow_up(tmp_path, monkeypatch, turn):
    results = {}
    for mode in ("async", "exit"):
        monkeypatch.setenv("OCTOP_CHECKPOINT_DURABILITY", mode)
        async with open_probe(tmp_path / mode / "memory.sqlite", tmp_path / "workspace") as (agent, model, _, tool):
            await agent.call(request("thread", "first"))
            blocked = "BLOCK_MODEL" if turn == "BLOCK_MODEL" else "BLOCK_TOOL"
            task = asyncio.create_task(_stream(agent, "thread", blocked))
            started = model._model_started if blocked == "BLOCK_MODEL" else tool
            try:
                await asyncio.wait_for(started.wait(), timeout=10)
            finally:
                agent.cancel("thread")
                await asyncio.wait_for(task, timeout=10)
            paused = None if turn == "interject" else await state(agent, "thread")
            await agent.call(request("thread", "immediate new input" if turn == "interject" else "follow-up"))
            results[mode] = (paused, await state(agent, "thread"), model._inputs)
    assert results["exit"] == results["async"]


@pytest.mark.parametrize("kind", ["TOOL_ERROR", "MODEL_ERROR"])
async def test_error_state_and_next_turn(tmp_path, monkeypatch, kind):
    results = {}
    for mode in ("async", "exit"):
        monkeypatch.setenv("OCTOP_CHECKPOINT_DURABILITY", mode)
        async with open_probe(tmp_path / mode / "memory.sqlite", tmp_path / "workspace") as (agent, model, memory, _):
            await agent.call(request("thread", "first"))
            before = rows(memory, "thread")
            with pytest.raises(RuntimeError, match=r"probe (tool|model) failed"):
                await _stream(agent, "thread", kind)
            assert rows(memory, "thread")[0] > before[0]
            failed = await state(agent, "thread")
            await agent.call(request("thread", "after the error"))
            results[mode] = (failed, await state(agent, "thread"), model._inputs)
    assert results["exit"] == results["async"]


@pytest.mark.parametrize(("approval", "decision"), [("parent", "approve"), ("parent", "reject"), ("child", "approve")])
async def test_approval_after_reopening_agent(tmp_path, monkeypatch, approval, decision):
    results = {}
    for mode in ("async", "exit"):
        monkeypatch.setenv("OCTOP_CHECKPOINT_DURABILITY", mode)
        db = tmp_path / mode / "memory.sqlite"
        workspace = tmp_path / "workspace"
        subagent = approval == "child"
        async with open_probe(db, workspace, approval=approval, subagent=subagent) as (agent, model, memory, _):
            await agent.call(request("thread", "first"))
            events = await _stream(agent, "thread", "SUBAGENT_APPROVAL" if subagent else "APPROVAL")
            assert any(event["type"] == "hitl_required" for event in events)
            paused = await state(agent, "thread")
            assert paused["next"] and paused["interrupts"]
            initial_inputs = model._inputs
            paused_rows = rows(memory, "thread")
        async with open_probe(db, workspace, approval=approval, subagent=subagent) as (agent, model, memory, _):
            assert await state(agent, "thread") == paused
            decisions = [{"type": decision, **({"message": "denied"} if decision == "reject" else {})}]
            events = [event async for event in agent.resume_hitl("thread", decisions)]
            assert all(event["type"] != "hitl_required" for event in events)
            finished = await state(agent, "thread")
            assert not finished["next"] and not finished["interrupts"]
            assert finished["values"]["messages"][-1]["type"] == "ai"
            results[mode] = (paused, finished, initial_inputs + model._inputs)
            print("approval rows", approval, decision, mode, paused_rows, rows(memory, "thread"))
    assert results["exit"] == results["async"]


async def test_delegated_tool_preserves_parent_state(tmp_path, monkeypatch):
    results = {}
    for mode in ("async", "exit"):
        monkeypatch.setenv("OCTOP_CHECKPOINT_DURABILITY", mode)
        async with open_probe(tmp_path / mode / "memory.sqlite", tmp_path / "workspace", subagent=True) as (
            agent,
            model,
            memory,
            _,
        ):
            await _stream(agent, "thread", "SUBAGENT")
            saved = await state(agent, "thread")
            assert "reply: CHILD" in saved["values"]["messages"][-1]["data"]["content"]
            results[mode] = (saved, model._inputs)
            print("delegation rows", mode, rows(memory, "thread"))
    assert results["exit"] == results["async"]


@pytest.mark.parametrize("mode", ["async", "exit"])
@pytest.mark.filterwarnings("ignore:The v3 streaming protocol on Pregel is experimental")
async def test_v3_stream_flushes_checkpoint_before_output_returns(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("OCTOP_CHECKPOINT_DURABILITY", mode)
    async with open_probe(tmp_path / "memory.sqlite", tmp_path / "workspace") as (agent, _, memory, _):
        stream = await agent.stream_events(request("thread", "first"))
        output = await stream.output()
        assert output["messages"][-1].content == "reply: first"
        assert (await agent.aget_history("thread"))[-1].content == "reply: first"
        checkpoint_count = rows(memory, "thread")[0]
        if mode == "exit":
            assert checkpoint_count == 2
        else:
            assert checkpoint_count > 2


@pytest.mark.parametrize("mode", ["async", "exit"])
async def test_pruning_preserves_state_and_pending_approval(tmp_path, monkeypatch, mode):
    import inspect

    from octop_memory.pipeline.lifecycle.checkpoint_gc import prune_checkpoints

    # Dependency-aware pruning with skip_pending ships in octop-memory 1.0.4.
    if "skip_pending" not in inspect.signature(prune_checkpoints).parameters:
        pytest.skip("requires octop-memory>=1.0.4")
    monkeypatch.setenv("OCTOP_CHECKPOINT_DURABILITY", mode)
    async with open_probe(tmp_path / "memory.sqlite", tmp_path / "workspace", approval="parent") as (
        agent,
        _,
        memory,
        _,
    ):
        for index in range(75):
            await agent.call(request("done", f"turn {index}"))
            await agent.call(request("pending", f"turn {index}"))
        events = await _stream(agent, "pending", "APPROVAL")
        assert any(event["type"] == "hitl_required" for event in events)
        before = {tid: await state(agent, tid) for tid in ("done", "pending")}
        pending_rows = rows(memory, "pending")
        stats = prune_checkpoints(memory, keep_last=20, skip_pending=True)
        print("pruning", mode, stats.checkpoints_deleted, stats.writes_deleted, stats.parent_prunes_skipped)
        if mode == "exit":
            assert stats.checkpoints_deleted > 0
        assert stats.parent_prunes_skipped >= 1
        assert rows(memory, "pending") == pending_rows
        assert {tid: await state(agent, tid) for tid in before} == before


@pytest.fixture
def retention_dsn(backend):
    if backend == "sqlite":
        yield None
        return
    dsn = os.environ.get("TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set TEST_POSTGRES_DSN to a disposable PostgreSQL database")
    psycopg = pytest.importorskip("psycopg")
    pytest.importorskip("langgraph.checkpoint.postgres")
    schema = "checkpoint_retention_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(psycopg.sql.SQL("CREATE SCHEMA {}").format(psycopg.sql.Identifier(schema)))
        try:
            yield psycopg.conninfo.make_conninfo(dsn, options=f"-csearch_path={schema}")
        finally:
            conn.execute(psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(psycopg.sql.Identifier(schema)))


@pytest.mark.parametrize("approval", ["parent", "child"])
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_pruning_snapshots_messages_and_preserves_approval(
    tmp_path, monkeypatch, approval, backend, retention_dsn
):
    import inspect

    from octop_memory.pipeline.lifecycle.checkpoint_gc import prune_checkpoints

    from octop_harness.middleware.memory import _prune_checkpoints_if_configured

    if "messages_channel" not in inspect.signature(prune_checkpoints).parameters:
        pytest.skip("installed octop-memory does not support message snapshots")
    monkeypatch.setenv("OCTOP_CHECKPOINT_DURABILITY", "async")
    monkeypatch.setenv("OCTOP_CHECKPOINT_RETENTION", "1")
    monkeypatch.setenv("OCTOP_CHECKPOINT_KEEP_LAST", "20")
    monkeypatch.delenv("OCTOP_CHECKPOINT_KEEP_DAYS", raising=False)
    monkeypatch.setenv("OCTOP_CHECKPOINT_BACKUP_BEFORE_PRUNE", "first")
    db = tmp_path / "memory.sqlite"
    async with open_probe(
        db, tmp_path / "workspace", approval=approval, subagent=approval == "child", postgres_dsn=retention_dsn
    ) as (
        agent,
        model,
        memory,
        _,
    ):
        for index in range(16):
            await agent.call(request("done", f"done {index}"))
            await agent.call(request("pending", f"pending {index}"))
        prompt = "SUBAGENT_APPROVAL" if approval == "child" else "APPROVAL"
        events = await _stream(agent, "pending", prompt)
        assert any(event["type"] == "hitl_required" for event in events)
        pending = await state(agent, "pending")
        pending_config = {"configurable": {"thread_id": "pending"}}
        pending_rows = list(memory.list(pending_config))
        done = {"configurable": {"thread_id": "done", "checkpoint_ns": ""}}
        done_rows = list(memory.list(done))
        retained = done_rows[:20]
        assert "messages" not in retained[-1].checkpoint["channel_values"]
        before_states = [(await agent.graph.aget_state(row.config)).values for row in retained]
        before_history = await agent.aget_history("done", limit=100)

        monkeypatch.setenv("OCTOP_CHECKPOINT_DURABILITY", "exit")
        _prune_checkpoints_if_configured(memory, messages_channel=agent.graph.channels["messages"])

        if backend == "sqlite":
            assert db.with_name("memory.sqlite.pre-prune.bak").is_file()
        else:
            assert list(memory.list(done)) == done_rows
            stats = prune_checkpoints(
                memory,
                thread_id="done",
                keep_last=20,
                skip_pending=True,
                offline=True,
                messages_channel=agent.graph.channels["messages"],
            )
            assert stats.checkpoints_deleted == len(done_rows) - 20 > 0
        remaining = list(memory.list(done))
        assert len(remaining) == 20
        assert remaining[-1].parent_config is None
        assert [(await agent.graph.aget_state(row.config)).values for row in retained] == before_states
        assert await agent.aget_history("done", limit=100) == before_history
        assert list(memory.list(pending_config)) == pending_rows
        assert await state(agent, "pending") == pending
        await agent.call(request("done", "continue"))
        assert model._inputs[-1][-2]["data"]["content"] == "reply: done 15"
        events = [event async for event in agent.resume_hitl("pending", [{"type": "approve"}])]
        assert all(event["type"] != "hitl_required" for event in events)
        assert not (await state(agent, "pending"))["next"]


async def _crash_child(db: Path, workspace: Path) -> None:
    async with open_probe(db, workspace) as (agent, _, memory, started):
        await agent.call(request("thread", "last completed turn"))
        baseline = rows(memory, "thread")[0]
        task = asyncio.create_task(_stream(agent, "thread", "CRASH"))
        await started.wait()
        if os.environ["OCTOP_CHECKPOINT_DURABILITY"] == "async":
            while rows(memory, "thread")[0] <= baseline:
                await asyncio.sleep(0.01)
        print("ready-to-kill", flush=True)
        await task


async def test_process_kill_keeps_last_finished_exit_turn(tmp_path, monkeypatch):
    recovered = {}
    for mode in ("async", "exit"):
        monkeypatch.setenv("OCTOP_CHECKPOINT_DURABILITY", mode)
        db = tmp_path / mode / "memory.sqlite"
        workspace = tmp_path / "workspace"
        child = await asyncio.create_subprocess_exec(
            sys.executable, __file__, str(db), str(workspace), stdout=asyncio.subprocess.PIPE
        )
        try:
            assert child.stdout is not None
            assert await asyncio.wait_for(child.stdout.readline(), 20) == b"ready-to-kill\n"
        finally:
            if child.returncode is None:
                child.kill()
            await child.wait()
        assert (db.parent / "tool-effect").read_text(encoding="utf-8") == "applied"
        async with open_probe(db, workspace) as (agent, model, memory, _):
            saved = await state(agent, "thread")
            before = rows(memory, "thread")
            await agent.call(request("thread", "after restart"))
            recovered[mode] = (saved, model._inputs[-1], before)
            print(
                "after kill", mode, [m["data"]["content"] for m in saved["values"]["messages"]], saved["next"], before
            )
    assert [m["data"]["content"] for m in recovered["exit"][0]["values"]["messages"]] == [
        "last completed turn",
        "reply: last completed turn",
    ]
    assert recovered["exit"][0]["next"] == ()
    assert any(m["data"]["content"] == "CRASH" for m in recovered["async"][0]["values"]["messages"])
    assert (
        recovered["exit"][1][-1]["data"]["content"] == recovered["async"][1][-1]["data"]["content"] == "after restart"
    )


if __name__ == "__main__":
    asyncio.run(_crash_child(Path(sys.argv[1]), Path(sys.argv[2])))
