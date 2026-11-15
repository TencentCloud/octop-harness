"""Managed ACP graph: one call node streams text deltas."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from octop_harness.acp.managed import _hitl_payload, build_managed_graph, option_id_for_decision
from octop_harness.acp.models import SuspendedPermission


class _Service:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def get_pending_permission(self, *, thread_id: str, runner: str) -> Any:
        del thread_id, runner
        return None

    async def get_session(self, thread_id: str, runner: str) -> Any:
        del thread_id, runner
        return None

    async def run_turn(self, **kwargs: Any) -> dict[str, Any]:
        blocks = kwargs["prompt_blocks"]
        self.prompts.append(str(blocks[0]["text"]))
        await kwargs["on_message"](
            {"type": "tool_start", "name": "Read", "call_id": "call-1", "status": "in_progress"},
            False,
        )
        await kwargs["on_message"](
            {
                "type": "tool_update",
                "name": "Read",
                "call_id": "call-1",
                "status": "completed",
                "output": "ok",
            },
            False,
        )
        await kwargs["on_message"]({"type": "text_delta", "text": "hello"}, False)
        await kwargs["on_message"]({"type": "text_delta", "text": " there"}, False)
        return {"status": "completed", "event": None}

    async def resume_permission(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"status": "completed", "event": None}


def test_fallback_permission_does_not_repeat_the_slug() -> None:
    payload = _hitl_payload(
        SuspendedPermission(
            payload={},
            options=[],
            runner="codebuddy",
            tool_name="external-agent",
            tool_kind="other",
        )
    )
    request = payload["action_requests"][0]
    assert request["name"] == "external-agent"
    assert request["args"] == {}
    assert "description" not in request


def test_named_permission_keeps_its_summary() -> None:
    payload = _hitl_payload(
        SuspendedPermission(
            payload={},
            options=[],
            runner="codebuddy",
            tool_name="Read",
            tool_kind="read",
            summary="Read README.md",
        )
    )
    request = payload["action_requests"][0]
    assert request["description"] == "Read README.md"
    assert request["args"] == {"summary": "Read README.md"}


def test_option_id_maps_approve_and_reject() -> None:
    permission = SuspendedPermission(
        payload={},
        options=[
            {"optionId": "allow", "kind": "allow_once"},
            {"optionId": "deny", "kind": "reject_once"},
        ],
        runner="opencode",
        tool_name="bash",
        tool_kind="execute",
    )
    assert option_id_for_decision(permission, {"decisions": [{"type": "approve"}]}) == "allow"
    assert option_id_for_decision(permission, {"decisions": [{"type": "reject"}]}) == "deny"
    assert option_id_for_decision(permission, None) == "deny"
    assert option_id_for_decision(permission, {"decisions": []}) == "deny"

    permanent_first = SuspendedPermission(
        payload={},
        options=[
            {"optionId": "always", "kind": "allow_always"},
            {"optionId": "once", "kind": "allow_once"},
            {"optionId": "deny", "kind": "reject_once"},
        ],
        runner="opencode",
        tool_name="bash",
        tool_kind="execute",
    )
    assert option_id_for_decision(permanent_first, {"decisions": [{"type": "approve"}]}) == "once"
    assert option_id_for_decision(permanent_first, {"decisions": [{"type": "edit"}]}) == "deny"


async def test_managed_graph_streams_acp_text() -> None:
    service = _Service()
    graph = build_managed_graph(
        runner="opencode",
        acp_config=None,  # type: ignore[arg-type]
        cwd="/tmp",
        service=service,
    )
    chunks: list[str] = []
    tools: list[str] = []
    results: list[dict[str, Any]] = []
    async for item in graph.astream(
        {"messages": [HumanMessage(content="summarize")]},
        config={"configurable": {"thread_id": "thread-1"}},
        stream_mode="messages",
    ):
        message = item[0] if isinstance(item, tuple) else item
        if not isinstance(message, AIMessageChunk):
            continue
        for call in getattr(message, "tool_call_chunks", None) or []:
            name = call.get("name") if isinstance(call, dict) else ""
            if name:
                tools.append(str(name))
        if isinstance(message.content, str) and message.content:
            chunks.append(message.content)
        extra = getattr(message, "additional_kwargs", None)
        if isinstance(extra, dict) and isinstance(extra.get("octop_tool_result"), dict):
            results.append(extra["octop_tool_result"])
    assert chunks == ["hello", " there"]
    assert tools == ["Read"]
    assert results == [{"tool_call_id": "call-1", "content": "ok", "status": "success"}]
    assert service.prompts == ["summarize"]


class _SilentService(_Service):
    async def run_turn(self, **kwargs: Any) -> dict[str, Any]:
        blocks = kwargs["prompt_blocks"]
        self.prompts.append(str(blocks[0]["text"]))
        return {"status": "completed", "event": None}


class _ReasoningService(_Service):
    async def run_turn(self, **kwargs: Any) -> dict[str, Any]:
        blocks = kwargs["prompt_blocks"]
        self.prompts.append(str(blocks[0]["text"]))
        await kwargs["on_message"]({"type": "reasoning_delta", "text": "look first"}, False)
        await kwargs["on_message"](
            {"type": "tool_start", "name": "Read", "call_id": "call-1", "status": "in_progress"},
            False,
        )
        await kwargs["on_message"](
            {
                "type": "tool_update",
                "name": "Read",
                "call_id": "call-1",
                "status": "completed",
                "output": "ok",
            },
            False,
        )
        await kwargs["on_message"]({"type": "text_delta", "text": "hello there"}, False)
        return {"status": "completed", "event": None}


async def test_managed_graph_stores_reasoning_and_tools() -> None:
    service = _ReasoningService()
    graph = build_managed_graph(
        runner="opencode",
        acp_config=None,  # type: ignore[arg-type]
        cwd="/tmp",
        service=service,
    )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="summarize")]},
        config={"configurable": {"thread_id": "thread-3"}},
    )
    messages = result["messages"]
    reasoning = messages[1]
    tool_call = messages[2]
    tool_result = messages[3]
    answer = messages[4]
    assert reasoning.additional_kwargs["reasoning_content"] == "look first"
    assert reasoning.additional_kwargs["octop_managed_streamed"] is True
    assert tool_call.tool_calls[0]["id"] == "call-1"
    assert tool_call.tool_calls[0]["name"] == "Read"
    assert tool_result.type == "tool"
    assert tool_result.content == "ok"
    assert tool_result.tool_call_id == "call-1"
    assert tool_result.status == "success"
    assert tool_result.additional_kwargs["octop_managed_streamed"] is True
    assert answer.content == "hello there"
    assert answer.additional_kwargs["octop_managed_streamed"] is True


async def test_managed_graph_accepts_turn_without_text() -> None:
    service = _SilentService()
    graph = build_managed_graph(
        runner="opencode",
        acp_config=None,  # type: ignore[arg-type]
        cwd="/tmp",
        service=service,
    )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "thread-2"}},
    )
    assert result["messages"][-1].content == ""


def test_protocol_forwards_managed_tool_result() -> None:
    from langchain_core.messages import AIMessageChunk

    from octop_harness.protocols.langgraph import LangGraphProtocol
    from octop_harness.protocols.think_splitter import ThinkSplitter

    protocol = LangGraphProtocol.__new__(LangGraphProtocol)
    events = protocol._project_messages_chunk(
        (
            AIMessageChunk(
                content="",
                additional_kwargs={
                    "octop_tool_result": {
                        "tool_call_id": "call-1",
                        "content": "ok",
                        "status": "success",
                    }
                },
            ),
            {"langgraph_node": "call"},
        ),
        ThinkSplitter(),
    )
    assert events == [
        {
            "type": "tool_result",
            "node": "call",
            "messages": [{"tool_call_id": "call-1", "content": "ok", "status": "success"}],
        }
    ]


class _PermissionService(_Service):
    def __init__(self) -> None:
        super().__init__()
        self.pending: Any = None
        self.resumes: list[str] = []

    async def get_pending_permission(self, *, thread_id: str, runner: str) -> Any:
        del thread_id, runner
        return self.pending

    async def get_session(self, thread_id: str, runner: str) -> Any:
        del thread_id, runner
        return SimpleNamespace(acp_session_id="sess-1")

    async def run_turn(self, **kwargs: Any) -> dict[str, Any]:
        blocks = kwargs["prompt_blocks"]
        self.prompts.append(str(blocks[0]["text"]))
        await kwargs["on_message"]({"type": "text_delta", "text": "before"}, False)
        self.pending = SuspendedPermission(
            payload={},
            options=[
                {"optionId": "always", "kind": "allow_always"},
                {"optionId": "once", "kind": "allow_once"},
                {"optionId": "deny", "kind": "reject_once"},
            ],
            runner="opencode",
            tool_name="bash",
            tool_kind="execute",
        )
        return {"status": "permission_required", "suspended_permission": self.pending}

    async def resume_permission(self, **kwargs: Any) -> dict[str, Any]:
        self.resumes.append(str(kwargs["option_id"]))
        self.pending = None
        await kwargs["on_message"]({"type": "text_delta", "text": "after"}, False)
        return {"status": "completed", "event": None}


async def test_managed_graph_keeps_text_when_permission_pauses() -> None:
    service = _PermissionService()
    graph = build_managed_graph(
        runner="opencode",
        acp_config=None,  # type: ignore[arg-type]
        cwd="/tmp",
        checkpointer=InMemorySaver(),
        service=service,
    )
    config = {"configurable": {"thread_id": "thread-perm"}}
    paused = await graph.ainvoke({"messages": [HumanMessage(content="run tests")]}, config)
    assert paused.get("__interrupt__")
    saved = [getattr(message, "content", "") for message in paused["messages"]]
    assert "before" in saved
    assert service.prompts == ["run tests"]

    resumed = await graph.ainvoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
    assert service.resumes == ["once"]
    assert "after" in [getattr(message, "content", "") for message in resumed["messages"]]
    assert "before" in [getattr(message, "content", "") for message in resumed["messages"]]


def test_protocol_keeps_stored_managed_tools_off_the_live_channel() -> None:
    from langchain_core.messages import AIMessage, ToolMessage

    from octop_harness.protocols.langgraph import LangGraphProtocol

    protocol = LangGraphProtocol.__new__(LangGraphProtocol)
    events = protocol._project_updates_chunk(
        {
            "call": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "Read", "args": {}, "id": "call-1", "type": "tool_call"}],
                        additional_kwargs={"octop_managed_streamed": True},
                    ),
                    ToolMessage(
                        content="ok",
                        tool_call_id="call-1",
                        name="Read",
                        additional_kwargs={"octop_managed_streamed": True},
                    ),
                ]
            }
        }
    )
    assert [event["type"] for event in events] == ["state_update"]
