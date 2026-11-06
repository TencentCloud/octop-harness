"""History repair must change checkpoint state without replaying public output."""

from __future__ import annotations

from typing import Any

import pytest
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from tests.test_protocols._fakes import graph_with_astream

from octop_harness.protocols.langgraph import LangGraphProtocol
from octop_harness.protocols.mcp import MCPProtocol
from octop_harness.protocols.openai import OpenAIProtocol

_REPAIR_NODE = "PatchToolCallsMiddleware.before_agent"
_CURRENT_REPLY = "CURRENT_REPLY can legitimately quote OLD_REPLY"


@pytest.mark.asyncio
@pytest.mark.parametrize("dangling", [False, True])
async def test_real_history_repair_only_streams_current_reply(dangling: bool) -> None:
    """Use the actual middleware: a generic history-rewrite node does not reproduce this bug."""
    model = FakeMessagesListChatModel(
        responses=[AIMessage(content=_CURRENT_REPLY, additional_kwargs={"reasoning_content": "CURRENT_THOUGHT"})]
    )
    graph = create_agent(model, middleware=[PatchToolCallsMiddleware()], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "offline-history-repair"}}
    history = [
        HumanMessage(content="old question", id="old-user"),
        AIMessage(content="OLD_REPLY", id="old-answer", additional_kwargs={"reasoning_content": "OLD_THOUGHT"}),
        AIMessage(
            content="OLD_TOOL_PREFACE",
            id="old-tool-call",
            tool_calls=[{"id": "old-call", "name": "unused_tool", "args": {}}],
        ),
    ]
    if not dangling:
        history.append(ToolMessage(content="previous result", tool_call_id="old-call", id="old-result"))
    await graph.aupdate_state(config, {"messages": history})
    request = [HumanMessage(content="current question", id="current-user")]

    events = [event async for event in LangGraphProtocol(graph=graph).stream(request, config)]

    assert [event["content"] for event in events if event["type"] == "token"] == [_CURRENT_REPLY]
    assert [event["content"] for event in events if event["type"] == "reasoning"] == ["CURRENT_THOUGHT"]
    # Filtering presentation must not remove history or prevent the missing result from being repaired.
    state = await graph.aget_state(config)
    messages = state.values["messages"]
    assert any(message.id == "old-answer" and message.content == "OLD_REPLY" for message in messages)
    results = [
        message for message in messages if isinstance(message, ToolMessage) and message.tool_call_id == "old-call"
    ]
    assert len(results) == 1
    assert results[0].status == ("error" if dangling else "success")
    if dangling:
        assert any(event["type"] == "tool_result" and event["node"] == _REPAIR_NODE for event in events)
    assert any(event["type"] == "state_snapshot" for event in events)


def _stream_parts() -> list[dict[str, Any]]:
    """Mix repair output, state changes, and genuinely new output with matching words."""
    replay = AIMessageChunk(
        content="OLD_REPLY",
        additional_kwargs={"reasoning_content": "OLD_THOUGHT"},
        tool_call_chunks=[{"id": "old-call", "name": "unused_tool", "args": "{}", "index": 0}],
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
    )
    repaired = ToolMessage(content="missing result repaired", tool_call_id="old-call", status="error")
    return [
        {"type": "messages", "data": (replay, {"langgraph_node": _REPAIR_NODE})},
        {"type": "updates", "data": {_REPAIR_NODE: {"messages": [repaired]}}},
        {"type": "values", "data": {"messages": [repaired]}},
        {"type": "messages", "data": (AIMessageChunk(content=_CURRENT_REPLY), {"langgraph_node": "model"})},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol_type", [LangGraphProtocol, OpenAIProtocol, MCPProtocol])
async def test_protocol_adapters_do_not_replay_history(protocol_type: type[LangGraphProtocol]) -> None:
    """All adapters share the same source-based filter, without matching text content."""
    protocol = protocol_type(graph=graph_with_astream(_stream_parts()))
    events = [event async for event in protocol.stream([HumanMessage(content="current question")], {})]

    if protocol_type is OpenAIProtocol:
        text = "".join(event["choices"][0]["delta"].get("content", "") for event in events)
    elif protocol_type is MCPProtocol:
        text = "".join(event["content"]["text"] for event in events)
    else:
        text = "".join(event["content"] for event in events if event["type"] == "token")
        assert not any(event["type"] in {"reasoning", "tool_call_chunk", "usage"} for event in events)
        assert [event["type"] for event in events] == ["tool_result", "state_snapshot", "token"]
    assert text == _CURRENT_REPLY


@pytest.mark.asyncio
async def test_resumed_stream_drops_repair_replay_and_keeps_state() -> None:
    """Resuming a turn must apply the same filtering while preserving repair updates."""
    protocol = LangGraphProtocol(graph=graph_with_astream(_stream_parts()))
    events = [event async for event in protocol.resume_stream("offline-resume", [], {})]

    assert [event["type"] for event in events] == ["tool_result", "state_snapshot", "token"]
    assert events[-1]["content"] == _CURRENT_REPLY
