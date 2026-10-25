"""Exercise per-invocation tool ceilings through a real agent graph."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TypedDict, cast, override

import pytest
from langchain import agents
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from octop_harness.config import HarnessAgentConfig
from octop_harness.middleware.conversation_mode import ConversationModeMiddleware
from octop_harness.middleware.tools_filter import ToolsFilterMiddleware
from octop_harness.request import ChatRequest


class ForcedToolModel(BaseChatModel):
    """Return a tool call even when the host hid that tool from the model."""

    @property
    @override
    def _llm_type(self) -> str:
        return "forced-tool-test"

    @override
    def bind_tools(self, tools: Sequence[object], **kwargs: object) -> ForcedToolModel:
        del tools, kwargs
        return self

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        message = (
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"path": str(messages[-1].content)},
                        "id": "read-1",
                        "type": "tool_call",
                    }
                ],
            )
            if isinstance(messages[-1], HumanMessage)
            else AIMessage(content="done")
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class GraphState(TypedDict):
    messages: list[BaseMessage]


TestGraph = Runnable[dict[str, list[BaseMessage]], GraphState]


async def _invoke(graph: TestGraph, request: ChatRequest) -> GraphState:
    config = cast(RunnableConfig, cast(object, request.to_runnable_config()))
    return await graph.ainvoke({"messages": request.normalize_messages()}, config)


def _graph(calls: list[str], *, checkpointer: InMemorySaver | None = None) -> TestGraph:
    @tool
    def read_file(path: str) -> str:
        """Read a synthetic test file."""
        calls.append(path)
        return "synthetic contents"

    return cast(
        TestGraph,
        agents.create_agent(
            model=ForcedToolModel(),
            tools=[read_file],
            middleware=[ConversationModeMiddleware(), ToolsFilterMiddleware(config=HarnessAgentConfig())],
            checkpointer=checkpointer,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ask", "plan", "craft"])
async def test_concurrent_invocations_do_not_share_tool_permissions(mode: str) -> None:
    calls: list[str] = []
    graph = _graph(calls)

    async def invoke(path: str, allowed: list[str] | None) -> GraphState:
        request = ChatRequest(messages=path, thread_id=path, allowed_tools=allowed, conversation_mode=mode)
        return await _invoke(graph, request)

    restricted, owner = await asyncio.gather(invoke("group-private.md", []), invoke("owner-file.md", None))
    assert calls == ["owner-file.md"]
    restricted_tool = restricted["messages"][-2]
    owner_tool = owner["messages"][-2]
    assert isinstance(restricted_tool, ToolMessage)
    assert isinstance(owner_tool, ToolMessage)
    assert restricted_tool.status == "error"
    assert owner_tool.status == "success"
    assert restricted["messages"][-1].content == "done"


@pytest.mark.asyncio
async def test_explicit_tool_allowlist_still_allows_the_selected_tool() -> None:
    calls: list[str] = []
    graph = _graph(calls)
    request = ChatRequest(messages="public.md", allowed_tools=["read_file"])
    _ = await _invoke(graph, request)
    assert calls == ["public.md"]


@pytest.mark.asyncio
@pytest.mark.parametrize("policies", [[[], None], [None, []]])
async def test_tool_policy_is_recomputed_for_each_turn_of_the_same_thread(policies: list[list[str] | None]) -> None:
    calls: list[str] = []
    graph = _graph(calls, checkpointer=InMemorySaver())
    for index, allowed in enumerate(policies):
        request = ChatRequest(messages=f"turn-{index}", thread_id="shared-group", allowed_tools=allowed)
        _ = await _invoke(graph, request)
    assert calls == [f"turn-{index}" for index, allowed in enumerate(policies) if allowed is None]
