"""Tests for ``octop_harness.middleware.tools_filter.ToolsFilterMiddleware``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolRuntime
from langgraph.runtime import Runtime

from octop_harness.config import HarnessAgentConfig
from octop_harness.middleware.tools_filter import ToolsFilterMiddleware


class ConnectorContext(TypedDict, total=False):
    mcp_servers: list[str]


def _named_tool(name: str) -> BaseTool:
    def run() -> str:
        return "ok"

    return StructuredTool.from_function(run, name=name, description="Test tool")


def _tool_names(request: ModelRequest[ConnectorContext | None]) -> list[str]:
    names: list[str] = []
    for tool in request.tools:
        assert isinstance(tool, BaseTool)
        names.append(tool.name)
    return names


class TestToolsFilterMiddleware:
    def _middleware(self, *, disabled: frozenset[str] = frozenset()) -> ToolsFilterMiddleware:
        return ToolsFilterMiddleware(config=HarnessAgentConfig(tools_disabled=disabled))

    def _request(
        self, tools: Sequence[BaseTool | dict[str, object]], context: ConnectorContext | None = None
    ) -> ModelRequest[ConnectorContext | None]:
        return ModelRequest(
            model=MagicMock(),
            messages=[],
            system_message=SystemMessage(content="Base prompt"),
            tool_choice=None,
            tools=list(tools),
            response_format=None,
            state={"messages": []},
            runtime=Runtime[ConnectorContext | None](context=context),
            model_settings={},
        )

    def test_empty_disabled_passes_through(self) -> None:
        tools = [_named_tool("web_fetch"), _named_tool("read_file")]
        mw = self._middleware()
        captured: dict[str, list[BaseTool | dict[str, object]]] = {}

        def handler(r: ModelRequest[ConnectorContext | None]) -> ModelResponse:
            captured["tools"] = list(r.tools or [])
            return ModelResponse(result=[])

        _ = mw.wrap_model_call(self._request(tools), handler)
        assert captured["tools"] == tools

    def test_subagent_filter_preserves_legacy_tools_without_a_turn_ceiling(self) -> None:
        tools = [_named_tool("execute")]
        mw = ToolsFilterMiddleware(config=HarnessAgentConfig(tools_disabled=frozenset({"execute"})), turn_only=True)
        captured: list[BaseTool | dict[str, object]] = []

        def handler(request: ModelRequest[ConnectorContext | None]) -> ModelResponse:
            captured.extend(request.tools)
            return ModelResponse(result=[])

        _ = mw.wrap_model_call(self._request(tools), handler)
        assert captured == tools

    def test_graph_ceiling_wins_over_fallback_runtime_connector_context(self) -> None:
        request = self._request([_named_tool("read_file")], context={"mcp_servers": ["private"]})
        captured: list[BaseTool | dict[str, object]] = []

        def handler(filtered: ModelRequest[ConnectorContext | None]) -> ModelResponse:
            captured.extend(filtered.tools)
            return ModelResponse(result=[])

        token = var_child_runnable_config.set({"configurable": {"allowed_tools": []}})
        try:
            _ = self._middleware().wrap_model_call(request, handler)
            assert captured == []
        finally:
            var_child_runnable_config.reset(token)

    def test_filters_matching_tool_names(self) -> None:
        tools = [
            _named_tool("web_fetch"),
            _named_tool("read_file"),
            _named_tool("execute"),
        ]
        mw = self._middleware(disabled=frozenset({"web_fetch", "execute"}))
        captured: dict[str, list[str]] = {}

        def handler(r: ModelRequest[ConnectorContext | None]) -> ModelResponse:
            captured["names"] = _tool_names(r)
            return ModelResponse(result=[])

        _ = mw.wrap_model_call(self._request(tools), handler)
        assert captured["names"] == ["read_file"]

    def test_filters_openai_style_dict_tools(self) -> None:
        tools: list[BaseTool | dict[str, object]] = [
            {"type": "function", "function": {"name": "web_fetch"}},
            {"type": "function", "function": {"name": "ls"}},
            {"name": "browser_use"},
        ]
        mw = self._middleware(disabled=frozenset({"web_fetch", "browser_use"}))
        captured: dict[str, list[BaseTool | dict[str, object]]] = {}

        def handler(r: ModelRequest[ConnectorContext | None]) -> ModelResponse:
            captured["tools"] = list(r.tools or [])
            return ModelResponse(result=[])

        _ = mw.wrap_model_call(self._request(tools), handler)
        assert len(captured["tools"]) == 1
        remaining = captured["tools"][0]
        assert isinstance(remaining, dict)
        assert remaining == {"type": "function", "function": {"name": "ls"}}

    @pytest.mark.asyncio
    async def test_async_wrap_filters(self) -> None:
        tools = [_named_tool("web_fetch"), _named_tool("ls")]
        mw = self._middleware(disabled=frozenset({"web_fetch"}))
        captured: dict[str, list[str]] = {}

        async def handler(r: ModelRequest[ConnectorContext | None]) -> ModelResponse:
            captured["names"] = _tool_names(r)
            return ModelResponse(result=[])

        _ = await mw.awrap_model_call(self._request(tools), handler)
        assert captured["names"] == ["ls"]

    @pytest.mark.parametrize("allowed, expected", [([], []), (["read_file", "execute"], ["read_file"])])
    def test_turn_allowlist_intersects_existing_tool_permissions(self, allowed: list[str], expected: list[str]) -> None:
        mw = self._middleware(disabled=frozenset({"execute"}))
        request = self._request([_named_tool("read_file"), _named_tool("execute"), _named_tool("task")])
        captured: list[str] = []

        def handler(req: ModelRequest[ConnectorContext | None]) -> ModelResponse:
            captured.extend(_tool_names(req))
            return ModelResponse(result=[])

        token = var_child_runnable_config.set({"configurable": {"allowed_tools": allowed}})
        try:
            _ = mw.wrap_model_call(request, handler)
        finally:
            var_child_runnable_config.reset(token)
        assert captured == expected

    @pytest.mark.parametrize(
        "name", ["read_file", "grep", "task", "search_knowledge", "mcp/private_read", "tool_search"]
    )
    @pytest.mark.parametrize("async_call", [False, True])
    @pytest.mark.asyncio
    async def test_disallowed_tool_never_reaches_handler(self, name: str, async_call: bool) -> None:
        from unittest.mock import AsyncMock

        request = ToolCallRequest(
            tool_call={"name": name, "args": {}, "id": "call-1", "type": "tool_call"},
            tool=None,
            state={},
            runtime=MagicMock(),
        )
        handler = AsyncMock() if async_call else MagicMock()
        token = var_child_runnable_config.set({"configurable": {"allowed_tools": []}})
        try:
            mw = self._middleware()
            result = await mw.awrap_tool_call(request, handler) if async_call else mw.wrap_tool_call(request, handler)
        finally:
            var_child_runnable_config.reset(token)
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.tool_call_id == "call-1"
        handler.assert_not_called()

    def test_hot_update_via_config_mutation(self) -> None:
        cfg = HarnessAgentConfig(tools_disabled=frozenset())
        mw = ToolsFilterMiddleware(config=cfg)
        tools = [_named_tool("web_fetch"), _named_tool("ls")]
        captured: dict[str, list[str]] = {}

        def handler(r: ModelRequest[ConnectorContext | None]) -> ModelResponse:
            captured["names"] = _tool_names(r)
            return ModelResponse(result=[])

        _ = mw.wrap_model_call(self._request(tools), handler)
        assert captured["names"] == ["web_fetch", "ls"]

        cfg.tools_disabled = frozenset({"web_fetch"})
        _ = mw.wrap_model_call(self._request(tools), handler)
        assert captured["names"] == ["ls"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("allowed", [None, [], ["web_fetch"], ["read_file"]])
    @pytest.mark.parametrize("async_call", [False, True])
    async def test_tool_runtime_preserves_ceiling_without_graph_context(
        self, allowed: list[str] | None, async_call: bool
    ) -> None:
        def discard_output(_chunk: object) -> None:
            pass

        request = ToolCallRequest(
            tool_call={"name": "read_file", "args": {}, "id": "fallback", "type": "tool_call"},
            tool=None,
            state={},
            runtime=ToolRuntime(
                state={},
                context=None,
                config={"configurable": {"allowed_tools": allowed}},
                stream_writer=discard_output,
                tool_call_id="fallback",
                store=None,
            ),
        )
        success = ToolMessage(content="executed", tool_call_id="fallback")
        handler = AsyncMock(return_value=success) if async_call else MagicMock(return_value=success)
        with patch("octop_harness.middleware.tools_filter.get_config", side_effect=RuntimeError("no graph context")):
            mw = self._middleware()
            result = await mw.awrap_tool_call(request, handler) if async_call else mw.wrap_tool_call(request, handler)
        if allowed is None or "read_file" in allowed:
            assert result is success
            handler.assert_called_once_with(request)
        else:
            assert isinstance(result, ToolMessage)
            assert result.status == "error"
            assert result.tool_call_id == "fallback"
            handler.assert_not_called()
