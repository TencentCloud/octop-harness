"""Tests for ``octop_harness.middleware.tools_filter.ToolsFilterMiddleware``."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from octop_harness.config import HarnessAgentConfig
from octop_harness.middleware.tools_filter import ToolsFilterMiddleware


def _named_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


class TestToolsFilterMiddleware:
    def _middleware(self, *, disabled: frozenset[str] = frozenset()) -> ToolsFilterMiddleware:
        return ToolsFilterMiddleware(config=HarnessAgentConfig(tools_disabled=disabled))

    def _request(self, tools: list[Any]) -> ModelRequest:
        return ModelRequest(
            model=MagicMock(),
            messages=[],
            system_message=SystemMessage(content="Base prompt"),
            tool_choice=None,
            tools=tools,
            response_format=None,
            state={},
            runtime=MagicMock(),
            model_settings={},
        )

    def test_empty_disabled_passes_through(self) -> None:
        tools = [_named_tool("web_fetch"), _named_tool("read_file")]
        mw = self._middleware()
        captured: dict[str, object] = {}

        def handler(r: ModelRequest) -> ModelResponse:
            captured["tools"] = list(r.tools or [])
            return ModelResponse(result=[])

        mw.wrap_model_call(self._request(tools), handler)
        assert captured["tools"] == tools

    def test_filters_matching_tool_names(self) -> None:
        tools = [
            _named_tool("web_fetch"),
            _named_tool("read_file"),
            _named_tool("execute"),
        ]
        mw = self._middleware(disabled=frozenset({"web_fetch", "execute"}))
        captured: dict[str, object] = {}

        def handler(r: ModelRequest) -> ModelResponse:
            captured["names"] = [t.name for t in (r.tools or [])]
            return ModelResponse(result=[])

        mw.wrap_model_call(self._request(tools), handler)
        assert captured["names"] == ["read_file"]

    def test_filters_openai_style_dict_tools(self) -> None:
        tools: list[Any] = [
            {"type": "function", "function": {"name": "web_fetch"}},
            {"type": "function", "function": {"name": "ls"}},
            {"name": "browser_use"},
        ]
        mw = self._middleware(disabled=frozenset({"web_fetch", "browser_use"}))
        captured: dict[str, object] = {}

        def handler(r: ModelRequest) -> ModelResponse:
            captured["tools"] = list(r.tools or [])
            return ModelResponse(result=[])

        mw.wrap_model_call(self._request(tools), handler)
        assert len(captured["tools"]) == 1
        assert captured["tools"][0]["function"]["name"] == "ls"

    @pytest.mark.asyncio
    async def test_async_wrap_filters(self) -> None:
        tools = [_named_tool("web_fetch"), _named_tool("ls")]
        mw = self._middleware(disabled=frozenset({"web_fetch"}))
        captured: dict[str, object] = {}

        async def handler(r: ModelRequest) -> ModelResponse:
            captured["names"] = [t.name for t in (r.tools or [])]
            return ModelResponse(result=[])

        await mw.awrap_model_call(self._request(tools), handler)
        assert captured["names"] == ["ls"]

    def test_hot_update_via_config_mutation(self) -> None:
        cfg = HarnessAgentConfig(tools_disabled=frozenset())
        mw = ToolsFilterMiddleware(config=cfg)
        tools = [_named_tool("web_fetch"), _named_tool("ls")]
        captured: dict[str, object] = {}

        def handler(r: ModelRequest) -> ModelResponse:
            captured["names"] = [t.name for t in (r.tools or [])]
            return ModelResponse(result=[])

        mw.wrap_model_call(self._request(tools), handler)
        assert captured["names"] == ["web_fetch", "ls"]

        cfg.tools_disabled = frozenset({"web_fetch"})
        mw.wrap_model_call(self._request(tools), handler)
        assert captured["names"] == ["ls"]

    def _tool_call_request(self, name: str, call_id: str = "call_1") -> ToolCallRequest:
        return ToolCallRequest(
            tool_call={"name": name, "id": call_id, "args": {}},
            tool=None,
            state={},
            runtime=MagicMock(),
        )

    async def _noop_handler(self, request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="ran", tool_call_id=str(request.tool_call.get("id", "")))

    @pytest.mark.asyncio
    async def test_empty_disabled_passes_tool_call_through(self) -> None:
        mw = self._middleware()
        result = await mw.awrap_tool_call(
            self._tool_call_request("execute"), self._noop_handler
        )
        assert result.status != "error"
        assert result.content == "ran"

    @pytest.mark.asyncio
    async def test_rejects_disabled_tool_call(self) -> None:
        mw = self._middleware(disabled=frozenset({"execute"}))
        called: dict[str, bool] = {"hit": False}

        async def handler(request: ToolCallRequest) -> ToolMessage:
            called["hit"] = True
            return ToolMessage(content="ran", tool_call_id="call_1")

        result = await mw.awrap_tool_call(
            self._tool_call_request("execute"), handler
        )
        assert called["hit"] is False
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "disabled" in result.content

    @pytest.mark.asyncio
    async def test_allows_enabled_tool_call(self) -> None:
        mw = self._middleware(disabled=frozenset({"execute"}))
        result = await mw.awrap_tool_call(
            self._tool_call_request("read_file", call_id="call_2"), self._noop_handler
        )
        assert result.status != "error"
        assert result.content == "ran"

    @pytest.mark.asyncio
    async def test_rejects_disabled_namespaced_tool_call(self) -> None:
        mw = self._middleware(disabled=frozenset({"mcp__weather"}))
        result = await mw.awrap_tool_call(
            self._tool_call_request("mcp__weather", call_id="call_3"), self._noop_handler
        )
        assert result.status == "error"
