"""Filter configured tools and enforce an optional per-invocation tool ceiling."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, cast, override

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import BaseTool
from langgraph.config import get_config
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from langgraph.typing import ContextT

from octop_harness.middleware.runtime import runtime_config
from octop_harness.request import normalize_allowed_tools

if TYPE_CHECKING:
    from octop_harness.config import HarnessAgentConfig


def _tool_name(tool: BaseTool | dict[str, Any]) -> str:
    if isinstance(tool, dict):
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            return str(fn["name"])
        if tool.get("name"):
            return str(tool["name"])
        return ""
    return str(getattr(tool, "name", "") or "")


class ToolsFilterMiddleware(AgentMiddleware[Any, Any]):
    """Strip disabled tools from ``ModelRequest.tools`` before each model call.

    Reads ``HarnessAgentConfig.tools_disabled`` so hosts can hot-update via
    ``HarnessAgent.set_tools_disabled`` without recompiling the graph.

    Mount after MCP / user middleware and before tool search so disabled names
    neither reappear nor enter the deferred-tool catalog.

    ``configurable.allowed_tools`` further narrows each invocation and is
    checked again before tool execution, including forged or resumed calls.
    ``turn_only`` preserves legacy subagent behavior when no ceiling is set.
    """

    def __init__(self, *, config: HarnessAgentConfig, turn_only: bool = False) -> None:
        super().__init__()
        self._config: HarnessAgentConfig = config
        self._turn_only: bool = turn_only

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse],
    ) -> ModelResponse:
        return handler(self._apply_filter(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._apply_filter(request))

    def _apply_filter(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        disabled = self._config.tools_disabled
        config: Mapping[str, object]
        try:
            config = get_config()
        except RuntimeError:
            config = runtime_config(request)
        raw_configurable = config.get("configurable")
        configurable: Mapping[str, object] = (
            cast(Mapping[str, object], raw_configurable) if isinstance(raw_configurable, Mapping) else {}
        )
        allowed = normalize_allowed_tools(configurable.get("allowed_tools"))
        if allowed is None and self._turn_only:
            return request
        if not disabled and allowed is None:
            return request
        tools_in: list[BaseTool | dict[str, Any]] = list(request.tools or [])
        filtered = [
            t for t in tools_in if _tool_name(t) not in disabled and (allowed is None or _tool_name(t) in allowed)
        ]
        if allowed is not None and not filtered:
            return request.override(tools=filtered, tool_choice=None)
        if len(filtered) == len(tools_in):
            return request
        return request.override(tools=filtered)

    def _blocked_message(self, request: ToolCallRequest) -> ToolMessage | None:
        config: Mapping[str, object]
        try:
            config = get_config()
        except RuntimeError:
            tool_config = getattr(request.runtime, "config", None)
            config = cast(Mapping[str, object], tool_config) if isinstance(tool_config, Mapping) else {}
        raw_configurable = config.get("configurable")
        configurable: Mapping[str, object] = (
            cast(Mapping[str, object], raw_configurable) if isinstance(raw_configurable, Mapping) else {}
        )
        allowed = normalize_allowed_tools(configurable.get("allowed_tools"))
        name = str(request.tool_call.get("name") or "")
        if allowed is None or (name in allowed and name not in self._config.tools_disabled):
            return None
        return ToolMessage(
            content=f"Tool `{name}` is not allowed for this invocation.",
            name=name,
            tool_call_id=str(request.tool_call.get("id") or ""),
            status="error",
        )

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[object]],
    ) -> ToolMessage | Command[object]:
        blocked = self._blocked_message(request)
        return blocked if blocked is not None else handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[object]]],
    ) -> ToolMessage | Command[object]:
        blocked = self._blocked_message(request)
        return blocked if blocked is not None else await handler(request)


__all__ = ["ToolsFilterMiddleware"]
