"""``ToolsFilterMiddleware`` — hide tools listed in ``tools_disabled`` from the model and block them at dispatch."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import BaseTool
from langgraph.prebuilt.tool_node import ToolCallRequest

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
    """Strip disabled tools from ``ModelRequest.tools`` before each model call, and
    reject any tool call that names a disabled tool at dispatch time.

    Hiding a tool from the model only lowers the chance it is called; a model that
    names a disabled tool in a ``tool_call`` would otherwise still execute it
    (#2267). ``awrap_tool_call`` blocks such calls with an error ``ToolMessage`` —
    the same way the registry rejects a name it does not know.

    Reads ``HarnessAgentConfig.tools_disabled`` so hosts can hot-update via
    ``HarnessAgent.set_tools_disabled`` without recompiling the graph.

    Mount after MCP / user middleware and before tool search so disabled names
    neither reappear nor enter the deferred-tool catalog.
    """

    def __init__(self, *, config: HarnessAgentConfig) -> None:
        super().__init__()
        self._config = config

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._apply_filter(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._apply_filter(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        tool_call = request.tool_call
        name = _tool_name(tool_call)
        if name and name in self._config.tools_disabled:
            return ToolMessage(
                content=(
                    f"tool {name!r} is disabled for this agent and cannot be executed"
                ),
                tool_call_id=str(tool_call.get("id", "")),
                status="error",
            )
        return await handler(request)

    def _apply_filter(self, request: ModelRequest) -> ModelRequest:
        disabled = self._config.tools_disabled
        if not disabled:
            return request
        tools_in: list[BaseTool | dict[str, Any]] = list(request.tools or [])
        filtered = [t for t in tools_in if _tool_name(t) not in disabled]
        if len(filtered) == len(tools_in):
            return request
        return request.override(tools=filtered)


__all__ = ["ToolsFilterMiddleware"]
