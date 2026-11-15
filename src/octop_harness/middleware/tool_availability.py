"""``ToolAvailabilityMiddleware`` — hide tools whose env is missing this turn.

Hosts that mount search providers unconditionally (``web_search_require_env=
False``) rely on per-call credentials to light them up. This middleware drops
the provider tools that have neither a per-call value nor a global env var, so
the model never sees a tool that would only answer with an authentication
error.

Mount after tool filtering (``ToolsFilterMiddleware``) and before tool search,
so hidden tools cannot leak into the deferred-tool catalog.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.tools.base import BaseTool

from octop_harness.tool_env_scope import current_tool_env


class ToolAvailabilityMiddleware(AgentMiddleware[Any, Any]):
    """Drop tools whose required env vars are absent for this turn.

    *required_env* maps tool names to the env vars that must all be non-empty.
    Tools not in the mapping, empty tuples, and non-``BaseTool`` entries
    (provider tool dicts) are left untouched.
    """

    def __init__(self, *, required_env: Mapping[str, tuple[str, ...]]) -> None:
        super().__init__()
        self._required_env = {name: tuple(vars_) for name, vars_ in required_env.items()}

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

    def _apply_filter(self, request: ModelRequest) -> ModelRequest:
        tools_in: list[BaseTool | dict[str, Any]] = list(request.tools or [])
        kept = [tool for tool in tools_in if self._is_available(tool)]
        if len(kept) == len(tools_in):
            return request
        return request.override(tools=kept)

    def _is_available(self, tool: BaseTool | dict[str, Any]) -> bool:
        if not isinstance(tool, BaseTool):
            return True
        required = self._required_env.get(tool.name)
        if not required:
            return True
        return all(current_tool_env(name) for name in required)


__all__ = ["ToolAvailabilityMiddleware"]
