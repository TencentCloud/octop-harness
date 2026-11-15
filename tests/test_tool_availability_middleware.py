"""Tests for ``ToolAvailabilityMiddleware``."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.tools import StructuredTool

from octop_harness.middleware.tool_availability import ToolAvailabilityMiddleware
from octop_harness.tool_env_scope import bind_tool_env, tool_env_scope


def _tool(name: str) -> StructuredTool:
    return StructuredTool.from_function(func=lambda: "x", name=name, description=name)


def _request(tools: list[Any]) -> ModelRequest:
    return ModelRequest(model=MagicMock(), messages=[], tools=tools)


def _names(request: ModelRequest) -> list[str]:
    return [getattr(tool, "name", "") for tool in (request.tools or [])]


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DEEPSEEK_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CSE_ID"):
        monkeypatch.delenv(name, raising=False)
    bind_tool_env({})


def _middleware() -> ToolAvailabilityMiddleware:
    return ToolAvailabilityMiddleware(
        required_env={
            "deepseek_search": ("DEEPSEEK_API_KEY",),
            "google_search": ("GOOGLE_API_KEY", "GOOGLE_CSE_ID"),
            "searchfree_search": (),
            "you_search": (),
        },
    )


def _call(mw: ToolAvailabilityMiddleware, request: ModelRequest) -> list[str]:
    seen: dict[str, Any] = {}

    def handler(inner: ModelRequest) -> Any:
        seen["names"] = _names(inner)
        return MagicMock()

    mw.wrap_model_call(request, handler)
    return seen["names"]


def test_hides_tool_without_any_key() -> None:
    names = _call(
        _middleware(),
        _request([_tool("deepseek_search"), _tool("searchfree_search"), _tool("read_file")]),
    )
    assert names == ["searchfree_search", "read_file"]


def test_shows_tool_with_scoped_key() -> None:
    with tool_env_scope({"DEEPSEEK_API_KEY": "sk-personal"}):
        names = _call(_middleware(), _request([_tool("deepseek_search"), _tool("read_file")]))
    assert names == ["deepseek_search", "read_file"]


def test_shows_tool_with_global_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-global")
    names = _call(_middleware(), _request([_tool("deepseek_search")]))
    assert names == ["deepseek_search"]


def test_partial_multivar_key_stays_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    names = _call(_middleware(), _request([_tool("google_search")]))
    assert names == []


def test_scoped_values_complete_a_multivar_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "global-key")
    with tool_env_scope({"GOOGLE_CSE_ID": "personal-cse"}):
        names = _call(_middleware(), _request([_tool("google_search")]))
    assert names == ["google_search"]


def test_non_tool_entries_are_untouched() -> None:
    provider_dict = {"type": "function", "function": {"name": "deepseek_search"}}
    names = _call(_middleware(), _request([provider_dict]))
    assert names == [""]


def test_unmapped_tool_is_untouched() -> None:
    names = _call(_middleware(), _request([_tool("some_other_tool")]))
    assert names == ["some_other_tool"]


@pytest.mark.asyncio
async def test_async_path_filters_the_same_way() -> None:
    seen: dict[str, Any] = {}

    async def handler(inner: ModelRequest) -> Any:
        seen["names"] = _names(inner)
        return MagicMock()

    await _middleware().awrap_model_call(
        _request([_tool("deepseek_search"), _tool("read_file")]),
        handler,
    )
    assert seen["names"] == ["read_file"]
