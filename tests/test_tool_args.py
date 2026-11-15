"""Tests for ToolArgsMiddleware (placeholder argument guard)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from pydantic import BaseModel, Field

from octop_harness.middleware.tool_args import ToolArgsMiddleware, placeholder_arg_fields


class _GrepArgs(BaseModel):
    pattern: str = Field(description="Text pattern to search for (literal string, not regex).")
    path: str | None = Field(default=None, description="Directory to search in.")


def _grep(pattern: str, path: str | None = None) -> str:
    return f"{pattern}:{path}"


def _tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_grep,
        args_schema=_GrepArgs,
        name="grep",
        description="Search files for a literal pattern.",
    )


def _request(args: dict[str, Any], tool: Any = None) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": "grep", "args": args, "id": "call-1"},
        tool=tool if tool is not None else _tool(),
        state={},
        runtime=None,  # type: ignore[arg-type]
    )


def test_field_name_value_is_a_placeholder() -> None:
    assert placeholder_arg_fields(_tool(), {"pattern": "pattern"}) == ["pattern"]
    assert placeholder_arg_fields(_tool(), {"pattern": " Pattern "}) == ["pattern"]


def test_real_value_and_optional_field_are_left_alone() -> None:
    assert placeholder_arg_fields(_tool(), {"pattern": "Octop"}) == []
    assert placeholder_arg_fields(_tool(), {"pattern": "Octop", "path": "path"}) == []
    assert placeholder_arg_fields(_tool(), {}) == []


def test_guard_blocks_the_placeholder_call() -> None:
    def handler(_request: ToolCallRequest) -> ToolMessage:
        raise AssertionError("the tool should not run")

    result = ToolArgsMiddleware().wrap_tool_call(_request({"pattern": "pattern"}), handler)
    assert result.status == "error"
    assert "pattern" in str(result.content)


def test_guard_passes_a_real_value_through() -> None:
    def handler(request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="hit", tool_call_id=request.tool_call["id"], status="success")

    result = ToolArgsMiddleware().wrap_tool_call(_request({"pattern": "Octop"}), handler)
    assert result.status == "success"
    assert str(result.content) == "hit"


def test_guard_can_be_disabled() -> None:
    def handler(request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="hit", tool_call_id=request.tool_call["id"], status="success")

    middleware = ToolArgsMiddleware()
    middleware.configure(enabled=False)
    result = middleware.wrap_tool_call(_request({"pattern": "pattern"}), handler)
    assert result.status == "success"


@pytest.mark.asyncio
async def test_async_guard_blocks_the_placeholder_call() -> None:
    async def handler(_request: ToolCallRequest) -> ToolMessage:
        raise AssertionError("the tool should not run")

    result = await ToolArgsMiddleware(language="en").awrap_tool_call(_request({"pattern": "pattern"}), handler)
    assert result.status == "error"
    assert "pattern" in str(result.content)
