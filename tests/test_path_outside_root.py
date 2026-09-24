"""Tests for FilesystemGuard outside-root soft-fail."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage

from octop_harness.middleware.filesystem_guard import (
    FilesystemGuardMiddleware,
    is_path_outside_root_error,
    path_outside_root_tool_message,
)


def test_is_path_outside_root_error_matches_deepagents_message() -> None:
    exc = ValueError(r"Path:D:\octop-data\data\x.md outside root directory: C:\Users\Administrator")
    assert is_path_outside_root_error(exc) is True


def test_is_path_outside_root_error_matches_traversal() -> None:
    assert is_path_outside_root_error(ValueError("Path traversal not allowed")) is True


def test_is_path_outside_root_error_ignores_other_value_errors() -> None:
    assert is_path_outside_root_error(ValueError("invalid offset")) is False
    assert is_path_outside_root_error(RuntimeError("outside root directory")) is False


def test_path_outside_root_tool_message_shape() -> None:
    msg = path_outside_root_tool_message(
        tool_name="read_file",
        tool_call_id="call_1",
        exc=ValueError("Path:/tmp/x outside root directory: /home"),
    )
    assert isinstance(msg, ToolMessage)
    assert msg.status == "error"
    assert msg.name == "read_file"
    assert msg.tool_call_id == "call_1"
    assert "outside root directory" in str(msg.content)
    assert "storage root" in str(msg.content).lower()


@pytest.mark.asyncio
async def test_middleware_softens_outside_root_value_error() -> None:
    mw = FilesystemGuardMiddleware()
    request = MagicMock()
    request.tool_call = {"name": "read_file", "id": "call_99", "args": {}}

    async def boom(_req: Any) -> ToolMessage:
        raise ValueError("Path:/D:/data/x.md outside root directory: C:/Users/me")

    out = await mw.awrap_tool_call(request, boom)
    assert isinstance(out, ToolMessage)
    assert out.status == "error"
    assert out.tool_call_id == "call_99"
    assert "outside root directory" in str(out.content)


@pytest.mark.asyncio
async def test_middleware_reraises_unrelated_value_error() -> None:
    mw = FilesystemGuardMiddleware()
    request = MagicMock()
    request.tool_call = {"name": "read_file", "id": "call_1", "args": {}}

    async def boom(_req: Any) -> ToolMessage:
        raise ValueError("limit must be > 0")

    with pytest.raises(ValueError, match="limit must be > 0"):
        await mw.awrap_tool_call(request, boom)


@pytest.mark.asyncio
async def test_middleware_does_not_soften_non_fs_tool() -> None:
    """Outside-root text from a non-filesystem tool must still propagate."""
    mw = FilesystemGuardMiddleware()
    request = MagicMock()
    request.tool_call = {"name": "web_fetch", "id": "call_w", "args": {}}

    async def boom(_req: Any) -> ToolMessage:
        raise ValueError("Path:/tmp/x outside root directory: /home")

    with pytest.raises(ValueError, match="outside root directory"):
        await mw.awrap_tool_call(request, boom)


def test_sync_wrap_softens_outside_root() -> None:
    mw = FilesystemGuardMiddleware()
    request = MagicMock()
    request.tool_call = {"name": "write_file", "id": "call_w", "args": {}}

    def boom(_req: Any) -> ToolMessage:
        raise ValueError("Path:/etc/passwd outside root directory: /workspace")

    out = mw.wrap_tool_call(request, boom)
    assert isinstance(out, ToolMessage)
    assert out.status == "error"
    assert out.name == "write_file"
