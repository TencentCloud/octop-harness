"""Tests for FilesystemGuardMiddleware (execution-backend path policy)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from deepagents.middleware.filesystem import FilesystemPermission
from langgraph.prebuilt.tool_node import ToolCallRequest

from octop_harness.middleware.filesystem_guard import (
    FilesystemGuardMiddleware,
    filesystem_guard_block_reason,
    unbounded_glob_block_reason,
)


def test_basename_glob_requires_an_explicit_path() -> None:
    for pattern in ("*", "**", "**/*", " * ", "*.md"):
        reason = unbounded_glob_block_reason("glob", {"pattern": pattern})
        assert reason is not None
        assert "explicit path" in reason
    assert unbounded_glob_block_reason("glob", {"pattern": "*", "path": "/"}) is None
    assert unbounded_glob_block_reason("glob", {"pattern": "*", "path": "//"}) is None
    assert unbounded_glob_block_reason("glob", {"pattern": "**/*.md"}) is None
    assert unbounded_glob_block_reason("glob", {"pattern": "src/**/*.py"}) is None


def test_allows_glob_anchored_to_a_directory() -> None:
    assert unbounded_glob_block_reason("glob", {"pattern": "*", "path": "."}) is None
    assert unbounded_glob_block_reason("glob", {"pattern": "*.md", "path": "."}) is None
    assert unbounded_glob_block_reason("ls", {"pattern": "*"}) is None


def test_relative_glob_stays_in_workspace_and_root_is_explicit(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    def resolve_path(path: str) -> str:
        if path == ".":
            return str(workspace_root)
        if path == "/":
            return str(tmp_path)
        return str((workspace_root / path).resolve())

    workspace = SimpleNamespace(resolve_path=resolve_path)
    assert unbounded_glob_block_reason("glob", {"pattern": "*", "path": "."}, workspace=workspace) is None
    assert unbounded_glob_block_reason("glob", {"pattern": "*.md", "path": "."}, workspace=workspace) is None
    assert unbounded_glob_block_reason("glob", {"pattern": "*", "path": "/"}, workspace=workspace) is None
    escaped = unbounded_glob_block_reason("glob", {"pattern": "*", "path": "../secret"}, workspace=workspace)
    assert escaped is not None
    assert "leaves the workspace" in escaped


def test_guard_rejects_root_glob_before_the_tool_runs() -> None:
    request = ToolCallRequest(
        tool_call={"name": "glob", "args": {"pattern": "*"}, "id": "call-1"},
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )

    def handler(_request: ToolCallRequest) -> None:
        raise AssertionError("glob should not run")

    result = FilesystemGuardMiddleware().wrap_tool_call(request, handler)
    assert result.status == "error"
    assert "explicit path" in str(result.content)


def test_blocks_read_on_denied_sensitive_path() -> None:
    permissions = [
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/etc/**"],
            mode="deny",
        )
    ]
    reason = filesystem_guard_block_reason(
        permissions,
        tool_name="read_file",
        params={"file_path": "/etc/passwd"},
    )
    assert reason is not None
    assert "permission denied" in reason
    assert "/etc/passwd" in reason


def test_allows_read_outside_denied_prefix() -> None:
    permissions = [
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/etc/**"],
            mode="deny",
        )
    ]
    assert (
        filesystem_guard_block_reason(
            permissions,
            tool_name="read_file",
            params={"file_path": "/workspace/SOUL.md"},
        )
        is None
    )


def test_blocks_write_file_tool() -> None:
    permissions = [
        FilesystemPermission(
            operations=["write"],
            paths=["/**/.env"],
            mode="deny",
        )
    ]
    reason = filesystem_guard_block_reason(
        permissions,
        tool_name="write_file",
        params={"file_path": "/home/user/project/.env"},
    )
    assert reason is not None
    assert "write" in reason


def test_blocks_ls_on_denied_directory_root() -> None:
    """``/etc/**`` must also deny listing the directory itself (``ls /etc``)."""
    permissions = [
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/etc/**"],
            mode="deny",
        )
    ]
    reason = filesystem_guard_block_reason(
        permissions,
        tool_name="ls",
        params={"path": "/etc"},
    )
    assert reason is not None
    assert "permission denied" in reason


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS /etc → /private/etc alias")
def test_blocks_ls_on_macos_private_etc_alias() -> None:
    """macOS ``/etc`` → ``/private/etc``; deny rules must cover the real path."""
    permissions = [
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/etc/**"],
            mode="deny",
        )
    ]
    reason = filesystem_guard_block_reason(
        permissions,
        tool_name="ls",
        params={"path": "/private/etc"},
    )
    assert reason is not None
    assert "permission denied" in reason
