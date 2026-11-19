"""Tests for FilesystemGuardMiddleware (execution-backend path policy)."""

from __future__ import annotations

import shutil
import sys
import tempfile
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
from deepagents.middleware.filesystem import (
    ExecuteSchema,
    FilesystemPermission,
    GlobSchema,
    ReadFileSchema,
    ReadVideoFileSchema,
)
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
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


def test_relative_file_path_defaults_to_workspace_before_upstream_tool() -> None:
    request = ToolCallRequest(
        tool_call={"name": "read_file", "args": {"file_path": "docs/guide.md"}, "id": "call-1"},
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    seen: dict[str, object] = {}

    def handler(item: ToolCallRequest) -> ToolMessage:
        seen.update(item.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="call-1")

    FilesystemGuardMiddleware(workspace_dir="/agent/workspace").wrap_tool_call(request, handler)
    assert seen == {"file_path": "/agent/workspace/docs/guide.md"}


def test_base_dir_resolves_a_document_relative_file_without_persisting_cwd() -> None:
    request = ToolCallRequest(
        tool_call={
            "name": "read_file",
            "args": {"file_path": "references/sources.md", "base_dir": "/agent/.octop/skills/daily-hot-news"},
            "id": "call-1",
        },
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    seen: dict[str, object] = {}

    def handler(item: ToolCallRequest) -> ToolMessage:
        seen.update(item.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="call-1")

    FilesystemGuardMiddleware(workspace_dir="/agent/workspace").wrap_tool_call(request, handler)
    assert seen == {"file_path": "/agent/.octop/skills/daily-hot-news/references/sources.md"}


def test_glob_without_a_root_stays_in_workspace() -> None:
    request = ToolCallRequest(
        tool_call={"name": "glob", "args": {"pattern": "**/fetch_dailyhot.mjs"}, "id": "call-1"},
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    seen: dict[str, object] = {}

    def handler(item: ToolCallRequest) -> ToolMessage:
        seen.update(item.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="call-1")

    FilesystemGuardMiddleware(workspace_dir="/agent/workspace").wrap_tool_call(request, handler)
    assert seen == {"pattern": "**/fetch_dailyhot.mjs", "path": "/agent/workspace"}


def test_execute_cwd_is_consumed_before_the_upstream_schema() -> None:
    request = ToolCallRequest(
        tool_call={
            "name": "execute",
            "args": {"command": "node scripts/fetch_dailyhot.mjs weibo", "cwd": "/agent/.octop/skills/daily-hot-news"},
            "id": "call-1",
        },
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    seen: dict[str, object] = {}

    def handler(item: ToolCallRequest) -> ToolMessage:
        seen.update(item.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="call-1")

    FilesystemGuardMiddleware(workspace_dir="/agent/workspace").wrap_tool_call(request, handler)
    assert seen == {"command": "cd -- /agent/.octop/skills/daily-hot-news && node scripts/fetch_dailyhot.mjs weibo"}


def _sandbox_workspace(**backend_attrs: object) -> SimpleNamespace:
    return SimpleNamespace(backend=SimpleNamespace(sandbox_fs=True, **backend_attrs))


def test_sandbox_relative_path_stays_on_the_virtual_root() -> None:
    request = ToolCallRequest(
        tool_call={"name": "read_file", "args": {"file_path": "docs/guide.md"}, "id": "call-1"},
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    seen: dict[str, object] = {}

    def handler(item: ToolCallRequest) -> ToolMessage:
        seen.update(item.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="call-1")

    FilesystemGuardMiddleware(
        workspace_dir="/home/wally/.octop/agents/Q6084B",
        workspace=_sandbox_workspace(),  # type: ignore[arg-type]
    ).wrap_tool_call(request, handler)
    assert seen == {"file_path": "/docs/guide.md"}


def test_sandbox_base_dir_stays_virtual_until_the_backend_maps_it() -> None:
    request = ToolCallRequest(
        tool_call={
            "name": "read_file",
            "args": {
                "file_path": "references/sources.md",
                "base_dir": "/.octop/skills/daily-hot-news",
            },
            "id": "call-1",
        },
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    seen: dict[str, object] = {}

    def handler(item: ToolCallRequest) -> ToolMessage:
        seen.update(item.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="call-1")

    FilesystemGuardMiddleware(
        workspace_dir="/home/wally/.octop/agents/Q6084B",
        workspace=_sandbox_workspace(),  # type: ignore[arg-type]
    ).wrap_tool_call(request, handler)
    assert seen == {"file_path": "/.octop/skills/daily-hot-news/references/sources.md"}


def test_sandbox_glob_without_a_root_uses_the_virtual_workspace() -> None:
    request = ToolCallRequest(
        tool_call={"name": "glob", "args": {"pattern": "**/fetch_dailyhot.mjs"}, "id": "call-1"},
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    seen: dict[str, object] = {}

    def handler(item: ToolCallRequest) -> ToolMessage:
        seen.update(item.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="call-1")

    FilesystemGuardMiddleware(
        workspace_dir="/home/wally/.octop/agents/Q6084B",
        workspace=_sandbox_workspace(),  # type: ignore[arg-type]
    ).wrap_tool_call(request, handler)
    assert seen == {"pattern": "**/fetch_dailyhot.mjs", "path": "/"}


def test_execute_cwd_uses_the_sandbox_shell_path() -> None:
    request = ToolCallRequest(
        tool_call={
            "name": "execute",
            "args": {
                "command": "node scripts/fetch_dailyhot.mjs weibo",
                "cwd": "/.octop/skills/daily-hot-news",
            },
            "id": "call-1",
        },
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    seen: dict[str, object] = {}

    def handler(item: ToolCallRequest) -> ToolMessage:
        seen.update(item.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="call-1")

    def shell_path(path: str) -> str:
        if path == "/.octop/skills/daily-hot-news":
            return "/workspace/.octop/skills/daily-hot-news"
        return path

    FilesystemGuardMiddleware(
        workspace_dir="/home/wally/.octop/agents/Q6084B",
        workspace=_sandbox_workspace(shell_path=shell_path),  # type: ignore[arg-type]
    ).wrap_tool_call(request, handler)
    assert seen == {
        "command": "cd -- /workspace/.octop/skills/daily-hot-news && node scripts/fetch_dailyhot.mjs weibo",
    }


def _schema_tool(name: str, schema: type, func: object) -> StructuredTool:
    return StructuredTool.from_function(
        name=name,
        description=name,
        func=func,  # type: ignore[arg-type]
        args_schema=schema,
    )


def _json_schema(tool: StructuredTool) -> dict[str, object]:
    raw = tool.tool_call_schema
    assert not isinstance(raw, dict)
    return raw.model_json_schema()


def test_model_request_advertises_relative_paths_and_document_base() -> None:
    read_file = _schema_tool(
        "read_file",
        ReadFileSchema,
        lambda file_path, offset=0, limit=100: file_path,
    )
    read_video = _schema_tool(
        "read_file",
        ReadVideoFileSchema,
        lambda file_path, offset=0, limit=100: file_path,
    )
    execute = _schema_tool(
        "execute",
        ExecuteSchema,
        lambda command, timeout=None: command,
    )
    glob = _schema_tool(
        "glob",
        GlobSchema,
        lambda pattern, path=None: pattern,
    )
    other = _schema_tool("current_time", ReadFileSchema, lambda file_path, offset=0, limit=100: file_path)
    request = ModelRequest(model=object(), messages=[], tools=[read_file, read_video, execute, glob, other])  # type: ignore[arg-type]
    seen: list[ModelRequest] = []

    def handler(item: ModelRequest) -> ModelResponse:
        seen.append(item)
        return ModelResponse(result=[])

    FilesystemGuardMiddleware().wrap_model_call(request, handler)

    assert len(seen) == 1
    advertised = seen[0].tools
    assert advertised[4] is other
    read_schema = _json_schema(advertised[0])
    video_schema = _json_schema(advertised[1])
    execute_schema = _json_schema(advertised[2])
    glob_schema = _json_schema(advertised[3])
    assert "base_dir" in read_schema["properties"]
    assert "base_dir" in video_schema["properties"]
    assert "relative" in str(read_schema["properties"]["file_path"]["description"])
    assert "Must be absolute" not in str(read_schema["properties"]["file_path"]["description"])
    assert "seconds" in str(video_schema["properties"]["offset"]["description"])
    assert "cwd" in execute_schema["properties"]
    assert "base_dir" in glob_schema["properties"]
    assert "base_dir" not in _json_schema(read_file)["properties"]
    assert "cwd" not in _json_schema(execute)["properties"]


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


class _DictWorkspace:
    """Minimal in-memory workspace for capture tests.

    ``resolve_path``/``present_path`` mimic a virtual-root workspace: "/x" and
    "x" are two spellings of the same file and canonicalize to "x".
    ``materialize_local`` mirrors the BackendWorkspace contract by staging
    current content into a temp file for the snapshot read.
    """

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.fail_read = False
        self.probe_unknown = False
        self._tmp = tempfile.mkdtemp(prefix="dict-ws-")
        self._stage = Path(self._tmp)
        weakref.finalize(self, shutil.rmtree, self._tmp, True, True)

    def exists(self, path: str) -> bool:
        return path in self.files

    def exists_strict(self, path: str) -> bool | None:
        if self.probe_unknown:
            return None
        return path in self.files

    def materialize_local(self, path: str, *, map_system_paths: bool = True) -> Path | None:
        if path not in self.files:
            return None
        if self.fail_read:
            raise OSError("snapshot read failed")
        target = self._stage / path.replace("/", "_")
        target.write_text(self.files[path], encoding="utf-8")
        return target

    def read_text(self, path: str, *, limit: int = 10_000_000) -> str | None:
        if self.fail_read:
            raise OSError("snapshot read failed")
        return self.files.get(path)

    def write_text(self, path: str, content: str, *, force: bool = False) -> None:
        self.files[path] = content

    def system_rel(self, logical: str) -> str:
        return logical

    def resolve_path(self, path: str) -> str:
        return path.lstrip("/") or "/"

    def present_path(self, storage_path: str) -> str:
        return storage_path.lstrip("/") or "/"


def test_guard_captures_before_after_for_write_tools() -> None:
    from langchain_core.messages import ToolMessage

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["a.md"] = "old"
    tracker = FileChangeTracker()
    guard = FilesystemGuardMiddleware(workspace=ws, file_changes=tracker)

    def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"], args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    request = ToolCallRequest(
        tool_call={"name": "write_file", "args": {"file_path": "a.md", "content": "new"}, "id": "c1"},
        tool=None,
        state={},
        runtime=SimpleNamespace(),
    )
    guard.wrap_tool_call(request, handler)
    changes = tracker.drain()
    assert len(changes) == 1
    assert changes[0].path == "a.md"
    assert changes[0].before == "old"
    assert changes[0].after == "new"


def test_guard_capture_created_file_has_null_before() -> None:
    from langchain_core.messages import ToolMessage

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    tracker = FileChangeTracker()
    guard = FilesystemGuardMiddleware(workspace=ws, file_changes=tracker)

    def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"], args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    request = ToolCallRequest(
        tool_call={"name": "write_file", "args": {"file_path": "new.md", "content": "x"}, "id": "c2"},
        tool=None,
        state={},
        runtime=SimpleNamespace(),
    )
    guard.wrap_tool_call(request, handler)
    changes = tracker.drain()
    assert changes[0].before is None
    assert changes[0].after == "x"


def test_guard_skips_capture_on_tool_error() -> None:
    from langchain_core.messages import ToolMessage

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["a.md"] = "old"
    tracker = FileChangeTracker()
    guard = FilesystemGuardMiddleware(workspace=ws, file_changes=tracker)

    def handler(req: ToolCallRequest):
        return ToolMessage(content="boom", tool_call_id=req.tool_call["id"], status="error")

    request = ToolCallRequest(
        tool_call={"name": "write_file", "args": {"file_path": "a.md", "content": "new"}, "id": "c3"},
        tool=None,
        state={},
        runtime=SimpleNamespace(),
    )
    guard.wrap_tool_call(request, handler)
    assert tracker.drain() == []


def test_tracker_scopes_drains_by_invocation() -> None:
    from octop_harness.middleware.file_changes import FileChangeTracker

    tracker = FileChangeTracker()
    tracker.record("a.md", "1", "2", scope="t1")
    tracker.record("b.md", None, "x", scope="t2")
    assert [c.path for c in tracker.drain("t1")] == ["a.md"]
    assert [c.path for c in tracker.drain("t2")] == ["b.md"]
    assert tracker.drain("t1") == []


def _write_request(path: str, content: str, call_id: str, name: str = "write_file") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": {"file_path": path, "content": content}, "id": call_id},
        tool=None,
        state={},
        runtime=SimpleNamespace(),
    )


def test_guard_capture_accepts_prefixed_tool_name() -> None:
    from langchain_core.messages import ToolMessage

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["a.md"] = "old"
    tracker = FileChangeTracker()
    guard = FilesystemGuardMiddleware(workspace=ws, file_changes=tracker)

    def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"], args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    guard.wrap_tool_call(_write_request("a.md", "new", "c4", name="filesystem/write_file"), handler)
    changes = tracker.drain()
    assert len(changes) == 1
    assert changes[0].path == "a.md"
    assert changes[0].before == "old"
    assert changes[0].after == "new"


def test_guard_capture_merges_path_variants_of_one_file() -> None:
    from langchain_core.messages import ToolMessage

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["notes.txt"] = "v1"
    tracker = FileChangeTracker()
    guard = FilesystemGuardMiddleware(workspace=ws, file_changes=tracker)

    def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"].lstrip("/"), args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    guard.wrap_tool_call(_write_request("notes.txt", "v2", "c5"), handler)
    guard.wrap_tool_call(_write_request("/notes.txt", "v3", "c6"), handler)
    changes = tracker.drain()
    assert len(changes) == 1
    assert changes[0].path == "notes.txt"
    assert changes[0].before == "v1"
    assert changes[0].after == "v3"


def test_guard_skips_capture_when_before_snapshot_truncated() -> None:
    from langchain_core.messages import ToolMessage

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["big.md"] = "0123456789abcdef"
    tracker = FileChangeTracker()
    guard = FilesystemGuardMiddleware(workspace=ws, file_changes=tracker)
    guard._SNAPSHOT_LIMIT = 8

    def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"], args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    guard.wrap_tool_call(_write_request("big.md", "tiny", "c7"), handler)
    assert tracker.drain() == []


def test_guard_skips_capture_when_after_snapshot_truncated() -> None:
    from langchain_core.messages import ToolMessage

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    tracker = FileChangeTracker()
    guard = FilesystemGuardMiddleware(workspace=ws, file_changes=tracker)
    guard._SNAPSHOT_LIMIT = 4

    def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"], args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    guard.wrap_tool_call(_write_request("big.md", "0123456789abcdef", "c8"), handler)
    assert tracker.drain() == []


def test_guard_skips_capture_when_snapshot_unreadable() -> None:
    from langchain_core.messages import ToolMessage

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["a.md"] = "old"
    ws.fail_read = True
    tracker = FileChangeTracker()
    guard = FilesystemGuardMiddleware(workspace=ws, file_changes=tracker)

    def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"], args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    guard.wrap_tool_call(_write_request("a.md", "new", "c10"), handler)
    assert tracker.drain() == []


def test_guard_suppresses_later_records_after_snapshot_skip() -> None:
    from langchain_core.messages import ToolMessage

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["a.md"] = "0123456789abcdef"
    tracker = FileChangeTracker()
    guard = FilesystemGuardMiddleware(workspace=ws, file_changes=tracker)
    guard._SNAPSHOT_LIMIT = 8

    def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"], args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    guard.wrap_tool_call(_write_request("a.md", "tiny", "c12"), handler)
    guard.wrap_tool_call(_write_request("a.md", "small", "c13"), handler)
    assert tracker.drain() == []


def test_tracker_unrewindable_path_is_forgotten_for_the_window() -> None:
    from octop_harness.middleware.file_changes import FileChangeTracker

    tracker = FileChangeTracker()
    tracker.record("a.md", "1", "2", scope="t1")
    tracker.mark_unrewindable("a.md", scope="t1")
    tracker.record("a.md", "3", "4", scope="t1")
    assert tracker.drain("t1") == []


def test_tracker_orders_before_after_by_capture_sequence() -> None:
    from octop_harness.middleware.file_changes import FileChangeTracker

    tracker = FileChangeTracker()
    # Parallel calls: the later-started call (seq 1) completes and inserts
    # first; the earlier call (seq 0) must still own the pre-turn before.
    tracker.record("a.md", "mid", "b-after", scope="t1", seq_before=1, seq_after=3)
    tracker.record("a.md", "pre", "a-after", scope="t1", seq_before=0, seq_after=2)
    drained = tracker.drain("t1")
    assert len(drained) == 1
    assert drained[0].before == "pre"
    assert drained[0].after == "b-after"


def test_tracker_drops_net_noop_entries() -> None:
    from octop_harness.middleware.file_changes import FileChangeTracker

    tracker = FileChangeTracker()
    tracker.record("created.md", None, "x", scope="t1")
    tracker.record("created.md", "x", None, scope="t1")
    tracker.record("same.md", "a", "a", scope="t1")
    assert tracker.drain("t1") == []


def test_tracker_discard_drops_pending_scope() -> None:
    from octop_harness.middleware.file_changes import FileChangeTracker

    tracker = FileChangeTracker()
    tracker.record("a.md", "1", "2", scope="t1")
    tracker.record("b.md", "1", "2", scope="t2")
    tracker.discard("t1")
    assert tracker.drain("t1") == []
    assert [c.path for c in tracker.drain("t2")] == ["b.md"]
    tracker.record("c.md", None, "x", scope="t3")
    tracker.discard(None)
    assert tracker.drain("t3") == []


def test_guard_skips_capture_when_existence_probe_unknown() -> None:
    from langchain_core.messages import ToolMessage

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["a.md"] = "old"
    ws.probe_unknown = True
    tracker = FileChangeTracker()
    guard = FilesystemGuardMiddleware(workspace=ws, file_changes=tracker)

    def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"], args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    guard.wrap_tool_call(_write_request("a.md", "new", "c14"), handler)
    assert tracker.drain() == []


async def test_guard_capture_async_path_records_change() -> None:
    from langchain_core.messages import ToolMessage

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["a.md"] = "old"
    tracker = FileChangeTracker()
    guard = FilesystemGuardMiddleware(workspace=ws, file_changes=tracker)

    async def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"], args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    await guard.awrap_tool_call(_write_request("a.md", "new", "c11"), handler)
    changes = tracker.drain()
    assert len(changes) == 1
    assert changes[0].path == "a.md"
    assert changes[0].before == "old"
    assert changes[0].after == "new"


def test_guard_capture_covers_append_file() -> None:
    from langchain_core.messages import ToolMessage

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["log.md"] = "line1"
    tracker = FileChangeTracker()
    guard = FilesystemGuardMiddleware(workspace=ws, file_changes=tracker)

    def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"], ws.read_text(args["file_path"]) + args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    request = ToolCallRequest(
        tool_call={"name": "append_file", "args": {"file_path": "log.md", "content": "line2"}, "id": "c9"},
        tool=None,
        state={},
        runtime=SimpleNamespace(),
    )
    guard.wrap_tool_call(request, handler)
    changes = tracker.drain()
    assert len(changes) == 1
    assert changes[0].path == "log.md"
    assert changes[0].before == "line1"
    assert changes[0].after == "line1line2"
