"""Tests for the shared file-capture middleware (parent chain + subagents)."""

from __future__ import annotations

import shutil
import tempfile
import weakref
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from octop_harness.middleware.file_capture import FileCaptureMiddleware


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


def test_capture_middleware_captures_before_after_for_write_tools() -> None:

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["a.md"] = "old"
    tracker = FileChangeTracker()
    capture = FileCaptureMiddleware(ws, tracker)

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
    capture.wrap_tool_call(request, handler)
    changes = tracker.drain()
    assert len(changes) == 1
    assert changes[0].path == "a.md"
    assert changes[0].before == "old"
    assert changes[0].after == "new"


def test_capture_middleware_created_file_has_null_before() -> None:

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    tracker = FileChangeTracker()
    capture = FileCaptureMiddleware(ws, tracker)

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
    capture.wrap_tool_call(request, handler)
    changes = tracker.drain()
    assert changes[0].before is None
    assert changes[0].after == "x"


def test_capture_middleware_skips_capture_on_tool_error() -> None:

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["a.md"] = "old"
    tracker = FileChangeTracker()
    capture = FileCaptureMiddleware(ws, tracker)

    def handler(req: ToolCallRequest):
        return ToolMessage(content="boom", tool_call_id=req.tool_call["id"], status="error")

    request = ToolCallRequest(
        tool_call={"name": "write_file", "args": {"file_path": "a.md", "content": "new"}, "id": "c3"},
        tool=None,
        state={},
        runtime=SimpleNamespace(),
    )
    capture.wrap_tool_call(request, handler)
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


def test_capture_middleware_accepts_prefixed_tool_name() -> None:

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["a.md"] = "old"
    tracker = FileChangeTracker()
    capture = FileCaptureMiddleware(ws, tracker)

    def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"], args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    capture.wrap_tool_call(_write_request("a.md", "new", "c4", name="filesystem/write_file"), handler)
    changes = tracker.drain()
    assert len(changes) == 1
    assert changes[0].path == "a.md"
    assert changes[0].before == "old"
    assert changes[0].after == "new"


def test_capture_middleware_merges_path_variants_of_one_file() -> None:

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["notes.txt"] = "v1"
    tracker = FileChangeTracker()
    capture = FileCaptureMiddleware(ws, tracker)

    def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"].lstrip("/"), args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    capture.wrap_tool_call(_write_request("notes.txt", "v2", "c5"), handler)
    capture.wrap_tool_call(_write_request("/notes.txt", "v3", "c6"), handler)
    changes = tracker.drain()
    assert len(changes) == 1
    assert changes[0].path == "notes.txt"
    assert changes[0].before == "v1"
    assert changes[0].after == "v3"


def test_capture_middleware_skips_capture_when_before_snapshot_truncated() -> None:

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["big.md"] = "0123456789abcdef"
    tracker = FileChangeTracker()
    capture = FileCaptureMiddleware(ws, tracker)
    capture._SNAPSHOT_LIMIT = 8

    def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"], args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    capture.wrap_tool_call(_write_request("big.md", "tiny", "c7"), handler)
    assert tracker.drain() == []


def test_capture_middleware_skips_capture_when_after_snapshot_truncated() -> None:

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    tracker = FileChangeTracker()
    capture = FileCaptureMiddleware(ws, tracker)
    capture._SNAPSHOT_LIMIT = 4

    def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"], args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    capture.wrap_tool_call(_write_request("big.md", "0123456789abcdef", "c8"), handler)
    assert tracker.drain() == []


def test_capture_middleware_skips_capture_when_snapshot_unreadable() -> None:

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["a.md"] = "old"
    ws.fail_read = True
    tracker = FileChangeTracker()
    capture = FileCaptureMiddleware(ws, tracker)

    def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"], args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    capture.wrap_tool_call(_write_request("a.md", "new", "c10"), handler)
    assert tracker.drain() == []


def test_capture_middleware_suppresses_later_records_after_snapshot_skip() -> None:

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["a.md"] = "0123456789abcdef"
    tracker = FileChangeTracker()
    capture = FileCaptureMiddleware(ws, tracker)
    capture._SNAPSHOT_LIMIT = 8

    def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"], args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    capture.wrap_tool_call(_write_request("a.md", "tiny", "c12"), handler)
    capture.wrap_tool_call(_write_request("a.md", "small", "c13"), handler)
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


def test_capture_middleware_skips_capture_when_existence_probe_unknown() -> None:

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["a.md"] = "old"
    ws.probe_unknown = True
    tracker = FileChangeTracker()
    capture = FileCaptureMiddleware(ws, tracker)

    def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"], args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    capture.wrap_tool_call(_write_request("a.md", "new", "c14"), handler)
    assert tracker.drain() == []


async def test_capture_middleware_async_path_records_change() -> None:

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["a.md"] = "old"
    tracker = FileChangeTracker()
    capture = FileCaptureMiddleware(ws, tracker)

    async def handler(req: ToolCallRequest):
        args = req.tool_call["args"]
        ws.write_text(args["file_path"], args["content"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    await capture.awrap_tool_call(_write_request("a.md", "new", "c11"), handler)
    changes = tracker.drain()
    assert len(changes) == 1
    assert changes[0].path == "a.md"
    assert changes[0].before == "old"
    assert changes[0].after == "new"


def test_capture_middleware_covers_append_file() -> None:

    from octop_harness.middleware.file_changes import FileChangeTracker

    ws = _DictWorkspace()
    ws.files["log.md"] = "line1"
    tracker = FileChangeTracker()
    capture = FileCaptureMiddleware(ws, tracker)

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
    capture.wrap_tool_call(request, handler)
    changes = tracker.drain()
    assert len(changes) == 1
    assert changes[0].path == "log.md"
    assert changes[0].before == "line1"
    assert changes[0].after == "line1line2"


def test_capture_scope_reads_thread_from_request_runtime() -> None:
    from octop_harness.middleware.file_capture import _capture_scope

    request = SimpleNamespace(runtime=SimpleNamespace(config={"configurable": {"thread_id": "t-9"}}))
    assert _capture_scope(request) == "t-9"  # type: ignore[arg-type]


async def test_capture_scope_inherits_parent_thread_in_nested_graph() -> None:
    """Subagent writes land in the parent drain window.

    The task tool invokes sub-graphs with the ambient parent config
    (langgraph ``ensure_config`` merges ``configurable`` per key), so the
    scope resolved inside a nested run is the parent ``thread_id`` — this is
    the mechanism the per-spec capture middleware depends on.
    """
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    from octop_harness.middleware.runtime import current_configurable

    class _State(TypedDict):
        done: bool

    seen: dict[str, object] = {}

    def child_node(_state: _State) -> dict[str, bool]:
        seen["thread_id"] = current_configurable(None).get("thread_id")
        return {"done": True}

    child_builder = StateGraph(_State)
    child_builder.add_node("observe", child_node)
    child_builder.add_edge(START, "observe")
    child_builder.add_edge("observe", END)
    child = child_builder.compile()

    async def parent_node(_state: _State) -> dict[str, bool]:
        # config=None mirrors the task tool: the ambient parent config seeds
        # the nested run.
        await child.ainvoke({}, config=None)
        return {"done": True}

    parent_builder = StateGraph(_State)
    parent_builder.add_node("delegate", parent_node)
    parent_builder.add_edge(START, "delegate")
    parent_builder.add_edge("delegate", END)
    parent = parent_builder.compile()

    await parent.ainvoke({}, config={"configurable": {"thread_id": "t-parent"}})
    assert seen["thread_id"] == "t-parent"
