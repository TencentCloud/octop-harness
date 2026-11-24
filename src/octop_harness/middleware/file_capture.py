"""File-write capture middleware shared by the parent agent and its subagents.

Snapshots the before/after content of filesystem tool writes into a
:class:`~octop_harness.middleware.file_changes.FileChangeTracker` so the host
can persist per-turn change summaries and rewind files. The middleware is
attached both to the parent chain (right after the filesystem guard, so it
sees the guard's rewrites) and to every subagent spec — subagents do not
inherit the parent's middleware chain, but the parent graph's
``configurable`` reaches them (langgraph ``ensure_config`` merges it), so
captures land under the same invocation ``thread_id`` either way.

Capture is auxiliary: nothing here may break the tool call being wrapped.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from octop_harness.backends.workspace import BackendWorkspace
from octop_harness.middleware.file_changes import FileChangeTracker
from octop_harness.middleware.runtime import current_configurable
from octop_harness.middleware.tool_paths import (
    path_from_tool_args,
    tool_base_name,
    tool_call_meta,
)

logger = logging.getLogger(__name__)

__all__ = ["FileCaptureMiddleware"]


def _capture_scope(request: ToolCallRequest) -> str:
    """Per-invocation drain scope so concurrent turns never share a window.

    Resolves the current graph ``configurable`` through the shared helper and
    uses the invocation ``thread_id`` — the same key the host drains with via
    ``pop_file_changes(thread_id)`` — or ``""`` for scopeless invocations
    (direct harness use without a graph thread). Subagent tool calls resolve
    the parent thread id because the task tool's config inherits it.
    """
    value = current_configurable(request).get("thread_id")
    if isinstance(value, str) and value:
        return value
    return ""


class FileCaptureMiddleware(AgentMiddleware[Any, Any]):
    """Record before/after snapshots for filesystem write tools."""

    _CAPTURE_TOOLS = frozenset({"write_file", "edit_file", "append_file", "delete"})
    # Snapshots feed rewind, which writes the recorded content back verbatim.
    # read_text silently truncates at its limit, so a larger file is skipped
    # instead of risking truncated bytes being restored later. A truncated
    # ``after`` must also be skipped: recorded as ``None`` it would look like
    # "deleted this turn" and rewind would delete a live file.
    _SNAPSHOT_LIMIT = 10_000_000

    def __init__(
        self,
        workspace: BackendWorkspace,
        file_changes: FileChangeTracker,
    ) -> None:
        self._workspace = workspace
        self._file_changes = file_changes

    def _capture_path(self, request: ToolCallRequest) -> str | None:  # noqa: PLR0911
        tool_name, _ = tool_call_meta(request)
        if tool_base_name(tool_name) not in self._CAPTURE_TOOLS:
            return None
        raw_args: Any = request.tool_call.get("args") or {}
        if not isinstance(raw_args, dict):
            return None
        raw = path_from_tool_args(tool_name, raw_args)
        if not raw:
            return None
        # Canonical key: the same file addressed as "notes.txt" and "/notes.txt"
        # (virtual root) must share one entry per drain window so rewind
        # restores the pre-turn content, not an intermediate one. The presented
        # fragment is only used when it round trips back to the same file;
        # otherwise the resolved storage path is kept, which
        # ``exists``/``read_text`` always understand.
        try:
            resolved = self._workspace.resolve_path(raw)
        except Exception:
            # Capture is auxiliary: never break the tool call on a path the
            # backend cannot resolve — fall back to the raw spelling.
            logger.debug("file-change capture: resolve_path failed for %r", raw, exc_info=True)
            return raw
        try:
            presented = self._workspace.present_path(resolved)
        except Exception:
            logger.debug("file-change capture: present_path failed for %r", resolved, exc_info=True)
            return resolved
        if not presented or presented == resolved:
            return resolved
        try:
            if self._workspace.resolve_path(presented) == resolved:
                return presented
        except Exception:
            logger.debug("file-change capture: presented path %r does not round trip", presented)
        return resolved

    def _snapshot(self, path: str | None) -> tuple[str | None, bool]:
        """Content snapshot as ``(text, skip)``.

        ``(None, False)`` means the file is absent — positively confirmed, so
        it is a real deletion this window. ``(None, True)`` means no usable
        snapshot (existence unknown, unreadable or oversized): the caller must
        suppress the record — recording ``None`` on either side would make
        rewind delete a live file.

        Capture is auxiliary: any unexpected failure here fails closed
        (``skip``) so it can never break the tool call being wrapped.
        """
        if path is None:
            # Unknown state: fail closed rather than report a confirmed
            # deletion for something that was never observed.
            return None, True
        try:
            return self._snapshot_unchecked(path)
        except Exception:
            logger.debug("file-change capture: snapshot failed for %r", path, exc_info=True)
            return None, True

    def _snapshot_unchecked(self, path: str) -> tuple[str | None, bool]:
        # One strict probe (absence must be positive), then one
        # materialization shared by the size check and the content read —
        # remote backends download at most once per snapshot. Note the
        # download itself is unbounded for oversized remote objects (no
        # backend stat API yet); the stat() check only avoids the extra
        # decode+keep for local mounts.
        probe = self._workspace.exists_strict(path)
        if probe is None:
            return None, True
        if not probe:
            return None, False
        local = self._workspace.materialize_local(path, map_system_paths=False)
        if local is None:
            # Confirmed present but bytes unavailable: fail closed.
            return None, True
        if local.stat().st_size > self._SNAPSHOT_LIMIT:
            return None, True
        # Decode raw bytes: text mode would normalize CRLF line endings and
        # break the byte-exact restore rewind relies on.
        text = local.read_bytes().decode("utf-8")
        if len(text) > self._SNAPSHOT_LIMIT:
            return None, True
        return text, False

    async def _asnapshot(self, path: str | None) -> tuple[str | None, bool]:
        # Snapshot I/O (potentially a remote download) must not block the loop.
        return await asyncio.to_thread(self._snapshot, path)

    def _record_change(
        self,
        path: str,
        before: str | None,
        result: Any,
        request: ToolCallRequest,
        *,
        seq_before: int,
        seq_after: int,
        before_skip: bool = False,
    ) -> None:
        if isinstance(result, ToolMessage) and getattr(result, "status", None) == "error":
            return
        if before_skip:
            # The record is unusable either way; no after read needed.
            self._store_change(path, before, None, request, seq_before, seq_after, True, True)
            return
        after, after_skip = self._snapshot(path)
        self._store_change(path, before, after, request, seq_before, seq_after, before_skip, after_skip)

    async def _arecord_change(
        self,
        path: str,
        before: str | None,
        result: Any,
        request: ToolCallRequest,
        *,
        seq_before: int,
        seq_after: int,
        before_skip: bool = False,
    ) -> None:
        if isinstance(result, ToolMessage) and getattr(result, "status", None) == "error":
            return
        if before_skip:
            # The record is unusable either way; no after read needed.
            self._store_change(path, before, None, request, seq_before, seq_after, True, True)
            return
        after, after_skip = await self._asnapshot(path)
        self._store_change(path, before, after, request, seq_before, seq_after, before_skip, after_skip)

    def _store_change(
        self,
        path: str | None,
        before: str | None,
        after: str | None,
        request: ToolCallRequest,
        seq_before: int,
        seq_after: int,
        before_skip: bool,
        after_skip: bool,
    ) -> None:
        if path is None:
            return
        try:
            scope = _capture_scope(request)
            if before_skip or after_skip:
                # Pre-turn state for this path is unknown; suppress the
                # pending entry and any later record so rewind never
                # restores an intermediate state.
                self._file_changes.mark_unrewindable(path, scope)
                return
            self._file_changes.record(path, before, after, scope=scope, seq_before=seq_before, seq_after=seq_after)
        except Exception:
            logger.debug("file-change capture: record failed for %r", path, exc_info=True)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        capture_path = self._capture_path(request)
        if capture_path is None:
            return await handler(request)
        # The tracker keeps only the first before-content per path, so when an
        # earlier capture already pinned it (or the path is unrewindable), the
        # re-read cannot improve the record — skip the probe/download.
        scope = _capture_scope(request)
        tracker = self._file_changes
        if tracker.unrewindable(capture_path, scope):
            # Any record would be dropped anyway; skip the after read too.
            return await handler(request)
        try:
            before, before_skip = (
                (await self._asnapshot(capture_path))
                if not tracker.before_known(capture_path, scope)
                else (None, False)
            )
        except Exception:
            # to_thread can raise before the guarded snapshot runs (e.g.
            # executor shutdown); capture must never fail the tool call.
            logger.debug(
                "file-change capture: before snapshot failed for %r",
                capture_path,
                exc_info=True,
            )
            before, before_skip = None, True
        # Assign the sequence only after the snapshot completes so the
        # earliest-observed content owns the smallest seq, even when
        # concurrent calls interleave.
        seq_before = tracker.next_seq()
        result = await handler(request)
        seq_after = tracker.next_seq()
        try:
            await self._arecord_change(
                capture_path,
                before,
                result,
                request,
                seq_before=seq_before,
                seq_after=seq_after,
                before_skip=before_skip,
            )
        except Exception:
            logger.debug("file-change capture: record failed for %r", capture_path, exc_info=True)
        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        capture_path = self._capture_path(request)
        if capture_path is None:
            return handler(request)
        # Same first-before optimization and unrewindable short-circuit as
        # the async path.
        scope = _capture_scope(request)
        tracker = self._file_changes
        if tracker.unrewindable(capture_path, scope):
            return handler(request)
        before, before_skip = (
            self._snapshot(capture_path) if not tracker.before_known(capture_path, scope) else (None, False)
        )
        # Same ordering rule as the async path: sequence reflects observed
        # content, not call start.
        seq_before = tracker.next_seq()
        result = handler(request)
        seq_after = tracker.next_seq()
        self._record_change(
            capture_path,
            before,
            result,
            request,
            seq_before=seq_before,
            seq_after=seq_after,
            before_skip=before_skip,
        )
        return result
