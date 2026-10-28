"""Memory workers retain their database until shutdown work has finished (#484)."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from octop_harness.config import HarnessAgentConfig
from octop_harness.memory.runtime import MemoryRuntime
from octop_harness.memory.store import memory_identity, shared_memory_store
from octop_harness.middleware import memory as memory_module


@pytest.fixture
def queued(monkeypatch: pytest.MonkeyPatch) -> list[Callable[[], None]]:
    jobs: list[Callable[[], None]] = []
    monkeypatch.setattr(memory_module._BackgroundExecutor, "submit", lambda self, fn, **kwargs: jobs.append(fn))
    return jobs


def make_runtime(tmp_path: Path) -> MemoryRuntime:
    config = HarnessAgentConfig(
        name="shutdown_test",
        workspace_dir=tmp_path,
        memory_aux_model_enabled=False,
        memory_extract_on_session_end=True,
        session_log_enabled=False,
    )
    return MemoryRuntime(config=config, workspace_path=tmp_path, model_factory=MagicMock())


def middleware(runtime: MemoryRuntime) -> memory_module.MemoryMiddleware:
    result = runtime.build_middleware()
    assert result is not None
    result._stop_maintenance_timer()
    return result


def capture(
    mw: memory_module.MemoryMiddleware, text: str = "Project Aurora uses PostgreSQL for durable storage."
) -> None:
    mw._submit_capture(
        [HumanMessage(content=text), AIMessage(content="I recorded the project requirement.")],
        {"thread_id": "shutdown-thread", "session_id": "shutdown-session"},
    )


def assert_closed(memory: Any) -> None:
    with pytest.raises(sqlite3.ProgrammingError):
        memory.backend.list_raw(limit=1)


def test_close_keeps_queued_capture_alive_and_persists_it(tmp_path: Path, queued: list[Callable[[], None]]) -> None:
    runtime = make_runtime(tmp_path)
    memory = runtime.memory
    mw = middleware(runtime)
    capture(mw)
    runtime.close()
    try:
        # The accepted worker must still reach SQLite after the owner closes.
        queued.pop(0)()
        assert_closed(memory)
        with closing(sqlite3.connect(tmp_path / "memory.sqlite")) as conn:
            assert conn.execute("SELECT count(*) FROM shutdown_test_raw_events").fetchone()[0] == 2
        capture(mw, "This late submission must not reach the database.")
        assert queued == []
    finally:
        runtime.close()


def test_close_does_not_wait_for_a_running_worker(tmp_path: Path, queued: list[Callable[[], None]]) -> None:
    runtime = make_runtime(tmp_path)
    memory = runtime.memory
    mw = middleware(runtime)
    entered, release = threading.Event(), threading.Event()
    errors: list[BaseException] = []

    def work() -> None:
        entered.set()
        assert release.wait(5)
        memory.backend.list_candidates(limit=1)

    mw._submit_background(work, label="maintenance")

    def run() -> None:
        try:
            queued.pop(0)()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(2)
        runtime.close()
        memory.backend.list_raw(limit=1)
    finally:
        release.set()
        worker.join(timeout=3)
        runtime.close()
    assert not worker.is_alive()
    assert errors == []
    assert_closed(memory)


def test_close_includes_workers_from_replaced_middleware(tmp_path: Path, queued: list[Callable[[], None]]) -> None:
    runtime = make_runtime(tmp_path)
    memory = runtime.memory
    old = middleware(runtime)
    capture(old)
    runtime._config.memory_recall_inject_enabled = False
    new = middleware(runtime)
    assert new is not old
    capture(new)
    runtime.close()
    try:
        queued.pop(1)()
        memory.backend.list_raw(limit=1)
        queued.pop(0)()
        assert_closed(memory)
    finally:
        runtime.close()


def test_end_session_before_graph_build_is_included_in_close(
    tmp_path: Path, queued: list[Callable[[], None]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = make_runtime(tmp_path)
    memory = runtime.memory
    calls: list[str] = []

    def extract(session: str, **kwargs: object) -> dict[str, int]:
        memory.backend.list_raw(limit=1)
        calls.append(session)
        return {"candidates": 0}

    monkeypatch.setattr(runtime.service, "extract", extract)
    runtime.end_session("before-build")
    runtime.close()
    try:
        queued.pop(0)()
        assert calls == ["before-build"]
        assert_closed(memory)
    finally:
        runtime.close()


def test_pending_close_does_not_release_another_runtime_claim(tmp_path: Path, queued: list[Callable[[], None]]) -> None:
    first, second = make_runtime(tmp_path), make_runtime(tmp_path)
    memory = first.memory
    assert second.memory is memory
    identity = memory_identity(first._config, tmp_path)
    assert identity is not None
    capture(middleware(first))
    first.close()
    first.close()
    try:
        second.close()
        assert shared_memory_store().refs(identity) == 1
        queued.pop(0)()
        assert shared_memory_store().refs(identity) == 0
        assert_closed(memory)
    finally:
        first.close()
        second.close()


def test_failed_worker_still_releases_memory(tmp_path: Path, queued: list[Callable[[], None]]) -> None:
    runtime = make_runtime(tmp_path)
    memory = runtime.memory

    def fail() -> None:
        memory.backend.list_raw(limit=1)
        raise ValueError("worker failed")

    middleware(runtime)._submit_background(fail, label="capture")
    runtime.close()
    try:
        with pytest.raises(ValueError, match="worker failed"):
            queued.pop(0)()
        assert_closed(memory)
    finally:
        runtime.close()


def test_close_during_failed_worker_submission_releases_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = make_runtime(tmp_path)
    memory = runtime.memory
    mw = middleware(runtime)

    def reject(self: object, fn: Callable[[], None], **kwargs: object) -> None:
        runtime.close()
        memory.backend.list_raw(limit=1)
        raise RuntimeError("thread could not start")

    monkeypatch.setattr(memory_module._BackgroundExecutor, "submit", reject)
    try:
        with pytest.raises(RuntimeError, match="thread could not start"):
            capture(mw)
        assert_closed(memory)
    finally:
        runtime.close()


def test_worker_can_close_its_own_runtime(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    memory = runtime.memory
    mw = middleware(runtime)

    def work() -> None:
        runtime.close()
        memory.backend.list_raw(limit=1)
        assert runtime.build_middleware() is None

    try:
        assert mw._submit_background(work, label="extract", background=False)
        assert_closed(memory)
        assert not mw._submit_background(work, label="extract", background=False)
    finally:
        runtime.close()
