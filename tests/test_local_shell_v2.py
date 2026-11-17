"""Integration checks for the opt-in Windows v2 local-shell executor."""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest
from deepagents.backends import CompositeBackend
from deepagents.backends.protocol import ExecuteResponse

from octop_harness.backends import local_shell, windows_process
from octop_harness.backends.local_shell import HarnessLocalShellBackend

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "nt", "Windows integration test")
class LocalShellV2Tests(unittest.TestCase):
    def make_backend(self) -> HarnessLocalShellBackend:
        return HarnessLocalShellBackend(
            root_dir=str(ROOT),
            workspace_dir=str(ROOT),
            virtual_mode=False,
            windows_executor="v2_file_job",
        )

    def test_environment_switch(self) -> None:
        previous = os.environ.get("OCTOP_WINDOWS_EXECUTOR")
        os.environ["OCTOP_WINDOWS_EXECUTOR"] = "v2_file_job"
        try:
            backend = HarnessLocalShellBackend(
                root_dir=str(ROOT),
                workspace_dir=str(ROOT),
                virtual_mode=False,
            )
            self.assertEqual(backend._windows_executor, "v2_file_job")
        finally:
            if previous is None:
                os.environ.pop("OCTOP_WINDOWS_EXECUTOR", None)
            else:
                os.environ["OCTOP_WINDOWS_EXECUTOR"] = previous

    def test_execute_normal(self) -> None:
        backend = self.make_backend()
        result = backend.execute(
            subprocess.list2cmdline([sys.executable, "-c", "print('v2-normal')"]),
            timeout=5,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("v2-normal", result.output)

    def test_execute_timeout_with_descendant_holding_output(self) -> None:
        child = "import time; print('v2-child', flush=True); time.sleep(60)"
        code = (
            "import subprocess,sys,time; "
            "subprocess.Popen([sys.executable,'-c'," + repr(child) + "], close_fds=False); "
            "time.sleep(60)"
        )
        command = subprocess.list2cmdline([sys.executable, "-c", code])
        backend = self.make_backend()
        started = time.monotonic()
        result = backend.execute(command, timeout=0.5)
        self.assertEqual(result.exit_code, 124)
        self.assertIn("timed out", result.output.lower())
        self.assertLess(time.monotonic() - started, 8)
        cleanup = backend._last_windows_cleanup
        self.assertIsNotNone(cleanup)
        self.assertTrue(cleanup.root_reaped, cleanup)
        self.assertTrue(cleanup.job_empty, cleanup)
        self.assertFalse(cleanup.deadline_exceeded, cleanup)


@pytest.mark.skipif(os.name != "nt", reason="Windows executor")
@pytest.mark.parametrize(("encoding", "limit"), [("utf-8", 8), ("gbk", 5)])
def test_truncated_cjk_output_remains_readable(tmp_path: Path, encoding: str, limit: int) -> None:
    script = tmp_path / "emit.py"
    script.write_text(
        f"import sys\nsys.stdout.buffer.write('中文测试'.encode({encoding!r}))\n",
        encoding="utf-8",
    )
    backend = HarnessLocalShellBackend(
        root_dir=tmp_path, virtual_mode=False, windows_executor="v2_file_job", max_output_bytes=limit
    )

    result = backend.execute(subprocess.list2cmdline([sys.executable, str(script)]), timeout=5)

    assert result.exit_code == 0
    assert result.truncated
    assert result.output.startswith("中文\n")
    assert "\ufffd" not in result.output
    assert result.output.count(f"Output truncated at {limit} bytes.") == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows executor")
def test_cleanup_query_failure_is_reported(tmp_path: Path, monkeypatch, caplog) -> None:
    api = windows_process._WindowsApi()
    monkeypatch.setattr(api, "query", lambda *_: 0)
    monkeypatch.setattr(windows_process, "_WindowsApi", lambda: api)
    backend = HarnessLocalShellBackend(root_dir=tmp_path, virtual_mode=False, windows_executor="v2_file_job")

    with caplog.at_level(logging.WARNING, logger=windows_process.__name__):
        result = backend.execute("echo cleanup", timeout=5)

    assert result.exit_code == 0
    (warning,) = caplog.records
    assert warning.cleanup_errors[0].startswith("query:")
    assert warning.terminated
    assert warning.root_reaped
    assert not warning.deadline_exceeded


@pytest.mark.skipif(os.name != "nt", reason="Windows executor")
@pytest.mark.asyncio
@pytest.mark.parametrize("force_budget_expiry", [False, True], ids=["job-empty", "budget-expired"])
async def test_cancel_shell_waits_for_job_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    force_budget_expiry: bool,
) -> None:
    if force_budget_expiry:
        install_budget_expiry(monkeypatch)
    caplog.set_level(logging.WARNING, logger=windows_process.__name__)
    sentinel = tmp_path / "child-started.txt"
    child_code = (
        f"import os,time; from pathlib import Path; Path({str(sentinel)!r}).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    script = tmp_path / "spawn.py"
    script.write_text(
        f"import subprocess, sys, time\nsubprocess.Popen([sys.executable, '-c', {child_code!r}])\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    backend = HarnessLocalShellBackend(root_dir=tmp_path, virtual_mode=False, windows_executor="v2_file_job")
    composite = CompositeBackend(default=backend, routes={})
    command = subprocess.list2cmdline([sys.executable, str(script)])
    task = asyncio.create_task(composite.aexecute(command, timeout=8))
    try:
        async with asyncio.timeout(5):
            while not sentinel.exists() or not sentinel.read_text().strip():
                await asyncio.sleep(0.02)
        child = psutil.Process(int(sentinel.read_text()))
        started = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        elapsed = time.monotonic() - started
        assert elapsed < 4

        cleanup = backend._last_windows_cleanup
        assert cleanup is not None and cleanup.root_reaped
        assert cleanup.job_empty is not force_budget_expiry
        assert cleanup.deadline_exceeded is force_budget_expiry
        _gone, alive = psutil.wait_procs([child], timeout=2.0)
        assert not alive, alive
        if force_budget_expiry:
            assert 0.15 <= elapsed < 2
            (record,) = caplog.records
            assert record.levelno == logging.WARNING
            assert record.deadline_exceeded
            assert not record.job_empty
            monkeypatch.undo()
        assert (await composite.aexecute("echo recovered", timeout=5)).exit_code == 0
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.skipif(os.name != "nt", reason="Windows executor")
@pytest.mark.asyncio
async def test_repeated_cancel_waits_for_job_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = HarnessLocalShellBackend(root_dir=tmp_path, virtual_mode=False, windows_executor="v2_file_job")
    loop = asyncio.get_running_loop()
    worker_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    allow_cleanup = Event()
    cleanup_done = Event()

    def run(command: str, *, timeout: int | None, cancel_event: Event) -> ExecuteResponse:
        loop.call_soon_threadsafe(worker_started.set)
        assert cancel_event.wait(3)
        loop.call_soon_threadsafe(cleanup_started.set)
        assert allow_cleanup.wait(3)
        cleanup_done.set()
        return ExecuteResponse(output=command, exit_code=130, truncated=False)

    monkeypatch.setattr(backend, "_execute_command", run)
    task = asyncio.create_task(backend.aexecute("echo probe", timeout=8))
    try:
        await asyncio.wait_for(worker_started.wait(), timeout=3)
        task.cancel()
        await asyncio.wait_for(cleanup_started.wait(), timeout=3)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        allow_cleanup.set()
        if not task.done():
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cleanup_done.is_set()


def install_budget_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    api = windows_process._WindowsApi()
    execute_windows = windows_process.execute_windows

    def report_nonempty_job(_job, _kind, data, _size, _output) -> bool:
        ctypes.cast(data, ctypes.POINTER(windows_process.ACCOUNTING)).contents.ActiveProcesses = 1
        return True

    # Termination uses the real Job. Only the accounting response is simulated.
    monkeypatch.setattr(api, "query", report_nonempty_job)
    monkeypatch.setattr(windows_process, "_WindowsApi", lambda: api)

    def execute_with_short_cleanup(command: str, **kwargs):
        return execute_windows(command, cleanup_timeout=0.2, **kwargs)

    monkeypatch.setattr(windows_process, "execute_windows", execute_with_short_cleanup)


@pytest.mark.skipif(os.name != "nt", reason="Windows executor")
@pytest.mark.parametrize("method", ["terminate", "terminate_process", "close"])
def test_cleanup_failures_log_without_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, method: str
) -> None:
    api = windows_process._WindowsApi()
    native = getattr(api, method)

    def release_then_report_failure(*args) -> int:
        assert native(*args)
        ctypes.set_last_error(5)
        return 0

    # Release the real resources before injecting a native failure result.
    monkeypatch.setattr(api, method, release_then_report_failure)
    if method == "terminate_process":
        monkeypatch.setattr(api, "assign", lambda *_: 0)
    monkeypatch.setattr(windows_process, "_WindowsApi", lambda: api)
    backend = HarnessLocalShellBackend(root_dir=tmp_path, virtual_mode=False, windows_executor="v2_file_job")
    command_marker = "ISSUE004_TEST_COMMAND_SHOULD_NOT_APPEAR_IN_WARNING"

    with caplog.at_level(logging.WARNING, logger=windows_process.__name__):
        response = backend.execute(f"echo {command_marker}", timeout=5)

    assert response.exit_code == (1 if method == "terminate_process" else 0)
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert record.cleanup_errors
    error_label = "close" if method == "close" else "terminate"
    assert all(error == f"{error_label}: 5" for error in record.cleanup_errors)
    assert record.root_reaped
    assert not record.deadline_exceeded
    assert command_marker not in caplog.text


@pytest.mark.skipif(os.name != "nt", reason="Windows executor")
def test_job_creation_failure_is_not_a_cleanup_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    api = windows_process._WindowsApi()
    create = Mock(return_value=0)
    terminate = Mock()
    monkeypatch.setattr(api, "create_job", create)
    monkeypatch.setattr(api, "terminate", terminate)
    monkeypatch.setattr(windows_process, "_WindowsApi", lambda: api)
    backend = HarnessLocalShellBackend(root_dir=tmp_path, virtual_mode=False, windows_executor="v2_file_job")

    with caplog.at_level(logging.WARNING, logger=windows_process.__name__):
        response = backend.execute("echo should-not-start", timeout=5)

    assert response.exit_code == 1
    assert "Error executing command" in response.output
    create.assert_called_once()
    terminate.assert_not_called()
    assert not caplog.records


@pytest.mark.skipif(os.name != "nt", reason="Windows executor")
@pytest.mark.parametrize("stdout_truncated", [False, True], ids=["stderr-only", "both-streams"])
def test_emoji_truncation_tracks_streams_separately(tmp_path: Path, stdout_truncated: bool) -> None:
    script = tmp_path / "emit-both.py"
    stdout = "中" * 9 + "🙂tail" if stdout_truncated else ""
    stderr = "文" * 9 + "🙂tail"
    script.write_text(
        f"import sys\nsys.stdout.buffer.write({stdout.encode()!r})\nsys.stderr.buffer.write({stderr.encode()!r})\n",
        encoding="utf-8",
    )
    limit = 29  # 27 bytes of complete characters, then two bytes of a four-byte emoji.
    command = subprocess.list2cmdline([sys.executable, str(script)])
    raw = windows_process.execute_windows(
        command, cwd=str(tmp_path), env=dict(os.environ), timeout=5, max_output_bytes=limit
    )
    assert raw.stdout_truncated is stdout_truncated
    assert raw.stderr_truncated
    backend = HarnessLocalShellBackend(
        root_dir=tmp_path, virtual_mode=False, windows_executor="v2_file_job", max_output_bytes=limit
    )
    result = backend.execute(command, timeout=5)

    assert result.exit_code == 0 and result.truncated
    assert "[stderr] " + "文" * 9 in result.output
    if stdout_truncated:
        assert result.output.startswith("中" * 9 + "\n")
    assert "🙂" not in result.output
    assert "\ufffd" not in result.output
    assert result.output.count(f"Output truncated at {limit} bytes.") == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows executor")
@pytest.mark.asyncio
async def test_cancel_before_job_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = HarnessLocalShellBackend(root_dir=tmp_path, virtual_mode=False, windows_executor="v2_file_job")
    original_execute = backend._execute_command
    api_factory = Mock()
    monkeypatch.setattr(windows_process, "_WindowsApi", api_factory)
    loop = asyncio.get_running_loop()
    worker_started = asyncio.Event()

    def defer_until_cancel(command: str, *, timeout: int | None, cancel_event: Event) -> ExecuteResponse:
        loop.call_soon_threadsafe(worker_started.set)
        assert cancel_event.wait(3)
        return original_execute(command, timeout=timeout, cancel_event=cancel_event)

    monkeypatch.setattr(backend, "_execute_command", defer_until_cancel)
    task = asyncio.create_task(backend.aexecute("echo not-started", timeout=5))
    try:
        await asyncio.wait_for(worker_started.wait(), timeout=3)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    api_factory.assert_not_called()
    cleanup = backend._last_windows_cleanup
    assert cleanup == windows_process.CleanupStatus()


@pytest.mark.asyncio
@pytest.mark.parametrize("pretend_posix", [False, True], ids=["default-executor", "non-windows"])
async def test_legacy_async_cancel_retains_original_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pretend_posix: bool
) -> None:
    monkeypatch.delenv("OCTOP_WINDOWS_EXECUTOR", raising=False)
    backend = HarnessLocalShellBackend(
        root_dir=tmp_path, virtual_mode=False, windows_executor="v2_file_job" if pretend_posix else None
    )
    if pretend_posix:
        monkeypatch.setattr(local_shell, "os", SimpleNamespace(name="posix"))
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    finish = Event()

    def execute(command: str, *, timeout: int | None = None) -> ExecuteResponse:
        loop.call_soon_threadsafe(started.set)
        assert finish.wait(3)
        return ExecuteResponse(output=command, exit_code=0, truncated=False)

    monkeypatch.setattr(backend, "execute", execute)
    task = asyncio.create_task(backend.aexecute("echo legacy", timeout=5))
    try:
        await asyncio.wait_for(started.wait(), timeout=3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
    finally:
        finish.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


if __name__ == "__main__":
    unittest.main()
