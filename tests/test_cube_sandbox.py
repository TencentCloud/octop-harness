"""Exercise CubeSandbox through the public backend resolver with mock SDKs."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from deepagents.backends.protocol import FILE_NOT_FOUND, INVALID_PATH, IS_DIRECTORY, PERMISSION_DENIED

from octop_harness.backends import CubeSandbox, resolve_backend, spec_supports_execution
from octop_harness.backends import cube_sandbox as backend


class MockCommandExitError(Exception):
    stdout = "partial"
    stderr = "failed"
    exit_code = 7


class MockTimeoutError(Exception):
    pass


class MockNotFoundError(Exception):
    pass


@pytest.fixture
def sandbox() -> Mock:
    mock = Mock(sandbox_id="cube-test")
    mock.commands.run.return_value = SimpleNamespace(stdout="", stderr="", exit_code=0)
    mock.files.read.return_value = bytearray(b"\x89\x00")
    return mock


def test_resolve_and_run(sandbox: Mock) -> None:
    spec = {"type": "cubesandbox", "sandbox": sandbox, "command_timeout": 45}
    remote = resolve_backend(spec)
    assert isinstance(remote, CubeSandbox)
    assert remote.sandbox_fs and remote.virtual_mode and spec_supports_execution(spec)
    sandbox.commands.run.return_value = SimpleNamespace(stdout="out\n", stderr="err\n", exit_code=3)
    response = remote.execute("test command")
    assert (response.output, response.exit_code) == ("out\nerr", 3)
    sandbox.commands.run.assert_called_once_with("test command", timeout=45)


def test_binary_io_and_path_errors(sandbox: Mock) -> None:
    remote = CubeSandbox(sandbox=sandbox)
    assert [r.error for r in remote.upload_files([("/a/b.png", b"\x89\x00"), ("../escape", b"x")])] == [
        None,
        INVALID_PATH,
    ]
    sandbox.files.write.assert_called_once_with("/a/b.png", b"\x89\x00")
    response = remote.download_files(["/a/b.png", "/a/../etc", "//a", "/a\x00z"])
    assert response[0].content == b"\x89\x00"
    assert [r.error for r in response] == [None, INVALID_PATH, INVALID_PATH, INVALID_PATH]
    sandbox.files.read.assert_called_once_with("/a/b.png", format="bytes")


def test_file_errors_and_quoted_mutations(sandbox: Mock) -> None:
    remote = CubeSandbox(sandbox=sandbox)
    sandbox.files.read.side_effect = [IsADirectoryError(), FileNotFoundError(), PermissionError()]
    sandbox.commands.run.return_value = SimpleNamespace(stdout="", stderr="", exit_code=1)
    assert [r.error for r in remote.download_files(["/dir", "/missing", "/denied"])] == [
        IS_DIRECTORY,
        FILE_NOT_FOUND,
        PERMISSION_DENIED,
    ]
    sandbox.commands.run.return_value.exit_code = 0
    remote.mkdir_path("/a b")
    remote.move_path("/a b/x", "/new dir/y")
    remote.delete_path("/new dir")
    assert sandbox.commands.run.call_args_list[-3:] == [
        call("mkdir -p -- '/a b'", timeout=120),
        call("mkdir -p -- '/new dir' && mv -- '/a b/x' '/new dir/y'", timeout=120),
        call("rm -rf -- '/new dir'", timeout=120),
    ]
    with pytest.raises(ValueError, match=r"sandbox root"):
        remote.delete_path("/")


def test_native_missing_optional_file(sandbox: Mock) -> None:
    remote = CubeSandbox(sandbox=sandbox)
    sandbox.files.read.side_effect = RuntimeError("Failed to read /SOUL.md: path '/SOUL.md' does not exist")
    sandbox.commands.run.return_value = SimpleNamespace(stdout="", stderr="", exit_code=1)
    assert remote.download_files(["/SOUL.md"])[0].error == FILE_NOT_FOUND


def test_native_is_preferred_and_e2b_is_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    mocked = Mock(__name__="e2b")
    calls = []

    def load(name: str) -> Mock:
        calls.append(name)
        if name == "cubesandbox":
            raise ModuleNotFoundError(name=name)
        return mocked

    monkeypatch.setattr(backend.importlib, "import_module", load)
    assert backend._load_sdk(None) is mocked
    assert calls == ["cubesandbox", "e2b"]


def test_e2b_command_exit_and_timeout(monkeypatch: pytest.MonkeyPatch, sandbox: Mock) -> None:
    sdk = SimpleNamespace(
        __name__="e2b",
        Sandbox=Mock(),
        CommandExitException=MockCommandExitError,
        TimeoutException=MockTimeoutError,
    )
    sdk.Sandbox.create.return_value = sandbox
    monkeypatch.setattr(backend, "_load_sdk", lambda _: sdk)
    remote = CubeSandbox(api_url="https://cube.example", template="base")
    sdk.Sandbox.create.assert_called_once_with(
        api_url="https://cube.example",
        template="base",
        timeout=300,
        metadata=None,
        lifecycle={"on_timeout": "pause", "auto_resume": True},
    )
    sandbox.commands.run.side_effect = [MockCommandExitError(), MockTimeoutError()]
    assert remote.execute("failed").output == "partial\nfailed"
    timeout = remote.execute("slow", timeout=2)
    assert timeout.exit_code == 124
    assert timeout.output == "Command timed out after 2s"


@pytest.fixture
def managed_sdk(monkeypatch: pytest.MonkeyPatch, sandbox: Mock) -> Mock:
    lock = Lock()

    @contextmanager
    def creation_lock(_dsn: str, _identity: dict[str, str]):
        with lock:
            yield

    sdk = Mock(__name__="cubesandbox", SandboxNotFoundError=MockNotFoundError)
    records = []

    def create(**kwargs: object) -> Mock:
        records.append({"sandboxID": sandbox.sandbox_id, "metadata": kwargs["metadata"]})
        return sandbox

    sdk.Sandbox.create.side_effect = create
    sdk.Sandbox.list_v2.side_effect = lambda **_: list(records)
    sdk.Sandbox.connect.return_value = sandbox
    monkeypatch.setattr(backend, "_load_sdk", lambda _: sdk)
    monkeypatch.setattr(backend, "_creation_lock", creation_lock)
    return sdk


def _open_managed() -> CubeSandbox:
    return CubeSandbox(octop_user="user", octop_agent="agent", lock_dsn="fixture")


def test_concurrent_open_and_reload_reuse_one_sandbox(managed_sdk: Mock, sandbox: Mock) -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        one, two = list(pool.map(lambda _: _open_managed(), range(2)))
    assert one.id == two.id
    assert managed_sdk.Sandbox.create.call_count == 1
    assert managed_sdk.Sandbox.create.call_args.kwargs["metadata"] == {
        "octop_user": "user",
        "octop_agent": "agent",
        "octop_scope": "agent",
    }
    one.close()
    sandbox.close.assert_called_once()
    sandbox.kill.assert_not_called()
    assert _open_managed().download_files(["/marker"])[0].content == b"\x89\x00"
    two.destroy()
    sandbox.kill.assert_called_once()


def test_missing_sandbox_is_recreated_and_workspace_restored(managed_sdk: Mock) -> None:
    _open_managed().close()
    managed_sdk.Sandbox.connect.side_effect = MockNotFoundError()
    restore = Mock()
    remote = CubeSandbox(octop_user="user", octop_agent="agent", lock_dsn="fixture", restore_workspace=restore)
    assert managed_sdk.Sandbox.create.call_count == 2
    restore.assert_called_once_with(remote)


def test_connect_errors_do_not_duplicate_sandbox(managed_sdk: Mock) -> None:
    _open_managed().close()
    managed_sdk.Sandbox.connect.side_effect = PermissionError("denied")
    with pytest.raises(PermissionError, match=r"denied"):
        _open_managed()
    assert managed_sdk.Sandbox.create.call_count == 1


def test_reuse_only_never_creates_a_sandbox(managed_sdk: Mock) -> None:
    with pytest.raises(backend.CubeSandboxMissingError, match=r"No CubeSandbox"):
        CubeSandbox(octop_user="user", octop_agent="agent", lock_dsn="fixture", reuse_only=True)
    managed_sdk.Sandbox.create.assert_not_called()


def _postgres_worker(dsn: str, state: Path, barrier: Barrier) -> None:
    sdk = Mock(__name__="cubesandbox")

    def discover(**_kwargs: object) -> list[dict[str, object]]:
        return json.loads(state.read_text(encoding="utf-8")) if state.exists() else []

    def create(**kwargs: object) -> Mock:
        with state.with_suffix(".created").open("a", encoding="utf-8") as log:
            log.write("create\n")
        time.sleep(0.2)
        records = discover()
        records.append({"sandboxID": "cube-test", "metadata": kwargs["metadata"]})
        state.write_text(json.dumps(records), encoding="utf-8")
        return Mock(sandbox_id="cube-test")

    sdk.Sandbox.list_v2.side_effect = discover
    sdk.Sandbox.create.side_effect = create
    sdk.Sandbox.connect.return_value = Mock(sandbox_id="cube-test")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backend, "_load_sdk", lambda _: sdk)
        barrier.wait(timeout=10)
        CubeSandbox(octop_user=state.parent.name, octop_agent="agent", lock_dsn=dsn).close()


@pytest.mark.skipif(not os.environ.get("TEST_POSTGRES_DSN"), reason="requires a disposable Postgres test database")
def test_postgres_lock_serializes_processes(tmp_path: Path) -> None:
    pytest.importorskip("psycopg")
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    state = tmp_path / "sandboxes.json"
    dsn = os.environ["TEST_POSTGRES_DSN"]
    workers = [ctx.Process(target=_postgres_worker, args=(dsn, state, barrier)) for _ in range(2)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(20)
        assert [worker.exitcode for worker in workers] == [0, 0]
        assert state.with_suffix(".created").read_text(encoding="utf-8").splitlines() == ["create"]
        assert len(json.loads(state.read_text(encoding="utf-8"))) == 1
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
