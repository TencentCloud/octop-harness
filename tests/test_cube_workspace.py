"""Exercise the object-storage mirror of a CubeSandbox workspace."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from deepagents.backends.protocol import FileDownloadResponse, FileUploadResponse

from octop_harness.backends import CubeSandbox, resolve_backend
from octop_harness.backends import cube_workspace as module
from octop_harness.backends.cube_workspace import CubeWorkspaceBackup, CubeWorkspaceError, default_workspace_prefix


class FakeStorage:
    """Object storage that keeps files in a dict keyed by virtual path."""

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.files = dict(files or {})
        self.uploaded: list[tuple[str, bytes]] = []

    def glob_info(self, pattern: str, path: str = "/") -> list[dict[str, Any]]:
        del pattern
        prefix = path.rstrip("/") or "/"
        return [{"path": key, "is_dir": False} for key in sorted(self.files) if key.startswith(prefix) or prefix == "/"]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return [
            FileDownloadResponse(
                path=path,
                content=self.files.get(path),
                error=None if path in self.files else "file_not_found",
            )
            for path in paths
        ]

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        self.uploaded.extend(files)
        self.files.update(files)
        return [FileUploadResponse(path=path) for path, _ in files]


class FakeSandbox:
    """Minimal sandbox surface used by restore/archive."""

    sandbox_id = "cube-workspace-test"

    def __init__(self, files: dict[str, bytes] | None = None, find_output: str = "") -> None:
        self.files = dict(files or {})
        self.find_output = find_output
        self.written: list[tuple[str, bytes]] = []
        self.commands: list[str] = []

    def kill(self) -> None:
        self.files.clear()

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        self.written.extend(files)
        self.files.update(files)
        return [FileUploadResponse(path=path) for path, _ in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return [
            FileDownloadResponse(
                path=path,
                content=self.files.get(path),
                error=None if path in self.files else "file_not_found",
            )
            for path in paths
        ]

    def execute(self, command: str, *, timeout: int | None = None) -> SimpleNamespace:
        del timeout
        self.commands.append(command)
        return SimpleNamespace(output=self.find_output, exit_code=0)


def test_prefix_and_path_mapping() -> None:
    backup = CubeWorkspaceBackup(storage=FakeStorage(), prefix="cube-workspaces/u/a/", sandbox_dir="/workspace/")
    assert backup.prefix == "/cube-workspaces/u/a"
    assert backup.sandbox_dir == "/workspace"
    assert backup.storage_path("/notes/today.md") == "/cube-workspaces/u/a/notes/today.md"
    assert backup.sandbox_path("notes/today.md") == "/workspace/notes/today.md"
    assert default_workspace_prefix("u", "a", "agent") == "/cube-workspaces/u/a"
    assert default_workspace_prefix("u", "a", "user") == "/cube-workspaces/u/_shared"


def test_restore_only_takes_its_own_prefix() -> None:
    storage = FakeStorage(
        {
            "/cube-workspaces/u/a/notes/today.md": b"hello",
            "/cube-workspaces/u/a/src/main.py": b"print(1)",
            "/cube-workspaces/u/b/notes/today.md": b"other agent",
        }
    )
    sandbox = FakeSandbox()
    backup = CubeWorkspaceBackup(storage=storage, prefix="/cube-workspaces/u/a")

    assert backup.restore(sandbox) == 2
    assert sorted(sandbox.written) == [
        ("/workspace/notes/today.md", b"hello"),
        ("/workspace/src/main.py", b"print(1)"),
    ]


def test_restore_fails_loudly_when_storage_cannot_be_read() -> None:
    storage = FakeStorage({"/cube-workspaces/u/a/ghost.md": b""})
    storage.download_files = lambda paths: [  # type: ignore[method-assign]
        FileDownloadResponse(path=paths[0], content=None, error="access_denied")
    ]
    backup = CubeWorkspaceBackup(storage=storage, prefix="/cube-workspaces/u/a")
    with pytest.raises(CubeWorkspaceError, match="cannot read"):
        backup.restore(FakeSandbox())


def test_archive_maps_sandbox_paths_back_to_storage() -> None:
    sandbox = FakeSandbox(
        files={"/workspace/notes/today.md": b"hello", "/workspace/src/main.py": b"print(1)"},
        find_output="/workspace/notes/today.md\n/workspace/src/main.py\n",
    )
    storage = FakeStorage()
    backup = CubeWorkspaceBackup(storage=storage, prefix="/cube-workspaces/u/a")

    assert backup.archive(sandbox) == 2
    assert sorted(storage.uploaded) == [
        ("/cube-workspaces/u/a/notes/today.md", b"hello"),
        ("/cube-workspaces/u/a/src/main.py", b"print(1)"),
    ]


def test_archive_refuses_a_runaway_workspace() -> None:
    sandbox = FakeSandbox(find_output="/workspace/a\n/workspace/b\n/workspace/c\n")
    backup = CubeWorkspaceBackup(storage=FakeStorage(), prefix="/cube-workspaces/u/a", max_files=2)
    with pytest.raises(CubeWorkspaceError, match="more than 2 files"):
        backup.archive(sandbox)


def test_backup_is_wired_into_a_managed_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_from_spec(spec: dict[str, Any], **kwargs: Any) -> CubeWorkspaceBackup:
        captured.update(spec=spec, **kwargs)
        return CubeWorkspaceBackup(storage=FakeStorage(), prefix=kwargs["prefix"])

    monkeypatch.setattr(module, "workspace_backup_from_spec", fake_from_spec)
    remote = resolve_backend(
        {
            "type": "cubesandbox",
            "sandbox": FakeSandbox(),
            "octop_user": "alice",
            "octop_agent": "bot-1",
            "workspace_backup": {"type": "cos", "bucket": "artifacts"},
        }
    )

    assert isinstance(remote, CubeSandbox)
    assert captured["spec"] == {"type": "cos", "bucket": "artifacts"}
    assert captured["prefix"] == "/cube-workspaces/alice/bot-1"
    assert captured["sandbox_dir"] == "/workspace"
    assert remote._workspace_backup.prefix == "/cube-workspaces/alice/bot-1"


def test_backup_options_reject_incomplete_specs() -> None:
    with pytest.raises(ValueError, match="require a workspace_backup spec"):
        resolve_backend({"type": "cubesandbox", "sandbox": FakeSandbox(), "workspace_backup_prefix": "/x"})
    with pytest.raises(ValueError, match="requires octop_user and octop_agent"):
        resolve_backend({"type": "cubesandbox", "sandbox": FakeSandbox(), "workspace_backup": {"type": "cos"}})
    with pytest.raises(ValueError, match="not both"):
        CubeSandbox(
            sandbox=FakeSandbox(),
            octop_user="alice",
            octop_agent="bot-1",
            restore_workspace=lambda _: None,
            workspace_backup=CubeWorkspaceBackup(storage=FakeStorage(), prefix="/p"),
        )


def test_archive_requires_a_configured_backup() -> None:
    remote = CubeSandbox(sandbox=FakeSandbox())
    assert remote.archives_workspace is False
    with pytest.raises(RuntimeError, match="No workspace_backup"):
        remote.archive_workspace()
    backed = CubeSandbox(
        sandbox=FakeSandbox(),
        octop_user="alice",
        octop_agent="bot-1",
        workspace_backup=CubeWorkspaceBackup(storage=FakeStorage(), prefix="/p"),
    )
    assert backed.archives_workspace is True
