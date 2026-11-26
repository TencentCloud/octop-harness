"""Copy a CubeSandbox workspace to and from an object-storage backend.

A CubeSandbox keeps the agent workspace inside the sandbox's writable layer
(``sandbox_fs``). Pausing preserves it, but a sandbox that is killed, crashes,
or is rebuilt on another node starts empty. This module makes object storage the
durable copy: :meth:`CubeWorkspaceBackup.restore` rehydrates a fresh sandbox and
:meth:`CubeWorkspaceBackup.archive` pushes the workspace out before it is lost.

The storage side is any backend implementing ``download_files`` / ``upload_files``
/ ``glob_info`` — the S3, COS, OSS and OBS backends already do.
"""

from __future__ import annotations

import logging
import posixpath
import shlex
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_READ_BATCH = 32
_EXEC_TIMEOUT = 120


class CubeWorkspaceError(RuntimeError):
    """A workspace copy could not be completed without losing data."""


@dataclass(frozen=True)
class CubeWorkspaceBackup:
    """A storage-side mirror of one sandbox directory.

    Args:
        storage: Object-storage backend (``s3`` / ``cos`` / ``oss`` / ``obs``).
        prefix: Storage prefix that holds this workspace, e.g.
            ``/cube-workspaces/alice/agent-7``.
        sandbox_dir: Absolute directory inside the sandbox to mirror.
        max_files: Refuse to archive more files than this in one pass, so a
            runaway workspace cannot saturate the uplink unnoticed.
    """

    storage: Any
    prefix: str
    sandbox_dir: str = "/workspace"
    max_files: int = 5000

    def __post_init__(self) -> None:
        prefix = "/" + self.prefix.strip("/") if self.prefix.strip("/") else ""
        root = "/" + self.sandbox_dir.strip("/") if self.sandbox_dir.strip("/") else ""
        if not root:
            raise ValueError("sandbox_dir must not be the sandbox root")
        if not prefix:
            raise ValueError("prefix must not be empty: it scopes which stored files are restored")
        if ".." in prefix.split("/") or ".." in root.split("/"):
            raise ValueError("prefix and sandbox_dir must not contain '..'")
        if self.max_files <= 0:
            raise ValueError("max_files must be positive")
        object.__setattr__(self, "prefix", prefix)
        object.__setattr__(self, "sandbox_dir", root)

    # -- paths -------------------------------------------------------------

    def storage_path(self, relative: str) -> str:
        """Storage key for a sandbox-relative path."""
        return posixpath.join(self.prefix, relative.lstrip("/"))

    def sandbox_path(self, relative: str) -> str:
        """Absolute sandbox path for a workspace-relative path."""
        return posixpath.join(self.sandbox_dir, relative.lstrip("/"))

    def _relative(self, storage_key: str) -> str | None:
        base = self.prefix.rstrip("/")
        if not storage_key.startswith(base + "/"):
            return None
        return storage_key[len(base) + 1 :] or None

    # -- restore -----------------------------------------------------------

    def restore(self, sandbox: Any) -> int:
        """Copy the stored workspace into *sandbox*; return the file count.

        Raises:
            CubeWorkspaceError: When the listing or any download fails. A partial
                workspace is worse than a loud failure: the agent would run with
                silent data loss.
        """
        keys = self._list_storage()
        restored = 0
        for start in range(0, len(keys), _READ_BATCH):
            batch = keys[start : start + _READ_BATCH]
            responses = self.storage.download_files([key for key, _ in batch])
            uploads: list[tuple[str, bytes]] = []
            for (key, relative), response in zip(batch, responses, strict=True):
                if response.error or response.content is None:
                    raise CubeWorkspaceError(f"cannot read {key} from storage: {response.error or 'empty'}")
                uploads.append((self.sandbox_path(relative), bytes(response.content)))
            if uploads:
                results = sandbox.upload_files(uploads)
                failed = [item for item in results if item.error]
                if failed:
                    raise CubeWorkspaceError(f"cannot restore {failed[0].path} into the sandbox: {failed[0].error}")
                restored += len(uploads)
        logger.info("Restored %d file(s) from %s into %s", restored, self.prefix, self.sandbox_dir)
        return restored

    def _list_storage(self) -> list[tuple[str, str]]:
        try:
            entries = self.storage.glob_info("**/*", path=self.prefix)
        except Exception as exc:  # storage SDK errors vary by provider
            raise CubeWorkspaceError(f"cannot list {self.prefix}: {exc}") from exc
        keys = sorted(
            entry["path"] for entry in entries if not entry.get("is_dir") and self._relative(entry["path"]) is not None
        )
        pairs: list[tuple[str, str]] = []
        for key in keys:
            relative = self._relative(key)
            if relative is not None:
                pairs.append((key, relative))
        return pairs

    # -- archive -----------------------------------------------------------

    def archive(self, sandbox: Any) -> int:
        """Copy the sandbox workspace into storage; return the file count."""
        paths = self._list_sandbox(sandbox)
        archived = 0
        for start in range(0, len(paths), _READ_BATCH):
            batch = paths[start : start + _READ_BATCH]
            responses = sandbox.download_files(batch)
            uploads: list[tuple[str, bytes]] = []
            for path, response in zip(batch, responses, strict=True):
                if response.error or response.content is None:
                    raise CubeWorkspaceError(f"cannot read {path} from the sandbox: {response.error or 'empty'}")
                relative = (
                    path[len(self.sandbox_dir) :].lstrip("/") if path.startswith(self.sandbox_dir) else path.lstrip("/")
                )
                uploads.append((self.storage_path(relative), bytes(response.content)))
            if uploads:
                results = self.storage.upload_files(uploads)
                failed = [item for item in results if getattr(item, "error", None)]
                if failed:
                    raise CubeWorkspaceError(f"cannot archive {failed[0].path}: {failed[0].error}")
                archived += len(uploads)
        logger.info("Archived %d file(s) from %s to %s", archived, self.sandbox_dir, self.prefix)
        return archived

    def _list_sandbox(self, sandbox: Any) -> list[str]:
        command = (
            f"test -d {shlex.quote(self.sandbox_dir)} && "
            f"find {shlex.quote(self.sandbox_dir)} -type f -print | head -n {self.max_files + 1} || true"
        )
        result = sandbox.execute(command, timeout=_EXEC_TIMEOUT)
        if result.exit_code != 0:
            raise CubeWorkspaceError(f"cannot list {self.sandbox_dir}: {result.output.strip()}")
        paths = [line.strip() for line in result.output.splitlines() if line.strip()]
        if len(paths) > self.max_files:
            raise CubeWorkspaceError(
                f"{self.sandbox_dir} holds more than {self.max_files} files; raise max_files or narrow sandbox_dir"
            )
        return paths


def workspace_backup_from_spec(
    spec: dict[str, Any],
    *,
    prefix: str,
    sandbox_dir: str = "/workspace",
    max_files: int = 5000,
) -> CubeWorkspaceBackup:
    """Build a backup from a nested backend spec such as ``{"type": "cos", ...}``.

    The nested spec goes through the normal resolver, so it accepts exactly the
    same keys as a top-level backend (and needs the matching extra installed).
    """
    from octop_harness.backends import resolve_backend

    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError("workspace_backup must be a backend spec dict with a 'type' key")
    storage = resolve_backend(spec, _scope_host_artifacts=False)
    return CubeWorkspaceBackup(storage=storage, prefix=prefix, sandbox_dir=sandbox_dir, max_files=max_files)


def default_workspace_prefix(octop_user: str, octop_agent: str, octop_scope: str) -> str:
    """Storage prefix shared by every harness replica of one Octop workspace."""
    if octop_scope == "user":
        return f"/cube-workspaces/{octop_user}/_shared"
    return f"/cube-workspaces/{octop_user}/{octop_agent}"


__all__ = [
    "CubeWorkspaceBackup",
    "CubeWorkspaceError",
    "default_workspace_prefix",
    "workspace_backup_from_spec",
]
