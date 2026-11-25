"""CubeSandbox shell and virtual filesystem backend for Deep Agents."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shlex
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import PurePosixPath
from typing import Any, cast

from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    INVALID_PATH,
    IS_DIRECTORY,
    PERMISSION_DENIED,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox


def _load_sdk(preferred: str | None) -> Any:
    if preferred not in (None, "cubesandbox", "e2b"):
        raise ValueError("sdk must be 'cubesandbox' or 'e2b'")
    for name in (preferred,) if preferred else ("cubesandbox", "e2b"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    raise ImportError("Install octop-harness[cubesandbox] or octop-harness[cubesandbox-native].")


def _virtual_path(path: str) -> PurePosixPath:
    parsed = PurePosixPath(path)
    if parsed.anchor != "/" or ".." in parsed.parts or "\x00" in path or "\\" in path:
        raise ValueError(f"Invalid sandbox path: {path!r}")
    return parsed


class CubeSandboxMissingError(LookupError):
    """No remote sandbox matches the requested Octop workspace."""


class CubeSandbox(BaseSandbox):
    """Run commands and file operations in a CubeSandbox sandbox.

    The native SDK takes precedence when installed. Each backend receives its
    own connection settings. Closing the backend releases the local handle.
    """

    sandbox_fs = True
    virtual_mode = True

    def __init__(
        self,
        *,
        template: str | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
        sandbox_domain: str | None = None,
        proxy_node_ip: str | None = None,
        timeout: int = 300,
        command_timeout: int = 120,
        sdk: str | None = None,
        sandbox: Any | None = None,
        octop_user: str | None = None,
        octop_agent: str | None = None,
        octop_scope: str = "agent",
        lock_dsn: str | None = None,
        restore_workspace: Callable[[CubeSandbox], None] | None = None,
        workspace_backup: Any | None = None,
        reuse_only: bool = False,
    ) -> None:
        if timeout <= 0 or command_timeout <= 0:
            raise ValueError("Sandbox and command timeouts must be positive")
        self._sdk = _load_sdk(sdk) if sandbox is None else None
        self._command_timeout = command_timeout
        if octop_scope not in {"agent", "user"}:
            raise ValueError("octop_scope must be 'agent' or 'user'")
        if bool(octop_user) != bool(octop_agent):
            raise ValueError("octop_user and octop_agent must be configured together")
        if workspace_backup is not None:
            if restore_workspace is not None:
                raise ValueError("Pass either restore_workspace or workspace_backup, not both")
            if not (octop_user and octop_agent):
                raise ValueError("workspace_backup requires managed sandbox metadata (octop_user and octop_agent)")
            restore_workspace = workspace_backup.restore
        self._workspace_backup = workspace_backup
        self._connection = self._connection_options(api_url, api_key, sandbox_domain, proxy_node_ip)
        if sandbox is not None:
            self._attach(sandbox)
        elif octop_user and octop_agent:
            metadata = {"octop_user": octop_user, "octop_agent": octop_agent, "octop_scope": octop_scope}
            self._attach_managed(template, timeout, metadata, lock_dsn, restore_workspace, reuse_only)
        else:
            if restore_workspace is not None or reuse_only:
                raise ValueError("Workspace restoration and reuse require managed sandbox metadata")
            self._attach(self._create(template, timeout, None))

    def _connection_options(
        self,
        api_url: str | None,
        api_key: str | None,
        sandbox_domain: str | None,
        proxy_node_ip: str | None,
    ) -> dict[str, Any]:
        if self._sdk is None:
            return {}
        endpoint = api_url or os.environ.get("CUBE_API_URL") or os.environ.get("E2B_API_URL")
        options = {"api_url": endpoint, "api_key": api_key}
        connection = {name: value for name, value in options.items() if value is not None}
        if self._sdk.__name__ == "cubesandbox":
            # The native SDK documents CUBE_PROXY_NODE_IP; honour it so a host
            # reached over a tunnel needs no per-agent spec keys.
            proxy_node_ip = proxy_node_ip or os.environ.get("CUBE_PROXY_NODE_IP")
            if sandbox_domain is not None:
                connection["sandbox_domain"] = sandbox_domain
            if proxy_node_ip is not None:
                connection["proxy_node_ip"] = proxy_node_ip
            return {"config": self._sdk.Config(**connection)}
        if not endpoint:
            raise ValueError("Set api_url, CUBE_API_URL or E2B_API_URL to the CubeSandbox endpoint")
        if proxy_node_ip is not None:
            raise ValueError("proxy_node_ip requires the native CubeSandbox SDK")
        if sandbox_domain is not None:
            connection["domain"] = sandbox_domain
        return connection

    def _attach_managed(
        self,
        template: str | None,
        timeout: int,
        metadata: dict[str, str],
        lock_dsn: str | None,
        restore_workspace: Callable[[CubeSandbox], None] | None,
        reuse_only: bool,
    ) -> None:
        dsn = lock_dsn or os.environ.get("OCTOP_CUBESANDBOX_LOCK_DSN")
        if not dsn:
            raise ValueError("Managed CubeSandbox requires OCTOP_CUBESANDBOX_LOCK_DSN or lock_dsn")
        identity = dict(metadata)
        if metadata["octop_scope"] == "user":
            del identity["octop_agent"]
        with _creation_lock(dsn, identity):
            sandbox = self._connect_existing(identity)
            if sandbox is not None:
                self._attach(sandbox)
                return
            if reuse_only:
                raise CubeSandboxMissingError("No CubeSandbox exists for this Octop workspace")
            self._attach(self._create(template, timeout, metadata))
            if restore_workspace is not None:
                try:
                    restore_workspace(self)
                except Exception:
                    self.destroy()
                    raise

    def _attach(self, sandbox: Any) -> None:
        self._sandbox: Any = sandbox
        self._id = str(sandbox.sandbox_id)
        self._kill = sandbox.kill

    def _create(self, template: str | None, timeout: int, metadata: dict[str, str] | None) -> Any:
        assert self._sdk is not None
        return self._sdk.Sandbox.create(
            template=template,
            timeout=timeout,
            metadata=metadata,
            lifecycle={"on_timeout": "pause", "auto_resume": True},
            **self._connection,
        )

    def _connect_existing(self, identity: dict[str, str]) -> Any:
        assert self._sdk is not None
        sandbox_cls = self._sdk.Sandbox
        if self._sdk.__name__ == "cubesandbox":
            # The v2 view includes paused sandboxes and paginates across the fleet.
            candidates = [
                item["sandboxID"]
                for item in sandbox_cls.list_v2(**self._connection)
                if all((item.get("metadata") or {}).get(key) == value for key, value in identity.items())
            ]
        else:
            pager = sandbox_cls.list(query=self._sdk.SandboxQuery(metadata=identity), **self._connection)
            candidates = []
            while pager.has_next:
                candidates.extend(
                    item.sandbox_id
                    for item in pager.next_items()
                    if all((item.metadata or {}).get(key) == value for key, value in identity.items())
                )
        if len(candidates) > 1:
            raise RuntimeError("Multiple CubeSandbox instances match this Octop workspace")
        if not candidates:
            return None
        try:
            return sandbox_cls.connect(candidates[0], **self._connection)
        except Exception as exc:
            # Authorization, transport and busy errors must not cause a duplicate.
            if not self._sdk_error(exc, "SandboxNotFoundError", "SandboxNotFoundException"):
                raise
            return None

    @property
    def sandbox_workspace_dir(self) -> str:
        """Return the persistent workspace path inside the guest."""
        return self._workspace_backup.sandbox_dir if self._workspace_backup is not None else "/workspace"

    @property
    def id(self) -> str:
        """Return the stable ID of the remote sandbox."""
        return self._id

    def _handle(self) -> Any:
        if self._sandbox is None:
            raise RuntimeError("CubeSandbox handle is closed")
        return self._sandbox

    def _sdk_error(self, exc: Exception, *names: str) -> bool:
        return self._sdk is not None and any(
            isinstance(exc, error_type)
            for name in names
            if isinstance(error_type := getattr(self._sdk, name, None), type)
        )

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run a shell command and preserve its exit status and output."""
        seconds = self._command_timeout if timeout is None else timeout
        try:
            result = self._handle().commands.run(command, timeout=seconds)
        except Exception as exc:
            if (
                isinstance(exc, TimeoutError)
                or self._sdk_error(exc, "TimeoutException")
                or (self._sdk is not None and self._sdk.__name__ == "cubesandbox" and _native_timeout(exc))
            ):
                return ExecuteResponse(output=f"Command timed out after {seconds}s", exit_code=124, truncated=False)
            if not self._sdk_error(exc, "CommandExitException"):
                raise
            result = exc

        output = "\n".join(part.rstrip("\n") for part in (result.stdout, result.stderr) if part)
        return ExecuteResponse(output=output, exit_code=int(result.exit_code), truncated=False)

    def mkdir_path(self, path: str) -> None:
        """Create a directory and any missing parents."""
        self._run_file_command(f"mkdir -p -- {shlex.quote(str(_virtual_path(path)))}")

    def move_path(self, src: str, dest: str) -> None:
        """Move a file or directory and create missing destination parents."""
        source, target = _virtual_path(src), _virtual_path(dest)
        if source == PurePosixPath("/") or target == PurePosixPath("/"):
            raise ValueError("Cannot move the sandbox root")
        self._run_file_command(
            f"mkdir -p -- {shlex.quote(str(target.parent))} && "
            f"mv -- {shlex.quote(str(source))} {shlex.quote(str(target))}"
        )

    def delete_path(self, path: str) -> None:
        """Remove a file or directory tree."""
        target = _virtual_path(path)
        if target == PurePosixPath("/"):
            raise ValueError("Cannot delete the sandbox root")
        self._run_file_command(f"rm -rf -- {shlex.quote(str(target))}")

    def _run_file_command(self, command: str) -> None:
        result = self.execute(command)
        if result.exit_code != 0:
            raise OSError(result.output)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Write binary files and report each path independently."""
        fs = self._handle().files
        responses = []
        for path, content in files:
            try:
                target = _virtual_path(path)
            except ValueError:
                responses.append(FileUploadResponse(path=path, error=INVALID_PATH))
                continue
            try:
                self.mkdir_path(str(target.parent))
                fs.write(str(target), content)
            except Exception as exc:
                responses.append(FileUploadResponse(path=path, error=self._file_error(target, exc)))
            else:
                responses.append(FileUploadResponse(path=path))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Read binary files and report each path independently."""
        fs = self._handle().files
        responses = []
        for path in paths:
            try:
                target = _virtual_path(path)
            except ValueError:
                responses.append(FileDownloadResponse(path=path, content=None, error=INVALID_PATH))
                continue
            try:
                content = bytes(fs.read(str(target), format="bytes"))
            except Exception as exc:
                responses.append(FileDownloadResponse(path=path, content=None, error=self._file_error(target, exc)))
            else:
                responses.append(FileDownloadResponse(path=path, content=content))
        return responses

    def _file_error(self, path: PurePosixPath, exc: Exception) -> str:
        message = str(exc)
        lower = message.lower()
        if isinstance(exc, IsADirectoryError) or "is a directory" in lower:
            return IS_DIRECTORY
        if isinstance(exc, PermissionError) or "permission denied" in lower:
            return PERMISSION_DENIED
        if (
            isinstance(exc, FileNotFoundError)
            or self._sdk_error(exc, "FilesystemNotFoundError", "FileNotFoundException")
            or "no such file" in lower
            or "file not found" in lower
            or ("does not exist" in lower and str(path).lower() in lower)
        ):
            # Some envd versions report a directory as not found when opened as a file.
            quoted = shlex.quote(str(path))
            try:
                if self.execute(f"test -d {quoted}").exit_code == 0:
                    return IS_DIRECTORY
            except Exception:
                pass
            return FILE_NOT_FOUND
        return message or type(exc).__name__

    def close(self) -> None:
        """Release the local SDK handle and leave the remote sandbox alive."""
        sandbox, self._sandbox = self._sandbox, None
        closer = getattr(sandbox, "close", None)
        if callable(closer):
            closer()

    def destroy(self) -> None:
        """Explicitly terminate the remote sandbox."""
        try:
            self._kill()
        except Exception as exc:
            if not self._sdk_error(exc, "SandboxNotFoundError", "SandboxNotFoundException"):
                raise
        self.close()

    @property
    def archives_workspace(self) -> bool:
        """Whether a workspace mirror is configured for this sandbox."""
        return self._workspace_backup is not None

    def archive_workspace(self) -> int:
        """Copy the sandbox workspace to object storage; return the file count.

        Call this before the sandbox is destroyed, and periodically while it
        runs: the sandbox's writable layer is the only copy until it lands in
        storage.
        """
        if self._workspace_backup is None:
            raise RuntimeError("No workspace_backup configured for this CubeSandbox")
        return int(self._workspace_backup.archive(self))


@contextmanager
def _creation_lock(dsn: str, identity: dict[str, str]) -> Iterator[None]:
    """Serialize discovery and creation across harness processes."""
    import psycopg

    payload = json.dumps(identity, sort_keys=True).encode()
    digest = hashlib.sha256(b"octop:cubesandbox:" + payload).digest()
    key = int.from_bytes(digest[:8], byteorder="big", signed=True)
    connect = cast(Callable[..., Any], psycopg.connect)
    with connect(dsn, connect_timeout=10) as connection:
        connection.execute("SET LOCAL lock_timeout = '60s'")
        connection.execute("SELECT pg_advisory_xact_lock(%s)", (key,))
        yield


def _native_timeout(exc: Exception) -> bool:
    if isinstance(exc, RuntimeError):
        return str(exc).lower().startswith(("deadline_exceeded:", "command failed: http 504"))
    import httpx

    return isinstance(exc, httpx.TimeoutException)


__all__ = ["CubeSandbox", "CubeSandboxMissingError"]
