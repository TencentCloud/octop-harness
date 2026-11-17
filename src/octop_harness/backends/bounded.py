"""Bounded binary reads: unsupported transports fail closed before downloading."""

from __future__ import annotations

from typing import Any, BinaryIO


class DownloadTooLargeError(ValueError):
    """A source exceeded the caller's byte budget."""


def read_capped(stream: BinaryIO, limit: int) -> bytes:
    """Consume at most limit + one bytes, even when size metadata is absent."""
    parts: list[bytes] = []
    remaining = limit
    while True:
        chunk = stream.read(min(65536, remaining + 1))
        if not chunk:
            return b"".join(parts)
        remaining -= len(chunk)
        if remaining < 0:
            raise DownloadTooLargeError("file exceeds download limit")
        parts.append(chunk)


def download_bytes_capped(backend: Any, path: str, max_bytes: int) -> bytes | None:
    """Read local/composite/cloud sources without an unbounded download fallback."""
    if max_bytes < 0:
        raise ValueError("max_bytes must be nonnegative")
    from deepagents.backends import CompositeBackend

    from octop_harness.backends.cloud_storage_base import CloudStorageBackend

    if isinstance(backend, CompositeBackend):
        child, key = backend._get_backend_and_key(path)
        return download_bytes_capped(child, key, max_bytes)
    if isinstance(backend, CloudStorageBackend):
        import inspect

        if "max_bytes" not in inspect.signature(backend._get_file_data).parameters:
            raise NotImplementedError("cloud backend does not support bounded binary downloads")
        # FileData is a JSON envelope (UTF-8 escaping can expand one byte to six).
        data = backend._get_file_data(path, max_bytes=max_bytes * 6 + 65536)
        if data is None:
            return None
        result = backend._file_data_to_bytes(data)
        if len(result) > max_bytes:
            raise DownloadTooLargeError("file exceeds download limit")
        return result
    from deepagents.backends import FilesystemBackend, LocalShellBackend

    if not isinstance(backend, (FilesystemBackend, LocalShellBackend)):
        raise NotImplementedError("backend does not support bounded binary downloads")
    local = backend._resolve_path(path)
    try:
        with local.open("rb") as stream:
            return read_capped(stream, max_bytes)
    except FileNotFoundError:
        return None
