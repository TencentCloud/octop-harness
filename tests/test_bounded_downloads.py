from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from octop_harness.backends.bounded import DownloadTooLargeError, download_bytes_capped, read_capped
from octop_harness.backends.workspace import BackendWorkspace


class TrackingStream(BytesIO):
    def read(self, size: int = -1) -> bytes:
        assert size >= 0
        return super().read(size)


def test_unknown_length_stream_stops_at_limit_plus_one() -> None:
    source = TrackingStream(b"x" * 1000000)
    with pytest.raises(DownloadTooLargeError):
        read_capped(source, 12)
    assert source.tell() == 13
    assert read_capped(TrackingStream(b"hello"), 5) == b"hello"


def test_unsupported_backend_never_invokes_full_download() -> None:
    backend = SimpleNamespace(download_files=Mock(side_effect=AssertionError("unbounded")))
    with pytest.raises(NotImplementedError):
        download_bytes_capped(backend, "/file.docx", 12)
    backend.download_files.assert_not_called()


def test_workspace_local_read_is_capped(tmp_path: Path) -> None:
    from deepagents.backends import LocalShellBackend

    workspace = BackendWorkspace(LocalShellBackend(root_dir=str(tmp_path)), workspace_dir=tmp_path)
    (tmp_path / "file.docx").write_bytes(b"x" * 1000)
    with pytest.raises(DownloadTooLargeError):
        workspace.download_bytes_capped("file.docx", 12)
    assert workspace.download_bytes_capped("file.docx", 1000) == b"x" * 1000


def test_s3_unknown_length_early_abort_closes_response() -> None:
    pytest.importorskip("botocore")
    from octop_harness.backends.s3_backend import S3Backend

    class Source(TrackingStream):
        consumed = 0

        def read(self, size: int = -1) -> bytes:
            chunk = super().read(size)
            self.consumed += len(chunk)
            return chunk

    source = Source(b"x" * 1000000)
    backend = object.__new__(S3Backend)
    backend._config = SimpleNamespace(bucket="bucket", prefix="")
    backend._client = SimpleNamespace(get_object=lambda **_kw: {"Body": source})
    with pytest.raises(DownloadTooLargeError):
        backend._get_object_bytes("file", max_bytes=12)
    assert source.consumed == 13
    assert source.closed
