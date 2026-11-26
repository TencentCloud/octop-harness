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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("outcome", ["success", "too_large", "unsupported"])
async def test_remote_workspace_never_materializes_unbounded(tmp_path: Path, asynchronous: bool, outcome: str) -> None:
    from octop_harness.backends.s3_backend import S3Backend

    legacy = Mock(side_effect=AssertionError("legacy remote download invoked"))
    source = TrackingStream(b"hello" if outcome == "success" else b"x" * 1000000)
    calls = []
    if outcome == "unsupported":
        backend = SimpleNamespace(download_files=legacy)
    else:
        backend = object.__new__(S3Backend)
        backend._config = SimpleNamespace(prefix="")
        backend.download_files = legacy

        def bounded_source(path: str, *, max_bytes: int | None = None):  # type: ignore[no-untyped-def]
            assert max_bytes is not None
            calls.append((path, max_bytes))
            return backend._parse_body(read_capped(source, max_bytes))

        backend._get_file_data = bounded_source
    # Remote storage must remain authoritative even when a host file exists.
    (tmp_path / "file.docx").write_bytes(b"stale host data")
    workspace = BackendWorkspace(backend, workspace_dir=tmp_path)

    async def download():  # type: ignore[no-untyped-def]
        if asynchronous:
            return await workspace.adownload_bytes_capped("file.docx", 5)
        return workspace.download_bytes_capped("file.docx", 5)

    if outcome == "success":
        assert await download() == b"hello"
        assert len(calls) == 1
    elif outcome == "too_large":
        with pytest.raises(DownloadTooLargeError):
            await download()
        assert len(calls) == 1
        assert source.tell() == calls[0][1] + 1
    else:
        with pytest.raises(NotImplementedError):
            await download()
    legacy.assert_not_called()
    assert not (tmp_path / ".harness-materialize").exists()


def test_materialize_local_keeps_delivery_download_default(tmp_path: Path) -> None:
    from deepagents.backends.protocol import FileDownloadResponse

    legacy = Mock(return_value=[FileDownloadResponse(path="/file.docx", content=b"document", error=None)])
    workspace = BackendWorkspace(SimpleNamespace(download_files=legacy), workspace_dir=tmp_path)
    assert workspace.materialize_local("file.docx", allow_download=False) is None
    legacy.assert_not_called()
    local = workspace.materialize_local("file.docx")
    assert local is not None
    assert local.read_bytes() == b"document"
    legacy.assert_called_once()
