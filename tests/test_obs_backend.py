"""Offline tests for the Huawei OBS client construction."""

from __future__ import annotations

import socket
import sys
import types

import pytest

from octop_harness.backends.obs_backend import ObsBackend, ObsConfig


class _RecordingObsClient:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _config(**overrides: object) -> ObsConfig:
    base: dict[str, object] = {
        "bucket": "bucket",
        "endpoint": "obs.cn-north-4.myhuaweicloud.com",
        "access_key_id": "ak",
        "secret_access_key": "sk",
    }
    base.update(overrides)
    return ObsConfig(**base)  # type: ignore[arg-type]


@pytest.fixture
def recording_obs(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("obs")
    module.ObsClient = _RecordingObsClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "obs", module)


def test_build_client_passes_scalar_socket_timeout(recording_obs: None) -> None:
    del recording_obs
    backend = ObsBackend(_config(connect_timeout=30, socket_timeout=45))
    timeout = backend._client.kwargs["timeout"]  # type: ignore[attr-defined]
    assert timeout == 45
    # http.client applies this value via socket.settimeout. A tuple raises
    # TypeError: 'tuple' object cannot be interpreted as an integer.
    sock = socket.socket()
    try:
        sock.settimeout(timeout)
        assert sock.gettimeout() == 45
    finally:
        sock.close()


def test_build_client_default_timeout_is_socket_timeout(recording_obs: None) -> None:
    del recording_obs
    backend = ObsBackend(_config())
    assert backend._client.kwargs["timeout"] == 60  # type: ignore[attr-defined]
