"""ACP session ids survive a process restart and resume the same agent session."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from octop_harness.acp.models import ACPSessionError
from octop_harness.acp.session_index import (
    AcpSessionIndex,
    resume_or_create_session,
    session_index_path,
)


def test_index_round_trip_and_forget(tmp_path: Path) -> None:
    index = AcpSessionIndex(session_index_path(str(tmp_path)))
    assert index.get("thread-1", "codebuddy") is None
    index.put("thread-1", "codebuddy", "sess-a")
    index.put("thread-1", "opencode", "sess-b")
    assert index.get("thread-1", "codebuddy") == "sess-a"
    index.forget("thread-1", "codebuddy")
    assert index.get("thread-1", "codebuddy") is None
    assert index.get("thread-1", "opencode") == "sess-b"


class _Conn:
    def __init__(self, *, fail_load: bool = False) -> None:
        self.fail_load = fail_load
        self.loaded: str | None = None
        self.created = False

    async def load_session(self, cwd: str, session_id: str, **_: Any) -> None:
        del cwd
        if self.fail_load:
            raise RuntimeError("gone")
        self.loaded = session_id

    async def new_session(self, cwd: str, **_: Any) -> Any:
        del cwd
        self.created = True

        class _Created:
            session_id = "sess-new"

        return _Created()


@pytest.mark.asyncio
async def test_resume_reuses_saved_session() -> None:
    conn = _Conn()
    opened = await resume_or_create_session(
        conn,
        cwd="/work",
        session_id="sess-old",
        load_supported=True,
    )
    assert opened.session_id == "sess-old"
    assert conn.loaded == "sess-old"
    assert conn.created is False


@pytest.mark.asyncio
async def test_resume_falls_back_when_load_fails() -> None:
    conn = _Conn(fail_load=True)
    opened = await resume_or_create_session(
        conn,
        cwd="/work",
        session_id="sess-old",
        load_supported=True,
    )
    assert opened.session_id == "sess-new"
    assert conn.created is True


@pytest.mark.asyncio
async def test_resume_creates_when_agent_cannot_load() -> None:
    conn = _Conn()
    opened = await resume_or_create_session(
        conn,
        cwd="/work",
        session_id="sess-old",
        load_supported=False,
    )
    assert opened.session_id == "sess-new"
    assert conn.loaded is None


@pytest.mark.asyncio
async def test_new_session_without_id_raises() -> None:
    class _Empty:
        async def new_session(self, cwd: str, **_: Any) -> Any:
            del cwd

            class _Created:
                session_id = ""

            return _Created()

    with pytest.raises(ACPSessionError):
        await resume_or_create_session(_Empty(), cwd="/work", session_id=None, load_supported=False)
