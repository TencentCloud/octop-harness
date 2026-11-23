"""Harness MCP specs become ACP session servers, and a changed set is reloaded."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from octop_harness.acp.managed import append_prompt_note, managed_prompt_note, managed_turn_mcp
from octop_harness.acp.mcp_servers import acp_mcp_servers_from_specs, mcp_fingerprint
from octop_harness.acp.models import ACPConfig, ACPSessionError
from octop_harness.acp.service import ACPService, _Conversation
from octop_harness.acp.session_index import resume_or_create_session


def test_specs_become_http_sse_and_stdio_servers() -> None:
    servers = acp_mcp_servers_from_specs(
        {
            "notes": {"transport": "streamable_http", "url": "https://notes.example/mcp"},
            "mail": {
                "transport": "sse",
                "url": "https://mail.example/sse",
                "headers": {"Authorization": "Bearer t"},
            },
            "yuque": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "yuque-mcp"],
                "env": {"YUQUE_PERSONAL_TOKEN": "secret"},
            },
            "placeholder": {},
        }
    )
    by_name = {server.name: server for server in servers}
    assert list(by_name) == ["mail", "notes", "yuque"]
    assert by_name["notes"].type == "http"
    assert by_name["notes"].url == "https://notes.example/mcp"
    assert by_name["mail"].type == "sse"
    assert by_name["mail"].headers[0].value == "Bearer t"
    assert by_name["yuque"].command == "npx"
    assert any(item.name == "YUQUE_PERSONAL_TOKEN" for item in by_name["yuque"].env)


def test_prompt_note_comes_from_configurable() -> None:
    config = {"configurable": {"acp_mcp_servers": {"notes": {"transport": "http"}}, "acp_prompt_note": "  @分析师  "}}
    assert managed_turn_mcp(config) == {"notes": {"transport": "http"}}
    assert managed_prompt_note(config) == "@分析师"
    assert append_prompt_note("看一下", "@分析师") == "看一下\n\n@分析师"
    assert append_prompt_note("", "@分析师") == ""


class _Conn:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.fail_first = fail_first
        self.calls: list[tuple[str, list[str]]] = []

    async def load_session(self, **kwargs: Any) -> None:
        raise RuntimeError("no session")

    async def new_session(self, **kwargs: Any) -> SimpleNamespace:
        servers = kwargs.get("mcp_servers")
        names = [str(getattr(item, "name", "")) for item in servers] if isinstance(servers, list) else []
        self.calls.append(("new", names))
        if self.fail_first and names:
            raise RuntimeError("mcp rejected")
        return SimpleNamespace(session_id="sess-1")


@pytest.mark.asyncio
async def test_new_session_attaches_servers_and_falls_back_when_rejected() -> None:
    servers = acp_mcp_servers_from_specs({"notes": {"transport": "http", "url": "http://127.0.0.1/mcp"}})
    ok = _Conn()
    opened = await resume_or_create_session(ok, cwd="/work", session_id=None, load_supported=False, mcp_servers=servers)
    assert opened.session_id == "sess-1"
    assert opened.mcp_degraded is False
    assert ok.calls == [("new", ["notes"])]

    rejected = _Conn(fail_first=True)
    opened = await resume_or_create_session(
        rejected, cwd="/work", session_id=None, load_supported=False, mcp_servers=servers
    )
    assert opened.session_id == "sess-1"
    assert opened.mcp_degraded is True
    assert rejected.calls == [("new", ["notes"]), ("new", [])]


@pytest.mark.asyncio
async def test_new_session_without_servers_still_raises() -> None:
    class _Empty:
        async def new_session(self, cwd: str, **_: Any) -> Any:
            del cwd

            class _Created:
                session_id = ""

            return _Created()

    with pytest.raises(ACPSessionError):
        await resume_or_create_session(_Empty(), cwd="/work", session_id=None, load_supported=False)


def _conversation(conn: Any, *, load_supported: bool, mcp_key: str) -> _Conversation:
    return _Conversation(
        thread_id="thr",
        runner="codebuddy",
        acp_session_id="sess",
        cwd="/work",
        conn=conn,
        process=SimpleNamespace(returncode=None),
        client=SimpleNamespace(),  # type: ignore[arg-type]
        exit_stack=SimpleNamespace(),  # type: ignore[arg-type]
        turn_lock=asyncio.Lock(),
        load_supported=load_supported,
        mcp_key=mcp_key,
    )


@pytest.mark.asyncio
async def test_changed_servers_reload_when_the_agent_can_load() -> None:
    servers = acp_mcp_servers_from_specs({"notes": {"transport": "http", "url": "http://127.0.0.1/mcp"}})

    class Conn:
        def __init__(self) -> None:
            self.loaded: list[str] = []

        async def load_session(self, **kwargs: object) -> None:
            raw = kwargs.get("mcp_servers")
            self.loaded = [str(getattr(item, "name", "")) for item in raw] if isinstance(raw, list) else []

    conn = Conn()
    conversation = _conversation(conn, load_supported=True, mcp_key="previous")
    restart = await ACPService(config=ACPConfig())._reload_mcp(conversation, servers)
    assert restart is False
    assert conn.loaded == ["notes"]
    assert conversation.mcp_key == mcp_fingerprint(servers)


@pytest.mark.asyncio
async def test_unchanged_servers_do_not_reload() -> None:
    servers = acp_mcp_servers_from_specs({"notes": {"transport": "http", "url": "http://127.0.0.1/mcp"}})

    class Conn:
        def __init__(self) -> None:
            self.loaded = False

        async def load_session(self, **kwargs: object) -> None:
            del kwargs
            self.loaded = True

    conn = Conn()
    conversation = _conversation(conn, load_supported=True, mcp_key=mcp_fingerprint(servers))
    restart = await ACPService(config=ACPConfig())._reload_mcp(conversation, servers)
    assert restart is False
    assert conn.loaded is False
