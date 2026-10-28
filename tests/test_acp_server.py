"""tests/test_acp_server.py — unit tests for ACP server helpers."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

import pytest
from acp import Agent, RequestError
from acp.schema import InitializeResponse, NewSessionResponse, SetSessionConfigOptionResponse

from octop_harness.acp.server import (
    ACP_AGENT_META_KEY,
    ACP_EXPERT_CATEGORY,
    ACP_EXPERT_CONFIG_ID,
    ACPExpertOption,
    HarnessACPAgent,
    _acp_agent_cls,
    _chunk_to_updates,
    _extract_prompt_text,
    _StreamTracker,
    _tool_output_text,
    build_harness_acp_agent,
)


def test_extract_prompt_text_from_dict_blocks() -> None:
    assert _extract_prompt_text([{"type": "text", "text": "hello"}]) == "hello"
    assert _extract_prompt_text([{"text": "a"}, {"text": "b"}]) == "a\nb"


def test_stream_tracker_text_delta() -> None:
    tracker = _StreamTracker()
    assert tracker.delta_text("hello") == "hello"
    assert tracker.delta_text("hello world") == " world"


def test_chunk_to_updates_token() -> None:
    tracker = _StreamTracker()
    updates = _chunk_to_updates(
        {"type": "token", "content": "hi"},
        tracker,
        tool_name_buf={},
        tool_id_buf={},
    )
    assert len(updates) == 1


def test_tool_output_text_from_objects() -> None:
    class _Msg:
        content = "done"

    assert _tool_output_text([_Msg()]) == "done"


def test_acp_agent_mro_prefers_harness_implementation() -> None:
    """Protocol stubs must not shadow HarnessACPAgent (regression for null ACP replies)."""
    cls = _acp_agent_cls(object(), agent_id="agent-1")
    mro = cls.__mro__
    assert mro.index(HarnessACPAgent) < mro.index(Agent), (
        f"HarnessACPAgent must precede acp.Agent in MRO, got {[c.__name__ for c in mro]}"
    )


@pytest.mark.asyncio
async def test_protocol_before_impl_returns_none_documents_bug() -> None:
    """Pin why ``(Agent, HarnessACPAgent)`` is wrong: Protocol methods return None."""

    class _Iface(Protocol):
        async def initialize(self, protocol_version: int) -> object:
            """Protocol stub: the body is ``...``, so a protocol-first MRO returns None."""
            ...

    class _Impl:
        async def initialize(self, protocol_version: int) -> object:
            return {"protocolVersion": protocol_version}

    class _Broken(_Iface, _Impl):
        def __init__(self) -> None:
            _Impl.__init__(self)

    class _Fixed(_Impl, _Iface):
        def __init__(self) -> None:
            _Impl.__init__(self)

    assert await _Broken().initialize(1) is None
    assert await _Fixed().initialize(1) == {"protocolVersion": 1}


@pytest.mark.asyncio
async def test_build_harness_acp_agent_initialize_not_null() -> None:
    from acp import PROTOCOL_VERSION

    agent = build_harness_acp_agent(object(), agent_id="main")
    result = await agent.initialize(protocol_version=1)
    assert result is not None
    assert isinstance(result, InitializeResponse)
    assert result.protocol_version == 1
    assert result.agent_capabilities is not None
    assert result.agent_capabilities.load_session is True
    assert result.agent_capabilities.session_capabilities is not None
    assert result.agent_capabilities.session_capabilities.close is not None
    assert result.agent_capabilities.session_capabilities.resume is not None
    assert result.agent_capabilities.session_capabilities.list is not None
    assert result.agent_info is not None
    assert result.agent_info.name == "octop"

    unsupported = await agent.initialize(protocol_version=65535)
    assert unsupported.protocol_version == PROTOCOL_VERSION
    assert unsupported.protocol_version != 65535


@pytest.mark.asyncio
async def test_build_harness_acp_agent_new_session_returns_id() -> None:
    agent = build_harness_acp_agent(object(), agent_id="main")
    result = await agent.new_session(cwd="/tmp")
    assert result is not None
    assert isinstance(result, NewSessionResponse)
    assert isinstance(result.session_id, str) and result.session_id
    assert result.field_meta == {ACP_AGENT_META_KEY: "main"}


@pytest.mark.asyncio
async def test_build_harness_acp_agent_load_and_close_session() -> None:
    agent = build_harness_acp_agent(object(), agent_id="main")
    created = await agent.new_session(cwd="/work")
    loaded = await agent.load_session(cwd="/work", session_id=created.session_id)
    assert loaded is not None
    assert loaded.field_meta == {ACP_AGENT_META_KEY: "main"}
    closed = await agent.close_session(session_id=created.session_id)
    assert closed is not None


@pytest.mark.asyncio
async def test_resume_and_list_sessions() -> None:
    agent = build_harness_acp_agent(object(), agent_id="main")
    created = await agent.new_session(cwd="/work")
    resumed = await agent.resume_session(cwd="/work", session_id=created.session_id)
    assert resumed is not None
    assert resumed.field_meta == {ACP_AGENT_META_KEY: "main"}

    # Resume after close rebinds a cold session id (process restart / Zed reopen).
    await agent.close_session(session_id=created.session_id)
    cold = await agent.resume_session(cwd="/other", session_id=created.session_id)
    assert cold is not None
    listed = await agent.list_sessions()
    assert any(s.session_id == created.session_id for s in listed.sessions)


@pytest.mark.asyncio
async def test_load_session_replays_history_via_session_update() -> None:
    class _Msg:
        def __init__(self, type_: str, content: str) -> None:
            self.type = type_
            self.content = content

    class _Harness:
        async def aget_history(self, thread_id: str, *, limit: int = 50) -> list[Any]:
            assert thread_id  # bound to session
            return [
                _Msg("human", "hello"),
                _Msg("ai", "world"),
                _Msg("tool", "skip-me"),
            ]

    class _Conn:
        def __init__(self) -> None:
            self.updates: list[Any] = []

        async def session_update(self, *, session_id: str, update: Any) -> None:
            self.updates.append((session_id, type(update).__name__))

    agent = build_harness_acp_agent(_Harness(), agent_id="main")
    conn = _Conn()
    agent.on_connect(conn)
    sid = "sess-replay-1"
    await agent.load_session(cwd="/work", session_id=sid)
    kinds = [name for _sid, name in conn.updates]
    assert kinds == ["UserMessageChunk", "AgentMessageChunk"]


@pytest.mark.asyncio
async def test_wrong_mro_with_real_acp_agent_protocol_returns_null() -> None:
    """With the real ``acp.Agent`` Protocol, wrong base order yields null handshakes."""

    class _Broken(Agent, HarnessACPAgent):
        def __init__(self) -> None:
            HarnessACPAgent.__init__(self, object(), agent_id="x")

    broken = _Broken()
    assert await broken.initialize(protocol_version=1) is None
    assert await broken.new_session(cwd="/tmp") is None


def test_run_and_build_share_same_mro_factory() -> None:
    """Both public entry points must use the same implementation-first class factory."""
    build_cls = type(build_harness_acp_agent(object(), agent_id="a"))
    run_cls = _acp_agent_cls(object(), agent_id="a")
    assert build_cls.__mro__[1:] == run_cls.__mro__[1:]
    assert build_cls.__mro__.index(HarnessACPAgent) < build_cls.__mro__.index(Agent)


@pytest.mark.asyncio
async def test_prompt_empty_returns_end_turn_without_harness_stream() -> None:
    class _BoomHarness:
        def stream(self, *_a: Any, **_k: Any) -> Any:
            raise AssertionError("empty prompt must not call harness.stream")

    agent = build_harness_acp_agent(_BoomHarness(), agent_id="main")
    created = await agent.new_session(cwd="/tmp")
    result = await agent.prompt(prompt=[], session_id=created.session_id)
    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_prompt_unknown_session_raises_invalid_params() -> None:
    agent = build_harness_acp_agent(object(), agent_id="main")
    with pytest.raises(RequestError) as excinfo:
        await agent.prompt(prompt=[{"type": "text", "text": "hi"}], session_id="missing")
    assert excinfo.value.code == -32602


def _two_experts() -> tuple[list[ACPExpertOption], dict[str, Any]]:
    harnesses = {"main": object(), "ops": object()}
    experts = [
        ACPExpertOption(value="main", name="Main", description="default"),
        ACPExpertOption(value="ops", name="Ops"),
    ]
    return experts, harnesses


@pytest.mark.asyncio
async def test_multi_expert_new_session_advertises_config_options() -> None:
    experts, harnesses = _two_experts()
    agent = build_harness_acp_agent(
        harnesses["main"],
        agent_id="main",
        experts=experts,
        resolve_harness=harnesses.__getitem__,
    )
    result = await agent.new_session(cwd="/tmp")
    assert result.config_options is not None
    assert len(result.config_options) == 1
    option = result.config_options[0]
    assert option.id == ACP_EXPERT_CONFIG_ID
    assert option.category == ACP_EXPERT_CATEGORY
    assert option.current_value == "main"
    assert [o.value for o in option.options] == ["main", "ops"]
    assert result.field_meta == {ACP_AGENT_META_KEY: "main"}


@pytest.mark.asyncio
async def test_multi_expert_set_config_option_switches_and_rejects_unknown() -> None:
    experts, _harnesses = _two_experts()
    seen: list[str] = []

    class _Harness:
        def __init__(self, expert_id: str) -> None:
            self.expert_id = expert_id

        async def stream(self, request: dict[str, Any]) -> Any:
            seen.append(str(request.get("agent_id")))
            if False:  # pragma: no cover - make this an async generator
                yield {}

    live = {"main": _Harness("main"), "ops": _Harness("ops")}
    agent = build_harness_acp_agent(
        live["main"],
        agent_id="main",
        experts=experts,
        resolve_harness=live.__getitem__,
    )
    created = await agent.new_session(cwd="/work")
    switched = await agent.set_config_option(
        config_id=ACP_EXPERT_CONFIG_ID,
        session_id=created.session_id,
        value="ops",
    )
    assert isinstance(switched, SetSessionConfigOptionResponse)
    assert switched.config_options[0].current_value == "ops"

    with pytest.raises(RequestError) as excinfo:
        await agent.set_config_option(
            config_id=ACP_EXPERT_CONFIG_ID,
            session_id=created.session_id,
            value="missing",
        )
    assert excinfo.value.code == -32602

    with pytest.raises(RequestError) as excinfo2:
        await agent.set_config_option(
            config_id="model",
            session_id=created.session_id,
            value="ops",
        )
    assert excinfo2.value.code == -32602

    result = await agent.prompt(
        prompt=[{"type": "text", "text": "hi"}],
        session_id=created.session_id,
    )
    assert result.stop_reason == "end_turn"
    assert seen == ["ops"]


@pytest.mark.asyncio
async def test_single_agent_mode_omits_config_options() -> None:
    agent = build_harness_acp_agent(object(), agent_id="main")
    result = await agent.new_session(cwd="/tmp")
    assert result.config_options is None


@pytest.mark.asyncio
async def test_async_resolve_harness_waits_before_prompt() -> None:
    """Hosts may finish embedding after initialize; resolve_harness may be async."""
    experts, _ = _two_experts()
    ready = asyncio.Event()
    live = {"main": object(), "ops": object()}

    async def resolve(expert_id: str) -> Any:
        await ready.wait()
        return live[expert_id]

    agent = build_harness_acp_agent(
        None,
        agent_id="main",
        experts=experts,
        resolve_harness=resolve,
    )
    created = await agent.new_session(cwd="/tmp")
    assert created.session_id

    async def _release() -> None:
        await asyncio.sleep(0.01)
        ready.set()

    release_task = asyncio.create_task(_release())
    harness = await agent._harness_for_session(created.session_id)
    await release_task
    assert harness is live["main"]
