"""Tests for per-invocation session-header injection."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from octop_harness.config import ModelConfig, ProviderConfig
from octop_harness.llm.factory import _get_reasoning_aware_chat_openai, build_chat_model
from octop_harness.llm.http_timeout import LLM_HTTP_TIMEOUT
from octop_harness.llm.session_header import (
    attach_session_http_clients,
    close_session_http_clients,
    current_session_id,
    install_request_hook,
    session_header_scope,
    session_http_clients,
)


def _apply_request_hooks(client: httpx.Client, request: httpx.Request) -> None:
    for hook in client.event_hooks.get("request", []):
        hook(request)


def test_session_header_scope_uses_thread_id() -> None:
    with session_header_scope("thread-abc"):
        assert current_session_id() == "thread-abc"
    # Outside the scope a throwaway id is still returned (never empty).
    assert current_session_id()


def test_session_header_scope_generates_id_when_thread_missing() -> None:
    with session_header_scope(None) as generated:
        assert generated
        assert current_session_id() == generated


def test_session_header_scope_empty_inner_keeps_outer_thread_id() -> None:
    with session_header_scope("outer-thread"):
        with session_header_scope(None) as inner:
            assert inner == "outer-thread"
            assert current_session_id() == "outer-thread"
        assert current_session_id() == "outer-thread"


def test_hook_injects_session_id_from_scope() -> None:
    client = httpx.Client()
    try:
        install_request_hook(client, "x-opencode-session")
        request = httpx.Request("POST", "https://example.com/v1/chat")
        with session_header_scope("tid-1"):
            _apply_request_hooks(client, request)
        assert request.headers["x-opencode-session"] == "tid-1"
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_client_hook_is_awaitable_on_send() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["session"] = request.headers.get("x-opencode-session", "")
        return httpx.Response(200, json={"ok": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        install_request_hook(client, "x-opencode-session")
        with session_header_scope("tid-async"):
            response = await client.get("https://example.com/")
        assert response.status_code == 200
        assert captured["session"] == "tid-async"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_async_client_stamps_header_outside_any_scope() -> None:
    """Auxiliary calls (memory extract, summarize) must still carry the header."""
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["session"] = request.headers.get("x-opencode-session", "")
        return httpx.Response(200, json={"ok": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        install_request_hook(client, "x-opencode-session")
        await client.get("https://example.com/")
        assert captured["session"]
    finally:
        await client.aclose()


def test_session_http_clients_match_hook_flavour_to_client() -> None:
    """``AsyncClient`` awaits its hooks; a sync hook there raises TypeError."""
    sync_client, async_client = session_http_clients("x-opencode-session")
    holder = SimpleNamespace()
    attach_session_http_clients(holder, sync_client, async_client)
    try:
        sync_hooks = sync_client.event_hooks["request"]
        async_hooks = async_client.event_hooks["request"]
        assert sync_hooks and not any(inspect.iscoroutinefunction(h) for h in sync_hooks)
        assert async_hooks and all(inspect.iscoroutinefunction(h) for h in async_hooks)
    finally:
        close_session_http_clients(holder)


def test_anthropic_binding_installs_async_hook_on_async_client() -> None:
    pytest.importorskip("langchain_anthropic")
    provider = ProviderConfig(
        id="go",
        protocol="anthropic",
        base_url="https://opencode.ai/zen/go",
        api_key="sk",
        session_header="x-opencode-session",
    )
    instance = build_chat_model(provider, ModelConfig(id="claude-sonnet-4"))
    try:
        sync_hooks = instance._client._client.event_hooks["request"]
        async_hooks = instance._async_client._client.event_hooks["request"]
        assert sync_hooks and not any(inspect.iscoroutinefunction(h) for h in sync_hooks)
        assert async_hooks and all(inspect.iscoroutinefunction(h) for h in async_hooks)
    finally:
        close_session_http_clients(instance)


def test_hook_does_not_overwrite_existing_header() -> None:
    client = httpx.Client()
    try:
        install_request_hook(client, "x-opencode-session")
        request = httpx.Request(
            "POST",
            "https://example.com/v1/chat",
            headers={"X-Opencode-Session": "user-set"},
        )
        with session_header_scope("tid-1"):
            _apply_request_hooks(client, request)
        assert request.headers["X-Opencode-Session"] == "user-set"
    finally:
        client.close()


def test_provider_config_round_trips_session_header() -> None:
    provider = ProviderConfig(
        id="go",
        base_url="https://opencode.ai/zen/go/v1",
        api_key="sk",
        session_header="x-opencode-session",
    )
    restored = ProviderConfig.from_dict(provider.to_dict())
    assert restored.session_header == "x-opencode-session"


def test_provider_config_from_dict_defaults_session_header() -> None:
    provider = ProviderConfig.from_dict({"id": "p", "base_url": "https://x", "api_key": "k"})
    assert provider.session_header is None


def test_provider_config_blank_session_header_becomes_none() -> None:
    provider = ProviderConfig(id="p", base_url="https://x", api_key="k", session_header="  ")
    assert provider.session_header is None


@pytest.mark.asyncio
async def test_openai_factory_passes_dedicated_clients_when_session_header_set() -> None:
    provider = ProviderConfig(
        id="go",
        base_url="https://opencode.ai/zen/go/v1",
        api_key="sk",
        protocol="openai",
        session_header="x-opencode-session",
    )
    model = ModelConfig(id="glm-5.2")
    from langchain_openai import ChatOpenAI

    cls = _get_reasoning_aware_chat_openai(ChatOpenAI)
    with patch.object(cls, "__init__", return_value=None) as mock_init:
        build_chat_model(provider, model)
    kwargs = mock_init.call_args.kwargs
    sync_client = kwargs["http_client"]
    async_client = kwargs["http_async_client"]
    try:
        assert isinstance(sync_client, httpx.Client)
        assert isinstance(async_client, httpx.AsyncClient)
        assert sync_client.timeout == LLM_HTTP_TIMEOUT
        # SDK request timeout must be set even with a custom http_client;
        # otherwise OpenAI writes timeout=None onto the request and ignores
        # the client's Timeout.
        assert kwargs["timeout"] == LLM_HTTP_TIMEOUT
        request = httpx.Request("POST", "https://opencode.ai/zen/go/v1/chat/completions")
        with session_header_scope("chat-thread"):
            _apply_request_hooks(sync_client, request)
        assert request.headers["x-opencode-session"] == "chat-thread"
    finally:
        sync_client.close()
        await async_client.aclose()


def test_openai_factory_skips_dedicated_clients_without_session_header() -> None:
    provider = ProviderConfig(
        id="example",
        base_url="https://api.example.com/v1",
        api_key="sk",
        protocol="openai",
    )
    model = ModelConfig(id="gpt-X")
    from langchain_openai import ChatOpenAI

    cls = _get_reasoning_aware_chat_openai(ChatOpenAI)
    with patch.object(cls, "__init__", return_value=None) as mock_init:
        build_chat_model(provider, model)
    kwargs = mock_init.call_args.kwargs
    assert "http_client" not in kwargs
    assert "http_async_client" not in kwargs
    assert kwargs["timeout"] == LLM_HTTP_TIMEOUT


def test_anthropic_factory_sets_explicit_timeout() -> None:
    pytest.importorskip("langchain_anthropic")
    from langchain_anthropic import ChatAnthropic

    provider = ProviderConfig(
        id="anthropic",
        protocol="anthropic",
        base_url="https://api.anthropic.com",
        api_key="sk",
    )
    with patch.object(ChatAnthropic, "__init__", return_value=None) as mock_init:
        build_chat_model(provider, ModelConfig(id="claude-sonnet-4"))
    kwargs = mock_init.call_args.kwargs
    assert kwargs["timeout"] == LLM_HTTP_TIMEOUT.read == 600.0


@pytest.mark.asyncio
async def test_session_header_middleware_uses_thread_id_from_runtime() -> None:
    from langchain.agents.middleware import ModelRequest

    from octop_harness.middleware.session_header import SessionHeaderMiddleware, thread_id_from_request

    request = ModelRequest(model=object(), messages=[], model_settings={})
    with patch(
        "octop_harness.middleware.session_header.runtime_config",
        return_value={"configurable": {"thread_id": "mw-thread"}},
    ):
        assert thread_id_from_request(request) == "mw-thread"
        seen: list[str] = []

        async def handler(_request: ModelRequest) -> object:
            seen.append(current_session_id())
            return object()

        await SessionHeaderMiddleware().awrap_model_call(request, handler)
    assert seen == ["mw-thread"]


@pytest.mark.asyncio
async def test_session_header_middleware_without_thread_id_preserves_outer_scope() -> None:
    from langchain.agents.middleware import ModelRequest

    from octop_harness.middleware.session_header import SessionHeaderMiddleware

    request = ModelRequest(model=object(), messages=[], model_settings={})
    seen: list[str] = []

    async def handler(_request: ModelRequest) -> object:
        seen.append(current_session_id())
        return object()

    with (
        session_header_scope("outer-thread"),
        patch(
            "octop_harness.middleware.session_header.runtime_config",
            return_value={},
        ),
    ):
        await SessionHeaderMiddleware().awrap_model_call(request, handler)
    assert seen == ["outer-thread"]


def test_session_http_clients_set_explicit_timeout() -> None:
    sync_client, async_client = session_http_clients("x-session")
    holder = SimpleNamespace()
    attach_session_http_clients(holder, sync_client, async_client)
    try:
        assert sync_client.timeout == LLM_HTTP_TIMEOUT
        assert async_client.timeout == LLM_HTTP_TIMEOUT
        assert sync_client.timeout.read == 600.0
        assert sync_client.timeout.connect == 5.0
    finally:
        close_session_http_clients(holder)
        assert sync_client.is_closed


def test_openai_sdk_honors_chat_timeout_over_http_client_default() -> None:
    """Regression: timeout=None on OpenAI ignores httpx.Client(timeout=...)."""
    pytest.importorskip("openai")
    from openai import OpenAI
    from openai._base_client import FinalRequestOptions

    http = httpx.Client(timeout=LLM_HTTP_TIMEOUT)
    try:
        none_client = OpenAI(
            api_key="sk-xxx",
            base_url="https://example.com/v1",
            http_client=http,
            timeout=None,
            max_retries=0,
        )
        none_opts = FinalRequestOptions.construct(
            method="post",
            url="/chat/completions",
            json_data={},
            timeout=none_client.timeout,
        )
        none_timeout = none_client._build_request(none_opts).extensions.get("timeout")
        assert none_timeout == {"connect": None, "read": None, "write": None, "pool": None}

        bounded = OpenAI(
            api_key="sk-xxx",
            base_url="https://example.com/v1",
            http_client=http,
            timeout=LLM_HTTP_TIMEOUT,
            max_retries=0,
        )
        bounded_opts = FinalRequestOptions.construct(
            method="post",
            url="/chat/completions",
            json_data={},
            timeout=bounded.timeout,
        )
        bounded_timeout = bounded._build_request(bounded_opts).extensions.get("timeout")
        assert bounded_timeout == {
            "connect": 5.0,
            "read": 600.0,
            "write": 600.0,
            "pool": 600.0,
        }
    finally:
        http.close()


def test_openai_sdk_timeout_aborts_hung_upstream() -> None:
    """CI-safe stand-in: SDK request timeout aborts a silent upstream near read=N."""
    import socket
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    pytest.importorskip("openai")
    from openai import APITimeoutError, OpenAI

    class _Hang(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            if length:
                self.rfile.read(length)
            time.sleep(3600)

        def log_message(self, format: str, *args: object) -> None:
            return

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    server = ThreadingHTTPServer(("127.0.0.1", port), _Hang)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    short = httpx.Timeout(connect=2.0, read=2.0, write=5.0, pool=2.0)
    client = OpenAI(
        api_key="sk-xxx",
        base_url=f"http://127.0.0.1:{port}/v1",
        timeout=short,
        max_retries=0,
    )
    t0 = time.perf_counter()
    with pytest.raises(APITimeoutError):
        client.chat.completions.create(
            model="dummy",
            messages=[{"role": "user", "content": "hi"}],
        )
    wall = time.perf_counter() - t0
    server.shutdown()
    assert 1.5 <= wall <= 5.0, wall


def test_bedrock_factory_sets_botocore_timeouts() -> None:
    import sys
    from unittest.mock import MagicMock

    fake_cls = MagicMock(name="ChatBedrockConverse")
    fake_cls.return_value = SimpleNamespace()
    fake_config_cls = MagicMock(name="Config")
    fake_config = SimpleNamespace(connect_timeout=5.0, read_timeout=600.0)
    fake_config_cls.return_value = fake_config
    botocore_config = SimpleNamespace(Config=fake_config_cls)
    provider = ProviderConfig(
        id="bedrock",
        protocol="bedrock",
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        api_key="aws-x",
    )
    with patch.dict(
        sys.modules,
        {
            "langchain_aws": SimpleNamespace(ChatBedrockConverse=fake_cls),
            "botocore": SimpleNamespace(config=botocore_config),
            "botocore.config": botocore_config,
        },
    ):
        build_chat_model(provider, ModelConfig(id="anthropic.claude-3"))
    fake_config_cls.assert_called_once_with(connect_timeout=5.0, read_timeout=600.0)
    assert fake_cls.call_args.kwargs["config"] is fake_config
