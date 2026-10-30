"""Memory request policy against real SDK serialization, without network."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from octop_memory.ports.llm import LLMClientError

from octop_harness.config import HarnessAgentConfig, ModelConfig, ProviderConfig
from octop_harness.llm.factory import ChatModelFactory
from octop_harness.memory.llm_client import HarnessAgentLLMClient


def _factory() -> ChatModelFactory:
    return ChatModelFactory(
        [
            ProviderConfig(
                id="local", base_url="https://local.example/v1", api_key="test", models=[ModelConfig(id="Qwen3")]
            ),
            ProviderConfig(
                id="other", base_url="https://other.example/v1", api_key="test", models=[ModelConfig(id="gpt-4.1")]
            ),
        ]
    )


def _response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        json={
            "id": "audit",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        },
    )


@pytest.mark.parametrize("retries", [0, 1, 2])
def test_timeout_attempts_do_not_multiply_sdk_retries(monkeypatch: pytest.MonkeyPatch, retries: int) -> None:
    calls: list[httpx.Request] = []
    delays: list[float] = []

    def send(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        calls.append(request)
        if len(calls) <= retries:
            raise httpx.ReadTimeout("probe timeout", request=request)
        return _response(request)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr("octop_harness.memory.llm_client.time.sleep", delays.append)
    factory = _factory()
    chat = factory.get_chat_model("local/Qwen3")
    client = HarnessAgentLLMClient(factory, default_model="local/Qwen3", max_retries=retries, retry_backoff_s=0.5)
    try:
        assert client.call_llm("probe", max_tokens=1024) == "ok"
        assert len(calls) == retries + 1
        assert delays == [0.5 * 2**i for i in range(retries)]
        assert all(json.loads(r.content)["max_completion_tokens"] == 1024 for r in calls)
        assert all("max_retries" not in json.loads(r.content) for r in calls)
        assert factory.get_chat_model("local/Qwen3") is chat
        assert chat.root_client.max_retries == 2
        assert factory.get_chat_model("local/Qwen3", max_retries=0).root_client.max_retries == 0
    finally:
        factory.close()


def test_fallback_drops_primary_vendor_options(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx.Request] = []

    def send(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        requests.append(request)
        if request.url.host == "local.example":
            return httpx.Response(
                400, request=request, json={"error": {"message": "model unavailable", "type": "invalid_request_error"}}
            )
        return _response(request)

    monkeypatch.setattr(httpx.Client, "send", send)
    factory = _factory()
    client = HarnessAgentLLMClient(
        factory,
        aux_model="local/Qwen3",
        default_model="other/gpt-4.1",
        thinking="off",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        max_retries=2,
    )
    try:
        assert client.call_llm("probe", max_tokens=512) == "ok"
        assert len(requests) == 2  # permanent 400 is not retried
        assert "chat_template_kwargs" in json.loads(requests[0].content)
        assert "chat_template_kwargs" not in json.loads(requests[1].content)
        assert json.loads(requests[1].content)["max_completion_tokens"] == 512
    finally:
        factory.close()


def test_retry_policy_roundtrip_and_validation() -> None:
    config = HarnessAgentConfig(name="test", memory_aux_max_retries=0, memory_aux_retry_backoff_s=0.25)
    restored = HarnessAgentConfig.from_dict(config.to_dict())
    assert restored.memory_aux_max_retries == 0
    assert restored.memory_aux_retry_backoff_s == 0.25
    for value in [-1, 11, True, 1.5]:
        with pytest.raises(ValueError, match="memory_aux_max_retries"):
            HarnessAgentConfig(name="test", memory_aux_max_retries=value)
    for value in [-1, 61, float("nan"), True]:
        with pytest.raises(ValueError, match="memory_aux_retry_backoff_s"):
            HarnessAgentConfig(name="test", memory_aux_retry_backoff_s=value)


@pytest.mark.parametrize("status", [408, 429, 503])
def test_transient_http_status_retries(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    calls = []

    def send(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                status, request=request, json={"error": {"message": "temporary", "type": "server_error"}}
            )
        return _response(request)

    monkeypatch.setattr(httpx.Client, "send", send)
    factory = _factory()
    try:
        client = HarnessAgentLLMClient(factory, default_model="local/Qwen3", max_retries=1, retry_backoff_s=0)
        assert client.call_llm("probe") == "ok"
        assert len(calls) == 2
    finally:
        factory.close()


@pytest.mark.parametrize("provider_wide", [False, True])
def test_model_cache_invalidation_removes_memory_variant(provider_wide: bool) -> None:
    factory = _factory()
    chat = factory.get_chat_model("local/Qwen3")
    memory = factory.get_chat_model("local/Qwen3", max_retries=0)
    assert chat is not memory
    other = factory.get_chat_model("other/gpt-4.1", max_retries=0)
    if provider_wide:
        factory.clear_provider_cache("local")
    else:
        factory.pop_cached("local/Qwen3")
    assert factory.get_chat_model("local/Qwen3") is not chat
    assert factory.get_chat_model("local/Qwen3", max_retries=0) is not memory
    assert factory.get_chat_model("other/gpt-4.1", max_retries=0) is other
    factory.close()


@pytest.mark.parametrize("use_aux", [False, True])
@pytest.mark.parametrize("tier", ["light", "heavy"])
def test_kimi_temperature_recovery_after_aux_timeout(monkeypatch: pytest.MonkeyPatch, use_aux: bool, tier: str) -> None:
    requests: list[httpx.Request] = []

    def send(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        requests.append(request)
        if request.url.host == "aux.example":
            raise httpx.ReadTimeout("aux unavailable", request=request)
        if "temperature" in json.loads(request.content):
            return httpx.Response(
                400,
                request=request,
                json={"error": {"message": "invalid temperature: only 1 is allowed", "type": "invalid_request_error"}},
            )
        return _response(request)

    monkeypatch.setattr(httpx.Client, "send", send)
    factory = ChatModelFactory(
        [
            ProviderConfig(
                id="aux", base_url="https://aux.example/v1", api_key="test", models=[ModelConfig(id="Qwen3")]
            ),
            ProviderConfig(
                id="moonshot",
                base_url="https://kimi.example/v1",
                api_key="test",
                models=[ModelConfig(id="kimi-for-coding", max_output_tokens=256)],
            ),
        ]
    )
    client = HarnessAgentLLMClient(
        factory,
        aux_model="aux/Qwen3" if use_aux else None,
        default_model="moonshot/kimi-for-coding",
        light_timeout_s=42,
        heavy_timeout_s=90,
        extra_body={"vendor_option": "primary-only"},
        max_retries=1,
        retry_backoff_s=0,
    )
    client.set_current_model("moonshot/kimi-for-coding")
    try:
        assert client.call_llm("extract", tier=tier, temperature=0.0, max_tokens=512, response_format="json") == "ok"
        assert [r.url.host for r in requests] == (["aux.example"] * 2 if use_aux else []) + ["kimi.example"] * 2
        original, retry = [json.loads(r.content) for r in requests[-2:]]
        assert original.pop("temperature") == 0.0
        assert original == retry
        assert "temperature" not in retry
        assert retry["max_completion_tokens"] == 256
        assert retry["response_format"] == {"type": "json_object"}
        assert ("vendor_option" in retry) is not use_aux
        assert "chat_template_kwargs" not in retry
        assert requests[-1].extensions["timeout"]["read"] == (42 if tier == "light" else 90)
        assert factory.get_chat_model("moonshot/kimi-for-coding").root_client.max_retries == 2
    finally:
        factory.close()


@pytest.mark.parametrize("second_status", [200, 400, 503])
@pytest.mark.parametrize("retries", [0, 1])
def test_temperature_retry_is_bounded_and_preserves_transient_retries(
    monkeypatch: pytest.MonkeyPatch, second_status: int, retries: int
) -> None:
    requests: list[httpx.Request] = []

    def send(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        requests.append(request)
        status = 400 if len(requests) == 1 else second_status
        if status == 200:
            return _response(request)
        return httpx.Response(
            status,
            request=request,
            json={"error": {"message": "invalid temperature", "type": "invalid_request_error"}},
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    factory = _factory()
    client = HarnessAgentLLMClient(factory, default_model="local/Qwen3", max_retries=retries, retry_backoff_s=0)
    try:
        if second_status == 200:
            assert client.call_llm("extract", temperature=0.0) == "ok"
        else:
            with pytest.raises(LLMClientError, match="invalid temperature"):
                client.call_llm("extract", temperature=0.0)
        assert len(requests) == (2 + retries if second_status == 503 else 2)
        assert all("temperature" not in json.loads(r.content) for r in requests[1:])
        assert all(json.loads(r.content)["chat_template_kwargs"] == {"enable_thinking": False} for r in requests)
    finally:
        factory.close()


def test_failed_temperature_retry_keeps_model_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx.Request] = []

    def send(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        requests.append(request)
        if request.url.host == "local.example":
            return httpx.Response(400, request=request, json={"error": {"message": "invalid temperature"}})
        return _response(request)

    monkeypatch.setattr(httpx.Client, "send", send)
    factory = _factory()
    try:
        client = HarnessAgentLLMClient(factory, aux_model="local/Qwen3", default_model="other/gpt-4.1")
        assert client.call_llm("extract", temperature=0.0) == "ok"
        assert [r.url.host for r in requests] == ["local.example", "local.example", "other.example"]
        assert "temperature" not in json.loads(requests[1].content)
        # A rejection by one provider must not change another provider's options.
        assert json.loads(requests[2].content)["temperature"] == 0.0
    finally:
        factory.close()


def test_auth_error_mentioning_temperature_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx.Request] = []

    def send(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            401, request=request, json={"error": {"message": "unauthorized request with temperature=0"}}
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    factory = _factory()
    try:
        client = HarnessAgentLLMClient(factory, default_model="local/Qwen3")
        with pytest.raises(LLMClientError, match="unauthorized"):
            client.call_llm("extract", temperature=0.0)
        assert len(requests) == 1
    finally:
        factory.close()
