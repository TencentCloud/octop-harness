"""Tests for ``octop_harness.llm.factory``."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from octop_harness.config import ModelConfig, ProviderConfig
from octop_harness.llm.factory import (
    ChatModelFactory,
    _get_reasoning_aware_chat_openai,
    build_chat_model,
)


class TestBuildChatModel:
    def test_openai_protocol_invokes_chat_openai(self) -> None:
        provider = ProviderConfig(
            id="example",
            base_url="https://api.example.com/v1",
            api_key="sk-x",
            protocol="openai",
            headers={"X-Trace": "1"},
        )
        model = ModelConfig(id="gpt-X")
        # The subclass is built lazily off the live ``ChatOpenAI`` and cached
        # on first use, so we patch the subclass directly to assert the kwargs
        # that reach the constructor.
        from langchain_openai import ChatOpenAI

        cls = _get_reasoning_aware_chat_openai(ChatOpenAI)
        with patch.object(cls, "__init__", return_value=None) as mock_init:
            build_chat_model(provider, model)
        mock_init.assert_called_once()
        kwargs = mock_init.call_args.kwargs
        assert kwargs["model"] == "gpt-X"
        assert kwargs["base_url"] == "https://api.example.com/v1"
        assert kwargs["api_key"] == "sk-x"
        assert kwargs["default_headers"] == {"X-Trace": "1"}
        # Custom base_url disables langchain-openai's stream_usage default,
        # so the factory must pass it explicitly or streamed responses carry
        # no usage_metadata.
        assert kwargs["stream_usage"] is True

    def test_openai_protocol_stream_usage_opt_out(self) -> None:
        provider = ProviderConfig(
            id="example",
            base_url="https://api.example.com/v1",
            api_key="sk-x",
            protocol="openai",
            stream_usage=False,
        )
        model = ModelConfig(id="gpt-X")
        from langchain_openai import ChatOpenAI

        cls = _get_reasoning_aware_chat_openai(ChatOpenAI)
        with patch.object(cls, "__init__", return_value=None) as mock_init:
            build_chat_model(provider, model)
        assert mock_init.call_args.kwargs["stream_usage"] is False

    def test_openai_native_tool_search_enables_responses_api(self) -> None:
        provider = ProviderConfig(
            id="openai",
            base_url="https://api.openai.com/v1",
            api_key="sk-x",
            protocol="openai",
        )
        model = ModelConfig(id="gpt-tool-search", native_tool_search=True)
        from langchain_openai import ChatOpenAI

        cls = _get_reasoning_aware_chat_openai(ChatOpenAI)
        with patch.object(cls, "__init__", return_value=None) as mock_init:
            instance = build_chat_model(provider, model)

        kwargs = mock_init.call_args.kwargs
        assert kwargs["use_responses_api"] is True
        assert kwargs["output_version"] == "responses/v1"
        assert instance._harness_native_tool_search is True
        assert instance._harness_tool_search_provider == "openai"

    def test_openai_profile_includes_distinct_context_limits(self) -> None:
        provider = ProviderConfig(
            id="example",
            base_url="https://api.example.com/v1",
            api_key="sk-x",
            protocol="openai",
        )
        model = ModelConfig(
            id="deepseek-v4-pro",
            max_input_tokens=1_000_000,
            context_window=1_000_000,
            max_output_tokens=384_000,
        )
        instance = build_chat_model(provider, model)
        assert instance.profile is not None
        assert instance.profile["max_input_tokens"] == 1_000_000
        assert instance.profile["context_window"] == 1_000_000
        assert instance.profile["max_output_tokens"] == 384_000

    def test_openai_returns_reasoning_aware_subclass(self) -> None:
        """``protocol="openai"`` must use the reasoning_content-aware subclass."""
        from langchain_openai import ChatOpenAI

        provider = ProviderConfig(
            id="example",
            base_url="https://api.example.com/v1",
            api_key="sk-x",
            protocol="openai",
        )
        model = ModelConfig(id="gpt-X")
        instance = build_chat_model(provider, model)
        assert isinstance(instance, ChatOpenAI)
        assert type(instance).__name__ == "_ReasoningAwareChatOpenAI"

    def test_anthropic_protocol_invokes_chat_anthropic(self) -> None:
        provider = ProviderConfig(
            id="anthropic",
            base_url="https://api.anthropic.com",
            api_key="sk-y",
            protocol="anthropic",
        )
        model = ModelConfig(id="claude-x")
        with patch("langchain_anthropic.ChatAnthropic") as mock_cls:
            build_chat_model(provider, model)
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == "claude-x"
        assert kwargs["api_key"] == "sk-y"

    def test_anthropic_native_tool_search_stamps_capability(self) -> None:
        provider = ProviderConfig(
            id="anthropic",
            base_url="https://api.anthropic.com",
            api_key="sk-y",
            protocol="anthropic",
        )
        model = ModelConfig(id="claude-tool-search", native_tool_search=True)
        with patch("langchain_anthropic.ChatAnthropic"):
            instance = build_chat_model(provider, model)

        assert instance._harness_native_tool_search is True
        assert instance._harness_tool_search_provider == "anthropic"

    def test_bedrock_protocol_reports_clear_import_error_when_missing(self) -> None:
        """When ``langchain_aws`` is not importable, surface a clear instruction."""
        provider = ProviderConfig(
            id="bedrock",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            api_key="aws-x",
            protocol="bedrock",
        )
        model = ModelConfig(id="anthropic.claude-3")
        # Force the import inside ``_build_bedrock`` to fail by patching the
        # builtin import. We only need to intercept the specific module name.
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "langchain_aws":
                raise ImportError("simulated missing langchain_aws")
            return real_import(name, *args, **kwargs)

        with (
            patch.object(builtins, "__import__", side_effect=fake_import),
            pytest.raises(ImportError, match="langchain-aws"),
        ):
            build_chat_model(provider, model)


class TestChatModelFactoryCaching:
    def _providers(self) -> list[ProviderConfig]:
        return [
            ProviderConfig(
                id="p",
                base_url="https://x",
                api_key="k",
                models=[ModelConfig(id="m")],
            )
        ]

    def test_caches_instances(self) -> None:
        factory = ChatModelFactory(self._providers())
        from langchain_openai import ChatOpenAI

        cls = _get_reasoning_aware_chat_openai(ChatOpenAI)
        with patch.object(cls, "__init__", return_value=None) as mock_init:
            first = factory.get_chat_model("p/m")
            second = factory.get_chat_model("p/m")
        assert first is second
        assert mock_init.call_count == 1

    def test_unknown_provider_raises(self) -> None:
        factory = ChatModelFactory(self._providers())
        with pytest.raises(ValueError, match="Unknown provider"):
            factory.get_chat_model("other/m")

    def test_unknown_model_raises(self) -> None:
        factory = ChatModelFactory(self._providers())
        with pytest.raises(ValueError, match=r"not found or disabled"):
            factory.get_chat_model("p/nonexistent")

    def test_accepts_list_directly(self) -> None:
        """Factory must accept list[ProviderConfig], not HarnessAgentConfig."""
        providers = self._providers()
        factory = ChatModelFactory(providers)
        assert factory is not None


class TestReasoningContentRoundTrip:
    """The DeepSeek thinking-mode bug: ``reasoning_content`` must round-trip.

    Upstream ``langchain-openai`` 1.x silently drops the field. Our subclass
    captures it on parse and re-emits it on serialize.
    """

    def _build(self) -> Any:
        provider = ProviderConfig(
            id="example",
            base_url="https://api.example.com/v1",
            api_key="sk-x",
            protocol="openai",
        )
        model = ModelConfig(id="thinking-model")
        return build_chat_model(provider, model)

    def test_create_chat_result_captures_reasoning_content(self) -> None:
        """A response with ``reasoning_content`` lands in ``additional_kwargs``."""
        from langchain_core.messages import AIMessage

        instance = self._build()
        response = {
            "id": "resp-1",
            "model": "thinking-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Beijing weather is sunny.",
                        "reasoning_content": "thinking aloud here",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = instance._create_chat_result(response)
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert msg.additional_kwargs.get("reasoning_content") == "thinking aloud here"

    def test_get_request_payload_re_emits_reasoning_content(self) -> None:
        """A round-tripped AIMessage must carry ``reasoning_content`` back out."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        instance = self._build()
        ai = AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "thinking again"},
            tool_calls=[
                {
                    "name": "get_weather",
                    "args": {"city": "Beijing"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )
        messages = [
            HumanMessage(content="weather?"),
            ai,
            ToolMessage(content="sunny", tool_call_id="call_1"),
        ]
        payload = instance._get_request_payload(messages)

        # Find the assistant entry in the rebuilt messages list.
        assistants = [m for m in payload["messages"] if m.get("role") == "assistant"]
        assert len(assistants) == 1
        assert assistants[0]["reasoning_content"] == "thinking again"

    def test_payload_unchanged_when_reasoning_content_absent(self) -> None:
        """No ``reasoning_content`` injected for normal AIMessages."""
        from langchain_core.messages import AIMessage, HumanMessage

        instance = self._build()
        messages = [
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
        ]
        payload = instance._get_request_payload(messages)
        assistants = [m for m in payload["messages"] if m.get("role") == "assistant"]
        assert len(assistants) == 1
        assert "reasoning_content" not in assistants[0]

    def test_create_chat_result_no_op_without_reasoning_content(self) -> None:
        """Standard OpenAI response shouldn't gain a phantom ``reasoning_content``."""
        from langchain_core.messages import AIMessage

        instance = self._build()
        response = {
            "id": "resp-2",
            "model": "thinking-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = instance._create_chat_result(response)
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert "reasoning_content" not in msg.additional_kwargs

    def test_subclass_is_memoised(self) -> None:
        """The subclass is built once per process and reused."""
        from langchain_openai import ChatOpenAI

        first = _get_reasoning_aware_chat_openai(ChatOpenAI)
        second = _get_reasoning_aware_chat_openai(ChatOpenAI)
        assert first is second
        assert issubclass(first, ChatOpenAI)
