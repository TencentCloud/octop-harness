"""``ModelRouterMiddleware`` — multimodal-aware model selection.

Priority chain (per design doc §3.4), highest first:
    1. ``configurable["model"]`` (set either explicitly by callers or
       auto-injected from ``ChatRequest.model``).
    2. ``HarnessAgentConfig.model_selector(state, config)`` — user-supplied hook.
    3. Built-in default selector — picks the multimodal model when the latest
       ``HumanMessage`` carries non-text content blocks.
    4. ``HarnessAgentConfig.default_model``.
    5. The first enabled model in providers (final fallback; computed lazily
       via :py:meth:`HarnessAgentConfig.pick_default_model_ref`).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

from octop_harness.api_key_scope import current_api_key_override
from octop_harness.config import HarnessAgentConfig, ModelSelector
from octop_harness.llm.factory import ChatModelFactory
from octop_harness.middleware.runtime import runtime_config
from octop_harness.middleware.turn_model import (
    content_has_non_text,
    latest_human_needs_multimodal,
    resolve_turn_model_ref,
)


class ModelRouterMiddleware(AgentMiddleware[Any, Any]):
    """Override the model on each ``ModelRequest`` based on the priority chain."""

    def __init__(
        self,
        config: HarnessAgentConfig,
        factory: ChatModelFactory,
        *,
        get_protocol: Callable[[str | None], Any] | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._factory = factory
        self._get_protocol = get_protocol
        self._user_selector: ModelSelector | None = config.model_selector

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        request.model = self._select_model(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        request.model = self._select_model(request)
        return await handler(request)

    def _select_model(self, request: ModelRequest) -> Any:
        ref = self._select_ref(request)
        api_key_override = self._api_key_override(request)
        if self._get_protocol is not None:
            return self._factory.get(ref, get_protocol=self._get_protocol, api_key_override=api_key_override).chat_model
        return self._factory.get_chat_model(ref, api_key_override=api_key_override)

    @staticmethod
    def _api_key_override(request: ModelRequest) -> str | None:
        """Per-call credential override (e.g. a per-user gateway key).

        A literal ``configurable["api_key_override"]`` wins when present
        (callers that build the graph config directly). Otherwise the key is
        read from the :mod:`octop_harness.api_key_scope` binding — the
        ``ChatRequest`` path keeps keys out of ``configurable`` because
        langgraph copies those values into checkpoint metadata.
        """
        configurable = runtime_config(request).get("configurable")
        if isinstance(configurable, dict):
            value = configurable.get("api_key_override")
            if isinstance(value, str) and value:
                return value
        return current_api_key_override()

    def _select_ref(self, request: ModelRequest) -> str:
        runtime_config_value = runtime_config(request)
        configurable = runtime_config_value.get("configurable")
        if not isinstance(configurable, dict):
            configurable = {}
            if isinstance(runtime_config_value, dict):
                runtime_config_value["configurable"] = configurable
        ref = resolve_turn_model_ref(
            pick_default_ref=self._config.pick_default_model_ref,
            configurable=configurable,
            messages=list(request.messages or []),
            user_selector=self._user_selector,
            state=getattr(request, "state", None) or {},
            runtime_config=runtime_config_value,
            pick_multimodal_ref=self._config.pick_multimodal_model_ref,
        )
        # Stamp the live chat ref so extract can fall back to it when aux fails.
        if ref:
            configurable["model"] = ref
        return ref


def _request_needs_multimodal(request: ModelRequest) -> bool:
    """Return True iff the latest user message contains non-text content."""
    return latest_human_needs_multimodal(list(request.messages or []))


# Back-compat alias for older tests / callers.
_content_has_non_text = content_has_non_text

__all__ = ["ModelRouterMiddleware"]
