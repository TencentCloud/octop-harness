"""The aux memory model must not report itself as part of the user's turn.

``HarnessAgentLLMClient`` runs inside a turn (``memory_save`` → manual document
reconciliation). Its runs used to inherit the turn's callback tree, so the
recorder stored the raw reconciliation JSON as an extra assistant message.
"""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda

from octop_harness.config import ModelConfig, ProviderConfig
from octop_harness.llm.factory import ChatModelFactory
from octop_harness.llm.invoke_helpers import bind_call_options, build_text_messages
from octop_harness.memory.llm_client import HarnessAgentLLMClient


class _Collector(BaseCallbackHandler):
    """Count chat-model starts the way the trajectory recorder would."""

    def __init__(self) -> None:
        self.chat_model_starts = 0

    def on_chat_model_start(self, *_args: Any, **_kwargs: Any) -> None:
        self.chat_model_starts += 1


def _factory() -> ChatModelFactory:
    return ChatModelFactory(
        [
            ProviderConfig(
                id="local",
                base_url="https://local.example/v1",
                api_key="test",
                models=[ModelConfig(id="Qwen3")],
            ),
        ]
    )


def _mock_ok_response(monkeypatch: Any) -> None:
    def send(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "aux",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            },
        )

    monkeypatch.setattr(httpx.Client, "send", send)


def test_aux_call_is_invisible_to_the_turn_callback_tree(monkeypatch: Any) -> None:
    _mock_ok_response(monkeypatch)
    factory = _factory()
    client = HarnessAgentLLMClient(factory, aux_model="local/Qwen3")
    collector = _Collector()
    try:
        text = RunnableLambda(lambda _payload: client.call_llm("probe", max_tokens=256)).invoke(
            {}, config={"callbacks": [collector]}
        )
    finally:
        factory.close()

    assert text == "ok"
    assert collector.chat_model_starts == 0


def test_a_bound_model_inside_a_turn_still_leaks_without_the_detach(monkeypatch: Any) -> None:
    """Pins the LangChain behaviour the client works around.

    If a future ``langchain-core`` stops merging the inherited callbacks, this
    test fails first — and the detach helper can be revisited then.
    """
    _mock_ok_response(monkeypatch)
    factory = _factory()
    collector = _Collector()
    try:
        model = factory.get_chat_model("local/Qwen3", max_retries=0)
        bound = bind_call_options(model, timeout_s=120.0)
        messages = build_text_messages("probe")
        RunnableLambda(lambda _payload: bound.invoke(messages, config={"callbacks": []})).invoke(
            {}, config={"callbacks": [collector]}
        )
    finally:
        factory.close()

    assert collector.chat_model_starts == 1
