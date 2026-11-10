"""Keep provider refusals as ordinary assistant replies."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import ModelRefusalMiddleware, ModelRequest, ModelResponse


class RefusalPassthroughMiddleware(ModelRefusalMiddleware):
    """Take deepagents' ``ModelRefusalMiddleware`` slot without raising.

    deepagents 0.7.35+ appends ``ModelRefusalMiddleware`` to the main agent,
    which turns a provider refusal into ``ModelRefusalError`` and fails the
    turn. ``create_deep_agent`` replaces middleware by ``name``, so this
    pass-through keeps refusals as the model's reply, as before 0.7.35.
    Subagents still report refusals through the ``task`` tool.
    """

    @property
    def name(self) -> str:
        return "ModelRefusalMiddleware"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(request)
