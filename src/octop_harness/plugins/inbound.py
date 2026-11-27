"""Host-neutral contracts for pre-agent admission plugins and optional scoring."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable


class InboundDispatch(Protocol):
    async def __call__(self, message: Any, /, *, activate: bool = False, receipt: Any = None) -> Any: ...


class InboundHandler(Protocol):
    async def handle(self, channel: Any, message: Any, dispatch: InboundDispatch, /) -> bool: ...

    def close(self) -> None:
        """Synchronously cancel owned work; hosts may additionally await wait_closed()."""
        ...


class InboundTextRenderer(Protocol):
    """Optional text assembly after admission; returning None keeps host defaults.

    The host supplies current text plus media/document hints, rendered background
    segments and the resolved locale. The plugin never needs to parse host labels.
    Routing, media blocks, commands and authorization remain host-owned.
    """

    def render_inbound_text(
        self, message: Any, current_text: str, *, background: list[dict[str, Any]], locale: str
    ) -> str | None: ...


class DecisionScorer(Protocol):
    async def __call__(
        self, *, state: str, statements: list[tuple[str, str]], strict: bool = False
    ) -> dict[str, float]: ...


DecisionScorerProvider = Callable[[], Awaitable[DecisionScorer | None]]


@runtime_checkable
class DecisionScorerAware(Protocol):
    def bind_decision_scorer(self, provider: DecisionScorerProvider) -> None: ...


def validate_inbound_handler(handler: InboundHandler) -> None:
    if not callable(getattr(handler, "handle", None)) or not callable(getattr(handler, "close", None)):
        raise ValueError("inbound handler must implement handle() and close()")
    if inspect.iscoroutinefunction(handler.close):
        raise ValueError("inbound close() must be synchronous; use wait_closed() for asynchronous cleanup")
    render = getattr(handler, "render_inbound_text", None)
    if render is not None and (not callable(render) or inspect.iscoroutinefunction(render)):
        raise ValueError("inbound render_inbound_text() must be synchronous")
