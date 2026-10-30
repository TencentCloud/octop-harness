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
