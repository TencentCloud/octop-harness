"""Turn-aware summary model for ``SummarizationMiddleware``.

deepagents builds ``SummarizationMiddleware`` with the compile-time seed model and
LangChain pins ``_summary_model = model.with_retry()`` in its constructor. A shared
agent compiled without a turn user therefore summarizes on the owner's default
(possibly placeholder) provider, even though ``ModelRouterMiddleware`` swaps the
main chat model per turn.

:class:`TurnAwareSummaryModel` resolves the model ref on every summary call with
the same priority chain as the router (``configurable["model"]`` → ``model_selector``
→ default) and delegates ``invoke`` / ``ainvoke``. Resolution failures fall back
to the seed so summaries never get worse than before.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from octop_harness.backends.utils import DEFENSIVE_OP_ERRORS
from octop_harness.middleware.turn_aware_profile import _configurable_from_runtime
from octop_harness.middleware.turn_model import resolve_turn_model_ref

logger = logging.getLogger(__name__)

UserSelector = Callable[[dict[str, Any], dict[str, Any]], str | None]


class TurnAwareSummaryModel:
    """Proxy that swaps the summary LLM to the current turn's model."""

    def __init__(
        self,
        seed: Any,
        *,
        factory: Any,
        pick_default_ref: Callable[[], str],
        user_selector: UserSelector | None = None,
    ) -> None:
        self._seed = seed
        self._factory = factory
        self._pick_default_ref = pick_default_ref
        self._user_selector = user_selector
        with_retry = getattr(seed, "with_retry", None)
        self._seed_model = with_retry() if callable(with_retry) else seed
        self._models: dict[str, Any] = {}

    def _current(self) -> Any:
        try:
            ref = resolve_turn_model_ref(
                pick_default_ref=self._pick_default_ref,
                configurable=_configurable_from_runtime(),
                user_selector=self._user_selector,
                state={},
            )
            model = self._factory.get_chat_model(ref)
        except DEFENSIVE_OP_ERRORS:
            logger.debug("summary model: turn resolution failed; using seed", exc_info=True)
            return self._seed_model
        if model is self._seed:
            return self._seed_model
        cached = self._models.get(ref)
        if cached is None:
            with_retry = getattr(model, "with_retry", None)
            cached = with_retry() if callable(with_retry) else model
            self._models[ref] = cached
        return cached

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self._current().invoke(*args, **kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return await self._current().ainvoke(*args, **kwargs)


def install_turn_aware_summary_model(
    mw: Any,
    *,
    seed: Any,
    factory: Any,
    pick_default_ref: Callable[[], str],
    user_selector: UserSelector | None = None,
) -> Any:
    """Replace the middleware's pinned summary model with a turn-aware proxy."""
    helper = getattr(mw, "_lc_helper", None)
    if helper is None or not hasattr(helper, "_summary_model"):
        return mw
    helper._summary_model = TurnAwareSummaryModel(
        seed,
        factory=factory,
        pick_default_ref=pick_default_ref,
        user_selector=user_selector,
    )
    return mw


__all__ = [
    "TurnAwareSummaryModel",
    "install_turn_aware_summary_model",
]
