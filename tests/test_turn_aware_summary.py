"""Tests for the turn-aware summary model used by SummarizationMiddleware."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from octop_harness.middleware.turn_aware_summary import (
    TurnAwareSummaryModel,
    install_turn_aware_summary_model,
)


def _model(name: str) -> MagicMock:
    model = MagicMock(name=name)
    retry = MagicMock(name=f"{name}-retry")
    retry.invoke.return_value = f"{name}-summary"
    retry.ainvoke = AsyncMock(return_value=f"{name}-summary")
    model.with_retry.return_value = retry
    return model


def _factory(models: dict[str, Any]) -> MagicMock:
    factory = MagicMock()
    factory.get_chat_model.side_effect = lambda ref: models[ref]
    return factory


def _proxy(seed: MagicMock, factory: MagicMock, **kwargs: Any) -> TurnAwareSummaryModel:
    return TurnAwareSummaryModel(
        seed,
        factory=factory,
        pick_default_ref=lambda: "p/seed",
        **kwargs,
    )


def test_summary_uses_configurable_model() -> None:
    seed = _model("seed")
    kimi = _model("kimi")
    proxy = _proxy(seed, _factory({"p/seed": seed, "p/kimi": kimi}))

    with patch(
        "octop_harness.middleware.turn_aware_summary._configurable_from_runtime",
        return_value={"model": "p/kimi"},
    ):
        assert proxy.invoke("prompt") == "kimi-summary"

    kimi.with_retry.return_value.invoke.assert_called_once_with("prompt")


async def test_async_summary_uses_configurable_model() -> None:
    seed = _model("seed")
    kimi = _model("kimi")
    proxy = _proxy(seed, _factory({"p/seed": seed, "p/kimi": kimi}))

    with patch(
        "octop_harness.middleware.turn_aware_summary._configurable_from_runtime",
        return_value={"model": "p/kimi"},
    ):
        assert await proxy.ainvoke("prompt") == "kimi-summary"

    kimi.with_retry.return_value.ainvoke.assert_awaited_once_with("prompt")


def test_summary_falls_back_to_seed_without_turn_override() -> None:
    seed = _model("seed")
    proxy = _proxy(seed, _factory({"p/seed": seed}))

    with patch(
        "octop_harness.middleware.turn_aware_summary._configurable_from_runtime",
        return_value={},
    ):
        assert proxy.invoke("prompt") == "seed-summary"

    seed.with_retry.return_value.invoke.assert_called_once_with("prompt")


def test_summary_falls_back_to_seed_when_factory_fails() -> None:
    seed = _model("seed")
    factory = MagicMock()
    factory.get_chat_model.side_effect = ValueError("unknown provider")
    proxy = _proxy(seed, factory)

    with patch(
        "octop_harness.middleware.turn_aware_summary._configurable_from_runtime",
        return_value={"model": "p/missing"},
    ):
        assert proxy.invoke("prompt") == "seed-summary"

    seed.with_retry.return_value.invoke.assert_called_once_with("prompt")


def test_summary_uses_model_selector() -> None:
    seed = _model("seed")
    kimi = _model("kimi")
    proxy = _proxy(
        seed,
        _factory({"p/seed": seed, "p/kimi": kimi}),
        user_selector=lambda _state, _config: "p/kimi",
    )

    with patch(
        "octop_harness.middleware.turn_aware_summary._configurable_from_runtime",
        return_value={},
    ):
        assert proxy.invoke("prompt") == "kimi-summary"

    kimi.with_retry.return_value.invoke.assert_called_once_with("prompt")


def test_install_replaces_pinned_summary_model() -> None:
    seed = _model("seed")
    helper = SimpleNamespace(_summary_model=seed.with_retry())
    mw = SimpleNamespace(_lc_helper=helper)

    install_turn_aware_summary_model(
        mw,
        seed=seed,
        factory=_factory({"p/seed": seed}),
        pick_default_ref=lambda: "p/seed",
    )

    assert isinstance(helper._summary_model, TurnAwareSummaryModel)
