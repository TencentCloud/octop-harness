"""Summary-input cap and failure fallback for context compaction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.exceptions import ContextOverflowError

from octop_harness.config import HarnessAgentConfig
from octop_harness.middleware.summarization_limits import (
    SUMMARY_FALLBACK_TEXT,
    SUMMARY_INPUT_RETRY_TOKEN_LIMIT,
    SUMMARY_INPUT_TOKEN_LIMIT,
    apply_summarization_policy,
    harden_summarization_middleware,
)


def _middleware(
    *,
    trim: int | None,
    sync_error: Exception | None = None,
    errors: list[Exception] | None = None,
    seen: list[int | None] | None = None,
) -> SimpleNamespace:
    helper = SimpleNamespace(trim_tokens_to_summarize=trim)
    pending = list(errors or [])

    def _fail() -> None:
        if sync_error is not None:
            raise sync_error
        if pending:
            raise pending.pop(0)

    def _create(_messages: object) -> str:
        if seen is not None:
            seen.append(helper.trim_tokens_to_summarize)
        _fail()
        return "summary"

    async def _acreate(_messages: object) -> str:
        if seen is not None:
            seen.append(helper.trim_tokens_to_summarize)
        _fail()
        return "summary"

    return SimpleNamespace(_lc_helper=helper, _create_summary=_create, _acreate_summary=_acreate)


def _upstream_apply(messages: list[object], event: dict[str, object] | None) -> list[object]:
    """The upstream out-of-range behavior: keep only the summary."""
    if event is None:
        return list(messages)
    summary = event["summary_message"]
    cutoff = event["cutoff_index"]
    assert isinstance(cutoff, int)
    if cutoff > len(messages):
        return [summary]
    return [summary, *messages[cutoff:]]


def _cutoff_middleware() -> SimpleNamespace:
    helper = SimpleNamespace(trim_tokens_to_summarize=SUMMARY_INPUT_TOKEN_LIMIT)

    async def _acreate(_messages: object) -> str:
        return "summary"

    return SimpleNamespace(
        _lc_helper=helper,
        _create_summary=lambda _messages: "summary",
        _acreate_summary=_acreate,
        _apply_event_to_messages=_upstream_apply,
    )


def test_cutoff_overflow_keeps_recent_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCTOP_SUMMARY_FALLBACK_KEEP", "20")
    mw = _cutoff_middleware()
    harden_summarization_middleware(mw)
    messages = [SimpleNamespace(type="human", content=str(index)) for index in range(264)]
    event = {"summary_message": SimpleNamespace(type="system", content="sum"), "cutoff_index": 296}
    result = mw._apply_event_to_messages(messages, event)
    assert len(result) == 21
    assert result[0] is event["summary_message"]
    assert result[-1] is messages[-1]


def test_cutoff_overflow_on_empty_history_keeps_summary_only() -> None:
    mw = _cutoff_middleware()
    harden_summarization_middleware(mw)
    summary = SimpleNamespace(type="system", content="sum")
    result = mw._apply_event_to_messages([], {"summary_message": summary, "cutoff_index": 5})
    assert result == [summary]


def test_cutoff_equal_to_message_count_stays_summary_only() -> None:
    mw = _cutoff_middleware()
    harden_summarization_middleware(mw)
    messages = [SimpleNamespace(type="human", content="only")]
    summary = SimpleNamespace(type="system", content="sum")
    result = mw._apply_event_to_messages(messages, {"summary_message": summary, "cutoff_index": 1})
    assert result == [summary]


def test_cutoff_inside_the_list_is_unchanged() -> None:
    mw = _cutoff_middleware()
    harden_summarization_middleware(mw)
    messages = [SimpleNamespace(type="human", content=str(index)) for index in range(30)]
    event = {"summary_message": SimpleNamespace(type="system", content="sum"), "cutoff_index": 10}
    result = mw._apply_event_to_messages(messages, event)
    assert result[0] is event["summary_message"]
    assert result[1:] == messages[10:]


def test_cutoff_overflow_includes_the_tool_call_that_owns_a_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCTOP_SUMMARY_FALLBACK_KEEP", "1")
    mw = _cutoff_middleware()
    harden_summarization_middleware(mw)
    messages = [
        SimpleNamespace(type="human", content="ask"),
        SimpleNamespace(type="ai", content="calling"),
        SimpleNamespace(type="tool", content="result"),
    ]
    event = {"summary_message": SimpleNamespace(type="system", content="sum"), "cutoff_index": 99}
    result = mw._apply_event_to_messages(messages, event)
    assert [message.type for message in result] == ["system", "ai", "tool"]


def test_caps_unset_summary_input() -> None:
    mw = _middleware(trim=None)
    harden_summarization_middleware(mw)
    assert mw._lc_helper.trim_tokens_to_summarize == SUMMARY_INPUT_TOKEN_LIMIT
    assert mw._create_summary([]) == "summary"


def test_keeps_explicit_summary_input_limit() -> None:
    mw = _middleware(trim=2000)
    harden_summarization_middleware(mw)
    assert mw._lc_helper.trim_tokens_to_summarize == 2000


def test_summary_failure_retries_then_uses_placeholder() -> None:
    seen: list[int | None] = []
    mw = _middleware(trim=None, sync_error=ContextOverflowError("prompt is too long"), seen=seen)
    harden_summarization_middleware(mw)
    assert mw._create_summary(["history"]) == SUMMARY_FALLBACK_TEXT
    assert seen == [SUMMARY_INPUT_TOKEN_LIMIT, SUMMARY_INPUT_RETRY_TOKEN_LIMIT]
    assert mw._lc_helper.trim_tokens_to_summarize == SUMMARY_INPUT_TOKEN_LIMIT


def test_summary_failure_skips_retry_when_window_already_small() -> None:
    seen: list[int | None] = []
    mw = _middleware(trim=2000, sync_error=ContextOverflowError("prompt is too long"), seen=seen)
    harden_summarization_middleware(mw)
    assert mw._create_summary(["history"]) == SUMMARY_FALLBACK_TEXT
    assert seen == [2000]


@pytest.mark.asyncio
async def test_async_summary_failure_uses_placeholder() -> None:
    seen: list[int | None] = []
    mw = _middleware(trim=None, sync_error=ContextOverflowError("prompt is too long"), seen=seen)
    harden_summarization_middleware(mw)
    assert await mw._acreate_summary(["history"]) == SUMMARY_FALLBACK_TEXT
    assert seen == [SUMMARY_INPUT_TOKEN_LIMIT, SUMMARY_INPUT_RETRY_TOKEN_LIMIT]


def test_rate_limit_failure_preserves_the_thread() -> None:
    mw = _middleware(trim=None, sync_error=RuntimeError("HTTP 429"))
    harden_summarization_middleware(mw)
    with pytest.raises(RuntimeError, match="HTTP 429"):
        mw._create_summary(["history"])


def test_rate_limit_after_overflow_preserves_the_thread() -> None:
    seen: list[int | None] = []
    mw = _middleware(
        trim=None,
        errors=[ContextOverflowError("prompt is too long"), RuntimeError("HTTP 429")],
        seen=seen,
    )
    harden_summarization_middleware(mw)
    with pytest.raises(RuntimeError, match="HTTP 429"):
        mw._create_summary(["history"])
    assert seen == [SUMMARY_INPUT_TOKEN_LIMIT, SUMMARY_INPUT_RETRY_TOKEN_LIMIT]
    assert mw._lc_helper.trim_tokens_to_summarize == SUMMARY_INPUT_TOKEN_LIMIT


@pytest.mark.asyncio
async def test_async_rate_limit_after_overflow_preserves_the_thread() -> None:
    mw = _middleware(
        trim=None,
        errors=[ContextOverflowError("prompt is too long"), RuntimeError("HTTP 429")],
    )
    harden_summarization_middleware(mw)
    with pytest.raises(RuntimeError, match="HTTP 429"):
        await mw._acreate_summary(["history"])
    assert mw._lc_helper.trim_tokens_to_summarize == SUMMARY_INPUT_TOKEN_LIMIT


def test_stale_index_with_resolvable_message_id_defers_to_upstream() -> None:
    """A cutoff id still in the list wins over the index-based approximation."""
    seen: list[dict[str, object]] = []

    def _id_aware_apply(messages: list[object], event: dict[str, object] | None) -> list[object]:
        assert event is not None
        seen.append(event)
        start = next(i for i, m in enumerate(messages) if getattr(m, "id", None) == event["cutoff_message_id"])
        return [event["summary_message"], *messages[start:]]

    mw = _cutoff_middleware()
    mw._apply_event_to_messages = _id_aware_apply
    harden_summarization_middleware(mw)
    messages = [SimpleNamespace(id=f"m{index}", type="human", content=str(index)) for index in range(30)]
    event = {
        "summary_message": SimpleNamespace(type="system", content="sum"),
        "cutoff_index": 296,  # stale: past the end of the list
        "cutoff_message_id": "m25",
    }
    result = mw._apply_event_to_messages(messages, event)
    assert seen == [event]
    assert result == [event["summary_message"], *messages[25:]]


def test_stale_index_with_missing_message_id_still_keeps_recent_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCTOP_SUMMARY_FALLBACK_KEEP", "20")
    mw = _cutoff_middleware()
    harden_summarization_middleware(mw)
    messages = [SimpleNamespace(id=f"m{index}", type="human", content=str(index)) for index in range(30)]
    event = {
        "summary_message": SimpleNamespace(type="system", content="sum"),
        "cutoff_index": 296,
        "cutoff_message_id": "gone",
    }
    result = mw._apply_event_to_messages(messages, event)
    assert len(result) == 21
    assert result[-1] is messages[-1]


def _real_summarization_middleware() -> Any:
    from deepagents.backends import StateBackend
    from deepagents.middleware.summarization import SummarizationMiddleware
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    model = FakeMessagesListChatModel(responses=[AIMessage(content="ok")])
    model.profile = {"max_input_tokens": 100_000}
    return SummarizationMiddleware(
        model=model,
        backend=StateBackend(),
        trigger=("fraction", 0.85),
        keep=("fraction", 0.10),
        trim_tokens_to_summarize=SUMMARY_INPUT_TOKEN_LIMIT,
        truncate_args_settings={"trigger": ("fraction", 0.85), "keep": ("fraction", 0.10)},
    )


def test_policy_overrides_trigger_keep_trim_and_argument_truncation(tmp_path: Path) -> None:
    config = HarnessAgentConfig(
        workspace_dir=tmp_path,
        summarization_trigger=("fraction", 0.5),
        summarization_keep=("fraction", 0.2),
        summarization_trim_tokens=3000,
    )
    mw = _real_summarization_middleware()
    apply_summarization_policy(mw, config)
    assert mw._lc_helper.trigger == ("fraction", 0.5)
    assert mw._lc_helper._trigger_clauses == [{"fraction": 0.5}]
    assert mw._lc_helper.keep == ("fraction", 0.2)
    assert mw._lc_helper.trim_tokens_to_summarize == 3000
    assert mw._truncate_args_trigger == ("fraction", 0.5)
    assert mw._truncate_args_keep == ("fraction", 0.2)


def test_disabling_clears_the_auto_trigger_and_keeps_the_middleware(tmp_path: Path) -> None:
    config = HarnessAgentConfig(workspace_dir=tmp_path, summarization_enabled=False)
    mw = _real_summarization_middleware()
    apply_summarization_policy(mw, config)
    assert mw._lc_helper._trigger_clauses == []
    assert mw._truncate_args_trigger is None
    assert mw._lc_helper.keep == ("fraction", 0.10)


def test_summarization_config_round_trip_and_rejects_keep_above_trigger(tmp_path: Path) -> None:
    config = HarnessAgentConfig(
        workspace_dir=tmp_path,
        summarization_trigger=("fraction", 0.6),
        summarization_keep=("messages", 4),
        summarization_trim_tokens=2500,
    )
    loaded = HarnessAgentConfig.from_dict(config.to_dict())
    assert loaded.summarization_trigger == ("fraction", 0.6)
    assert loaded.summarization_keep == ("messages", 4)
    assert loaded.summarization_trim_tokens == 2500
    with pytest.raises(ValueError, match="less than"):
        HarnessAgentConfig(
            workspace_dir=tmp_path,
            summarization_trigger=("fraction", 0.4),
            summarization_keep=("fraction", 0.4),
        )
