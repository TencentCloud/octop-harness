"""Summary-input cap and failure fallback for context compaction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.exceptions import ContextOverflowError

from octop_harness.middleware.summarization_limits import (
    SUMMARY_FALLBACK_TEXT,
    SUMMARY_INPUT_RETRY_TOKEN_LIMIT,
    SUMMARY_INPUT_TOKEN_LIMIT,
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
