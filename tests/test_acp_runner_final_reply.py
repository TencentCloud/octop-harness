"""An outbound ACP turn must keep the agent's final reply even when tools ran."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from octop_harness.acp.models import ACPConfig
from octop_harness.builtin.tools import acp_runner


class _FakeService:
    """Streams canned events the way ``ACPHostedClient`` would."""

    def __init__(self, events: list[dict[str, Any]], result: dict[str, Any]) -> None:
        self._events = events
        self._result = result

    async def run_turn(self, *, on_message: Any, **_kwargs: Any) -> dict[str, Any]:
        for event in self._events:
            await on_message(event, False)
        return self._result

    async def get_session(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def get_pending_permission(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def cancel_turn(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _config() -> ACPConfig:
    return ACPConfig.from_dict(
        {
            "runners": {
                "codebuddy": {"enabled": True, "command": "codebuddy", "args": ["--acp"]},
            },
        },
    )


def _patch_service(monkeypatch: pytest.MonkeyPatch, service: _FakeService) -> None:
    monkeypatch.setattr(acp_runner, "_get_service", lambda *_args, **_kwargs: service)
    monkeypatch.setattr(acp_runner, "_thread_id_from_config", lambda: "thr-1")


async def _run(service: _FakeService, monkeypatch: pytest.MonkeyPatch) -> str:
    _patch_service(monkeypatch, service)
    return await acp_runner._run_streaming_action(
        service_key="svc",
        config=_config(),
        action_name="start",
        runner_name="codebuddy",
        message_text="fix it",
        execution_cwd=Path("."),
        timeout_seconds=None,
    )


async def test_turn_with_tool_calls_keeps_the_final_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _FakeService(
        events=[{"type": "tool_call", "kind": "edit", "detail": "edit_file"}],
        result={"status": "ok", "event": {"type": "text", "text": "已修改 src/app.py"}},
    )

    out = await _run(service, monkeypatch)

    assert "[tool_call] edit (edit_file)" in out
    assert "[assistant]" in out
    assert "已修改 src/app.py" in out


async def test_turn_without_tools_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _FakeService(
        events=[],
        result={"status": "ok", "event": {"type": "text", "text": "done"}},
    )

    out = await _run(service, monkeypatch)

    assert out.startswith("runner: codebuddy working directory:")
    assert "[assistant]\ndone" in out
    assert "[tool_call]" not in out


async def test_tool_only_turn_does_not_invent_a_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _FakeService(
        events=[{"type": "tool_call", "kind": "read", "detail": "read_file"}],
        result={"status": "ok"},
    )

    out = await _run(service, monkeypatch)

    assert out == "[tool_call] read (read_file)"


async def test_error_event_after_tools_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _FakeService(
        events=[{"type": "tool_call", "kind": "edit", "detail": "edit_file"}],
        result={"status": "ok", "event": {"type": "error", "message": "runner crashed"}},
    )

    out = await _run(service, monkeypatch)

    assert "[tool_call] edit (edit_file)" in out
    assert "[error] runner crashed" in out
