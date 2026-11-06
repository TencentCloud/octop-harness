"""Regression coverage for the outbound ACP tool result."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from octop_harness.acp.models import ACPConfig
from octop_harness.builtin.tools import acp_runner


async def _run_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    streamed: tuple[dict[str, Any], ...] = (),
    final: dict[str, Any] | None = None,
    permission: bool = False,
) -> str:
    async def run_turn(**kwargs: Any) -> dict[str, Any]:
        for payload in streamed:
            await kwargs["on_message"](payload, True)
        if permission:
            return {
                "status": "permission_required",
                "event": final,
                "suspended_permission": SimpleNamespace(summary="Allow the edit?", options=[]),
            }
        return {"status": "completed", "event": final}

    service = SimpleNamespace(run_turn=run_turn)
    monkeypatch.setattr(acp_runner, "_get_service", lambda *_: service)
    monkeypatch.setattr(acp_runner, "_thread_id_from_config", lambda: "test-thread")
    config = ACPConfig.from_dict({"runners": {"test": {"enabled": True, "command": "unused"}}})
    return await acp_runner._run_streaming_action(
        service_key="test",
        config=config,
        action_name="message",
        runner_name="test",
        message_text="do work",
        execution_cwd=tmp_path,
        timeout_seconds=None,
    )


_TOOL = {"type": "tool_call", "kind": "edit", "detail": "edit_file"}
_REPLY = {"type": "text", "text": "Updated app.py."}


@pytest.mark.asyncio
async def test_tool_events_and_final_assistant_reply_are_both_returned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = await _run_turn(monkeypatch, tmp_path, streamed=(_TOOL,), final=_REPLY)
    assert result == "[tool_call] edit (edit_file)\n\n[assistant]\nUpdated app.py."


@pytest.mark.asyncio
async def test_terminal_error_follows_tool_events(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = await _run_turn(
        monkeypatch, tmp_path, streamed=(_TOOL,), final={"type": "error", "message": "runner failed"}
    )
    assert result == "[tool_call] edit (edit_file)\n\n[error] runner failed"


@pytest.mark.asyncio
async def test_tool_free_and_tool_only_output_stays_the_same(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert await _run_turn(monkeypatch, tmp_path, final=_REPLY) == (
        f"runner: test working directory: {tmp_path}\n\n[assistant]\nUpdated app.py."
    )
    assert await _run_turn(monkeypatch, tmp_path, streamed=(_TOOL,)) == "[tool_call] edit (edit_file)"


@pytest.mark.asyncio
async def test_already_streamed_reply_is_not_repeated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = await _run_turn(monkeypatch, tmp_path, streamed=(_TOOL, _REPLY), final=_REPLY)
    assert result.count("[assistant]\nUpdated app.py.") == 1


@pytest.mark.asyncio
async def test_permission_output_still_asks_for_an_option(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = await _run_turn(monkeypatch, tmp_path, streamed=(_TOOL,), final=_REPLY, permission=True)
    assert result == (
        "[tool_call] edit (edit_file)\n\n[permission_required] Allow the edit?\n"
        "Choose an option id and call acp_runner(action='respond', ...):"
    )
