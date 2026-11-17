"""YOLO mode selects an unattended ACP session mode and auto-allows permissions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from octop_harness.acp.client import ACPHostedClient
from octop_harness.acp.models import ACPRunnerConfig
from octop_harness.acp.permission_mode import (
    ask_mode_id,
    mode_state,
    select_session_mode,
    yolo_mode_id,
    yolo_option_id,
)


def _mode(mode_id: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(id=mode_id, name=name)


def test_explicit_session_mode_wins_when_the_agent_offers_it() -> None:
    modes = [
        _mode("default", "Default"),
        _mode("plan", "Plan"),
        _mode("bypassPermissions", "Bypass Permissions"),
    ]
    assert select_session_mode(modes, "ask", "plan") == ("ask", "plan")
    assert select_session_mode(modes, "yolo", "plan") == ("yolo", "plan")
    assert select_session_mode(modes, "ask", "bypassPermissions") == ("yolo", "bypassPermissions")
    assert select_session_mode(modes, "yolo", "missing") == ("yolo", "bypassPermissions")
    assert select_session_mode([], "ask", "build") == ("ask", "build")
    assert select_session_mode([], "yolo", "build") == ("yolo", "build")
    full = [_mode("read-only", "Read only"), _mode("full-access", "Full access")]
    assert select_session_mode(full, "ask", "full-access") == ("yolo", "full-access")


def test_yolo_mode_matches_bypass_and_yolo_names() -> None:
    modes = [
        _mode("default", "Default"),
        _mode("plan", "Plan"),
        _mode("bypassPermissions", "Bypass Permissions"),
    ]
    assert yolo_mode_id(modes) == "bypassPermissions"
    assert yolo_mode_id([_mode("yolo", "YOLO"), _mode("code", "Code")]) == "yolo"
    assert yolo_mode_id([_mode("full-access", "Full access")]) == "full-access"
    assert yolo_mode_id([_mode("dontAsk", "Don't ask"), _mode("default", "Default")]) is None
    assert yolo_mode_id([_mode("code", "Code")]) is None


def test_ask_mode_restores_the_agents_default() -> None:
    modes = [_mode("default", "Default"), _mode("yolo", "YOLO")]
    assert ask_mode_id(modes, "default") == "default"
    assert ask_mode_id(modes, "yolo") == "default"
    assert ask_mode_id([_mode("yolo", "YOLO")], "yolo") is None
    deny_first = [_mode("dontAsk", "Don't ask"), _mode("default", "Default")]
    assert ask_mode_id(deny_first, "yolo") == "default"
    assert ask_mode_id(deny_first, "dontAsk") == "default"


def test_mode_state_reads_the_session_response() -> None:
    response = SimpleNamespace(
        modes=SimpleNamespace(
            current_mode_id="default",
            available_modes=[_mode("default", "Default"), _mode("yolo", "YOLO")],
        )
    )
    current, available, default = mode_state(response)
    assert current == "default"
    assert default == "default"
    assert [item.id for item in available] == ["default", "yolo"]
    assert mode_state(SimpleNamespace()) == (None, [], None)


def test_yolo_option_prefers_a_permanent_allow() -> None:
    options = [
        {"optionId": "allow-once", "kind": "allow_once"},
        {"optionId": "allow-always", "kind": "allow_always"},
        {"optionId": "reject", "kind": "reject_once"},
    ]
    assert yolo_option_id(options) == "allow-always"


class _Option:
    def model_dump(self, **_kwargs: Any) -> dict[str, str]:
        return {"optionId": "allow-always", "kind": "allow_always", "name": "Always allow"}


class _Call:
    def model_dump(self, **_kwargs: Any) -> dict[str, str]:
        return {"title": "Edit README.md", "kind": "edit"}


@pytest.mark.asyncio
async def test_yolo_answers_permission_without_waiting() -> None:
    client = ACPHostedClient(
        runner_name="codebuddy",
        runner_config=ACPRunnerConfig(command="codebuddy"),
        cwd=".",
    )
    client.set_permission_mode("yolo")
    response = await client.request_permission([_Option()], "session", _Call())
    assert response.outcome.outcome == "selected"
    assert response.outcome.option_id == "allow-always"
    assert client.pending_permission is None
