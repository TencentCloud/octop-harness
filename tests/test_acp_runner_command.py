"""Windows ACP runner command resolution (PATHEXT / .cmd shims)."""

from __future__ import annotations

from typing import Any

import pytest

from octop_harness.acp.models import ACPSessionError
from octop_harness.acp.service import resolve_acp_runner_argv


def test_non_windows_leaves_bare_command_unchanged() -> None:
    command, args = resolve_acp_runner_argv(
        "opencode",
        ["acp"],
        env={"PATH": r"C:\Users\liuc\AppData\Roaming\npm"},
        is_windows=False,
        which=lambda *a, **k: (_ for _ in ()).throw(AssertionError("which must not run off Windows")),
    )
    assert command == "opencode"
    assert args == ("acp",)


def test_windows_cmd_shim_is_launched_through_comspec() -> None:
    seen: dict[str, Any] = {}

    def fake_which(command: str, *, path: str | None = None, pathext: str | None = None) -> str:
        seen["command"] = command
        seen["path"] = path
        seen["pathext"] = pathext
        return r"C:\Users\liuc\AppData\Roaming\npm\opencode.CMD"

    command, args = resolve_acp_runner_argv(
        "opencode",
        ["acp", "--verbose"],
        env={
            "PATH": r"C:\Users\liuc\AppData\Roaming\npm;C:\Windows\System32",
            "PATHEXT": ".COM;.EXE;.BAT;.CMD",
            "COMSPEC": r"C:\Windows\System32\cmd.exe",
        },
        is_windows=True,
        which=fake_which,
    )
    assert seen == {
        "command": "opencode",
        "path": r"C:\Users\liuc\AppData\Roaming\npm;C:\Windows\System32",
        "pathext": ".COM;.EXE;.BAT;.CMD",
    }
    assert command == r"C:\Windows\System32\cmd.exe"
    assert args == ("/c", r"C:\Users\liuc\AppData\Roaming\npm\opencode.CMD", "acp", "--verbose")


def test_windows_bat_shim_is_launched_through_cmd() -> None:
    command, args = resolve_acp_runner_argv(
        "npx",
        ["-y", "@zed-industries/claude-agent-acp"],
        is_windows=True,
        which=lambda *a, **k: r"C:\Program Files\nodejs\npx.bat",
    )
    assert command == "cmd.exe"
    assert args == ("/c", r"C:\Program Files\nodejs\npx.bat", "-y", "@zed-industries/claude-agent-acp")


def test_windows_exe_uses_the_resolved_absolute_path() -> None:
    resolved = r"C:\Users\liuc\AppData\Roaming\npm\node_modules\opencode-ai\bin\opencode.exe"
    command, args = resolve_acp_runner_argv(
        "opencode",
        ["acp"],
        is_windows=True,
        which=lambda *a, **k: resolved,
    )
    assert command == resolved
    assert args == ("acp",)


def test_windows_missing_command_raises_actionable_error() -> None:
    with pytest.raises(ACPSessionError, match=r"opencode") as exc:
        resolve_acp_runner_argv(
            "opencode",
            ["acp"],
            is_windows=True,
            which=lambda *a, **k: None,
        )
    message = str(exc.value)
    assert "PATH" in message
    assert ".cmd" in message
    assert "CreateProcess" in message or "bare name" in message
