"""Unit tests for octop_harness.plugins."""

from __future__ import annotations

from pathlib import Path

import pytest

from octop_harness.plugins.context import PluginContext
from octop_harness.plugins.loader import load_plugin_dir
from octop_harness.plugins.manifest import PluginManifest
from octop_harness.plugins.registry import PluginRegistry
from octop_harness.plugins.tools import _tool_enabled

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "plugins" / "echo-tool"


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    PluginRegistry.reset()
    yield
    PluginRegistry.reset()


def test_manifest_from_yaml(tmp_path: Path) -> None:
    (tmp_path / "plugin.yaml").write_text(
        "id: demo\nversion: 0.1.0\nname: Demo\nkind: tool\nentry: main.py\n",
        encoding="utf-8",
    )
    manifest = PluginManifest.load(tmp_path / "plugin.yaml")
    assert manifest.id == "demo"
    assert manifest.kind == "tool"


def test_context_rejects_wrong_kind(tmp_path: Path) -> None:
    manifest = PluginManifest(
        id="x",
        version="1",
        name="x",
        kind="hook",
        entry="main.py",
    )
    ctx = PluginContext(manifest=manifest, source_path=tmp_path)
    with pytest.raises(ValueError, match="not tool"):
        ctx.tool("t", lambda: None)


def test_load_echo_fixture() -> None:
    loaded = load_plugin_dir(_FIXTURE, install_deps=False)
    assert loaded.manifest.id == "echo-tool"
    assert loaded.tools[0].name == "echo_message"


def test_tool_enabled_defaults_on_when_plugin_globally_on() -> None:
    assert _tool_enabled("echo", "echo_message", agent_plugins={}, global_plugins={}) is True
    assert (
        _tool_enabled(
            "echo",
            "echo_message",
            agent_plugins={"echo": {"tools": {"echo_message": {"config": {"x": 1}}}}},
            global_plugins={"echo": True},
        )
        is True
    )
    assert (
        _tool_enabled(
            "echo",
            "echo_message",
            agent_plugins={"echo": {"tools": {"echo_message": {"enabled": False}}}},
            global_plugins={"echo": True},
        )
        is False
    )
    assert (
        _tool_enabled(
            "echo",
            "echo_message",
            agent_plugins={},
            global_plugins={"echo": False},
        )
        is False
    )
