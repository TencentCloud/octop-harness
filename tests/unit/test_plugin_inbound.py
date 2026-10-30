"""Registration and cleanup of host-neutral admission hooks."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from octop_harness.plugins import PluginContext, PluginManifest, PluginRegistry


def context(tmp_path: Path, kind: str = "hook") -> PluginContext:
    return PluginContext(
        manifest=PluginManifest(id="admission", name="Admission", version="1", kind=kind, entry="main.py"),
        source_path=tmp_path,
    )


class Handler:
    def __init__(self) -> None:
        self.close = Mock()

    async def handle(self, channel, message, dispatch):
        return False


@pytest.fixture(autouse=True)
def registry():
    PluginRegistry.reset()
    yield PluginRegistry()
    PluginRegistry().clear()
    PluginRegistry.reset()


def test_replace_unregister_and_clear_close_owned_handlers(tmp_path, registry):
    old, new = Handler(), Handler()
    first = context(tmp_path)
    first.inbound(old)
    registry.register(first.to_loaded())
    assert registry.all_inbound() == [old]
    second = context(tmp_path)
    second.inbound(new)
    registry.register(second.to_loaded())
    old.close.assert_called_once_with()
    assert registry.all_inbound() == [new]
    registry.clear()
    registry.unregister("admission")
    new.close.assert_called_once_with()
    assert registry.all_inbound() == []


def test_bad_replacement_preserves_running_plugin(tmp_path, registry):
    old = Handler()
    ctx = context(tmp_path)
    ctx.inbound(old)
    registry.register(ctx.to_loaded())
    replacement = context(tmp_path).to_loaded()
    replacement.inbound = [object()]
    with pytest.raises(ValueError, match=r"handle.*close"):
        registry.register(replacement)
    old.close.assert_not_called()
    assert registry.all_inbound() == [old]


def test_cleanup_failure_does_not_skip_other_handlers(tmp_path, registry):
    first, second = Handler(), Handler()
    first.close.side_effect = RuntimeError("cleanup failed")
    ctx = context(tmp_path)
    ctx.inbound(first)
    ctx.inbound(second)
    registry.register(ctx.to_loaded())
    registry.clear()
    second.close.assert_called_once_with()
    assert registry.all_inbound() == []


def test_registration_rejects_non_hook_and_async_close(tmp_path):
    with pytest.raises(ValueError, match="kind=hook"):
        context(tmp_path, "tool").inbound(Handler())

    class AsyncClose:
        async def handle(self, channel, message, dispatch):
            return False

        async def close(self):
            pass

    handler = AsyncClose()
    with pytest.raises(ValueError, match="synchronous"):
        context(tmp_path).inbound(handler)
