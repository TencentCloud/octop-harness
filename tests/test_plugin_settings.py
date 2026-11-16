"""Generic settings registration does not depend on a particular plugin kind."""

from pathlib import Path

import pytest

from octop_harness.plugins import PluginContext, PluginManifest


class Provider:
    async def get_settings(self, context):
        return {"value": "initial"}

    async def save_settings(self, context, values):
        pass


@pytest.mark.parametrize("kind", ["tool", "hook", "channel", "skill"])
def test_settings_provider_registration_for_each_plugin_kind(kind):
    manifest = PluginManifest.from_dict({"id": "arbitrary-plugin", "version": "1", "kind": kind})
    context = PluginContext(manifest=manifest, source_path=Path("unused"))
    provider = Provider()
    context.settings(provider)
    assert context.to_loaded().settings is provider
    with pytest.raises(ValueError, match="already registered"):
        context.settings(provider)


def test_invalid_provider_is_rejected():
    context = PluginContext(
        manifest=PluginManifest.from_dict({"id": "sample", "version": "1"}), source_path=Path("unused")
    )
    with pytest.raises(ValueError, match="async get_settings"):
        context.settings(object())
