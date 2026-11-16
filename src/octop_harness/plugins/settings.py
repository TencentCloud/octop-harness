"""Optional host-neutral settings provider; plugins own their schema and storage."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Protocol


class PluginSettingsProvider(Protocol):
    async def get_settings(self, context: Mapping[str, Any]) -> dict[str, Any]: ...

    async def save_settings(self, context: Mapping[str, Any], values: dict[str, Any]) -> None: ...


def validate_settings_provider(provider: PluginSettingsProvider) -> None:
    for name in ("get_settings", "save_settings"):
        if not inspect.iscoroutinefunction(getattr(provider, name, None)):
            raise ValueError(f"plugin settings provider must implement async {name}()")
