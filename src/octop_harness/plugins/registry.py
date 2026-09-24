"""In-process plugin registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from octop_harness.plugins.manifest import PluginManifest

ConfigField = dict[str, Any]
ToolFn = Callable[..., Any]


@dataclass
class ToolRegistration:
    plugin_id: str
    name: str
    fn: ToolFn
    description: str
    config_fields: list[ConfigField] = field(default_factory=list)


@dataclass
class MiddlewareRegistration:
    plugin_id: str
    instance: Any
    priority: int = 100


@dataclass
class SkillRegistration:
    plugin_id: str
    skills_dir: Path


@dataclass
class LoadedPlugin:
    manifest: PluginManifest
    source_path: Path
    tools: list[ToolRegistration] = field(default_factory=list)
    middleware: list[MiddlewareRegistration] = field(default_factory=list)
    skills_dir: Path | None = None
    diagnostics: list[str] = field(default_factory=list)
    context: Any | None = None


class PluginRegistry:
    """Global registry of loaded plugins."""

    _instance: PluginRegistry | None = None
    _plugins: dict[str, LoadedPlugin]

    def __new__(cls) -> PluginRegistry:
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._plugins = {}
            cls._instance = inst
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def clear(self) -> None:
        self._plugins.clear()

    def register(self, loaded: LoadedPlugin) -> None:
        self._plugins[loaded.manifest.id] = loaded

    def unregister(self, plugin_id: str) -> LoadedPlugin | None:
        return self._plugins.pop(plugin_id, None)

    def get(self, plugin_id: str) -> LoadedPlugin | None:
        return self._plugins.get(plugin_id)

    def list_plugins(self) -> list[LoadedPlugin]:
        return list(self._plugins.values())

    def all_loaded(self) -> list[LoadedPlugin]:
        return self.list_plugins()

    def all_tools(self) -> list[ToolRegistration]:
        out: list[ToolRegistration] = []
        for plugin in self._plugins.values():
            out.extend(plugin.tools)
        return out

    def build_middleware_chain(self, *, global_enabled: dict[str, bool] | None = None) -> list[Any]:
        """Return middleware instances sorted by priority (lower runs earlier)."""
        enabled = global_enabled or {}
        entries: list[MiddlewareRegistration] = []
        for plugin in self._plugins.values():
            if enabled.get(plugin.manifest.id) is False:
                continue
            entries.extend(plugin.middleware)
        entries.sort(key=lambda e: e.priority)
        return [e.instance for e in entries]
