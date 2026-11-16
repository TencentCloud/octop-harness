"""Octop / octop-harness plugin system."""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware

from octop_harness.plugins.context import PluginContext
from octop_harness.plugins.inbound import (
    DecisionScorer,
    DecisionScorerAware,
    DecisionScorerProvider,
    InboundDispatch,
    InboundHandler,
    InboundTextRenderer,
)
from octop_harness.plugins.loader import discover_plugin_dirs, load_all, load_plugin_dir, unload_plugin
from octop_harness.plugins.manifest import PluginManifest
from octop_harness.plugins.registry import ChannelRegistration, LoadedPlugin, PluginRegistry
from octop_harness.plugins.settings import PluginSettingsProvider
from octop_harness.plugins.tools import build_plugin_tools, collect_plugin_tool_configs, get_tool_config

__all__ = [
    "AgentMiddleware",
    "ChannelRegistration",
    "DecisionScorer",
    "DecisionScorerAware",
    "DecisionScorerProvider",
    "InboundDispatch",
    "InboundHandler",
    "InboundTextRenderer",
    "LoadedPlugin",
    "PluginContext",
    "PluginManifest",
    "PluginRegistry",
    "PluginSettingsProvider",
    "build_plugin_tools",
    "collect_plugin_tool_configs",
    "discover_plugin_dirs",
    "get_tool_config",
    "load_all",
    "load_plugin_dir",
    "unload_plugin",
]
