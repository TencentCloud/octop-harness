"""Tests for ``octop_harness.tool_env_scope``."""

from __future__ import annotations

import asyncio

import pytest

from octop_harness.tool_env_scope import (
    MAX_TOOL_ENV_ENTRIES,
    MAX_TOOL_ENV_NAME_LENGTH,
    MAX_TOOL_ENV_VALUE_LENGTH,
    bind_tool_env,
    current_tool_env,
    normalize_tool_env,
    tool_env_from,
    tool_env_scope,
)


class TestNormalize:
    def test_accepts_string_pairs(self) -> None:
        assert normalize_tool_env({"A": "x", "B": "y"}) == {"A": "x", "B": "y"}

    def test_drops_empty_and_non_string_entries(self) -> None:
        assert normalize_tool_env({"A": "", "B": None, 3: "x", "C": "z"}) == {"C": "z"}
        assert normalize_tool_env(None) == {}
        assert normalize_tool_env("not-a-mapping") == {}
        assert normalize_tool_env(["A", "x"]) == {}

    def test_caps_entry_count(self) -> None:
        raw = {f"K{i}": "v" for i in range(MAX_TOOL_ENV_ENTRIES + 5)}
        assert len(normalize_tool_env(raw)) == MAX_TOOL_ENV_ENTRIES

    def test_drops_oversized_values(self) -> None:
        assert normalize_tool_env({"K": "x" * (MAX_TOOL_ENV_VALUE_LENGTH + 1)}) == {}
        assert normalize_tool_env({"K": "x" * MAX_TOOL_ENV_VALUE_LENGTH}) == {"K": "x" * MAX_TOOL_ENV_VALUE_LENGTH}

    def test_drops_oversized_names(self) -> None:
        assert normalize_tool_env({"K" * (MAX_TOOL_ENV_NAME_LENGTH + 1): "x"}) == {}

    def test_strips_names_and_values(self) -> None:
        assert normalize_tool_env({" KEY ": "  sk-x  "}) == {"KEY": "sk-x"}

    def test_drops_whitespace_only_entries(self) -> None:
        assert normalize_tool_env({"A": "   ", "B": "\t\n", "C": "z"}) == {"C": "z"}


class TestFrom:
    def test_does_not_mutate(self) -> None:
        configurable = {"tool_env": {"A": "x"}}
        assert tool_env_from(configurable) == {"A": "x"}
        assert configurable == {"tool_env": {"A": "x"}}

    def test_missing_key_yields_empty_mapping(self) -> None:
        assert tool_env_from(None) == {}


class TestCurrent:
    def test_scope_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCTOP_TEST_GLOBAL_KEY", "global")
        assert current_tool_env("OCTOP_TEST_GLOBAL_KEY") == "global"
        bind_tool_env({"OCTOP_TEST_GLOBAL_KEY": "scoped"})
        try:
            assert current_tool_env("OCTOP_TEST_GLOBAL_KEY") == "scoped"
        finally:
            bind_tool_env({})
        assert current_tool_env("OCTOP_TEST_GLOBAL_KEY") == "global"

    def test_empty_scope_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCTOP_TEST_GLOBAL_KEY", "global")
        bind_tool_env({})
        assert current_tool_env("OCTOP_TEST_GLOBAL_KEY") == "global"

    def test_missing_everywhere_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OCTOP_TEST_MISSING_KEY", raising=False)
        bind_tool_env({})
        assert current_tool_env("OCTOP_TEST_MISSING_KEY") == ""
        assert current_tool_env("OCTOP_TEST_MISSING_KEY", default="fallback") == "fallback"

    def test_empty_env_is_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCTOP_TEST_EMPTY_KEY", "")
        bind_tool_env({})
        assert current_tool_env("OCTOP_TEST_EMPTY_KEY", default="fallback") == "fallback"

    def test_whitespace_scope_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCTOP_TEST_GLOBAL_KEY", "global")
        bind_tool_env({"OCTOP_TEST_GLOBAL_KEY": "   "})
        try:
            assert current_tool_env("OCTOP_TEST_GLOBAL_KEY") == "global"
        finally:
            bind_tool_env({})

    def test_bind_replaces_previous_values(self) -> None:
        bind_tool_env({"OCTOP_TEST_A": "1", "OCTOP_TEST_B": "2"})
        bind_tool_env({"OCTOP_TEST_A": "3"})
        try:
            assert current_tool_env("OCTOP_TEST_A") == "3"
            assert current_tool_env("OCTOP_TEST_B") == ""
        finally:
            bind_tool_env({})


class TestScope:
    def test_scope_restores_previous_binding(self) -> None:
        bind_tool_env({"OCTOP_TEST_A": "outer"})
        try:
            with tool_env_scope({"OCTOP_TEST_A": "inner", "OCTOP_TEST_B": "b"}):
                assert current_tool_env("OCTOP_TEST_A") == "inner"
                assert current_tool_env("OCTOP_TEST_B") == "b"
            assert current_tool_env("OCTOP_TEST_A") == "outer"
            assert current_tool_env("OCTOP_TEST_B") == ""
        finally:
            bind_tool_env({})


@pytest.mark.asyncio
async def test_bindings_are_task_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    ready = asyncio.Event()
    release = asyncio.Event()
    seen: dict[str, str] = {}

    async def worker(name: str, key: str) -> None:
        bind_tool_env({"DEEPSEEK_API_KEY": key})
        ready.set()
        await release.wait()
        seen[name] = current_tool_env("DEEPSEEK_API_KEY")

    bind_tool_env({})
    tasks = [asyncio.create_task(worker("a", "key-a")), asyncio.create_task(worker("b", "key-b"))]
    await ready.wait()
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(*tasks)
    assert seen == {"a": "key-a", "b": "key-b"}
    # The parent context is not polluted by either child.
    assert current_tool_env("DEEPSEEK_API_KEY") == ""


@pytest.mark.asyncio
async def test_child_task_inherits_binding() -> None:
    bind_tool_env({"DEEPSEEK_API_KEY": "parent"})
    try:

        async def child() -> str:
            return current_tool_env("DEEPSEEK_API_KEY")

        assert await asyncio.create_task(child()) == "parent"
    finally:
        bind_tool_env({})
