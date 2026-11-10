"""Tests for ``octop_harness.api_key_scope`` — per-call keys stay out of checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from octop_harness.api_key_scope import (
    api_key_override_from,
    api_key_override_scope,
    bind_api_key_override,
    current_api_key_override,
    normalize_api_key_override,
    take_api_key_override,
)
from octop_harness.request import ChatRequest

CANARY = "sk-CANARY-MUST-NOT-PERSIST"


class _State(TypedDict):
    messages: list


async def _echo(state: _State) -> _State:
    return state


class TestNormalize:
    def test_keeps_non_empty_string(self) -> None:
        assert normalize_api_key_override("sk-1") == "sk-1"

    def test_rejects_blank_and_non_string(self) -> None:
        # Whitespace-only passes through: matches the middleware's historical
        # truthiness rule (hosts strip keys before sending).
        assert normalize_api_key_override("") is None
        assert normalize_api_key_override("   ") == "   "
        assert normalize_api_key_override(None) is None
        assert normalize_api_key_override(42) is None


class TestConfigurableHelpers:
    def test_from_reads_without_mutation(self) -> None:
        cfg = {"api_key_override": "sk-1", "thread_id": "t"}
        assert api_key_override_from(cfg) == "sk-1"
        assert cfg == {"api_key_override": "sk-1", "thread_id": "t"}

    def test_take_pops_the_key(self) -> None:
        cfg = {"api_key_override": "sk-1", "thread_id": "t"}
        assert take_api_key_override(cfg) == "sk-1"
        assert cfg == {"thread_id": "t"}

    def test_take_normalizes(self) -> None:
        assert take_api_key_override({"api_key_override": ""}) is None
        assert take_api_key_override({}) is None


class TestScopeBinding:
    def test_scope_sets_and_resets(self) -> None:
        assert current_api_key_override() is None
        with api_key_override_scope("sk-scope"):
            assert current_api_key_override() == "sk-scope"
        assert current_api_key_override() is None

    def test_sticky_bind_until_rebound(self) -> None:
        with api_key_override_scope(None):
            bind_api_key_override("sk-sticky")
            assert current_api_key_override() == "sk-sticky"
            bind_api_key_override(None)
            assert current_api_key_override() is None


class TestRunnableConfigNeverCarriesTheKey:
    def test_configurable_key_is_stripped(self) -> None:
        req = ChatRequest(
            messages="hi",
            thread_id="t1",
            configurable={"api_key_override": CANARY, "locale": "zh"},
        )
        config = req.to_runnable_config()
        assert "api_key_override" not in config["configurable"]
        # The seam stays readable on the request itself for scope binding.
        assert api_key_override_from(req.configurable) == CANARY
        assert config["configurable"]["locale"] == "zh"


@pytest.mark.asyncio
async def test_key_never_persisted_to_checkpoint_metadata(tmp_path: Path) -> None:
    """Canary: a per-call key must not appear anywhere in ``checkpoints.sqlite``."""
    db_path = tmp_path / "checkpoints.sqlite"
    builder = StateGraph(_State)
    builder.add_node("echo", _echo)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)

    config = ChatRequest(
        messages="hi",
        thread_id="t1",
        configurable={"api_key_override": CANARY},
    ).to_runnable_config()

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        graph = builder.compile(checkpointer=saver)
        await graph.ainvoke({"messages": [HumanMessage("hi")]}, config=config)

    assert CANARY.encode("utf-8") not in db_path.read_bytes(), "key leaked into checkpoints.sqlite"
