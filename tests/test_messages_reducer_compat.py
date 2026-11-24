"""Regression for the DeepAgentState messages-channel reducer import.

``deepagents`` 0.7.41 dropped ``deepagents._messages_reducer`` and switched to
LangGraph's ``_messages_delta_reducer``. MemoryRuntime must import the same
symbol or fresh installs fail at ``from octop_harness.memory.runtime import ...``.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.graph.message import _messages_delta_reducer as langgraph_reducer

from octop_harness.memory import runtime as memory_runtime


def test_memory_runtime_uses_langgraph_messages_delta_reducer() -> None:
    assert memory_runtime._messages_delta_reducer is langgraph_reducer
    assert memory_runtime._CHECKPOINT_MESSAGES_CHANNEL is not None


def test_messages_delta_reducer_accepts_batched_writes() -> None:
    # Mirror how MemoryRuntime wraps the reducer for DeltaChannel.
    out = memory_runtime._messages_delta_reducer(
        [],
        [[HumanMessage(content="hi", id="m1")]],
    )
    assert len(out) == 1
    assert out[0].id == "m1"
