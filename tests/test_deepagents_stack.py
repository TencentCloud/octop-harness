"""Harness middleware inside a real ``deepagents.create_deep_agent`` stack.

Most agent tests stub ``create_deep_agent``. These build the real graph so a
deepagents upgrade that reorders or adds middleware fails here.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.exceptions import ModelRefusalError
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from octop_harness.agent import HarnessAgent
from octop_harness.config import HarnessAgentConfig
from octop_harness.context_usage import CONTEXT_USAGE_KEY, message_content_text, partition_system_prompt_segments
from octop_harness.decision.config import DecisionConfig
from octop_harness.middleware.model_refusal import RefusalPassthroughMiddleware
from octop_harness.middleware.skill_filter import SkillFilterMiddleware


class _Model(FakeMessagesListChatModel):
    seen: list[list[BaseMessage]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
        self.seen.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
        return self._generate(messages, stop, run_manager, **kwargs)


def _refusal() -> AIMessage:
    return AIMessage(
        content="",
        additional_kwargs={"refusal": "I can't help with that."},
        response_metadata={"model_provider": "openai"},
    )


def _system_text(messages: list[BaseMessage]) -> str:
    return "\n".join(message_content_text(m.content) for m in messages if isinstance(m, SystemMessage))


@pytest.mark.asyncio
async def test_refusal_stays_a_reply(tmp_path: Path) -> None:
    graph = create_deep_agent(
        model=_Model(responses=[_refusal()]),
        middleware=[RefusalPassthroughMiddleware()],
        backend=FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True),
    )
    result = await graph.ainvoke({"messages": [HumanMessage(content="hi")]})
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert last.additional_kwargs["refusal"] == "I can't help with that."


@pytest.mark.asyncio
async def test_deepagents_raises_on_refusal_without_passthrough(tmp_path: Path) -> None:
    """If this stops raising, deepagents dropped the slot and the passthrough can go."""
    graph = create_deep_agent(
        model=_Model(responses=[_refusal()]),
        backend=FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True),
    )
    with pytest.raises(ModelRefusalError):
        await graph.ainvoke({"messages": [HumanMessage(content="hi")]})


def _write_skill(root: Path, slug: str) -> None:
    skill_dir = root / "skills" / slug
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {slug}\ndescription: {slug} skill description\n---\n\nBody.\n")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configurable", "expected"),
    [
        ({}, {"alpha"}),
        ({"skills": []}, set()),
    ],
)
async def test_skill_filter_shapes_skills_prompt(
    tmp_path: Path,
    configurable: dict[str, Any],
    expected: set[str],
) -> None:
    _write_skill(tmp_path, "alpha")
    _write_skill(tmp_path, "beta")
    model = _Model(responses=[AIMessage(content="ok")])
    model.seen = []
    graph = create_deep_agent(
        model=model,
        middleware=[SkillFilterMiddleware(config=HarnessAgentConfig(skills_disabled=frozenset({"beta"})))],
        skills=["/skills/"],
        backend=FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True),
    )
    await graph.ainvoke({"messages": [HumanMessage(content="hi")]}, {"configurable": configurable})

    prompt = _system_text(model.seen[0])
    shown = {slug for slug in ("alpha", "beta") if f"{slug} skill description" in prompt}
    assert shown == expected


@pytest.fixture
def harness_with_skills(
    sample_config: HarnessAgentConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[HarnessAgentConfig, _Model]:
    _write_skill(tmp_path, "alpha")
    _write_skill(tmp_path, "beta")
    (tmp_path / "_builtin_skills").mkdir()
    model = _Model(responses=[AIMessage(content="one"), AIMessage(content="two"), AIMessage(content="three")])
    monkeypatch.setattr("octop_harness.llm.factory.ChatModelFactory.get_chat_model", lambda *args, **kwargs: model)
    config = replace(
        sample_config,
        backend={"type": "filesystem", "root_dir": str(tmp_path), "virtual_mode": False},
        bootstrap_enabled=False,
        checkpointer=InMemorySaver(),
        language="en",
    )
    return config, model


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore:Direct attribute assignment to ModelRequest.model is deprecated:DeprecationWarning")
async def test_decision_keeps_skill_catalog_out_of_the_cached_prefix(
    harness_with_skills: tuple[HarnessAgentConfig, _Model], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, model = harness_with_skills
    scores = iter([{"skill:alpha": 0.95, "skill:beta": 0.1}, {"skill:alpha": 0.1, "skill:beta": 0.95}, {}])

    async def score(*args, **kwargs):
        return next(scores)

    monkeypatch.setattr("octop_harness.decision.client.HttpDecisionClient.score_async", score)
    decision = DecisionConfig(enabled=True, endpoint="https://example.invalid", tools=False, team=False)
    agent = HarnessAgent(replace(config, decision=decision))
    graph_config = {"configurable": {"thread_id": "decision-skills"}}
    for text in ("Use alpha and beta, prefer alpha", "Use alpha and beta, prefer beta", "Use alpha and beta again"):
        await agent._graph.ainvoke({"messages": [HumanMessage(content=text)]}, graph_config)

    prompts = [_system_text(messages) for messages in model.seen]
    assert prompts[0] == prompts[1] == prompts[2]
    assert "alpha skill description" not in prompts[0]
    assert "beta skill description" not in prompts[0]
    for messages, selected, unselected in zip(model.seen[:2], ("alpha", "beta"), ("beta", "alpha"), strict=True):
        note = message_content_text(messages[-1].content)
        assert note.startswith("[turn note]")
        assert f"{selected} skill description" in note
        assert f"{unselected} skill description" not in note


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore:Direct attribute assignment to ModelRequest.model is deprecated:DeprecationWarning")
async def test_context_usage_counts_the_rendered_skill_catalog(
    harness_with_skills: tuple[HarnessAgentConfig, _Model], tmp_path: Path
) -> None:
    config, model = harness_with_skills
    (tmp_path / "AGENTS.md").write_text("Keep project rules in their own context segment.")
    agent = HarnessAgent(config)
    result = await agent._graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]}, {"configurable": {"thread_id": "skills-usage"}}
    )

    prompt = _system_text(model.seen[0])
    assert "alpha skill description" in prompt
    assert "Keep project rules in their own context segment." in prompt
    expected = partition_system_prompt_segments(prompt)
    assert expected["skills"] > 0
    assert expected["rules"] > 0
    usage = result["messages"][-1].additional_kwargs[CONTEXT_USAGE_KEY]
    assert usage["segments"].get("skills", 0) == expected["skills"]
    assert usage["segments"].get("rules", 0) == expected["rules"]
    assert usage["input_tokens"] == sum(usage["segments"].values())
