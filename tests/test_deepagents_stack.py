"""Harness middleware inside a real ``deepagents.create_deep_agent`` stack.

Most agent tests stub ``create_deep_agent``. These build the real graph so a
deepagents upgrade that reorders or adds middleware fails here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.exceptions import ModelRefusalError
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from octop_harness.config import HarnessAgentConfig
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
    return "\n".join(str(m.content) for m in messages if isinstance(m, SystemMessage))


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
