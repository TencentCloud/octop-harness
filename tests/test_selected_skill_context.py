"""Verify selected skill bodies with a real DeepAgents graph and a local fake model."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from octop_harness.config import HarnessAgentConfig
from octop_harness.middleware.skill_filter import SkillFilterMiddleware


class SelectedSkillContextTests(unittest.TestCase):
    filter_class = SkillFilterMiddleware

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="selected-skill-context-")
        self.addCleanup(temp.cleanup)
        self.workspace = Path(temp.name)
        for slug, body in (
            ("current-consultation", "CURRENT_BODY_DO_NOT_FABRICATE_BUSINESS_ANSWERS"),
            ("historical-analysis", "HISTORY_BODY_DO_NOT_AUTO_INJECT"),
        ):
            path = self.workspace / ".octop/skills" / slug / "SKILL.md"
            path.parent.mkdir(parents=True)
            path.write_text(f"---\nname: {slug}\ndescription: Synthetic test skill\n---\n\n{body}\n", encoding="utf-8")
        self.backend = FilesystemBackend(root_dir=self.workspace, virtual_mode=True)
        reference = self.workspace / ".octop/skills/current-consultation/references/detail.md"
        reference.parent.mkdir()
        reference.write_text("REFERENCE_BODY_DO_NOT_AUTO_EXPAND", encoding="utf-8")

    def graph(self, *, disabled=frozenset()):
        class CaptureModel(FakeMessagesListChatModel):
            captured: list = []

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                self.captured.append(messages)
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        model = CaptureModel(responses=[AIMessage(content="Synthetic response")])
        graph = create_deep_agent(
            model,
            backend=self.backend,
            skills=["/.octop/skills"],
            middleware=[self.filter_class(config=HarnessAgentConfig(skills_disabled=disabled), backend=self.backend)],
            checkpointer=InMemorySaver(),
        )
        return graph, model

    def test_first_request_contains_native_pinned_body_and_unselect_does_not_leak(self):
        graph, model = self.graph()
        config = {"configurable": {"thread_id": "same-thread", "skills": ["current-consultation"]}}
        result = graph.invoke({"messages": [HumanMessage("Synthetic question")]}, config=config)
        pinned = [m for m in model.captured[0] if m.additional_kwargs.get("lc_source") == "pinned_skill"]
        self.assertEqual(len(pinned), 1)
        self.assertIn("CURRENT_BODY_", pinned[0].content)
        self.assertEqual(pinned[0].additional_kwargs["skill"]["path"], "/.octop/skills/current-consultation/SKILL.md")
        self.assertNotIn("HISTORY_BODY_", str(model.captured[0]))
        self.assertNotIn("REFERENCE_BODY_", str(model.captured[0]))
        self.assertFalse(any(m.additional_kwargs.get("lc_source") == "pinned_skill" for m in result["messages"]))
        config["configurable"]["skills"] = []
        graph.invoke({"messages": [HumanMessage("Synthetic question after deselection")]}, config=config)
        self.assertNotIn("CURRENT_BODY_", str(model.captured[1]))

    def test_async_first_request_contains_body(self):
        graph, model = self.graph()
        asyncio.run(
            graph.ainvoke(
                {"messages": [HumanMessage("Synthetic question")]},
                config={"configurable": {"thread_id": "async", "skills": ["current-consultation"]}},
            )
        )
        self.assertIn("CURRENT_BODY_", str(model.captured[0]))
        self.assertNotIn("HISTORY_BODY_", str(model.captured[0]))

    def test_disabled_skill_never_injects_body(self):
        graph, model = self.graph(disabled=frozenset({"current-consultation"}))
        graph.invoke(
            {"messages": [HumanMessage("Synthetic question")]},
            config={
                "configurable": {"thread_id": "disabled", "skills": ["current-consultation"]},
            },
        )
        self.assertNotIn("CURRENT_BODY_", str(model.captured[0]))

    def test_no_selection_keeps_progressive_disclosure(self):
        graph, model = self.graph()
        graph.invoke(
            {"messages": [HumanMessage("Synthetic question")]},
            config={
                "configurable": {"thread_id": "unselected"},
            },
        )
        self.assertNotIn("CURRENT_BODY_", str(model.captured[0]))
        self.assertNotIn("HISTORY_BODY_", str(model.captured[0]))

    def test_next_turn_reads_updated_body_without_caching_previous_content(self):
        graph, model = self.graph()
        config = {"configurable": {"thread_id": "updated", "skills": ["current-consultation"]}}
        graph.invoke({"messages": [HumanMessage("First turn")]}, config=config)
        path = self.workspace / ".octop/skills/current-consultation/SKILL.md"
        path.write_text(path.read_text(encoding="utf-8").replace("CURRENT_BODY_", "UPDATED_BODY_"), encoding="utf-8")
        graph.invoke({"messages": [HumanMessage("Second turn")]}, config=config)
        self.assertIn("UPDATED_BODY_", str(model.captured[1]))
        self.assertNotIn("CURRENT_BODY_", str(model.captured[1]))

    def test_unreadable_selected_skill_stops_before_model(self):
        graph, model = self.graph()
        config = {"configurable": {"thread_id": "missing", "skills": ["current-consultation"]}}
        graph.invoke({"messages": [HumanMessage("First turn")]}, config=config)
        (self.workspace / ".octop/skills/current-consultation/SKILL.md").unlink()
        with self.assertRaisesRegex(ValueError, "Could not read all selected skill"):
            graph.invoke({"messages": [HumanMessage("The skill file is no longer readable")]}, config=config)
        self.assertEqual(len(model.captured), 1)

    def test_unknown_selection_never_treats_input_as_file_path(self):
        graph, model = self.graph()
        graph.invoke(
            {"messages": [HumanMessage("Synthetic question")]},
            config={
                "configurable": {"thread_id": "unknown", "skills": ["/.octop/skills/historical-analysis/SKILL.md"]},
            },
        )
        self.assertNotIn("HISTORY_BODY_", str(model.captured[0]))
