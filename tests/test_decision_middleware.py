"""Decision shortlist: confidence cutoff, HTTP parse, and middleware."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.config import var_child_runnable_config
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from octop_harness.config import HarnessAgentConfig
from octop_harness.decision import decision_active, parse_probabilities
from octop_harness.decision.client import _ATTEMPTS, DecisionRequestError, HttpDecisionClient, _timeout_for
from octop_harness.decision.config import DecisionConfig
from octop_harness.decision.select import Candidate, keep_confident, lexical_shortlist
from octop_harness.middleware.decision import DecisionMiddleware, decision_dispatch_snapshot
from octop_harness.middleware.tool_search import ToolSearchMiddleware
from octop_harness.middleware.tools_filter import ToolsFilterMiddleware


class _Scorer:
    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities
        self.calls = 0
        self.statements: list[tuple[str, str]] = []

    def score(self, *, state: str, statements: list[tuple[str, str]]) -> dict[str, float]:
        self.calls += 1
        self.state = state
        self.statements = list(statements)
        return dict(self.probabilities)

    async def score_async(self, *, state: str, statements: list[tuple[str, str]]) -> dict[str, float]:
        return self.score(state=state, statements=statements)


@contextmanager
def _configurable(*, run_id: str = "", **kwargs: object):
    config: dict[str, object] = {"configurable": kwargs}
    if run_id:
        config["run_id"] = run_id
    token = var_child_runnable_config.set(config)
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


def _request(
    *,
    tools: list[Any] | None = None,
    state: dict[str, Any] | None = None,
    text: str = "帮我读 pdf",
    messages: list[Any] | None = None,
) -> ModelRequest:
    return ModelRequest(
        model=MagicMock(),
        messages=messages if messages is not None else [HumanMessage(content=text)],
        system_message=SystemMessage(content="base"),
        tool_choice=None,
        tools=tools or [],
        response_format=None,
        state=state or {},
        runtime=MagicMock(),
        model_settings={},
    )


def _tool(name: str, description: str = "") -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.description = description or name
    return tool


def _note(request: ModelRequest) -> str:
    if not request.messages:
        return ""
    content = request.messages[-1].content
    return content if isinstance(content, str) else str(content)


def _config(**overrides: Any) -> DecisionConfig:
    raw: dict[str, Any] = {
        "enabled": True,
        "endpoint": "http://127.0.0.1:9/v1/decisions",
        "skills": True,
        "tools": True,
        "team": True,
        "threshold": 0.6,
        "tool_min_count": 2,
        "max_selected": 8,
    }
    raw.update(overrides)
    return DecisionConfig(**raw)


def test_decision_active_requires_switch_endpoint_and_a_task() -> None:
    assert decision_active(None) is False
    assert decision_active(DecisionConfig(enabled=False, endpoint="http://localhost/d")) is False
    assert decision_active(DecisionConfig(enabled=True, endpoint="")) is False
    assert (
        decision_active(
            DecisionConfig(enabled=True, endpoint="http://localhost/d", skills=False, tools=False, team=False)
        )
        is False
    )
    assert decision_active(_config()) is True


def test_keep_confident_falls_back_when_nothing_clears_the_bar() -> None:
    assert keep_confident({"a": 0.2, "b": 0.4}, ["a", "b"], threshold=0.6, limit=2) is None
    assert keep_confident({"a": 0.9, "b": 0.7, "c": 0.1}, ["a", "b", "c"], threshold=0.6, limit=1) == ["a"]


def test_lexical_shortlist_skips_when_nothing_overlaps() -> None:
    items = [Candidate(id=str(i), label=f"tool-{i}", description="unrelated") for i in range(20)]
    assert lexical_shortlist(items, "读 pdf 合同") is None
    matched = lexical_shortlist(
        [*items, Candidate(id="pdf", label="read_pdf", description="读取 pdf 合同")],
        "读 pdf 合同",
    )
    assert matched is not None
    assert matched[0].id == "pdf"


def test_parse_probabilities_accepts_common_shapes() -> None:
    assert parse_probabilities({"answers": {"a": {"noul": 0.8}}}) == {"a": 0.8}
    assert parse_probabilities({"answers": [{"id": "a", "probability": 0.8}]}) == {"a": 0.8}
    assert parse_probabilities({"results": [{"id": "b", "p_yes": "0.4"}]}) == {"b": 0.4}
    assert parse_probabilities({"questions": [{"id": "c", "answer": {"yes": 0.7}}]}) == {"c": 0.7}


def test_local_laya_allows_a_full_cpu_batch_to_finish() -> None:
    assert _timeout_for("laya").read == 60
    assert _timeout_for("jev").read == 15


def test_build_decision_body_uses_noul_questions() -> None:
    from octop_harness.decision import build_decision_body

    body = build_decision_body(
        state="帮我读 pdf",
        statements=[("skill:pdf", "这轮需要技能「pdf」。")],
        model="typesafe/jev-1.13",
    )
    assert body["model"] == "typesafe/jev-1.13"
    question = body["questions"]["skill:pdf"]
    assert question["type"] == "noul"
    assert question["criteria"]["true"] == "适用"


def test_middleware_narrows_skills_and_leaves_them_when_unsure(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[dict[str, Any]] = []
    monkeypatch.setattr("langgraph.config.get_stream_writer", lambda: events.append)
    state = {
        "skills_metadata": [
            {"name": "pdf", "description": "读取 pdf"},
            {"name": "xlsx", "description": "表格"},
        ]
    }
    sure = DecisionMiddleware(
        config=_config(tools=False, team=False), scorer=_Scorer({"skill:pdf": 0.9, "skill:xlsx": 0.1})
    )
    narrowed = sure.wrap_model_call(_request(state=state), lambda request: request)
    assert [row["name"] for row in narrowed.state["skills_metadata"]] == ["pdf"]

    unsure = DecisionMiddleware(
        config=_config(tools=False, team=False), scorer=_Scorer({"skill:pdf": 0.2, "skill:xlsx": 0.2})
    )
    kept = unsure.wrap_model_call(_request(state=state), lambda request: request)
    assert [row["name"] for row in kept.state["skills_metadata"]] == ["pdf", "xlsx"]
    assert capsys.readouterr().err == ""
    assert events[0]["kind"] == "decision"
    assert events[0]["empty"] is False
    assert events[0]["parts"] == [{"kind": "skill", "kept": [{"name": "pdf", "score": 0.9}]}]
    assert events[1]["parts"] == [{"kind": "skill", "kept": None}]


def test_middleware_drops_skill_definitions_and_keeps_that_cut() -> None:
    prompt = (
        "**Available Skills:**\n"
        "\n"
        "- **pdf**: 读取 pdf\n"
        "  -> Read `/skills/pdf/SKILL.md` for full instructions\n"
        "- **xlsx**: 表格\n"
        "  -> Read `/skills/xlsx/SKILL.md` for full instructions\n"
        "\n"
        "**How to Use Skills (Progressive Disclosure):**\n"
    )
    state = {
        "skills_metadata": [
            {"name": "pdf", "description": "读取 pdf"},
            {"name": "xlsx", "description": "表格"},
        ]
    }
    scorer = _Scorer({"skill:pdf": 0.9, "skill:xlsx": 0.1})
    mw = DecisionMiddleware(config=_config(tools=False, team=False), scorer=scorer)
    request = ModelRequest(
        model=MagicMock(),
        messages=[HumanMessage(content="读 pdf")],
        system_message=SystemMessage(content=prompt),
        tool_choice=None,
        tools=[],
        response_format=None,
        state=state,
        runtime=MagicMock(),
        model_settings={},
    )
    first = mw.wrap_model_call(request, lambda item: item)
    assert "**How to Use Skills" in first.system_message.content
    assert "xlsx" not in first.system_message.content
    assert "- **pdf**:" not in first.system_message.content
    assert "pdf" in _note(first)
    assert "xlsx" not in _note(first)
    assert [row["name"] for row in first.state["skills_metadata"]] == ["pdf"]
    second = mw.wrap_model_call(request, lambda item: item)
    assert second.system_message.content == first.system_message.content
    assert "xlsx" not in _note(second)
    assert scorer.calls == 1


def test_middleware_skips_tools_below_the_minimum() -> None:
    scorer = _Scorer({"tool:read_file": 0.99})
    mw = DecisionMiddleware(config=_config(skills=False, team=False, tool_min_count=8), scorer=scorer)
    tools = [_tool("read_file", "读取 pdf"), _tool("execute")]
    out = mw.wrap_model_call(_request(tools=tools), lambda request: request)
    assert out.tools == tools
    assert scorer.calls == 0


def test_middleware_keeps_skill_runtime_tools_beside_the_shortlist() -> None:
    tools = [
        _tool("task", "派一个子任务"),
        _tool("read_file", "读取文件"),
        _tool("glob", "按模式找文件"),
        _tool("write_todos", "记录待办"),
        _tool("execute", "运行命令"),
        _tool("web_fetch", "抓取网页"),
        _tool("browser_use", "打开网页"),
    ]
    scorer = _Scorer(
        {
            "tool:task": 0.91,
            "tool:read_file": 0.1,
            "tool:glob": 0.1,
            "tool:write_todos": 0.1,
            "tool:execute": 0.1,
            "tool:web_fetch": 0.5,
            "tool:browser_use": 0.2,
        }
    )
    mw = DecisionMiddleware(config=_config(skills=False, team=False, tool_min_count=2), scorer=scorer)
    out = mw.wrap_model_call(_request(tools=tools, text="今日热点早报"), lambda request: request)
    assert [tool.name for tool in out.tools] == [
        "task",
        "read_file",
        "glob",
        "write_todos",
        "execute",
        "web_fetch",
        "browser_use",
    ]


def test_low_tool_scores_keep_capability_tools() -> None:
    tools = [_tool("read_file", "读取文件"), _tool("web_fetch", "抓取网页"), _tool("execute", "运行命令")]
    tools.extend(_tool(f"misc_{i}", "unrelated") for i in range(6))
    tools.append(_tool("browser_use", "打开网页"))
    tools.append(_tool("send_file_to_user", "把文件发给用户"))
    scorer = _Scorer({"tool:browser_use": 0.12, **{f"tool:misc_{i}": 0.1 for i in range(6)}})
    mw = DecisionMiddleware(config=_config(skills=False, team=False, tool_min_count=2), scorer=scorer)
    request = _request(tools=tools, text="今天天气")
    events: list[dict[str, Any]] = []

    def reply(item: ModelRequest) -> ModelResponse:
        events.append({"tools": [tool.name for tool in item.tools], "note": _note(item)})
        return ModelResponse(result=[AIMessage(content="好")])

    first = mw.wrap_model_call(request, reply)
    assert events[0]["tools"] == [
        "read_file",
        "web_fetch",
        "execute",
        "browser_use",
        "send_file_to_user",
        "call_deferred_tool",
    ]
    assert "misc_0" not in events[0]["note"]
    part = first.result[0].additional_kwargs["octop_decision"]["parts"][0]
    assert part == {"kind": "tool", "kept": []}
    scorer.probabilities = {"tool:browser_use": 0.95}
    mw.wrap_model_call(request, reply)
    assert events[1]["tools"] == [
        "read_file",
        "web_fetch",
        "execute",
        "browser_use",
        "send_file_to_user",
        "call_deferred_tool",
    ]
    assert scorer.calls == 1


def test_unscored_tools_stay_in_the_catalog() -> None:
    tools = [_tool(f"misc_{i}", "unrelated") for i in range(8)]
    tools.append(_tool("read_pdf", "读取 pdf 合同"))
    scorer = _Scorer({"tool:not-asked": 0.2})
    mw = DecisionMiddleware(config=_config(skills=False, team=False, tool_min_count=2), scorer=scorer)
    out = mw.wrap_model_call(_request(tools=tools, text="读取 pdf 合同"), lambda item: item)
    assert [tool.name for tool in out.tools] == ["call_deferred_tool"]
    assert "read_pdf" in _note(out)
    assert "misc_0" in _note(out)


def test_middleware_keeps_tools_it_already_allowed_this_turn() -> None:
    tools = [_tool(f"misc_{i}", "unrelated") for i in range(8)]
    tools.extend([_tool("read_pdf", "读取 pdf 合同"), _tool("plugin_browser", "打开网页")])
    scorer = _Scorer({"tool:read_pdf": 0.91})
    mw = DecisionMiddleware(config=_config(skills=False, team=False, tool_min_count=2), scorer=scorer)
    request = _request(tools=tools, text="读取 pdf 合同")
    first = mw.wrap_model_call(request, lambda item: item)
    assert [tool.name for tool in first.tools] == ["call_deferred_tool"]
    assert "read_pdf" in _note(first)
    assert "plugin_browser" not in _note(first)
    scorer.probabilities = {"tool:plugin_browser": 0.93}
    second = mw.wrap_model_call(request, lambda item: item)
    assert [tool.name for tool in second.tools] == ["call_deferred_tool"]
    assert _note(second) == _note(first)
    assert second.system_message.content == first.system_message.content
    assert scorer.calls == 1


def test_question_budget_keeps_unasked_catalog_entries() -> None:
    skills = [{"name": f"skill-{index}", "description": "读取 pdf"} for index in range(12)]
    tools = [_tool("read_file", "读取文件")]
    tools.extend(_tool(f"tool-{index}", "读取 pdf") for index in range(12))
    scorer = _Scorer({"skill:skill-0": 0.91, "tool:tool-0": 0.91})
    mw = DecisionMiddleware(config=_config(team=False, tool_min_count=2), scorer=scorer)
    out = mw.wrap_model_call(
        _request(tools=tools, state={"skills_metadata": skills}, text="读取 pdf"),
        lambda request: request,
    )
    asked = [item_id for item_id, _ in scorer.statements]
    assert sum(item_id.startswith("skill:") for item_id in asked) == 10
    assert sum(item_id.startswith("tool:") for item_id in asked) == 10
    assert "tool:read_file" not in asked
    assert [row["name"] for row in out.state["skills_metadata"]] == ["skill-0", "skill-10", "skill-11"]
    assert [tool.name for tool in out.tools] == ["read_file", "call_deferred_tool"]
    assert "- tool-0:" in _note(out)
    assert "- tool-10:" in _note(out)
    assert "- tool-11:" in _note(out)
    assert "- tool-1:" not in _note(out)


def test_turn_cache_drops_the_oldest_entry() -> None:
    from octop_harness.middleware.decision import _TurnFilter

    mw = DecisionMiddleware(config=_config(skills=False, tools=False, team=False), scorer=_Scorer({}))
    for index in range(129):
        mw._remember_turn(str(index), _TurnFilter())
    assert "0" not in mw._turns
    assert "1" in mw._turns
    assert "128" in mw._turns
    assert len(mw._turns) == 128


def test_repeated_user_text_starts_a_new_decision_pass() -> None:
    tools = [_tool(f"misc_{i}", "unrelated") for i in range(8)]
    tools.append(_tool("read_pdf", "读取 pdf 合同"))
    scorer = _Scorer({"tool:read_pdf": 0.91})
    mw = DecisionMiddleware(config=_config(skills=False, team=False, tool_min_count=2), scorer=scorer)

    mw.wrap_model_call(_request(tools=tools, text="读取 pdf 合同"), lambda item: item)
    repeated = _request(
        tools=tools,
        messages=[
            HumanMessage(content="读取 pdf 合同"),
            AIMessage(content="第一次回答"),
            HumanMessage(content="读取 pdf 合同"),
        ],
    )
    mw.wrap_model_call(repeated, lambda item: item)

    assert scorer.calls == 2


def test_later_user_turn_sends_prior_conversation_to_the_decision_model() -> None:
    tools = [_tool(f"misc_{i}", "unrelated") for i in range(16)]
    tools.append(_tool("smtp_submit", "发送邮件"))
    scorer = _Scorer({"tool:smtp_submit": 0.91})
    mw = DecisionMiddleware(config=_config(skills=False, team=False, tool_min_count=2), scorer=scorer)
    request = _request(
        tools=tools,
        messages=[
            HumanMessage(content="帮我给张三写一封请假邮件"),
            AIMessage(content="草稿已经写好了"),
            HumanMessage(content="按刚才说的做"),
        ],
    )
    out = mw.wrap_model_call(request, lambda item: item)
    assert "请假邮件" in scorer.state
    assert "按刚才说的做" in scorer.state
    assert "草稿已经写好了" in scorer.state
    assert "smtp_submit" in [item_id.split(":", 1)[1] for item_id, _ in scorer.statements]
    assert "smtp_submit" in _note(out)


def test_decision_state_keeps_the_latest_ask_when_history_overflows() -> None:
    from octop_harness.middleware.decision import _STATE_LIMIT, _state_text

    earlier = "帮我处理这份很长的合同。" + ("条款" * (_STATE_LIMIT))
    request = _request(
        messages=[
            HumanMessage(content=earlier),
            AIMessage(content="草稿已经写好了。" + ("好" * (_STATE_LIMIT))),
            HumanMessage(content="按刚才说的做"),
        ]
    )
    state = _state_text(request, chinese=True)
    assert len(state) <= _STATE_LIMIT
    assert "按刚才说的做" in state
    assert state.rstrip("…").endswith("按刚才说的做")


def test_middleware_keeps_only_confident_tools() -> None:
    tools = [_tool(f"misc_{i}", "unrelated") for i in range(8)]
    tools.append(_tool("read_pdf", "读取 pdf 合同"))
    scorer = _Scorer({"tool:read_pdf": 0.91})
    mw = DecisionMiddleware(config=_config(skills=False, team=False, tool_min_count=2), scorer=scorer)
    out = mw.wrap_model_call(_request(tools=tools, text="读取 pdf 合同"), lambda request: request)
    assert [tool.name for tool in out.tools] == ["call_deferred_tool"]
    assert "read_pdf" in _note(out)
    assert "misc_0" not in _note(out)


def test_middleware_pins_members_and_blocks_the_rest() -> None:
    researcher = MagicMock(agent_id="agent-research", metadata={"description": "查资料"})
    writer = MagicMock(agent_id="agent-writer", metadata={"description": "写稿"})
    team = MagicMock()
    team.list_peers.return_value = [researcher, writer]
    team.peer_display_name.side_effect = lambda entry: "研究员" if entry is researcher else "写作者"
    scorer = _Scorer({"member:agent-research": 0.91, "member:agent-writer": 0.2})
    mw = DecisionMiddleware(
        config=_config(skills=False, tools=False),
        team=team,
        scorer=scorer,
    )
    with _configurable(peer_invoke_mode="async", user=1, agent_id="host", thread_id="room-1"):
        hinted = mw.wrap_model_call(_request(text="请安排人查一下资料"), lambda request: request)
        assert hinted.system_message.content == "base"
        assert "研究员" in _note(hinted)
        listed = mw.wrap_tool_call(
            ToolCallRequest(
                tool_call={"name": "agent_list", "args": {}, "id": "l1"},
                tool=None,  # type: ignore[arg-type]
                state={},  # type: ignore[arg-type]
                runtime=None,  # type: ignore[arg-type]
            ),
            lambda _request: ToolMessage(
                content='[{"agent_id": "agent-research", "short_id": "search", "name": "研究员"},'
                '{"agent_id": "agent-writer", "short_id": "writer", "name": "写作者"}]',
                tool_call_id="l1",
            ),
        )
        assert "agent-writer" not in listed.content
        blocked = mw.wrap_tool_call(
            ToolCallRequest(
                tool_call={"name": "ask_agent", "args": {"expert": "写作者"}, "id": "a1"},
                tool=None,  # type: ignore[arg-type]
                state={},  # type: ignore[arg-type]
                runtime=None,  # type: ignore[arg-type]
            ),
            lambda _request: ToolMessage(content="ok", tool_call_id="a1"),
        )
        assert blocked.status == "error"


def test_team_host_without_request_mode_still_pins_members() -> None:
    researcher = MagicMock(agent_id="agent-research", metadata={"description": "查资料"})
    writer = MagicMock(agent_id="agent-writer", metadata={"description": "写稿"})
    team = MagicMock()
    team.list_peers.return_value = [researcher, writer]
    team.peer_display_name.side_effect = lambda entry: "研究员" if entry is researcher else "写作者"
    team.peer_invoke_mode_for.return_value = "async"
    scorer = _Scorer({"member:agent-research": 0.91, "member:agent-writer": 0.2})
    mw = DecisionMiddleware(config=_config(skills=False, tools=False), team=team, scorer=scorer)
    with _configurable(user=1, agent_id="host", thread_id="room-1"):
        hinted = mw.wrap_model_call(_request(text="请安排人查一下资料"), lambda request: request)
    assert hinted.system_message.content == "base"
    assert "研究员" in _note(hinted)
    assert scorer.calls == 1


def test_sync_dispatch_does_not_pin_members() -> None:
    team = MagicMock()
    team.peer_invoke_mode_for.return_value = "async"
    scorer = _Scorer({"member:agent-research": 0.99})
    mw = DecisionMiddleware(config=_config(skills=False, tools=False), team=team, scorer=scorer)
    with _configurable(peer_invoke_mode="sync", user=1, agent_id="host", thread_id="room-1"):
        mw.wrap_model_call(_request(text="请安排人查一下资料"), lambda request: request)
    assert scorer.calls == 0
    team.list_peers.assert_not_called()


def _team() -> MagicMock:
    researcher = MagicMock(agent_id="agent-research", metadata={"description": "查资料"})
    writer = MagicMock(agent_id="agent-writer", metadata={"description": "写稿"})
    team = MagicMock()
    team.list_peers.return_value = [researcher, writer]
    team.peer_display_name.side_effect = lambda entry: "研究员" if entry is researcher else "写作者"
    team.researcher = researcher
    return team


def _host_tools() -> list[Any]:
    return [_tool("ask_agent", "派给成员"), _tool("agent_list", "列出成员"), _tool("current_time", "现在")]


def test_host_decision_lists_dispatch_tools() -> None:
    team = _team()
    scorer = _Scorer({"member:agent-research": 0.91, "member:agent-writer": 0.2})
    mw = DecisionMiddleware(config=_config(skills=False), team=team, scorer=scorer)
    seen: dict[str, list[str]] = {}

    def reply(item: ModelRequest) -> ModelResponse:
        seen["tools"] = [tool.name for tool in item.tools]
        return ModelResponse(result=[AIMessage(content="好")])

    with _configurable(peer_invoke_mode="async", user=1, agent_id="host", thread_id="room-1"):
        out = mw.wrap_model_call(_request(tools=_host_tools(), text="请安排人查一下资料"), reply)
    assert seen["tools"] == ["ask_agent", "agent_list", "current_time"]
    parts = out.result[0].additional_kwargs["octop_decision"]["parts"]
    assert parts[0] == {"kind": "tool", "kept": [{"name": "ask_agent"}, {"name": "agent_list"}]}
    assert parts[1]["kind"] == "member"
    assert "tool:ask_agent" not in [item_id for item_id, _ in scorer.statements]


def test_followup_scores_the_next_member_not_the_one_who_just_finished() -> None:
    team = _team()
    scorer = _Scorer({"result:sufficient": 0.2, "member:agent-writer": 0.91})
    mw = DecisionMiddleware(config=_config(skills=False), team=team, scorer=scorer)
    request = _request(
        tools=_host_tools(),
        messages=[
            HumanMessage(content="请安排人查资料并写一份简报"),
            AIMessage(content="交给研究员"),
            HumanMessage(content="[系统唤醒] 团队成员 研究员 已完成. 任务: 查资料"),
        ],
    )
    seen: dict[str, str] = {}

    def reply(item: ModelRequest) -> ModelResponse:
        seen["note"] = _note(item)
        return ModelResponse(result=[AIMessage(content="好")])

    with _configurable(
        peer_invoke_mode="async",
        user=1,
        agent_id="host",
        thread_id="room-1",
        octop_decision_followup=True,
        octop_decision_finished_member="agent-research",
    ):
        out = mw.wrap_model_call(request, reply)
    assert scorer.calls == 1
    assert "请安排人查资料并写一份简报" in scorer.state
    assert "研究员 已完成" in scorer.state
    assert [item_id for item_id, _ in scorer.statements] == [
        "result:sufficient",
        "member:agent-writer",
    ]
    assert "写作者" in seen["note"]
    assert "研究员" not in seen["note"].split("这轮只派给")[-1]
    parts = out.result[0].additional_kwargs["octop_decision"]["parts"]
    assert parts[0]["kind"] == "result"
    assert parts[1]["kind"] == "tool"
    assert parts[2]["kept"][0]["name"] == "写作者"


def test_followup_filters_catalogs_with_the_dispatched_task() -> None:
    skills = [{"name": f"misc-{index}", "description": "unrelated"} for index in range(12)]
    skills.append({"name": "pdf", "description": "读取 pdf 合同"})
    state = {"skills_metadata": skills}
    scorer = _Scorer({"skill:pdf": 0.9})
    mw = DecisionMiddleware(
        config=_config(tools=False, team=False),
        scorer=scorer,
    )
    request = _request(
        state=state,
        messages=[HumanMessage(content="[系统唤醒] 团队成员已完成")],
    )
    with _configurable(
        thread_id="room-1",
        octop_decision_followup=True,
        octop_decision_task="读取 pdf 合同",
    ):
        mw.wrap_model_call(request, lambda item: item)
    assert scorer.state.startswith("读取 pdf 合同")
    assert [item_id for item_id, _ in scorer.statements] == ["skill:pdf"]


def test_followup_low_scores_close_dispatch() -> None:
    team = _team()
    scorer = _Scorer({"result:sufficient": 0.2, "member:agent-writer": 0.1})
    mw = DecisionMiddleware(config=_config(skills=False), team=team, scorer=scorer)
    request = _request(
        tools=_host_tools(),
        messages=[
            HumanMessage(content="请安排人查资料"),
            HumanMessage(content="[系统唤醒] 团队成员 研究员 已完成"),
        ],
    )
    seen: dict[str, str] = {}

    def reply(item: ModelRequest) -> ModelResponse:
        seen["note"] = _note(item)
        seen["tools"] = [tool.name for tool in item.tools]
        return ModelResponse(result=[AIMessage(content="好")])

    with _configurable(
        peer_invoke_mode="async",
        user=1,
        agent_id="host",
        thread_id="room-1",
        octop_decision_followup=True,
        octop_decision_finished_member="agent-research",
    ):
        out = mw.wrap_model_call(request, reply)
        blocked = mw.wrap_tool_call(
            ToolCallRequest(
                tool_call={"name": "ask_agent", "args": {"expert": "写作者"}, "id": "a1"},
                tool=None,  # type: ignore[arg-type]
                state={},  # type: ignore[arg-type]
                runtime=None,  # type: ignore[arg-type]
            ),
            lambda _request: ToolMessage(content="ok", tool_call_id="a1"),
        )
    assert "不必再派工" in seen["note"]
    assert seen["tools"] == ["ask_agent", "agent_list", "current_time"]
    assert blocked.status == "error"
    parts = out.result[0].additional_kwargs["octop_decision"]["parts"]
    assert parts == [
        {
            "kind": "result",
            "kept": [{"name": "结果仍需补充", "score": 0.2}],
        },
        {"kind": "member", "kept": []},
    ]


def test_followup_evaluates_result_when_no_member_remains() -> None:
    researcher = MagicMock(agent_id="agent-research", metadata={"description": "查资料"})
    team = MagicMock()
    team.list_peers.return_value = [researcher]
    team.peer_display_name.return_value = "研究员"
    scorer = _Scorer({"result:sufficient": 0.91})
    mw = DecisionMiddleware(config=_config(skills=False), team=team, scorer=scorer)
    request = _request(
        tools=_host_tools(),
        messages=[
            HumanMessage(content="请安排人查资料"),
            HumanMessage(content="[系统唤醒] 团队成员 研究员 已完成"),
        ],
    )
    seen: dict[str, str] = {}

    def reply(item: ModelRequest) -> ModelResponse:
        seen["note"] = _note(item)
        return ModelResponse(result=[AIMessage(content="好")])

    with _configurable(
        peer_invoke_mode="async",
        user=1,
        agent_id="host",
        thread_id="room-1",
        octop_decision_followup=True,
        octop_decision_finished_member="agent-research",
    ):
        out = mw.wrap_model_call(request, reply)
    assert [item_id for item_id, _ in scorer.statements] == ["result:sufficient"]
    assert "结果已经足够" in seen["note"]
    parts = out.result[0].additional_kwargs["octop_decision"]["parts"]
    assert parts == [
        {
            "kind": "result",
            "kept": [{"name": "结果充分, 可以总结", "score": 0.91}],
        }
    ]


def test_named_member_is_evaluated_without_overriding_user_choice() -> None:
    researcher = MagicMock(agent_id="agent-research", metadata={})
    writer = MagicMock(agent_id="agent-writer", metadata={})
    team = MagicMock()
    team.list_peers.return_value = [researcher, writer]
    team.peer_display_name.side_effect = lambda entry: "研究员" if entry is researcher else "写作者"
    scorer = _Scorer({"member:agent-research": 0.1})
    mw = DecisionMiddleware(config=_config(skills=False, tools=False), team=team, scorer=scorer)
    with _configurable(peer_invoke_mode="async", user=1, agent_id="host", thread_id="room-1"):
        out = mw.wrap_model_call(
            _request(text="请 @研究员 处理"),
            lambda _request: ModelResponse(result=[AIMessage(content="好")]),
        )
    assert scorer.calls == 1
    assert [item_id for item_id, _ in scorer.statements] == ["member:agent-research"]
    part = out.result[0].additional_kwargs["octop_decision"]["parts"][0]
    assert part == {
        "kind": "member",
        "kept": [{"name": "研究员", "score": 0.1}],
    }


def test_member_name_without_at_sign_does_not_force_routing() -> None:
    team = _team()
    scorer = _Scorer({"member:agent-research": 0.1, "member:agent-writer": 0.9})
    mw = DecisionMiddleware(config=_config(skills=False, tools=False), team=team, scorer=scorer)
    with _configurable(peer_invoke_mode="async", user=1, agent_id="host", thread_id="room-1"):
        out = mw.wrap_model_call(
            _request(text="研究员的方案可以参考。请安排最合适的人处理"),
            lambda _request: ModelResponse(result=[AIMessage(content="好")]),
        )
    part = out.result[0].additional_kwargs["octop_decision"]["parts"][0]
    assert part == {
        "kind": "member",
        "kept": [{"name": "写作者", "score": 0.9}],
    }


@pytest.mark.asyncio
async def test_empty_score_does_not_change_the_request(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {
        "skills_metadata": [
            {"name": "pdf", "description": "读取 pdf"},
            {"name": "xlsx", "description": "表格"},
        ]
    }
    events: list[dict[str, Any]] = []
    monkeypatch.setattr("langgraph.config.get_stream_writer", lambda: events.append)
    mw = DecisionMiddleware(config=_config(tools=False, team=False), scorer=_Scorer({}))

    async def passthrough(request: ModelRequest) -> ModelRequest:
        return request

    out = await mw.awrap_model_call(_request(state=state), passthrough)
    assert [row["name"] for row in out.state["skills_metadata"]] == ["pdf", "xlsx"]
    assert capsys.readouterr().err == ""
    assert events == [{"kind": "decision", "model": "laya", "empty": True, "parts": []}]


class _FailingScorer:
    def score(self, *, state: str, statements: list[tuple[str, str]]) -> dict[str, float]:
        raise DecisionRequestError("http://127.0.0.1:9/v1/decisions", "ConnectError")

    async def score_async(self, *, state: str, statements: list[tuple[str, str]]) -> dict[str, float]:
        raise DecisionRequestError("http://127.0.0.1:9/v1/decisions", "ConnectError")


@pytest.mark.asyncio
async def test_failed_decision_request_keeps_catalog_and_reports_the_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "skills_metadata": [
            {"name": "pdf", "description": "读取 pdf"},
            {"name": "xlsx", "description": "表格"},
        ]
    }
    events: list[dict[str, Any]] = []
    monkeypatch.setattr("langgraph.config.get_stream_writer", lambda: events.append)
    mw = DecisionMiddleware(config=_config(tools=False, team=False), scorer=_FailingScorer())
    seen: list[ModelRequest] = []

    async def reply(request: ModelRequest) -> ModelResponse:
        seen.append(request)
        return ModelResponse(result=[AIMessage(content="好")])

    out = await mw.awrap_model_call(_request(state=state), reply)
    assert [row["name"] for row in seen[0].state["skills_metadata"]] == ["pdf", "xlsx"]
    assert events == [
        {
            "kind": "decision",
            "model": "laya",
            "empty": True,
            "parts": [],
            "error": "request_failed",
            "detail": "ConnectError",
        }
    ]
    stamped = out.result[0].additional_kwargs["octop_decision"]
    assert stamped["error"] == "request_failed"
    assert stamped["detail"] == "ConnectError"


def test_provider_native_tools_stay_callable_and_are_not_scored() -> None:
    native = {"type": "function", "function": {"name": "web_search", "description": "搜索网页", "parameters": {}}}
    tools: list[Any] = [_tool(f"misc_{index}", "unrelated") for index in range(8)]
    tools.append(native)
    scorer = _Scorer({f"tool:misc_{index}": 0.1 for index in range(8)})
    mw = DecisionMiddleware(config=_config(skills=False, team=False, tool_min_count=2), scorer=scorer)
    out = mw.wrap_model_call(_request(tools=tools, text="搜索网页"), lambda item: item)
    assert native in out.tools
    assert "tool:web_search" not in {item_id for item_id, _text in scorer.statements}


def test_deferred_catalogs_stay_with_their_run() -> None:
    pdf = _tool("read_pdf", "读取 pdf")
    sheet = _tool("read_sheet", "读取表格")
    tools_pdf = [*(_tool(f"misc_{index}", "unrelated") for index in range(8)), pdf]
    tools_sheet = [*(_tool(f"misc_{index}", "unrelated") for index in range(8)), sheet]
    mw = DecisionMiddleware(
        config=_config(skills=False, team=False, tool_min_count=2),
        scorer=_Scorer({"tool:read_pdf": 0.95, "tool:read_sheet": 0.95}),
    )
    with _configurable(run_id="run-a", thread_id="room"):
        mw.wrap_model_call(_request(tools=tools_pdf, text="读取 pdf"), lambda item: item)
    with _configurable(run_id="run-b", thread_id="room"):
        mw.wrap_model_call(_request(tools=tools_sheet, text="读取表格"), lambda item: item)

    def forward(name: str, *, run_id: str) -> ToolMessage:
        def handler(item: ToolCallRequest) -> ToolMessage:
            return ToolMessage(content=item.tool_call["name"], name=str(item.tool_call["name"]), tool_call_id="c")

        with _configurable(run_id=run_id, thread_id="room"):
            result = mw.wrap_tool_call(
                ToolCallRequest(
                    tool_call={"name": "call_deferred_tool", "args": {"name": name, "arguments": {}}, "id": "c"},
                    tool=None,  # type: ignore[arg-type]
                    state={},  # type: ignore[arg-type]
                    runtime=None,  # type: ignore[arg-type]
                ),
                handler,
            )
        assert isinstance(result, ToolMessage)
        return result

    assert forward("read_pdf", run_id="run-a").content == "read_pdf"
    assert forward("read_pdf", run_id="run-b").status == "error"
    assert forward("read_sheet", run_id="run-b").content == "read_sheet"


def test_laya_keeps_its_read_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    seen: dict[str, httpx.Timeout] = {}

    def post(*_args: object, **kwargs: object) -> httpx.Response:
        timeout = kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        seen["timeout"] = timeout
        request = httpx.Request("POST", "http://127.0.0.1:9/v1/decisions")
        return httpx.Response(200, json={"answers": {"skill:pdf": {"noul": 0.8}}}, request=request)

    monkeypatch.setattr(httpx, "post", post)
    client = HttpDecisionClient(endpoint="http://127.0.0.1:9/v1/decisions", provider="laya")
    assert client.score(state="读 pdf", statements=[("skill:pdf", "需要 pdf")]) == {"skill:pdf": 0.8}
    assert seen["timeout"].read == pytest.approx(60, abs=1)

    def remote_post(*_args: object, **kwargs: object) -> httpx.Response:
        timeout = kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        seen["remote"] = timeout
        request = httpx.Request("POST", "http://127.0.0.1:9/v1/decisions")
        return httpx.Response(200, json={"answers": {}}, request=request)

    monkeypatch.setattr(httpx, "post", remote_post)
    remote = HttpDecisionClient(endpoint="http://127.0.0.1:9/v1/decisions")
    assert remote.score(state="读 pdf", statements=[("skill:pdf", "需要 pdf")]) == {}
    assert seen["remote"].read == pytest.approx(5, abs=0.5)


def test_decision_client_retries_transport_errors_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    calls = {"n": 0}

    def post(*_args: object, **_kwargs: object) -> object:
        calls["n"] += 1
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr("octop_harness.decision.client.time.sleep", lambda _delay: None)
    client = HttpDecisionClient(endpoint="http://127.0.0.1:9/v1/decisions", provider="laya")
    with pytest.raises(DecisionRequestError):
        client.score(state="读 pdf", statements=[("skill:pdf", "需要 pdf")])
    assert calls["n"] == _ATTEMPTS


def test_decision_client_strict_mode_raises_the_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    def post(*_args: object, **_kwargs: object) -> object:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "post", post)
    client = HttpDecisionClient(endpoint="http://127.0.0.1:9/v1/decisions")
    with pytest.raises(httpx.ConnectError):
        client.score(state="读 pdf", statements=[("skill:pdf", "需要 pdf")], strict=True)


@pytest.mark.asyncio
async def test_decision_client_async_retries_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    calls = {"n": 0}

    class _Client:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def post(self, *_args: object, **_kwargs: object) -> object:
            calls["n"] += 1
            raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    client = HttpDecisionClient(endpoint="http://127.0.0.1:9/v1/decisions")
    with pytest.raises(DecisionRequestError):
        await client.score_async(state="读 pdf", statements=[("skill:pdf", "需要 pdf")])
    assert calls["n"] == _ATTEMPTS
    calls["n"] = 0
    with pytest.raises(httpx.ConnectError):
        await client.score_async(state="读 pdf", statements=[("skill:pdf", "需要 pdf")], strict=True)
    assert calls["n"] == 1


def test_decision_request_error_is_public() -> None:
    from octop_harness import decision as decision_api

    assert decision_api.DecisionRequestError is DecisionRequestError


def test_decision_client_does_not_retry_a_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    calls = {"n": 0}

    def post(*_args: object, **_kwargs: object) -> httpx.Response:
        calls["n"] += 1
        request = httpx.Request("POST", "http://127.0.0.1:9/v1/decisions")
        return httpx.Response(400, request=request)

    monkeypatch.setattr(httpx, "post", post)
    client = HttpDecisionClient(endpoint="http://127.0.0.1:9/v1/decisions")
    with pytest.raises(DecisionRequestError):
        client.score(state="读 pdf", statements=[("skill:pdf", "需要 pdf")])
    assert calls["n"] == 1


def test_middleware_reports_tokens_kept_out_while_the_model_runs() -> None:
    from octop_harness.context_usage import pending_decision_saved

    tools = [_tool(f"misc_{index}", "unrelated tool") for index in range(8)]
    tools.append(_tool("read_pdf", "读取 pdf 合同"))
    scorer = _Scorer({"tool:read_pdf": 0.91})
    mw = DecisionMiddleware(config=_config(skills=False, team=False, tool_min_count=2), scorer=scorer)
    request = _request(tools=tools, text="读取 pdf 合同")
    seen: dict[str, int] = {}

    def handler(item: ModelRequest) -> ModelRequest:
        seen["saved"] = pending_decision_saved.get()
        return item

    mw.wrap_model_call(request, handler)
    assert seen["saved"] > 0
    mw.wrap_model_call(request, handler)
    assert seen["saved"] > 0
    assert scorer.calls == 1


def test_middleware_stamps_the_reply_once_per_turn() -> None:
    tools = [_tool(f"misc_{i}", "unrelated") for i in range(8)]
    tools.append(_tool("read_pdf", "读取 pdf 合同"))
    scorer = _Scorer({"tool:read_pdf": 0.91})
    mw = DecisionMiddleware(config=_config(skills=False, team=False, tool_min_count=2), scorer=scorer)
    request = _request(tools=tools, text="读取 pdf 合同")

    def reply(_request: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="好")])

    first = mw.wrap_model_call(request, reply)
    stamped = first.result[0].additional_kwargs["octop_decision"]
    assert stamped["kind"] == "decision"
    assert stamped["parts"][0]["kind"] == "tool"
    assert stamped["parts"][0]["kept"][0]["name"] == "read_pdf"
    second = mw.wrap_model_call(request, reply)
    assert "octop_decision" not in second.result[0].additional_kwargs
    assert scorer.calls == 1


def test_turn_selection_does_not_change_the_cached_prefix() -> None:
    prompt = (
        "persona\n"
        "**Available Skills:**\n"
        "\n"
        "- **pdf**: 读取 pdf\n"
        "- **xlsx**: 表格\n"
        "\n"
        "**How to Use Skills (Progressive Disclosure):**\n"
        "tail\n"
    )
    tools = [_tool("read_file", "读取文件"), _tool("read_pdf", "读取 pdf"), _tool("plugin_browser", "打开网页")]
    state = {
        "skills_metadata": [
            {"name": "pdf", "description": "读取 pdf"},
            {"name": "xlsx", "description": "表格"},
        ]
    }
    scorer = _Scorer({"skill:pdf": 0.95, "skill:xlsx": 0.1, "tool:read_pdf": 0.95, "tool:plugin_browser": 0.1})
    mw = DecisionMiddleware(config=_config(team=False, tool_min_count=2), scorer=scorer)

    def make(messages: list[Any]) -> ModelRequest:
        return ModelRequest(
            model=MagicMock(),
            messages=messages,
            system_message=SystemMessage(content=prompt),
            tool_choice=None,
            tools=list(tools),
            response_format=None,
            state=dict(state),
            runtime=MagicMock(),
            model_settings={},
        )

    first = mw.wrap_model_call(make([HumanMessage(content="读 pdf")]), lambda item: item)
    scorer.probabilities = {"skill:pdf": 0.1, "skill:xlsx": 0.95, "tool:read_pdf": 0.1, "tool:plugin_browser": 0.95}
    second = mw.wrap_model_call(
        make(
            [
                HumanMessage(content="读 pdf"),
                AIMessage(content="好"),
                HumanMessage(content="打开网页"),
            ]
        ),
        lambda item: item,
    )
    assert first.system_message.content == second.system_message.content
    assert "tail" in first.system_message.content
    assert "- **pdf**:" not in first.system_message.content
    assert (
        [tool.name for tool in first.tools]
        == [tool.name for tool in second.tools]
        == [
            "read_file",
            "call_deferred_tool",
        ]
    )
    assert "pdf" in _note(first) and "xlsx" not in _note(first)
    assert "read_pdf" in _note(first) and "plugin_browser" not in _note(first)
    assert "xlsx" in _note(second) and "plugin_browser" in _note(second)
    assert scorer.calls == 2


def test_deferred_tool_call_forwards_to_the_selected_tool() -> None:
    tools = [_tool(f"misc_{index}", "unrelated") for index in range(8)]
    real = _tool("read_pdf", "读取 pdf")
    tools.append(real)
    mw = DecisionMiddleware(
        config=_config(skills=False, team=False, tool_min_count=2),
        scorer=_Scorer({"tool:read_pdf": 0.95}),
    )
    mw.wrap_model_call(_request(tools=tools, text="读取 pdf 合同"), lambda item: item)
    seen: dict[str, Any] = {}

    def handler(item: ToolCallRequest) -> ToolMessage:
        seen["name"] = item.tool_call["name"]
        seen["args"] = item.tool_call["args"]
        seen["tool"] = item.tool
        return ToolMessage(content="ok", name=str(item.tool_call["name"]), tool_call_id=str(item.tool_call["id"]))

    out = mw.wrap_tool_call(
        ToolCallRequest(
            tool_call={
                "name": "call_deferred_tool",
                "args": {"name": "read_pdf", "arguments": {"path": "a.pdf"}},
                "id": "c1",
            },
            tool=None,  # type: ignore[arg-type]
            state={},  # type: ignore[arg-type]
            runtime=None,  # type: ignore[arg-type]
        ),
        handler,
    )
    assert seen["name"] == "read_pdf"
    assert seen["args"] == {"path": "a.pdf"}
    assert seen["tool"] is real
    assert isinstance(out, ToolMessage)
    assert out.name == "call_deferred_tool"
    assert out.tool_call_id == "c1"
    assert out.content == "ok"

    missing = mw.wrap_tool_call(
        ToolCallRequest(
            tool_call={"name": "call_deferred_tool", "args": {"name": "misc_0", "arguments": {}}, "id": "c2"},
            tool=None,  # type: ignore[arg-type]
            state={},  # type: ignore[arg-type]
            runtime=None,  # type: ignore[arg-type]
        ),
        handler,
    )
    assert isinstance(missing, ToolMessage)
    assert missing.status == "error"
    assert missing.content == "没有名为 misc_0 的本轮工具。"


def test_misrouted_core_tool_runs_instead_of_reporting_it_missing() -> None:
    reader = _tool("read_file", "读取文件")
    clock = _tool("current_time", "当前时间")
    tools: list[Any] = [reader, clock]
    tools.extend(_tool(f"misc_{index}", "unrelated") for index in range(6))
    tools.append(_tool("cronjob_list", "列出定时任务"))
    mw = DecisionMiddleware(
        config=_config(skills=False, team=False, tool_min_count=2),
        scorer=_Scorer({"tool:cronjob_list": 0.95, **{f"tool:misc_{index}": 0.1 for index in range(6)}}),
    )
    narrowed = mw.wrap_model_call(_request(tools=tools, text="今日热点早报"), lambda item: item)
    assert [tool.name for tool in narrowed.tools] == ["read_file", "current_time", "call_deferred_tool"]
    note = _note(narrowed)
    assert "cronjob_list" in note
    assert "read_file" not in note.split("请直接按原名调用。", 1)[-1]
    assert "函数列表里已有的工具" in note
    seen: dict[str, Any] = {}

    def handler(item: ToolCallRequest) -> ToolMessage:
        seen["name"] = item.tool_call["name"]
        seen["args"] = item.tool_call["args"]
        seen["tool"] = item.tool
        return ToolMessage(content="file", name=str(item.tool_call["name"]), tool_call_id=str(item.tool_call["id"]))

    before = decision_dispatch_snapshot()
    out = mw.wrap_tool_call(
        ToolCallRequest(
            tool_call={
                "name": "call_deferred_tool",
                "args": {"name": "read_file", "arguments": {"file_path": "SKILL.md"}},
                "id": "c1",
            },
            tool=None,  # type: ignore[arg-type]
            state={},  # type: ignore[arg-type]
            runtime=None,  # type: ignore[arg-type]
        ),
        handler,
    )
    assert seen == {"name": "read_file", "args": {"file_path": "SKILL.md"}, "tool": reader}
    assert isinstance(out, ToolMessage)
    assert out.name == "call_deferred_tool"
    assert out.tool_call_id == "c1"
    assert out.content == "file"
    assert out.status != "error"
    assert out.additional_kwargs["octop_forwarded_tool"] == "read_file"
    forwarded = decision_dispatch_snapshot()
    assert forwarded.core_forwards.get("read_file", 0) == before.core_forwards.get("read_file", 0) + 1

    clock_out = mw.wrap_tool_call(
        ToolCallRequest(
            tool_call={"name": "call_deferred_tool", "args": {"name": "current_time"}, "id": "c2"},
            tool=None,  # type: ignore[arg-type]
            state={},  # type: ignore[arg-type]
            runtime=None,  # type: ignore[arg-type]
        ),
        handler,
    )
    assert seen["name"] == "current_time"
    assert seen["tool"] is clock
    assert isinstance(clock_out, ToolMessage)
    assert clock_out.content == "file"
    assert clock_out.status != "error"
    assert clock_out.additional_kwargs["octop_forwarded_tool"] == "current_time"

    dropped = mw.wrap_tool_call(
        ToolCallRequest(
            tool_call={"name": "call_deferred_tool", "args": {"name": "misc_0", "arguments": {}}, "id": "c3"},
            tool=None,  # type: ignore[arg-type]
            state={},  # type: ignore[arg-type]
            runtime=None,  # type: ignore[arg-type]
        ),
        handler,
    )
    assert isinstance(dropped, ToolMessage)
    assert dropped.status == "error"
    assert dropped.content == "没有名为 misc_0 的本轮工具。"


def test_unknown_deferred_name_does_not_invent_a_direct_tool() -> None:
    mw = DecisionMiddleware(config=_config(skills=False, tools=False, team=False), scorer=_Scorer({}))
    before = decision_dispatch_snapshot()
    missing = mw.wrap_tool_call(
        ToolCallRequest(
            tool_call={"name": "call_deferred_tool", "args": {"name": "read_file", "arguments": {}}, "id": "c1"},
            tool=None,  # type: ignore[arg-type]
            state={},  # type: ignore[arg-type]
            runtime=None,  # type: ignore[arg-type]
        ),
        lambda _request: ToolMessage(content="unused", tool_call_id="c1"),
    )
    assert isinstance(missing, ToolMessage)
    assert missing.status == "error"
    assert missing.content == "没有名为 read_file 的本轮工具。"
    assert decision_dispatch_snapshot().unknown == before.unknown + 1

    english = DecisionMiddleware(
        config=_config(skills=False, tools=False, team=False),
        language="en",
        scorer=_Scorer({}),
    )
    english_missing = english.wrap_tool_call(
        ToolCallRequest(
            tool_call={"name": "call_deferred_tool", "args": {"name": "web_fetch"}, "id": "c2"},
            tool=None,  # type: ignore[arg-type]
            state={},  # type: ignore[arg-type]
            runtime=None,  # type: ignore[arg-type]
        ),
        lambda _request: ToolMessage(content="unused", tool_call_id="c2"),
    )
    assert isinstance(english_missing, ToolMessage)
    assert english_missing.content == "No turn tool named web_fetch."


def _structured_tool(name: str, description: str = "") -> StructuredTool:
    def run() -> str:
        return name

    return StructuredTool.from_function(run, name=name, description=description or name)


def _deferred_request(name: str, arguments: dict[str, Any] | None = None, *, call_id: str) -> ToolCallRequest:
    args: dict[str, Any] = {"name": name}
    if arguments is not None:
        args["arguments"] = arguments
    return ToolCallRequest(
        tool_call={"name": "call_deferred_tool", "args": args, "id": call_id},
        tool=None,  # type: ignore[arg-type]
        state={},  # type: ignore[arg-type]
        runtime=None,  # type: ignore[arg-type]
    )


def test_disabled_core_tool_is_not_offered_through_the_dispatcher() -> None:
    tools: list[Any] = [_structured_tool("read_file", "读取文件"), _structured_tool("current_time", "当前时间")]
    tools.extend(_structured_tool(f"misc_{index}", "unrelated") for index in range(6))
    tools.append(_structured_tool("cronjob_list", "列出定时任务"))
    gate = ToolsFilterMiddleware(config=HarnessAgentConfig(tools_disabled=frozenset({"read_file"})))
    decision = DecisionMiddleware(
        config=_config(skills=False, team=False, tool_min_count=2),
        scorer=_Scorer({"tool:cronjob_list": 0.95, **{f"tool:misc_{index}": 0.1 for index in range(6)}}),
    )

    def through_decision(filtered: ModelRequest) -> ModelRequest:
        return decision.wrap_model_call(filtered, lambda item: item)

    narrowed = gate.wrap_model_call(_request(tools=tools, text="今日热点早报"), through_decision)
    assert [tool.name for tool in narrowed.tools] == ["current_time", "call_deferred_tool"]
    before = decision_dispatch_snapshot()
    missing = decision.wrap_tool_call(
        _deferred_request("read_file", {}, call_id="c1"),
        lambda _item: ToolMessage(content="unused", tool_call_id="c1"),
    )
    assert isinstance(missing, ToolMessage)
    assert missing.status == "error"
    assert missing.content == "没有名为 read_file 的本轮工具。"
    assert decision_dispatch_snapshot().unknown == before.unknown + 1
    assert decision_dispatch_snapshot().core_forwards.get("read_file", 0) == before.core_forwards.get("read_file", 0)


async def test_async_misrouted_core_tool_runs() -> None:
    reader = _tool("read_file", "读取文件")
    tools: list[Any] = [reader, *(_tool(f"misc_{index}", "unrelated") for index in range(6)), _tool("cronjob_list")]
    mw = DecisionMiddleware(
        config=_config(skills=False, team=False, tool_min_count=2),
        scorer=_Scorer({"tool:cronjob_list": 0.95, **{f"tool:misc_{index}": 0.1 for index in range(6)}}),
    )
    await mw.awrap_model_call(_request(tools=tools, text="今日热点早报"), _async_echo)
    before = decision_dispatch_snapshot()

    async def handler(item: ToolCallRequest) -> ToolMessage:
        assert item.tool is reader
        return ToolMessage(content="file", name=str(item.tool_call["name"]), tool_call_id=str(item.tool_call["id"]))

    out = await mw.awrap_tool_call(_deferred_request("read_file", {"file_path": "SKILL.md"}, call_id="c1"), handler)
    assert isinstance(out, ToolMessage)
    assert out.content == "file"
    assert out.name == "call_deferred_tool"
    assert out.additional_kwargs["octop_forwarded_tool"] == "read_file"
    assert (
        decision_dispatch_snapshot().core_forwards.get("read_file", 0) == before.core_forwards.get("read_file", 0) + 1
    )


async def _async_echo(request: ModelRequest) -> ModelResponse:
    return ModelResponse(result=[AIMessage(content=str(len(request.tools or [])))])


def test_tool_search_still_guards_a_deferred_core_tool() -> None:
    reader = _structured_tool("read_file", "读取文件")
    tools: list[Any] = [reader, *(_structured_tool(f"misc_{index}", "unrelated") for index in range(6))]
    tools.append(_structured_tool("cronjob_list", "列出定时任务"))
    decision = DecisionMiddleware(
        config=_config(skills=False, team=False, tool_min_count=2),
        scorer=_Scorer({"tool:cronjob_list": 0.95, **{f"tool:misc_{index}": 0.1 for index in range(6)}}),
    )
    search = ToolSearchMiddleware(deferred_tools=frozenset({"read_file"}))
    seen: list[str] = []

    def tool_node(item: ToolCallRequest) -> ToolMessage:
        seen.append(str(item.tool_call["name"]))
        return ToolMessage(content="ran", name=str(item.tool_call["name"]), tool_call_id=str(item.tool_call["id"]))

    def through_search(item: ToolCallRequest) -> ToolMessage | Command[Any]:
        return search.wrap_tool_call(item, tool_node)

    runtime = type("Runtime", (), {"config": {"configurable": {"thread_id": "room"}}})()
    with _configurable(run_id="run-1", thread_id="room"):
        narrowed = decision.wrap_model_call(_request(tools=tools, text="今日热点早报"), lambda item: item)
        visible: list[Any] = []

        def capture(item: ModelRequest) -> ModelRequest:
            visible.extend(item.tools or [])
            return item

        search.wrap_model_call(narrowed, capture)
        blocked = decision.wrap_tool_call(
            ToolCallRequest(
                tool_call={
                    "name": "call_deferred_tool",
                    "args": {"name": "read_file", "arguments": {}},
                    "id": "c1",
                },
                tool=None,  # type: ignore[arg-type]
                state={},  # type: ignore[arg-type]
                runtime=runtime,  # type: ignore[arg-type]
            ),
            through_search,
        )
        loaded = decision.wrap_tool_call(
            ToolCallRequest(
                tool_call={
                    "name": "call_deferred_tool",
                    "args": {"name": "read_file", "arguments": {}},
                    "id": "c2",
                },
                tool=None,  # type: ignore[arg-type]
                state={"loaded_tool_names": ["read_file"]},  # type: ignore[arg-type]
                runtime=runtime,  # type: ignore[arg-type]
            ),
            through_search,
        )

    assert any(
        getattr(tool, "name", "") == "read_file" and "Deferred reference" in tool.description for tool in visible
    )
    assert seen == ["read_file"]
    assert isinstance(blocked, Command)
    blocked_message = blocked.update["messages"][0]
    assert isinstance(blocked_message, ToolMessage)
    assert blocked_message.name == "call_deferred_tool"
    assert blocked_message.additional_kwargs["octop_forwarded_tool"] == "read_file"
    assert "tool_schema_not_loaded" in blocked_message.content
    assert isinstance(loaded, ToolMessage)
    assert loaded.content == "ran"
    assert loaded.name == "call_deferred_tool"
    assert loaded.additional_kwargs["octop_forwarded_tool"] == "read_file"
