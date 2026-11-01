"""Narrow skills, tools, and team members with a decision model before the chat model speaks.

Mounted only when :func:`octop_harness.decision.decision_active` is true.
A low score, a failed request, or a user who already named a teammate leaves
that catalog as it was.

The system prompt and the core tool list stay byte-stable across turns so a
provider prefix cache keeps hitting. Skills and long-tail tools selected for
this turn are appended after the conversation. Long-tail tools are called
through ``call_deferred_tool``, whose schema does not change.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import AgentState
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from pydantic import BaseModel, Field

from octop_harness.context_usage import catalog_saved_tokens, pending_decision_saved
from octop_harness.decision.client import DecisionRequestError, HttpDecisionClient
from octop_harness.decision.config import DecisionConfig
from octop_harness.decision.select import Candidate, keep_confident, lexical_shortlist
from octop_harness.middleware.peer import has_agent_at_mention
from octop_harness.middleware.runtime import runtime_config
from octop_harness.teams.profile import peer_description
from octop_harness.teams.util import coerce_user_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionDispatchSnapshot:
    """Misroutes of core tools, and dispatcher calls that named nothing runnable."""

    core_forwards: dict[str, int]
    unknown: int


class _DispatchCounters:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._core: dict[str, int] = {}
        self._unknown = 0

    def core_forward(self, tool: str) -> None:
        with self._lock:
            self._core[tool] = self._core.get(tool, 0) + 1
        logger.info("decision forwarded core tool through call_deferred_tool: %s", tool)

    def unknown(self) -> None:
        with self._lock:
            self._unknown += 1

    def snapshot(self) -> DecisionDispatchSnapshot:
        with self._lock:
            return DecisionDispatchSnapshot(dict(self._core), self._unknown)


_DISPATCH = _DispatchCounters()


def decision_dispatch_snapshot() -> DecisionDispatchSnapshot:
    """Process-wide counts for dispatcher misroutes. Tool names stay low-cardinality."""
    return _DISPATCH.snapshot()


# Set while one model call is being scored, then copied onto that reply so
# history can show the same row the live stream already showed.
_PENDING_DECISION: ContextVar[dict[str, Any] | None] = ContextVar(
    "octop_pending_decision",
    default=None,
)

# Decision APIs take a short situation string, not the chat model window.
# Newest turns win when this budget is exceeded.
_STATE_LIMIT = 2400
_BLURB_LIMIT = 240
_QUESTION_LIMIT = 20
_CATALOG_LIMIT = 12
# Near misses stay. 0.6 still prefers a clear yes; 0.45 is kept while there is room.
_KEEP_SLACK = 0.15
# Filesystem, todos, memory, team dispatch, web lookup, and irreplaceable
# delivery/browse/screenshot tools. Decision may still drop the long tail
# (MCP/plugins, env file, media generation, etc.).
# Dropping the core set makes the chat model print a raw tool call and stop.
# ask_agent / agent_list are how a team host schedules; they stay even when
# every other tool is filtered out.
_TOOLS_ALWAYS_KEPT = frozenset(
    {
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "delete",
        "glob",
        "grep",
        "execute",
        "write_todos",
        "task",
        "current_time",
        "tool_search",
        "memory_search",
        "memory_get",
        "agent_list",
        "ask_agent",
        "web_fetch",
        "tavily_search",
        "brave_search",
        "google_search",
        "kimi_search",
        "deepseek_search",
        "yuanbao_search",
        "searchfree_search",
        "you_search",
        "send_file_to_user",
        "browser_use",
        "desktop_screenshot",
    }
)
# Shown on the host decision card. They are not scored: a low score must not
# take away the host's ability to dispatch.
_HOST_DISPATCH_TOOLS = ("ask_agent", "agent_list")
# Stable entry for long-tail tools. Its schema never changes, so it can sit in
# the cached tool prefix while the selected schemas move to the turn note.
_DEFERRED_TOOL_NAME = "call_deferred_tool"
# Matches no teammate. A wrap-up that clears nobody closes dispatch for the
# rest of that turn.
_DISPATCH_CLOSED = frozenset({"\x00"})
_AGENT_AT_TOKEN_RE = re.compile(r"@([^\W\d_][\w\-]*)(?![\w./\-~]*\.\w)")


def _call_with_savings(
    before: ModelRequest,
    after: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    token = pending_decision_saved.set(catalog_saved_tokens(before, after))
    try:
        return handler(after)
    finally:
        pending_decision_saved.reset(token)


async def _call_with_savings_async(
    before: ModelRequest,
    after: ModelRequest,
    handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
) -> ModelResponse:
    token = pending_decision_saved.set(catalog_saved_tokens(before, after))
    try:
        return await handler(after)
    finally:
        pending_decision_saved.reset(token)


def _stamp_decision(response: ModelResponse, payload: dict[str, Any] | None) -> ModelResponse:
    """Copy this turn's decision onto the reply so history can render it."""
    if not payload:
        return response
    result = getattr(response, "result", None)
    if not isinstance(result, list):
        return response
    stamped: list[Any] = []
    attached = False
    for message in result:
        if not attached and isinstance(message, AIMessage):
            extra = dict(message.additional_kwargs or {})
            extra["octop_decision"] = payload
            stamped.append(message.model_copy(update={"additional_kwargs": extra}))
            attached = True
        else:
            stamped.append(message)
    if not attached:
        return response
    return replace(response, result=stamped)


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


def _latest_user_text(request: ModelRequest) -> str:
    messages = getattr(request, "messages", None) or []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return _stringify_content(msg.content).strip()
    return ""


def _deferrable(tool: Any) -> bool:
    """Provider-native dict tools stay in the cached tool list and are not scored."""
    return not isinstance(tool, dict)


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            return str(fn["name"])
        if tool.get("name"):
            return str(tool["name"])
        return ""
    return str(getattr(tool, "name", "") or "")


def _tool_description(tool: Any) -> str:
    if isinstance(tool, dict):
        fn = tool.get("function")
        if isinstance(fn, dict):
            return str(fn.get("description") or "")
        return str(tool.get("description") or "")
    return str(getattr(tool, "description", "") or "")


def _clip(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _statement(kind: str, label: str, description: str, *, chinese: bool) -> str:
    blurb = _clip(description, _BLURB_LIMIT)
    detail = f" {blurb}" if blurb else ""
    if chinese:
        if kind == "skill":
            return f"这轮需要技能「{label}」。{detail}".strip()
        if kind == "tool":
            return f"这轮需要工具「{label}」。{detail}".strip()
        return f"这轮应该派给成员「{label}」。{detail}".strip()
    if kind == "skill":
        return f'The skill "{label}" is needed for this request.{detail}'
    if kind == "tool":
        return f'The tool "{label}" is needed for this request.{detail}'
    return f'The teammate "{label}" should work on this request.{detail}'


class _DeferredCallArgs(BaseModel):
    """Arguments for the stable long-tail tool entry."""

    name: str = Field(description="Tool name from the turn note.")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments for that tool.")


def _unused_deferred_call(name: str, arguments: dict[str, Any] | None = None) -> str:
    """Placeholder. ``wrap_tool_call`` forwards the call before this runs."""
    if arguments:
        return name
    return name


def _make_deferred_tool(chinese: bool) -> StructuredTool:
    description = (
        "调用本轮说明里列出的长尾工具。函数列表里已有的工具请按原名直接调用。"
        if chinese
        else (
            "Call a long-tail tool listed in the turn note. Call tools already in the function list by their own names."
        )
    )
    return StructuredTool.from_function(
        name=_DEFERRED_TOOL_NAME,
        description=description,
        func=_unused_deferred_call,
        args_schema=_DeferredCallArgs,
        infer_schema=False,
    )


@dataclass
class _TurnFilter:
    """Frozen catalog for one user message, including the tool loop after it.

    ``tools is None`` means this turn did not score tools. A set is the tool
    names that stay callable: core tools stay in ``request.tools``, and the
    long-tail names in the set are described in the turn note.
    ``rewrite_skills`` moves the skill catalog out of the system prompt.
    ``skills is None`` then keeps every skill; a set keeps those names.
    """

    tools: set[str] | None = None
    skills: set[str] | None = None
    rewrite_skills: bool = False
    hints: tuple[str, ...] = ()


@dataclass
class _Work:
    statements: list[tuple[str, str]] = field(default_factory=list)
    skill_ids: list[str] | None = None
    skill_rows: list[dict[str, Any]] | None = None
    tool_ids: list[str] | None = None
    tools_in: list[Any] | None = None
    member_ids: list[str] | None = None
    forced_member_ids: list[str] = field(default_factory=list)
    member_labels: dict[str, str] = field(default_factory=dict)
    member_keys: dict[str, frozenset[str]] = field(default_factory=dict)
    thread_key: str = ""
    turn_key: str = ""
    asked_skills: list[str] = field(default_factory=list)
    asked_tools: list[str] = field(default_factory=list)
    asked_members: list[str] = field(default_factory=list)
    result_check: bool = False


class DecisionMiddleware(AgentMiddleware[Any, Any]):
    """Ask one decision model which skills, tools, and teammates to keep."""

    def __init__(
        self,
        *,
        config: DecisionConfig,
        team: Any | None = None,
        language: str = "zh",
        scorer: Any | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._team = team
        self._chinese = language != "en"
        self._scorer = scorer or HttpDecisionClient(
            endpoint=config.endpoint,
            api_key=config.api_key,
            model=config.model,
            provider=config.provider,
        )
        self._pins: dict[str, frozenset[str]] = {}
        self._turns: dict[str, _TurnFilter] = {}
        self._deferred_by_thread: dict[str, dict[str, Any]] = {}
        # Not registered on the graph. The model only sees it on turns that
        # have long-tail tools; wrap_tool_call forwards the call itself.
        self._deferred_call_tool = _make_deferred_tool(self._chinese)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        token = _PENDING_DECISION.set(None)
        try:
            cached = self._turns.get(_turn_key(request))
            if cached is not None:
                narrowed = self._replay(request, cached)
            else:
                narrowed = self._finish(request, self._scorer.score)
            response = _call_with_savings(request, narrowed, handler)
            return _stamp_decision(response, _PENDING_DECISION.get())
        finally:
            _PENDING_DECISION.reset(token)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        token = _PENDING_DECISION.set(None)
        try:
            cached = self._turns.get(_turn_key(request))
            if cached is not None:
                narrowed = self._replay(request, cached)
            else:
                work = self._collect(request)
                if work is None:
                    narrowed = request
                else:
                    try:
                        probabilities = await self._scorer.score_async(
                            state=_state_text(request, chinese=self._chinese),
                            statements=work.statements,
                        )
                    except DecisionRequestError as exc:
                        narrowed = self._unavailable(request, work, exc)
                    else:
                        narrowed = self._apply(request, work, probabilities)
            response = await _call_with_savings_async(request, narrowed, handler)
            return _stamp_decision(response, _PENDING_DECISION.get())
        finally:
            _PENDING_DECISION.reset(token)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        forwarded = self._forward_deferred(request)
        if isinstance(forwarded, ToolMessage):
            return forwarded
        if isinstance(forwarded, ToolCallRequest):
            return self._finish_forwarded(request, forwarded, handler(forwarded))
        blocked = self._blocked_ask(request)
        if blocked is not None:
            return blocked
        result = handler(request)
        return self._filter_agent_list(request, result)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        forwarded = self._forward_deferred(request)
        if isinstance(forwarded, ToolMessage):
            return forwarded
        if isinstance(forwarded, ToolCallRequest):
            return self._finish_forwarded(request, forwarded, await handler(forwarded))
        blocked = self._blocked_ask(request)
        if blocked is not None:
            return blocked
        result = await handler(request)
        return self._filter_agent_list(request, result)

    def _finish(
        self,
        request: ModelRequest,
        score: Callable[..., dict[str, float]],
    ) -> ModelRequest:
        work = self._collect(request)
        if work is None:
            return request
        try:
            probabilities = score(
                state=_state_text(request, chinese=self._chinese),
                statements=work.statements,
            )
        except DecisionRequestError as exc:
            return self._unavailable(request, work, exc)
        return self._apply(request, work, probabilities)

    def _collect(self, request: ModelRequest) -> _Work | None:
        text = _latest_user_text(request)
        thread_key = _thread_key(request)
        self._pins.pop(thread_key, None)
        if not text:
            return None
        work = _Work(thread_key=thread_key, turn_key=_turn_key(request))
        cfg = self._config
        catalog_text = _followup_task(request) if _is_followup(request) else _user_history_text(request)
        if not catalog_text:
            catalog_text = _earlier_user_text(request) if _is_followup(request) else text
        if cfg.skills:
            self._collect_skills(request, catalog_text, work)
        if cfg.tools:
            self._collect_tools(request, catalog_text, work)
        if cfg.team:
            self._collect_members(request, text, work)
        if not work.statements:
            return None
        _fit_questions(work)
        if not work.statements:
            return None
        return work

    def _collect_skills(self, request: ModelRequest, text: str, work: _Work) -> None:
        raw = request.state.get("skills_metadata")
        rows = [row for row in raw if isinstance(row, dict)] if isinstance(raw, list) else []
        candidates = [
            Candidate(
                id=name,
                label=name,
                description=_clip(str(row.get("description") or ""), _BLURB_LIMIT),
            )
            for row in rows
            if (name := str(row.get("name") or "").strip())
        ]
        if len(candidates) < 2:
            return
        chosen = lexical_shortlist(candidates, text, limit=_CATALOG_LIMIT)
        if not chosen:
            return
        work.skill_rows = rows
        work.skill_ids = [item.id for item in chosen]
        for item in chosen:
            work.statements.append(
                ("skill:" + item.id, _statement("skill", item.label, item.description, chinese=self._chinese))
            )

    def _collect_tools(self, request: ModelRequest, text: str, work: _Work) -> None:
        tools_in = list(request.tools or [])
        named = [name for tool in tools_in if (name := _tool_name(tool))]
        if len(named) <= self._config.tool_min_count:
            return
        # Builtin tools stay in context. Spending a question on them only
        # crowds out the long tail the decision model can actually drop.
        candidates = [
            Candidate(id=name, label=name, description=_clip(_tool_description(tool), _BLURB_LIMIT))
            for tool in tools_in
            if (name := _tool_name(tool))
            and _deferrable(tool)
            and name not in _TOOLS_ALWAYS_KEPT
            and name != _DEFERRED_TOOL_NAME
        ]
        if not candidates:
            return
        chosen = lexical_shortlist(candidates, text, limit=_CATALOG_LIMIT)
        if not chosen:
            return
        work.tools_in = tools_in
        work.tool_ids = [item.id for item in chosen]
        for item in chosen:
            work.statements.append(
                ("tool:" + item.id, _statement("tool", item.label, item.description, chinese=self._chinese))
            )

    def _collect_members(self, request: ModelRequest, text: str, work: _Work) -> None:
        if self._team is None:
            return
        configurable = runtime_config(request).get("configurable") or {}
        if not self._async_dispatch(configurable):
            return
        user_id = coerce_user_id(configurable.get("user"))
        agent_id = str(configurable.get("agent_id") or "")
        if user_id is None or not agent_id:
            return
        # A member completion always gets a sufficiency check, even when the
        # roster has nobody left to dispatch. This is the host's final-result
        # evaluation and must not disappear merely because the team is small.
        followup = _is_followup(request)
        if followup:
            work.result_check = True
            statement = (
                "当前成员结果已经足以回答用户原始请求, 主持人现在可以总结, 不需要继续派工."
                if self._chinese
                else (
                    "The current member result is sufficient to answer the user's original request; "
                    "the host can summarize now without further dispatch."
                )
            )
            work.statements.append(("result:sufficient", statement))
        try:
            peers = self._team.list_peers(user_id, exclude_agent_id=agent_id)
        except Exception:
            logger.warning("decision member list failed", exc_info=True)
            return
        candidates: list[Candidate] = []
        labels: dict[str, str] = {}
        keys: dict[str, frozenset[str]] = {}
        for entry in peers:
            agent = str(getattr(entry, "agent_id", "") or "")
            if not agent:
                continue
            label = str(self._team.peer_display_name(entry) or agent)
            description = peer_description(getattr(entry, "metadata", None) or {}, "zh" if self._chinese else "en")
            candidates.append(Candidate(id=agent, label=label, description=_clip(description, _BLURB_LIMIT)))
            labels[agent] = label
            keys[agent] = frozenset(item.lower() for item in (agent, agent[-6:], label) if str(item).strip())
        # A wrap-up names whoever just finished. That is not the user picking
        # the next teammate, and that member is not a candidate to pin again.
        if followup:
            finished = _finished_member(request)
            if finished:
                candidates = [item for item in candidates if item.id != finished]
                labels.pop(finished, None)
                keys.pop(finished, None)
        naming = _earlier_user_text(request) if followup else text
        candidates, labels, keys, forced, stop = _restrict_named_members(
            naming,
            candidates,
            labels,
            keys,
            followup=followup,
        )
        if stop:
            return
        work.forced_member_ids = forced
        work.member_ids = [item.id for item in candidates]
        work.member_labels = labels
        work.member_keys = keys
        for item in candidates:
            work.statements.append(
                ("member:" + item.id, _statement("member", item.label, item.description, chinese=self._chinese))
            )

    def _apply(self, request: ModelRequest, work: _Work, probabilities: dict[str, float]) -> ModelRequest:
        if not probabilities:
            self._remember_turn(work.turn_key, _TurnFilter())
            self._publish(empty=True, parts=[])
            return request
        parts: list[dict[str, Any]] = []
        hints: list[str] = []
        skill_kept: set[str] | None = None
        tool_kept: set[str] | None = None
        rewrite_skills = False
        result_sufficient = False
        if work.skill_ids and work.skill_rows is not None:
            part, skill_kept = self._apply_skills(work, probabilities)
            rewrite_skills = True
            parts.append(part)
        if work.tool_ids and work.tools_in is not None:
            part, tool_kept = self._apply_tools(work, probabilities)
            parts.append(part)
        if work.result_check:
            part, result_sufficient, hint = self._apply_result(work, probabilities)
            parts.append(part)
            if hint:
                hints.append(hint)
        if work.member_ids and not result_sufficient:
            part, hint = self._apply_members(request, work, probabilities)
            parts.append(part)
            if hint:
                hints.append(hint)
        closed = result_sufficient or any(part.get("kind") == "member" and part.get("kept") == [] for part in parts)
        dispatch = None if closed else self._host_dispatch_part(request)
        if dispatch is not None and not any(part.get("kind") == "tool" for part in parts):
            member_at = next((index for index, part in enumerate(parts) if part.get("kind") == "member"), len(parts))
            parts.insert(member_at, dispatch)
        turn = _TurnFilter(
            tools=tool_kept,
            skills=skill_kept,
            rewrite_skills=rewrite_skills,
            hints=tuple(hints),
        )
        self._remember_turn(work.turn_key, turn)
        if parts:
            self._publish(empty=False, parts=parts)
        return self._present(request, turn)

    def _apply_result(
        self,
        work: _Work,
        probabilities: dict[str, float],
    ) -> tuple[dict[str, Any], bool, str | None]:
        score = probabilities.get("result:sufficient")
        if score is None:
            return {"kind": "result", "kept": None}, False, None
        sufficient = float(score) >= self._config.threshold
        name = (
            ("结果充分, 可以总结" if sufficient else "结果仍需补充")
            if self._chinese
            else ("Sufficient; ready to summarize" if sufficient else "More evidence needed")
        )
        part = {
            "kind": "result",
            "kept": [{"name": name, "score": round(float(score), 2)}],
        }
        if not sufficient:
            return part, False, None
        self._pins[work.thread_key] = _DISPATCH_CLOSED
        hint = (
            "决策模型认为成员结果已经足够. 请直接总结, 不要再调用 ask_agent."
            if self._chinese
            else (
                "The decision model found the member result sufficient. Summarize now and do not call ask_agent again."
            )
        )
        return part, True, hint

    def _remember_turn(self, key: str, turn: _TurnFilter) -> None:
        self._turns.pop(key, None)
        self._turns[key] = turn
        while len(self._turns) > 128:
            self._turns.pop(next(iter(self._turns)))

    def _replay(self, request: ModelRequest, turn: _TurnFilter) -> ModelRequest:
        return self._present(request, turn)

    def _present(self, request: ModelRequest, turn: _TurnFilter) -> ModelRequest:
        """Keep the cached prefix stable and append this turn's variable note."""
        sections: list[str] = []
        if turn.rewrite_skills:
            request, skill_text = _relocate_skills(request, turn.skills, chinese=self._chinese)
            if skill_text:
                sections.append(skill_text)
        if turn.tools is not None:
            request, tool_text = self._relocate_tools(request, turn.tools)
            if tool_text:
                sections.append(tool_text)
        sections.extend(hint for hint in turn.hints if hint)
        if not sections:
            return request
        title = "[本轮说明]\n" if self._chinese else "[turn note]\n"
        note = HumanMessage(content=title + "\n\n".join(sections))
        return request.override(messages=[*list(request.messages or []), note])

    def _relocate_tools(self, request: ModelRequest, kept: set[str]) -> tuple[ModelRequest, str]:
        incoming = list(request.tools or [])
        stable: list[Any] = []
        deferred: list[Any] = []
        direct: list[Any] = []
        has_long_tail = False
        for tool in incoming:
            name = _tool_name(tool)
            if name == _DEFERRED_TOOL_NAME:
                continue
            # Provider-executed tools stay in the tool list. Their schemas are
            # not callable through ``call_deferred_tool``, and the set does not
            # depend on this turn's scores.
            if not name or isinstance(tool, dict) or name in _TOOLS_ALWAYS_KEPT:
                stable.append(tool)
                # Core tools stay directly callable. Models still route them
                # through the dispatcher after reading the turn note; keep the
                # objects so that misroute runs the tool instead of erroring.
                if name in _TOOLS_ALWAYS_KEPT and not isinstance(tool, dict):
                    direct.append(tool)
                continue
            has_long_tail = True
            if name in kept:
                deferred.append(tool)
        saw_dispatcher = any(_tool_name(tool) == _DEFERRED_TOOL_NAME for tool in incoming)
        if has_long_tail:
            stable.append(self._deferred_call_tool)
        if has_long_tail or saw_dispatcher:
            request = request.override(tools=stable)
        self._remember_deferred(request, [*direct, *deferred] if has_long_tail else [])
        return request, _deferred_tool_note(deferred, chinese=self._chinese)

    def _remember_deferred(self, request: ModelRequest, tools: list[Any]) -> None:
        key = _execution_key(request)
        catalog = {name: tool for tool in tools if (name := _tool_name(tool))}
        self._deferred_by_thread.pop(key, None)
        self._deferred_by_thread[key] = catalog
        while len(self._deferred_by_thread) > 128:
            self._deferred_by_thread.pop(next(iter(self._deferred_by_thread)))

    def _deferred_catalog(self, request: ToolCallRequest) -> dict[str, Any]:
        key = _execution_key(cast(ModelRequest, request))
        return dict(self._deferred_by_thread.get(key, {}))

    def _forward_deferred(self, request: ToolCallRequest) -> ToolCallRequest | ToolMessage | None:
        name = str(request.tool_call.get("name") or "")
        if name != _DEFERRED_TOOL_NAME:
            return None
        args = request.tool_call.get("args") or {}
        target = str(args.get("name") or args.get("tool") or "").strip()
        payload = args.get("arguments", args.get("args"))
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                parsed = None
            payload = parsed
        if not isinstance(payload, dict):
            payload = {}
        call_id = str(request.tool_call.get("id") or "")
        tool = self._deferred_catalog(request).get(target)
        if tool is None or not target:
            _DISPATCH.unknown()
            text = _missing_deferred_tool(target, chinese=self._chinese)
            return ToolMessage(content=text, name=_DEFERRED_TOOL_NAME, tool_call_id=call_id, status="error")
        forwarded = cast(ToolCall, {**request.tool_call, "name": target, "args": payload})
        return request.override(tool_call=forwarded, tool=cast(BaseTool, tool))

    def _finish_forwarded(
        self,
        request: ToolCallRequest,
        forwarded: ToolCallRequest,
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
        target = str(forwarded.tool_call.get("name") or "")
        if target in _TOOLS_ALWAYS_KEPT:
            _DISPATCH.core_forward(target)
        call_id = str(request.tool_call.get("id") or "")
        return _retarget_tool_result(result, call_id, forwarded_name=target)

    def _apply_skills(self, work: _Work, probabilities: dict[str, float]) -> tuple[dict[str, Any], set[str] | None]:
        assert work.skill_ids is not None and work.skill_rows is not None
        scores = _prefixed(probabilities, "skill")
        asked, unasked = _split_asked(work.skill_ids, work.asked_skills)
        chosen = keep_confident(
            scores,
            asked,
            threshold=self._config.threshold,
            limit=self._config.max_selected,
            slack=_KEEP_SLACK,
        )
        if chosen is None:
            return self._part("skill", None, scores), None
        return self._part("skill", chosen, scores), set(chosen) | set(unasked)

    def _apply_tools(self, work: _Work, probabilities: dict[str, float]) -> tuple[dict[str, Any], set[str]]:
        assert work.tool_ids is not None and work.tools_in is not None
        scores = _prefixed(probabilities, "tool")
        asked, unasked = _split_asked(work.tool_ids, work.asked_tools)
        chosen = keep_confident(
            scores,
            asked,
            threshold=self._config.threshold,
            limit=self._config.max_selected,
            slack=_KEEP_SLACK,
        )
        if chosen is None:
            # A real "no" on every asked tool drops the long tail. Missing
            # scores keep the catalog — a partial reply is not a verdict.
            # Either way the tool list stays the stable core set; kept
            # long-tail schemas move to the turn note.
            if _fully_scored(scores, asked):
                kept = set(unasked) | _TOOLS_ALWAYS_KEPT
                return self._part("tool", [], scores), kept
            names = {name for tool in work.tools_in if (name := _tool_name(tool)) and name != _DEFERRED_TOOL_NAME}
            return self._part("tool", None, scores), names
        kept = set(chosen) | set(unasked) | _TOOLS_ALWAYS_KEPT
        return self._part("tool", chosen, scores), kept

    def _apply_members(
        self, request: ModelRequest, work: _Work, probabilities: dict[str, float]
    ) -> tuple[dict[str, Any], str | None]:
        assert work.member_ids is not None
        scores = _prefixed(probabilities, "member")
        asked, unasked = _split_asked(work.member_ids, work.asked_members)
        if work.forced_member_ids:
            # Explicit user routing always wins. The decision model still
            # evaluates and displays that choice, but cannot silently reroute.
            return self._pin_members(work, list(work.forced_member_ids), scores)
        if unasked or any(item_id not in scores for item_id in asked):
            return self._part("member", None, scores, work.member_labels), None
        chosen = keep_confident(
            scores,
            asked,
            threshold=self._config.threshold,
            limit=self._config.max_selected,
            slack=_KEEP_SLACK,
        )
        if chosen is None:
            # Wrap-up is an evaluation: a full miss means stop, not "keep everyone".
            if _is_followup(request) and _fully_scored(scores, asked):
                self._pins[work.thread_key] = _DISPATCH_CLOSED
                if self._chinese:
                    hint = "决策模型评估成员结果后认为不必再派工. 请直接收口, 不要再调用 ask_agent."
                else:
                    hint = (
                        "The decision model reviewed the member result and found no further dispatch. "
                        "Wrap up now and do not call ask_agent."
                    )
                return self._part("member", [], scores, work.member_labels), hint
            return self._part("member", None, scores, work.member_labels), None
        return self._pin_members(work, chosen, scores)

    def _pin_members(
        self,
        work: _Work,
        chosen: list[str],
        scores: dict[str, float],
    ) -> tuple[dict[str, Any], str]:
        keys: set[str] = set()
        labels: list[str] = []
        for member_id in chosen:
            keys.update(work.member_keys.get(member_id, frozenset()))
            labels.append(work.member_labels.get(member_id, member_id))
        self._pins[work.thread_key] = frozenset(keys)
        if self._chinese:
            hint = "这轮只派给这些成员, 可以同时派给多位: " + "、".join(labels)
        else:
            hint = "Dispatch this turn only to these members; several may run at once: " + ", ".join(labels)
        return self._part("member", chosen, scores, work.member_labels), hint

    def _model_label(self) -> str:
        return self._config.model or self._config.provider

    def _unavailable(self, request: ModelRequest, work: _Work, error: DecisionRequestError) -> ModelRequest:
        """Keep the catalog when the decision service cannot be reached."""
        self._remember_turn(work.turn_key, _TurnFilter())
        self._publish(empty=True, parts=[], error="request_failed", detail=error.detail)
        return request

    def _publish(
        self,
        *,
        empty: bool,
        parts: list[dict[str, Any]],
        error: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Hand one decision result to the chat stream. No log line."""
        payload = {
            "kind": "decision",
            "model": self._model_label(),
            "empty": empty,
            "parts": parts,
        }
        if error:
            payload["error"] = error
        if detail:
            payload["detail"] = detail
        _PENDING_DECISION.set(payload)
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
        except Exception:
            return
        try:
            writer(payload)
        except Exception:
            return

    def _part(
        self,
        kind: str,
        chosen: list[str] | None,
        scores: dict[str, float],
        labels: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if chosen is None:
            return {"kind": kind, "kept": None}
        items: list[dict[str, Any]] = []
        for item_id in chosen:
            name = labels.get(item_id, item_id) if labels else item_id
            item: dict[str, Any] = {"name": name}
            score = scores.get(item_id)
            if score is not None:
                item["score"] = round(float(score), 2)
            items.append(item)
        return {"kind": kind, "kept": items}

    def _host_dispatch_part(self, request: ModelRequest) -> dict[str, Any] | None:
        """Name the tools a host uses to schedule, without scoring them away."""
        if not self._config.tools:
            return None
        configurable = runtime_config(request).get("configurable") or {}
        if not isinstance(configurable, dict) or not self._async_dispatch(configurable):
            return None
        present = {_tool_name(tool) for tool in list(request.tools or [])}
        kept = [name for name in _HOST_DISPATCH_TOOLS if name in present]
        if "ask_agent" not in kept:
            return None
        return {"kind": "tool", "kept": [{"name": name} for name in kept]}

    def _async_dispatch(self, configurable: dict[str, Any]) -> bool:
        """Team hosts dispatch in the background. One-to-one chats do not.

        A request-scoped ``sync`` always wins, so a member answering inside
        a room cannot pin its own teammates. When the request omits the mode,
        the agent config is the host's async dispatch.
        """
        mode = str(configurable.get("peer_invoke_mode") or "")
        if mode == "async":
            return True
        if mode:
            return False
        agent_id = str(configurable.get("agent_id") or "")
        reader = getattr(self._team, "peer_invoke_mode_for", None)
        if self._team is None or not agent_id or not callable(reader):
            return False
        mode = reader(agent_id)
        return isinstance(mode, str) and mode == "async"

    def _pin(self) -> frozenset[str] | None:
        try:
            from langgraph.config import get_config

            cfg = get_config().get("configurable") or {}
        except Exception:
            return None
        thread_key = str(cfg.get("thread_id") or cfg.get("agent_id") or "")
        return self._pins.get(thread_key)

    def _blocked_ask(self, request: ToolCallRequest) -> ToolMessage | None:
        pin = self._pin()
        if not pin:
            return None
        tool_call = request.tool_call
        if str(tool_call.get("name") or "") != "ask_agent":
            return None
        if pin == _DISPATCH_CLOSED:
            if self._chinese:
                content = "决策模型评估后这轮不再派工. 请直接收口, 不要改派."
            else:
                content = "The decision model closed dispatch for this turn. Wrap up and do not dispatch again."
            return ToolMessage(
                content=content,
                tool_call_id=str(tool_call.get("id") or ""),
                status="error",
            )
        expert = str((tool_call.get("args") or {}).get("expert") or "").strip().lstrip("@").lower()
        if expert and expert in pin:
            return None
        if self._chinese:
            content = "这轮没有把该成员选进派工名单. 请改派已选中的成员, 或先调用 agent_list."
        else:
            content = "That member was not selected for this turn. Dispatch a selected member, or call agent_list."
        return ToolMessage(
            content=content,
            tool_call_id=str(tool_call.get("id") or ""),
            status="error",
        )

    def _filter_agent_list(
        self, request: ToolCallRequest, result: ToolMessage | Command[Any]
    ) -> ToolMessage | Command[Any]:
        pin = self._pin()
        if not pin or not isinstance(result, ToolMessage):
            return result
        if str(request.tool_call.get("name") or "") != "agent_list":
            return result
        content = result.content
        if not isinstance(content, str):
            return result
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return result
        if not isinstance(data, list):
            return result
        kept = []
        for item in data:
            if not isinstance(item, dict):
                continue
            names = {
                str(item.get("agent_id") or "").lower(),
                str(item.get("short_id") or "").lower(),
                str(item.get("name") or "").lower(),
            }
            if names & pin:
                kept.append(item)
        return ToolMessage(
            content=json.dumps(kept, ensure_ascii=False, indent=2),
            tool_call_id=result.tool_call_id,
            status=result.status,
        )


def _allocate(lengths: list[int], limit: int) -> list[int]:
    """Split ``limit`` question slots across groups, then give leftovers back."""
    count = len(lengths)
    if count == 0 or limit <= 0:
        return [0] * count
    share, extra = divmod(limit, count)
    alloc = [min(length, share + (1 if index < extra else 0)) for index, length in enumerate(lengths)]
    leftover = limit - sum(alloc)
    while leftover > 0:
        progressed = False
        for index, length in enumerate(lengths):
            if alloc[index] >= length or leftover <= 0:
                continue
            alloc[index] += 1
            leftover -= 1
            progressed = True
        if not progressed:
            break
    return alloc


def _fit_questions(work: _Work) -> None:
    """Keep each catalog inside the question budget.

    Members are omitted when the roster cannot be asked in full. A partial
    pin would block teammates the model never scored. Skills and tools that
    do not fit stay in the catalog; only scored misses are dropped.
    """
    grouped = {
        "skill": [item for item in work.statements if item[0].startswith("skill:")],
        "tool": [item for item in work.statements if item[0].startswith("tool:")],
        "result": [item for item in work.statements if item[0].startswith("result:")],
        "member": [item for item in work.statements if item[0].startswith("member:")],
    }
    order = [kind for kind in ("result", "skill", "tool", "member") if grouped[kind]]
    if not order:
        return
    alloc = dict(zip(order, _allocate([len(grouped[kind]) for kind in order], _QUESTION_LIMIT), strict=True))
    if "member" in alloc and alloc["member"] < len(grouped["member"]):
        order = [kind for kind in order if kind != "member"]
        grouped["member"] = []
        work.member_ids = None
        if not order:
            work.statements = []
            return
        alloc = dict(zip(order, _allocate([len(grouped[kind]) for kind in order], _QUESTION_LIMIT), strict=True))
    kept: list[tuple[str, str]] = []
    asked: dict[str, list[str]] = {"result": [], "skill": [], "tool": [], "member": []}
    for kind in ("result", "skill", "tool", "member"):
        chosen = grouped[kind][: alloc.get(kind, 0)]
        kept.extend(chosen)
        prefix = kind + ":"
        asked[kind] = [item_id[len(prefix) :] for item_id, _ in chosen]
    work.statements = kept
    work.asked_skills = asked["skill"]
    work.asked_tools = asked["tool"]
    work.asked_members = asked["member"]


def _split_asked(ids: list[str], asked: list[str]) -> tuple[list[str], list[str]]:
    """Ids the model was asked about, and the shortlist entries that did not fit."""
    if not asked:
        return list(ids), []
    asked_set = set(asked)
    return (
        [item_id for item_id in ids if item_id in asked_set],
        [item_id for item_id in ids if item_id not in asked_set],
    )


def _configurable_of(request: ModelRequest) -> dict[str, Any]:
    raw = runtime_config(request).get("configurable") or {}
    return raw if isinstance(raw, dict) else {}


def _is_followup(request: ModelRequest) -> bool:
    """Host wrap-up after a member finishes. Not a new user question."""
    return bool(_configurable_of(request).get("octop_decision_followup"))


def _finished_member(request: ModelRequest) -> str:
    return str(_configurable_of(request).get("octop_decision_finished_member") or "").strip()


def _followup_task(request: ModelRequest) -> str:
    return str(_configurable_of(request).get("octop_decision_task") or "").strip()


def _earlier_user_text(request: ModelRequest) -> str:
    """User text before the latest human message (the wrap-up wakeup)."""
    messages = getattr(request, "messages", None) or []
    seen_latest = False
    for msg in reversed(messages):
        if not isinstance(msg, HumanMessage):
            continue
        if not seen_latest:
            seen_latest = True
            continue
        return _stringify_content(msg.content).strip()
    return ""


def _message_line(msg: Any, *, chinese: bool) -> str:
    text = _stringify_content(getattr(msg, "content", "")).strip()
    if not text:
        return ""
    if isinstance(msg, HumanMessage):
        prefix = "用户: " if chinese else "User: "
        return prefix + text
    if isinstance(msg, AIMessage):
        prefix = "助手: " if chinese else "Assistant: "
        return prefix + text
    return ""


def _fit_lines(lines: list[str], limit: int, *, head: str = "") -> str:
    """Keep ``head`` and the newest line. Older lines fill what is left.

    A line that does not fit is clipped from the start, then anything older
    is dropped. The chat model still sees the full thread; this is only the
    decision ``state`` budget.
    """
    if limit <= 0:
        return ""
    head = _clip(head, min(len(head), max(64, limit // 4))) if head else ""
    body_limit = limit - len(head) - (1 if head else 0)
    if not lines:
        return head
    latest = _clip(lines[-1], body_limit)
    budget = body_limit - len(latest)
    older: list[str] = []
    if budget > 1:
        budget -= 1
        for line in reversed(lines[:-1]):
            if budget <= 0:
                break
            sep = 1 if older else 0
            room = budget - sep
            if room <= 0:
                break
            piece = line if len(line) <= room else _clip(line, room)
            older.append(piece)
            budget -= len(piece) + sep
            if len(line) > room:
                break
        older.reverse()
    parts = [part for part in (head, *older, latest) if part]
    return "\n".join(parts)


def _user_texts(request: ModelRequest) -> list[str]:
    texts: list[str] = []
    for msg in getattr(request, "messages", None) or []:
        if not isinstance(msg, HumanMessage):
            continue
        text = _stringify_content(msg.content).strip()
        if text:
            texts.append(text)
    return texts


def _conversation_lines(request: ModelRequest, *, chinese: bool) -> list[str]:
    lines: list[str] = []
    for msg in getattr(request, "messages", None) or []:
        if isinstance(msg, ToolMessage):
            continue
        line = _message_line(msg, chinese=chinese)
        if line:
            lines.append(line)
    return lines


def _user_history_text(request: ModelRequest) -> str:
    """User turns in this thread. The latest ask stays when the budget is tight."""
    return _fit_lines(_user_texts(request), _STATE_LIMIT)


def _state_text(request: ModelRequest, *, chinese: bool = True) -> str:
    lines = _conversation_lines(request, chinese=chinese)
    head = ""
    if _is_followup(request):
        earlier = _followup_task(request) or _earlier_user_text(request)
        if earlier and not any(earlier in line for line in lines):
            head = earlier
    return _fit_lines(lines, _STATE_LIMIT, head=head) or _clip(_latest_user_text(request), _STATE_LIMIT)


def _thread_key(request: ModelRequest) -> str:
    configurable = runtime_config(request).get("configurable") or {}
    return str(configurable.get("thread_id") or configurable.get("agent_id") or "")


def _execution_key(request: ModelRequest) -> str:
    """Keep one run's deferred tools when another run shares the thread."""
    config = runtime_config(request)
    run_id = str(config.get("run_id") or "")
    thread = _thread_key(request)
    return f"{thread}\n{run_id}" if run_id else thread


_SKILL_ENTRY_RE = re.compile(r"^- \*\*([^*]+)\*\*:")
# Same bytes on every turn. The selected bullets move to the turn note, so the
# system prompt no longer changes when the decision model picks a different set.
_STABLE_SKILLS_ZH = "**Available Skills:**\n\n这一轮要使用的技能列在对话末尾的本轮说明里。\n\n"
_STABLE_SKILLS_EN = (
    "**Available Skills:**\n\nSkills for this turn are listed in the turn note at the end of the conversation.\n\n"
)


def _skill_region(text: str) -> tuple[int, int] | None:
    start = text.find("**Available Skills:**")
    if start < 0:
        return None
    end = text.find("**How to Use Skills", start)
    if end < 0:
        return None
    return start, end


def _selected_skill_entries(region: str, kept: set[str] | None) -> str:
    """Skill bullets inside ``**Available Skills:**``. ``kept is None`` keeps all."""
    lines = region.split("\n")
    blocks: list[str] = []
    entry: list[str] = []
    name: str | None = None

    def flush() -> None:
        nonlocal entry, name
        if entry and name and (kept is None or name in kept):
            blocks.append("\n".join(entry).strip("\n"))
        entry = []
        name = None

    for line in lines[1:]:
        match = _SKILL_ENTRY_RE.match(line)
        if match:
            flush()
            name = match.group(1).strip()
            entry = [line]
        else:
            entry.append(line)
    flush()
    return "\n".join(blocks).strip()


def _rewrite_skill_text(text: str, kept: set[str] | None, stub: str) -> tuple[str, str]:
    region = _skill_region(text)
    if region is None:
        return text, ""
    start, end = region
    note = _selected_skill_entries(text[start:end], kept)
    return text[:start] + stub + text[end:], note


def _rewrite_skill_content(content: Any, kept: set[str] | None, stub: str) -> tuple[Any, str]:
    if isinstance(content, str):
        return _rewrite_skill_text(content, kept, stub)
    if not isinstance(content, list):
        return content, ""
    notes: list[str] = []
    blocks: list[Any] = []
    changed = False
    for block in content:
        if isinstance(block, str):
            updated, note = _rewrite_skill_text(block, kept, stub)
            changed = changed or updated != block
            blocks.append(updated)
        elif isinstance(block, dict) and block.get("type") == "text":
            text = str(block.get("text") or "")
            updated, note = _rewrite_skill_text(text, kept, stub)
            changed = changed or updated != text
            blocks.append({**block, "text": updated})
        else:
            note = ""
            blocks.append(block)
        if note:
            notes.append(note)
    return (blocks if changed else content), "\n".join(notes)


def _metadata_skill_lines(request: ModelRequest, kept: set[str] | None) -> str:
    raw = request.state.get("skills_metadata")
    if not isinstance(raw, list):
        return ""
    lines: list[str] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name or (kept is not None and name not in kept):
            continue
        description = str(row.get("description") or "").strip()
        lines.append(f"- **{name}**: {description}" if description else f"- **{name}**")
    return "\n".join(lines)


def _narrow_skill_state(request: ModelRequest, kept: set[str]) -> ModelRequest:
    raw = request.state.get("skills_metadata")
    if not isinstance(raw, list):
        return request
    narrowed = [
        row
        for row in raw
        if not isinstance(row, dict) or not str(row.get("name") or "").strip() or str(row.get("name")) in kept
    ]
    if narrowed == raw:
        return request
    state = cast(AgentState[Any], {**request.state, "skills_metadata": narrowed})
    return request.override(state=state)


def _relocate_skills(
    request: ModelRequest,
    kept: set[str] | None,
    *,
    chinese: bool,
) -> tuple[ModelRequest, str]:
    """Replace the skill catalog with a stable stub and return the selected bullets."""
    system = request.system_message
    stub = _STABLE_SKILLS_ZH if chinese else _STABLE_SKILLS_EN
    note = ""
    if system is not None:
        updated, note = _rewrite_skill_content(system.content, kept, stub)
        if updated != system.content:
            request = request.override(system_message=system.model_copy(update={"content": updated}))
    if not note:
        note = _metadata_skill_lines(request, kept)
    if kept is not None:
        request = _narrow_skill_state(request, kept)
    if not note.strip():
        return request, ""
    title = "技能:" if chinese else "Skills:"
    return request, f"{title}\n{note.strip()}"


def _tool_args_schema(tool: Any) -> str:
    schema = getattr(tool, "args_schema", None)
    model_json_schema = getattr(schema, "model_json_schema", None)
    if callable(model_json_schema):
        try:
            raw = model_json_schema()
        except (TypeError, ValueError):
            return ""
        if isinstance(raw, dict):
            return json.dumps(raw, ensure_ascii=False, sort_keys=True)
    if isinstance(tool, dict):
        return json.dumps(tool, ensure_ascii=False, sort_keys=True)
    return ""


def _deferred_tool_note(tools: list[Any], *, chinese: bool) -> str:
    if not tools:
        return ""
    lines: list[str] = []
    for tool in tools:
        name = _tool_name(tool)
        description = _tool_description(tool).strip()
        line = f"- {name}: {description}" if description else f"- {name}"
        schema = _tool_args_schema(tool)
        if schema:
            line += "\n  " + schema
        lines.append(line)
    header = (
        "工具: 下面列出的才用 call_deferred_tool 调用, name 填工具名, arguments 填参数。"
        "函数列表里已有的工具, 例如 read_file、current_time、web_fetch, 请直接按原名调用。"
        if chinese
        else (
            "Tools: call call_deferred_tool only for the tools listed below, with name and arguments. "
            "Tools already in the function list, such as read_file, current_time, and web_fetch, "
            "are called by their own names."
        )
    )
    return header + "\n" + "\n".join(lines)


def _missing_deferred_tool(target: str, *, chinese: bool) -> str:
    """Name a dispatcher miss without claiming a tool the model cannot see."""
    missing = target or "(empty)"
    if chinese:
        return f"没有名为 {missing} 的本轮工具。"
    return f"No turn tool named {missing}."


def _retarget_tool_result(
    result: ToolMessage | Command[Any],
    call_id: str,
    *,
    forwarded_name: str,
) -> ToolMessage | Command[Any]:
    """Keep the tool message paired with the model's ``call_deferred_tool`` call."""
    if isinstance(result, ToolMessage):
        extra = dict(result.additional_kwargs or {})
        if forwarded_name:
            extra["octop_forwarded_tool"] = forwarded_name
        return result.model_copy(
            update={
                "name": _DEFERRED_TOOL_NAME,
                "tool_call_id": call_id or result.tool_call_id,
                "additional_kwargs": extra,
            }
        )
    if isinstance(result, Command) and isinstance(result.update, dict):
        messages = result.update.get("messages")
        if isinstance(messages, list):
            rewritten = dict(result.update)
            rewritten["messages"] = [
                _retarget_tool_result(item, call_id, forwarded_name=forwarded_name)
                if isinstance(item, ToolMessage)
                else item
                for item in messages
            ]
            return replace(result, update=rewritten)
    return result


def _turn_key(request: ModelRequest) -> str:
    """One user message, stable across the tool loop that follows it."""
    thread = _thread_key(request)
    messages = getattr(request, "messages", None) or []
    for index in range(len(messages) - 1, -1, -1):
        msg = messages[index]
        if isinstance(msg, HumanMessage):
            ident = str(getattr(msg, "id", "") or "") or _clip(_stringify_content(msg.content), 80)
            # The graph may omit message ids. The position is stable while
            # tool/assistant messages are appended, and distinguishes a later
            # user who sends the same text again.
            return f"{thread}\n{index}\n{ident}"
    return thread


def _fully_scored(scores: dict[str, float], ids: list[str]) -> bool:
    return bool(ids) and all(item_id in scores for item_id in ids)


def _prefixed(probabilities: dict[str, float], kind: str) -> dict[str, float]:
    prefix = kind + ":"
    return {key[len(prefix) :]: value for key, value in probabilities.items() if key.startswith(prefix)}


def _named_member_ids(text: str, candidates: list[Candidate]) -> list[str]:
    mentions = [match.group(1).lower() for match in _AGENT_AT_TOKEN_RE.finditer(text)]
    if not mentions:
        return []
    out: list[str] = []
    for item in candidates:
        keys = [key.lower() for key in (item.label.strip(), item.id.strip(), item.id[-6:].strip()) if len(key) >= 2]
        if any(mention in key or key in mention for mention in mentions for key in keys):
            out.append(item.id)
    return out


def _restrict_named_members(
    text: str,
    candidates: list[Candidate],
    labels: dict[str, str],
    keys: dict[str, frozenset[str]],
    *,
    followup: bool,
) -> tuple[
    list[Candidate],
    dict[str, str],
    dict[str, frozenset[str]],
    list[str],
    bool,
]:
    forced = _named_member_ids(text, candidates) if text else []
    if forced:
        forced_set = set(forced)
        chosen = [item for item in candidates if item.id in forced_set]
        chosen_labels = {item_id: label for item_id, label in labels.items() if item_id in forced_set}
        chosen_keys = {item_id: value for item_id, value in keys.items() if item_id in forced_set}
        return chosen, chosen_labels, chosen_keys, [item.id for item in chosen], False
    # An unknown/hidden @mention is still an explicit user choice. Do not let
    # the decision model redirect it to a visible peer.
    unknown_mention = not followup and has_agent_at_mention(text)
    too_few = len(candidates) < (1 if followup else 2)
    return candidates, labels, keys, [], unknown_mention or too_few


__all__ = ["DecisionMiddleware"]
