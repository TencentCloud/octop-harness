"""Graph that forwards a chat turn to a configured ACP runner.

The compiled graph is a stand-in for ``create_deep_agent``. Text fragments from
the ACP session are yielded as chat-model chunks so the existing messages
stream becomes ``token`` events. A permission pause uses the same interrupt
payload shape as tool approval.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from pydantic import ConfigDict, PrivateAttr
from typing_extensions import TypedDict

from octop_harness.acp.models import ACPConfig
from octop_harness.acp.service import get_acp_service, init_acp_service


class ManagedState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    needs_permission: bool
    resume_option: str


def managed_session_mode(config: RunnableConfig) -> str:
    """ACP ``modeId`` chosen for this turn. Empty keeps the agent's current mode."""
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return ""
    raw = configurable.get("acp_session_mode")
    return raw.strip() if isinstance(raw, str) else ""


def managed_permission_mode(config: RunnableConfig) -> str:
    """``yolo`` skips ACP approval cards for this turn."""
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return "ask"
    return "yolo" if configurable.get("acp_permission_mode") == "yolo" else "ask"


def managed_turn_mcp(config: RunnableConfig) -> dict[str, Any] | None:
    """Harness MCP specs the host attached for this managed turn."""
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None
    raw = configurable.get("acp_mcp_servers")
    return raw if isinstance(raw, dict) else None


def _locale(config: RunnableConfig) -> str:
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return ""
    raw = configurable.get("locale")
    return raw if isinstance(raw, str) else ""


def managed_prompt_note(config: RunnableConfig) -> str:
    """Extra instruction the host wants appended to the ACP prompt."""
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return ""
    raw = configurable.get("acp_prompt_note")
    return raw.strip() if isinstance(raw, str) else ""


def append_prompt_note(text: str, note: str) -> str:
    extra = note.strip()
    if text and extra:
        return f"{text}\n\n{extra}"
    return text


def _thread_id(config: RunnableConfig) -> str:
    configurable = config.get("configurable")
    if isinstance(configurable, dict):
        raw = configurable.get("thread_id")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return "managed"


def _tool_finished(payload: dict[str, Any]) -> bool:
    return str(payload.get("status") or "") in {"completed", "failed"}


def _tool_chunks(
    payload: dict[str, Any],
    opened: dict[str, int],
    tool_index: int,
) -> tuple[list[ChatGenerationChunk], int]:
    """Open a tool card once, then close it when ACP reports the call finished."""
    call_id = str(payload.get("call_id") or "")
    chunks: list[ChatGenerationChunk] = []
    if call_id not in opened:
        opened[call_id] = tool_index
        tool_index += 1
        chunks.append(
            ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": str(payload.get("name") or "tool"),
                            "args": "{}",
                            "id": call_id,
                            "index": opened[call_id],
                        }
                    ],
                )
            )
        )
    if _tool_finished(payload):
        opened.pop(call_id, None)
        chunks.append(_tool_result_chunk(payload))
    return chunks, tool_index


def _tool_result_chunk(payload: dict[str, Any]) -> ChatGenerationChunk:
    failed = str(payload.get("status") or "") == "failed"
    return ChatGenerationChunk(
        message=AIMessageChunk(
            content="",
            additional_kwargs={
                "octop_tool_result": {
                    "tool_call_id": str(payload.get("call_id") or ""),
                    "content": str(payload.get("output") or ""),
                    "status": "error" if failed else "success",
                }
            },
        )
    )


async def _iter_queued_chunks(
    model: AcpRunnerChatModel,
    queue: asyncio.Queue[tuple[str, Any] | None],
) -> Any:
    """Turn ACP events into chat chunks, and close tool cards when the prompt ends."""
    yielded = False
    tool_index = 0
    opened: dict[str, int] = {}
    while True:
        item = await queue.get()
        if item is None:
            break
        kind, payload = item
        yielded = True
        if kind == "tool" and isinstance(payload, dict):
            tool_chunks, tool_index = _tool_chunks(payload, opened, tool_index)
            for tool_chunk in tool_chunks:
                yield tool_chunk
            continue
        if kind == "reasoning":
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    additional_kwargs={"reasoning_content": str(payload)},
                )
            )
            continue
        yield ChatGenerationChunk(message=AIMessageChunk(content=str(payload)))
    if model.permission is None:
        for call_id in list(opened):
            yield _tool_result_chunk({"call_id": call_id, "status": "completed", "output": ""})
    if not yielded:
        yield ChatGenerationChunk(message=AIMessageChunk(content=""))


def _stream_message(chunk: Any) -> Any:
    message = getattr(chunk, "message", None)
    return message if message is not None else chunk


def _collect_managed_chunk(
    chunk: Any,
    text_parts: list[str],
    reasoning_parts: list[str],
    tools: list[dict[str, Any]],
    tools_by_id: dict[str, dict[str, Any]],
) -> None:
    """Remember reasoning and tool cards so the checkpoint can show them later."""
    message = _stream_message(chunk)
    extra = getattr(message, "additional_kwargs", None)
    if not isinstance(extra, dict):
        extra = {}
    result = extra.get("octop_tool_result")
    if isinstance(result, dict):
        call_id = str(result.get("tool_call_id") or "")
        if not call_id:
            return
        record = tools_by_id.get(call_id)
        if record is None:
            record = {"id": call_id, "name": "tool", "output": "", "failed": False}
            tools_by_id[call_id] = record
            tools.append(record)
        record["output"] = str(result.get("content") or "")
        record["failed"] = result.get("status") == "error"
        return
    opened = False
    for call in getattr(message, "tool_call_chunks", None) or []:
        if not isinstance(call, dict):
            continue
        call_id = str(call.get("id") or "")
        if not call_id or call_id in tools_by_id:
            opened = True
            continue
        record = {
            "id": call_id,
            "name": str(call.get("name") or "tool"),
            "output": "",
            "failed": False,
        }
        tools_by_id[call_id] = record
        tools.append(record)
        opened = True
    if opened:
        return
    reasoning = extra.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        reasoning_parts.append(reasoning)
        return
    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        text_parts.append(content)


def _managed_history_messages(
    text_parts: list[str],
    reasoning_parts: list[str],
    tools: list[dict[str, Any]],
) -> list[BaseMessage]:
    """Store the turn that was already streamed. The flag keeps it off the live channel."""
    stored: list[BaseMessage] = []
    reasoning = "".join(reasoning_parts).strip()
    if reasoning:
        stored.append(
            AIMessage(
                content="",
                additional_kwargs={
                    "octop_managed_streamed": True,
                    "reasoning_content": reasoning,
                },
            )
        )
    for record in tools:
        call_id = str(record.get("id") or "")
        if not call_id:
            continue
        name = str(record.get("name") or "tool")
        stored.append(
            AIMessage(
                content="",
                tool_calls=[{"name": name, "args": {}, "id": call_id, "type": "tool_call"}],
                additional_kwargs={"octop_managed_streamed": True},
            )
        )
        stored.append(
            ToolMessage(
                content=str(record.get("output") or ""),
                tool_call_id=call_id,
                name=name,
                status="error" if record.get("failed") else "success",
                additional_kwargs={"octop_managed_streamed": True},
            )
        )
    stored.append(
        AIMessage(
            content="".join(text_parts),
            additional_kwargs={"octop_managed_streamed": True},
        )
    )
    return stored


def _user_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        role = getattr(message, "type", None)
        if role not in {"human", "user"}:
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
            return "".join(parts)
    return ""


def _decision_type(decision: Any) -> str | None:
    """Read the card decision. Anything other than an explicit approve or reject is ignored."""
    if not isinstance(decision, dict):
        return None
    decisions = decision.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        return None
    first = decisions[0]
    if not isinstance(first, dict):
        return None
    kind = first.get("type")
    if kind in {"approve", "reject"}:
        return str(kind)
    return None


def _option_id(option: dict[str, Any]) -> str:
    return str(option.get("optionId") or option.get("option_id") or "")


def _option_label(option: dict[str, Any]) -> str:
    return str(option.get("kind") or option.get("name") or option.get("title") or "").lower()


def _is_deny(label: str) -> bool:
    return any(token in label for token in ("reject", "cancel", "deny"))


def _pick_option(options: list[dict[str, Any]], *, allow: bool) -> str:
    """Choose one ACP option. Approve prefers a one-time allow over a permanent one."""
    if allow:
        once = ""
        plain = ""
        always = ""
        for option in options:
            label = _option_label(option)
            option_id = _option_id(option)
            if not option_id or _is_deny(label):
                continue
            if "always" in label:
                always = always or option_id
            elif "once" in label:
                once = once or option_id
            else:
                plain = plain or option_id
        return once or plain or always
    for option in options:
        option_id = _option_id(option)
        if option_id and _is_deny(_option_label(option)):
            return option_id
    return ""


def option_id_for_decision(permission: Any, decision: Any) -> str:
    """Map an approve/reject card onto one ACP permission option id.

    A missing or unexpected decision is treated as a rejection. Approval never
    selects ``allow_always`` while a one-time allow is also offered.
    """
    options = [item for item in (getattr(permission, "options", None) or []) if isinstance(item, dict)]
    if not options:
        return ""
    return _pick_option(options, allow=_decision_type(decision) == "approve")


def _hitl_payload(permission: Any) -> dict[str, Any]:
    name = str(getattr(permission, "tool_name", None) or "external-agent")
    raw_summary = getattr(permission, "summary", None)
    summary = raw_summary.strip() if isinstance(raw_summary, str) else ""
    if summary == "external-agent":
        summary = ""
    elif not summary and name != "external-agent":
        summary = name
    request: dict[str, Any] = {"name": name, "args": {"summary": summary} if summary else {}}
    if summary:
        request["description"] = summary
    return {
        "action_requests": [request],
        "review_configs": [
            {
                "action_name": name,
                "allowed_decisions": ["approve", "reject"],
            }
        ],
    }


class AcpRunnerChatModel(BaseChatModel):
    """Stream one ACP turn as chat-model chunks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    runner: str
    cwd: str
    thread_id: str
    service: Any
    option_id: str | None = None
    mcp_servers: dict[str, Any] | None = None
    prompt_note: str = ""
    locale: str = ""
    permission_mode: str = "ask"
    session_mode: str = ""
    _permission: Any = PrivateAttr(default=None)

    @property
    def permission(self) -> Any:
        return self._permission

    @property
    def _llm_type(self) -> str:
        return "acp-managed"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        raise RuntimeError("ACP managed agent only supports async streaming")

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        del stop, run_manager, kwargs
        self._permission = None
        queue: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()

        async def on_message(payload: dict[str, Any], _is_last: bool) -> None:
            kind = str(payload.get("type") or "")
            if kind == "text_delta":
                delta = str(payload.get("text") or "")
                if delta:
                    await queue.put(("text", delta))
                return
            if kind == "reasoning_delta":
                delta = str(payload.get("text") or "")
                if delta:
                    await queue.put(("reasoning", delta))
                return
            if kind in {"tool_start", "tool_update"}:
                await queue.put(("tool", payload))

        async def execute() -> None:
            try:
                if self.option_id:
                    session = await self.service.get_session(self.thread_id, self.runner)
                    if session is None:
                        await queue.put(("text", "ACP session is no longer active."))
                        return
                    result = await self.service.resume_permission(
                        acp_session_id=session.acp_session_id,
                        option_id=self.option_id,
                        on_message=on_message,
                    )
                else:
                    result = await self.service.run_turn(
                        thread_id=self.thread_id,
                        runner=self.runner,
                        prompt_blocks=[
                            {"type": "text", "text": append_prompt_note(_user_text(list(messages)), self.prompt_note)}
                        ],
                        cwd=self.cwd,
                        on_message=on_message,
                        mcp_servers=self.mcp_servers,
                        locale=self.locale,
                        permission_mode=self.permission_mode,
                        session_mode=self.session_mode,
                    )
                if result.get("status") == "permission_required":
                    self._permission = result.get("suspended_permission")
            except Exception as exc:
                await queue.put(("text", str(exc)))
            finally:
                await queue.put(None)

        task = asyncio.create_task(execute())
        try:
            async for chunk in _iter_queued_chunks(self, queue):
                yield chunk
        finally:
            await task


def build_managed_graph(
    *,
    runner: str,
    acp_config: ACPConfig,
    cwd: str,
    checkpointer: Any = None,
    service: Any = None,
) -> Any:
    """Compile the managed-agent graph bound to one ACP runner."""
    bound = service if service is not None else get_acp_service(cwd) or init_acp_service(cwd, acp_config)

    async def call(state: ManagedState, config: RunnableConfig) -> dict[str, Any]:
        thread_id = _thread_id(config)
        option_id = (state.get("resume_option") or "").strip() or None
        model = AcpRunnerChatModel(
            runner=runner,
            cwd=cwd,
            thread_id=thread_id,
            service=bound,
            option_id=option_id,
            mcp_servers=managed_turn_mcp(config),
            prompt_note=managed_prompt_note(config),
            locale=_locale(config),
            permission_mode=managed_permission_mode(config),
            session_mode=managed_session_mode(config),
        )
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tools: list[dict[str, Any]] = []
        tools_by_id: dict[str, dict[str, Any]] = {}
        async for chunk in model.astream(list(state.get("messages") or []), config):
            _collect_managed_chunk(chunk, text_parts, reasoning_parts, tools, tools_by_id)
            # Let the graph emit this chunk before the next ACP update is pulled.
            # Otherwise a burst of deltas stays inside the node and the chat
            # paints the whole reply at once.
            await asyncio.sleep(0)
        return {
            "messages": _managed_history_messages(text_parts, reasoning_parts, tools),
            "needs_permission": model.permission is not None,
            "resume_option": "",
        }

    async def pause(state: ManagedState, config: RunnableConfig) -> dict[str, Any]:
        """Pause after the turn's text is already in the checkpoint."""
        if not state.get("needs_permission"):
            return {}
        pending = await bound.get_pending_permission(thread_id=_thread_id(config), runner=runner)
        if pending is None:
            return {
                "needs_permission": False,
                "resume_option": "",
                "messages": [AIMessage(content="ACP permission is no longer pending.")],
            }
        decision = interrupt(_hitl_payload(pending))
        option_id = option_id_for_decision(pending, decision)
        if not option_id:
            return {
                "needs_permission": False,
                "resume_option": "",
                "messages": [AIMessage(content="ACP permission has no selectable option.")],
            }
        return {"needs_permission": False, "resume_option": option_id}

    def _after_call(state: ManagedState) -> str:
        return "pause" if state.get("needs_permission") else END

    def _after_pause(state: ManagedState) -> str:
        return "call" if (state.get("resume_option") or "").strip() else END

    graph: StateGraph[ManagedState] = StateGraph(ManagedState)
    graph.add_node("call", call)
    graph.add_node("pause", pause)
    graph.add_edge(START, "call")
    graph.add_conditional_edges("call", _after_call, {"pause": "pause", END: END})
    graph.add_conditional_edges("pause", _after_pause, {"call": "call", END: END})
    return graph.compile(checkpointer=checkpointer)
