"""Expose a :class:`~octop_harness.HarnessAgent` as an ACP agent over stdio."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from uuid import uuid4

from octop_harness.backends.utils import ASYNC_DEFENSIVE_OP_ERRORS

logger = logging.getLogger(__name__)

ACP_AGENT_META_KEY = "octop.agent"
ACP_ERROR_META_KEY = "octop.error"
ACP_EXPERT_CONFIG_ID = "expert"
ACP_EXPERT_CATEGORY = "_expert"

# Sync or async: hosts may finish embedding the agent after ACP initialize.
ExpertResolver = Callable[[str], Any | Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class ACPExpertOption:
    """One selectable Octop expert / agent advertised via ACP ``configOptions``."""

    value: str
    name: str
    description: str = ""


def _package_version() -> str:
    try:
        return version("octop-harness")
    except PackageNotFoundError:
        return "0.0.0"


class _StreamTracker:
    """Convert cumulative harness text into ACP incremental deltas."""

    def __init__(self) -> None:
        self._prev_text = ""
        self._prev_thinking = ""
        self._seen_tool_ids: set[str] = set()
        self._tool_inputs: dict[str, Any] = {}

    def delta_text(self, cumulative: str) -> str:
        delta = cumulative[len(self._prev_text) :] if cumulative.startswith(self._prev_text) else cumulative
        self._prev_text = cumulative
        return delta

    def delta_thinking(self, cumulative: str) -> str:
        delta = cumulative[len(self._prev_thinking) :] if cumulative.startswith(self._prev_thinking) else cumulative
        self._prev_thinking = cumulative
        return delta

    def is_new_tool_call(self, tool_id: str) -> bool:
        if tool_id in self._seen_tool_ids:
            return False
        self._seen_tool_ids.add(tool_id)
        return True

    def tool_input_changed(self, tool_id: str, raw_input: Any) -> bool:
        if not raw_input:
            return False
        if self._tool_inputs.get(tool_id) == raw_input:
            return False
        self._tool_inputs[tool_id] = raw_input
        return True


def _extract_prompt_text(prompt: Any) -> str:
    parts: list[str] = []
    for block in prompt or []:
        text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def _message_text(content: Any) -> str:
    """Flatten LangChain / OpenAI-style message content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def _tool_output_text(messages: list[Any]) -> str:
    parts: list[str] = []
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else msg.get("content") if isinstance(msg, dict) else ""
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        elif content is not None:
            parts.append(str(content))
    return "\n".join(parts)


def _chunk_to_updates(
    chunk: dict[str, Any],
    tracker: _StreamTracker,
    *,
    tool_name_buf: dict[str, str],
    tool_id_buf: dict[str, str],
) -> list[Any]:
    from acp import (
        start_tool_call,
        text_block,
        tool_content,
        update_agent_message,
        update_agent_thought,
        update_tool_call,
    )

    ctype = str(chunk.get("type") or "")
    updates: list[Any] = []

    if ctype == "token":
        text = str(chunk.get("content") or "")
        delta = tracker.delta_text(text)
        if delta:
            updates.append(update_agent_message(text_block(delta)))
    elif ctype == "reasoning":
        text = str(chunk.get("content") or "")
        delta = tracker.delta_thinking(text)
        if delta:
            updates.append(update_agent_thought(text_block(delta)))
    elif ctype == "tool_call_chunk":
        idx_key = f"_idx_{chunk.get('index', 0)}"
        name_frag = str(chunk.get("name") or "")
        tool_id = str(chunk.get("id") or "") or tool_id_buf.get(idx_key, "")
        if name_frag:
            tool_name_buf[idx_key] = tool_name_buf.get(idx_key, "") + name_frag
        if tool_id:
            tool_id_buf[idx_key] = tool_id
        args_frag = chunk.get("args")
        resolved_id = tool_id_buf.get(idx_key) or idx_key
        resolved_name = tool_name_buf.get(idx_key, "tool")
        if name_frag and tracker.is_new_tool_call(resolved_id):
            updates.append(
                start_tool_call(
                    resolved_id,
                    resolved_name,
                    status="in_progress",
                    raw_input=args_frag or None,
                ),
            )
            tracker.tool_input_changed(resolved_id, args_frag)
        elif args_frag and tracker.tool_input_changed(resolved_id, args_frag):
            updates.append(update_tool_call(resolved_id, raw_input=args_frag))
    elif ctype == "tool_result":
        messages = chunk.get("messages") or []
        output = _tool_output_text(messages)
        tool_id = next(iter(tool_id_buf.values()), uuid4().hex[:8])
        updates.append(
            update_tool_call(
                tool_id,
                status="completed",
                content=[tool_content(text_block(output or "(no output)"))],
            ),
        )
    return updates


class HarnessACPAgent:
    """ACP Agent implementation backed by one or more live :class:`~octop_harness.HarnessAgent`s.

    Single-harness mode (default): pass ``harness`` + optional ``agent_id`` — same as before.

    Multi-expert mode: pass ``experts`` (selectable ACP ``configOptions``) and
    ``resolve_harness`` to resolve the live harness for the selected expert id.
    ``agent_id`` is the default ``currentValue`` (must be one of ``experts``).
    """

    def __init__(
        self,
        harness: Any | None = None,
        *,
        agent_id: str = "",
        experts: Sequence[ACPExpertOption] | None = None,
        resolve_harness: ExpertResolver | None = None,
    ) -> None:
        expert_list = tuple(experts or ())
        if expert_list:
            values = [e.value for e in expert_list]
            if len(set(values)) != len(values):
                raise ValueError("ACP expert option values must be unique")
            default = agent_id or values[0]
            if default not in values:
                raise ValueError(f"default agent_id {default!r} is not in experts")
            if resolve_harness is None and harness is None:
                raise ValueError("multi-expert ACP requires resolve_harness or a default harness")
            self._experts = expert_list
            self._agent_id = default
            self._resolve_harness = resolve_harness
            self._harness = harness
        else:
            if harness is None:
                raise ValueError("harness is required when experts are not provided")
            self._experts = ()
            self._agent_id = agent_id
            self._resolve_harness = None
            self._harness = harness

        self._conn: Any = None
        self._sessions: dict[str, dict[str, Any]] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}

    def on_connect(self, conn: Any) -> None:
        self._conn = conn

    def _session_expert_id(self, session_id: str) -> str:
        session = self._sessions.get(session_id) or {}
        expert_id = str(session.get("expert_id") or self._agent_id or "")
        return expert_id

    def _thread_id_for_session(self, session_id: str, *, expert_id: str | None = None) -> str:
        """Map ACP session id → harness checkpoint thread id."""
        eid = expert_id if expert_id is not None else self._session_expert_id(session_id)
        if self._experts and eid:
            return f"acp:{session_id}:{eid}"
        return session_id

    def _bind_session(
        self,
        session_id: str,
        *,
        cwd: str,
        expert_id: str | None = None,
        title: str | None = None,
    ) -> str:
        """Register or refresh an active ACP session; return the bound expert id."""
        existing = self._sessions.get(session_id) or {}
        bound = str(
            expert_id
            or existing.get("expert_id")
            or self._agent_id
            or (self._experts[0].value if self._experts else ""),
        )
        row: dict[str, Any] = {"cwd": cwd, "expert_id": bound}
        if title is not None:
            row["title"] = title
        elif existing.get("title"):
            row["title"] = existing["title"]
        self._sessions[session_id] = row
        return bound

    async def _harness_for_session(self, session_id: str) -> Any:
        expert_id = self._session_expert_id(session_id)
        if self._resolve_harness is not None and expert_id:
            try:
                harness = self._resolve_harness(expert_id)
                if inspect.isawaitable(harness):
                    harness = await harness
                return harness
            except Exception as exc:
                from acp import RequestError

                raise RequestError.invalid_params(
                    {"configId": ACP_EXPERT_CONFIG_ID, "value": expert_id, "reason": str(exc)},
                ) from exc
        if self._harness is None:
            from acp import RequestError

            raise RequestError.internal_error({"reason": "no harness bound for ACP session"})
        return self._harness

    async def _replay_history(self, session_id: str) -> None:
        """Stream prior turns to the client (required by ACP ``session/load``)."""
        if self._conn is None:
            return
        from acp import text_block, update_agent_message, update_user_message

        harness = await self._harness_for_session(session_id)
        aget_history = getattr(harness, "aget_history", None)
        if not callable(aget_history):
            return
        thread_id = self._thread_id_for_session(session_id)
        try:
            messages = await aget_history(thread_id, limit=500)
        except ASYNC_DEFENSIVE_OP_ERRORS:
            logger.exception("ACP load_session: history read failed session=%s", session_id)
            return

        for msg in messages or []:
            role = getattr(msg, "type", None) or getattr(msg, "role", None) or ""
            text = _message_text(getattr(msg, "content", None)).strip()
            if not text:
                continue
            role_s = str(role).lower()
            update: Any
            if role_s in {"human", "user"}:
                update = update_user_message(text_block(text))
            elif role_s in {"ai", "assistant"}:
                update = update_agent_message(text_block(text))
            else:
                # Skip tool / system / unknown — load must still complete.
                continue
            await self._conn.session_update(session_id=session_id, update=update)

    def _expert_config_options(self, *, current_value: str) -> list[Any] | None:
        if not self._experts:
            return None
        from acp.schema import SessionConfigOptionSelect, SessionConfigSelectOption

        options = [
            SessionConfigSelectOption(
                value=item.value,
                name=item.name,
                description=item.description or None,
            )
            for item in self._experts
        ]
        return [
            SessionConfigOptionSelect(
                id=ACP_EXPERT_CONFIG_ID,
                name="Expert",
                description="Octop expert (agent) that handles this session",
                category=ACP_EXPERT_CATEGORY,
                type="select",
                current_value=current_value,
                options=options,
            ),
        ]

    def _session_meta(self, expert_id: str) -> dict[str, str] | None:
        return {ACP_AGENT_META_KEY: expert_id} if expert_id else None

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any | None = None,
        client_info: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        from acp import PROTOCOL_VERSION
        from acp.schema import (
            AgentCapabilities,
            Implementation,
            InitializeResponse,
            SessionCapabilities,
            SessionCloseCapabilities,
            SessionListCapabilities,
            SessionResumeCapabilities,
        )

        logger.info("ACP initialize: version=%s client=%s", protocol_version, client_info)
        # Negotiate: echo a supported version; if the client asks for an unsupported
        # version, answer with our latest supported (ACP-INIT-003).
        supported = {PROTOCOL_VERSION}
        negotiated = protocol_version if protocol_version in supported else PROTOCOL_VERSION
        return InitializeResponse(
            protocol_version=negotiated,
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=None,
                mcp_capabilities=None,
                session_capabilities=SessionCapabilities(
                    close=SessionCloseCapabilities(),
                    resume=SessionResumeCapabilities(),
                    list=SessionListCapabilities(),
                ),
                auth=None,
            ),
            agent_info=Implementation(
                name="octop",
                title="Octop Agent",
                version=_package_version(),
            ),
        )

    async def new_session(self, cwd: str, **kwargs: Any) -> Any:
        from acp.schema import NewSessionResponse

        session_id = uuid4().hex
        expert_id = self._bind_session(session_id, cwd=cwd, expert_id=self._agent_id)
        logger.info("ACP new_session: id=%s cwd=%s expert=%s", session_id, cwd, expert_id)
        return NewSessionResponse(
            session_id=session_id,
            config_options=self._expert_config_options(current_value=expert_id),
            field_meta=self._session_meta(expert_id),
        )

    async def load_session(self, cwd: str, session_id: str, **kwargs: Any) -> Any:
        """Resume a prior conversation and replay history (ACP ``session/load``)."""
        from acp.schema import LoadSessionResponse

        token = str(session_id or "").strip()
        if not token:
            from acp import RequestError

            raise RequestError.invalid_params({"sessionId": session_id, "reason": "empty sessionId"})

        expert_id = self._bind_session(token, cwd=cwd)
        logger.info("ACP load_session: id=%s cwd=%s expert=%s", token, cwd, expert_id)
        # Spec: stream the entire conversation via session/update, then respond.
        await self._replay_history(token)
        return LoadSessionResponse(
            config_options=self._expert_config_options(current_value=expert_id),
            field_meta=self._session_meta(expert_id),
        )

    async def resume_session(self, cwd: str, session_id: str, **kwargs: Any) -> Any:
        """Reconnect without replaying history (ACP ``session/resume``)."""
        from acp.schema import ResumeSessionResponse

        token = str(session_id or "").strip()
        if not token:
            from acp import RequestError

            raise RequestError.invalid_params({"sessionId": session_id, "reason": "empty sessionId"})

        expert_id = self._bind_session(token, cwd=cwd)
        logger.info("ACP resume_session: id=%s cwd=%s expert=%s", token, cwd, expert_id)
        return ResumeSessionResponse(
            config_options=self._expert_config_options(current_value=expert_id),
            field_meta=self._session_meta(expert_id),
        )

    async def list_sessions(
        self,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """List active ACP sessions known to this process."""
        from acp.schema import ListSessionsResponse, SessionInfo

        _ = cursor  # pagination not required while the set is small / process-local
        sessions: list[Any] = []
        for sid, meta in self._sessions.items():
            session_cwd = str(meta.get("cwd") or "")
            if cwd and session_cwd and session_cwd != cwd:
                continue
            sessions.append(
                SessionInfo(
                    session_id=sid,
                    cwd=session_cwd or (cwd or "."),
                    title=meta.get("title"),
                    field_meta=self._session_meta(str(meta.get("expert_id") or "")),
                ),
            )
        return ListSessionsResponse(sessions=sessions, next_cursor=None)

    async def set_config_option(
        self,
        config_id: str,
        session_id: str,
        value: str | bool,
        **kwargs: Any,
    ) -> Any:
        from acp import RequestError
        from acp.schema import SetSessionConfigOptionResponse

        if not self._experts:
            raise RequestError.invalid_params({"configId": config_id, "reason": "no config options"})
        if config_id != ACP_EXPERT_CONFIG_ID:
            raise RequestError.invalid_params({"configId": config_id, "reason": "unknown configId"})
        if not isinstance(value, str):
            raise RequestError.invalid_params(
                {"configId": config_id, "reason": "expert value must be a string id"},
            )
        allowed = {item.value for item in self._experts}
        if value not in allowed:
            raise RequestError.invalid_params(
                {"configId": config_id, "value": value, "reason": "value not in options"},
            )

        session = self._sessions.get(session_id)
        if session is None:
            raise RequestError.invalid_params(
                {"sessionId": session_id, "reason": "unknown session"},
            )

        # Validate the target harness is resolvable before committing the switch.
        if self._resolve_harness is not None:
            try:
                resolved = self._resolve_harness(value)
                if inspect.isawaitable(resolved):
                    await resolved
            except Exception as exc:
                raise RequestError.invalid_params(
                    {"configId": config_id, "value": value, "reason": str(exc)},
                ) from exc

        session["expert_id"] = value
        logger.info("ACP set_config_option: session=%s expert=%s", session_id, value)
        options = self._expert_config_options(current_value=value)
        assert options is not None
        return SetSessionConfigOptionResponse(config_options=options)

    async def prompt(self, prompt: Any, session_id: str, **kwargs: Any) -> Any:  # noqa: PLR0912
        from acp import RequestError
        from acp.schema import PromptResponse

        text = _extract_prompt_text(prompt)
        if not text:
            return PromptResponse(stop_reason="end_turn")

        if session_id not in self._sessions:
            raise RequestError.invalid_params(
                {"sessionId": session_id, "reason": "unknown session"},
            )

        expert_id = self._session_expert_id(session_id)
        harness = await self._harness_for_session(session_id)

        cancel_event = asyncio.Event()
        self._cancel_events[session_id] = cancel_event
        tracker = _StreamTracker()
        tool_name_buf: dict[str, str] = {}
        tool_id_buf: dict[str, str] = {}

        thread_id = self._thread_id_for_session(session_id, expert_id=expert_id)
        request: dict[str, Any] = {
            "messages": [{"role": "user", "content": text}],
            "thread_id": thread_id,
            "source": "acp",
        }
        if expert_id:
            request["agent_id"] = expert_id

        cancelled = False
        stream = harness.stream(request)
        try:
            async for chunk in stream:
                if cancel_event.is_set():
                    cancelled = True
                    logger.info("ACP prompt cancelled: session=%s", session_id)
                    break
                for update in _chunk_to_updates(
                    chunk,
                    tracker,
                    tool_name_buf=tool_name_buf,
                    tool_id_buf=tool_id_buf,
                ):
                    if cancel_event.is_set():
                        cancelled = True
                        break
                    await self._conn.session_update(session_id=session_id, update=update)
                if cancelled:
                    break
        except ASYNC_DEFENSIVE_OP_ERRORS as exc:
            logger.exception("ACP prompt error: session=%s", session_id)
            await self._report_prompt_error(session_id, exc)
        finally:
            self._cancel_events.pop(session_id, None)
            aclose = getattr(stream, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except ASYNC_DEFENSIVE_OP_ERRORS:
                    logger.debug("ACP: stream aclose failed for session=%s", session_id, exc_info=True)

        if cancel_event.is_set():
            cancelled = True
        return PromptResponse(stop_reason="cancelled" if cancelled else "end_turn")

    async def close_session(self, session_id: str, **kwargs: Any) -> Any:
        from acp.schema import CloseSessionResponse

        logger.info("ACP close_session: session=%s", session_id)
        # Cancel any in-flight prompt so it can resolve with stopReason=cancelled
        # before we drop the session (ACP-CLOSE-002).
        event = self._cancel_events.get(session_id)
        if event is not None:
            event.set()
        self._sessions.pop(session_id, None)
        return CloseSessionResponse()

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        logger.info("ACP cancel: session=%s", session_id)
        event = self._cancel_events.get(session_id)
        if event is not None:
            event.set()

    async def _report_prompt_error(self, session_id: str, exc: BaseException) -> None:
        from acp import text_block
        from acp.schema import AgentMessageChunk

        try:
            await self._conn.session_update(
                session_id=session_id,
                update=AgentMessageChunk(
                    session_update="agent_message_chunk",
                    content=text_block(f"Error: {exc}"),
                    field_meta={ACP_ERROR_META_KEY: True},
                ),
            )
        except ASYNC_DEFENSIVE_OP_ERRORS:
            logger.exception("ACP: failed to report prompt error (session=%s)", session_id)


def _acp_agent_cls(
    harness: Any | None,
    *,
    agent_id: str = "",
    experts: Sequence[ACPExpertOption] | None = None,
    resolve_harness: ExpertResolver | None = None,
) -> type:
    """Concrete ACP agent class with implementation-first MRO.

    ``acp.Agent`` is a :class:`~typing.Protocol`. Putting it *before*
    :class:`HarnessACPAgent` makes Protocol stub methods (returning
    ``None``) win lookup, so ``initialize`` / ``session/new`` serialize
    as JSON ``null`` and IDE clients (Zed, Multica, …) fail to handshake.
    Implementation must come first: ``(HarnessACPAgent, Agent)``.
    """
    from acp import Agent

    class _Agent(HarnessACPAgent, Agent):  # type: ignore[misc]
        def __init__(self) -> None:
            HarnessACPAgent.__init__(
                self,
                harness,
                agent_id=agent_id,
                experts=experts,
                resolve_harness=resolve_harness,
            )

    return _Agent


async def run_harness_acp_server(
    harness: Any | None = None,
    *,
    agent_id: str = "",
    experts: Sequence[ACPExpertOption] | Mapping[str, Mapping[str, str]] | None = None,
    resolve_harness: ExpertResolver | None = None,
) -> None:
    """Run as an ACP agent on stdio until the client disconnects."""
    from acp import run_agent

    parsed = _coerce_experts(experts)
    await run_agent(
        _acp_agent_cls(
            harness,
            agent_id=agent_id,
            experts=parsed,
            resolve_harness=resolve_harness,
        )(),
        use_unstable_protocol=True,
    )


def build_harness_acp_agent(
    harness: Any | None = None,
    *,
    agent_id: str = "",
    experts: Sequence[ACPExpertOption] | Mapping[str, Mapping[str, str]] | None = None,
    resolve_harness: ExpertResolver | None = None,
) -> Any:
    """Build an ACP :class:`acp.Agent` wrapper (for tests)."""
    return _acp_agent_cls(
        harness,
        agent_id=agent_id,
        experts=_coerce_experts(experts),
        resolve_harness=resolve_harness,
    )()


def _coerce_experts(
    experts: Sequence[ACPExpertOption] | Mapping[str, Mapping[str, str]] | None,
) -> tuple[ACPExpertOption, ...] | None:
    if experts is None:
        return None
    if isinstance(experts, Mapping):
        return tuple(
            ACPExpertOption(
                value=key,
                name=str(meta.get("name") or key),
                description=str(meta.get("description") or ""),
            )
            for key, meta in experts.items()
        )
    return tuple(experts)
