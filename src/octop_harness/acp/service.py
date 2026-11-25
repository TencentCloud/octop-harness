"""High-level ACP service for delegated external agent sessions."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from octop_harness.acp.client import ACPHostedClient
from octop_harness.acp.mcp_servers import acp_mcp_servers_from_specs, mcp_fingerprint
from octop_harness.acp.models import ACPConfig, ACPConfigurationError, ACPRunnerConfig, ACPSessionError
from octop_harness.acp.permission_mode import select_session_mode
from octop_harness.acp.session_index import (
    AcpSessionIndex,
    collaboration_notice,
    resume_or_create_session,
    session_index_path,
)
from octop_harness.backends.utils import ASYNC_DEFENSIVE_OP_ERRORS
from octop_harness.runtime_env import load_dotenv_path, overlay_env, prepend_user_cli_path

logger = logging.getLogger(__name__)

MessageHandler = Callable[[dict[str, Any], bool], Awaitable[None]]


def _kill_process_tree(pid: int) -> None:
    try:
        import psutil
    except ImportError:
        return
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    for child in parent.children(recursive=True):
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()
    with contextlib.suppress(psutil.NoSuchProcess):
        parent.kill()


@dataclass
class _Conversation:
    thread_id: str
    runner: str
    acp_session_id: str
    cwd: str
    conn: Any
    process: Any
    client: ACPHostedClient
    exit_stack: AsyncExitStack
    turn_lock: asyncio.Lock
    prompt_task: asyncio.Task[Any] | None = None
    load_supported: bool = False
    mcp_key: str = ""
    mcp_degraded: bool = False
    mcp_notice_sent: bool = False
    current_mode_id: str | None = None
    available_modes: tuple[Any, ...] = ()
    default_mode_id: str | None = None


class ACPService:
    def __init__(self, *, config: ACPConfig) -> None:
        self.config = config
        self._lock = asyncio.Lock()
        self._sessions: dict[tuple[str, str], _Conversation] = {}

    async def run_turn(
        self,
        *,
        thread_id: str,
        runner: str,
        prompt_blocks: list[dict[str, Any]],
        cwd: str,
        on_message: MessageHandler,
        restart: bool = False,
        require_existing: bool = False,
        mcp_servers: Mapping[str, Any] | None = None,
        locale: str = "",
        permission_mode: str = "ask",
        session_mode: str = "",
    ) -> dict[str, Any]:
        attached = None if mcp_servers is None else acp_mcp_servers_from_specs(mcp_servers)
        reopened = False
        if restart:
            await self.close_thread_session(thread_id=thread_id, runner=runner, cwd=cwd, forget=True)
        elif attached is not None:
            existing = await self.get_session(thread_id, runner)
            if existing is not None and await self._reload_mcp(existing, attached):
                await self.close_thread_session(thread_id=thread_id, runner=runner, cwd=cwd, forget=True)
                reopened = True

        conversation = await self._get_or_create_session(
            thread_id=thread_id,
            runner=runner,
            cwd=cwd,
            require_existing=require_existing,
            mcp_servers=attached,
        )
        async with conversation.turn_lock:
            if conversation.client.pending_permission is not None:
                raise ACPSessionError(
                    f"Session {conversation.acp_session_id} is waiting for permission",
                )
            if conversation.prompt_task is not None and not conversation.prompt_task.done():
                raise ACPSessionError(
                    f"Session {conversation.acp_session_id} is already processing a turn",
                )

            conversation.cwd = cwd or conversation.cwd
            conversation.client.update_cwd(conversation.cwd)
            await self._apply_permission_mode(conversation, permission_mode, session_mode=session_mode)
            conversation.client.start_prompt(on_message)
            if not conversation.mcp_notice_sent:
                notice = collaboration_notice(
                    degraded=conversation.mcp_degraded,
                    reopened=reopened,
                    locale=locale,
                )
                if notice:
                    await on_message({"type": "text_delta", "text": notice + "\n\n"}, False)
                    conversation.mcp_notice_sent = True
            conversation.prompt_task = asyncio.create_task(
                conversation.conn.prompt(
                    session_id=conversation.acp_session_id,
                    prompt=self._prompt_blocks_to_models(prompt_blocks),
                ),
            )
            return await self._wait_for_prompt_outcome(conversation=conversation)

    async def resume_permission(
        self,
        *,
        acp_session_id: str,
        option_id: str,
        on_message: MessageHandler,
    ) -> dict[str, Any]:
        conversation = await self._find_session_by_acp_id(acp_session_id)
        if conversation is None:
            raise ACPSessionError(f"Session not found: {acp_session_id}")
        if conversation.client.pending_permission is None:
            raise ACPSessionError(f"Session {acp_session_id} has no pending permission request")
        if conversation.prompt_task is None or conversation.prompt_task.done():
            raise ACPSessionError(f"Session {acp_session_id} is not awaiting permission resume")

        async with conversation.turn_lock:
            conversation.client.resume_prompt(on_message)
            conversation.client.resolve_permission(option_id)
            return await self._wait_for_prompt_outcome(conversation=conversation)

    async def close_thread_session(
        self,
        *,
        thread_id: str,
        runner: str,
        cwd: str | None = None,
        forget: bool = False,
    ) -> None:
        async with self._lock:
            conversation = self._sessions.pop((thread_id, runner), None)
        try:
            if forget:
                store_cwd = cwd or (conversation.cwd if conversation is not None else None)
                if store_cwd:
                    AcpSessionIndex(session_index_path(store_cwd)).forget(thread_id, runner)
        finally:
            if conversation is not None:
                await self._close_conversation(conversation)

    async def close_all_sessions(self) -> None:
        async with self._lock:
            conversations = list(self._sessions.values())
            self._sessions.clear()
        for conversation in conversations:
            await self._close_conversation(conversation)

    async def get_session(self, thread_id: str, runner: str) -> _Conversation | None:
        async with self._lock:
            return self._sessions.get((thread_id, runner))

    async def get_pending_permission(self, *, thread_id: str, runner: str) -> Any | None:
        conversation = await self.get_session(thread_id, runner)
        if conversation is None:
            return None
        return conversation.client.pending_permission

    async def cancel_turn(self, *, thread_id: str, runner: str) -> bool:
        conversation = await self.get_session(thread_id, runner)
        if conversation is None:
            return False
        prompt_task = conversation.prompt_task
        if prompt_task is None or prompt_task.done():
            return False
        try:
            await conversation.conn.cancel(session_id=conversation.acp_session_id)
        except (OSError, RuntimeError, asyncio.CancelledError):
            return False
        try:
            await asyncio.wait_for(asyncio.shield(prompt_task), timeout=0.5)
        except TimeoutError:
            return prompt_task.done()
        except (OSError, RuntimeError, asyncio.CancelledError):
            return prompt_task.done()
        return prompt_task.done()

    def _get_runner_config(self, runner: str) -> ACPRunnerConfig:
        runner_config = self.config.runners.get(runner)
        if runner_config is None:
            raise ACPConfigurationError(f"Unknown ACP runner: {runner}", runner=runner)
        if not runner_config.enabled:
            raise ACPConfigurationError(f"ACP runner '{runner}' is disabled", runner=runner)
        return runner_config

    async def _reload_mcp(self, conversation: _Conversation, servers: Sequence[Any]) -> bool:
        """Attach ``servers`` when they differ. True means the session must be reopened."""
        key = mcp_fingerprint(servers)
        if conversation.mcp_key == key:
            return False
        if conversation.load_supported:
            try:
                await conversation.conn.load_session(
                    cwd=conversation.cwd,
                    session_id=conversation.acp_session_id,
                    mcp_servers=list(servers),
                )
            except Exception:
                logger.warning(
                    "ACP could not refresh MCP servers for %s",
                    conversation.acp_session_id,
                    exc_info=True,
                )
            else:
                conversation.mcp_key = key
                return False
        return True

    async def _get_or_create_session(
        self,
        *,
        thread_id: str,
        runner: str,
        cwd: str,
        require_existing: bool,
        mcp_servers: Sequence[Any] | None = None,
    ) -> _Conversation:
        runner_config = self._get_runner_config(runner)
        async with self._lock:
            existing = self._sessions.get((thread_id, runner))

        if existing is not None:
            if existing.process.returncode is None:
                return existing
            await self.close_thread_session(thread_id=thread_id, runner=runner)
            if require_existing:
                raise ACPSessionError(
                    f"ACP session for runner '{runner}' is no longer active; call start first",
                )
        elif require_existing:
            raise ACPSessionError(
                f"no bound ACP session found for runner '{runner}' in current thread",
            )

        conversation = await self._open_conversation(
            thread_id=thread_id,
            runner=runner,
            cwd=cwd or ".",
            runner_config=runner_config,
            mcp_servers=mcp_servers,
        )
        async with self._lock:
            self._sessions[(thread_id, runner)] = conversation
        return conversation

    async def _find_session_by_acp_id(self, acp_session_id: str) -> _Conversation | None:
        async with self._lock:
            for session in self._sessions.values():
                if session.acp_session_id == acp_session_id:
                    return session
        return None

    async def _open_conversation(
        self,
        *,
        thread_id: str,
        runner: str,
        cwd: str,
        runner_config: ACPRunnerConfig,
        mcp_servers: Sequence[Any] | None = None,
    ) -> _Conversation:
        from acp import PROTOCOL_VERSION, spawn_agent_process
        from acp.schema import ClientCapabilities, Implementation

        client = ACPHostedClient(runner_name=runner, runner_config=runner_config, cwd=cwd)
        exit_stack = AsyncExitStack()
        spawn_env = overlay_env(os.environ, load_dotenv_path(os.path.join(cwd, ".env")), protect=True)
        spawn_env = overlay_env(spawn_env, runner_config.env, protect=False)
        spawn_env = prepend_user_cli_path(spawn_env)
        try:
            conn, process = await exit_stack.enter_async_context(
                spawn_agent_process(
                    client,  # type: ignore[arg-type]
                    runner_config.command,
                    *runner_config.args,
                    cwd=cwd,
                    env=spawn_env,
                    transport_kwargs={"limit": runner_config.stdio_buffer_limit_bytes},
                ),
            )
            initialized = await conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                capabilities=ClientCapabilities(),
                client_info=Implementation(name="octop-harness-acp", version="0.1.0"),
            )
            if initialized.protocol_version != PROTOCOL_VERSION:
                raise ACPSessionError(f"Protocol mismatch: {initialized.protocol_version}")
            capabilities = getattr(initialized, "agent_capabilities", None)
            load_supported = bool(getattr(capabilities, "load_session", False))
            index = AcpSessionIndex(session_index_path(cwd))
            opened = await resume_or_create_session(
                conn,
                cwd=cwd,
                session_id=index.get(thread_id, runner),
                load_supported=load_supported,
                mcp_servers=mcp_servers,
            )
            index.put(thread_id, runner, opened.session_id)
            return _Conversation(
                thread_id=thread_id,
                runner=runner,
                acp_session_id=opened.session_id,
                cwd=cwd,
                conn=conn,
                process=process,
                client=client,
                exit_stack=exit_stack,
                turn_lock=asyncio.Lock(),
                load_supported=load_supported,
                mcp_key="" if mcp_servers is None else mcp_fingerprint(mcp_servers),
                mcp_degraded=opened.mcp_degraded,
                current_mode_id=opened.current_mode_id,
                available_modes=opened.available_modes,
                default_mode_id=opened.default_mode_id,
            )
        except ASYNC_DEFENSIVE_OP_ERRORS:
            await exit_stack.aclose()
            raise

    async def _apply_permission_mode(self, conversation: _Conversation, mode: str, *, session_mode: str = "") -> None:
        """Switch the agent to the requested session mode, then auto-allow YOLO."""
        selected, target = select_session_mode(
            list(conversation.available_modes),
            mode,
            session_mode,
            conversation.default_mode_id,
        )
        conversation.client.set_permission_mode(selected)
        if not target or target == conversation.current_mode_id:
            return
        try:
            await conversation.conn.set_session_mode(
                session_id=conversation.acp_session_id,
                mode_id=target,
            )
        except Exception:
            logger.warning(
                "ACP set_session_mode %s failed for %s",
                target,
                conversation.acp_session_id,
                exc_info=True,
            )
            return
        conversation.current_mode_id = target

    async def _wait_for_prompt_outcome(self, *, conversation: _Conversation) -> dict[str, Any]:
        prompt_task = conversation.prompt_task
        if prompt_task is None:
            raise ACPSessionError("ACP prompt task is missing")

        permission_task = asyncio.create_task(conversation.client.wait_for_permission_request())
        try:
            done, _ = await asyncio.wait(
                {prompt_task, permission_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if permission_task in done and conversation.client.pending_permission is not None:
                finished_event = await conversation.client.finish_prompt()
                return {
                    "status": "permission_required",
                    "suspended_permission": conversation.client.pending_permission,
                    "event": finished_event,
                }

            permission_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await permission_task

            try:
                await prompt_task
            except ASYNC_DEFENSIVE_OP_ERRORS as exc:
                conversation.prompt_task = None
                await conversation.client.finish_prompt()
                raise ACPSessionError(str(exc)) from exc

            conversation.prompt_task = None
            finished_event = await conversation.client.finish_prompt()
            pending = conversation.client.pending_permission
            if pending is not None:
                return {
                    "status": "permission_required",
                    "suspended_permission": pending,
                    "event": finished_event,
                }
            return {"status": "completed", "event": finished_event}
        finally:
            if not permission_task.done():
                permission_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await permission_task

    async def _close_conversation(self, conversation: _Conversation) -> None:
        try:
            if conversation.prompt_task is not None and not conversation.prompt_task.done():
                conversation.prompt_task.cancel()
                with contextlib.suppress(Exception):
                    await conversation.prompt_task
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    conversation.conn.close_session(session_id=conversation.acp_session_id),
                    timeout=5.0,
                )
        finally:
            _kill_process_tree(conversation.process.pid)
            await conversation.exit_stack.aclose()

    @staticmethod
    def _prompt_blocks_to_models(blocks: list[dict[str, Any]]) -> list[Any]:
        from acp import text_block

        out: list[Any] = []
        for block in blocks:
            if block.get("type") != "text":
                raise ACPConfigurationError("Only text prompt blocks are currently supported")
            out.append(text_block(str(block.get("text", ""))))
        return out


_services: dict[str, ACPService] = {}


def get_acp_service(service_key: str) -> ACPService | None:
    return _services.get(service_key)


def init_acp_service(service_key: str, config: ACPConfig) -> ACPService:
    previous = _services.get(service_key)
    _services[service_key] = ACPService(config=config)
    if previous is not None:
        _schedule_close(previous)
    return _services[service_key]


def pop_acp_service(service_key: str) -> ACPService | None:
    """Detach the service for ``service_key`` without closing it yet."""
    return _services.pop(service_key, None)


def close_acp_service(service_key: str) -> None:
    previous = pop_acp_service(service_key)
    if previous is not None:
        _schedule_close(previous)


def _schedule_close(service: ACPService) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and not loop.is_closed():
        if loop.is_running():
            loop.create_task(service.close_all_sessions())
        else:
            loop.run_until_complete(service.close_all_sessions())
        return
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(service.close_all_sessions())
        loop.close()
    except (OSError, RuntimeError):
        pass


def _shutdown_services() -> None:
    services = list(_services.values())
    _services.clear()
    if not services:
        return
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            for service in services:
                loop.create_task(service.close_all_sessions())
            return
    except RuntimeError:
        pass
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(asyncio.gather(*(s.close_all_sessions() for s in services)))
        loop.close()
    except (OSError, RuntimeError):
        pass


atexit.register(_shutdown_services)
