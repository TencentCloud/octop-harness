"""Persist the ACP session id for one chat thread so a later process can resume it."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from octop_harness.acp.models import ACPSessionError
from octop_harness.acp.permission_mode import mode_state

logger = logging.getLogger(__name__)

_INDEX_LOCKS: dict[str, threading.Lock] = {}
_INDEX_LOCKS_GUARD = threading.Lock()


def _index_lock(path: Path) -> threading.Lock:
    """Serialize read-modify-write of one index inside this process."""
    key = str(path)
    with _INDEX_LOCKS_GUARD:
        lock = _INDEX_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _INDEX_LOCKS[key] = lock
        return lock


class AcpSessionIndex:
    """Map ``(thread, runner)`` to the ACP session id inside a workspace."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self, thread_id: str, runner: str) -> str | None:
        entry = self._read().get(thread_id)
        if not isinstance(entry, dict):
            return None
        raw = entry.get(runner)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return None

    def put(self, thread_id: str, runner: str, session_id: str) -> None:
        with _index_lock(self.path):
            data = self._read()
            entry = data.get(thread_id)
            if not isinstance(entry, dict):
                entry = {}
            entry[runner] = session_id
            data[thread_id] = entry
            self._write(data)

    def forget(self, thread_id: str, runner: str) -> None:
        with _index_lock(self.path):
            data = self._read()
            entry = data.get(thread_id)
            if not isinstance(entry, dict) or runner not in entry:
                return
            entry.pop(runner, None)
            if entry:
                data[thread_id] = entry
            else:
                data.pop(thread_id, None)
            self._write(data)

    def _read(self) -> dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError:
            logger.warning("ACP session index unreadable: %s", self.path, exc_info=True)
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("ACP session index is not JSON: %s", self.path)
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)


def session_index_path(cwd: str) -> Path:
    return Path(cwd).expanduser() / ".octop" / "acp-sessions.json"


@dataclass(frozen=True)
class OpenedSession:
    session_id: str
    mcp_degraded: bool = False
    current_mode_id: str | None = None
    available_modes: tuple[Any, ...] = ()
    default_mode_id: str | None = None


def collaboration_notice(*, degraded: bool, reopened: bool, locale: str) -> str:
    """One user-visible line when collaboration tools were dropped or the session restarted."""
    if not degraded and not reopened:
        return ""
    chinese = locale.lower().startswith("zh")
    parts: list[str] = []
    if reopened:
        parts.append(
            "协作工具有变化, 外部智能体的当前会话已重新开始."
            if chinese
            else "Collaboration tools changed, so this external agent's session started over."
        )
    if degraded:
        parts.append(
            "外部智能体没有接上这次的协作工具, 连接器、知识库和专家调用暂不可用."
            if chinese
            else (
                "This agent did not accept the collaboration tools, "
                "so connectors, knowledge, and expert calls are unavailable."
            )
        )
    return "\n".join(parts)


async def resume_or_create_session(
    conn: Any,
    *,
    cwd: str,
    session_id: str | None,
    load_supported: bool,
    mcp_servers: Sequence[Any] | None = None,
) -> OpenedSession:
    """Reuse ``session_id`` when the agent can load it; otherwise open a new one.

    ``mcp_servers`` is the ACP server list for this session. ``None`` leaves the
    agent's own servers alone. When the agent rejects the list, the session opens
    once without it and ``mcp_degraded`` is set.
    """
    attached = None if mcp_servers is None else list(mcp_servers)
    if session_id and load_supported:
        try:
            kwargs: dict[str, Any] = {}
            if attached is not None:
                kwargs["mcp_servers"] = attached
            loaded = await conn.load_session(cwd=cwd, session_id=session_id, **kwargs)
        except Exception:
            logger.warning(
                "ACP load_session failed for %s; opening a new session",
                session_id,
                exc_info=True,
            )
        else:
            logger.info("ACP resumed session %s", session_id)
            current, available, default = mode_state(loaded)
            return OpenedSession(
                session_id,
                current_mode_id=current,
                available_modes=tuple(available),
                default_mode_id=default,
            )
    degraded = False
    try:
        kwargs = {} if attached is None else {"mcp_servers": attached}
        created = await conn.new_session(cwd=cwd, **kwargs)
    except Exception:
        if not attached:
            raise
        logger.warning("ACP new_session rejected MCP servers; opening the session without them", exc_info=True)
        created = await conn.new_session(cwd=cwd, mcp_servers=[])
        degraded = True
    new_id = str(getattr(created, "session_id", "") or "").strip()
    if not new_id:
        raise ACPSessionError("ACP new_session returned no session id")
    current, available, default = mode_state(created)
    return OpenedSession(
        new_id,
        mcp_degraded=degraded,
        current_mode_id=current,
        available_modes=tuple(available),
        default_mode_id=default,
    )
