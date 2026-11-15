"""Persist the ACP session id for one chat thread so a later process can resume it."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from octop_harness.acp.models import ACPSessionError

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


async def resume_or_create_session(
    conn: Any,
    *,
    cwd: str,
    session_id: str | None,
    load_supported: bool,
) -> str:
    """Reuse ``session_id`` when the agent can load it; otherwise open a new one."""
    if session_id and load_supported:
        try:
            await conn.load_session(cwd=cwd, session_id=session_id)
        except Exception:
            logger.warning(
                "ACP load_session failed for %s; opening a new session",
                session_id,
                exc_info=True,
            )
        else:
            logger.info("ACP resumed session %s", session_id)
            return session_id
    created = await conn.new_session(cwd=cwd)
    new_id = str(getattr(created, "session_id", "") or "").strip()
    if not new_id:
        raise ACPSessionError("ACP new_session returned no session id")
    return new_id
