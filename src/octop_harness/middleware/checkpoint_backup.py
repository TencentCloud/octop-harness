"""Manage the backup taken before scheduled checkpoint pruning."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager, suppress
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Literal

logger = logging.getLogger(__name__)

BackupMode = Literal["first", "always", "off"]
_BACKUP_MODE_ENV = "OCTOP_CHECKPOINT_BACKUP_BEFORE_PRUNE"
_BACKUP_KEEP_DAYS_ENV = "OCTOP_CHECKPOINT_BACKUP_KEEP_DAYS"
_DAY_SECONDS = 24 * 60 * 60
_COPY_TIMEOUT_SECONDS = 5 * 60
_BACKUP_LOCKS: weakref.WeakValueDictionary[Path, threading.Lock] = weakref.WeakValueDictionary()
_BACKUP_LOCKS_GUARD = threading.Lock()
_UNWRITTEN_MARKERS: dict[Path, str] = {}
_ALWAYS_WARNINGS: dict[tuple[Path, str], float] = {}


@lru_cache(maxsize=32)
def _parse_backup_mode(value: str) -> BackupMode:
    if value in {"", "first"}:
        return "first"
    if value in {"always", "1", "true", "on", "yes"}:
        return "always"
    if value in {"0", "false", "off", "no"}:
        return "off"
    logger.warning("invalid %s=%r; using first", _BACKUP_MODE_ENV, value)
    return "first"


@lru_cache(maxsize=32)
def _parse_backup_keep_days(value: str) -> int:
    if not value:
        return 7
    try:
        days = int(value)
        if days >= 0:
            return days
    except ValueError:
        pass
    logger.warning("invalid %s=%r; using 7", _BACKUP_KEEP_DAYS_ENV, value)
    return 7


def checkpoint_backup_policy_from_env() -> tuple[BackupMode, int]:
    return (
        _parse_backup_mode(os.environ.get(_BACKUP_MODE_ENV, "").strip().lower()),
        _parse_backup_keep_days(os.environ.get(_BACKUP_KEEP_DAYS_ENV, "").strip()),
    )


def _backup_lock(path: Path) -> threading.Lock:
    # Different Memory instances in this process can share one SQLite file.
    with _BACKUP_LOCKS_GUARD:
        lock = _BACKUP_LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _BACKUP_LOCKS[path] = lock
        return lock


def _backup_path(path: Path) -> Path:
    return path.with_name(path.name + ".pre-prune.bak")


def _remove_file(path: Path, *, max_age_seconds: int | None = None) -> None:
    try:
        if max_age_seconds is None or time.time() - path.lstat().st_mtime > max_age_seconds:
            path.unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("could not remove checkpoint backup file %s", path, exc_info=True)


def expire_checkpoint_backups(path: Path, keep_days: int) -> None:
    backup = _backup_path(path)
    with _backup_lock(path):
        if keep_days:
            _remove_file(backup, max_age_seconds=keep_days * _DAY_SECONDS)
        # Leave recent temporary files alone in case another process is copying.
        prefix = backup.name + ".tmp-"
        try:
            for candidate in backup.parent.iterdir():
                if candidate.name.startswith(prefix):
                    _remove_file(candidate, max_age_seconds=_DAY_SECONDS)
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("could not inspect temporary checkpoint backups in %s", backup.parent, exc_info=True)


@contextmanager
def _temporary_backup_file(backup: Path) -> Iterator[Path]:
    fd, name = tempfile.mkstemp(dir=backup.parent, prefix=backup.name + ".tmp-")
    path = Path(name)
    try:
        os.close(fd)
        yield path
    finally:
        _remove_file(path)


def _backup_checkpoint_file(path: Path, backup: Path, keep_days: int) -> None:
    started = time.monotonic()

    def check_timeout(status: int, _remaining: int, _total: int) -> None:
        if status != sqlite3.SQLITE_DONE and time.monotonic() - started >= _COPY_TIMEOUT_SECONDS:
            raise TimeoutError("SQLite checkpoint backup copy exceeded five minutes")

    with _temporary_backup_file(backup) as temp:
        with (
            closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as source,
            closing(sqlite3.connect(temp)) as target,
        ):
            source.backup(target, pages=1024, progress=check_timeout)
            check = target.execute("PRAGMA quick_check").fetchone()
            if check != ("ok",):
                raise sqlite3.DatabaseError(f"checkpoint backup quick_check failed: {check!r}")
        size = temp.stat().st_size
        os.replace(temp, backup)
    logger.info(
        "checkpoint backup %s: %d bytes in %.3f seconds; %s",
        backup,
        size,
        time.monotonic() - started,
        f"expires after {keep_days} days" if keep_days else "automatic expiration disabled",
    )


def _required_backup_space(path: Path) -> int:
    size = path.stat().st_size
    with suppress(FileNotFoundError):
        size += path.with_name(path.name + "-wal").stat().st_size
    return size + (size + 9) // 10


def _warn_always(path: Path, reason: str, message: str, *args: object) -> None:
    key = (path, reason)
    now = time.monotonic()
    previous = _ALWAYS_WARNINGS.get(key)
    if previous is None or now - previous >= _DAY_SECONDS:
        _ALWAYS_WARNINGS[key] = now
        logger.warning(message, *args)


def _write_marker(backup: Path, marker: Path, record: str) -> None:
    try:
        with _temporary_backup_file(backup) as temp:
            temp.write_text(record, encoding="utf-8")
            os.replace(temp, marker)
    except OSError:
        # Even a small marker may be unwritable on a full disk. Remember the
        # result in this process and retry only the marker on a later pass.
        if marker not in _UNWRITTEN_MARKERS:
            logger.warning("could not write checkpoint backup marker %s", marker, exc_info=True)
        _UNWRITTEN_MARKERS[marker] = record
    else:
        _UNWRITTEN_MARKERS.pop(marker, None)


def _marker_exists(marker: Path) -> bool:
    try:
        return marker.exists()
    except OSError:
        # The backup attempt can still proceed if the marker cannot be read.
        return False


def prepare_checkpoint_backup(
    path: Path, mode: BackupMode, keep_days: int, *, write_dump: Callable[[BinaryIO], None] | None = None
) -> bool:
    """Return whether pruning may proceed after applying the configured backup policy."""
    if mode == "off":
        return True
    backup = _backup_path(path)
    marker = path.with_name(path.name + ".pre-prune.done")
    with _backup_lock(path):
        marked = _marker_exists(marker)
        if marked:
            _UNWRITTEN_MARKERS.pop(marker, None)
        elif record := _UNWRITTEN_MARKERS.get(marker):
            _write_marker(backup, marker, record)
            marked = True
        if mode == "first" and marked:
            return True

        result = (
            _attempt_backup(path, backup, mode, keep_days)
            if write_dump is None
            else _attempt_logical_backup(backup, mode, write_dump)
        )
        if not marked and (mode == "first" or result == "backed_up"):
            record = json.dumps({"completed_at": datetime.now(UTC).isoformat(), "result": result}) + "\n"
            _write_marker(backup, marker, record)
        return mode == "first" or result == "backed_up"


def _attempt_logical_backup(backup: Path, mode: BackupMode, write_dump: Callable[[BinaryIO], None]) -> str:
    try:
        backup.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with _temporary_backup_file(backup) as temp:
            with temp.open("wb") as output:
                write_dump(output)
            os.replace(temp, backup)
        logger.info("checkpoint backup %s: %d bytes", backup, backup.stat().st_size)
        return "backed_up"
    except Exception as exc:
        # Connection errors can contain details about the configured server.
        reason = type(exc).__name__
        message = "PostgreSQL checkpoint backup failed for %s (%s); %s"
        if mode == "first":
            logger.warning(message, backup, reason, "continuing without a backup")
        else:
            _warn_always(backup, "backup_failed", message, backup, reason, "skipping checkpoint pruning")
        return "failed_then_skipped"


def _attempt_backup(path: Path, backup: Path, mode: BackupMode, keep_days: int) -> str:
    try:
        required = _required_backup_space(path)
        free = shutil.disk_usage(path.parent).free
        if free >= required:
            _backup_checkpoint_file(path, backup, keep_days)
            return "backed_up"
    except Exception as exc:
        message = "checkpoint backup failed for %s (%s); %s"
        if mode == "first":
            logger.warning(message, path, exc, "continuing without a backup")
        else:
            _warn_always(path, "backup_failed", message, path, exc, "skipping checkpoint pruning")
        return "failed_then_skipped"

    message = "insufficient space for checkpoint backup %s (need %d bytes, have %d bytes); %s"
    if mode == "first":
        logger.warning(message, path, required, free, "continuing without a backup")
    else:
        _warn_always(path, "low_space", message, path, required, free, "skipping checkpoint pruning")
    return "skipped_low_space"
