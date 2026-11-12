"""Scheduled checkpoint backup behavior against disposable SQLite databases."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint
from octop_memory import Memory

from octop_harness.middleware import checkpoint_backup
from octop_harness.middleware.memory import _prune_checkpoints_if_configured, checkpoint_retention_policy

_MODE = "OCTOP_CHECKPOINT_BACKUP_BEFORE_PRUNE"
_KEEP_DAYS = "OCTOP_CHECKPOINT_BACKUP_KEEP_DAYS"


def test_postgres_logical_backup_uses_existing_modes_and_expiration(tmp_path):
    path = tmp_path / "pg" / "checkpoint.sql"
    backup = path.with_suffix(".sql.pre-prune.bak")
    marker = path.with_suffix(".sql.pre-prune.done")
    attempts = []

    def write(output):
        attempts.append(1)
        output.write(b"complete dump")

    def fail(output):
        attempts.append(1)
        output.write(b"partial dump")
        raise OSError("injected failure")

    assert checkpoint_backup.prepare_checkpoint_backup(path, "off", 7, write_dump=write)
    assert not attempts
    assert checkpoint_backup.prepare_checkpoint_backup(path, "first", 7, write_dump=write)
    assert checkpoint_backup.prepare_checkpoint_backup(path, "first", 7, write_dump=fail)
    assert len(attempts) == 1
    assert not checkpoint_backup.prepare_checkpoint_backup(path, "always", 7, write_dump=fail)
    assert backup.read_bytes() == b"complete dump"
    assert _marker_result(marker) == "backed_up"
    os.utime(backup, (time.time() - 8 * 86400,) * 2)
    checkpoint_backup.expire_checkpoint_backups(path, 7)
    assert not backup.exists()
    assert checkpoint_backup.prepare_checkpoint_backup(path, "first", 7, write_dump=write)
    assert len(attempts) == 2
    assert checkpoint_backup.prepare_checkpoint_backup(path, "always", 7, write_dump=write)
    assert backup.read_bytes() == b"complete dump"


def _append(memory: Memory, count: int = 1) -> None:
    config: dict[str, Any] = {"configurable": {"thread_id": "thread", "checkpoint_ns": ""}}
    latest = memory.get_tuple(config)
    if latest:
        config = latest.config
    for step in range(count):
        config = memory.put(config, create_checkpoint(empty_checkpoint(), None, step), {}, {})


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Memory, Path, Path, Path]]:
    monkeypatch.delenv(_MODE, raising=False)
    monkeypatch.delenv(_KEEP_DAYS, raising=False)
    monkeypatch.setenv("OCTOP_CHECKPOINT_RETENTION", "1")
    monkeypatch.setenv("OCTOP_CHECKPOINT_KEEP_LAST", "3")
    monkeypatch.delenv("OCTOP_CHECKPOINT_KEEP_DAYS", raising=False)
    monkeypatch.delenv("OCTOP_CHECKPOINT_VACUUM", raising=False)
    monkeypatch.setattr(checkpoint_backup.shutil, "disk_usage", lambda _path: SimpleNamespace(free=2**40))
    path = tmp_path / "memory.sqlite"
    memory = Memory("test", backend_config={"db_path": str(path)})
    try:
        _append(memory, 6)
        yield memory, path, path.with_suffix(".sqlite.pre-prune.bak"), path.with_suffix(".sqlite.pre-prune.done")
    finally:
        memory._checkpointer.conn.close()
        memory.backend.close()


def _marker_result(marker: Path) -> str:
    record = json.loads(marker.read_text(encoding="utf-8"))
    assert datetime.fromisoformat(record["completed_at"]).utcoffset() is not None
    return str(record["result"])


def _checkpoint_count(path: Path) -> int:
    with closing(sqlite3.connect(path)) as conn:
        return int(conn.execute("SELECT count(*) FROM checkpoints").fetchone()[0])


def test_first_backup_is_complete_and_is_never_rewritten(store):
    memory, path, backup, marker = store
    before = _checkpoint_count(path)
    _prune_checkpoints_if_configured(memory)
    assert _checkpoint_count(path) == 3
    assert _marker_result(marker) == "backed_up"
    assert _checkpoint_count(backup) == before
    with closing(sqlite3.connect(backup)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]

    os.utime(backup, (time.time() - 60,) * 2)
    mtime = backup.stat().st_mtime_ns
    content = backup.read_bytes()
    _append(memory)
    assert _checkpoint_count(path) == 4
    _prune_checkpoints_if_configured(memory)
    assert _checkpoint_count(path) == 3
    assert backup.stat().st_mtime_ns == mtime
    assert backup.read_bytes() == content


def test_expiration_runs_without_prunable_checkpoints_and_preserves_marker(store):
    memory, path, backup, marker = store
    _prune_checkpoints_if_configured(memory)
    stale = path.parent / (backup.name + ".tmp-abandoned")
    recent = path.parent / (backup.name + ".tmp-recent")
    unrelated = path.parent / "operator-backup.sqlite"
    for file in (stale, recent, unrelated):
        file.touch()
    for file in (backup, stale, unrelated):
        os.utime(file, (time.time() - 8 * 86400,) * 2)
    _prune_checkpoints_if_configured(memory)
    assert not backup.exists()
    assert not stale.exists()
    assert recent.exists() and unrelated.exists()
    assert _marker_result(marker) == "backed_up"
    _append(memory)
    _prune_checkpoints_if_configured(memory)
    assert _checkpoint_count(path) == 3
    assert not backup.exists()

    marker.unlink()
    _append(memory)
    _prune_checkpoints_if_configured(memory)
    assert backup.exists()
    assert _marker_result(marker) == "backed_up"


def test_keep_days_zero_keeps_old_backup_even_with_retention_disabled(store, monkeypatch):
    memory, _path, backup, _marker = store
    _prune_checkpoints_if_configured(memory)
    os.utime(backup, (time.time() - 8 * 86400,) * 2)
    monkeypatch.setenv(_KEEP_DAYS, "0")
    monkeypatch.setenv("OCTOP_CHECKPOINT_RETENTION", "0")
    _prune_checkpoints_if_configured(memory)
    assert backup.exists()
    monkeypatch.setenv(_KEEP_DAYS, "7")
    _prune_checkpoints_if_configured(memory)
    assert not backup.exists()


def test_first_low_space_warns_once_and_still_prunes(store, monkeypatch, caplog):
    memory, path, backup, marker = store
    checks = []

    def disk_usage(directory):
        checks.append(directory)
        return SimpleNamespace(free=1)

    monkeypatch.setattr(checkpoint_backup.shutil, "disk_usage", disk_usage)
    size = path.stat().st_size + path.with_suffix(".sqlite-wal").stat().st_size
    needed = size + (size + 9) // 10
    for _ in range(2):
        _prune_checkpoints_if_configured(memory)
        assert _checkpoint_count(path) == 3
        _append(memory)
    assert checks == [path.parent.resolve()]
    assert not backup.exists()
    assert _marker_result(marker) == "skipped_low_space"
    warnings = [record.message for record in caplog.records if "insufficient space" in record.message]
    assert len(warnings) == 1
    assert f"need {needed} bytes, have 1 bytes" in warnings[0]


def test_first_failed_copy_leaves_no_backup_or_temporary_file_and_still_prunes(store, monkeypatch, caplog):
    memory, path, backup, marker = store
    connect = sqlite3.connect
    copies = []

    class FailingCopy(sqlite3.Connection):
        def backup(self, target, *, pages, progress):
            copies.append(pages)
            target.execute("CREATE TABLE partial_copy (data BLOB)")
            raise sqlite3.OperationalError("simulated write failure")

    def connect_for_backup(database, **kwargs):
        if kwargs.get("uri"):
            kwargs["factory"] = FailingCopy
        return connect(database, **kwargs)

    monkeypatch.setattr(checkpoint_backup.sqlite3, "connect", connect_for_backup)
    for _ in range(2):
        _prune_checkpoints_if_configured(memory)
        assert _checkpoint_count(path) == 3
        _append(memory)
    assert copies == [1024]
    assert not backup.exists()
    assert not list(path.parent.glob(backup.name + ".tmp-*"))
    assert _marker_result(marker) == "failed_then_skipped"
    assert caplog.text.count("simulated write failure") == 1


def test_first_does_not_repeat_backup_when_marker_cannot_be_written(store, monkeypatch, caplog):
    memory, path, backup, marker = store
    replace = os.replace

    def full_disk_on_marker(source, destination):
        if destination.name == marker.name:
            raise OSError("no space left for marker")
        replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint_backup.os, "replace", full_disk_on_marker)
        _prune_checkpoints_if_configured(memory)
        assert _checkpoint_count(path) == 3
        assert not marker.exists()
        _append(memory)
        _prune_checkpoints_if_configured(memory)
        assert _checkpoint_count(path) == 3
        assert _checkpoint_count(backup) == 6
    assert caplog.text.count("could not write checkpoint backup marker") == 1
    _append(memory)
    _prune_checkpoints_if_configured(memory)
    assert _marker_result(marker) == "backed_up"
    assert _checkpoint_count(backup) == 6


def test_always_replaces_backup_each_time(store, monkeypatch):
    memory, path, backup, _marker = store
    monkeypatch.setenv(_MODE, "always")
    _prune_checkpoints_if_configured(memory)
    assert _checkpoint_count(backup) == 6
    _append(memory)
    _prune_checkpoints_if_configured(memory)
    assert _checkpoint_count(path) == 3
    assert _checkpoint_count(backup) == 4


def test_always_keeps_previous_backup_when_quick_check_fails(store, monkeypatch, caplog):
    memory, path, backup, _marker = store
    monkeypatch.setenv(_MODE, "always")
    _prune_checkpoints_if_configured(memory)
    previous = (backup.stat().st_mtime_ns, backup.read_bytes())
    _append(memory)
    connect = sqlite3.connect

    class InvalidBackup(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql == "PRAGMA quick_check":
                return super().execute("SELECT 'simulated corrupt page'")
            return super().execute(sql, parameters)

    def connect_for_backup(database, **kwargs):
        if isinstance(database, Path) and database.name.startswith(backup.name + ".tmp-"):
            kwargs["factory"] = InvalidBackup
        return connect(database, **kwargs)

    monkeypatch.setattr(checkpoint_backup.sqlite3, "connect", connect_for_backup)
    for _ in range(2):
        _prune_checkpoints_if_configured(memory)
    assert _checkpoint_count(path) == 4
    assert (backup.stat().st_mtime_ns, backup.read_bytes()) == previous
    assert not list(path.parent.glob(backup.name + ".tmp-*"))
    assert caplog.text.count("simulated corrupt page") == 1


def test_always_low_space_retries_and_warns_at_most_once_per_day(store, monkeypatch, caplog):
    memory, path, backup, marker = store
    monkeypatch.setenv(_MODE, "always")
    now = 1.0
    free = 1
    checks = []

    def disk_usage(directory):
        checks.append(directory)
        return SimpleNamespace(free=free)

    monkeypatch.setattr(checkpoint_backup.time, "monotonic", lambda: now)
    monkeypatch.setattr(checkpoint_backup.shutil, "disk_usage", disk_usage)
    _prune_checkpoints_if_configured(memory)
    now += 3600
    _prune_checkpoints_if_configured(memory)
    assert caplog.text.count("insufficient space for checkpoint backup") == 1
    assert _checkpoint_count(path) == 6
    assert not backup.exists() and not marker.exists()
    now += 86400
    _prune_checkpoints_if_configured(memory)
    assert caplog.text.count("insufficient space for checkpoint backup") == 2
    assert _checkpoint_count(path) == 6
    free = 2**40
    _prune_checkpoints_if_configured(memory)
    assert _checkpoint_count(path) == 3
    assert backup.exists()
    assert len(checks) == 4


def test_off_prunes_without_backup_or_marker(store, monkeypatch):
    memory, path, backup, marker = store
    monkeypatch.setenv(_MODE, "off")
    _prune_checkpoints_if_configured(memory)
    assert _checkpoint_count(path) == 3
    assert not backup.exists() and not marker.exists()


@pytest.mark.parametrize(
    ("value", "mode", "warning"),
    [
        (None, "first", False),
        ("first", "first", False),
        ("always", "always", False),
        ("1", "always", False),
        (" TRUE ", "always", False),
        ("on", "always", False),
        ("yes", "always", False),
        ("0", "off", False),
        ("false", "off", False),
        ("off", "off", False),
        ("no", "off", False),
        ("unrecognized", "first", True),
    ],
)
def test_policy_mode(value, mode, warning, monkeypatch, caplog):
    if value is None:
        monkeypatch.delenv(_MODE, raising=False)
    else:
        monkeypatch.setenv(_MODE, value)
    monkeypatch.delenv(_KEEP_DAYS, raising=False)
    assert checkpoint_retention_policy()["backup_before_prune"] == mode
    assert checkpoint_retention_policy()["backup_before_prune"] == mode
    assert checkpoint_retention_policy()["backup_keep_days"] == 7
    assert caplog.text.count(_MODE) == int(warning)


@pytest.mark.parametrize(("value", "expected"), [("0", 0), (" 14 ", 14), ("-1", 7), ("invalid", 7)])
def test_policy_keep_days(value, expected, monkeypatch):
    monkeypatch.setenv(_KEEP_DAYS, value)
    assert checkpoint_retention_policy()["backup_keep_days"] == expected
