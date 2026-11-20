"""Per-turn file write checkpoints (before/after snapshots).

Within a drain window (one agent turn) each path keeps its FIRST
before-content and LAST after-content, so a file edited several times in a
turn rewinds to its pre-turn state. ``before=None`` means the file did not
exist (created this window); ``after=None`` means it was removed.

Entries are keyed by invocation scope (thread id) because one agent can run
concurrent invocations; a drain always returns exactly the writes of the
scope it names, never another turn's in-flight changes. Snapshots carry a
capture sequence so parallel tool calls cannot let an intermediate "before"
win over the pre-turn one.

The tracker is drained by the host (Octop) at turn end and persisted there;
the harness only captures at the write point. Retention is bounded so hosts
that never drain cannot accumulate snapshots without limit.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class FileChange:
    path: str
    before: str | None
    after: str | None
    # Capture-order sequence numbers: ``before`` must come from the earliest
    # start, ``after`` from the latest completion, even when parallel tool
    # calls complete out of order.
    seq_first: int = 0
    seq_last: int = 0


@dataclass
class FileChangeTracker:
    _pending: dict[str, dict[str, FileChange]] = field(default_factory=dict)
    _skipped: dict[str, dict[str, None]] = field(default_factory=dict)
    # Scopes whose window failed closed (too many unusable snapshots): no
    # further records, because their pre-turn states are partly unknown.
    _unrewindable_scopes: set[str] = field(default_factory=set)
    _scope_chars: dict[str, int] = field(default_factory=dict)
    _total_chars: int = 0
    _last_touch: dict[str, float] = field(default_factory=dict)
    # Sync and async tool paths can run on different threads; re-entrant
    # because eviction may drop scopes while a record is in flight.
    _lock: threading.RLock = field(default_factory=threading.RLock)
    # Monotonic capture order for before/after contents: owned here so the
    # sequence survives guard middleware rebuilds (graph recompiles).
    _capture_seq: Iterator[int] = field(default_factory=itertools.count)

    # Bounds for hosts that never drain: eviction drops the oldest captured
    # path (insertion order) or the idlest scope, which only degrades rewind
    # coverage for abandoned windows — never for the active one.
    _MAX_ENTRIES_PER_SCOPE = 128
    _MAX_SNAPSHOT_CHARS_PER_SCOPE = 64 * 1024 * 1024
    # Global budget across scopes: without it the effective bound would be
    # _MAX_SCOPES x _MAX_SNAPSHOT_CHARS_PER_SCOPE.
    _MAX_SNAPSHOT_CHARS_TOTAL = 256 * 1024 * 1024
    _MAX_SKIPPED_PER_SCOPE = 512
    _MAX_SCOPES = 256
    _SCOPE_IDLE_SECONDS = 3600.0

    def record(
        self,
        path: str,
        before: str | None,
        after: str | None,
        scope: str = "",
        *,
        seq_before: int = 0,
        seq_after: int = 0,
    ) -> None:
        with self._lock:
            if scope in self._unrewindable_scopes:
                return
            if path in self._skipped.get(scope, {}):
                # Pre-turn state unknown for this path: a pending entry would
                # restore an intermediate state on rewind, so keep nothing.
                self._drop_pending(path, scope)
                self._touch(scope)
                self._evict_overflow(scope)
                return
            entries = self._pending.setdefault(scope, {})
            existing = entries.get(path)
            if existing is None:
                entries[path] = FileChange(
                    path=path,
                    before=before,
                    after=after,
                    seq_first=seq_before,
                    seq_last=seq_after,
                )
                self._add_scope_chars(scope, len(before or "") + len(after or ""))
            else:
                # Keep the earliest before and the latest after by capture
                # order, not by whichever parallel call inserted first.
                if seq_before < existing.seq_first:
                    self._add_scope_chars(scope, len(before or "") - len(existing.before or ""))
                    existing.before = before
                    existing.seq_first = seq_before
                if seq_after >= existing.seq_last:
                    self._add_scope_chars(scope, len(after or "") - len(existing.after or ""))
                    existing.after = after
                    existing.seq_last = seq_after
            self._touch(scope)
            self._evict_overflow(scope)

    def rebind_scope(self, old: str, new: str) -> None:
        """Move a window's pending captures to another scope.

        Used when a resumed HITL turn re-keys the graph scope: the paused
        turn's before/after snapshots must keep applying to the resumed
        writes instead of being stranded under the old scope.
        """
        with self._lock:
            entries = self._pending.pop(old, None)
            self._total_chars -= self._scope_chars.pop(old, 0)
            if entries:
                target = self._pending.setdefault(new, {})
                self._scope_chars[new] = self._scope_chars.get(new, 0)
                for path, ch in entries.items():
                    existing = target.get(path)
                    if existing is None:
                        target[path] = ch
                        self._add_scope_chars(new, len(ch.before or "") + len(ch.after or ""))
                    else:
                        previous = len(existing.before or "") + len(existing.after or "")
                        if ch.seq_first < existing.seq_first:
                            existing.before = ch.before
                            existing.seq_first = ch.seq_first
                        if ch.seq_last >= existing.seq_last:
                            existing.after = ch.after
                            existing.seq_last = ch.seq_last
                        self._add_scope_chars(
                            new,
                            len(existing.before or "") + len(existing.after or "") - previous,
                        )
            skipped = self._skipped.pop(old, None)
            if skipped:
                self._skipped.setdefault(new, {}).update(skipped)
            if old in self._unrewindable_scopes:
                self._unrewindable_scopes.discard(old)
                self._unrewindable_scopes.add(new)
            self._last_touch.pop(old, None)
            self._touch(new)

    def unrewindable(self, path: str, scope: str = "") -> bool:
        """True when this path/window can never produce a usable record.

        Covers paths with an unusable snapshot and scopes that failed closed:
        callers can skip the whole capture (no before, no after read) since
        any record would be dropped anyway.
        """
        with self._lock:
            return path in self._skipped.get(scope, {}) or scope in self._unrewindable_scopes

    def next_seq(self) -> int:
        """Allocate the next capture-order sequence number."""
        return next(self._capture_seq)

    def mark_unrewindable(self, path: str, scope: str = "") -> None:
        """Forget this path for the drain window after an unusable snapshot."""
        with self._lock:
            skipped = self._skipped.setdefault(scope, {})
            if len(skipped) >= self._MAX_SKIPPED_PER_SCOPE:
                # Too many unknowns to reason about this window: fail closed
                # instead of silently recording wrong snapshots.
                self._fail_scope_closed(scope)
            else:
                skipped[path] = None
            self._drop_pending(path, scope)
            self._touch(scope)
            self._evict_overflow(scope)

    def before_known(self, path: str, scope: str = "") -> bool:
        """True when an earlier capture already pinned this path's before.

        The tracker keeps only the first before-content per path, so re-reading
        the file for a later write to the same path cannot improve the record.
        """
        with self._lock:
            return (
                path in self._pending.get(scope, {})
                or path in self._skipped.get(scope, {})
                or scope in self._unrewindable_scopes
            )

    def drain(self, scope: str = "") -> list[FileChange]:
        with self._lock:
            entries = self._pending.pop(scope, None)
            self._skipped.pop(scope, None)
            self._total_chars -= self._scope_chars.pop(scope, 0)
            self._unrewindable_scopes.discard(scope)
            self._last_touch.pop(scope, None)
        if not entries:
            return []
        # Net no-ops (rewritten with identical content, created then removed)
        # have nothing to rewind and would only show up as phantom changes.
        return [c for c in entries.values() if c.before != c.after]

    def discard(self, scope: str | None = None) -> None:
        """Drop undrained captures — one scope, or every scope when ``None``."""
        with self._lock:
            if scope is None:
                self._pending.clear()
                self._skipped.clear()
                self._unrewindable_scopes.clear()
                self._scope_chars.clear()
                self._total_chars = 0
                self._last_touch.clear()
            else:
                self._pending.pop(scope, None)
                self._skipped.pop(scope, None)
                self._unrewindable_scopes.discard(scope)
                self._total_chars -= self._scope_chars.pop(scope, 0)
                self._last_touch.pop(scope, None)

    def _drop_pending(self, path: str, scope: str) -> None:
        # Keep the per-scope size counter exact: an inflated counter would
        # trip the budget and evict live entries on every later write.
        dropped = self._pending.get(scope, {}).pop(path, None)
        if dropped is not None:
            self._add_scope_chars(scope, -(len(dropped.before or "") + len(dropped.after or "")))

    def _fail_scope_closed(self, scope: str) -> None:
        while len(self._unrewindable_scopes) >= self._MAX_SCOPES:
            self._unrewindable_scopes.discard(next(iter(self._unrewindable_scopes)))
        self._unrewindable_scopes.add(scope)
        self._pending.pop(scope, None)
        # record/before_known already short-circuit on the flag; release the
        # set so bounded retention still holds for hosts that never drain.
        self._skipped.pop(scope, None)
        self._total_chars -= self._scope_chars.pop(scope, 0)

    def _touch(self, scope: str) -> None:
        self._last_touch[scope] = time.monotonic()

    def _evict_overflow(self, scope: str) -> None:
        entries = self._pending.get(scope)
        if entries:
            while entries:
                over_entries = len(entries) > self._MAX_ENTRIES_PER_SCOPE
                over_chars = self._scope_chars.get(scope, 0) > self._MAX_SNAPSHOT_CHARS_PER_SCOPE
                if not over_entries and not over_chars:
                    break
                evicted = next(iter(entries))
                # The evicted path's pre-turn state is now unknown; suppress
                # later records for it instead of restoring an intermediate.
                skipped = self._skipped.setdefault(scope, {})
                if len(skipped) >= self._MAX_SKIPPED_PER_SCOPE:
                    self._fail_scope_closed(scope)
                    entries = self._pending.get(scope)
                    continue
                skipped[evicted] = None
                chars = entries.pop(evicted)
                self._add_scope_chars(scope, -(len(chars.before or "") + len(chars.after or "")))
        # Abandoned scopes (hosts that never drain) must not accumulate:
        # idle scopes go first; once a cap is exceeded with none idle, the
        # least-recently-touched other scope is evicted so a cycling host
        # still stays bounded. The current window is never a victim.
        while len(self._last_touch) > self._MAX_SCOPES or (self._total_chars > self._MAX_SNAPSHOT_CHARS_TOTAL):
            candidates = [s for s in self._last_touch if s != scope]
            if not candidates:
                break
            now = time.monotonic()
            idle = [s for s in candidates if now - self._last_touch[s] > self._SCOPE_IDLE_SECONDS]
            victim = min(idle or candidates, key=lambda s: self._last_touch[s])
            if idle:
                self.discard(victim)
            else:
                # Still possibly in flight: fail the window closed so later
                # records cannot start from an intermediate snapshot.
                self._fail_scope_closed(victim)
                self._last_touch.pop(victim, None)

    def _add_scope_chars(self, scope: str, delta: int) -> None:
        """Keep the per-scope and global char counters exact."""
        if not delta:
            return
        self._scope_chars[scope] = self._scope_chars.get(scope, 0) + delta
        self._total_chars += delta


__all__ = ["FileChange", "FileChangeTracker"]
