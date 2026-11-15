"""Per-call tool environment carried outside ``RunnableConfig``.

``ChatRequest.configurable["tool_env"]`` is the documented seam for per-call
credential overrides of built-in tools (e.g. per-user web-search keys). That
entry must never reach ``RunnableConfig.configurable``: langgraph copies
configurable values into checkpoint metadata, which is persisted to
``checkpoints.sqlite``.

Instead :meth:`ChatRequest.to_runnable_config` drops the key from the run
config and :class:`~octop_harness.agent.HarnessAgent` binds it here at each
entry point; built-in tools read it through :func:`current_tool_env`, which
falls back to ``os.environ`` so a global administrator key keeps working.

The binding is task-local and sticky: every ``HarnessAgent`` entry point
rebinds (including to an empty mapping), so sequential invocations in one task
never serve stale values. Child tasks created mid-invocation inherit the
binding.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any

TOOL_ENV_KEY = "tool_env"

# Defensive bounds so an abnormal caller cannot pin an unbounded mapping in a
# context or in request memory. Values are secrets: never log them.
MAX_TOOL_ENV_ENTRIES = 16
MAX_TOOL_ENV_NAME_LENGTH = 256
MAX_TOOL_ENV_VALUE_LENGTH = 4096

_EMPTY_ENV: Mapping[str, str] = MappingProxyType({})
_CURRENT_TOOL_ENV: ContextVar[Mapping[str, str]] = ContextVar("harness_tool_env", default=_EMPTY_ENV)


def normalize_tool_env(value: object) -> dict[str, str]:
    """Return a clean ``str -> str`` mapping from *value*.

    Names and values are stripped: a whitespace-only entry is a missing
    credential, and letting it through would shadow the ``os.environ`` fallback
    with a value that can only fail authentication. Non-mapping input,
    non-string / empty names, non-string / empty values, entries longer than
    :data:`MAX_TOOL_ENV_NAME_LENGTH` / :data:`MAX_TOOL_ENV_VALUE_LENGTH`, and
    entries beyond :data:`MAX_TOOL_ENV_ENTRIES` are dropped.
    """
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, str] = {}
    for name, raw in value.items():
        if not isinstance(name, str) or not isinstance(raw, str):
            continue
        key, val = name.strip(), raw.strip()
        if not key or not val:
            continue
        if len(key) > MAX_TOOL_ENV_NAME_LENGTH or len(val) > MAX_TOOL_ENV_VALUE_LENGTH:
            continue
        out[key] = val
        if len(out) >= MAX_TOOL_ENV_ENTRIES:
            break
    return out


def tool_env_from(configurable: Mapping[str, Any] | None) -> dict[str, str]:
    """Read the mapping out of a request ``configurable`` without mutating it."""
    return normalize_tool_env((configurable or {}).get(TOOL_ENV_KEY))


def current_tool_env(name: str, default: str = "") -> str:
    """Return the per-call value for *name*, falling back to the process env.

    Priority: current-task binding > ``os.environ`` > *default*. Empty values
    are treated as unset at both levels.
    """
    scoped = _CURRENT_TOOL_ENV.get().get(name)
    if scoped:
        return scoped
    return os.getenv(name) or default


def bind_tool_env(value: object) -> None:
    """Bind *value* for the current task until rebound (empty mapping clears)."""
    _CURRENT_TOOL_ENV.set(normalize_tool_env(value))


@contextmanager
def tool_env_scope(value: object) -> Iterator[None]:
    """Bind *value* for the duration of the block, restoring the prior value."""
    token = _CURRENT_TOOL_ENV.set(normalize_tool_env(value))
    try:
        yield
    finally:
        _CURRENT_TOOL_ENV.reset(token)


__all__ = [
    "MAX_TOOL_ENV_ENTRIES",
    "MAX_TOOL_ENV_NAME_LENGTH",
    "MAX_TOOL_ENV_VALUE_LENGTH",
    "TOOL_ENV_KEY",
    "bind_tool_env",
    "current_tool_env",
    "normalize_tool_env",
    "tool_env_from",
    "tool_env_scope",
]
