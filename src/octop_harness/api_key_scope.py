"""Per-call API-key override carried outside ``RunnableConfig``.

``ChatRequest.configurable["api_key_override"]`` is the documented seam for
per-call credential overrides (e.g. per-user gateway keys). That entry must
never reach ``RunnableConfig.configurable``: langgraph copies configurable
values into checkpoint metadata, which is persisted to ``checkpoints.sqlite``.

Instead :meth:`ChatRequest.to_runnable_config` drops the key from the run
config and :class:`~octop_harness.agent.HarnessAgent` binds it here at each
entry point; :mod:`octop_harness.middleware.model_router` reads it from the
scope and passes it to the factory explicitly.

The binding is task-local and sticky: every ``HarnessAgent`` entry point
rebinds (including to ``None``), so sequential invocations in one task never
serve a stale key. Child tasks created mid-invocation inherit the binding.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

API_KEY_OVERRIDE_KEY = "api_key_override"

_CURRENT_API_KEY_OVERRIDE: ContextVar[str | None] = ContextVar("harness_api_key_override", default=None)


def normalize_api_key_override(value: object) -> str | None:
    """Return *value* when it is a non-empty string, else ``None``."""
    return value if isinstance(value, str) and value else None


def api_key_override_from(configurable: Mapping[str, Any] | None) -> str | None:
    """Read the key out of a request ``configurable`` mapping without mutating it."""
    return normalize_api_key_override((configurable or {}).get(API_KEY_OVERRIDE_KEY))


def take_api_key_override(configurable: MutableMapping[str, Any]) -> str | None:
    """Pop the key out of *configurable* so it cannot ride into ``RunnableConfig``."""
    return normalize_api_key_override(configurable.pop(API_KEY_OVERRIDE_KEY, None))


def current_api_key_override() -> str | None:
    """The per-call key bound to the current task, if any."""
    return _CURRENT_API_KEY_OVERRIDE.get()


def bind_api_key_override(value: object) -> None:
    """Bind *value* for the current task until rebound (``None`` clears)."""
    _CURRENT_API_KEY_OVERRIDE.set(normalize_api_key_override(value))


@contextmanager
def api_key_override_scope(value: object) -> Iterator[None]:
    """Bind *value* for the duration of the block, restoring the prior value."""
    token = _CURRENT_API_KEY_OVERRIDE.set(normalize_api_key_override(value))
    try:
        yield
    finally:
        _CURRENT_API_KEY_OVERRIDE.reset(token)


__all__ = [
    "API_KEY_OVERRIDE_KEY",
    "api_key_override_from",
    "api_key_override_scope",
    "bind_api_key_override",
    "current_api_key_override",
    "normalize_api_key_override",
    "take_api_key_override",
]
