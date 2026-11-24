"""Plugin manifest (``plugin.yaml``)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

PluginKind = Literal["tool", "skill", "hook", "channel"]

PLUGIN_KINDS = ("tool", "skill", "hook", "channel")

# Capability names: the four built-in kinds, or a name a host application
# defines (for example Octop's ``server``). The harness only enforces the
# built-in ones; hosts read the rest from ``LoadedPlugin.provided``.
_CAPABILITY_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class PluginManifest:
    id: str
    version: str
    name: str
    kind: PluginKind
    entry: str
    description: str = ""
    requires: tuple[str, ...] = field(default_factory=tuple)
    capabilities: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.capabilities:
            object.__setattr__(self, "capabilities", (self.kind,))

    def has(self, capability: str) -> bool:
        """Whether the manifest declares ``capability``."""
        return capability in self.capabilities

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginManifest:
        plugin_id = str(data.get("id") or "").strip()
        version = str(data.get("version") or "").strip()
        if not plugin_id or not version:
            raise ValueError("plugin.yaml requires id and version")

        capabilities = _parse_capabilities(data.get("capabilities"))
        raw_kind = str(data.get("kind") or "").strip().lower()
        # Without ``kind``, the first built-in capability is the display kind.
        kind = raw_kind or next((c for c in capabilities if c in PLUGIN_KINDS), "tool")
        if kind not in PLUGIN_KINDS:
            raise ValueError(f"unsupported plugin kind: {kind!r}")
        if not capabilities:
            capabilities = (kind,)
        elif raw_kind and kind not in capabilities:
            raise ValueError(f"kind {kind!r} must also be listed in capabilities")

        entry = str(data.get("entry") or "main.py").strip()
        name = str(data.get("name") or plugin_id).strip()
        description = str(data.get("description") or "").strip()

        raw_requires = data.get("requires") or []
        if not isinstance(raw_requires, list):
            raise ValueError("requires must be a list of pip requirement strings")
        requires = tuple(str(r).strip() for r in raw_requires if str(r).strip())

        return cls(
            id=plugin_id,
            version=version,
            name=name,
            kind=kind,  # type: ignore[arg-type]
            entry=entry,
            description=description,
            requires=requires,
            capabilities=capabilities,
        )

    @classmethod
    def load(cls, path: Path) -> PluginManifest:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid plugin manifest: {path}")
        return cls.from_dict(data)


def _parse_capabilities(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not raw:
        raise ValueError("capabilities must be a non-empty list of names")
    out: list[str] = []
    for item in raw:
        name = str(item).strip().lower()
        if not _CAPABILITY_RE.match(name):
            raise ValueError(f"invalid capability name: {item!r}")
        if name not in out:
            out.append(name)
    return tuple(out)
