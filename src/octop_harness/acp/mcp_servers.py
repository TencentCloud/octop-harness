"""Convert harness MCP specs into the server list an ACP session accepts.

Hosts keep the same dict they already give the in-process tool loader
(``transport``, ``url``, ``headers``, ``command``, ``args``, ``env``). A managed
turn passes that dict through ``configurable["acp_mcp_servers"]``. Entries with
no transport are in-process placeholders and are skipped.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def acp_mcp_servers_from_specs(specs: Mapping[str, Any] | None) -> list[Any]:
    """Return ACP servers for ``specs``, sorted by name."""
    if not specs:
        return []
    servers: list[Any] = []
    for name in sorted(specs):
        spec = specs[name]
        if not isinstance(spec, dict):
            continue
        server = _one(str(name), spec)
        if server is not None:
            servers.append(server)
    return servers


def mcp_fingerprint(servers: Sequence[Any]) -> str:
    """Stable identity for one ACP server list."""
    payload = [
        server.model_dump(mode="json", exclude_none=True) for server in sorted(servers, key=lambda item: str(item.name))
    ]
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _header_list(raw: Any) -> list[Any]:
    from acp.schema import HttpHeader

    if isinstance(raw, dict):
        pairs = [(str(key), str(value)) for key, value in raw.items() if str(key).strip()]
    elif isinstance(raw, list):
        pairs = []
        for item in raw:
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("key") or "").strip()
                if name:
                    pairs.append((name, str(item.get("value") or "")))
    else:
        pairs = []
    return [HttpHeader(name=name, value=value) for name, value in pairs]


def _env_list(raw: Any) -> list[Any]:
    from acp.schema import EnvVariable

    if not isinstance(raw, dict):
        return []
    return [EnvVariable(name=str(key), value=str(value)) for key, value in raw.items() if str(key).strip()]


def _args(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def _one(name: str, spec: dict[str, Any]) -> Any | None:
    from acp.schema import HttpMcpServer, McpServerStdio, SseMcpServer

    label = name.strip()
    if not label:
        return None
    transport = str(spec.get("transport") or "").strip().lower()
    if transport in {"http", "streamable_http"}:
        url = str(spec.get("url") or "").strip()
        if not url:
            return None
        return HttpMcpServer(type="http", name=label, url=url, headers=_header_list(spec.get("headers")))
    if transport == "sse":
        url = str(spec.get("url") or "").strip()
        if not url:
            return None
        return SseMcpServer(type="sse", name=label, url=url, headers=_header_list(spec.get("headers")))
    if transport == "stdio":
        command = str(spec.get("command") or "").strip()
        if not command:
            return None
        from octop_harness.runtime_env import merge_host_stdio_env

        raw_env = spec.get("env") if isinstance(spec.get("env"), dict) else {}
        return McpServerStdio(
            name=label,
            command=command,
            args=_args(spec.get("args")),
            env=_env_list(merge_host_stdio_env(raw_env)),
        )
    return None
