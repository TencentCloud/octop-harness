"""Tool name/path helpers shared by the filesystem middlewares."""

from __future__ import annotations

from typing import Any

from langgraph.prebuilt.tool_node import ToolCallRequest

__all__ = ["path_from_tool_args", "tool_base_name", "tool_call_meta"]


def tool_base_name(name: str) -> str:
    trimmed = name.strip()
    slash = trimmed.rfind("/")
    return trimmed[slash + 1 :] if slash >= 0 else trimmed


def path_from_tool_args(tool_name: str, params: dict[str, Any]) -> str | None:
    base = tool_base_name(tool_name)
    if base in {"read_file", "write_file", "append_file", "edit_file", "delete"}:
        for key in ("file_path", "path"):
            raw = params.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        return None
    if base in {"ls", "glob", "grep"}:
        raw = params.get("path")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def tool_call_meta(request: ToolCallRequest) -> tuple[str, str]:
    tool_call = request.tool_call
    return str(tool_call.get("name") or ""), str(tool_call.get("id") or "")
