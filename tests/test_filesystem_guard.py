"""Tests for FilesystemGuardMiddleware (execution-backend path policy)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from deepagents.middleware.filesystem import (
    ExecuteSchema,
    FilesystemPermission,
    GlobSchema,
    ReadFileSchema,
    ReadVideoFileSchema,
)
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt.tool_node import ToolCallRequest

from octop_harness.middleware.filesystem_guard import (
    FilesystemGuardMiddleware,
    filesystem_guard_block_reason,
    unbounded_glob_block_reason,
)


def test_basename_glob_requires_an_explicit_path() -> None:
    for pattern in ("*", "**", "**/*", " * ", "*.md"):
        reason = unbounded_glob_block_reason("glob", {"pattern": pattern})
        assert reason is not None
        assert "explicit path" in reason
    assert unbounded_glob_block_reason("glob", {"pattern": "*", "path": "/"}) is None
    assert unbounded_glob_block_reason("glob", {"pattern": "*", "path": "//"}) is None
    assert unbounded_glob_block_reason("glob", {"pattern": "**/*.md"}) is None
    assert unbounded_glob_block_reason("glob", {"pattern": "src/**/*.py"}) is None


def test_allows_glob_anchored_to_a_directory() -> None:
    assert unbounded_glob_block_reason("glob", {"pattern": "*", "path": "."}) is None
    assert unbounded_glob_block_reason("glob", {"pattern": "*.md", "path": "."}) is None
    assert unbounded_glob_block_reason("ls", {"pattern": "*"}) is None


def test_relative_glob_stays_in_workspace_and_root_is_explicit(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    def resolve_path(path: str) -> str:
        if path == ".":
            return str(workspace_root)
        if path == "/":
            return str(tmp_path)
        return str((workspace_root / path).resolve())

    workspace = SimpleNamespace(resolve_path=resolve_path)
    assert unbounded_glob_block_reason("glob", {"pattern": "*", "path": "."}, workspace=workspace) is None
    assert unbounded_glob_block_reason("glob", {"pattern": "*.md", "path": "."}, workspace=workspace) is None
    assert unbounded_glob_block_reason("glob", {"pattern": "*", "path": "/"}, workspace=workspace) is None
    escaped = unbounded_glob_block_reason("glob", {"pattern": "*", "path": "../secret"}, workspace=workspace)
    assert escaped is not None
    assert "leaves the workspace" in escaped


def test_guard_rejects_root_glob_before_the_tool_runs() -> None:
    request = ToolCallRequest(
        tool_call={"name": "glob", "args": {"pattern": "*"}, "id": "call-1"},
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )

    def handler(_request: ToolCallRequest) -> None:
        raise AssertionError("glob should not run")

    result = FilesystemGuardMiddleware().wrap_tool_call(request, handler)
    assert result.status == "error"
    assert "explicit path" in str(result.content)


def test_relative_file_path_defaults_to_workspace_before_upstream_tool() -> None:
    request = ToolCallRequest(
        tool_call={"name": "read_file", "args": {"file_path": "docs/guide.md"}, "id": "call-1"},
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    seen: dict[str, object] = {}

    def handler(item: ToolCallRequest) -> ToolMessage:
        seen.update(item.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="call-1")

    FilesystemGuardMiddleware(workspace_dir="/agent/workspace").wrap_tool_call(request, handler)
    assert seen == {"file_path": "/agent/workspace/docs/guide.md"}


def test_base_dir_resolves_a_document_relative_file_without_persisting_cwd() -> None:
    request = ToolCallRequest(
        tool_call={
            "name": "read_file",
            "args": {"file_path": "references/sources.md", "base_dir": "/agent/.octop/skills/daily-hot-news"},
            "id": "call-1",
        },
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    seen: dict[str, object] = {}

    def handler(item: ToolCallRequest) -> ToolMessage:
        seen.update(item.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="call-1")

    FilesystemGuardMiddleware(workspace_dir="/agent/workspace").wrap_tool_call(request, handler)
    assert seen == {"file_path": "/agent/.octop/skills/daily-hot-news/references/sources.md"}


def test_glob_without_a_root_stays_in_workspace() -> None:
    request = ToolCallRequest(
        tool_call={"name": "glob", "args": {"pattern": "**/fetch_dailyhot.mjs"}, "id": "call-1"},
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    seen: dict[str, object] = {}

    def handler(item: ToolCallRequest) -> ToolMessage:
        seen.update(item.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="call-1")

    FilesystemGuardMiddleware(workspace_dir="/agent/workspace").wrap_tool_call(request, handler)
    assert seen == {"pattern": "**/fetch_dailyhot.mjs", "path": "/agent/workspace"}


def test_execute_cwd_is_consumed_before_the_upstream_schema() -> None:
    request = ToolCallRequest(
        tool_call={
            "name": "execute",
            "args": {"command": "node scripts/fetch_dailyhot.mjs weibo", "cwd": "/agent/.octop/skills/daily-hot-news"},
            "id": "call-1",
        },
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    seen: dict[str, object] = {}

    def handler(item: ToolCallRequest) -> ToolMessage:
        seen.update(item.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="call-1")

    FilesystemGuardMiddleware(workspace_dir="/agent/workspace").wrap_tool_call(request, handler)
    assert seen == {"command": "cd -- /agent/.octop/skills/daily-hot-news && node scripts/fetch_dailyhot.mjs weibo"}


def _sandbox_workspace(**backend_attrs: object) -> SimpleNamespace:
    return SimpleNamespace(backend=SimpleNamespace(sandbox_fs=True, **backend_attrs))


def test_sandbox_relative_path_stays_on_the_virtual_root() -> None:
    request = ToolCallRequest(
        tool_call={"name": "read_file", "args": {"file_path": "docs/guide.md"}, "id": "call-1"},
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    seen: dict[str, object] = {}

    def handler(item: ToolCallRequest) -> ToolMessage:
        seen.update(item.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="call-1")

    FilesystemGuardMiddleware(
        workspace_dir="/home/wally/.octop/agents/Q6084B",
        workspace=_sandbox_workspace(),  # type: ignore[arg-type]
    ).wrap_tool_call(request, handler)
    assert seen == {"file_path": "/docs/guide.md"}


def test_sandbox_base_dir_stays_virtual_until_the_backend_maps_it() -> None:
    request = ToolCallRequest(
        tool_call={
            "name": "read_file",
            "args": {
                "file_path": "references/sources.md",
                "base_dir": "/.octop/skills/daily-hot-news",
            },
            "id": "call-1",
        },
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    seen: dict[str, object] = {}

    def handler(item: ToolCallRequest) -> ToolMessage:
        seen.update(item.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="call-1")

    FilesystemGuardMiddleware(
        workspace_dir="/home/wally/.octop/agents/Q6084B",
        workspace=_sandbox_workspace(),  # type: ignore[arg-type]
    ).wrap_tool_call(request, handler)
    assert seen == {"file_path": "/.octop/skills/daily-hot-news/references/sources.md"}


def test_sandbox_glob_without_a_root_uses_the_virtual_workspace() -> None:
    request = ToolCallRequest(
        tool_call={"name": "glob", "args": {"pattern": "**/fetch_dailyhot.mjs"}, "id": "call-1"},
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    seen: dict[str, object] = {}

    def handler(item: ToolCallRequest) -> ToolMessage:
        seen.update(item.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="call-1")

    FilesystemGuardMiddleware(
        workspace_dir="/home/wally/.octop/agents/Q6084B",
        workspace=_sandbox_workspace(),  # type: ignore[arg-type]
    ).wrap_tool_call(request, handler)
    assert seen == {"pattern": "**/fetch_dailyhot.mjs", "path": "/"}


def test_execute_cwd_uses_the_sandbox_shell_path() -> None:
    request = ToolCallRequest(
        tool_call={
            "name": "execute",
            "args": {
                "command": "node scripts/fetch_dailyhot.mjs weibo",
                "cwd": "/.octop/skills/daily-hot-news",
            },
            "id": "call-1",
        },
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )
    seen: dict[str, object] = {}

    def handler(item: ToolCallRequest) -> ToolMessage:
        seen.update(item.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id="call-1")

    def shell_path(path: str) -> str:
        if path == "/.octop/skills/daily-hot-news":
            return "/workspace/.octop/skills/daily-hot-news"
        return path

    FilesystemGuardMiddleware(
        workspace_dir="/home/wally/.octop/agents/Q6084B",
        workspace=_sandbox_workspace(shell_path=shell_path),  # type: ignore[arg-type]
    ).wrap_tool_call(request, handler)
    assert seen == {
        "command": "cd -- /workspace/.octop/skills/daily-hot-news && node scripts/fetch_dailyhot.mjs weibo",
    }


class _AlwaysYes:
    """Decision scorer that keeps every candidate."""

    def score(self, *, state: str, statements: list[tuple[str, str]]) -> dict[str, float]:
        return {}

    async def score_async(self, *, state: str, statements: list[tuple[str, str]]) -> dict[str, float]:
        return {}


def _schema_tool(name: str, schema: type, func: object) -> StructuredTool:
    return StructuredTool.from_function(
        name=name,
        description=name,
        func=func,  # type: ignore[arg-type]
        args_schema=schema,
    )


def _json_schema(tool: StructuredTool) -> dict[str, object]:
    raw = tool.tool_call_schema
    assert not isinstance(raw, dict)
    return raw.model_json_schema()


def test_model_request_advertises_relative_paths_and_document_base() -> None:
    read_file = _schema_tool(
        "read_file",
        ReadFileSchema,
        lambda file_path, offset=0, limit=100: file_path,
    )
    read_video = _schema_tool(
        "read_file",
        ReadVideoFileSchema,
        lambda file_path, offset=0, limit=100: file_path,
    )
    execute = _schema_tool(
        "execute",
        ExecuteSchema,
        lambda command, timeout=None: command,
    )
    glob = _schema_tool(
        "glob",
        GlobSchema,
        lambda pattern, path=None: pattern,
    )
    other = _schema_tool("current_time", ReadFileSchema, lambda file_path, offset=0, limit=100: file_path)
    request = ModelRequest(model=object(), messages=[], tools=[read_file, read_video, execute, glob, other])  # type: ignore[arg-type]
    seen: list[ModelRequest] = []

    def handler(item: ModelRequest) -> ModelResponse:
        seen.append(item)
        return ModelResponse(result=[])

    FilesystemGuardMiddleware().wrap_model_call(request, handler)

    assert len(seen) == 1
    advertised = seen[0].tools
    assert advertised[4] is other
    read_schema = _json_schema(advertised[0])
    video_schema = _json_schema(advertised[1])
    execute_schema = _json_schema(advertised[2])
    glob_schema = _json_schema(advertised[3])
    assert "base_dir" in read_schema["properties"]
    assert "base_dir" in video_schema["properties"]
    assert "relative" in str(read_schema["properties"]["file_path"]["description"])
    assert "Must be absolute" not in str(read_schema["properties"]["file_path"]["description"])
    assert "seconds" in str(video_schema["properties"]["offset"]["description"])
    assert "cwd" in execute_schema["properties"]
    assert "base_dir" in glob_schema["properties"]
    assert "base_dir" not in _json_schema(read_file)["properties"]
    assert "cwd" not in _json_schema(execute)["properties"]


def test_blocks_read_on_denied_sensitive_path() -> None:
    permissions = [
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/etc/**"],
            mode="deny",
        )
    ]
    reason = filesystem_guard_block_reason(
        permissions,
        tool_name="read_file",
        params={"file_path": "/etc/passwd"},
    )
    assert reason is not None
    assert "permission denied" in reason
    assert "/etc/passwd" in reason


def test_allows_read_outside_denied_prefix() -> None:
    permissions = [
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/etc/**"],
            mode="deny",
        )
    ]
    assert (
        filesystem_guard_block_reason(
            permissions,
            tool_name="read_file",
            params={"file_path": "/workspace/SOUL.md"},
        )
        is None
    )


def test_blocks_write_file_tool() -> None:
    permissions = [
        FilesystemPermission(
            operations=["write"],
            paths=["/**/.env"],
            mode="deny",
        )
    ]
    reason = filesystem_guard_block_reason(
        permissions,
        tool_name="write_file",
        params={"file_path": "/home/user/project/.env"},
    )
    assert reason is not None
    assert "write" in reason


def test_blocks_ls_on_denied_directory_root() -> None:
    """``/etc/**`` must also deny listing the directory itself (``ls /etc``)."""
    permissions = [
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/etc/**"],
            mode="deny",
        )
    ]
    reason = filesystem_guard_block_reason(
        permissions,
        tool_name="ls",
        params={"path": "/etc"},
    )
    assert reason is not None
    assert "permission denied" in reason


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS /etc → /private/etc alias")
def test_blocks_ls_on_macos_private_etc_alias() -> None:
    """macOS ``/etc`` → ``/private/etc``; deny rules must cover the real path."""
    permissions = [
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/etc/**"],
            mode="deny",
        )
    ]
    reason = filesystem_guard_block_reason(
        permissions,
        tool_name="ls",
        params={"path": "/private/etc"},
    )
    assert reason is not None
    assert "permission denied" in reason


async def test_deferred_tool_dispatch_runs_inside_the_filesystem_guard(tmp_path: Path) -> None:
    """``call_deferred_tool`` must be forwarded *through* the guard.

    ``DecisionMiddleware`` re-enters the chain at the next handler for the
    forwarded tool, so it has to sit outside ``FilesystemGuardMiddleware``. When
    it sat inside, the guard never saw the forwarded call: deny rules were
    skipped, and the ``base_dir`` / ``cwd`` fields the request copy advertises
    reached deepagents' tool functions, which reject them with ``TypeError`` and
    abort the turn.
    """
    from unittest.mock import MagicMock, patch

    from deepagents.middleware.filesystem import GrepSchema
    from langchain_core.messages import HumanMessage, SystemMessage

    from octop_harness.agent import HarnessAgent
    from octop_harness.config import HarnessAgentConfig, ModelConfig, ProviderConfig
    from octop_harness.decision.config import DecisionConfig
    from octop_harness.middleware.decision import DecisionMiddleware

    cfg = HarnessAgentConfig(
        name="test-agent",
        workspace_dir=tmp_path,
        backend={"type": "filesystem", "root_dir": str(tmp_path), "virtual_mode": False},
        providers=[
            ProviderConfig(
                id="p",
                base_url="https://x",
                api_key="k",
                models=[ModelConfig(id="text", input=["text"])],
            )
        ],
        default_model="p/text",
        memory_enabled=False,
        checkpointer=False,
        decision=DecisionConfig(
            enabled=True,
            endpoint="http://127.0.0.1:9/v1/decisions",
            skills=False,
            tools=True,
            team=False,
            tool_min_count=1,
        ),
    )
    with (
        patch("langchain_openai.ChatOpenAI", return_value=MagicMock()),
        patch("deepagents.create_deep_agent", return_value=MagicMock(name="fake-graph")),
    ):
        agent = HarnessAgent(cfg)
    chain = agent._build_middleware()
    names = [type(mw).__name__ for mw in chain]
    assert "DecisionMiddleware" in names
    assert "FilesystemGuardMiddleware" in names
    assert names.index("DecisionMiddleware") < names.index("FilesystemGuardMiddleware")

    async def grep(pattern: str, path: str | None = None) -> str:
        return f"{pattern}@{path}"

    live = StructuredTool.from_function(coroutine=grep, name="grep", args_schema=GrepSchema, description="grep")

    async def widget(query: str) -> str:
        return query

    long_tail = StructuredTool.from_function(coroutine=widget, name="widget_search", description="widget")
    guard = FilesystemGuardMiddleware(workspace_dir=str(tmp_path))
    decision = DecisionMiddleware(config=cfg.decision, scorer=_AlwaysYes())
    request = ModelRequest(
        model=MagicMock(),
        messages=[HumanMessage(content="用 grep 搜 Octop")],
        system_message=SystemMessage(content="base"),
        tool_choice=None,
        tools=[live, long_tail],
        response_format=None,
        state={},
        runtime=MagicMock(),
        model_settings={},
    )

    seen: dict[str, object] = {}

    async def model_handler(item: ModelRequest) -> ModelResponse:
        seen["model_tools"] = list(item.tools or [])
        return ModelResponse(result=[])

    await decision.awrap_model_call(request, lambda item: guard.awrap_model_call(item, model_handler))
    model_tools = seen["model_tools"]
    assert any(
        "base_dir" in _json_schema(tool).get("properties", {})
        for tool in model_tools
        if isinstance(tool, StructuredTool)
    )

    async def tool_handler(item: ToolCallRequest) -> ToolMessage:
        seen["name"] = item.tool_call["name"]
        seen["args"] = item.tool_call["args"]
        seen["schema"] = _json_schema(item.tool) if isinstance(item.tool, StructuredTool) else {}
        return ToolMessage(content="ok", name=str(item.tool_call["name"]), tool_call_id="c1")

    await decision.awrap_tool_call(
        ToolCallRequest(
            tool_call={
                "name": "call_deferred_tool",
                "args": {
                    "name": "grep",
                    "arguments": {"pattern": "Octop", "path": "docs", "base_dir": "/tmp/base"},
                },
                "id": "c1",
            },
            tool=None,
            state={},
            runtime=None,  # type: ignore[arg-type]
        ),
        lambda item: guard.awrap_tool_call(item, tool_handler),
    )
    assert seen["name"] == "grep"
    assert seen["args"] == {"pattern": "Octop", "path": "/tmp/base/docs"}
    assert "base_dir" not in seen["schema"].get("properties", {})
