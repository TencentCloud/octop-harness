"""Tests for MemoryService-backed LLM tool functions."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from octop_memory import Memory, MemoryService
from octop_memory.application.path_projection import raw_to_path
from pydantic import ValidationError

from octop_harness.builtin.tools.memory_tools import build_memory_tools


class _FakeService:
    def __init__(self) -> None:
        self.search_result: dict[str, Any] = {"hits": []}
        self.get_result: dict[str, Any] = {"content": ""}
        self.save_result = type(
            "Node", (), {"content": "User prefers dark mode", "topic": "preferences", "metadata": {}}
        )()
        self.save_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    def save(self, content: str, *, topic: str | None = None) -> Any:
        self.save_calls.append({"content": content, "topic": topic})
        return self.save_result

    def search(self, query: str, *, max_results: int = 5, corpus: str = "all") -> dict[str, Any]:
        self.search_calls.append({"query": query, "max_results": max_results, "corpus": corpus})
        return self.search_result

    def get(self, path: str, *, start: int | None = None, lines: int | None = None) -> dict[str, Any]:
        self.get_calls.append({"path": path, "start": start, "lines": lines})
        return self.get_result


class TestBuildMemoryTools:
    @pytest.mark.parametrize("kind", ["Preference", "Decision", "Task", "Fact"])
    def test_classified_save_passes_kind_to_service(self, kind: str) -> None:
        service = _FakeService()
        with patch.object(service, "save") as save_call:
            save = next(t for t in build_memory_tools(service) if t.name == "memory_save")
            assert "saved" in save.invoke({"content": "Remember this", "kind": kind}).lower()
            if kind == "Fact":
                save_call.assert_called_once_with("Remember this", topic=None)
            else:
                save_call.assert_called_once_with("Remember this", topic=None, kind=kind)

    def test_invalid_classification_never_saves(self) -> None:
        service = _FakeService()
        save = next(t for t in build_memory_tools(service) if t.name == "memory_save")
        with pytest.raises(ValidationError):
            save.invoke({"content": "Remember this", "kind": "ConflictCandidate"})
        assert service.save_calls == []

    def test_old_memory_version_rejects_typed_save_without_silent_fact_fallback(self) -> None:
        service = _FakeService()
        save = next(t for t in build_memory_tools(service) if t.name == "memory_save")
        result = save.invoke({"content": "User prefers Chinese", "kind": "Preference"})
        assert "upgrade octop-memory" in result
        assert service.save_calls == []

    def test_memory_save_does_not_report_document_review_to_the_model(self) -> None:
        service = _FakeService()
        service.save_result.metadata = {
            "manual_document_conflict": True,
            "manual_document_review": {"status": "partial"},
        }
        save = next(t for t in build_memory_tools(service) if t.name == "memory_save")
        assert save.invoke({"content": "A conflicting fact"}) == "Memory saved to octop-memory."

    def test_returns_recall_tools(self) -> None:
        tools = build_memory_tools(_FakeService())  # type: ignore[arg-type]
        assert {t.name for t in tools} == {"memory_search", "memory_get", "memory_save"}

    def test_memory_save_writes_through_service(self) -> None:
        service = _FakeService()
        save = next(t for t in build_memory_tools(service) if t.name == "memory_save")

        result = save.invoke({"content": "User prefers dark mode", "topic": "preferences"})

        assert service.save_calls == [{"content": "User prefers dark mode", "topic": "preferences"}]
        assert "saved" in result.lower()

    def test_memory_save_rejects_empty_content(self) -> None:
        service = _FakeService()
        save = next(t for t in build_memory_tools(service) if t.name == "memory_save")

        assert "empty" in save.invoke({"content": "  "}).lower()
        assert service.save_calls == []

    def test_memory_save_reports_failure_without_claiming_success(self) -> None:
        service = _FakeService()
        save = next(t for t in build_memory_tools(service) if t.name == "memory_save")
        with patch.object(service, "save", side_effect=OSError("unavailable")):
            result = save.invoke({"content": "Remember this project decision"})
        assert result == "memory_save failed: OSError"

    def test_memory_search_formats_hits(self) -> None:
        service = _FakeService()
        service.search_result = {
            "hits": [
                {
                    "path": "atom/a.md",
                    "layer": "atom",
                    "snippet": "User prefers dark mode\nfor dashboards.",
                },
            ],
        }
        tools = build_memory_tools(service)
        search_tool = next(t for t in tools if t.name == "memory_search")  # type: ignore[arg-type]

        result = search_tool.invoke({"query": "dark mode", "max_results": 3})

        assert "Memory hits:" in result
        assert "[atom] atom/a.md" in result
        assert "User prefers dark mode for dashboards." in result
        assert service.search_calls == [{"query": "dark mode", "max_results": 3, "corpus": "all"}]

    @pytest.mark.parametrize(
        "image",
        [
            '{"type": "image_url", "workspace_path": "inbound/screenshot.png"}',
            '{"image_url": {"url": "data:image/png;base64,AAAA"}, "type": "image_url"}',
            '{"type":"image","source":{"type":"base64","data":"AAAA"}}',
            '{"type":"input_image","image_url":"data:image/png;base64,AAAA"}',
        ],
    )
    def test_memory_search_keeps_text_around_raw_image_blocks(self, image: str) -> None:
        service = _FakeService()
        service.search_result = {
            "hits": [{"path": "raw/example.md", "layer": "raw", "snippet": f"分析截图\n{image}\n重启后恢复"}]
        }
        search = next(t for t in build_memory_tools(service) if t.name == "memory_search")

        assert search.invoke({"query": "截图"}) == "Memory hits:\n- [raw] raw/example.md\n  分析截图 重启后恢复"

    @pytest.mark.parametrize(
        "image",
        [
            '{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA…',
            '{"image_url":{"url":"data:image/png;base64,AAAA…',
            '{"type": "image_ur…',
        ],
    )
    def test_memory_search_omits_image_blocks_clipped_by_memory(self, image: str) -> None:
        service = _FakeService()
        service.search_result = {"hits": [{"path": "raw/example.md", "layer": "raw", "snippet": f"分析截图\n{image}"}]}
        search = next(t for t in build_memory_tools(service) if t.name == "memory_search")

        assert search.invoke({"query": "截图"}) == "Memory hits:\n- [raw] raw/example.md\n  分析截图"

    @pytest.mark.parametrize("caption", ["", "用户发送了图片。\n", "User sent an image.\n"])
    def test_memory_search_skips_image_only_hits(self, caption: str) -> None:
        service = _FakeService()
        service.search_result = {
            "hits": [
                {
                    "path": "raw/screenshot.md",
                    "layer": "raw",
                    "snippet": caption + '{"type":"image_url","workspace_path":"inbound/screenshot.png"}',
                }
            ]
        }
        search = next(t for t in build_memory_tools(service) if t.name == "memory_search")

        assert search.invoke({"query": "截图"}) == "No matching memory entries. (empty)"

        example = '接口示例\n{"type":"image_url","image_url":{"url":"https://example.invalid/p.png"}}'
        service.search_result["hits"].append({"path": "atom/decision.md", "layer": "atom", "snippet": example})
        assert search.invoke({"query": "截图"}) == "Memory hits:\n- [atom] atom/decision.md\n  " + example.replace(
            "\n", " "
        )

    def test_memory_search_preserves_other_json_in_raw_text(self) -> None:
        service = _FakeService()
        text = '{"type":"file","workspace_path":"inbound/report.pdf"}\n{"type":"image_url","description":"example"}'
        service.search_result = {"hits": [{"path": "raw/example.md", "layer": "raw", "snippet": text}]}
        search = next(t for t in build_memory_tools(service) if t.name == "memory_search")

        assert text.replace("\n", " ") in search.invoke({"query": "example"})

    @pytest.mark.parametrize("with_hits", [True, False])
    def test_search_reports_pending_index_upgrade(self, with_hits: bool) -> None:
        service = _FakeService()
        hint = "Raw memory search index needs an upgrade. Run `octop-memory db upgrade-index`."
        hit = {"layer": "atom", "path": "atom/a.md", "snippet": "A saved preference."}
        service.search_result = {
            "hits": [hit] if with_hits else [],
            "warning": hint,
            "empty_reason": None if with_hits else hint,
        }
        search = next(t for t in build_memory_tools(service) if t.name == "memory_search")
        result = search.invoke({"query": "preference"})
        assert result.count(hint) == 1
        if with_hits:
            assert "A saved preference." in result

    def test_memory_search_empty(self) -> None:
        service = _FakeService()
        service.search_result = {"hits": [], "empty_reason": "no_index"}
        tools = build_memory_tools(service)
        search_tool = next(t for t in tools if t.name == "memory_search")  # type: ignore[arg-type]

        result = search_tool.invoke({"query": "nothing here"})

        assert result == "No matching memory entries. (no_index)"

    def test_memory_get_returns_content(self) -> None:
        service = _FakeService()
        service.get_result = {"content": "# Memory\ncontent"}
        get_tool = next(t for t in build_memory_tools(service) if t.name == "memory_get")  # type: ignore[arg-type]

        result = get_tool.invoke({"path": "atom/a.md", "start": 2, "lines": 5})

        assert result == "# Memory\ncontent"
        assert service.get_calls == [{"path": "atom/a.md", "start": 2, "lines": 5}]

    def test_memory_get_filters_images_when_opening_a_search_result(self, tmp_path: Path) -> None:
        memory = Memory("test", backend_config={"db_path": str(tmp_path / "memory.sqlite")})
        service = MemoryService(memory, host="octop-harness")
        image = (
            '{"type":"image_url","workspace_path":"inbound/screenshot.png",'
            '"image_url":{"url":"data:image/png;base64,AAAA"}}'
        )
        try:
            service.capture_turn(
                user=f"orchidcatalyst caption\n{image}\nMore details after the screenshot.",
                assistant="Synthetic response.",
                session_id="previous-session",
                thread_id="previous-thread",
            )
            tools = {tool.name: tool for tool in build_memory_tools(service)}
            hits = tools["memory_search"].invoke({"query": "orchidcatalyst"})
            path = hits.splitlines()[1].split()[-1]
            full = ast.literal_eval(tools["memory_get"].invoke({"path": path}))
            assert full["excerpt"].startswith("---\nid: ")
            assert full["excerpt"].endswith("orchidcatalyst caption\n\nMore details after the screenshot.")
            assert "workspace_path" not in str(full) and "base64" not in str(full)
            source = service.get(path)["excerpt"].splitlines()
            start = source.index("orchidcatalyst caption") + 1
            result = ast.literal_eval(tools["memory_get"].invoke({"path": path, "start": start, "lines": 3}))

            assert result["excerpt"] == "orchidcatalyst caption\n\nMore details after the screenshot."
            assert result["from_line"] == start
            assert result["to_line"] == start + 2
            assert "workspace_path" not in str(result)
            assert "base64" not in str(result)
        finally:
            memory.backend.close()

    @pytest.mark.parametrize("caption", ["", "用户发送了图片。\n", "User sent an image.\n"])
    def test_memory_get_reports_no_usable_content_for_image_only_records(self, tmp_path: Path, caption: str) -> None:
        memory = Memory("test", backend_config={"db_path": str(tmp_path / "memory.sqlite")})
        service = MemoryService(memory, host="octop-harness")
        image = (
            '{"type":"image_url","workspace_path":"inbound/septentrion.png",'
            '"image_url":{"url":"data:image/png;base64,AAAA"}}'
        )
        try:
            service.capture_turn(
                user=caption + image,
                assistant="Synthetic response.",
                session_id="image-session",
                thread_id="image-thread",
            )
            [raw] = memory.list_raw(thread_id="image-thread", event_type="user_message")
            path = raw_to_path(raw)
            tools = {tool.name: tool for tool in build_memory_tools(service)}
            image_line = service.get(path)["excerpt"].splitlines().index(image) + 1

            assert tools["memory_search"].invoke({"query": "septentrion"}).startswith("No matching memory entries.")
            assert tools["memory_get"].invoke({"path": path}) == "No usable memory content."
            assert (
                tools["memory_get"].invoke({"path": path, "start": image_line, "lines": 1})
                == "No usable memory content."
            )
        finally:
            memory.backend.close()

    def test_memory_get_filters_legacy_raw_content(self) -> None:
        service = _FakeService()
        service.get_result = {"content": '配文\n{"type":"image","base64":"AAAA"}'}
        get_tool = next(t for t in build_memory_tools(service) if t.name == "memory_get")

        assert get_tool.invoke({"path": "raw/2026-11-19/example.md"}).strip() == "配文"

    def test_memory_get_formats_error(self) -> None:
        service = _FakeService()
        service.get_result = {"error": "not found"}
        get_tool = next(t for t in build_memory_tools(service) if t.name == "memory_get")  # type: ignore[arg-type]

        result = get_tool.invoke({"path": "missing.md"})

        assert result == "memory_get error: not found"

    def test_memory_search_swallows_unexpected_errors(self) -> None:
        class _Boom(_FakeService):
            def search(self, query: str, *, max_results: int = 5, corpus: str = "all") -> dict[str, Any]:
                raise Exception("InvalidSubscription")

        tools = build_memory_tools(_Boom())  # type: ignore[arg-type]
        search_tool = next(t for t in tools if t.name == "memory_search")
        result = search_tool.invoke({"query": "anything"})
        assert result == "memory_search failed: Exception"
