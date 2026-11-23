"""Tests for ``octop_harness.skills.catalog``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from octop_harness.backends.workspace import BackendWorkspace
from octop_harness.skills.catalog import list_skill_summaries


@pytest.fixture
def fs_backend(tmp_path: Path) -> Any:
    from deepagents.backends import FilesystemBackend

    return FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)


class TestSkillCatalog:
    async def test_lists_builtin_and_workspace_skills(self, fs_backend: Any, tmp_path: Path) -> None:
        ws = BackendWorkspace(fs_backend, tmp_path)
        await ws.aupload_many(
            [
                (
                    "_builtin_skills/builtin-one/SKILL.md",
                    b"---\nname: builtin-one\ndescription: built-in\n---\n# One\n",
                ),
                (
                    "skills/custom-one/SKILL.md",
                    b"---\nname: custom-one\ndescription: custom\n---\n# Custom\n",
                ),
            ]
        )

        rows = await list_skill_summaries(ws)
        by_slug = {row["slug"]: row for row in rows}
        assert set(by_slug) == {"builtin-one", "custom-one"}
        assert by_slug["custom-one"]["kind"] == "workspace"
        assert by_slug["builtin-one"]["kind"] == "builtin"

    async def test_workspace_overrides_builtin_slug(self, fs_backend: Any, tmp_path: Path) -> None:
        ws = BackendWorkspace(fs_backend, tmp_path)
        await ws.aupload_many(
            [
                (
                    "_builtin_skills/shared/SKILL.md",
                    b"---\nname: shared\ndescription: builtin copy\n---\n",
                ),
                (
                    "skills/shared/SKILL.md",
                    b"---\nname: shared\ndescription: workspace copy\n---\n",
                ),
            ]
        )

        rows = await list_skill_summaries(ws)
        shared = next(row for row in rows if row["slug"] == "shared")
        assert shared["description"] == "workspace copy"
        assert shared["kind"] == "workspace"

    async def test_skills_disabled_marks_enabled_false(self, fs_backend: Any, tmp_path: Path) -> None:
        ws = BackendWorkspace(fs_backend, tmp_path)
        await ws.aupload_many(
            [
                (
                    "skills/off-skill/SKILL.md",
                    b"---\nname: display-name\ndescription: off\n---\n",
                ),
            ]
        )

        rows = await list_skill_summaries(ws, skills_disabled=frozenset({"off-skill"}))
        row = rows[0]
        assert row["slug"] == "off-skill"
        assert row["name"] == "display-name"
        assert row["enabled"] is False

    async def test_removed_skills_are_omitted(self, fs_backend: Any, tmp_path: Path) -> None:
        ws = BackendWorkspace(fs_backend, tmp_path)
        await ws.awrite_text(
            "skills/gone/SKILL.md",
            "---\nremoved: true\n---\n",
            force=True,
        )

        rows = await list_skill_summaries(ws)
        assert rows == []

    async def test_skills_dir_scans_extra_root_directories(self, fs_backend: Any, tmp_path: Path) -> None:
        ws = BackendWorkspace(fs_backend, tmp_path)
        await ws.aupload_many(
            [
                (
                    "skills_extra/alpha/SKILL.md",
                    b"---\nname: alpha\ndescription: from extra root\n---\n",
                ),
                (
                    "skills_extra/beta/SKILL.md",
                    b"---\nname: beta\ndescription: also extra\n---\n",
                ),
            ]
        )

        rows = await list_skill_summaries(ws, skills_dir="skills_extra")
        by_slug = {row["slug"]: row for row in rows}
        assert set(by_slug) == {"alpha", "beta"}
        assert all(row["kind"] == "workspace" for row in rows)

    async def test_results_sorted_by_kind_then_slug(self, fs_backend: Any, tmp_path: Path) -> None:
        ws = BackendWorkspace(fs_backend, tmp_path)
        await ws.aupload_many(
            [
                (
                    "skills/z-skill/SKILL.md",
                    b"---\nname: z-skill\ndescription: z\n---\n",
                ),
                (
                    "_builtin_skills/a-skill/SKILL.md",
                    b"---\nname: a-skill\ndescription: a\n---\n",
                ),
            ]
        )

        rows = await list_skill_summaries(ws)
        assert [row["slug"] for row in rows] == ["a-skill", "z-skill"]

    async def test_harness_emoji_in_summary(self, fs_backend: Any, tmp_path: Path) -> None:
        ws = BackendWorkspace(fs_backend, tmp_path)
        await ws.awrite_text(
            "skills/icon-skill/SKILL.md",
            ('---\nname: icon-skill\ndescription: has icon\nmetadata:\n  harness:\n    emoji: "🎯"\n---\n'),
            force=True,
        )

        rows = await list_skill_summaries(ws)
        assert rows[0]["emoji"] == "🎯"

    async def test_octop_presentation_metadata_in_summary(self, fs_backend: Any, tmp_path: Path) -> None:
        ws = BackendWorkspace(fs_backend, tmp_path)
        await ws.awrite_text(
            "skills/pdf-reader/SKILL.md",
            (
                "---\n"
                "name: pdf-reader\n"
                "description: Agent trigger description\n"
                "metadata:\n"
                "  octop:\n"
                "    label:\n"
                "      zh: PDF 阅读\n"
                "      en: PDF Reader\n"
                "    summary:\n"
                "      zh: 阅读和处理 PDF\n"
                "      en: Read and process PDFs\n"
                "    emoji: 📄\n"
                "    icon_url: https://cdn.example.com/pdf.png\n"
                "---\n"
            ),
            force=True,
        )

        rows = await list_skill_summaries(ws)

        assert rows == [
            {
                "slug": "pdf-reader",
                "name": "pdf-reader",
                "description": "Agent trigger description",
                "enabled": True,
                "kind": "workspace",
                "label": {"zh": "PDF 阅读", "en": "PDF Reader"},
                "summary": {"zh": "阅读和处理 PDF", "en": "Read and process PDFs"},
                "emoji": "📄",
                "icon_url": "https://cdn.example.com/pdf.png",
            }
        ]

    async def test_octop_display_name_does_not_replace_identity(self, fs_backend: Any, tmp_path: Path) -> None:
        ws = BackendWorkspace(fs_backend, tmp_path)
        await ws.awrite_text(
            "skills/pdf-reader/SKILL.md",
            (
                "---\n"
                "name: pdf-reader\n"
                "description: Agent trigger\n"
                "metadata:\n"
                "  octop:\n"
                "    display_name: PDF Reader\n"
                "---\n"
            ),
            force=True,
        )

        rows = await list_skill_summaries(ws)

        assert rows[0]["name"] == "pdf-reader"
        assert rows[0]["display_name"] == "PDF Reader"

    async def test_absolute_skills_dir_outside_workspace(self, tmp_path: Path) -> None:
        """Host-absolute skills_dir entries must stay listable (not rewritten relative)."""
        from deepagents.backends import FilesystemBackend

        ws_dir = tmp_path / "workspace"
        ws_dir.mkdir()
        (ws_dir / "skills" / "local").mkdir(parents=True)
        (ws_dir / "skills" / "local" / "SKILL.md").write_text(
            "---\nname: local\ndescription: workspace\n---\n",
            encoding="utf-8",
        )
        package_skills = tmp_path / "skill-packages" / "PACK01" / "skills"
        (package_skills / "pdf-reader").mkdir(parents=True)
        (package_skills / "pdf-reader" / "SKILL.md").write_text(
            "---\nname: pdf-reader\ndescription: Read PDF\n---\n# PDF\n",
            encoding="utf-8",
        )
        backend = FilesystemBackend(root_dir="/", virtual_mode=True)
        ws = BackendWorkspace(backend, ws_dir)

        abs_root = str(package_skills.resolve())
        rows = await list_skill_summaries(ws, skills_dir=[abs_root])
        by_slug = {row["slug"]: row for row in rows}
        assert set(by_slug) == {"local", "pdf-reader"}
        assert by_slug["pdf-reader"]["kind"] == "workspace"
        assert by_slug["local"]["kind"] == "workspace"

    async def test_skips_skill_with_invalid_utf8_manifest(self, fs_backend: Any, tmp_path: Path) -> None:
        ws = BackendWorkspace(fs_backend, tmp_path)
        good = b"---\nname: good\ndescription: ok\n---\n"
        bad = b"---\nname: bad\n---\n" + b"\xe2j$broken"
        await ws.aupload_many(
            [
                ("skills/good/SKILL.md", good),
                ("skills/bad/SKILL.md", bad),
            ]
        )

        rows = await list_skill_summaries(ws)
        assert [row["slug"] for row in rows] == ["good"]


class TestSkillPlatformCompatibility:
    def test_parses_known_compatibility_strings(self) -> None:
        from octop_harness.skills.catalog import skill_compatible_platforms

        assert skill_compatible_platforms(None) is None
        assert skill_compatible_platforms("") is None
        assert skill_compatible_platforms("Python 3.10+") is None
        assert skill_compatible_platforms("macos only") == frozenset({"Darwin"})
        assert skill_compatible_platforms("macos only \u2014 requires Reminders.app and remindctl CLI") == frozenset(
            {"Darwin"}
        )
        assert skill_compatible_platforms("linux, macos") == frozenset({"Linux", "Darwin"})
        assert skill_compatible_platforms("linux, macos, windows") == frozenset({"Linux", "Darwin", "Windows"})

    def test_platform_predicate(self) -> None:
        from octop_harness.skills.catalog import is_skill_platform_compatible

        assert is_skill_platform_compatible("macos only", system="Linux") is False
        assert is_skill_platform_compatible("macos only", system="Darwin") is True
        assert is_skill_platform_compatible("linux, macos", system="Windows") is False
        assert is_skill_platform_compatible("linux, macos", system="Linux") is True
        assert is_skill_platform_compatible(None, system="Linux") is True
        assert is_skill_platform_compatible("Python 3.10+", system="Windows") is True

    async def test_catalog_omits_macos_only_on_linux(
        self, fs_backend: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from octop_harness.skills import catalog as catalog_mod

        monkeypatch.setattr(catalog_mod.platform, "system", lambda: "Linux")
        ws = BackendWorkspace(fs_backend, tmp_path)
        await ws.aupload_many(
            [
                (
                    "_builtin_skills/apple-notes/SKILL.md",
                    b"---\nname: apple-notes\ndescription: notes\ncompatibility: macos only\n---\n",
                ),
                (
                    "_builtin_skills/apple-reminders/SKILL.md",
                    (
                        "---\n"
                        "name: apple-reminders\n"
                        "description: reminders\n"
                        "compatibility: macos only \u2014 requires Reminders.app and remindctl CLI\n"
                        "---\n"
                    ).encode(),
                ),
                (
                    "_builtin_skills/watchers/SKILL.md",
                    b"---\nname: watchers\ndescription: poll\ncompatibility: linux, macos\n---\n",
                ),
                (
                    "_builtin_skills/plan/SKILL.md",
                    b"---\nname: plan\ndescription: plan\ncompatibility: linux, macos, windows\n---\n",
                ),
                (
                    "skills/custom-one/SKILL.md",
                    b"---\nname: custom-one\ndescription: custom\n---\n",
                ),
            ]
        )

        rows = await list_skill_summaries(ws)
        assert [row["slug"] for row in rows] == ["plan", "watchers", "custom-one"]
        assert all(row["enabled"] is True for row in rows)

    async def test_catalog_keeps_macos_only_on_darwin(
        self, fs_backend: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from octop_harness.skills import catalog as catalog_mod

        monkeypatch.setattr(catalog_mod.platform, "system", lambda: "Darwin")
        ws = BackendWorkspace(fs_backend, tmp_path)
        await ws.awrite_text(
            "_builtin_skills/apple-notes/SKILL.md",
            "---\nname: apple-notes\ndescription: notes\ncompatibility: macos only\n---\n",
            force=True,
        )

        rows = await list_skill_summaries(ws)
        assert [row["slug"] for row in rows] == ["apple-notes"]
        assert rows[0]["enabled"] is True

    async def test_catalog_omits_non_windows_skill_on_windows(
        self, fs_backend: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from octop_harness.skills import catalog as catalog_mod

        monkeypatch.setattr(catalog_mod.platform, "system", lambda: "Windows")
        ws = BackendWorkspace(fs_backend, tmp_path)
        await ws.aupload_many(
            [
                (
                    "_builtin_skills/watchers/SKILL.md",
                    b"---\nname: watchers\ndescription: poll\ncompatibility: linux, macos\n---\n",
                ),
                (
                    "_builtin_skills/plan/SKILL.md",
                    b"---\nname: plan\ndescription: plan\ncompatibility: linux, macos, windows\n---\n",
                ),
            ]
        )

        rows = await list_skill_summaries(ws)
        assert [row["slug"] for row in rows] == ["plan"]
