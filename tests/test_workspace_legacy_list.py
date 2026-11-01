"""Workspace listing of legacy/system-dir names with ``system_files_path`` set.

Regression tests for the workspace-panel symptom (TencentCloud/Octop#2612):
entering ``skills/`` listed only the canonical ``.octop/skills`` tree, so
user-authored skills in the workspace root were invisible while the root
listing still advertised ``skills/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from deepagents.backends import FilesystemBackend

from octop_harness.backends.workspace import BackendWorkspace


@pytest.fixture
def seeded(tmp_path: Path) -> BackendWorkspace:
    (tmp_path / "skills" / "demo").mkdir(parents=True)
    (tmp_path / "skills" / "demo" / "SKILL.md").write_text("user skill", encoding="utf-8")
    (tmp_path / "skills" / "loose.md").write_text("loose", encoding="utf-8")
    (tmp_path / ".octop" / "skills" / "octop-assistant").mkdir(parents=True)
    (tmp_path / ".octop" / "skills" / "octop-assistant" / "SKILL.md").write_text("builtin", encoding="utf-8")
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    return BackendWorkspace(backend, tmp_path, system_files_path=".octop")


class TestLegacyDirListings:
    def test_list_dir_legacy_skills_root(self, seeded: BackendWorkspace, tmp_path: Path) -> None:
        """``list_dir('skills')`` lists the real workspace-root ``skills/`` tree."""
        rows = seeded.list_dir("skills")
        assert rows is not None
        paths = {row["path"] for row in rows}
        assert paths == {"skills/demo", "skills/loose.md"}
        assert all(not row["path"].startswith(".octop/") for row in rows)

    def test_list_dir_root_still_shows_both(self, seeded: BackendWorkspace) -> None:
        rows = seeded.list_dir(".")
        assert rows is not None
        paths = {row["path"] for row in rows}
        assert paths == {".octop", "skills"}

    def test_list_dir_canonical_explicit_path(self, seeded: BackendWorkspace) -> None:
        """An explicit ``.octop/...`` request keeps listing the canonical tree."""
        rows = seeded.list_dir(".octop/skills")
        assert rows is not None
        paths = {row["path"] for row in rows}
        assert paths == {".octop/skills/octop-assistant"}

    async def test_als_legacy_skills_root(self, seeded: BackendWorkspace) -> None:
        """``als`` (the async path used by the dashboard tree API) agrees."""
        result = await seeded.als("skills")
        assert result is not None
        paths = set()
        for entry in result.entries:
            raw = entry.get("path", "") if isinstance(entry, dict) else getattr(entry, "path", "")
            paths.add(str(raw or ""))
        assert paths == {"skills/demo", "skills/loose.md"}
