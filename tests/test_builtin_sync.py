"""Tests for ``octop_harness.builtin._sync.sync_builtin_skills_to_backend``."""

from __future__ import annotations

from collections import Counter
from importlib import resources
from pathlib import Path

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import FileUploadResponse, WriteResult

from octop_harness.backends.workspace import DEFAULT_BUILTIN_SKILLS_DIR, BackendWorkspace
from octop_harness.builtin._sync import sync_builtin_skills_to_backend
from octop_harness.builtin.templates import iter_builtin_skills


class CountingBackend(FilesystemBackend):
    def __init__(self, root: Path) -> None:
        super().__init__(root_dir=str(root), virtual_mode=False)
        self.checks: Counter[str] = Counter()
        self.puts: Counter[str] = Counter()

    def write(self, file_path: str, content: str) -> WriteResult:
        self.checks[file_path] += 1
        result = super().write(file_path, content)
        if result.error is None:
            self.puts[file_path] += 1
        return result

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        self.puts.update(path for path, _ in files)
        return super().upload_files(files)


@pytest.fixture
def workspace(tmp_path: Path) -> BackendWorkspace:
    return BackendWorkspace(
        FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False),
        tmp_path,
    )


class TestSyncBuiltinSkillsToBackend:
    @pytest.mark.parametrize("language", ["en", "zh"])
    def test_first_sync_writes_each_file_once_without_file_checks(self, tmp_path: Path, language: str) -> None:
        backend = CountingBackend(tmp_path)
        workspace = BackendWorkspace(backend, tmp_path)
        assert sync_builtin_skills_to_backend(workspace, "0.1.0", language=language) == (True, [])
        skill_puts = {path: count for path, count in backend.puts.items() if not path.endswith("/.version")}
        assert skill_puts and set(skill_puts.values()) == {1}
        assert all(path.endswith("/.version") for path in backend.checks)
        expected_files = iter_builtin_skills(language)
        assert len(skill_puts) == len(expected_files)
        for _, relative_path, content, _ in expected_files:
            assert workspace.read_text(f"{DEFAULT_BUILTIN_SKILLS_DIR}/{relative_path}") == content.decode("utf-8")
        if language == "zh":
            path = f"{DEFAULT_BUILTIN_SKILLS_DIR}/watchers/SKILL.md"
            expected = resources.files("octop_harness.builtin.skills").joinpath("zh/watchers/SKILL.md")
            assert workspace.read_text(path) == expected.read_text(encoding="utf-8")

    def test_existing_marker_preserves_version_and_upgrade_behavior(self, tmp_path: Path) -> None:
        backend = CountingBackend(tmp_path)
        workspace = BackendWorkspace(backend, tmp_path)
        sync_builtin_skills_to_backend(workspace, "0.1.0")
        path = f"{DEFAULT_BUILTIN_SKILLS_DIR}/watchers/SKILL.md"
        workspace.write_text(path, "user edit", force=True)
        backend.checks.clear()
        backend.puts.clear()
        assert sync_builtin_skills_to_backend(workspace, "0.1.0") == (False, [])
        assert not backend.checks and not backend.puts
        assert workspace.read_text(path) == "user edit"
        assert sync_builtin_skills_to_backend(workspace, "0.2.0") == (True, [])
        assert backend.checks
        expected = resources.files("octop_harness.builtin.skills").joinpath("en/watchers/SKILL.md")
        assert workspace.read_text(path) == expected.read_text(encoding="utf-8")

    def test_syncs_skills_to_backend(self, workspace: BackendWorkspace) -> None:
        synced, fallback = sync_builtin_skills_to_backend(workspace, agent_version="0.1.0")
        assert synced is True
        assert fallback == []
        assert workspace.exists(f"{DEFAULT_BUILTIN_SKILLS_DIR}/using-workspace/SKILL.md")
        assert workspace.read_text(f"{DEFAULT_BUILTIN_SKILLS_DIR}/.version") == "0.1.0"

    def test_skips_when_version_matches(self, workspace: BackendWorkspace) -> None:
        sync_builtin_skills_to_backend(workspace, agent_version="0.1.0")
        synced, fallback = sync_builtin_skills_to_backend(workspace, agent_version="0.1.0")
        assert synced is False
        assert fallback == []

    def test_re_syncs_when_version_differs(self, workspace: BackendWorkspace) -> None:
        sync_builtin_skills_to_backend(workspace, agent_version="0.1.0")
        synced, fallback = sync_builtin_skills_to_backend(workspace, agent_version="0.2.0")
        assert synced is True
        assert fallback == []
        assert workspace.read_text(f"{DEFAULT_BUILTIN_SKILLS_DIR}/.version") == "0.2.0"

    def test_zh_overlays_skill_md_and_inherits_en_scripts(self, workspace: BackendWorkspace) -> None:
        from importlib import resources

        synced, fallback = sync_builtin_skills_to_backend(workspace, agent_version="0.1.0", language="zh")
        assert synced is True
        assert fallback == []

        package = resources.files("octop_harness.builtin.skills")
        zh_skill = package.joinpath("zh/watchers/SKILL.md").read_text(encoding="utf-8")
        en_script = package.joinpath("en/watchers/scripts/watch_github.py").read_text(encoding="utf-8")
        assert workspace.read_text(f"{DEFAULT_BUILTIN_SKILLS_DIR}/watchers/SKILL.md") == zh_skill
        assert workspace.read_text(f"{DEFAULT_BUILTIN_SKILLS_DIR}/watchers/scripts/watch_github.py") == en_script
        assert workspace.exists(f"{DEFAULT_BUILTIN_SKILLS_DIR}/jupyter-live-kernel/scripts/jupyter_live_kernel.py")
