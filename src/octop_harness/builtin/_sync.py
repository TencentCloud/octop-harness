"""Sync packaged built-in skills and agents into the workspace via :class:`BackendWorkspace`."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib import resources
from importlib.resources.abc import Traversable
from typing import TYPE_CHECKING

from octop_harness.backends.workspace import DEFAULT_AGENTS_DIR, DEFAULT_BUILTIN_SKILLS_DIR

if TYPE_CHECKING:
    from octop_harness.backends.workspace import BackendWorkspace

logger = logging.getLogger(__name__)

VERSION_FILE = ".version"
BUILTIN_SKILLS_PACKAGE = "octop_harness.builtin.skills"
BUILTIN_AGENTS_PACKAGE = "octop_harness.builtin.agents"


@dataclass
class SeedAgentsResult:
    """Outcome of seeding packaged agent markdown into the workspace."""

    created: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    overwritten: list[str] = field(default_factory=list)
    fallback: list[str] = field(default_factory=list)


def sync_builtin_skills_to_backend(
    workspace: BackendWorkspace,
    agent_version: str,
    *,
    language: str = "en",
    target_dir: str | None = None,
) -> tuple[bool, list[str]]:
    """Copy packaged built-in skills into the workspace backend.

    Args:
        workspace: Scoped workspace facade (all writes go through it).
        agent_version: Package version used as cache key.
        language: Language code (``"en"`` / ``"zh"``).  When the
            language subdirectory does not exist, falls back to ``"en"``
            and records the fallback paths.

            When *language* is not ``"en"``, implements **file-level
            fallback**: *language* files replace matching files in the
            ``"en"`` tree before copying. This allows
            ``zh/`` to contain only translated ``SKILL.md`` files while
            inheriting supplemental files (``templates/``,
            ``references/``, ``.sh`` scripts, etc.) from ``en/``.
        target_dir: Relative directory under ``workspace_dir``; defaults to
            ``DEFAULT_BUILTIN_SKILLS_DIR``.

    Returns:
        ``(synced, fallback_paths)`` where *fallback_paths* lists the
        workspace paths served from the ``"en"`` fallback.

        Note: when *language* directory exists but only contains a
        subset of files (the normal case for ``zh/``), individual
        missing files are silently filled from ``en/`` and are **not**
        recorded in *fallback_paths*.  Only the case where the entire
        language directory is absent populates *fallback_paths*.
    """
    root = target_dir or DEFAULT_BUILTIN_SKILLS_DIR
    fallback_paths: list[str] = []
    try:
        version_path = f"{root}/{VERSION_FILE}"
        has_marker = workspace.exists(version_path)
        existing = workspace.read_text(version_path) if has_marker else None
        if existing is not None and existing.strip() == agent_version:
            return False, []

        package_root: Traversable = resources.files(BUILTIN_SKILLS_PACKAGE)
        en_dir = package_root.joinpath("en")
        files: dict[str, Traversable] = {}
        _collect_tree(en_dir, root, files)
        if language != "en":
            lang_dir = package_root.joinpath(language)
            try:
                list(lang_dir.iterdir())
            except (FileNotFoundError, TypeError, OSError):
                fallback_paths.extend(_collect_skill_paths(en_dir, root))
            else:
                _collect_tree(lang_dir, root, files)

        for path, source in files.items():
            if has_marker:
                workspace.write_text(path, source.read_text(encoding="utf-8"), force=True)
            else:
                workspace.upload_bytes(path, source.read_text(encoding="utf-8").encode("utf-8"))
        workspace.write_text(version_path, agent_version, force=True)
        return True, fallback_paths
    except (OSError, ModuleNotFoundError):
        logger.warning(
            "Failed to sync built-in skills to %s; agent will continue without them.",
            root,
            exc_info=True,
        )
        return False, []


def _collect_skill_paths(lang_dir: Traversable, dest_root: str) -> list[str]:
    """Return workspace paths for every file under *lang_dir*."""
    out: list[str] = []
    for entry in lang_dir.iterdir():
        if entry.name.startswith(("__", ".")):
            continue
        if not entry.is_dir():
            continue
        _collect_paths_recursive(entry, f"{dest_root}/{entry.name}", out)
    return out


def _collect_paths_recursive(source: Traversable, dest_prefix: str, out: list[str]) -> None:
    """Recursively collect workspace paths into *out*."""
    for entry in source.iterdir():
        if entry.name.startswith(("__", ".")):
            continue
        if entry.is_dir():
            _collect_paths_recursive(entry, f"{dest_prefix}/{entry.name}", out)
        else:
            out.append(f"{dest_prefix}/{entry.name}")


def _collect_tree(
    source: Traversable,
    dest_dir: str,
    files: dict[str, Traversable],
) -> None:
    """Collect files by target path, replacing earlier language variants."""
    for entry in source.iterdir():
        if entry.name.startswith(("__", ".")):
            continue
        entry_path = f"{dest_dir}/{entry.name}"
        if entry.is_dir():
            _collect_tree(entry, entry_path, files)
        else:
            files[entry_path] = entry


def seed_builtin_agents_to_workspace(
    workspace: BackendWorkspace,
    *,
    language: str = "en",
    target_dir: str | None = None,
    overwrite: bool = False,
) -> SeedAgentsResult:
    """Copy packaged built-in agent markdown into ``agents/`` (idempotent).

    Existing files are left untouched unless ``overwrite=True``.
    When *language* is not found, falls back to ``"en"`` and records
    fallback paths in ``result.fallback``.
    """
    root = target_dir or DEFAULT_AGENTS_DIR
    result = SeedAgentsResult()
    try:
        package_root: Traversable = resources.files(BUILTIN_AGENTS_PACKAGE)
        lang_dir = package_root.joinpath(language)

        try:
            list(lang_dir.iterdir())
        except (FileNotFoundError, TypeError, OSError):
            logger.warning(
                "Built-in agents language %r not found, falling back to 'en'",
                language,
            )
            lang_dir = package_root.joinpath("en")
            # Collect fallback paths.
            for entry in lang_dir.iterdir():
                if entry.name.startswith(("__", ".", "README")):
                    continue
                if entry.is_file() and entry.name.endswith(".md"):
                    result.fallback.append(f"{root}/{entry.name}")

        _seed_tree_to_workspace(workspace, lang_dir, root, overwrite=overwrite, result=result)
    except (OSError, ModuleNotFoundError):
        logger.warning(
            "Failed to seed built-in agents to %s; agent will continue without them.",
            root,
            exc_info=True,
        )
    return result


def _seed_tree_to_workspace(
    workspace: BackendWorkspace,
    source: Traversable,
    dest_dir: str,
    *,
    overwrite: bool,
    result: SeedAgentsResult,
) -> None:
    for entry in source.iterdir():
        if entry.name.startswith("__"):
            continue
        entry_path = f"{dest_dir}/{entry.name}"
        if entry.is_dir():
            _seed_tree_to_workspace(workspace, entry, entry_path, overwrite=overwrite, result=result)
            continue
        exists = workspace.exists(entry_path)
        resolved = workspace.resolve_path(entry_path)
        if exists and not overwrite:
            result.skipped.append(resolved)
            continue
        workspace.write_text(entry_path, entry.read_text(encoding="utf-8"), force=True)
        if exists:
            result.overwritten.append(resolved)
        else:
            result.created.append(resolved)


__all__ = ["SeedAgentsResult", "seed_builtin_agents_to_workspace", "sync_builtin_skills_to_backend"]
