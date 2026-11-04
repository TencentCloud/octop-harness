"""Sync packaged built-in skills and agents into the workspace via :class:`BackendWorkspace`."""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from importlib import resources
from importlib.resources.abc import Traversable
from typing import TYPE_CHECKING

import yaml

from octop_harness.backends.workspace import DEFAULT_AGENTS_DIR, DEFAULT_BUILTIN_SKILLS_DIR

if TYPE_CHECKING:
    from octop_harness.backends.workspace import BackendWorkspace

logger = logging.getLogger(__name__)

VERSION_FILE = ".version"
BUILTIN_SKILLS_PACKAGE = "octop_harness.builtin.skills"
BUILTIN_AGENTS_PACKAGE = "octop_harness.builtin.agents"

# Platform identifiers used in SKILL.md ``compatibility`` frontmatter values
# that indicate macOS-only skills.  Case-insensitive substring match.
_MACOS_TOKENS = ("macos", "darwin", "osx")


def _skill_platform_compatible(skill_dir: Traversable) -> bool:
    """Return True when the skill at *skill_dir* is compatible with this host."""
    skill_md = skill_dir.joinpath("SKILL.md")
    if not skill_md.is_file():
        return True
    try:
        text = skill_md.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, UnicodeError):
        return True
    if not text.startswith("---\n"):
        return True
    end = text.find("\n---", 4)
    if end == -1:
        return True
    raw = text[4:end]
    try:
        meta = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        return True
    if not isinstance(meta, dict):
        return True
    compat_raw = meta.get("compatibility")
    if not compat_raw:
        return True
    compat_str = str(compat_raw).lower()
    is_macos_compat = any(tok in compat_str for tok in _MACOS_TOKENS)
    is_this_macos = platform.system().lower() == "darwin"
    if is_macos_compat and not is_this_macos:
        logger.debug(
            "skipping builtin skill %s: compatibility=%r but host is %s",
            skill_dir.name,
            compat_raw,
            platform.system(),
        )
        return False
    return True


def _skill_entry_filter(entry: Traversable) -> bool:
    """Entry filter: keep everything except macOS-only skill directories."""
    if not entry.is_dir():
        return True
    if not entry.joinpath("SKILL.md").is_file():
        return True
    return _skill_platform_compatible(entry)


def _copy_tree_to_workspace_filtered(
    workspace: "BackendWorkspace",
    source: Traversable,
    dest_dir: str,
    entry_filter,
) -> None:
    """Like :func:`_copy_tree_to_workspace` but skips entries failing *entry_filter*."""
    for entry in source.iterdir():
        if entry.name.startswith(("__", ".")):
            continue
        if not entry_filter(entry):
            continue
        entry_path = f"{dest_dir}/{entry.name}"
        if entry.is_dir():
            _copy_tree_to_workspace_filtered(workspace, entry, entry_path, entry_filter)
        else:
            workspace.write_text(entry_path, entry.read_text(encoding="utf-8"), force=True)


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
            fallback**: the ``"en"`` tree is copied first, then
            *language* files are overlaid on top.  This allows
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
        if _is_up_to_date(workspace, root, agent_version):
            return False, []

        package_root: Traversable = resources.files(BUILTIN_SKILLS_PACKAGE)

        if language == "en":
            # Simple case: just copy "en".
            _copy_tree_to_workspace_filtered(workspace, package_root.joinpath("en"), root, _skill_entry_filter)
            workspace.write_text(f"{root}/{VERSION_FILE}", agent_version, force=True)
            return True, []

        # File-level merge: copy "en" first, then overlay "language".
        lang_dir = package_root.joinpath(language)
        try:
            list(lang_dir.iterdir())
            has_language = True
        except (FileNotFoundError, TypeError, OSError):
            has_language = False

        # Always start with "en" as the base.
        en_dir = package_root.joinpath("en")
        _copy_tree_to_workspace_filtered(workspace, en_dir, root, _skill_entry_filter)

        if has_language:
            # Overlay "language" files on top of "en".
            _copy_tree_to_workspace_filtered(workspace, lang_dir, root, _skill_entry_filter)
        else:
            # No language directory at all -> record all paths as fallback.
            fallback_paths.extend(_collect_skill_paths(en_dir, root))

        workspace.write_text(f"{root}/{VERSION_FILE}", agent_version, force=True)
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
        if entry.joinpath("SKILL.md").is_file() and not _skill_platform_compatible(entry):
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


def _is_up_to_date(workspace: BackendWorkspace, target_dir: str, agent_version: str) -> bool:
    version_path = f"{target_dir}/{VERSION_FILE}"
    if not workspace.exists(version_path):
        return False
    existing = workspace.read_text(version_path)
    if existing is None:
        return False
    return existing.strip() == agent_version


def _copy_tree_to_workspace(
    workspace: BackendWorkspace,
    source: Traversable,
    dest_dir: str,
) -> None:
    """Recursively write a packaged Traversable tree through *workspace*."""
    for entry in source.iterdir():
        if entry.name.startswith(("__", ".")):
            continue
        entry_path = f"{dest_dir}/{entry.name}"
        if entry.is_dir():
            _copy_tree_to_workspace(workspace, entry, entry_path)
        else:
            workspace.write_text(entry_path, entry.read_text(encoding="utf-8"), force=True)


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
