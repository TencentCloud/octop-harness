"""LLM tool functions for the memory system.

These are registered as agent tools at construction time. They use
``langchain_core.tools.tool`` to produce ``StructuredTool`` instances.

The agent exposes read-only service-backed tools:

* ``memory_search(query, max_results)`` routes through the
  :class:`MemoryService` ranker (atom + page + raw).
* ``memory_get(path, start, lines)`` resolves a virtual path returned by
  ``memory_search``.

Durable writes are not exposed as a tool. The agent persists user-visible
memory by editing the workspace markdown files (USER.md / MEMORY.md /
AGENTS.md). Structured AtomCards are produced asynchronously by the idle
distillation pipeline from the captured raw conversation, not by the LLM
directly.

Each tool wraps the underlying service call in a broad ``except`` so that
storage / DB faults degrade into a human-readable string instead of
crashing the agent turn or leaking a stack trace into the conversation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool, tool

if TYPE_CHECKING:
    from octop_memory import MemoryService

logger = logging.getLogger(__name__)


def build_memory_tools(service: MemoryService) -> list[StructuredTool]:
    """Build :class:`MemoryService`-backed LLM tools.

    Exposes read-only recall tools:

    * ``memory_search(query, max_results)`` — ranked hits across
      atom / entity_page / raw. Each hit carries a virtual ``path`` you
      can pass to ``memory_get``.
    * ``memory_get(path, start, lines)`` — resolve a virtual path to its
      source markdown excerpt.

    Durable writes are intentionally not exposed. Use ``edit_file`` /
    ``append_file`` to update USER.md / MEMORY.md / AGENTS.md;
    AtomCards are produced asynchronously by idle distillation.

    Args:
        service: A :class:`MemoryService` to bind the tools to.
    """

    @tool
    def memory_search(query: str, max_results: int = 5) -> str:
        """Search the agent's memory across atoms, entity pages, raw events.

        Use this when the user references something that happened in a
        previous turn or session, or asks about a fact / preference /
        decision they expect the agent to remember.

        Args:
            query: Natural-language search query.
            max_results: Cap on returned hits (default 5, max ~20).
        """
        try:
            result = service.search(query, max_results=max_results)
        except Exception as exc:  # pragma: no cover - storage / ranker / model fault; pylint: disable=broad-except
            logger.warning("memory_search failed: %s", exc, exc_info=True)
            return f"memory_search failed: {exc.__class__.__name__}"
        hits = result.get("hits", []) if isinstance(result, dict) else []
        if not hits:
            empty_reason = result.get("empty_reason") if isinstance(result, dict) else None
            return f"No matching memory entries. ({empty_reason or 'empty'})"
        lines: list[str] = []
        for h in hits:
            path = h.get("path", "?")
            layer = h.get("layer") or "?"
            snippet = (h.get("snippet") or "").strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            lines.append(f"- [{layer}] {path}\n  {snippet}")
        return "Memory hits:\n" + "\n".join(lines)

    @tool
    def memory_get(path: str, start: int | None = None, lines: int | None = None) -> str:
        """Resolve a virtual memory path to its source text.

        Pass a path returned by ``memory_search`` (e.g.
        ``atom/<id>.md`` / ``page/<entity>.md`` / ``raw/<date>/<id>.md``)
        to read the underlying markdown.

        Args:
            path: The virtual path string from a ``memory_search`` hit.
            start: Optional 1-based line number to start at.
            lines: Optional number of lines to return.
        """
        try:
            result = service.get(path, start=start, lines=lines)
        except Exception as exc:  # pragma: no cover - storage / IO fault; pylint: disable=broad-except
            logger.warning("memory_get failed: %s", exc, exc_info=True)
            return f"memory_get failed: {exc.__class__.__name__}"
        if not isinstance(result, dict):
            return str(result)
        if result.get("error"):
            return f"memory_get error: {result.get('error')}"
        content = result.get("content")
        if isinstance(content, str):
            return content
        return str(result)

    return [memory_search, memory_get]  # type: ignore[list-item]


__all__ = ["build_memory_tools"]
