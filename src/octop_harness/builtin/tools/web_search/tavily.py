"""Tavily AI search tool. Requires ``TAVILY_API_KEY``.

Tavily is an AI-tuned search API that bundles result reranking and an
optional one-shot synthesised answer. The ``langchain-tavily`` package
provides the async client we wrap here.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from octop_harness.tool_env_scope import current_tool_env


@tool
async def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_answer: bool = True,
) -> str:
    """Search the web with Tavily AI and return the structured results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).
        search_depth: ``"basic"`` (cheaper, faster) or ``"advanced"`` —
            advanced does an extra retrieval pass.
        include_answer: When ``True``, Tavily also returns a short
            AI-synthesised answer alongside the result list.

    Returns:
        Either a JSON string with the result payload, or a plain-text
        error if the API key is missing or the package is not
        installed.
    """
    api_key = current_tool_env("TAVILY_API_KEY")
    if not api_key:
        return (
            "Error: TAVILY_API_KEY is not set. Configure a personal search key "
            "in the host application or ask the administrator to set the global key."
        )

    try:
        from langchain_tavily import TavilySearch
    except ImportError:
        return (
            "Error: tavily_search requires the 'langchain-tavily' package. "
            "Install with: pip install 'octop-harness[tavily]' "
            "(or: pip install langchain-tavily)."
        )

    search = TavilySearch(
        max_results=max_results,
        search_depth=search_depth,
        include_answer=include_answer,
        api_key=api_key,
    )
    result = await search.ainvoke({"query": query})
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)
