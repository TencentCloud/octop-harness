"""Google Custom Search tool. Requires ``GOOGLE_API_KEY`` + ``GOOGLE_CSE_ID``.

Google's Custom Search Engine (CSE) is the supported public API for
programmatic Google search. You need an API key from Google Cloud and
a CSE configured at https://programmablesearchengine.google.com/.
"""

from __future__ import annotations

import asyncio
import json

from langchain_core.tools import tool

from octop_harness.tool_env_scope import current_tool_env


@tool
async def google_search(
    query: str,
    num_results: int = 5,
) -> str:
    """Search the web with Google Custom Search.

    Args:
        query: The search query string.
        num_results: Number of results to return (default 5).

    Returns:
        Either a JSON string with the result list, or an error
        message if env vars are missing or the package is not
        installed.
    """
    api_key = current_tool_env("GOOGLE_API_KEY")
    cse_id = current_tool_env("GOOGLE_CSE_ID")
    missing = [name for name, val in (("GOOGLE_API_KEY", api_key), ("GOOGLE_CSE_ID", cse_id)) if not val]
    if missing:
        return (
            f"Error: missing search credential(s): {', '.join(missing)}. Configure a personal "
            "search key in the host application or ask the administrator to set the global key."
        )

    try:
        from langchain_community.utilities import GoogleSearchAPIWrapper
    except ImportError:
        return (
            "Error: google_search requires the 'langchain-community' and "
            "'google-api-python-client' packages. Install with: "
            "pip install 'octop-harness[google-search]' "
            "(or: pip install langchain-community google-api-python-client)."
        )

    search = GoogleSearchAPIWrapper(
        google_api_key=api_key,
        google_cse_id=cse_id,
        k=num_results,
    )
    results = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: search.results(query, num_results=num_results),
    )
    if not results:
        return "No results found."
    return json.dumps(results, ensure_ascii=False)
