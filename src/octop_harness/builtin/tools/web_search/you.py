"""You.com web search. Works without an API key; ``YDC_API_KEY`` is optional.

Without a key, requests go to the free ``api.you.com/v1/agents/search``
endpoint (100 searches a day, counted per IP). With ``YDC_API_KEY`` set,
requests go to the keyed ``ydc-index.io/v1/search`` endpoint with higher
limits. Uses ``httpx`` only, no extra dependencies.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from langchain_core.tools import tool

DEFAULT_TIMEOUT = float(os.environ.get("YOU_SEARCH_TIMEOUT", "30"))
FREE_ENDPOINT = "https://api.you.com/v1/agents/search"
KEYED_ENDPOINT = "https://ydc-index.io/v1/search"
USER_AGENT = "youdotcom-integration/tencentcloud-octop-harness"

_HTTP_ERRORS = {
    401: "invalid YDC_API_KEY",
    402: "You.com API credits exhausted",
    403: "YDC_API_KEY is not allowed for this endpoint",
}


@tool
async def you_search(query: str, max_results: int = 5) -> str:
    """Search the web with You.com and return a JSON array of results.

    Works with no API key (free tier). Set ``YDC_API_KEY`` for higher limits.
    Each result has ``title``, ``url``, ``description``, ``snippets`` and,
    when known, ``page_age``.

    Args:
        query: The search query string.
        max_results: Max number of results to return (default 5, max 20).

    Returns:
        A JSON string with the results, or a plain-text error message when
        the upstream is unreachable or returns non-2xx.
    """
    count = max(1, min(int(max_results), 20))
    api_key = os.getenv("YDC_API_KEY", "").strip()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            if api_key:
                headers["X-API-Key"] = api_key
                response = await client.post(KEYED_ENDPOINT, json={"query": query, "count": count}, headers=headers)
            else:
                response = await client.get(FREE_ENDPOINT, params={"query": query, "count": count}, headers=headers)
    except httpx.HTTPError as exc:
        return f"Error: you.com request failed: {exc}"

    status = response.status_code
    if status == 429:
        retry_after = response.headers.get("Retry-After")
        hint = f" Retry after {retry_after}s." if retry_after else ""
        if not api_key:
            hint += " The free tier allows 100 searches a day; set YDC_API_KEY for more."
        return f"Error: you.com rate limit reached (HTTP 429).{hint}"
    if status >= 400:
        reason = _HTTP_ERRORS.get(status, "request failed")
        return f"Error: you.com returned HTTP {status} ({reason})."

    try:
        body = response.json()
    except ValueError:
        return "Error: you.com returned a non-JSON response."

    web = (body.get("results") or {}).get("web") or [] if isinstance(body, dict) else []
    results: list[dict[str, Any]] = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "description": item.get("description", ""),
            "snippets": item.get("snippets") or [],
            **({"page_age": item["page_age"]} if item.get("page_age") else {}),
        }
        for item in web[:count]
        if isinstance(item, dict) and item.get("url")
    ]
    return json.dumps(results, ensure_ascii=False) if results else "No results found."
