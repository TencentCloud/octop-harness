"""cloro web search: live Google results. Requires ``CLORO_API_KEY``.

cloro (https://cloro.dev) loads the live Google results page for a query
and country and returns the organic results
(``POST api.cloro.dev/v1/monitor/google``). Uses ``httpx`` only.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from langchain_core.tools import tool

# cloro loads the live results page during the request, so allow more time than a search index needs.
DEFAULT_TIMEOUT = float(os.environ.get("CLORO_SEARCH_TIMEOUT", "120"))
ENDPOINT = "https://api.cloro.dev/v1/monitor/google"
USER_AGENT = "cloro-integration/tencentcloud-octop-harness"


def _error_message(response: httpx.Response) -> str:
    # Failures arrive as {"success": false, "error": {"code": "...", "message": "..."}}.
    try:
        error = response.json().get("error") or {}
    except (ValueError, AttributeError):
        error = {}
    message = error.get("message") if isinstance(error, dict) else None
    return f"Error: cloro returned HTTP {response.status_code}" + (f" ({message})." if message else ".")


def _organic_hits(body: Any, count: int) -> list[dict[str, Any]]:
    if not isinstance(body, dict):
        return []
    organic = (body.get("result") or {}).get("organicResults") or []
    hits: list[dict[str, Any]] = []
    for item in organic:
        if not isinstance(item, dict):
            continue
        url = item.get("link") or ""
        if not url.startswith(("http://", "https://")):
            continue
        hits.append(
            {
                "position": len(hits) + 1,
                "title": item.get("title") or "",
                "url": url,
                "snippet": item.get("snippet") or "",
            }
        )
        if len(hits) >= count:
            break
    return hits


@tool
async def cloro_search(query: str, max_results: int = 5, country: str = "US") -> str:
    """Search Google with cloro and return a JSON array of live organic results.

    Each result has ``position``, ``title``, ``url`` and ``snippet``.

    Args:
        query: The search query string.
        max_results: Max number of results to return (default 5).
        country: 2-letter ISO 3166-1 code for the location of the results (default ``"US"``).

    Returns:
        A JSON string with the results, or a plain-text error message when
        the key is missing, the upstream is unreachable or it returns non-2xx.
    """
    api_key = os.getenv("CLORO_API_KEY", "").strip()
    if not api_key:
        return "Error: CLORO_API_KEY environment variable is not set."
    count = max(1, int(max_results))
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(
                ENDPOINT, json={"query": query, "country": country.strip().upper()}, headers=headers
            )
    except httpx.HTTPError as exc:
        return f"Error: cloro request failed: {exc}"

    if response.status_code >= 400:
        return _error_message(response)

    try:
        body = response.json()
    except ValueError:
        return "Error: cloro returned a non-JSON response."

    results = _organic_hits(body, count)
    return json.dumps(results, ensure_ascii=False) if results else "No results found."
