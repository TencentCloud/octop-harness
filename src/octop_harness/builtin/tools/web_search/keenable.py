"""Keenable web search. Works without an API key; ``KEENABLE_API_KEY`` is optional.

Without a key, requests use Keenable's public search endpoint
(``POST api.keenable.ai/v1/search/public``, 1000 searches/hour per IP).
With ``KEENABLE_API_KEY`` set, requests use the keyed endpoint
(``POST api.keenable.ai/v1/search``) with higher limits. Uses ``httpx`` only.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from langchain_core.tools import tool

DEFAULT_TIMEOUT = float(os.environ.get("KEENABLE_SEARCH_TIMEOUT", "30"))
FREE_ENDPOINT = "https://api.keenable.ai/v1/search/public"
KEYED_ENDPOINT = "https://api.keenable.ai/v1/search"
APP_TITLE = "octop-harness"

_HTTP_ERRORS = {
    401: "invalid KEENABLE_API_KEY",
    403: "KEENABLE_API_KEY is not allowed for this endpoint",
}


def _web_hits(body: Any, count: int) -> list[dict[str, Any]]:
    if not isinstance(body, dict):
        return []
    hits: list[dict[str, Any]] = []
    for item in body.get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        row: dict[str, Any] = {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": (item.get("snippet") or "").strip(),
        }
        if item.get("published_at"):
            row["published_at"] = item["published_at"]
        hits.append(row)
        if len(hits) >= count:
            break
    return hits


@tool
async def keenable_search(query: str, max_results: int = 5) -> str:
    """Search the web with Keenable and return a JSON array of results.

    Works with no API key. Set ``KEENABLE_API_KEY`` for higher limits.
    Each result has ``title``, ``url``, ``snippet`` (page text) and,
    when known, ``published_at``.

    Args:
        query: The search query string.
        max_results: Max number of results to return (default 5, max 20).

    Returns:
        A JSON string with the results, or a plain-text error message when
        the upstream is unreachable or returns non-2xx.
    """
    count = max(1, min(int(max_results), 20))
    api_key = os.getenv("KEENABLE_API_KEY", "").strip()
    headers = {"X-Keenable-Title": APP_TITLE, "Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    endpoint = KEYED_ENDPOINT if api_key else FREE_ENDPOINT
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(endpoint, json={"query": query, "max_results": count}, headers=headers)
    except httpx.HTTPError as exc:
        return f"Error: keenable request failed: {exc}"

    status = response.status_code
    if status == 429:
        retry_after = response.headers.get("Retry-After")
        hint = f" Retry after {retry_after}s." if retry_after else ""
        if not api_key:
            hint += " Keyless use allows 1000 searches an hour per IP; set KEENABLE_API_KEY for more."
        return f"Error: keenable rate limit reached (HTTP 429).{hint}"
    if status >= 400:
        reason = _HTTP_ERRORS.get(status, "request failed")
        return f"Error: keenable returned HTTP {status} ({reason})."

    try:
        body = response.json()
    except ValueError:
        return "Error: keenable returned a non-JSON response."

    results = _web_hits(body, count)
    return json.dumps(results, ensure_ascii=False) if results else "No results found."
