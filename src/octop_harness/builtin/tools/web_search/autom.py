"""Autom web search: live Google results. Requires ``AUTOM_API_KEY``.

Autom (https://www.autom.dev) returns the live Google results page for a
query, country and language as JSON
(``GET api.autom.dev/v1/google/search``). Uses ``httpx`` only.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from langchain_core.tools import tool

DEFAULT_TIMEOUT = float(os.environ.get("AUTOM_SEARCH_TIMEOUT", "60"))
ENDPOINT = "https://api.autom.dev/v1/google/search"
USER_AGENT = "autom-integration/tencentcloud-octop-harness"


def _error_message(response: httpx.Response) -> str:
    # Failures arrive as {"error": "message"}.
    try:
        error = response.json().get("error")
    except (ValueError, AttributeError):
        error = None
    message = error if isinstance(error, str) and error else None
    return f"Error: Autom returned HTTP {response.status_code}" + (f" ({message})." if message else ".")


def _organic_hits(body: Any, count: int) -> list[dict[str, Any]]:
    if not isinstance(body, dict):
        return []
    hits: list[dict[str, Any]] = []
    for item in body.get("organic_results") or []:
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
async def autom_search(query: str, max_results: int = 5, country: str = "us", language: str = "") -> str:
    """Search Google with Autom and return a JSON array of live organic results.

    Each result has ``position``, ``title``, ``url`` and ``snippet``.

    Args:
        query: The search query string.
        max_results: Max number of results to return (default 5, at most 10).
        country: 2-letter country code for the location of the results (default ``"us"``).
        language: Optional language code for the results, for example ``"en"`` or ``"fr"``.

    Returns:
        A JSON string with the results, or a plain-text error message when
        the key is missing, the upstream is unreachable or it returns non-2xx.
    """
    api_key = os.getenv("AUTOM_API_KEY", "").strip()
    if not api_key:
        return "Error: AUTOM_API_KEY environment variable is not set."
    count = min(max(1, int(max_results)), 10)
    params = {"query": query, "gl": country.strip().lower()}
    if language.strip():
        params["hl"] = language.strip().lower()
    headers = {"x-api-key": api_key, "User-Agent": USER_AGENT, "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(ENDPOINT, params=params, headers=headers)
    except httpx.HTTPError as exc:
        return f"Error: Autom request failed: {exc}"

    if response.status_code >= 400:
        return _error_message(response)

    try:
        body = response.json()
    except ValueError:
        return "Error: Autom returned a non-JSON response."

    results = _organic_hits(body, count)
    return json.dumps(results, ensure_ascii=False) if results else "No results found."
