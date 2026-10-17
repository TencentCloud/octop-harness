"""DeepSeek first-party web tools: Search + Fetch. Requires ``DEEPSEEK_API_KEY``.

DeepSeek's search stack (https://api-docs.deepseek.com/guides/search) exposes
two read-only JSON endpoints under ``https://api.deepseek.com``:

- ``POST /v1/tools/search`` — DeepSeek's self-built search engine. Returns
  reranked results (``rerank_score`` ≥ 1.5 marks a good hit) with snippet +
  summary, and optional full page content when ``need_content`` is true.
- ``POST /v1/tools/fetch`` — fetches one URL and returns its body as
  Markdown (``{title, content}``).

Both authenticate with a regular DeepSeek API key (Bearer) and are billed
per successful request. Each endpoint is rate-limited to 5 QPS per account;
``408``/``429``/``5xx`` are transient and safe to retry.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from langchain_core.tools import tool

DEFAULT_TIMEOUT = float(os.environ.get("DEEPSEEK_TOOLS_TIMEOUT", "30"))
DEFAULT_BASE_URL = os.environ.get(
    "DEEPSEEK_TOOLS_BASE_URL",
    "https://api.deepseek.com",
).rstrip("/")

_SEARCH_MAX_RESULTS_CAP = 50


def _error_message(response: httpx.Response) -> str:
    """Human-readable message from an OpenAI-style error body."""
    detail = ""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            detail = str(error["message"])
    if detail:
        return f"HTTP {response.status_code}: {detail}"
    return f"HTTP {response.status_code}"


def _strip_empty_content(result: dict[str, Any]) -> dict[str, Any]:
    """Drop the ``content`` key when the API returned no page body.

    With ``need_content=false`` every hit carries ``content: ""``; keeping
    those empty strings just wastes the model's context window.
    """
    if not result.get("content"):
        return {key: value for key, value in result.items() if key != "content"}
    return result


@tool
async def deepseek_search(
    query: str,
    max_results: int = 5,
    need_content: bool = False,
    ori_question: str = "",
) -> str:
    """Search the web with DeepSeek's self-built search engine.

    Supports advanced operators in ``query``: ``site:`` / ``-site:``,
    ``-"excluded phrase"``, ``"exact phrase"``, ``lang:zh,en`` and
    ``after:``/``before:`` date ranges (e.g. ``after:2025-01-01``).

    Args:
        query: The search query string (1-128 characters).
        max_results: Maximum number of results, clamped to 1-50 (default 5).
        need_content: When ``True``, each result also carries the full page
            text — costs extra tokens; leave off unless needed.
        ori_question: Optional original user question (helps intent
            understanding when ``query`` is a rewritten search string).

    Returns:
        A JSON array of result objects (``title``, ``url``, ``snippet``,
        ``summary``, ``rerank_score``, ``authority``, ...) or a plain-text
        error message.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return "Error: DEEPSEEK_API_KEY environment variable is not set."

    payload: dict[str, Any] = {
        "query": query,
        "count": max(1, min(max_results, _SEARCH_MAX_RESULTS_CAP)),
        "need_content": need_content,
    }
    if ori_question.strip():
        payload["ori_question"] = ori_question.strip()

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{DEFAULT_BASE_URL}/v1/tools/search",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    except httpx.HTTPError as exc:
        return f"Error: deepseek_search request failed: {exc}"

    if response.status_code >= 400:
        return f"Error: deepseek_search returned {_error_message(response)}."

    try:
        body = response.json()
    except ValueError:
        return "Error: deepseek_search returned a non-JSON response."

    results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(results, list) or not results:
        return "No results found."
    return json.dumps(
        [_strip_empty_content(item) for item in results if isinstance(item, dict)],
        ensure_ascii=False,
    )


@tool
async def deepseek_fetch(url: str) -> str:
    """Fetch a single web page via DeepSeek and return its content as Markdown.

    Args:
        url: Absolute HTTP/HTTPS URL of the page to fetch (no credentials
            in the URL; max length 8192).

    Returns:
        A JSON object ``{"title": ..., "content": ...}`` where ``content``
        is the page body in Markdown, or a plain-text error message.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return "Error: DEEPSEEK_API_KEY environment variable is not set."

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{DEFAULT_BASE_URL}/v1/tools/fetch",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"url": url},
            )
    except httpx.HTTPError as exc:
        return f"Error: deepseek_fetch request failed: {exc}"

    if response.status_code >= 400:
        return f"Error: deepseek_fetch returned {_error_message(response)}."

    try:
        body = response.json()
    except ValueError:
        return "Error: deepseek_fetch returned a non-JSON response."

    if not isinstance(body, dict) or not body.get("content"):
        return "Error: deepseek_fetch returned no usable content."
    return json.dumps(body, ensure_ascii=False)
