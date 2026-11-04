"""Kimi (Moonshot AI) built-in web search tool.

Kimi's latest generation models (``kimi-k3.1-pro``, ``kimi-k3``, etc.) have
a native ``web_search`` tool that the model invokes itself; we treat the
whole thing as a single black-box "give me an answer with citations" call.
Requires ``MOONSHOT_API_KEY``.

The old ``moonshot-v1`` series (including ``moonshot-v1-128k``) was
officially discontinued on 2026-08-31 and returns 404 for all new API
keys. The default model is now ``kimi-k3.1-pro``; operators can override
via the ``KIMI_MODEL`` environment variable.
"""

from __future__ import annotations

import os
from typing import Any, cast

from langchain_core.tools import tool


@tool
async def kimi_search(query: str) -> str:
    """Ask Kimi (Moonshot) to web-search and synthesise an answer.

    Unlike the other search tools, Kimi returns a *synthesised answer*
    rather than a list of results — its model performs the search
    internally and writes a paragraph with citations.

    Args:
        query: The question to answer using Kimi's built-in web search.

    Returns:
        Either Kimi's synthesised answer, or a plain-text error if the
        API key is missing or the OpenAI SDK isn't installed.
    """
    api_key = os.getenv("MOONSHOT_API_KEY", "")
    if not api_key:
        return "Error: MOONSHOT_API_KEY environment variable is not set."

    try:
        from openai import AsyncOpenAI
    except ImportError:
        return "Error: kimi_search requires the 'openai' package. Install with: pip install openai."

    model_name = os.getenv("KIMI_MODEL", "kimi-k3.1-pro")
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")
    # ``{"type": "web_search"}`` is a Moonshot-specific extension to the
    # tools field; it isn't part of OpenAI's typed schema. ``cast`` keeps
    # mypy quiet without weakening other type checks in this file.
    tools_payload = cast(Any, [{"type": "web_search"}])
    response = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": query}],
        tools=tools_payload,
        temperature=0.3,
    )
    message = response.choices[0].message
    return message.content or "No answer returned."
