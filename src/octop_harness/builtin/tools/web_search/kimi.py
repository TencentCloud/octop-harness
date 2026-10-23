"""Kimi (Moonshot AI) built-in web search tool.

Kimi exposes a ``$web_search`` *builtin function*: the model decides when
to search, and the caller only has to echo the arguments it produced back
as a ``tool`` message. We treat the whole exchange as a single black-box
"give me an answer with citations" call. Requires ``MOONSHOT_API_KEY``.

The model name is read from ``MOONSHOT_SEARCH_MODEL`` rather than being
pinned in the source: Moonshot stopped serving the ``moonshot-v1-*``
family to newly issued keys, so a hard-coded name turns every fresh key
into a ``resource_not_found_error`` that looks like a permission problem
(see TencentCloud/Octop#1926).
"""

from __future__ import annotations

import os
from typing import Any, cast

from langchain_core.tools import tool

#: Env var that overrides the model Kimi's built-in search runs on, so a
#: Moonshot-side model retirement can be absorbed without a release.
KIMI_MODEL_ENV = "MOONSHOT_SEARCH_MODEL"

#: Used when :data:`KIMI_MODEL_ENV` is unset or blank. ``moonshot-v1-128k``
#: is no longer served to new accounts; ``kimi-k2.6`` is the current
#: general-purpose model (256K context). Use ``kimi-k3`` (1M context) when
#: search results are large enough to risk ``Input token length too long``.
DEFAULT_KIMI_MODEL = "kimi-k2.6"

#: Upper bound on search rounds. The model normally finishes in one; the
#: cap only exists so a model that keeps re-issuing ``$web_search`` cannot
#: spin forever.
_MAX_SEARCH_ROUNDS = 5


def kimi_search_model() -> str:
    """Resolve the Moonshot model used by :func:`kimi_search`.

    Returns ``MOONSHOT_SEARCH_MODEL`` when it is set to something
    non-blank, otherwise :data:`DEFAULT_KIMI_MODEL`.
    """
    return os.getenv(KIMI_MODEL_ENV, "").strip() or DEFAULT_KIMI_MODEL


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

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")
    # ``$web_search`` is declared as a ``builtin_function`` — a plain
    # ``function`` name may not contain ``$``. This isn't part of OpenAI's
    # typed schema, so ``cast`` keeps mypy quiet without weakening other
    # type checks in this file.
    tools_payload = cast(
        Any,
        [{"type": "builtin_function", "function": {"name": "$web_search"}}],
    )
    messages: list[Any] = [{"role": "user", "content": query}]
    for _ in range(_MAX_SEARCH_ROUNDS):
        response = await client.chat.completions.create(
            model=kimi_search_model(),
            messages=messages,
            tools=tools_payload,
        )
        choice = response.choices[0]
        if choice.finish_reason != "tool_calls":
            return choice.message.content or "No answer returned."
        # Echoing the assistant turn back verbatim is what makes Moonshot
        # actually run the search: the arguments it produced *are* the
        # result. K2.x/K3 also expect any ``reasoning_content`` to be
        # replayed, which appending the whole message preserves.
        messages.append(choice.message)
        for call in choice.message.tool_calls or []:
            # The OpenAI types union function calls with custom-tool calls, and
            # the latter carry no ``function``. Kimi's built-in ``$web_search``
            # only ever produces function calls, so anything else is skipped
            # rather than read through an attribute it does not have.
            function = getattr(call, "function", None)
            if function is None:
                continue
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": function.name,
                    "content": function.arguments or "{}",
                },
            )
    return "Error: kimi_search exceeded the web-search round limit."
