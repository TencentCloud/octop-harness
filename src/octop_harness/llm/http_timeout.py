"""Shared HTTP timeout policy for LLM provider clients.

Kept outside ``session_header`` so every protocol path (with or without a
session header) uses the same knobs.
"""

from __future__ import annotations

import httpx

# Match OpenAI / Anthropic SDK defaults: a 10-minute overall request budget
# with a short connect bound. Caps hung upstreams without mistaking slow
# first-token / long-context reasoning for a stall. Retries (SDK + harness
# middleware) multiply wall time, so stay generous.
LLM_HTTP_TIMEOUT = httpx.Timeout(timeout=600.0, connect=5.0)

__all__ = ["LLM_HTTP_TIMEOUT"]
