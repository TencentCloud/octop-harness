"""Custom PII detector for ``PIIMiddleware`` (catches LLM provider API keys).

Why this exists
---------------
``PIIMiddleware`` ships built-in detectors for ``email``/``credit_card``/``ip``/
``mac_address``/``url``, but **not** for API keys. Yet user inputs (and worse,
tool outputs) frequently contain keys — pasted error logs, copy-pasted .env
contents, etc. — which then end up in session logs, traces, and the next
LLM turn's context.

This module fills that gap. ``detect_pii`` recognizes the common public
key formats and reports matches in the ``PIIMatch`` shape ``PIIMiddleware``
consumes. Combine it with ``strategy="mask"`` for the canonical "show the
prefix and last 4 chars, star out the middle" behavior:

    sk-4b829b7b-b0aa-4064-8d53-18b0025594f2
    →
    sk-4********************************5f2

Detected formats
----------------
- OpenAI / OpenAI-compatible: ``sk-...`` (40+ chars), ``sk-proj-...``
- Anthropic: ``sk-ant-...``
- AWS Access Key ID: ``AKIA...`` / ``ASIA...`` (20 chars)
- Google Cloud / Gemini: ``AIzaSy...`` (39 chars)
- HuggingFace: ``hf_...``
- Tencent / Aliyun-style hex secrets after explicit assignment (``api_key=...``)

The pattern intentionally stays conservative: aggressive matching produces
false positives that censor benign hex-y strings (commit hashes, UUIDs).

Future detectors (phone numbers, government IDs, etc.) can be added to
``_PATTERNS`` below — that's why the module is named ``pii`` rather than
``api_key_detector``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain.agents.middleware.pii import PIIMatch

# Each entry is ``(label, compiled_pattern)``. Order matters only for nicer
# ``type`` labels in matches when multiple patterns collide.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # OpenAI project keys: ``sk-proj-XXXX...``
    ("openai_project", re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}")),
    # Anthropic: ``sk-ant-XXXX...``
    ("anthropic", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    # OpenAI / OpenAI-compatible: ``sk-XXXX...`` (catches HAI Hub etc.).
    # Excludes ``sk-proj-`` / ``sk-ant-`` thanks to negative lookahead.
    ("openai", re.compile(r"sk-(?!proj-|ant-)[A-Za-z0-9_-]{20,}")),
    # AWS access key id: AKIA / ASIA + 16 uppercase alnum, exactly 20 chars total.
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    # Google Cloud / Gemini: AIza + 35 chars.
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    # HuggingFace: hf_ + 30+ chars.
    ("huggingface", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    # Generic ``api_key=...`` / ``apikey: ...`` style assignments. Captures the
    # value after the operator; we trim quotes from the captured group below.
    (
        "generic_api_key_assignment",
        re.compile(
            r"""
            (?:api[_\s-]?key|secret[_\s-]?key|access[_\s-]?token)
            \s*[=:]\s*
            ['"]?
            (?P<value>[A-Za-z0-9_\-+/=]{20,})
            ['"]?
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
)


def detect_pii(text: str) -> list[PIIMatch]:
    """Return all suspected PII spans inside ``text``.

    Currently detects LLM provider API keys (see module docstring for the
    list of formats). The output shape matches what ``PIIMiddleware``
    expects from a custom detector:
    ``[{"type": str, "value": str, "start": int, "end": int}, ...]``.
    Spans are returned in left-to-right order; overlapping matches from
    different patterns are de-duplicated by start position (first label wins).
    """
    seen_starts: set[int] = set()
    matches: list[dict[str, object]] = []
    for label, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            # If the pattern uses a named group, prefer that span (so we mask
            # only the secret, not the surrounding ``api_key=`` boilerplate).
            if "value" in m.groupdict():
                start, end = m.span("value")
                value = m.group("value")
            else:
                start, end = m.span()
                value = m.group(0)
            if start in seen_starts:
                continue
            seen_starts.add(start)
            matches.append({"type": label, "value": value, "start": start, "end": end})

    matches.sort(key=lambda m: m["start"])  # type: ignore[arg-type, return-value]
    return matches  # type: ignore[return-value]


__all__ = ["detect_pii"]
