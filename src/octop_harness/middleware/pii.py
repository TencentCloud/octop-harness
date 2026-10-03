"""Custom PII detector for provider API keys and Chinese personal information.

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
- Mainland China mobile numbers (11 digits, optional ``+86`` / ``0086`` prefix)
- 18-character Chinese resident IDs with a valid date and checksum

The pattern intentionally stays conservative: aggressive matching produces
false positives that censor benign hex-y strings (commit hashes, UUIDs).

Personal-number detection is limited to these formats; it does not verify
identity, allocation, or ownership. Legacy 15-digit IDs are not detected.
"""

from __future__ import annotations

import re
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain.agents.middleware.pii import PIIMatch

# Each entry is ``(label, compiled_pattern)``. Existing key patterns take
# precedence over personal-number matches inside their spans.
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
    (
        "cn_mobile_phone",
        re.compile(r"(?<![A-Za-z\d_])(?:(?:\+86|0086)[ -]?)?1[3-9][0-9]{9}(?![A-Za-z\d_])"),
    ),
    (
        "cn_resident_id",
        re.compile(r"(?<![A-Za-z\d_])[1-9][0-9]{5}[12][0-9]{10}[0-9Xx](?![A-Za-z\d_])"),
    ),
)


def _valid_resident_id(value: str) -> bool:
    """Validate the date, sequence and MOD 11-2 check digit of an 18-digit ID."""
    try:
        date(int(value[6:10]), int(value[10:12]), int(value[12:14]))
    except ValueError:
        return False
    if value[14:17] == "000":
        return False
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    checksum = sum(int(digit) * weight for digit, weight in zip(value[:17], weights, strict=True))
    return value[-1].upper() == "10X98765432"[checksum % 11]


def detect_pii(text: str) -> list[PIIMatch]:
    """Return all suspected PII spans inside ``text``.

    Detects provider API keys, mainland mobile numbers and resident IDs
    (see module docstring for supported formats). The output shape matches what ``PIIMiddleware``
    expects from a custom detector:
    ``[{"type": str, "value": str, "start": int, "end": int}, ...]``.
    Spans are returned in left-to-right order; overlapping matches from
    key patterns with the same start are de-duplicated (first label wins).
    Personal-number matches overlapping an API key are omitted.
    """
    seen_spans: list[tuple[int, int]] = []
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
            if label == "cn_resident_id" and not _valid_resident_id(value):
                continue
            if any(
                start == previous_start
                or (label in ("cn_mobile_phone", "cn_resident_id") and start < previous_end and end > previous_start)
                for previous_start, previous_end in seen_spans
            ):
                continue
            seen_spans.append((start, end))
            matches.append({"type": label, "value": value, "start": start, "end": end})

    matches.sort(key=lambda m: m["start"])  # type: ignore[arg-type, return-value]
    return matches  # type: ignore[return-value]


__all__ = ["detect_pii"]
