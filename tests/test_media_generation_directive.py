"""System-prompt guidance about media generation (#1983)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from octop_harness.agent import media_generation_directive


class _Media:
    def __init__(self, has_key: bool) -> None:
        self._has_key = has_key

    def has_api_key(self) -> bool:
        return self._has_key


def _cfg(media: Any) -> Any:
    return SimpleNamespace(media_generation=media)


def test_no_directive_when_media_generation_is_ready() -> None:
    assert media_generation_directive(_cfg(_Media(True))) == ""


def test_unconfigured_wording_is_scoped_to_builtin_tools() -> None:
    """The old text read as "no media tool exists" and blocked connector tools."""
    text = media_generation_directive(_cfg(None))
    assert "built-in" in text
    assert "tool list" in text
    assert "connector" in text
    assert "do not claim that media was generated" in text


def test_missing_credentials_wording_is_scoped_to_builtin_tools() -> None:
    text = media_generation_directive(_cfg(_Media(False)))
    assert "credentials are missing" in text
    assert "built-in" in text
    assert "connector" in text
    assert "do not retry or claim success" in text


def test_directive_never_denies_tools_from_the_request() -> None:
    """Both branches must point the model at its own tool list, not deny it."""
    for media in (None, _Media(False)):
        text = media_generation_directive(_cfg(media))
        assert "use it normally" in text
