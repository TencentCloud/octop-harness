"""Tests for ``octop_harness.builtin.tools.web_search``."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from octop_harness.builtin.tools.web_search import load_web_search_tools
from octop_harness.builtin.tools.web_search._registry import (
    WEB_SEARCH_PROVIDERS,
    _missing_env,
    _provider_available,
)
from octop_harness.builtin.tools.web_search.brave import brave_search
from octop_harness.builtin.tools.web_search.deepseek import deepseek_fetch, deepseek_search
from octop_harness.builtin.tools.web_search.google import google_search
from octop_harness.builtin.tools.web_search.kimi import kimi_search
from octop_harness.builtin.tools.web_search.searchfree import searchfree_search
from octop_harness.builtin.tools.web_search.tavily import tavily_search

# ---------------------------------------------------------------------------
# Registry — provider lookup, env-var gating, policy dispatch
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_known_providers(self) -> None:
        assert set(WEB_SEARCH_PROVIDERS) == {
            "tavily",
            "brave",
            "google",
            "kimi",
            "searchfree",
            "deepseek_search",
            "deepseek_fetch",
        }

    def test_provider_available_all_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FOO", "x")
        monkeypatch.setenv("BAR", "y")
        assert _provider_available(("FOO", "BAR")) is True

    def test_provider_available_one_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FOO", "x")
        monkeypatch.delenv("BAR", raising=False)
        assert _provider_available(("FOO", "BAR")) is False

    def test_missing_env_lists_blanks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FOO", "x")
        monkeypatch.setenv("BAR", "")
        monkeypatch.delenv("BAZ", raising=False)
        assert _missing_env(("FOO", "BAR", "BAZ")) == ["BAR", "BAZ"]


class TestLoadWebSearchTools:
    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "TAVILY_API_KEY",
            "BRAVE_API_KEY",
            "GOOGLE_API_KEY",
            "GOOGLE_CSE_ID",
            "MOONSHOT_API_KEY",
            "DEEPSEEK_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_false_returns_empty(self) -> None:
        assert load_web_search_tools(False) == []

    def test_auto_with_no_env_still_includes_searchfree(self) -> None:
        # ``searchfree`` has no env requirement, so it always loads under
        # ``auto`` even when every other provider is unconfigured.
        names_auto = {t.name for t in load_web_search_tools("auto")}
        names_true = {t.name for t in load_web_search_tools(True)}
        assert names_auto == {"searchfree_search"}
        assert names_true == {"searchfree_search"}

    def test_auto_loads_only_configured_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tav")
        monkeypatch.setenv("MOONSHOT_API_KEY", "kim")
        loaded = load_web_search_tools("auto")
        names = {t.name for t in loaded}
        # ``searchfree`` rides along regardless because its tuple is empty.
        assert names == {"tavily_search", "kimi_search", "searchfree_search"}

    def test_all_loads_every_provider(self) -> None:
        loaded = load_web_search_tools("all")
        names = {t.name for t in loaded}
        assert names == {
            "tavily_search",
            "brave_search",
            "google_search",
            "kimi_search",
            "searchfree_search",
            "deepseek_search",
            "deepseek_fetch",
        }

    def test_explicit_list_loads_named_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tav")
        monkeypatch.setenv("BRAVE_API_KEY", "brv")
        loaded = load_web_search_tools(["tavily", "brave"])
        names = {t.name for t in loaded}
        assert names == {"tavily_search", "brave_search"}

    def test_explicit_list_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown web-search provider"):
            load_web_search_tools(["bogus"])

    def test_explicit_list_missing_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match=r"env var.*missing"):
            load_web_search_tools(["tavily"])


# ---------------------------------------------------------------------------
# tavily_search
# ---------------------------------------------------------------------------


class TestTavilySearch:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_error(self) -> None:
        result = await tavily_search.ainvoke({"query": "x"})
        assert isinstance(result, str)
        assert "TAVILY_API_KEY" in result

    @pytest.mark.asyncio
    async def test_calls_sdk_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "secret")

        captured: dict[str, Any] = {}

        class _FakeTavily:
            def __init__(self, **kwargs: Any) -> None:
                captured["init"] = kwargs

            async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
                captured["payload"] = payload
                return {"answer": "42", "results": []}

        fake_module = ModuleType("langchain_tavily")
        fake_module.TavilySearch = _FakeTavily  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"langchain_tavily": fake_module}):
            result = await tavily_search.ainvoke(
                {"query": "what is the answer", "max_results": 3, "search_depth": "advanced"},
            )
        parsed = json.loads(result)
        assert parsed == {"answer": "42", "results": []}
        assert captured["init"]["api_key"] == "secret"
        assert captured["init"]["max_results"] == 3
        assert captured["init"]["search_depth"] == "advanced"
        assert captured["payload"] == {"query": "what is the answer"}

    @pytest.mark.asyncio
    async def test_missing_package_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "secret")
        with patch.dict(sys.modules, {"langchain_tavily": None}):
            result = await tavily_search.ainvoke({"query": "x"})
        assert "langchain-tavily" in result


# ---------------------------------------------------------------------------
# brave_search
# ---------------------------------------------------------------------------


class TestBraveSearch:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_error(self) -> None:
        result = await brave_search.ainvoke({"query": "x"})
        assert "BRAVE_API_KEY" in result

    @pytest.mark.asyncio
    async def test_calls_sdk_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAVE_API_KEY", "secret")
        captured: dict[str, Any] = {}

        class _FakeBraveSearch:
            @classmethod
            def from_api_key(cls, *, api_key: str, search_kwargs: dict[str, Any]) -> _FakeBraveSearch:
                captured["api_key"] = api_key
                captured["search_kwargs"] = search_kwargs
                return cls()

            def run(self, query: str) -> str:
                captured["query"] = query
                return "raw brave result"

        fake_tools_module = ModuleType("langchain_community.tools")
        fake_tools_module.BraveSearch = _FakeBraveSearch  # type: ignore[attr-defined]
        # Building a fake parent package so the ``from langchain_community.tools import …``
        # import works.
        fake_pkg = ModuleType("langchain_community")
        fake_pkg.tools = fake_tools_module  # type: ignore[attr-defined]
        with patch.dict(
            sys.modules,
            {"langchain_community": fake_pkg, "langchain_community.tools": fake_tools_module},
        ):
            result = await brave_search.ainvoke({"query": "hello", "count": 7})
        assert result == "raw brave result"
        assert captured == {
            "api_key": "secret",
            "search_kwargs": {"count": 7},
            "query": "hello",
        }

    @pytest.mark.asyncio
    async def test_missing_package_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAVE_API_KEY", "secret")
        with patch.dict(sys.modules, {"langchain_community.tools": None}):
            result = await brave_search.ainvoke({"query": "x"})
        assert "langchain-community" in result


# ---------------------------------------------------------------------------
# google_search
# ---------------------------------------------------------------------------


class TestGoogleSearch:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)

    @pytest.mark.asyncio
    async def test_missing_both_env_vars_lists_them(self) -> None:
        result = await google_search.ainvoke({"query": "x"})
        assert "GOOGLE_API_KEY" in result
        assert "GOOGLE_CSE_ID" in result

    @pytest.mark.asyncio
    async def test_missing_one_env_var_lists_only_that_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "k")
        result = await google_search.ainvoke({"query": "x"})
        assert "GOOGLE_CSE_ID" in result
        assert "GOOGLE_API_KEY" not in result

    @pytest.mark.asyncio
    async def test_calls_sdk_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "k")
        monkeypatch.setenv("GOOGLE_CSE_ID", "c")
        captured: dict[str, Any] = {}

        class _FakeWrapper:
            def __init__(self, **kwargs: Any) -> None:
                captured["init"] = kwargs

            def results(self, query: str, num_results: int) -> list[dict[str, Any]]:
                captured["call"] = (query, num_results)
                return [{"title": "Hello", "link": "https://example.com"}]

        fake_utilities = ModuleType("langchain_community.utilities")
        fake_utilities.GoogleSearchAPIWrapper = _FakeWrapper  # type: ignore[attr-defined]
        fake_pkg = ModuleType("langchain_community")
        fake_pkg.utilities = fake_utilities  # type: ignore[attr-defined]
        with patch.dict(
            sys.modules,
            {"langchain_community": fake_pkg, "langchain_community.utilities": fake_utilities},
        ):
            result = await google_search.ainvoke({"query": "hi", "num_results": 3})
        parsed = json.loads(result)
        assert parsed == [{"title": "Hello", "link": "https://example.com"}]
        assert captured["init"]["google_api_key"] == "k"
        assert captured["init"]["google_cse_id"] == "c"
        assert captured["call"] == ("hi", 3)


# ---------------------------------------------------------------------------
# kimi_search
# ---------------------------------------------------------------------------


class TestKimiSearch:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_error(self) -> None:
        result = await kimi_search.ainvoke({"query": "x"})
        assert "MOONSHOT_API_KEY" in result

    @pytest.mark.asyncio
    async def test_calls_sdk_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MOONSHOT_API_KEY", "secret")
        captured: dict[str, Any] = {}

        # Build a fake ``openai`` module exposing ``AsyncOpenAI``.
        class _FakeChatCompletions:
            async def create(self, **kwargs: Any) -> Any:
                captured["call"] = kwargs
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
                )

        class _FakeChat:
            def __init__(self) -> None:
                self.completions = _FakeChatCompletions()

        class _FakeAsyncOpenAI:
            def __init__(self, **kwargs: Any) -> None:
                captured["init"] = kwargs
                self.chat = _FakeChat()

        fake_module = ModuleType("openai")
        fake_module.AsyncOpenAI = _FakeAsyncOpenAI  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"openai": fake_module}):
            result = await kimi_search.ainvoke({"query": "hello"})
        assert result == "answer"
        assert captured["init"]["api_key"] == "secret"
        assert captured["init"]["base_url"].startswith("https://api.moonshot.cn")
        assert captured["call"]["model"].startswith("moonshot-")
        assert any(t.get("type") == "web_search" for t in captured["call"]["tools"])


# ---------------------------------------------------------------------------
# searchfree (zero-config public fallback)
# ---------------------------------------------------------------------------


class TestSearchFreeSearch:
    @pytest.mark.asyncio
    async def test_returns_results_array_when_wrapped(self) -> None:
        class _Resp:
            status_code = 200

            def json(self) -> dict[str, Any]:
                return {"results": [{"title": "Hi", "url": "https://example.com"}]}

        class _FakeClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> _FakeClient:
                return self

            async def __aexit__(self, *args: object) -> None:
                """No-op."""

            async def post(self, url: str, json: dict[str, Any]) -> _Resp:
                assert url.endswith("/api/search")
                assert json["query"] == "hello"
                return _Resp()

        from octop_harness.builtin.tools.web_search import searchfree as mod

        with patch.object(mod.httpx, "AsyncClient", _FakeClient):
            result = await searchfree_search.ainvoke({"query": "hello"})
        parsed = json.loads(result)
        assert parsed == [{"title": "Hi", "url": "https://example.com"}]

    @pytest.mark.asyncio
    async def test_returns_body_when_no_results_key(self) -> None:
        class _Resp:
            status_code = 200

            def json(self) -> list[dict[str, str]]:
                return [{"title": "Bare list"}]

        class _FakeClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> _FakeClient:
                return self

            async def __aexit__(self, *args: object) -> None:
                """No-op."""

            async def post(self, url: str, json: dict[str, Any]) -> _Resp:
                return _Resp()

        from octop_harness.builtin.tools.web_search import searchfree as mod

        with patch.object(mod.httpx, "AsyncClient", _FakeClient):
            result = await searchfree_search.ainvoke({"query": "hello"})
        parsed = json.loads(result)
        assert parsed == [{"title": "Bare list"}]

    @pytest.mark.asyncio
    async def test_http_error_returns_plain_text(self) -> None:
        class _Resp:
            status_code = 503

            def json(self) -> dict[str, Any]:
                return {}

        class _FakeClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> _FakeClient:
                return self

            async def __aexit__(self, *args: object) -> None:
                """No-op."""

            async def post(self, url: str, json: dict[str, Any]) -> _Resp:
                return _Resp()

        from octop_harness.builtin.tools.web_search import searchfree as mod

        with patch.object(mod.httpx, "AsyncClient", _FakeClient):
            result = await searchfree_search.ainvoke({"query": "hi"})
        assert "HTTP 503" in result

    @pytest.mark.asyncio
    async def test_network_failure_returns_plain_text(self) -> None:
        from octop_harness.builtin.tools.web_search import searchfree as mod

        class _FakeClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> _FakeClient:
                return self

            async def __aexit__(self, *args: object) -> None:
                """No-op."""

            async def post(self, url: str, json: dict[str, Any]) -> Any:
                raise mod.httpx.ConnectError("connection refused")

        with patch.object(mod.httpx, "AsyncClient", _FakeClient):
            result = await searchfree_search.ainvoke({"query": "hi"})
        assert "searchfree request failed" in result


# ---------------------------------------------------------------------------
# deepseek_search / deepseek_fetch (first-party DeepSeek Search + Fetch APIs)
# ---------------------------------------------------------------------------


def _fake_deepseek_client(
    captured: dict[str, Any],
    *,
    status_code: int = 200,
    body: Any = None,
) -> type:
    """Build a fake ``httpx.AsyncClient`` recording posts to DeepSeek."""

    class _Resp:
        def __init__(self) -> None:
            self.status_code = status_code

        def json(self) -> Any:
            if isinstance(body, Exception):
                raise body
            return body

        @property
        def text(self) -> str:
            return "" if body is None else str(body)

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            """No-op."""

        async def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> _Resp:
            captured.setdefault("calls", []).append(
                {"url": url, "headers": headers, "json": json},
            )
            return _Resp()

    return _FakeClient


class TestDeepSeekSearch:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_error(self) -> None:
        result = await deepseek_search.ainvoke({"query": "x"})
        assert "DEEPSEEK_API_KEY" in result

    @pytest.mark.asyncio
    async def test_returns_results_and_strips_empty_content(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-secret")
        captured: dict[str, Any] = {}
        results = [
            {
                "title": "DeepSeek V4",
                "url": "https://example.com/v4",
                "snippet": "V4 is out",
                "summary": "DeepSeek V4 released",
                "rerank_score": 2.4,
                "content": "",
                "authority": 3,
            },
            {
                "title": "With body",
                "url": "https://example.com/body",
                "snippet": "s",
                "summary": "sum",
                "rerank_score": 1.8,
                "content": "full page text",
                "authority": 2,
            },
        ]
        client = _fake_deepseek_client(
            captured,
            body={"results": results},
        )
        from octop_harness.builtin.tools.web_search import deepseek as mod

        with patch.object(mod.httpx, "AsyncClient", client):
            result = await deepseek_search.ainvoke(
                {"query": "DeepSeek V4 release", "max_results": 3, "ori_question": "最新模型是啥"},
            )
        parsed = json.loads(result)
        assert "content" not in parsed[0]
        assert parsed[1]["content"] == "full page text"
        call = captured["calls"][0]
        assert call["url"].endswith("/v1/tools/search")
        assert call["headers"]["Authorization"] == "Bearer ds-secret"
        assert call["json"]["count"] == 3
        assert call["json"]["need_content"] is False
        assert call["json"]["ori_question"] == "最新模型是啥"

    @pytest.mark.asyncio
    async def test_omits_ori_question_when_blank_and_clamps_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-secret")
        captured: dict[str, Any] = {}
        client = _fake_deepseek_client(captured, body={"results": [{"title": "t"}]})
        from octop_harness.builtin.tools.web_search import deepseek as mod

        with patch.object(mod.httpx, "AsyncClient", client):
            await deepseek_search.ainvoke({"query": "q", "max_results": 999})
        call = captured["calls"][0]
        assert "ori_question" not in call["json"]
        assert call["json"]["count"] == 50

    @pytest.mark.asyncio
    async def test_empty_results_returns_no_results_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-secret")
        captured: dict[str, Any] = {}
        client = _fake_deepseek_client(captured, body={"results": []})
        from octop_harness.builtin.tools.web_search import deepseek as mod

        with patch.object(mod.httpx, "AsyncClient", client):
            result = await deepseek_search.ainvoke({"query": "q"})
        assert result == "No results found."

    @pytest.mark.asyncio
    async def test_upstream_error_includes_api_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-secret")
        captured: dict[str, Any] = {}
        client = _fake_deepseek_client(
            captured,
            status_code=401,
            body={"error": {"message": "auth failed", "type": "authentication_error"}},
        )
        from octop_harness.builtin.tools.web_search import deepseek as mod

        with patch.object(mod.httpx, "AsyncClient", client):
            result = await deepseek_search.ainvoke({"query": "q"})
        assert "HTTP 401" in result
        assert "auth failed" in result

    @pytest.mark.asyncio
    async def test_network_failure_returns_plain_text(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-secret")
        from octop_harness.builtin.tools.web_search import deepseek as mod

        class _BrokenClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> _BrokenClient:
                return self

            async def __aexit__(self, *args: object) -> None:
                """No-op."""

            async def post(self, *args: Any, **kwargs: Any) -> Any:
                raise mod.httpx.ConnectError("connection reset")

        with patch.object(mod.httpx, "AsyncClient", _BrokenClient):
            result = await deepseek_search.ainvoke({"query": "q"})
        assert "deepseek_search request failed" in result


class TestDeepSeekFetch:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_error(self) -> None:
        result = await deepseek_fetch.ainvoke({"url": "https://example.com"})
        assert "DEEPSEEK_API_KEY" in result

    @pytest.mark.asyncio
    async def test_returns_title_and_markdown_content(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-secret")
        captured: dict[str, Any] = {}
        client = _fake_deepseek_client(
            captured,
            body={"title": "Example", "content": "# Example\n\nBody."},
        )
        from octop_harness.builtin.tools.web_search import deepseek as mod

        with patch.object(mod.httpx, "AsyncClient", client):
            result = await deepseek_fetch.ainvoke({"url": "https://example.com/article"})
        assert json.loads(result) == {"title": "Example", "content": "# Example\n\nBody."}
        call = captured["calls"][0]
        assert call["url"].endswith("/v1/tools/fetch")
        assert call["json"] == {"url": "https://example.com/article"}

    @pytest.mark.asyncio
    async def test_not_found_error_is_surfaced(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-secret")
        captured: dict[str, Any] = {}
        client = _fake_deepseek_client(
            captured,
            status_code=404,
            body={"error": {"message": "no extractable content", "type": "invalid_request_error"}},
        )
        from octop_harness.builtin.tools.web_search import deepseek as mod

        with patch.object(mod.httpx, "AsyncClient", client):
            result = await deepseek_fetch.ainvoke({"url": "https://example.com/404"})
        assert "HTTP 404" in result
        assert "no extractable content" in result

    @pytest.mark.asyncio
    async def test_empty_content_is_an_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-secret")
        captured: dict[str, Any] = {}
        client = _fake_deepseek_client(captured, body={"title": "", "content": ""})
        from octop_harness.builtin.tools.web_search import deepseek as mod

        with patch.object(mod.httpx, "AsyncClient", client):
            result = await deepseek_fetch.ainvoke({"url": "https://example.com"})
        assert "no usable content" in result


# ---------------------------------------------------------------------------
# Integration: HarnessAgentConfig.web_search_tools
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_default_is_auto(self) -> None:
        from octop_harness import HarnessAgentConfig, ModelConfig, ProviderConfig

        cfg = HarnessAgentConfig(
            providers=[
                ProviderConfig(
                    id="p",
                    base_url="http://x",
                    api_key="x",
                    models=[ModelConfig(id="m", input=["text"])],
                ),
            ],
        )
        assert cfg.web_search_tools == "auto"

    def test_round_trips_through_to_dict(self) -> None:
        from octop_harness import HarnessAgentConfig, ModelConfig, ProviderConfig

        cfg = HarnessAgentConfig(
            providers=[
                ProviderConfig(
                    id="p",
                    base_url="http://x",
                    api_key="x",
                    models=[ModelConfig(id="m", input=["text"])],
                ),
            ],
            web_search_tools=["tavily", "kimi"],
        )
        data = cfg.to_dict()
        assert data["web_search_tools"] == ["tavily", "kimi"]
        restored = HarnessAgentConfig.from_dict(data)
        assert restored.web_search_tools == ["tavily", "kimi"]

    def test_round_trip_auto(self) -> None:
        from octop_harness import HarnessAgentConfig, ModelConfig, ProviderConfig

        cfg = HarnessAgentConfig(
            providers=[
                ProviderConfig(
                    id="p",
                    base_url="http://x",
                    api_key="x",
                    models=[ModelConfig(id="m", input=["text"])],
                ),
            ],
            web_search_tools="auto",
        )
        assert cfg.to_dict()["web_search_tools"] == "auto"
        restored = HarnessAgentConfig.from_dict(cfg.to_dict())
        assert restored.web_search_tools == "auto"


def test_load_web_search_tools_invokes_lazy_loader_only_when_needed() -> None:
    """``langchain_tavily`` etc. must NOT be imported just by importing the registry."""
    # Sanity: importing the registry should not have pulled the optional deps in.
    assert "langchain_tavily" not in sys.modules
    # And explicitly clear before we test.
    sys.modules.pop("langchain_tavily", None)
    # ``"all"`` policy imports the *@tool wrappers* but each tool only imports
    # its SDK at invocation time, so the SDK module still shouldn't be present.
    tools = load_web_search_tools("all")
    assert any(t.name == "tavily_search" for t in tools)
    assert "langchain_tavily" not in sys.modules
