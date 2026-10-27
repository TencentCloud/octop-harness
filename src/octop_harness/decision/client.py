"""HTTP client for a yes/no decision model.

The body matches the OpenRouter decisions API (also used by Jev): ``questions``
is a map of noul items, and each answer's ``noul`` field is P(yes). A transport
or HTTP failure is retried, then raised so the caller can say the catalog was
left unchanged. A successful body that parses to nothing stays an empty map.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = httpx.Timeout(15.0)
_LAYA_TIMEOUT = httpx.Timeout(60.0)
_ATTEMPTS = 3
_RETRY_DELAY_S = 0.2
# Remote scoring must not stall a chat turn. Local Laya keeps its own read budget.
_REMOTE_BUDGET_S = 5.0


class DecisionRequestError(Exception):
    """The decision endpoint failed after retries. The catalog stays unchanged."""

    def __init__(self, endpoint: str, detail: str) -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(detail)


def _retryable(exc: BaseException) -> bool:
    """Connection drops and 5xx are worth another try. A 4xx or bad body is not."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


def _failure_detail(exc: BaseException) -> str:
    """A user-safe failure reason; do not expose URLs or response bodies."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.TransportError):
        return type(exc).__name__
    return "invalid response"


def _timeout_for(provider: str) -> httpx.Timeout:
    """Local CPU Laya can take about a second per question."""
    return _LAYA_TIMEOUT if provider == "laya" else _DEFAULT_TIMEOUT


def _criteria(text: str) -> dict[str, str]:
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return {"true": "适用", "false": "不适用"}
    return {"true": "Yes, this applies.", "false": "No, this does not apply."}


def build_decision_body(
    *,
    state: str,
    statements: list[tuple[str, str]],
    model: str = "",
) -> dict[str, Any]:
    """Build an OpenRouter-style decisions request."""
    body: dict[str, Any] = {
        "state": state,
        "questions": {
            item_id: {
                "type": "noul",
                "instructions": text,
                "criteria": _criteria(text),
            }
            for item_id, text in statements
        },
    }
    if model:
        body["model"] = model
    return body


def _read_probability(row: object) -> float | None:
    if not isinstance(row, dict):
        return None
    raw = row.get("noul", row.get("probability", row.get("p_yes", row.get("yes"))))
    answer = row.get("answer")
    if raw is None and isinstance(answer, dict):
        raw = answer.get("noul", answer.get("probability", answer.get("p_yes", answer.get("yes"))))
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_probabilities(payload: object) -> dict[str, float]:
    """Read ``id -> P(yes)`` from an OpenRouter answers object or a list of rows."""
    if not isinstance(payload, dict):
        return {}
    answers = payload.get("answers")
    out: dict[str, float] = {}
    if isinstance(answers, dict):
        for item_id, row in answers.items():
            probability = _read_probability(row)
            if probability is not None:
                out[str(item_id)] = probability
        if out:
            return out
    rows = answers if isinstance(answers, list) else payload.get("results") or payload.get("questions") or []
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        item_id = str(row.get("id") or "").strip()
        probability = _read_probability(row)
        if item_id and probability is not None:
            out[item_id] = probability
    return out


class HttpDecisionClient:
    """POST a batch of yes/no questions to ``endpoint``."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str = "",
        model: str = "",
        provider: str = "",
    ) -> None:
        self._endpoint = endpoint.strip()
        self._api_key = api_key
        self._model = model.strip()
        self._provider = provider.strip()
        self._timeout = _timeout_for(self._provider)

    def score(self, *, state: str, statements: list[tuple[str, str]], strict: bool = False) -> dict[str, float]:
        if not statements or not self._endpoint:
            return {}
        deadline = time.monotonic() + self._request_budget_s()

        def call() -> dict[str, float]:
            response = httpx.post(
                self._endpoint,
                json=self._body(state, statements),
                headers=self._headers(),
                timeout=self._remaining_timeout(deadline),
            )
            response.raise_for_status()
            return parse_probabilities(response.json())

        return self._run(call, strict=strict, pause=time.sleep, deadline=deadline)

    async def score_async(
        self,
        *,
        state: str,
        statements: list[tuple[str, str]],
        strict: bool = False,
    ) -> dict[str, float]:
        if not statements or not self._endpoint:
            return {}
        deadline = time.monotonic() + self._request_budget_s()

        async def call() -> dict[str, float]:
            async with httpx.AsyncClient(timeout=self._remaining_timeout(deadline)) as client:
                response = await client.post(
                    self._endpoint,
                    json=self._body(state, statements),
                    headers=self._headers(),
                )
            response.raise_for_status()
            return parse_probabilities(response.json())

        return await self._run_async(call, strict=strict, deadline=deadline)

    def _run(
        self,
        call: Callable[[], dict[str, float]],
        *,
        strict: bool,
        pause: Callable[[float], None],
        deadline: float,
    ) -> dict[str, float]:
        last: BaseException | None = None
        for attempt in range(1, _ATTEMPTS + 1):
            try:
                return call()
            except Exception as exc:
                last = exc
                if strict or not _retryable(exc) or attempt == _ATTEMPTS:
                    self._log_failure(attempt, exc, final=True)
                    if strict:
                        raise
                    raise DecisionRequestError(self._endpoint, _failure_detail(exc)) from exc
                self._log_failure(attempt, exc, final=False)
                pause(min(_RETRY_DELAY_S, self._remaining_seconds(deadline)))
        raise DecisionRequestError(self._endpoint, _failure_detail(last or RuntimeError())) from last

    async def _run_async(
        self,
        call: Callable[[], Awaitable[dict[str, float]]],
        *,
        strict: bool,
        deadline: float,
    ) -> dict[str, float]:
        last: BaseException | None = None
        for attempt in range(1, _ATTEMPTS + 1):
            try:
                return await call()
            except Exception as exc:
                last = exc
                if strict or not _retryable(exc) or attempt == _ATTEMPTS:
                    self._log_failure(attempt, exc, final=True)
                    if strict:
                        raise
                    raise DecisionRequestError(self._endpoint, _failure_detail(exc)) from exc
                self._log_failure(attempt, exc, final=False)
                await asyncio.sleep(min(_RETRY_DELAY_S, self._remaining_seconds(deadline)))
        raise DecisionRequestError(self._endpoint, _failure_detail(last or RuntimeError())) from last

    def _request_budget_s(self) -> float:
        read = self._timeout.read
        limit = float(read) if isinstance(read, int | float) else _REMOTE_BUDGET_S
        if self._provider == "laya":
            return limit
        return min(limit, _REMOTE_BUDGET_S)

    def _remaining_seconds(self, deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())

    def _remaining_timeout(self, deadline: float) -> httpx.Timeout:
        connect_timeout = self._timeout.connect or _REMOTE_BUDGET_S
        return httpx.Timeout(max(0.001, min(self._remaining_seconds(deadline), float(connect_timeout))))

    def _log_failure(self, attempt: int, exc: BaseException, *, final: bool) -> None:
        logger.warning(
            "decision request failed endpoint=%s attempt=%s/%s error=%s",
            self._endpoint,
            attempt,
            _ATTEMPTS,
            type(exc).__name__,
            exc_info=final,
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _body(self, state: str, statements: list[tuple[str, str]]) -> dict[str, Any]:
        return build_decision_body(state=state, statements=statements, model=self._model)
