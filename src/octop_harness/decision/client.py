"""HTTP client for a yes/no decision model.

The body matches the OpenRouter decisions API (also used by Jev): ``questions``
is a map of noul items, and each answer's ``noul`` field is P(yes). A transport
or parse failure returns an empty map so the caller keeps the current catalog.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = httpx.Timeout(15.0)
_LAYA_TIMEOUT = httpx.Timeout(60.0)
_JEV_MIN_SIDE = 280
_JEV_MAX_SIDE = 1568


def prepare_jev_image_asset(raw_bytes: bytes) -> dict[str, str] | None:
    """Normalize raw image bytes to Jev's 280..1568 px baseline JPEG asset spec."""
    if not raw_bytes:
        return None
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(raw_bytes)) as src:
            img = ImageOps.exif_transpose(src)
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgba = img.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")

            if max(img.size) > _JEV_MAX_SIDE:
                img.thumbnail((_JEV_MAX_SIDE, _JEV_MAX_SIDE), Image.Resampling.LANCZOS)

            w, h = img.size
            if w < _JEV_MIN_SIDE or h < _JEV_MIN_SIDE:
                new_w = max(w, _JEV_MIN_SIDE)
                new_h = max(h, _JEV_MIN_SIDE)
                canvas = Image.new("RGB", (new_w, new_h), (255, 255, 255))
                canvas.paste(img, ((new_w - w) // 2, (new_h - h) // 2))
                img = canvas

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return {"type": "image", "format": "jpg", "data_base64": b64}
    except Exception:
        logger.debug("prepare_jev_image_asset failed", exc_info=True)
        return None


def _timeout_for(provider: str) -> httpx.Timeout:
    """Local CPU Laya can take about a second per question."""
    return _LAYA_TIMEOUT if provider == "laya" else _DEFAULT_TIMEOUT


def _criteria(text: str) -> dict[str, str]:
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return {"true": "适用", "false": "不适用"}
    return {"true": "Yes, this applies.", "false": "No, this does not apply."}


def build_decision_body(
    *,
    state: Any,
    statements: list[tuple[str, str]],
    model: str = "",
    assets: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an OpenRouter/TypeSafe decisions request (with optional image assets)."""
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
    if assets:
        body["assets"] = assets
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

    def _supports_images(self) -> bool:
        if self._provider in ("microsoft", "laya"):
            return False
        if self._provider == "jev":
            return True
        return "jev" in self._model.lower()

    def _effective_state_and_assets(
        self,
        state: str,
        assets: dict[str, dict[str, Any]] | None,
    ) -> tuple[Any, dict[str, dict[str, Any]] | None]:
        if not assets or not self._supports_images():
            return state, None
        ref_lines = [f"[Image {idx + 1}: {ref}]" for idx, ref in enumerate(assets)]
        combined = f"{state}\n" + "\n".join(ref_lines) if state else "\n".join(ref_lines)
        return combined, assets

    def score(
        self,
        *,
        state: str,
        statements: list[tuple[str, str]],
        strict: bool = False,
        assets: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        if not statements or not self._endpoint:
            return {}
        eff_state, eff_assets = self._effective_state_and_assets(state, assets)
        try:
            response = httpx.post(
                self._endpoint,
                json=self._body(eff_state, statements, assets=eff_assets),
                headers=self._headers(has_images=bool(eff_assets)),
                timeout=self._timeout,
            )
            response.raise_for_status()
            return parse_probabilities(response.json())
        except Exception:
            if eff_assets:
                logger.warning(
                    "decision vision request failed endpoint=%s; retrying text-only",
                    self._endpoint,
                    exc_info=True,
                )
                return self.score(state=state, statements=statements, strict=strict, assets=None)
            logger.warning("decision request failed endpoint=%s", self._endpoint, exc_info=True)
            if strict:
                raise
            return {}

    async def score_async(
        self,
        *,
        state: str,
        statements: list[tuple[str, str]],
        strict: bool = False,
        assets: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        if not statements or not self._endpoint:
            return {}
        eff_state, eff_assets = self._effective_state_and_assets(state, assets)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._endpoint,
                    json=self._body(eff_state, statements, assets=eff_assets),
                    headers=self._headers(has_images=bool(eff_assets)),
                )
            response.raise_for_status()
            return parse_probabilities(response.json())
        except Exception:
            if eff_assets:
                logger.warning(
                    "decision vision request failed endpoint=%s; retrying text-only",
                    self._endpoint,
                    exc_info=True,
                )
                return await self.score_async(
                    state=state,
                    statements=statements,
                    strict=strict,
                    assets=None,
                )
            logger.warning("decision request failed endpoint=%s", self._endpoint, exc_info=True)
            if strict:
                raise
            return {}

    def _headers(self, *, has_images: bool = False) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if has_images:
            headers["TypeSafe-Danger-Zone-Images"] = "true"
        return headers

    def _body(
        self,
        state: Any,
        statements: list[tuple[str, str]],
        *,
        assets: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return build_decision_body(
            state=state,
            statements=statements,
            model=self._model,
            assets=assets,
        )
