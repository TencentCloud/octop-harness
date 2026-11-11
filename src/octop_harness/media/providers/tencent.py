"""Tencent Cloud TokenHub adapter for Hunyuan image models (image generation only)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import httpx

from octop_harness.media.errors import MediaGenerationError
from octop_harness.media.http import BearerJsonMediaProvider, provider_payload_error
from octop_harness.media.models import (
    ImageGenerationRequest,
    MediaCapabilities,
    MediaJobUpdate,
    MediaSubmission,
    ProviderTask,
    RemoteArtifact,
    VideoGenerationRequest,
)

_IMAGE_SIZE = re.compile(r"^(\d+)[xX](\d+)$")
# Hy-Image-3.5-preview: width and height in [256, 8192], area up to 4096x4096.
_V35_SIDE = (256, 8192)
_V35_MAX_AREA = 16_777_216
# Hy-Image-3.0: width and height in [512, 2048], area up to 1024x1024, at most 3 reference images.
_V3_SIDE = (512, 2048)
_V3_MAX_AREA = 1024 * 1024
_V3_MAX_REFERENCES = 3
_V3_MAX_SEED = 4_294_967_295


class TencentMediaProvider(BearerJsonMediaProvider):
    """Call the TokenHub Hunyuan image endpoints (synchronous, Bearer API key).

    ``hy-image-v3.5*`` models use ``/wand/hunyuan-image/v35-generation`` with a Chat-style
    ``messages`` body and return ``choices[0].delta.image.url``. ``hy-image-v3`` uses
    ``/wand/hunyuan-image/v3-generation`` with ``prompt``/``images`` and returns ``data[].url``.
    Result URLs are signed and expire after 12 hours, so artifacts are downloaded right away.
    One request returns one image, so ``count`` issues that many requests. The custom watermark
    footnote is not mapped; ``watermark`` is ignored. Other TokenHub image models use different
    endpoints and are rejected.
    """

    provider_id = "tencent"
    capabilities = MediaCapabilities(video_generation=False, image_references=True)

    async def _submit_image(
        self,
        client: httpx.AsyncClient,
        request: ImageGenerationRequest,
    ) -> MediaSubmission:
        model = self.image_model
        if model.startswith("hy-image-v3.5"):
            path, body, parse = "/wand/hunyuan-image/v35-generation", _v35_body(model, request), _v35_artifact
        elif model == "hy-image-v3":
            path, body, parse = "/wand/hunyuan-image/v3-generation", _v3_body(model, request), _v3_artifacts
        else:
            raise _unsupported(
                f"TokenHub image model {model!r} is not supported; use hy-image-v3.5-preview or hy-image-v3."
            )
        artifacts: list[RemoteArtifact] = []
        for _ in range(max(request.count, 1)):
            payload = await self._request_json(client, "POST", path, json=body)
            artifacts.extend(parse(payload))
        return MediaSubmission(artifacts=tuple(artifacts))

    async def _submit_video(
        self,
        client: httpx.AsyncClient,
        request: VideoGenerationRequest,
    ) -> MediaSubmission:
        del client, request
        raise _unsupported("The Tencent TokenHub media adapter only generates images.")

    async def _poll_task(
        self,
        client: httpx.AsyncClient,
        task: ProviderTask,
    ) -> MediaJobUpdate:
        del client
        raise MediaGenerationError(
            "Tencent TokenHub image generation has no asynchronous tasks to poll",
            provider_task_id=task.task_id,
        )


def _v35_body(model: str, request: ImageGenerationRequest) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
    content.extend({"type": "image_url", "image_url": {"url": image}} for image in request.reference_images)
    body: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": content}]}
    if request.size:
        body["size"] = _checked_size(request.size, side=_V35_SIDE, max_area=_V35_MAX_AREA, model=model)
    if request.seed is not None:
        if request.seed < 0:
            raise _unsupported("Hy-Image-3.5 seed must not be negative.")
        body["seed"] = request.seed
    return body


def _v3_body(model: str, request: ImageGenerationRequest) -> dict[str, Any]:
    if len(request.reference_images) > _V3_MAX_REFERENCES:
        raise _unsupported(f"Hy-Image-3.0 accepts at most {_V3_MAX_REFERENCES} reference images.")
    body: dict[str, Any] = {"model": model, "prompt": request.prompt}
    if request.reference_images:
        body["images"] = list(request.reference_images)
    if request.size:
        body["size"] = _checked_size(request.size, side=_V3_SIDE, max_area=_V3_MAX_AREA, model=model)
    if request.seed is not None and request.count <= 1:
        if not 1 <= request.seed <= _V3_MAX_SEED:
            raise _unsupported(f"Hy-Image-3.0 seed must be between 1 and {_V3_MAX_SEED}.")
        body["seed"] = request.seed
    return body


def _checked_size(size: str, *, side: tuple[int, int], max_area: int, model: str) -> str:
    match = _IMAGE_SIZE.fullmatch(size.strip())
    if not match:
        raise _unsupported(f"{model} size must be WIDTHxHEIGHT.")
    width, height = int(match.group(1)), int(match.group(2))
    low, high = side
    if not (low <= width <= high and low <= height <= high) or width * height > max_area:
        raise _unsupported(
            f"{model} size must keep width and height in [{low}, {high}] and the area at most {max_area} pixels."
        )
    return f"{width}x{height}"


def _check_payload_error(payload: Mapping[str, Any]) -> None:
    error = payload.get("error")
    if isinstance(error, Mapping) and error:
        raise provider_payload_error(error, fallback="Tencent TokenHub image generation failed")


def _v35_artifact(payload: Mapping[str, Any]) -> list[RemoteArtifact]:
    _check_payload_error(payload)
    choices = payload.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    delta = first.get("delta") if isinstance(first, Mapping) else None
    image = delta.get("image") if isinstance(delta, Mapping) else None
    url = image.get("url") if isinstance(image, Mapping) else None
    if not isinstance(url, str) or not url:
        reason = first.get("finish_reason") if isinstance(first, Mapping) else None
        detail = f" (finish_reason={reason})" if reason else ""
        raise MediaGenerationError(f"Tencent TokenHub image generation returned no image{detail}")
    return [RemoteArtifact(url=url, media_type="image/png", extension="png")]


def _v3_artifacts(payload: Mapping[str, Any]) -> list[RemoteArtifact]:
    _check_payload_error(payload)
    rows = payload.get("data")
    artifacts = [
        RemoteArtifact(url=row["url"], media_type="image/png", extension="png")
        for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, Mapping) and isinstance(row.get("url"), str) and row["url"]
    ]
    if not artifacts:
        raise MediaGenerationError("Tencent TokenHub image generation returned no images")
    return artifacts


def _unsupported(message: str) -> MediaGenerationError:
    return MediaGenerationError(
        message,
        code="unsupported_input",
        category="validation",
        safe_to_resubmit=True,
        remediation="adjust_request",
        model_instruction="Adjust the unsupported arguments, then call the tool at most once more.",
    )


__all__ = ["TencentMediaProvider"]
