"""Gemini Interactions API adapter for Nano Banana image models (image generation only)."""

from __future__ import annotations

import base64
import binascii
import math
import mimetypes
import re
from collections.abc import Mapping
from typing import Any

import httpx

from octop_harness.media.errors import MediaGenerationError
from octop_harness.media.http import BearerJsonMediaProvider
from octop_harness.media.models import (
    ImageGenerationRequest,
    MediaCapabilities,
    MediaJobUpdate,
    MediaSubmission,
    ProviderTask,
    RemoteArtifact,
    VideoGenerationRequest,
)

# Interactions API revision the request and response shapes below follow.
GEMINI_API_REVISION = "2026-05-20"

_DATA_URL = re.compile(r"^data:(?P<mime>image/[\w.+-]+);base64,(?P<data>.+)$", re.DOTALL)
_IMAGE_SIZE = re.compile(r"^(\d+)[xX](\d+)$")
_ASPECT_RATIO = re.compile(r"^\d+:\d+$")
_RESOLUTION = re.compile(r"^(?:512|0\.5[kK]|[124][kK])$")


class GeminiMediaProvider(BearerJsonMediaProvider):
    """Call the Gemini Interactions API (``POST /interactions``) for native image generation.

    The prompt and reference images form the ``input`` list (``uri`` for HTTP(S) URLs, inline
    ``data`` for data URLs), and ``response_format`` requests an image. ``size`` may be an aspect
    ratio (``16:9``), a resolution tier (``1K``/``2K``/``4K``), or ``WIDTHxHEIGHT``, which is reduced
    to its aspect ratio. One request returns one image, so ``count`` issues that many requests.
    The API has no seed or watermark parameters, so those request fields are ignored.
    """

    provider_id = "gemini"
    capabilities = MediaCapabilities(video_generation=False, image_references=True)

    def _auth_headers(self) -> dict[str, str]:
        headers = super()._auth_headers()
        api_key = headers.pop("Authorization").removeprefix("Bearer ")
        headers["x-goog-api-key"] = api_key
        headers["Api-Revision"] = GEMINI_API_REVISION
        return headers

    async def _submit_image(
        self,
        client: httpx.AsyncClient,
        request: ImageGenerationRequest,
    ) -> MediaSubmission:
        body: dict[str, Any] = {
            "model": self.image_model,
            "input": [
                {"type": "text", "text": request.prompt},
                *(_image_input(image) for image in request.reference_images),
            ],
            "response_format": _response_format(request.size),
        }
        artifacts: list[RemoteArtifact] = []
        for _ in range(max(request.count, 1)):
            payload = await self._request_json(client, "POST", "/interactions", json=body)
            artifacts.extend(_image_artifacts(payload))
        return MediaSubmission(artifacts=tuple(artifacts))

    async def _submit_video(
        self,
        client: httpx.AsyncClient,
        request: VideoGenerationRequest,
    ) -> MediaSubmission:
        del client, request
        raise _unsupported("The Gemini media adapter only generates images.")

    async def _poll_task(
        self,
        client: httpx.AsyncClient,
        task: ProviderTask,
    ) -> MediaJobUpdate:
        del client
        raise MediaGenerationError(
            "Gemini image generation has no asynchronous tasks to poll",
            provider_task_id=task.task_id,
        )


def _response_format(size: str | None) -> dict[str, str]:
    response_format = {"type": "image"}
    if not size:
        return response_format
    value = size.strip()
    if _ASPECT_RATIO.fullmatch(value):
        response_format["aspect_ratio"] = value
    elif _RESOLUTION.fullmatch(value):
        response_format["image_size"] = "512" if value in {"512", "0.5k", "0.5K"} else value.upper()
    elif match := _IMAGE_SIZE.fullmatch(value):
        width, height = int(match.group(1)), int(match.group(2))
        if width <= 0 or height <= 0:
            raise _unsupported("Gemini image size must use positive WIDTHxHEIGHT values.")
        divisor = math.gcd(width, height)
        response_format["aspect_ratio"] = f"{width // divisor}:{height // divisor}"
    else:
        raise _unsupported("Gemini image size must be an aspect ratio (16:9), 1K/2K/4K, or WIDTHxHEIGHT.")
    return response_format


def _image_input(image: str) -> dict[str, str]:
    match = _DATA_URL.fullmatch(image.strip())
    if match:
        return {"type": "image", "data": match.group("data"), "mime_type": match.group("mime")}
    mime_type = mimetypes.guess_type(image.split("?", 1)[0])[0] or "image/png"
    return {"type": "image", "uri": image, "mime_type": mime_type}


def _image_artifacts(payload: Mapping[str, Any]) -> list[RemoteArtifact]:
    status = payload.get("status")
    if status not in {None, "completed"}:
        raise MediaGenerationError(f"Gemini image generation ended with status {status!r}")
    artifacts: list[RemoteArtifact] = []
    texts: list[str] = []
    steps = payload.get("steps")
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, Mapping) or step.get("type") != "model_output":
            continue
        content = step.get("content")
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                texts.append(block["text"])
                continue
            if block.get("type") != "image":
                continue
            mime_type = block.get("mime_type")
            media_type = mime_type if isinstance(mime_type, str) and mime_type else "image/png"
            extension = media_type.split("/", 1)[-1].replace("jpeg", "jpg")
            data = block.get("data")
            if isinstance(data, str) and data:
                try:
                    decoded = base64.b64decode(data, validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise MediaGenerationError("Gemini returned invalid base64 image data") from exc
                artifacts.append(RemoteArtifact(data=decoded, media_type=media_type, extension=extension))
                continue
            uri = block.get("uri")
            if isinstance(uri, str) and uri:
                artifacts.append(RemoteArtifact(url=uri, media_type=media_type, extension=extension))
    if not artifacts:
        reason = " ".join(texts).strip()
        message = "Gemini image generation returned no image"
        raise MediaGenerationError(f"{message}: {reason[:300]}" if reason else message)
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


__all__ = ["GEMINI_API_REVISION", "GeminiMediaProvider"]
