"""OpenAI Images API adapter for GPT Image models (image generation only)."""

from __future__ import annotations

import base64
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

_IMAGE_SIZE = re.compile(r"^\d+[xX]\d+$")


class OpenAIMediaProvider(BearerJsonMediaProvider):
    """Call the OpenAI Images API (``/images/generations`` and ``/images/edits``).

    Text-only prompts use ``/images/generations``. Requests with reference images use
    ``/images/edits`` with JSON ``images: [{"image_url": ...}]`` entries, so URLs and data URLs
    are passed through without a multipart upload. GPT Image models return base64 data. The
    Images API has no seed or watermark parameters, so those request fields are ignored.
    """

    provider_id = "openai"
    capabilities = MediaCapabilities(video_generation=False, image_references=True)

    async def _submit_image(
        self,
        client: httpx.AsyncClient,
        request: ImageGenerationRequest,
    ) -> MediaSubmission:
        body: dict[str, Any] = {
            "model": self.image_model,
            "prompt": request.prompt,
            "n": request.count,
        }
        if request.size:
            size = request.size.strip()
            if size != "auto" and not _IMAGE_SIZE.fullmatch(size):
                raise _unsupported("OpenAI image size must be WIDTHxHEIGHT (for example 1024x1024) or auto.")
            body["size"] = size.lower()
        path = "/images/generations"
        if request.reference_images:
            body["images"] = [{"image_url": image} for image in request.reference_images]
            path = "/images/edits"

        payload = await self._request_json(client, "POST", path, json=body)
        error = payload.get("error")
        if isinstance(error, Mapping) and error:
            raise provider_payload_error(error, fallback="OpenAI image generation failed")
        rows = payload.get("data")
        if not isinstance(rows, list) or not rows:
            raise MediaGenerationError("OpenAI image generation returned no images")

        artifacts: list[RemoteArtifact] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            encoded = row.get("b64_json")
            if isinstance(encoded, str) and encoded:
                try:
                    data = base64.b64decode(encoded, validate=True)
                except ValueError as exc:
                    raise MediaGenerationError("OpenAI returned invalid base64 image data") from exc
                artifacts.append(RemoteArtifact(data=data, media_type="image/png", extension="png"))
                continue
            url = row.get("url")
            if isinstance(url, str) and url:
                artifacts.append(RemoteArtifact(url=url, media_type="image/png", extension="png"))
        if not artifacts:
            raise MediaGenerationError("OpenAI image generation returned no usable images")
        return MediaSubmission(artifacts=tuple(artifacts))

    async def _submit_video(
        self,
        client: httpx.AsyncClient,
        request: VideoGenerationRequest,
    ) -> MediaSubmission:
        del client, request
        raise _unsupported("The OpenAI media adapter only generates images.")

    async def _poll_task(
        self,
        client: httpx.AsyncClient,
        task: ProviderTask,
    ) -> MediaJobUpdate:
        del client
        raise MediaGenerationError(
            "OpenAI image generation has no asynchronous tasks to poll",
            provider_task_id=task.task_id,
        )


def _unsupported(message: str) -> MediaGenerationError:
    return MediaGenerationError(
        message,
        code="unsupported_input",
        category="validation",
        safe_to_resubmit=True,
        remediation="adjust_request",
        model_instruction="Adjust the unsupported arguments, then call the tool at most once more.",
    )


__all__ = ["OpenAIMediaProvider"]
