"""Tencent Cloud TokenHub adapters for image, video, and 3D generation."""

from __future__ import annotations

import base64
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx

from octop_harness.media.errors import MediaGenerationError
from octop_harness.media.http import (
    BearerJsonMediaProvider,
    classified_provider_error,
    provider_payload_error,
)
from octop_harness.media.models import (
    ImageGenerationRequest,
    MediaCapabilities,
    MediaJobUpdate,
    MediaSubmission,
    ProviderTask,
    RemoteArtifact,
    ThreeDGenerationRequest,
    VideoGenerationRequest,
)

_IMAGE_SIZE = re.compile(r"^(\d+)[xX*](\d+)$")
_ASPECT_RATIO = re.compile(r"^(\d+):(\d+)$")
_HY_VIDEO_RATIOS = frozenset({"16:9", "9:16", "1:1", "4:3", "3:4"})
_WAND_VIDEO_RATIOS = frozenset({"21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "adaptive"})
_KLING_RATIOS = frozenset({"16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9"})
_VIDU_IMAGE_RATIOS = frozenset({"16:9", "9:16", "1:1", "3:4", "4:3", "21:9", "2:3", "3:2", "auto"})
_VIDU_VIDEO_RATIOS = frozenset({"16:9", "9:16", "1:1", "3:4", "4:3"})
_PIXVERSE_V56_RATIOS = frozenset({"16:9", "4:3", "1:1", "3:4", "9:16"})
_PIXVERSE_V6_RATIOS = frozenset({"16:9", "4:3", "1:1", "3:4", "9:16", "2:3", "3:2", "21:9"})
_HIDREAM_SIZES: dict[str, str] = {
    "1:1": "2048x2048",
    "4:3": "2304x1728",
    "3:4": "1728x2304",
    "16:9": "2560x1440",
    "9:16": "1440x2560",
    "3:2": "2496x1664",
    "2:3": "1664x2496",
}
_HIDREAM_EXACT_SIZES = frozenset(_HIDREAM_SIZES.values())
_THREE_D_FORMAT_TYPES: dict[str, tuple[str, str]] = {
    "glb": ("model/gltf-binary", "glb"),
    "obj": ("model/obj", "obj"),
    "stl": ("model/stl", "stl"),
    "usdz": ("model/vnd.usdz+zip", "usdz"),
    "fbx": ("application/octet-stream", "fbx"),
    "3mf": ("model/3mf", "3mf"),
    "mp4": ("video/mp4", "mp4"),
}
_IMAGE_EXT_TYPES: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}
_HI3D_FORMATS: dict[str, int] = {
    "obj": 1,
    "glb": 2,
    "stl": 3,
    "fbx": 4,
    "usdz": 5,
    "3mf": 6,
}
_TEX_RES_VALUES = frozenset({"2k", "4k", "8k"})
_VIDU_I2V_ONLY = frozenset({"vidu-video-q4-preview", "vidu-video-q3", "vidu-video-q2-pro", "vidu-video-q2-turbo"})
_KLING_OMNI_ONLY = frozenset({"kling-video-v3-omni", "kling-video-o1"})


class TencentTokenHubMediaProvider(BearerJsonMediaProvider):
    """Call Tencent Cloud TokenHub multimodal image, video, and 3D HTTP APIs."""

    provider_id = "tencent-tokenhub"
    capabilities = MediaCapabilities(
        image_generation=True,
        video_generation=True,
        image_references=True,
        video_first_frame=True,
        video_last_frame=True,
        video_references=True,
        video_audio=True,
        three_d_generation=True,
        three_d_text_to_3d=True,
        three_d_image_to_3d=True,
        three_d_multi_view=True,
        three_d_pbr=True,
    )

    def _v1_path(self, subpath: str) -> str:
        base = self._config.base_url.rstrip("/")
        return subpath if base.endswith("/v1") else f"/v1{subpath}"

    # ------------------------------------------------------------------
    # Image submission
    # ------------------------------------------------------------------

    async def _submit_image(
        self,
        client: httpx.AsyncClient,
        request: ImageGenerationRequest,
    ) -> MediaSubmission:
        family = _image_family(self.image_model)
        handlers = {
            "hy-v35": self._submit_image_hy_v35,
            "hy-v3": self._submit_image_hy_v3,
            "wand": self._submit_image_wand,
            "vidu": self._submit_image_vidu,
            "seedream": self._submit_image_seedream,
            "kling": self._submit_image_kling,
            "hidream": self._submit_image_hidream,
            "minimax": self._submit_image_minimax,
        }
        handler = handlers.get(family)
        if handler is None:
            raise _unsupported(f"Unsupported TokenHub image model: {self.image_model}")
        return await handler(client, request)

    async def _submit_image_hy_v35(
        self,
        client: httpx.AsyncClient,
        request: ImageGenerationRequest,
    ) -> MediaSubmission:
        if request.count > 1:
            raise _unsupported("Hy-Image-3.5 returns one image per request; set count=1.")
        content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": _normalize_data_url_mime(image)}}
            for image in request.reference_images
        )
        body: dict[str, Any] = {
            "model": self.image_model,
            "messages": [{"role": "user", "content": content}],
        }
        if request.size:
            body["size"] = _normalize_hy_image_size(request.size, is_v35=True)
        if request.seed is not None:
            body["seed"] = _validate_non_negative_seed(request.seed)

        payload = await self._request_json(
            client,
            "POST",
            self._v1_path("/wand/hunyuan-image/v35-generation"),
            json=body,
        )
        _raise_tokenhub_payload_error(payload, fallback="TokenHub image generation failed")
        url = _extract_v35_image_url(payload)
        return MediaSubmission(artifacts=(RemoteArtifact(url=url, media_type="image/png", extension="png"),))

    async def _submit_image_hy_v3(
        self,
        client: httpx.AsyncClient,
        request: ImageGenerationRequest,
    ) -> MediaSubmission:
        if request.count > 1:
            raise _unsupported("Hy-Image-3.0 returns one image per request; set count=1.")
        if len(request.reference_images) > 3:
            raise _unsupported("Hy-Image-3.0 accepts at most 3 reference images.")
        body: dict[str, Any] = {"model": self.image_model, "prompt": request.prompt}
        if request.reference_images:
            body["images"] = [_normalize_data_url_mime(image) for image in request.reference_images]
        if request.size:
            body["size"] = _normalize_hy_image_size(request.size, is_v35=False)
        if request.seed is not None:
            body["seed"] = _validate_non_negative_seed(request.seed)

        payload = await self._request_json(
            client,
            "POST",
            self._v1_path("/wand/hunyuan-image/v3-generation"),
            json=body,
        )
        _raise_tokenhub_payload_error(payload, fallback="TokenHub image generation failed")
        return _require_image_artifacts(_extract_data_image_artifacts(payload))

    async def _submit_image_wand(
        self,
        client: httpx.AsyncClient,
        request: ImageGenerationRequest,
    ) -> MediaSubmission:
        if request.count > 1:
            raise _unsupported("WAND-Vega image models return one image per request; set count=1.")
        if len(request.reference_images) > 10:
            raise _unsupported("WAND-Vega image models accept at most 10 reference images.")
        body: dict[str, Any] = {"model": self.image_model, "prompt": request.prompt}
        if request.reference_images:
            body["input"] = [
                {
                    "content": [
                        {"type": "input_image", "image_url": _normalize_data_url_mime(img)}
                        for img in request.reference_images
                    ]
                }
            ]
        if request.size:
            body["size"] = _normalize_hy_image_size(request.size, is_v35=True)

        payload = await self._request_json(
            client,
            "POST",
            self._v1_path("/wand/vega-images/generations"),
            json=body,
        )
        _raise_tokenhub_payload_error(payload, fallback="TokenHub WAND image generation failed")
        task_id = _require_str(payload.get("task_id"), "TokenHub WAND image generation did not return a task_id")
        return MediaSubmission(
            task=ProviderTask(task_id=task_id, kind="image", model=self.image_model, metadata={"family": "wand"})
        )

    async def _submit_image_vidu(
        self,
        client: httpx.AsyncClient,
        request: ImageGenerationRequest,
    ) -> MediaSubmission:
        if request.count > 1:
            raise _unsupported("Vidu image models return one image per request; set count=1.")
        if len(request.reference_images) > 7:
            raise _unsupported("Vidu-Image-q2 accepts at most 7 reference images.")
        body: dict[str, Any] = {"model": self.image_model, "prompt": request.prompt}
        if request.reference_images:
            body["images"] = [_normalize_data_url_mime(img) for img in request.reference_images]
        if request.size:
            ratio, tier = _infer_ratio_and_tier(request.size)
            if ratio in _VIDU_IMAGE_RATIOS:
                body["aspect_ratio"] = ratio
            body["resolution"] = {"4k": "4K", "2k": "2K"}.get(tier, "1080p")
        if request.seed is not None:
            body["seed"] = _validate_non_negative_seed(request.seed)

        payload = await self._request_json(
            client,
            "POST",
            self._v1_path("/wand/vidu-image/generation"),
            json=body,
        )
        _raise_tokenhub_payload_error(payload, fallback="TokenHub Vidu image generation failed")
        task_id = _require_str(payload.get("task_id"), "TokenHub Vidu image generation did not return a task_id")
        return MediaSubmission(
            task=ProviderTask(task_id=task_id, kind="image", model=self.image_model, metadata={"family": "vidu"})
        )

    async def _submit_image_seedream(
        self,
        client: httpx.AsyncClient,
        request: ImageGenerationRequest,
    ) -> MediaSubmission:
        if request.count > 1:
            raise _unsupported("TokenHub Seedream image models return one image per request; set count=1.")
        max_refs = 14 if "v5.0-lite" in self.image_model else 10
        if len(request.reference_images) > max_refs:
            raise _unsupported(f"{self.image_model} accepts at most {max_refs} reference images.")
        body: dict[str, Any] = {
            "model": self.image_model,
            "prompt": request.prompt,
            "response_format": "url",
            "watermark": request.watermark,
        }
        if request.reference_images:
            body["images"] = [_normalize_data_url_mime(img) for img in request.reference_images]
        if request.size:
            cleaned = request.size.strip()
            if cleaned.upper() in {"1K", "1.5K", "2K", "3K", "4K"}:
                body["size"] = cleaned.upper()
            else:
                body["size"] = _normalize_image_size(cleaned)

        payload = await self._request_json(
            client,
            "POST",
            self._v1_path("/wand/si-image/generation"),
            json=body,
        )
        _raise_tokenhub_payload_error(payload, fallback="TokenHub Seedream image generation failed")
        return _require_image_artifacts(_extract_data_image_artifacts(payload))

    async def _submit_image_kling(
        self,
        client: httpx.AsyncClient,
        request: ImageGenerationRequest,
    ) -> MediaSubmission:
        model = self.image_model.lower()
        if model.startswith("kling-image-v4"):
            path, body = _build_kling_v4_image_request(self.image_model, request)
        else:
            path, body = _build_kling_v3_image_request(self.image_model, request)

        payload = await self._request_json(client, "POST", self._v1_path(path), json=body)
        _raise_tokenhub_payload_error(payload, fallback="TokenHub Kling image generation failed")
        data = payload.get("data")
        task_id = data.get("id") or data.get("task_id") if isinstance(data, Mapping) else None
        resolved_id = _require_str(task_id, "TokenHub Kling image generation did not return a task id")
        return MediaSubmission(
            task=ProviderTask(task_id=resolved_id, kind="image", model=self.image_model, metadata={"family": "kling"})
        )

    async def _submit_image_hidream(
        self,
        client: httpx.AsyncClient,
        request: ImageGenerationRequest,
    ) -> MediaSubmission:
        if request.count > 1:
            raise _unsupported("HiDream image models return one image per request; set count=1.")
        is_edit = self.image_model.lower().endswith("-edit")
        if is_edit:
            if not request.reference_images:
                raise _unsupported("hidream-image-o1-1.5-edit requires 1 to 6 reference images.")
            if len(request.reference_images) > 6:
                raise _unsupported("hidream-image-o1-1.5-edit accepts at most 6 reference images.")
            path = "/wand/hidream-image/images/edits"
        else:
            if request.reference_images:
                raise _unsupported(
                    "hidream-image-o1-1.5 is text-to-image only; use hidream-image-o1-1.5-edit for image editing."
                )
            path = "/wand/hidream-image/images/generations"

        body: dict[str, Any] = {"model": self.image_model, "prompt": request.prompt}
        if is_edit:
            body["images"] = [_normalize_data_url_mime(img) for img in request.reference_images]
        if request.size:
            mapped = _map_hidream_size(request.size)
            if mapped:
                body["size"] = mapped

        payload = await self._request_json(client, "POST", self._v1_path(path), json=body)
        _raise_tokenhub_payload_error(payload, fallback="TokenHub HiDream image generation failed")
        return _require_image_artifacts(_extract_data_image_artifacts(payload))

    async def _submit_image_minimax(
        self,
        client: httpx.AsyncClient,
        request: ImageGenerationRequest,
    ) -> MediaSubmission:
        if request.count > 10:
            raise _unsupported("TokenHub MiniMax image models accept count between 1 and 10.")
        if len(request.reference_images) > 16:
            raise _unsupported("TokenHub MiniMax image edit accepts at most 16 reference images.")
        body: dict[str, Any] = {
            "model": self.image_model,
            "prompt": request.prompt,
            "n": request.count,
            "response_format": "url",
            "aigc_watermark": request.watermark,
        }
        if request.size:
            cleaned = request.size.strip()
            if _ASPECT_RATIO.fullmatch(cleaned):
                body["aspect_ratio"] = cleaned
            elif cleaned.upper() in {"1K", "2K", "4K"}:
                body["area"] = cleaned.upper()
            else:
                body["size"] = _normalize_image_size(cleaned)

        if request.reference_images:
            body["images"] = [{"image_url": _normalize_data_url_mime(img)} for img in request.reference_images]
            path = "/wand/minimax-image/edits"
        else:
            path = "/wand/minimax-image/generation"

        payload = await self._request_json(client, "POST", self._v1_path(path), json=body)
        _raise_tokenhub_payload_error(payload, fallback="TokenHub MiniMax image generation failed")
        return _require_image_artifacts(_extract_data_image_artifacts(payload))

    # ------------------------------------------------------------------
    # Video submission
    # ------------------------------------------------------------------

    async def _submit_video(
        self,
        client: httpx.AsyncClient,
        request: VideoGenerationRequest,
    ) -> MediaSubmission:
        family = _video_family(self.video_model)
        if family in {"minimax-v2", "minimax-v1"}:
            return await self._submit_video_minimax(client, request, family=family)
        handlers = {
            "hy": self._submit_video_hy,
            "wand": self._submit_video_wand,
            "kling": self._submit_video_kling,
            "vidu": self._submit_video_vidu,
            "pixverse": self._submit_video_pixverse,
            "yt": self._submit_video_yt,
        }
        handler = handlers.get(family)
        if handler is None:
            raise _unsupported(f"Unsupported TokenHub video model: {self.video_model}")
        return await handler(client, request)

    async def _submit_video_hy(
        self,
        client: httpx.AsyncClient,
        request: VideoGenerationRequest,
    ) -> MediaSubmission:
        if request.last_frame:
            raise _unsupported("Tencent Cloud TokenHub Hy-Video does not support last_frame.")
        if request.generate_audio:
            raise _unsupported("Tencent Cloud TokenHub Hy-Video does not support generate_audio.")
        images = ([request.first_frame] if request.first_frame else []) + list(request.reference_images)
        if len(images) > 1:
            raise _unsupported("Tencent Cloud TokenHub Hy-Video accepts at most one input image.")

        body: dict[str, Any] = {
            "model": self.video_model,
            "prompt": request.prompt,
            "duration": 5,
            "resolution": "720p",
        }
        if images:
            image = images[0]
            if image.startswith(("http://", "https://")):
                body["image_url"] = image
            else:
                body["image"] = _normalize_data_url_mime(image)
        else:
            if request.aspect_ratio not in _HY_VIDEO_RATIOS:
                raise _unsupported("Tencent Cloud TokenHub Hy-Video aspect_ratio must be 16:9, 9:16, 1:1, 4:3, or 3:4.")
            body["aspect_ratio"] = request.aspect_ratio
        if request.seed is not None:
            body["seed"] = request.seed

        payload = await self._request_json(
            client,
            "POST",
            self._v1_path("/wand/hunyuan-video/generation"),
            json=body,
        )
        _raise_tokenhub_payload_error(payload, fallback="TokenHub video generation failed")
        task_id = _require_str(payload.get("task_id"), "TokenHub video generation did not return a task_id")
        return MediaSubmission(
            task=ProviderTask(task_id=task_id, kind="video", model=self.video_model, metadata={"family": "hy"})
        )

    async def _submit_video_wand(
        self,
        client: httpx.AsyncClient,
        request: VideoGenerationRequest,
    ) -> MediaSubmission:
        content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
        if request.first_frame:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _normalize_data_url_mime(request.first_frame)},
                    "role": "first_frame",
                }
            )
        if request.last_frame:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _normalize_data_url_mime(request.last_frame)},
                    "role": "last_frame",
                }
            )
        if request.reference_images:
            if request.first_frame or request.last_frame:
                raise _unsupported("WAND-Vega video cannot combine first/last_frame with reference_images.")
            if len(request.reference_images) > 9:
                raise _unsupported("WAND-Vega video accepts at most 9 reference images.")
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": _normalize_data_url_mime(img)},
                    "role": "reference_image",
                }
                for img in request.reference_images
            )

        body: dict[str, Any] = {
            "model": self.video_model,
            "content": content,
            "resolution": _normalize_wand_video_resolution(request.resolution),
            "duration": _clamp_int(request.duration, 4, 15),
        }
        if request.aspect_ratio in _WAND_VIDEO_RATIOS:
            body["ratio"] = request.aspect_ratio

        payload = await self._request_json(
            client,
            "POST",
            self._v1_path("/wand/vega-videos/tasks"),
            json=body,
        )
        _raise_tokenhub_payload_error(payload, fallback="TokenHub WAND video generation failed")
        task_id = _require_str(payload.get("id") or payload.get("task_id"), "WAND video did not return a task id")
        return MediaSubmission(
            task=ProviderTask(task_id=task_id, kind="video", model=self.video_model, metadata={"family": "wand"})
        )

    async def _submit_video_kling(
        self,
        client: httpx.AsyncClient,
        request: VideoGenerationRequest,
    ) -> MediaSubmission:
        path, body = _build_kling_video_request(self.video_model, request)
        payload = await self._request_json(client, "POST", self._v1_path(path), json=body)
        _raise_tokenhub_payload_error(payload, fallback="TokenHub Kling video generation failed")
        data = payload.get("data")
        task_id = data.get("id") or data.get("task_id") if isinstance(data, Mapping) else None
        resolved_id = _require_str(task_id, "TokenHub Kling video generation did not return a task id")
        return MediaSubmission(
            task=ProviderTask(task_id=resolved_id, kind="video", model=self.video_model, metadata={"family": "kling"})
        )

    async def _submit_video_vidu(
        self,
        client: httpx.AsyncClient,
        request: VideoGenerationRequest,
    ) -> MediaSubmission:
        path, body = _build_vidu_video_request(self.video_model, request)
        payload = await self._request_json(client, "POST", self._v1_path(path), json=body)
        _raise_tokenhub_payload_error(payload, fallback="TokenHub Vidu video generation failed")
        task_id = _require_str(payload.get("task_id"), "TokenHub Vidu video generation did not return a task_id")
        return MediaSubmission(
            task=ProviderTask(task_id=task_id, kind="video", model=self.video_model, metadata={"family": "vidu"})
        )

    async def _submit_video_pixverse(
        self,
        client: httpx.AsyncClient,
        request: VideoGenerationRequest,
    ) -> MediaSubmission:
        path, body = _build_pixverse_video_request(self.video_model, request)
        payload = await self._request_json(client, "POST", self._v1_path(path), json=body)
        _raise_tokenhub_payload_error(payload, fallback="TokenHub PixVerse video generation failed")
        resp = payload.get("Resp")
        video_id = resp.get("video_id") if isinstance(resp, Mapping) else None
        task_id = _require_str(
            str(video_id) if video_id not in (None, "") else None,
            "TokenHub PixVerse video generation did not return a video_id",
        )
        return MediaSubmission(
            task=ProviderTask(task_id=task_id, kind="video", model=self.video_model, metadata={"family": "pixverse"})
        )

    async def _submit_video_minimax(
        self,
        client: httpx.AsyncClient,
        request: VideoGenerationRequest,
        *,
        family: str,
    ) -> MediaSubmission:
        if family == "minimax-v2":
            path = "/wand/minimax-video-v2/generation"
            body = _build_minimax_v2_video_body(self.video_model, request)
        else:
            path = "/wand/minimax-video/generation"
            body = _build_minimax_v1_video_body(self.video_model, request)

        payload = await self._request_json(client, "POST", self._v1_path(path), json=body)
        _raise_tokenhub_payload_error(payload, fallback="TokenHub MiniMax video generation failed")
        task_id = _require_str(payload.get("task_id"), "TokenHub MiniMax video generation did not return a task_id")
        return MediaSubmission(
            task=ProviderTask(task_id=task_id, kind="video", model=self.video_model, metadata={"family": family})
        )

    async def _submit_video_yt(
        self,
        client: httpx.AsyncClient,
        request: VideoGenerationRequest,
    ) -> MediaSubmission:
        images = ([request.first_frame] if request.first_frame else []) + list(request.reference_images)
        if not images:
            raise _image_first_required_error(
                self.video_model,
                kind="video",
                alternative="hy-video-v1.5 or kling-video-v4",
            )
        body: dict[str, Any] = {
            "model": self.video_model,
            "image": {"url": _normalize_data_url_mime(images[0])},
        }
        if request.prompt:
            body["prompt"] = request.prompt
        payload = await self._request_json(
            client,
            "POST",
            self._v1_path("/api/video/submit"),
            json=body,
        )
        _raise_tokenhub_payload_error(payload, fallback="TokenHub YT video generation failed")
        task_id = _require_str(payload.get("id") or payload.get("task_id"), "TokenHub YT video did not return an id")
        return MediaSubmission(
            task=ProviderTask(task_id=task_id, kind="video", model=self.video_model, metadata={"family": "yt"})
        )

    # ------------------------------------------------------------------
    # 3D submission
    # ------------------------------------------------------------------

    async def _submit_3d(
        self,
        client: httpx.AsyncClient,
        request: ThreeDGenerationRequest,
    ) -> MediaSubmission:
        body = _build_3d_submit_body(self.three_d_model, request)
        payload = await self._request_json(
            client,
            "POST",
            self._v1_path("/api/3d/submit"),
            json=body,
        )
        _raise_tokenhub_payload_error(payload, fallback="TokenHub 3D generation failed")
        task_id = _require_str(
            payload.get("id") or payload.get("task_id"),
            "TokenHub 3D generation did not return a task id",
        )
        return MediaSubmission(
            task=ProviderTask(
                task_id=task_id,
                kind="3d",
                model=self.three_d_model,
                metadata={"family": _three_d_family(self.three_d_model)},
            )
        )

    # ------------------------------------------------------------------
    # Unified async polling
    # ------------------------------------------------------------------

    async def _poll_task(
        self,
        client: httpx.AsyncClient,
        task: ProviderTask,
    ) -> MediaJobUpdate:
        if task.kind == "image":
            return await self._poll_image_task(client, task)
        if task.kind == "video":
            return await self._poll_video_task(client, task)
        if task.kind == "3d":
            return await self._poll_3d_task(client, task)
        raise MediaGenerationError(f"Unsupported TokenHub task kind: {task.kind}")

    async def _poll_image_task(
        self,
        client: httpx.AsyncClient,
        task: ProviderTask,
    ) -> MediaJobUpdate:
        family = task.metadata.get("family") or _image_family(task.model)
        if family == "wand":
            return await self._poll_wand_image_task(client, task)
        if family == "vidu":
            return await self._poll_vidu_task(
                client,
                task,
                subpath=f"/wand/vidu-image/tasks/{task.task_id}",
                kind="image",
            )
        if family == "kling":
            return await self._poll_kling_image_task(client, task)
        raise MediaGenerationError(f"Unsupported TokenHub async image family: {family}")

    async def _poll_wand_image_task(
        self,
        client: httpx.AsyncClient,
        task: ProviderTask,
    ) -> MediaJobUpdate:
        payload = await self._request_json(
            client,
            "GET",
            self._v1_path(f"/wand/vega-images/tasks/{task.task_id}"),
            provider_task_id=task.task_id,
        )
        _raise_tokenhub_payload_error(
            payload,
            fallback=f"TokenHub WAND image task {task.task_id} failed",
            provider_task_id=task.task_id,
        )
        status = str(payload.get("status", "")).lower()
        if status == "completed":
            artifacts = _extract_data_image_artifacts(payload)
            if not artifacts:
                raise MediaGenerationError("TokenHub WAND image task completed without a download URL")
            return MediaJobUpdate(status="succeeded", artifacts=artifacts)
        if status in {"failed", "cancelled", "incomplete"}:
            raise provider_payload_error(
                payload.get("error") or payload,
                fallback=f"TokenHub WAND image task {task.task_id} {status}",
                provider_task_id=task.task_id,
            )
        return MediaJobUpdate(status="running" if status == "in_progress" else "pending")

    async def _poll_kling_image_task(
        self,
        client: httpx.AsyncClient,
        task: ProviderTask,
    ) -> MediaJobUpdate:
        payload = await self._request_json(
            client,
            "GET",
            self._v1_path(f"/wand/kling-image/tasks/{task.task_id}"),
            provider_task_id=task.task_id,
        )
        _raise_tokenhub_payload_error(
            payload,
            fallback=f"TokenHub Kling image task {task.task_id} failed",
            provider_task_id=task.task_id,
        )
        data = payload.get("data")
        if isinstance(data, list) and data and isinstance(data[0], Mapping):
            return _parse_kling_v4_task_entry(data[0], task_id=task.task_id, media_kind="image")
        if isinstance(data, Mapping):
            status = str(data.get("task_status", "")).lower()
            if status in {"succeed", "succeeded"}:
                result = data.get("task_result")
                rows = (result.get("images") or result.get("series_images")) if isinstance(result, Mapping) else None
                artifacts = _extract_url_artifacts(rows, media_type="image/png", extension="png")
                if not artifacts:
                    raise MediaGenerationError("TokenHub Kling image task completed without a download URL")
                return MediaJobUpdate(status="succeeded", artifacts=artifacts)
            if status == "failed":
                msg = str(data.get("task_status_msg") or f"TokenHub Kling image task {task.task_id} failed")
                raise classified_provider_error(msg, provider_task_id=task.task_id)
            return MediaJobUpdate(status="running" if status == "processing" else "pending")
        return MediaJobUpdate(status="pending")

    async def _poll_video_task(
        self,
        client: httpx.AsyncClient,
        task: ProviderTask,
    ) -> MediaJobUpdate:
        family = task.metadata.get("family") or _video_family(task.model)
        if family == "vidu":
            return await self._poll_vidu_task(
                client,
                task,
                subpath=f"/wand/vidu/tasks/{task.task_id}",
                kind="video",
            )
        handlers = {
            "hy": self._poll_hy_video_task,
            "wand": self._poll_wand_video_task,
            "kling": self._poll_kling_video_task,
            "pixverse": self._poll_pixverse_video_task,
            "minimax-v2": self._poll_minimax_v2_video_task,
            "minimax-v1": self._poll_minimax_v1_video_task,
            "yt": self._poll_yt_video_task,
        }
        handler = handlers.get(family)
        if handler is None:
            raise MediaGenerationError(f"Unsupported TokenHub video family: {family}")
        return await handler(client, task)

    async def _poll_hy_video_task(
        self,
        client: httpx.AsyncClient,
        task: ProviderTask,
    ) -> MediaJobUpdate:
        payload = await self._request_json(
            client,
            "GET",
            self._v1_path(f"/wand/hunyuan-video/tasks/{task.task_id}"),
            provider_task_id=task.task_id,
        )
        _raise_tokenhub_payload_error(
            payload,
            fallback=f"TokenHub video task {task.task_id} failed",
            provider_task_id=task.task_id,
        )
        status = str(payload.get("status", "")).lower()
        if status == "succeeded":
            artifacts = _extract_url_artifacts(payload.get("videos"), media_type="video/mp4", extension="mp4")
            if not artifacts:
                raise MediaGenerationError("TokenHub video task completed without a download URL")
            return MediaJobUpdate(status="succeeded", artifacts=artifacts)
        if status == "failed":
            raise provider_payload_error(
                payload.get("error") or payload,
                fallback=f"TokenHub video task {task.task_id} failed",
                provider_task_id=task.task_id,
            )
        return MediaJobUpdate(status="running" if status == "running" else "pending")

    async def _poll_wand_video_task(
        self,
        client: httpx.AsyncClient,
        task: ProviderTask,
    ) -> MediaJobUpdate:
        payload = await self._request_json(
            client,
            "GET",
            self._v1_path(f"/wand/vega-videos/tasks/{task.task_id}"),
            provider_task_id=task.task_id,
        )
        _raise_tokenhub_payload_error(
            payload,
            fallback=f"TokenHub WAND video task {task.task_id} failed",
            provider_task_id=task.task_id,
        )
        status = str(payload.get("status", "")).lower()
        if status == "succeeded":
            content = payload.get("content")
            url = content.get("video_url") if isinstance(content, Mapping) else None
            if not isinstance(url, str) or not url:
                raise MediaGenerationError("TokenHub WAND video task completed without a video_url")
            return MediaJobUpdate(
                status="succeeded",
                artifacts=(RemoteArtifact(url=url, media_type="video/mp4", extension="mp4"),),
            )
        if status == "failed":
            raise provider_payload_error(
                payload.get("error") or payload,
                fallback=f"TokenHub WAND video task {task.task_id} failed",
                provider_task_id=task.task_id,
            )
        return MediaJobUpdate(status="running" if status == "running" else "pending")

    async def _poll_kling_video_task(
        self,
        client: httpx.AsyncClient,
        task: ProviderTask,
    ) -> MediaJobUpdate:
        payload = await self._request_json(
            client,
            "GET",
            self._v1_path(f"/wand/kling/tasks/{task.task_id}"),
            provider_task_id=task.task_id,
        )
        _raise_tokenhub_payload_error(
            payload,
            fallback=f"TokenHub Kling video task {task.task_id} failed",
            provider_task_id=task.task_id,
        )
        data = payload.get("data")
        entry = data[0] if isinstance(data, list) and data and isinstance(data[0], Mapping) else data
        if isinstance(entry, Mapping):
            return _parse_kling_v4_task_entry(entry, task_id=task.task_id, media_kind="video")
        return MediaJobUpdate(status="pending")

    async def _poll_vidu_task(
        self,
        client: httpx.AsyncClient,
        task: ProviderTask,
        *,
        subpath: str,
        kind: str,
    ) -> MediaJobUpdate:
        payload = await self._request_json(
            client,
            "GET",
            self._v1_path(subpath),
            provider_task_id=task.task_id,
        )
        _raise_tokenhub_payload_error(
            payload,
            fallback=f"TokenHub Vidu {kind} task {task.task_id} failed",
            provider_task_id=task.task_id,
        )
        state = str(payload.get("state") or payload.get("status") or "").lower()
        if state in {"success", "succeeded"}:
            media_type, ext = ("image/png", "png") if kind == "image" else ("video/mp4", "mp4")
            artifacts = _extract_url_artifacts(payload.get("creations"), media_type=media_type, extension=ext)
            if not artifacts:
                raise MediaGenerationError(f"TokenHub Vidu {kind} task completed without a download URL")
            return MediaJobUpdate(status="succeeded", artifacts=artifacts)
        if state == "failed":
            raise provider_payload_error(
                payload.get("error") or payload.get("err_code") or payload,
                fallback=f"TokenHub Vidu {kind} task {task.task_id} failed",
                provider_task_id=task.task_id,
            )
        return MediaJobUpdate(status="running" if state == "processing" else "pending")

    async def _poll_pixverse_video_task(
        self,
        client: httpx.AsyncClient,
        task: ProviderTask,
    ) -> MediaJobUpdate:
        payload = await self._request_json(
            client,
            "GET",
            self._v1_path(f"/wand/pixverse/tasks/{task.task_id}"),
            provider_task_id=task.task_id,
        )
        _raise_tokenhub_payload_error(
            payload,
            fallback=f"TokenHub PixVerse video task {task.task_id} failed",
            provider_task_id=task.task_id,
        )
        resp = payload.get("Resp")
        if not isinstance(resp, Mapping):
            return MediaJobUpdate(status="pending")
        status = resp.get("status")
        if status == 1:
            url = resp.get("url")
            if not isinstance(url, str) or not url:
                raise MediaGenerationError("TokenHub PixVerse video task completed without a URL")
            return MediaJobUpdate(
                status="succeeded",
                artifacts=(RemoteArtifact(url=url, media_type="video/mp4", extension="mp4"),),
            )
        if status == 7:
            raise classified_provider_error(
                "PixVerse content moderation rejected the video request.",
                provider_code="content_policy",
                provider_task_id=task.task_id,
            )
        if status in {6, 8}:
            raise MediaGenerationError(
                f"TokenHub PixVerse video task {task.task_id} failed (status={status}).",
                provider_task_id=task.task_id,
            )
        return MediaJobUpdate(status="running" if status == 5 else "pending")

    async def _poll_minimax_v2_video_task(
        self,
        client: httpx.AsyncClient,
        task: ProviderTask,
    ) -> MediaJobUpdate:
        payload = await self._request_json(
            client,
            "GET",
            self._v1_path(f"/wand/minimax-video-v2/tasks/{task.task_id}"),
            provider_task_id=task.task_id,
        )
        _raise_tokenhub_payload_error(
            payload,
            fallback=f"TokenHub MiniMax video task {task.task_id} failed",
            provider_task_id=task.task_id,
        )
        task_obj = payload.get("task") if isinstance(payload.get("task"), Mapping) else payload
        assert isinstance(task_obj, Mapping)
        status = str(task_obj.get("status", "")).lower()
        if status == "succeeded":
            content = task_obj.get("content")
            url = content.get("url") if isinstance(content, Mapping) else None
            if not isinstance(url, str) or not url:
                raise MediaGenerationError("TokenHub MiniMax video task completed without a URL")
            return MediaJobUpdate(
                status="succeeded",
                artifacts=(RemoteArtifact(url=url, media_type="video/mp4", extension="mp4"),),
            )
        if status in {"failed", "cancelled"}:
            raise provider_payload_error(
                task_obj.get("error") or task_obj,
                fallback=f"TokenHub MiniMax video task {task.task_id} {status}",
                provider_task_id=task.task_id,
            )
        return MediaJobUpdate(status="running" if status == "running" else "pending")

    async def _poll_minimax_v1_video_task(
        self,
        client: httpx.AsyncClient,
        task: ProviderTask,
    ) -> MediaJobUpdate:
        payload = await self._request_json(
            client,
            "GET",
            self._v1_path(f"/wand/minimax-video/tasks/{task.task_id}"),
            provider_task_id=task.task_id,
        )
        _raise_tokenhub_payload_error(
            payload,
            fallback=f"TokenHub MiniMax video task {task.task_id} failed",
            provider_task_id=task.task_id,
        )
        status = str(payload.get("status", "")).lower()
        if status == "success":
            content = payload.get("content")
            url = content.get("url") if isinstance(content, Mapping) else None
            if not isinstance(url, str) or not url:
                raise MediaGenerationError("TokenHub MiniMax video task completed without a URL")
            return MediaJobUpdate(
                status="succeeded",
                artifacts=(RemoteArtifact(url=url, media_type="video/mp4", extension="mp4"),),
            )
        if status == "fail":
            base_resp = payload.get("base_resp")
            detail = base_resp.get("status_msg") if isinstance(base_resp, Mapping) else None
            raise classified_provider_error(
                str(detail or f"TokenHub MiniMax video task {task.task_id} failed"),
                provider_task_id=task.task_id,
            )
        return MediaJobUpdate(status="running" if status == "processing" else "pending")

    async def _poll_yt_video_task(
        self,
        client: httpx.AsyncClient,
        task: ProviderTask,
    ) -> MediaJobUpdate:
        payload = await self._request_json(
            client,
            "POST",
            self._v1_path("/api/video/query"),
            json={"model": task.model, "id": task.task_id},
            provider_task_id=task.task_id,
        )
        _raise_tokenhub_payload_error(
            payload,
            fallback=f"TokenHub YT video task {task.task_id} failed",
            provider_task_id=task.task_id,
        )
        status = str(payload.get("status", "")).lower()
        if status in {"completed", "succeeded"}:
            data = payload.get("data")
            url = data.get("url") if isinstance(data, Mapping) else None
            if not isinstance(url, str) or not url:
                raise MediaGenerationError("TokenHub YT video task completed without a URL")
            return MediaJobUpdate(
                status="succeeded",
                artifacts=(RemoteArtifact(url=url, media_type="video/mp4", extension="mp4"),),
            )
        if status == "failed":
            raise provider_payload_error(
                payload.get("error") or payload,
                fallback=f"TokenHub YT video task {task.task_id} failed",
                provider_task_id=task.task_id,
            )
        return MediaJobUpdate(status="running" if status in {"in_progress", "processing", "running"} else "pending")

    async def _poll_3d_task(
        self,
        client: httpx.AsyncClient,
        task: ProviderTask,
    ) -> MediaJobUpdate:
        payload = await self._request_json(
            client,
            "POST",
            self._v1_path("/api/3d/query"),
            json={"model": task.model, "id": task.task_id},
            provider_task_id=task.task_id,
        )
        _raise_tokenhub_payload_error(
            payload,
            fallback=f"TokenHub 3D task {task.task_id} failed",
            provider_task_id=task.task_id,
        )
        status = str(payload.get("status", "")).lower()
        if status == "completed":
            artifacts = _extract_3d_artifacts(payload)
            if not artifacts:
                raise MediaGenerationError("TokenHub 3D task completed without downloadable artifacts")
            return MediaJobUpdate(status="succeeded", artifacts=artifacts)
        if status == "failed":
            raise provider_payload_error(
                payload.get("error") or payload,
                fallback=f"TokenHub 3D task {task.task_id} failed",
                provider_task_id=task.task_id,
            )
        return MediaJobUpdate(status="running" if status in {"in_progress", "processing", "running"} else "pending")


# ----------------------------------------------------------------------
# Family classification
# ----------------------------------------------------------------------


_IMAGE_FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("hy-image-v3.5", "hy-v35"),
    ("hy-image", "hy-v3"),
    ("wand-vega-image", "wand"),
    ("vidu-image", "vidu"),
    ("seedream-image", "seedream"),
    ("kling-image", "kling"),
    ("hidream-image", "hidream"),
    ("minimax-image", "minimax"),
)

_VIDEO_FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("hy-video", "hy"),
    ("wand-vega-video", "wand"),
    ("kling-video", "kling"),
    ("vidu-video", "vidu"),
    ("pixverse-video", "pixverse"),
    ("minimax-video-h3", "minimax-v2"),
    ("minimax-video", "minimax-v1"),
    ("yt-video", "yt"),
)


def _match_family(model: str, prefixes: tuple[tuple[str, str], ...]) -> str:
    lower = model.strip().lower()
    for prefix, family in prefixes:
        if lower.startswith(prefix):
            return family
    return "unknown"


def _image_family(model: str) -> str:
    return _match_family(model, _IMAGE_FAMILY_PREFIXES)


def _video_family(model: str) -> str:
    return _match_family(model, _VIDEO_FAMILY_PREFIXES)


def _three_d_family(model: str) -> str:
    lower = model.strip().lower()
    if lower.startswith("hy-3d"):
        return "hy"
    if lower.startswith("tripo-3d"):
        return "tripo"
    if lower.startswith("hi3d"):
        return "hi3d"
    if lower.startswith("rodin-gen-3d"):
        return "rodin"
    return "unknown"


# ----------------------------------------------------------------------
# Request builders (Image / Video / 3D)
# ----------------------------------------------------------------------


def _build_kling_v4_image_request(model: str, request: ImageGenerationRequest) -> tuple[str, dict[str, Any]]:
    is_long = model.lower().endswith("-long")
    contents: list[dict[str, Any]] = [{"type": "prompt", "text": request.prompt}]
    if is_long:
        if len(request.reference_images) > 3:
            raise _unsupported("kling-image-v4-long accepts at most 3 reference images.")
        for idx, img in enumerate(request.reference_images, start=1):
            contents.append({"type": "refer_image", "url": _normalize_data_url_mime(img), "id": f"ref_{idx}"})
        settings: dict[str, Any] = {
            "skill": "ecommerce" if request.reference_images else "webpage",
            "resolution": "2k" if _infer_tier(request.size) in {"2k", "4k", "8k"} else "1k",
        }
        return (
            "/wand/kling/long-image-generation",
            {
                "model": model,
                "contents": contents,
                "settings": settings,
                "options": {"watermark_info": {"enabled": request.watermark}},
            },
        )

    if len(request.reference_images) > 10:
        raise _unsupported("kling-image-v4 accepts at most 10 reference images.")
    if request.count > 9:
        raise _unsupported("kling-image-v4 accepts count between 1 and 9.")
    for idx, img in enumerate(request.reference_images, start=1):
        contents.append({"type": "refer_image", "url": _normalize_data_url_mime(img), "id": f"img_{idx}"})
    settings = {
        "result_type": "single",
        "n": request.count,
        "resolution": "4k" if _infer_tier(request.size) == "4k" else "2k",
    }
    if request.size:
        ratio, _ = _infer_ratio_and_tier(request.size)
        if ratio in _KLING_RATIOS:
            settings["aspect_ratio"] = ratio
    return (
        "/wand/kling/image-generation",
        {
            "model": model,
            "contents": contents,
            "settings": settings,
            "options": {"watermark_info": {"enabled": request.watermark}},
        },
    )


def _build_kling_v3_image_request(model: str, request: ImageGenerationRequest) -> tuple[str, dict[str, Any]]:
    lower = model.lower()
    if request.count > 9:
        raise _unsupported(f"{model} accepts count between 1 and 9.")
    body: dict[str, Any] = {
        "model": model,
        "prompt": request.prompt,
        "n": request.count,
        "watermark_info": {"enabled": request.watermark},
    }
    if lower == "kling-image-v3":
        if len(request.reference_images) > 1:
            raise _unsupported(
                "kling-image-v3 accepts at most 1 reference image; use kling-image-v3-omni for multi-image."
            )
        if request.reference_images:
            body["image"] = _strip_data_url_base64(request.reference_images[0])
    else:
        if len(request.reference_images) > 10:
            raise _unsupported(f"{model} accepts at most 10 reference images.")
        if request.reference_images:
            body["image_list"] = [_strip_data_url_base64(img) for img in request.reference_images]

    if request.size:
        ratio, tier = _infer_ratio_and_tier(request.size)
        if ratio in _KLING_RATIOS:
            body["aspect_ratio"] = ratio
        if tier == "4k" and lower == "kling-image-v3-omni":
            body["resolution"] = "4k"
        elif tier in {"2k", "4k"}:
            body["resolution"] = "2k"
        else:
            body["resolution"] = "1k"
    return "/wand/kling-image/generation", body


def _build_kling_video_request(model: str, request: VideoGenerationRequest) -> tuple[str, dict[str, Any]]:
    lower = model.lower()
    if "motion-control" in lower:
        raise _unsupported(f"{model} requires a motion reference video, which is not supported by generate_video.")

    is_v4 = lower == "kling-video-v4"
    is_v4_flash = lower == "kling-video-v4-flash"
    use_omni = lower in _KLING_OMNI_ONLY or (bool(request.reference_images) and (is_v4 or is_v4_flash))
    options = {"watermark_info": {"enabled": request.watermark}}

    if use_omni:
        return "/wand/kling/omni-video", _build_kling_omni_video_body(model, request, options=options)

    if request.first_frame or request.reference_images:
        if request.reference_images and not request.first_frame:
            first_frame = request.reference_images[0]
        else:
            first_frame = request.first_frame or ""
        contents: list[dict[str, Any]] = [
            {"type": "prompt", "text": request.prompt},
            {"type": "first_frame", "url": _normalize_data_url_mime(first_frame)},
        ]
        if request.last_frame:
            if lower in {"kling-video-v3-turbo", "kling-video-v4-flash"}:
                raise _unsupported(f"{model} does not support last_frame.")
            contents.append({"type": "last_frame", "url": _normalize_data_url_mime(request.last_frame)})
        settings = _kling_video_settings(lower, request, include_aspect_ratio=False)
        return (
            "/wand/kling/image-to-video",
            {"model": model, "contents": contents, "settings": settings, "options": options},
        )

    settings = _kling_video_settings(lower, request, include_aspect_ratio=True)
    return (
        "/wand/kling/text-to-video",
        {"model": model, "prompt": request.prompt, "settings": settings, "options": options},
    )


def _build_kling_omni_video_body(
    model: str,
    request: VideoGenerationRequest,
    *,
    options: dict[str, Any],
) -> dict[str, Any]:
    lower = model.lower()
    contents: list[dict[str, Any]] = [{"type": "prompt", "text": request.prompt}]
    if request.first_frame:
        contents.append({"type": "first_frame", "url": _normalize_data_url_mime(request.first_frame)})
    if request.last_frame:
        if lower == "kling-video-v4-flash":
            raise _unsupported("kling-video-v4-flash does not support last_frame.")
        contents.append({"type": "last_frame", "url": _normalize_data_url_mime(request.last_frame)})
    for img in request.reference_images:
        contents.append({"type": "refer_image", "url": _normalize_data_url_mime(img)})
    settings = _kling_video_settings(lower, request, include_aspect_ratio=not bool(request.first_frame))
    return {"model": model, "contents": contents, "settings": settings, "options": options}


def _kling_video_settings(
    lower: str,
    request: VideoGenerationRequest,
    *,
    include_aspect_ratio: bool,
) -> dict[str, Any]:
    res = request.resolution.strip().lower()
    if lower == "kling-video-v4-flash":
        resolution = "720p"
    elif lower in {"kling-video-v4", "kling-video-v3", "kling-video-v3-omni"}:
        resolution = res if res in {"720p", "1080p", "4k"} else "720p"
    else:
        resolution = "1080p" if res in {"1080p", "4k"} else "720p"

    if lower in {"kling-video-v2.6", "kling-video-v2.5-turbo"}:
        duration = 10 if request.duration >= 8 else 5
    elif lower == "kling-video-v4":
        duration = _clamp_int(request.duration, 3, 30)
    elif lower == "kling-video-v4-flash":
        duration = _clamp_int(request.duration, 3, 20)
    elif lower == "kling-video-o1":
        duration = _clamp_int(request.duration, 3, 10)
    else:
        duration = _clamp_int(request.duration, 3, 15)

    settings: dict[str, Any] = {"resolution": resolution, "duration": duration}
    if include_aspect_ratio:
        allowed = {"16:9", "9:16", "1:1", "21:9"} if lower.startswith("kling-video-v4") else {"16:9", "9:16", "1:1"}
        settings["aspect_ratio"] = request.aspect_ratio if request.aspect_ratio in allowed else "16:9"
    if request.generate_audio and lower in {"kling-video-v3", "kling-video-v3-omni", "kling-video-v2.6"}:
        settings["audio"] = "native"
        if lower == "kling-video-v2.6":
            settings["resolution"] = "1080p"
    return settings


def _vidu_reference_video_request(
    model: str,
    lower: str,
    request: VideoGenerationRequest,
    common: dict[str, Any],
    *,
    is_q3: bool,
    is_q4: bool,
) -> tuple[str, dict[str, Any]]:
    if lower in {"vidu-video-q3-pro", "vidu-video-q2-turbo"}:
        raise _unsupported(f"{model} does not support reference-to-video; use first_frame or vidu-video-q3-turbo.")
    imgs = [_normalize_data_url_mime(img) for img in request.reference_images]
    if is_q4:
        common["subjects"] = [{"name": f"ref{i}", "images": [img]} for i, img in enumerate(imgs, start=1)]
    else:
        common["images"] = imgs
    common["duration"] = _clamp_int(request.duration, 3 if (is_q3 or is_q4) else 1, 16 if (is_q3 or is_q4) else 10)
    if request.aspect_ratio in _VIDU_VIDEO_RATIOS:
        common["aspect_ratio"] = request.aspect_ratio
    return "/wand/vidu/reference-to-video", common


def _build_vidu_video_request(model: str, request: VideoGenerationRequest) -> tuple[str, dict[str, Any]]:
    lower = model.lower()
    is_q4 = lower == "vidu-video-q4-preview"
    is_q3 = lower.startswith("vidu-video-q3")
    common: dict[str, Any] = {
        "model": model,
        "prompt": request.prompt,
        "resolution": _normalize_vidu_video_resolution(request.resolution, allow_uhd=is_q4),
        "watermark": request.watermark,
    }
    if request.seed is not None:
        common["seed"] = _validate_non_negative_seed(request.seed)
    if is_q3 or is_q4:
        common["audio"] = request.generate_audio

    if request.first_frame and request.last_frame:
        if lower in {"vidu-video-q4-preview", "vidu-video-q3", "vidu-video-q2"}:
            raise _unsupported(f"{model} does not support start-end-to-video (first_frame + last_frame).")
        common["images"] = [_normalize_data_url_mime(request.first_frame), _normalize_data_url_mime(request.last_frame)]
        common["duration"] = _clamp_int(request.duration, 1, 16 if is_q3 else 8)
        return "/wand/vidu/start-end-to-video", common

    if request.reference_images:
        return _vidu_reference_video_request(model, lower, request, common, is_q3=is_q3, is_q4=is_q4)

    if request.first_frame:
        common["images"] = [_normalize_data_url_mime(request.first_frame)]
        use_ref = lower in {"vidu-video-q3", "vidu-video-q2"}
        min_dur = 3 if (is_q3 if use_ref else is_q4) else 1
        max_dur = 16 if (is_q3 or (is_q4 and not use_ref)) else 10
        common["duration"] = _clamp_int(request.duration, min_dur, max_dur)
        return ("/wand/vidu/reference-to-video" if use_ref else "/wand/vidu/image-to-video"), common

    if lower in _VIDU_I2V_ONLY:
        raise _image_first_required_error(model, kind="video", alternative="vidu-video-q3-pro or vidu-video-q3-turbo")
    common["duration"] = _clamp_int(request.duration, 1, 16 if is_q3 else 10)
    if request.aspect_ratio in _VIDU_VIDEO_RATIOS:
        common["aspect_ratio"] = request.aspect_ratio
    return "/wand/vidu/text-to-video", common


def _build_pixverse_video_request(model: str, request: VideoGenerationRequest) -> tuple[str, dict[str, Any]]:
    lower = model.lower()
    is_v56 = lower == "pixverse-video-v5.6"
    is_v6_max = lower == "pixverse-video-v6.0-max"
    quality = _normalize_pixverse_quality(request.resolution)
    if is_v56:
        if request.duration >= 10 and quality != "1080p":
            duration = 10
        elif request.duration >= 7:
            duration = 8
        else:
            duration = 5
    else:
        duration = _clamp_int(request.duration, 1, 15)

    body: dict[str, Any] = {
        "model": model,
        "prompt": request.prompt,
        "duration": duration,
        "quality": quality,
        "generate_audio_switch": request.generate_audio,
    }
    if request.seed is not None:
        body["seed"] = _validate_non_negative_seed(request.seed)

    if request.first_frame and request.last_frame:
        if is_v6_max:
            raise _unsupported("pixverse-video-v6.0-max only supports text-to-video and image-to-video.")
        body["first_frame_img"] = _normalize_data_url_mime(request.first_frame)
        body["last_frame_img"] = _normalize_data_url_mime(request.last_frame)
        return "/wand/pixverse/start-end-to-video", body

    if request.reference_images and not request.first_frame:
        if is_v6_max:
            raise _unsupported("pixverse-video-v6.0-max does not support reference-to-video.")
        if len(request.reference_images) > 7:
            raise _unsupported("PixVerse reference-to-video accepts at most 7 reference images.")
        body["image_references"] = [
            {"img_id": _normalize_data_url_mime(img), "type": "subject"} for img in request.reference_images
        ]
        allowed_ratios = _PIXVERSE_V56_RATIOS if is_v56 else _PIXVERSE_V6_RATIOS
        body["aspect_ratio"] = request.aspect_ratio if request.aspect_ratio in allowed_ratios else "16:9"
        return "/wand/pixverse/reference-to-video", body

    if request.first_frame:
        body["img_id"] = _normalize_data_url_mime(request.first_frame)
        return "/wand/pixverse/image-to-video", body

    allowed_ratios = _PIXVERSE_V56_RATIOS if is_v56 else _PIXVERSE_V6_RATIOS
    body["aspect_ratio"] = request.aspect_ratio if request.aspect_ratio in allowed_ratios else "16:9"
    return "/wand/pixverse/text-to-video", body


def _build_minimax_v2_video_body(model: str, request: VideoGenerationRequest) -> dict[str, Any]:
    lower = model.lower()
    content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
    has_frames = bool(request.first_frame or request.last_frame)
    if request.first_frame:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _normalize_data_url_mime(request.first_frame)},
                "role": "first_frame",
            }
        )
    if request.last_frame:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _normalize_data_url_mime(request.last_frame)},
                "role": "last_frame",
            }
        )
    if request.reference_images:
        if has_frames:
            raise _unsupported("MiniMax H3 video cannot combine first/last_frame with reference_images.")
        if len(request.reference_images) > 9:
            raise _unsupported("MiniMax H3 video accepts at most 9 reference images.")
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": _normalize_data_url_mime(img)},
                "role": "reference_image",
            }
            for img in request.reference_images
        )

    res_lower = request.resolution.strip().lower()
    if lower == "minimax-video-h3":
        resolution = "2K" if res_lower in {"2k", "4k"} else "768P"
    else:
        resolution = "480P" if res_lower == "480p" else "768P"

    min_dur = 5 if lower == "minimax-video-h3-max" else 4
    body: dict[str, Any] = {
        "model": model,
        "content": content,
        "resolution": resolution,
        "duration": _clamp_int(request.duration, min_dur, 15),
        "aigc_watermark": request.watermark,
    }
    if not has_frames:
        allowed_ratios = {"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}
        body["ratio"] = request.aspect_ratio if request.aspect_ratio in allowed_ratios else "16:9"
    return body


def _build_minimax_v1_video_body(model: str, request: VideoGenerationRequest) -> dict[str, Any]:
    lower = model.lower()
    is_fast = lower == "minimax-video-v2.3-fast"
    first_frame = request.first_frame or (request.reference_images[0] if request.reference_images else None)
    if is_fast and not first_frame:
        raise _image_first_required_error(model, kind="video", alternative="minimax-video-v2.3 or minimax-video-h3")
    if is_fast and request.last_frame:
        raise _unsupported("minimax-video-v2.3-fast does not support last_frame.")

    resolution = "1080P" if request.resolution.strip().lower() in {"1080p", "2k", "4k"} else "768P"
    duration = 10 if (request.duration >= 8 and resolution == "768P") else 6
    body: dict[str, Any] = {
        "model": model,
        "prompt": request.prompt,
        "duration": duration,
        "resolution": resolution,
        "aigc_watermark": request.watermark,
    }
    if first_frame:
        body["first_frame_image"] = _normalize_data_url_mime(first_frame)
    if request.last_frame:
        body["last_frame_image"] = _normalize_data_url_mime(request.last_frame)
    return body


def _build_3d_submit_body(model: str, request: ThreeDGenerationRequest) -> dict[str, Any]:
    family = _three_d_family(model)
    if family == "hy":
        return _build_hy_3d_submit_body(model, request)
    if family == "tripo":
        return _build_tripo_3d_submit_body(model, request)
    if family == "hi3d":
        return _build_hi3d_submit_body(model, request)
    if family == "rodin":
        return _build_rodin_3d_submit_body(model, request)
    raise _unsupported(f"Unsupported TokenHub 3D model: {model}")


def _build_hy_3d_submit_body(model: str, request: ThreeDGenerationRequest) -> dict[str, Any]:
    is_v36 = model.startswith("hy-3d-3.6")
    is_express = model.startswith("hy-3d-express")

    if is_v36 and not request.image:
        raise _image_first_required_error(
            "HY-3D-3.6",
            kind="3d",
            alternative="hy-3d-3.1, hy-3d-express, tripo-3d-3.1, or rodin-gen-3d-2.5",
        )

    body: dict[str, Any] = {"model": model}
    if request.image:
        if request.image.startswith(("http://", "https://")):
            body["image_url"] = request.image
        else:
            body["image_base64"] = _strip_data_url_base64(request.image)
    if request.multi_view_images:
        if "image_url" not in body:
            raise _unsupported("multi_view_images requires an HTTP(S) image URL as the primary front image.")
        body["multi_view_images"] = [
            {"view_type": view_type, "view_image_url": url} for view_type, url in request.multi_view_images
        ]

    if not is_v36 and request.prompt and (not request.image or (not is_express and request.generate_type == "Sketch")):
        if len(request.prompt) > 1024:
            raise _unsupported("TokenHub 3D text prompt must be at most 1024 characters.")
        body["prompt"] = request.prompt

    _apply_hy_3d_options(body, model=model, request=request)
    return body


def _apply_express_3d_options(body: dict[str, Any], request: ThreeDGenerationRequest) -> None:
    if request.generate_type == "Geometry":
        body["enable_geometry"] = True
    elif request.generate_type != "Normal":
        raise _unsupported(f"HY-3D-Express does not support generate_type={request.generate_type!r}.")
    if request.enable_pbr and request.generate_type == "Normal":
        body["enable_pbr"] = True
    if request.result_format:
        body["result_format"] = request.result_format.lower()


def _apply_hy_3d_options(body: dict[str, Any], *, model: str, request: ThreeDGenerationRequest) -> None:
    if model.startswith("hy-3d-express"):
        _apply_express_3d_options(body, request)
        return

    is_v36 = model.startswith("hy-3d-3.6")
    allowed_types = {"Normal", "Geometry"} if is_v36 else {"Normal", "Geometry", "Sketch", "LowPoly"}
    if model.startswith("hy-3d-3.1"):
        allowed_types = {"Normal", "Geometry", "Sketch"}
    if request.generate_type not in allowed_types:
        raise _unsupported(f"Model {model} does not support generate_type={request.generate_type!r}.")
    body["generate_type"] = request.generate_type

    if request.generate_type == "Normal":
        if request.enable_pbr:
            body["enable_pbr"] = True
        if is_v36:
            _apply_v36_texture_options(body, request)

    if request.face_count is not None and request.generate_type != "LowPoly":
        min_faces, max_faces = (1000, 2_000_000) if is_v36 else (3000, 1_500_000)
        if not min_faces <= request.face_count <= max_faces:
            raise _unsupported(f"Model {model} face_count must be between {min_faces} and {max_faces}.")
        body["face_count"] = request.face_count

    if request.result_format:
        fmt = request.result_format.lower()
        if fmt not in {"glb", "obj", "stl", "usdz", "fbx"}:
            raise _unsupported(f"Model {model} does not support result_format={request.result_format!r}.")
        # Hy3D 3.6/3.1/3.0 return glb/obj by default when result_format is omitted;
        # the API enum only accepts stl, usdz, or fbx when explicitly provided.
        if fmt not in {"glb", "obj"}:
            body["result_format"] = fmt


def _apply_v36_texture_options(body: dict[str, Any], request: ThreeDGenerationRequest) -> None:
    if request.tex_res is not None:
        if request.tex_res not in _TEX_RES_VALUES:
            raise _unsupported("HY-3D-3.6 tex_res must be one of 2k, 4k, or 8k.")
        body["tex_res"] = request.tex_res
    if request.seam_less is not None:
        body["seam_less"] = request.seam_less
    if request.bake_lighting is not None:
        body["bake_lighting"] = request.bake_lighting


def _build_tripo_3d_submit_body(model: str, request: ThreeDGenerationRequest) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model}
    if request.multi_view_images:
        if not request.image:
            raise _unsupported("Tripo multi-view generation requires a front image in image.")
        inputs: dict[str, str] = {"front": _normalize_data_url_mime(request.image)}
        for view_type, url in request.multi_view_images:
            if view_type in {"back", "left", "right"}:
                inputs[view_type] = _normalize_data_url_mime(url)
        body["inputs"] = inputs
    elif request.image:
        body["input"] = _normalize_data_url_mime(request.image)
    elif request.prompt:
        if len(request.prompt) > 1024:
            raise _unsupported("Tripo 3D prompt must be at most 1024 characters.")
        body["prompt"] = request.prompt
    else:
        raise _unsupported("Tripo 3D requires prompt or image.")

    if request.generate_type == "Geometry":
        body["texture"] = False
        body["pbr"] = False
    else:
        body["texture"] = True
        body["pbr"] = request.enable_pbr
        if request.generate_type == "LowPoly" and model.lower() == "tripo-3d-3.1":
            body["smart_low_poly"] = True
    if request.face_count is not None:
        body["face_limit"] = request.face_count
    return body


def _build_hi3d_submit_body(model: str, request: ThreeDGenerationRequest) -> dict[str, Any]:
    if not request.image:
        raise _image_first_required_error(
            model,
            kind="3d",
            alternative="hy-3d-3.1, tripo-3d-3.1, or rodin-gen-3d-2.5",
        )
    body: dict[str, Any] = {"model": model}
    if request.multi_view_images:
        views: dict[str, str] = {"front": request.image}
        for view_type, url in request.multi_view_images:
            if view_type in {"back", "left", "right"}:
                views[view_type] = url
        order = ("front", "back", "left", "right")
        body["multi_images"] = [views[k] for k in order if k in views]
        body["multi_images_bit"] = "".join("1" if k in views else "0" for k in order)
    elif request.image.startswith(("http://", "https://")):
        body["image_url"] = request.image
    else:
        body["image_base64"] = _strip_data_url_base64(request.image)

    body["request_type"] = 1 if request.generate_type == "Geometry" else 3
    body["pbr"] = 1 if request.enable_pbr else 0
    if request.face_count is not None:
        max_faces = 5_000_000 if model.lower().startswith("hi3d-3.0") else 2_000_000
        if not 100_000 <= request.face_count <= max_faces:
            raise _unsupported(f"{model} face_count must be between 100000 and {max_faces}.")
        body["face"] = request.face_count
    fmt_key = (request.result_format or "glb").lower()
    if fmt_key not in _HI3D_FORMATS:
        raise _unsupported(f"{model} does not support result_format={request.result_format!r}.")
    body["format"] = _HI3D_FORMATS[fmt_key]
    return body


def _build_rodin_3d_submit_body(model: str, request: ThreeDGenerationRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "tier": "Gen-2.5-Medium",
        "preview_render": True,
    }
    if request.image:
        images = [_normalize_data_url_mime(request.image)]
        images.extend(_normalize_data_url_mime(url) for _, url in request.multi_view_images)
        if len(images) > 5:
            raise _unsupported("Rodin Gen-3D 2.5 accepts at most 5 input images.")
        body["images"] = images
    elif request.prompt:
        if len(request.prompt) > 1024:
            raise _unsupported("Rodin Gen-3D 2.5 prompt must be at most 1024 characters.")
        body["prompt"] = request.prompt
    else:
        raise _unsupported("Rodin Gen-3D 2.5 requires prompt or image.")

    if request.generate_type == "Geometry":
        body["material"] = "None"
    else:
        body["material"] = "PBR" if request.enable_pbr else "Shaded"
    if request.face_count is not None:
        if not 500 <= request.face_count <= 2_000_000:
            raise _unsupported("Rodin Gen-3D 2.5 face_count must be between 500 and 2000000.")
        body["quality_override"] = request.face_count
    fmt = (request.result_format or "glb").lower()
    if fmt not in {"glb", "usdz", "fbx", "obj", "stl"}:
        raise _unsupported(f"Rodin Gen-3D 2.5 does not support result_format={request.result_format!r}.")
    body["geometry_file_format"] = fmt
    return body


# ----------------------------------------------------------------------
# Response & artifact parsing helpers
# ----------------------------------------------------------------------


def _parse_kling_v4_task_entry(
    entry: Mapping[str, Any],
    *,
    task_id: str,
    media_kind: str,
) -> MediaJobUpdate:
    status = str(entry.get("status", "")).lower()
    if status in {"succeeded", "succeed"}:
        media_type, ext = ("image/png", "png") if media_kind == "image" else ("video/mp4", "mp4")
        artifacts = _extract_url_artifacts(entry.get("outputs"), media_type=media_type, extension=ext)
        if not artifacts:
            raise MediaGenerationError(f"TokenHub Kling {media_kind} task completed without a download URL")
        return MediaJobUpdate(status="succeeded", artifacts=artifacts)
    if status == "failed":
        msg = str(entry.get("message") or f"TokenHub Kling {media_kind} task {task_id} failed")
        raise classified_provider_error(msg, provider_task_id=task_id)
    return MediaJobUpdate(status="running" if status == "processing" else "pending")


def _extract_v35_image_url(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise MediaGenerationError("TokenHub Hy-Image-3.5 response returned no choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise MediaGenerationError("TokenHub Hy-Image-3.5 response returned an unexpected choice")
    delta = first.get("delta")
    image = delta.get("image") if isinstance(delta, Mapping) else None
    url = image.get("url") if isinstance(image, Mapping) else None
    if not isinstance(url, str) or not url:
        raise MediaGenerationError("TokenHub Hy-Image-3.5 response returned no image URL")
    return url


def _extract_data_image_artifacts(payload: Mapping[str, Any]) -> tuple[RemoteArtifact, ...]:
    rows = payload.get("data")
    if not isinstance(rows, list):
        return ()
    artifacts: list[RemoteArtifact] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        url = row.get("url")
        if isinstance(url, str) and url:
            artifacts.append(RemoteArtifact(url=url, media_type="image/png", extension="png"))
            continue
        b64 = row.get("b64_json")
        if isinstance(b64, str) and b64:
            artifacts.append(
                RemoteArtifact(
                    data=base64.b64decode(_strip_data_url_base64(b64)),
                    media_type="image/png",
                    extension="png",
                )
            )
    return tuple(artifacts)


def _extract_url_artifacts(
    rows: object,
    *,
    media_type: str,
    extension: str,
) -> tuple[RemoteArtifact, ...]:
    if not isinstance(rows, list):
        return ()
    artifacts: list[RemoteArtifact] = []
    for row in rows:
        raw_url = row.get("url") if isinstance(row, Mapping) else None
        if isinstance(raw_url, str) and raw_url:
            artifacts.append(RemoteArtifact(url=raw_url, media_type=media_type, extension=extension))
    return tuple(artifacts)


def _extract_3d_artifacts(payload: Mapping[str, Any]) -> tuple[RemoteArtifact, ...]:
    # 1. Tripo shape: payload["output"] = {"model_url", "rendered_image_url", "generated_image_url"}
    output = payload.get("output")
    if isinstance(output, Mapping):
        return _extract_tripo_3d_artifacts(output)

    data = payload.get("data")
    # 2. Hi3D shape: payload["data"] = {"url", "cover_url"}
    if isinstance(data, Mapping):
        return _extract_hi3d_artifacts(data)

    # 3. Hy3D & Rodin shape: payload["data"] = [...]
    if isinstance(data, list):
        return _extract_3d_list_artifacts(data)
    return ()


def _extract_tripo_3d_artifacts(output: Mapping[str, Any]) -> tuple[RemoteArtifact, ...]:
    primaries: list[RemoteArtifact] = []
    previews: list[RemoteArtifact] = []
    model_url = output.get("model_url")
    if isinstance(model_url, str) and model_url:
        media_type, ext = _infer_3d_media_type(model_url, raw_type="")
        primaries.append(RemoteArtifact(url=model_url, media_type=media_type, extension=ext, role="primary"))
    preview_url = output.get("rendered_image_url") or output.get("generated_image_url")
    if isinstance(preview_url, str) and preview_url:
        previews.append(_preview_artifact(preview_url))
    return tuple(primaries + previews)


def _extract_hi3d_artifacts(data: Mapping[str, Any]) -> tuple[RemoteArtifact, ...]:
    primaries: list[RemoteArtifact] = []
    previews: list[RemoteArtifact] = []
    url = data.get("url")
    if isinstance(url, str) and url:
        media_type, ext = _infer_3d_media_type(url, raw_type="")
        primaries.append(RemoteArtifact(url=url, media_type=media_type, extension=ext, role="primary"))
    cover = data.get("cover_url")
    if isinstance(cover, str) and cover:
        previews.append(_preview_artifact(cover))
    return tuple(primaries + previews)


def _extract_3d_list_artifacts(rows: list[Any]) -> tuple[RemoteArtifact, ...]:
    primaries: list[RemoteArtifact] = []
    previews: list[RemoteArtifact] = []
    seen_previews: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        raw_url = row.get("url")
        if isinstance(raw_url, str) and raw_url:
            name_or_url = str(row.get("name") or raw_url)
            ext_hint = _path_extension(name_or_url)
            if ext_hint in _IMAGE_EXT_TYPES and not row.get("type"):
                if raw_url not in seen_previews:
                    seen_previews.add(raw_url)
                    previews.append(_preview_artifact(raw_url, ext_hint=ext_hint))
            else:
                raw_type = str(row.get("type") or "").strip().lower()
                media_type, extension = _infer_3d_media_type(name_or_url, raw_type=raw_type)
                primaries.append(
                    RemoteArtifact(url=raw_url, media_type=media_type, extension=extension, role="primary")
                )
        preview_url = row.get("preview_image_url")
        if isinstance(preview_url, str) and preview_url and preview_url not in seen_previews:
            seen_previews.add(preview_url)
            previews.append(_preview_artifact(preview_url))
    return tuple(primaries + previews)


def _infer_3d_media_type(url_or_name: str, *, raw_type: str) -> tuple[str, str]:
    ext = _path_extension(url_or_name)
    if ext == "zip":
        return "application/zip", "zip"
    if raw_type in _THREE_D_FORMAT_TYPES:
        return _THREE_D_FORMAT_TYPES[raw_type]
    if ext in _THREE_D_FORMAT_TYPES:
        return _THREE_D_FORMAT_TYPES[ext]
    return "application/octet-stream", raw_type or ext or "glb"


def _preview_artifact(url: str, *, ext_hint: str = "") -> RemoteArtifact:
    ext = ext_hint or _path_extension(url)
    media_type = _IMAGE_EXT_TYPES.get(ext, "image/png")
    normalized_ext = "jpg" if ext == "jpeg" else (ext if ext in _IMAGE_EXT_TYPES else "png")
    return RemoteArtifact(url=url, media_type=media_type, extension=normalized_ext, role="preview")


def _path_extension(value: str) -> str:
    path = urlsplit(value).path.lower()
    if "." not in path.rsplit("/", 1)[-1]:
        return ""
    return path.rsplit(".", 1)[-1]


# ----------------------------------------------------------------------
# Error & normalization utilities
# ----------------------------------------------------------------------


def _raise_tokenhub_payload_error(
    payload: Mapping[str, Any],
    *,
    fallback: str,
    provider_task_id: str | None = None,
) -> None:
    error = payload.get("error")
    if isinstance(error, Mapping) and error:
        raise provider_payload_error(error, fallback=fallback, provider_task_id=provider_task_id)
    err_code = payload.get("ErrCode")
    if err_code not in (None, 0, "0"):
        raise provider_payload_error(
            {"code": err_code, "message": payload.get("ErrMsg") or fallback},
            fallback=fallback,
            provider_task_id=provider_task_id,
        )
    code = payload.get("code")
    if code not in (None, 0, "0"):
        raise provider_payload_error(payload, fallback=fallback, provider_task_id=provider_task_id)
    base_resp = payload.get("base_resp")
    if isinstance(base_resp, Mapping) and base_resp.get("status_code") not in (None, 0, "0"):
        raise provider_payload_error(base_resp, fallback=fallback, provider_task_id=provider_task_id)
    if payload.get("status") == "failed" and (payload.get("code") or payload.get("message")):
        raise provider_payload_error(payload, fallback=fallback, provider_task_id=provider_task_id)


def _require_image_artifacts(artifacts: tuple[RemoteArtifact, ...]) -> MediaSubmission:
    if not artifacts:
        raise MediaGenerationError("TokenHub image generation returned no usable images")
    return MediaSubmission(artifacts=artifacts)


def _require_str(value: object, error_message: str) -> str:
    if not isinstance(value, str) or not value:
        raise MediaGenerationError(error_message)
    return value


def _normalize_data_url_mime(url: str) -> str:
    if url.lower().startswith("data:image/jpg;"):
        return "data:image/jpeg;" + url[len("data:image/jpg;") :]
    return url


def _strip_data_url_base64(value: str) -> str:
    if value.startswith("data:") and ";base64," in value:
        return value.split(";base64,", 1)[1]
    return value


_HY_V35_SIZE_ALIASES: dict[str, str] = {
    "1K": "1024x1024",
    "2K": "2048x2048",
    "4K": "4096x4096",
    "1:1": "1536x1536",
    "16:9": "2048x1152",
    "9:16": "1152x2048",
    "4:3": "1792x1344",
    "3:4": "1344x1792",
    "3:2": "1920x1280",
    "2:3": "1280x1920",
    "21:9": "2304x988",
}

_HY_V3_SIZE_ALIASES: dict[str, str] = {
    "1K": "1024x1024",
    "1:1": "1024x1024",
    "16:9": "1280x720",
    "9:16": "720x1280",
    "4:3": "1024x768",
    "3:4": "768x1024",
    "3:2": "1216x832",
    "2:3": "832x1216",
    "21:9": "1472x576",
}


def _normalize_image_size(size: str) -> str:
    match = _IMAGE_SIZE.fullmatch(size.strip())
    if not match:
        raise _unsupported("Tencent Cloud TokenHub image size must use WIDTHxHEIGHT format (e.g. 1024x1024).")
    return f"{match.group(1)}x{match.group(2)}"


def _normalize_hy_image_size(size: str, *, is_v35: bool) -> str:
    cleaned = size.strip()
    aliases = _HY_V35_SIZE_ALIASES if is_v35 else _HY_V3_SIZE_ALIASES
    mapped = aliases.get(cleaned.upper()) or aliases.get(cleaned)
    if mapped:
        return mapped
    match = _IMAGE_SIZE.fullmatch(cleaned)
    if not match:
        raise _unsupported("Tencent Cloud TokenHub image size must use WIDTHxHEIGHT format (e.g. 1024x1024).")
    width, height = int(match.group(1)), int(match.group(2))
    min_side, max_side, max_area = (256, 8192, 16_777_216) if is_v35 else (512, 2048, 1_048_576)
    if not (min_side <= width <= max_side and min_side <= height <= max_side and width * height <= max_area):
        raise _unsupported(
            f"Image size {width}x{height} exceeds TokenHub limits (side {min_side}-{max_side}, area <= {max_area})."
        )
    return f"{width}x{height}"


def _infer_tier(size: str | None) -> str:
    if not size:
        return "2k"
    return _infer_ratio_and_tier(size)[1]


def _infer_ratio_and_tier(size: str) -> tuple[str, str]:
    cleaned = size.strip()
    if _ASPECT_RATIO.fullmatch(cleaned):
        return cleaned, "2k"
    upper = cleaned.upper()
    if upper in {"1K", "2K", "4K", "8K"}:
        return "1:1", upper.lower()
    match = _IMAGE_SIZE.fullmatch(cleaned)
    if not match:
        return "1:1", "2k"
    width, height = int(match.group(1)), int(match.group(2))
    max_side = max(width, height)
    tier = "4k" if max_side >= 3200 else ("2k" if max_side >= 1600 else "1k")
    ratio = _closest_ratio(width, height)
    return ratio, tier


def _closest_ratio(width: int, height: int) -> str:
    if height <= 0:
        return "1:1"
    target = width / height
    candidates = (
        ("1:1", 1.0),
        ("16:9", 16 / 9),
        ("9:16", 9 / 16),
        ("4:3", 4 / 3),
        ("3:4", 3 / 4),
        ("3:2", 3 / 2),
        ("2:3", 2 / 3),
        ("21:9", 21 / 9),
    )
    return min(candidates, key=lambda item: abs(item[1] - target))[0]


def _map_hidream_size(size: str) -> str | None:
    cleaned = size.strip()
    if cleaned.lower() == "auto":
        return "auto"
    if cleaned in _HIDREAM_EXACT_SIZES:
        return cleaned
    ratio, _ = _infer_ratio_and_tier(cleaned)
    return _HIDREAM_SIZES.get(ratio)


def _normalize_wand_video_resolution(resolution: str) -> str:
    lower = resolution.strip().lower()
    if lower in {"1080p", "2k", "4k"}:
        return lower.upper()
    return "768P"


def _normalize_vidu_video_resolution(resolution: str, *, allow_uhd: bool) -> str:
    lower = resolution.strip().lower()
    if allow_uhd and lower in {"2k", "4k"}:
        return lower.upper()
    if lower in {"540p", "720p", "1080p"}:
        return lower
    if lower == "480p":
        return "540p"
    return "720p"


def _normalize_pixverse_quality(resolution: str) -> str:
    lower = resolution.strip().lower()
    if lower in {"360p", "540p", "720p", "1080p"}:
        return lower
    if lower == "480p":
        return "540p"
    return "720p"


def _validate_non_negative_seed(seed: int) -> int:
    if seed < 0:
        raise _unsupported("Tencent Cloud TokenHub seed must be non-negative.")
    return seed


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _image_first_required_error(model: str, *, kind: str, alternative: str) -> MediaGenerationError:
    tool_name = "generate_3d" if kind == "3d" else "generate_video"
    label = "3D" if kind == "3d" else "video"
    return MediaGenerationError(
        f"{model} only supports image-to-{label} generation and requires an input image.",
        code="unsupported_input",
        category="validation",
        safe_to_resubmit=True,
        remediation="adjust_request",
        model_instruction=(
            f"{model} requires an input image (image-to-{label}). First call generate_image to produce "
            f"a clean reference image, then call {tool_name} with that workspace image path "
            f"(or ask the admin to switch the {label} model to {alternative} for direct text-to-{label})."
        ),
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


__all__ = ["TencentTokenHubMediaProvider"]
