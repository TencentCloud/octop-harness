"""Provider-neutral media generation with built-in vendor adapters."""

from octop_harness.media.base import BaseMediaProvider, MediaGenerationProvider, ThreeDMediaProvider
from octop_harness.media.errors import MediaGenerationError
from octop_harness.media.manager import InMemoryMediaJobStore, MediaJobStore, MediaManager, MediaRouteInfo
from octop_harness.media.models import (
    GeneratedMedia,
    ImageGenerationRequest,
    MediaArtifactRole,
    MediaCapabilities,
    MediaGenerationResult,
    MediaJob,
    MediaJobUpdate,
    MediaSubmission,
    ProviderTask,
    RemoteArtifact,
    StoredMedia,
    ThreeDGenerationRequest,
    VideoGenerationRequest,
)
from octop_harness.media.providers.dashscope import DashScopeMediaProvider
from octop_harness.media.providers.minimax import MiniMaxMediaProvider
from octop_harness.media.providers.tencent_tokenhub import TencentTokenHubMediaProvider
from octop_harness.media.providers.volcengine import VolcengineMediaProvider
from octop_harness.media.registry import BUILTIN_MEDIA_PROVIDERS, create_media_manager, create_media_provider

__all__ = [
    "BUILTIN_MEDIA_PROVIDERS",
    "BaseMediaProvider",
    "DashScopeMediaProvider",
    "GeneratedMedia",
    "ImageGenerationRequest",
    "InMemoryMediaJobStore",
    "MediaArtifactRole",
    "MediaCapabilities",
    "MediaGenerationError",
    "MediaGenerationProvider",
    "MediaGenerationResult",
    "MediaJob",
    "MediaJobStore",
    "MediaJobUpdate",
    "MediaManager",
    "MediaRouteInfo",
    "MediaSubmission",
    "MiniMaxMediaProvider",
    "ProviderTask",
    "RemoteArtifact",
    "StoredMedia",
    "TencentTokenHubMediaProvider",
    "ThreeDGenerationRequest",
    "ThreeDMediaProvider",
    "VideoGenerationRequest",
    "VolcengineMediaProvider",
    "create_media_manager",
    "create_media_provider",
]
