"""S3-compatible backend (boto3) — covers AWS S3, MinIO, and custom stores.

This module also provides :func:`cos_spec_to_s3_compat` as a fallback spec
converter for Tencent Cloud COS when the native ``cos-python-sdk-v5`` is
unavailable.

Inherits all ``BackendProtocol`` methods from
:class:`~octop_harness.backends.cloud_storage_base.CloudStorageBackend`;
only the boto3 SDK calls are implemented here.

Storage layout
--------------
Every virtual file at ``/foo/bar.txt`` maps to an S3 object whose key is
``<prefix>foo/bar.txt``. The object body is JSON matching deepagents'
``FileData`` shape::

    {"content": "...", "encoding": "utf-8",
     "created_at": "...", "modified_at": "..."}

Compatibility notes
-------------------
- Uses S3 v2 signature (``signature_version="s3"``) and virtual-hosted style
  addressing to avoid the ``aws-chunked`` transfer encoding that SigV4 adds,
  which many non-AWS stores reject.
- Pagination uses ``list_objects_v2`` with the ``ContinuationToken`` pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from deepagents.backends.protocol import FileData, FileInfo

from octop_harness.backends.cloud_storage_base import CloudStorageBackend

logger = logging.getLogger(__name__)


@dataclass
class S3Config:
    """Connection parameters for an S3-compatible object store.

    Field names intentionally match the keys emitted by
    ``octop.infra.backend.adapter`` so that
    ``S3Backend(S3Config(**spec_kwargs))`` works directly.

    Attributes:
        bucket: Bucket name.
        access_key_id: Access key ID.
        secret_access_key: Secret access key.
        region: Region identifier (e.g. ``"us-east-1"``).
        endpoint_url: Custom endpoint URL including scheme.
        prefix: Key prefix applied to every virtual path. Empty by default.
        addressing_style: ``"virtual"`` (default) or ``"path"``.
            Virtual-hosted style is required by OSS, OBS, and most
            non-AWS stores.
    """

    bucket: str
    access_key_id: str
    secret_access_key: str
    region: str = ""
    endpoint_url: str = ""
    prefix: str = ""
    addressing_style: str = "virtual"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.bucket:
            raise ValueError("S3Config.bucket is required")
        if not self.access_key_id or not self.secret_access_key:
            raise ValueError("S3Config.access_key_id / secret_access_key are required")
        object.__setattr__(self, "prefix", self.prefix.strip("/"))

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> S3Config:
        """Build from a flat spec dict, ignoring unrecognised keys.

        Also accepts legacy boto3-style field names (``aws_access_key_id`` /
        ``aws_secret_access_key``) for backward compatibility.
        """
        # Normalise legacy boto3-style key names.
        if "aws_access_key_id" in kwargs and "access_key_id" not in kwargs:
            kwargs["access_key_id"] = kwargs.pop("aws_access_key_id")
        if "aws_secret_access_key" in kwargs and "secret_access_key" not in kwargs:
            kwargs["secret_access_key"] = kwargs.pop("aws_secret_access_key")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        extra = {k: v for k, v in kwargs.items() if k not in known}
        filtered = {k: v for k, v in kwargs.items() if k in known}
        return cls(**filtered, extra=extra)


class S3Backend(CloudStorageBackend):
    """S3-compatible virtual filesystem backend (boto3).

    Covers AWS S3, MinIO, and any S3-compatible store that accepts
    virtual-hosted style requests. All ``BackendProtocol`` methods are
    inherited from :class:`CloudStorageBackend`; only boto3 SDK calls
    are implemented here.
    """

    def __init__(self, config: S3Config) -> None:
        self._config = config
        self._client = self._build_client(config)

    @property
    def _prefix(self) -> str:
        return self._config.prefix

    def _storage_error_types(self) -> tuple[type[BaseException], ...]:
        from botocore.exceptions import ClientError

        return (*super()._storage_error_types(), ClientError)

    @staticmethod
    def _build_client(config: S3Config) -> Any:
        import boto3
        import botocore.config

        kwargs: dict[str, Any] = {
            "aws_access_key_id": config.access_key_id,
            "aws_secret_access_key": config.secret_access_key,
        }
        if config.region:
            kwargs["region_name"] = config.region
        if config.endpoint_url:
            kwargs["endpoint_url"] = config.endpoint_url
        kwargs["config"] = botocore.config.Config(
            signature_version="s3",
            s3={"addressing_style": config.addressing_style},
        )
        return boto3.client("s3", **kwargs)

    # ------------------------------------------------------------------
    # SDK primitives
    # ------------------------------------------------------------------

    def _get_file_data(self, path: str) -> FileData | None:
        from botocore.exceptions import ClientError

        try:
            resp = self._client.get_object(Bucket=self._config.bucket, Key=self._key(path))
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            http_status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in {"NoSuchKey", "404", "NotFound"} or http_status == 404:
                return None
            raise
        return self._parse_body(resp["Body"].read())

    def _put_file_data(self, path: str, file_data: FileData) -> None:
        self._client.put_object(
            Bucket=self._config.bucket,
            Key=self._key(path),
            Body=self._encode_file_data(file_data),
        )

    def _put_dir_marker(self, virtual_dir: str) -> None:
        self._client.put_object(
            Bucket=self._config.bucket,
            Key=self._prefix_key(virtual_dir),
            Body=b"",
        )

    def _ls_entries(self, path: str) -> list[FileInfo]:
        prefix_key = self._key(path)
        if prefix_key and not prefix_key.endswith("/"):
            prefix_key += "/"

        entries: list[FileInfo] = []
        continuation_token: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "Bucket": self._config.bucket,
                "Prefix": prefix_key,
                "Delimiter": "/",
                "MaxKeys": 1000,
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            resp = self._client.list_objects_v2(**kwargs)

            for cp in resp.get("CommonPrefixes") or []:
                key = cp["Prefix"] if isinstance(cp, dict) else cp
                entries.append({"path": self._virtual_path(key.rstrip("/")), "is_dir": True})
            for obj in resp.get("Contents") or []:
                key = obj["Key"] if isinstance(obj, dict) else obj
                if key == prefix_key:
                    continue
                info: FileInfo = {"path": self._virtual_path(key), "is_dir": False}
                if isinstance(obj, dict):
                    if (size := obj.get("Size")) is not None:
                        info["size"] = int(size)
                    if last_mod := obj.get("LastModified"):
                        info["modified_at"] = str(last_mod)
                entries.append(info)

            if not resp.get("IsTruncated"):
                break
            continuation_token = resp.get("NextContinuationToken") or ""
            if not continuation_token:
                break

        entries.sort(key=lambda e: e["path"])
        return entries

    def _collect_recursive(self, path: str) -> dict[str, FileData]:
        from botocore.exceptions import ClientError

        prefix_key = self._key(path)
        if prefix_key and not prefix_key.endswith("/") and path != "/":
            prefix_key += "/"

        files: dict[str, FileData] = {}
        continuation_token: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "Bucket": self._config.bucket,
                "Prefix": prefix_key,
                "MaxKeys": 1000,
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            resp = self._client.list_objects_v2(**kwargs)

            for obj in resp.get("Contents") or []:
                key = obj["Key"] if isinstance(obj, dict) else obj
                if key.endswith("/"):
                    continue
                vp = self._virtual_path(key)
                try:
                    fd = self._get_file_data(vp)
                except ClientError:
                    continue
                if fd is not None:
                    files[vp] = fd

            if not resp.get("IsTruncated"):
                break
            continuation_token = resp.get("NextContinuationToken") or ""
            if not continuation_token:
                break
        return files

    def _iter_prefix_object_keys(self, path: str) -> Any:
        prefix_key = self._prefix_key(path)
        continuation_token: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "Bucket": self._config.bucket,
                "Prefix": prefix_key,
                "MaxKeys": 1000,
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            resp = self._client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents") or []:
                yield obj["Key"] if isinstance(obj, dict) else obj
            if not resp.get("IsTruncated"):
                break
            continuation_token = resp.get("NextContinuationToken") or ""
            if not continuation_token:
                break

    def _prefix_has_objects(self, path: str) -> bool:
        resp = self._client.list_objects_v2(
            Bucket=self._config.bucket,
            Prefix=self._prefix_key(path),
            MaxKeys=1,
        )
        return bool(resp.get("Contents"))

    def _copy_object(self, src: str, dest: str) -> None:
        self._client.copy_object(
            Bucket=self._config.bucket,
            Key=self._key(dest),
            CopySource={"Bucket": self._config.bucket, "Key": self._key(src)},
        )

    def delete_object(self, path: str) -> None:
        """Delete the S3 object backing ``path``."""
        self._client.delete_object(Bucket=self._config.bucket, Key=self._key(path))

    def delete_prefix(self, path: str) -> int:
        """Delete every object under ``path``; return count of removed keys."""
        deleted = 0
        for key in self._iter_prefix_object_keys(path):
            self._client.delete_object(Bucket=self._config.bucket, Key=key)
            deleted += 1
        return deleted


# ---------------------------------------------------------------------------
# COS fallback spec converter
# ---------------------------------------------------------------------------


def cos_spec_to_s3_compat(spec: dict[str, Any]) -> dict[str, Any] | None:
    """Return an S3-protocol spec equivalent to a COS spec.

    Used when the native ``cos-python-sdk-v5`` is unavailable but boto3
    can still talk to Tencent COS via the S3-compatible API.
    """
    if spec.get("type") != "cos":
        return None
    bucket = spec.get("bucket")
    region = spec.get("region")
    secret_id = spec.get("secret_id")
    secret_key = spec.get("secret_key")
    if not all((bucket, region, secret_id, secret_key)):
        return None
    endpoint = spec.get("endpoint") or f"cos.{region}.myqcloud.com"
    if "://" not in str(endpoint):
        endpoint = f"https://{endpoint}"
    out: dict[str, Any] = {
        "type": "s3",
        "bucket": bucket,
        "access_key_id": secret_id,
        "secret_access_key": secret_key,
        "region": region,
        "endpoint_url": endpoint,
        "addressing_style": "virtual",
    }
    prefix = spec.get("prefix")
    if prefix:
        out["prefix"] = prefix
    return out


# Backward-compat aliases for code that imported from the old module names.
S3CompatConfig = S3Config
S3CompatBackend = S3Backend

__all__ = [
    "S3Backend",
    "S3CompatBackend",  # alias
    "S3CompatConfig",  # alias
    "S3Config",
    "cos_spec_to_s3_compat",
]
