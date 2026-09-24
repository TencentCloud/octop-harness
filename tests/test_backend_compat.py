"""Tests for cos_spec_to_s3_compat (now in octop_harness.backends.s3_backend)."""

from __future__ import annotations

from octop_harness.backends.s3_backend import cos_spec_to_s3_compat


def test_cos_spec_to_s3_compat() -> None:
    spec = {
        "type": "cos",
        "bucket": "b-125",
        "region": "ap-guangzhou",
        "secret_id": "AKID",
        "secret_key": "SECRET",
        "prefix": "agents/",
    }
    out = cos_spec_to_s3_compat(spec)
    assert out is not None
    assert out["type"] == "s3"
    assert out["bucket"] == "b-125"
    assert out["endpoint_url"] == "https://cos.ap-guangzhou.myqcloud.com"
    assert out["prefix"] == "agents/"


def test_cos_spec_to_s3_compat_non_cos() -> None:
    assert cos_spec_to_s3_compat({"type": "filesystem", "root_dir": "/tmp"}) is None
