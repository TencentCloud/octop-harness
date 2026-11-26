"""Tests for memory extraction keep/skip rule config forwarding (TencentCloud/Octop#2963)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from octop_memory import MemoryService

from octop_harness.config import HarnessAgentConfig
from octop_harness.memory.runtime import MemoryRuntime, _extraction_service_config


def test_extract_rule_defaults_and_whitespace_normalization(tmp_path: Path) -> None:
    cfg = HarnessAgentConfig(
        name="rules-default",
        workspace_dir=tmp_path,
        memory_extract_keep_rules="   \n\t ",
        memory_extract_skip_rules="",
    )
    assert cfg.memory_extract_keep_rules is None
    assert cfg.memory_extract_skip_rules is None
    assert _extraction_service_config(cfg) is None


def test_extract_rules_roundtrip_and_forwarding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MemoryService, "supports_extraction_rules", True, raising=False)
    cfg = HarnessAgentConfig(
        name="rules-roundtrip",
        workspace_dir=tmp_path,
        memory_extract_max_tokens=1024,
        memory_extract_keep_rules="  Keep stable preferences and architecture decisions.  ",
        memory_extract_skip_rules="  Skip casual greetings and one-off lookups.  ",
    )
    assert cfg.memory_extract_keep_rules == "Keep stable preferences and architecture decisions."
    assert cfg.memory_extract_skip_rules == "Skip casual greetings and one-off lookups."

    rebuilt = HarnessAgentConfig.from_dict(cfg.to_dict())
    assert rebuilt.memory_extract_keep_rules == cfg.memory_extract_keep_rules
    assert rebuilt.memory_extract_skip_rules == cfg.memory_extract_skip_rules

    assert _extraction_service_config(cfg) == {
        "extraction": {
            "max_tokens": 1024,
            "keep_rules": "Keep stable preferences and architecture decisions.",
            "skip_rules": "Skip casual greetings and one-off lookups.",
        }
    }


def test_extract_rules_only_populates_present_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MemoryService, "supports_extraction_rules", True, raising=False)
    keep_only = HarnessAgentConfig(
        name="keep-only",
        workspace_dir=tmp_path,
        memory_extract_keep_rules="Keep product requirements",
    )
    assert _extraction_service_config(keep_only) == {"extraction": {"keep_rules": "Keep product requirements"}}

    skip_only = HarnessAgentConfig(
        name="skip-only",
        workspace_dir=tmp_path,
        memory_extract_skip_rules="Skip temporary debugging noise",
    )
    assert _extraction_service_config(skip_only) == {"extraction": {"skip_rules": "Skip temporary debugging noise"}}


@pytest.mark.parametrize("field_name", ["memory_extract_keep_rules", "memory_extract_skip_rules"])
def test_extract_rules_validation(tmp_path: Path, field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        HarnessAgentConfig(name="bad-type", workspace_dir=tmp_path, **{field_name: 123})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=field_name):
        HarnessAgentConfig(name="too-long", workspace_dir=tmp_path, **{field_name: "x" * 2001})


def test_legacy_memory_rejects_rules_but_preserves_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(MemoryService, "supports_extraction_rules", raising=False)
    config = HarnessAgentConfig(name="legacy", workspace_dir=tmp_path, memory_extract_max_tokens=512)
    assert _extraction_service_config(config) == {"extraction": {"max_tokens": 512}}
    with pytest.raises(ValueError, match="octop-memory"):
        _extraction_service_config(
            HarnessAgentConfig(name="legacy", workspace_dir=tmp_path, memory_extract_skip_rules="skip")
        )


@pytest.mark.skipif(
    not getattr(MemoryService, "supports_extraction_rules", False), reason="requires companion octop-memory"
)
def test_rules_reach_both_model_calls_through_harness_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    llm = Mock()
    llm.call_llm.side_effect = ['{"candidates": []}', '{"episodes": []}']
    monkeypatch.setattr(MemoryRuntime, "_build_llm_client", lambda self: llm)
    config = HarnessAgentConfig(
        name="rules",
        workspace_dir=tmp_path,
        memory_extract_keep_rules="Keep office preferences",
        memory_extract_skip_rules="Skip casual greetings",
    )
    runtime = MemoryRuntime(config=config, workspace_path=tmp_path, model_factory=Mock())
    try:
        runtime.service.capture_turn(user="I prefer quiet offices when writing.", session_id="s", thread_id="t")
        result = runtime.service.extract("s", promote=False, regen_pages=False)
        assert result["failure_reason"] is None
        assert llm.call_llm.call_count == 2
        for call in llm.call_llm.call_args_list:
            assert "Keep office preferences" in call.args[0]
            assert "Skip casual greetings" in call.args[0]
    finally:
        runtime.close()
