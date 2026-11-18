"""Integration tests for the structured-form HITL lifecycle.

Covers the chain: HarnessAgent construction → interrupt_on registration →
tool mounting → submission validation → headless fallback.

These tests do *not* require a real LLM — they exercise the integration
points between agent wiring, tool schema, and the validators.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from octop_harness.config import HarnessAgentConfig, ProviderConfig, ModelConfig
from octop_harness.builtin.tools.structured_form import (
    STRUCTURED_FORM_HEADLESS_FALLBACK,
    STRUCTURED_FORM_PROTOCOL,
    STRUCTURED_FORM_SCHEMA_VERSION,
    STRUCTURED_FORM_TOOL_NAME,
    StructuredBooleanField,
    StructuredEnumField,
    StructuredFormField,
    StructuredFormRequestArgs,
    StructuredIntegerField,
    StructuredStringField,
    validate_structured_form_submission,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _base_cfg(tmp_path: Path, **overrides: object) -> HarnessAgentConfig:
    kwargs: dict[str, object] = {
        "name": "test-agent",
        "workspace_dir": tmp_path,
        "backend": {"type": "filesystem", "root_dir": str(tmp_path), "virtual_mode": False},
        "providers": [
            ProviderConfig(
                id="p",
                base_url="https://x",
                api_key="sk-test",
                models=[ModelConfig(id="text", input=["text"])],
            ),
        ],
        "default_model": "p/text",
        "memory_enabled": False,
        "checkpointer": False,
        "structured_form_enabled": True,
        **overrides,
    }
    return HarnessAgentConfig(**kwargs)  # type: ignore[arg-type]


# ── Interrupt registration ───────────────────────────────────────────


class TestInterruptRegistration:
    """Agent must register request_structured_form interrupt when enabled."""

    @pytest.mark.asyncio
    async def test_agent_registers_structured_form_interrupt(
        self, tmp_path: Path
    ) -> None:
        cfg = _base_cfg(tmp_path)

        with (
            patch("octop_harness.agent.HarnessAgent._wire_workspace_dotenv_for_execute"),
            patch("octop_harness.agent.HarnessAgent._build_backend"),
            patch("octop_harness.agent.HarnessAgent._init_model_factory"),
            patch("octop_harness.agent.MemoryRuntime"),
            patch("octop_harness.agent.HarnessAgent._build_graph"),
            patch("octop_harness.agent.HarnessAgent._init_protocols"),
            patch("octop_harness.agent.HarnessAgent._bind_plugin_model_factory"),
        ):
            from octop_harness.agent import HarnessAgent

            agent = HarnessAgent(cfg)
            interrupt_on = agent._resolve_interrupt_on()

        # Must contain the structured-form tool entry.
        assert STRUCTURED_FORM_TOOL_NAME in interrupt_on
        entry = interrupt_on[STRUCTURED_FORM_TOOL_NAME]
        assert entry["allowed_decisions"] == ["respond", "cancel"]
        assert "structured form" in entry["description"]

    @pytest.mark.asyncio
    async def test_agent_does_not_register_when_disabled(
        self, tmp_path: Path
    ) -> None:
        cfg = _base_cfg(tmp_path, structured_form_enabled=False)

        with (
            patch("octop_harness.agent.HarnessAgent._wire_workspace_dotenv_for_execute"),
            patch("octop_harness.agent.HarnessAgent._build_backend"),
            patch("octop_harness.agent.HarnessAgent._init_model_factory"),
            patch("octop_harness.agent.MemoryRuntime"),
            patch("octop_harness.agent.HarnessAgent._build_graph"),
            patch("octop_harness.agent.HarnessAgent._init_protocols"),
            patch("octop_harness.agent.HarnessAgent._bind_plugin_model_factory"),
        ):
            from octop_harness.agent import HarnessAgent

            agent = HarnessAgent(cfg)
            interrupt_on = agent._resolve_interrupt_on()

        assert STRUCTURED_FORM_TOOL_NAME not in interrupt_on

    @pytest.mark.asyncio
    async def test_tool_is_mounted_when_active(
        self, tmp_path: Path
    ) -> None:
        cfg = _base_cfg(tmp_path)

        with (
            patch("octop_harness.agent.HarnessAgent._wire_workspace_dotenv_for_execute"),
            patch("octop_harness.agent.HarnessAgent._build_backend"),
            patch("octop_harness.agent.HarnessAgent._init_model_factory"),
            patch("octop_harness.agent.MemoryRuntime"),
            patch("octop_harness.agent.HarnessAgent._build_graph"),
            patch("octop_harness.agent.HarnessAgent._init_protocols"),
            patch("octop_harness.agent.HarnessAgent._bind_plugin_model_factory"),
            patch("octop_harness.agent.build_media_generation_tools", return_value=[]),
        ):
            from octop_harness.agent import HarnessAgent

            agent = HarnessAgent(cfg)
            tools = agent._build_tools()

        tool_names = [
            getattr(t, "name", None) or getattr(t, "__name__", None) for t in tools
        ]
        assert STRUCTURED_FORM_TOOL_NAME in tool_names

    @pytest.mark.asyncio
    async def test_tool_is_not_mounted_when_disabled(
        self, tmp_path: Path
    ) -> None:
        cfg = _base_cfg(tmp_path, structured_form_enabled=False)

        with (
            patch("octop_harness.agent.HarnessAgent._wire_workspace_dotenv_for_execute"),
            patch("octop_harness.agent.HarnessAgent._build_backend"),
            patch("octop_harness.agent.HarnessAgent._init_model_factory"),
            patch("octop_harness.agent.MemoryRuntime"),
            patch("octop_harness.agent.HarnessAgent._build_graph"),
            patch("octop_harness.agent.HarnessAgent._init_protocols"),
            patch("octop_harness.agent.HarnessAgent._bind_plugin_model_factory"),
            patch("octop_harness.agent.build_media_generation_tools", return_value=[]),
        ):
            from octop_harness.agent import HarnessAgent

            agent = HarnessAgent(cfg)
            tools = agent._build_tools()

        tool_names = [
            getattr(t, "name", None) or getattr(t, "__name__", None) for t in tools
        ]
        assert STRUCTURED_FORM_TOOL_NAME not in tool_names


# ── Headless fallback ────────────────────────────────────────────────


class TestHeadlessFallback:
    """When there is no HITL channel the tool returns a safe fallback."""

    def test_fallback_returns_static_message(self) -> None:
        from octop_harness.builtin.tools.structured_form import request_structured_form

        result = request_structured_form.func(
            form_protocol=STRUCTURED_FORM_PROTOCOL,
            form_id="test",
            schema_version=STRUCTURED_FORM_SCHEMA_VERSION,
            title="Test Form",
            fields=[],
        )
        assert result == STRUCTURED_FORM_HEADLESS_FALLBACK
        assert "Do not call" in result

    def test_fallback_tells_model_not_to_invent_values(self) -> None:
        from octop_harness.builtin.tools.structured_form import request_structured_form

        result = request_structured_form.func(
            form_protocol=STRUCTURED_FORM_PROTOCOL,
            form_id="deploy-config",
            schema_version=STRUCTURED_FORM_SCHEMA_VERSION,
            title="Deploy Params",
            fields=[],
            timeout_seconds=300,
        )
        assert "inventing field values" in result


# ── Submission validation lifecycle ──────────────────────────────────


class TestSubmissionLifecycle:
    """Integration of field schema → value validation."""

    def test_all_four_field_types_validate_together(self) -> None:
        fields: list[StructuredFormField] = [
            StructuredIntegerField(key="replicas", label="Replicas", required=True, nullable=False, min=1, max=50, default=3),
            StructuredStringField(key="env", label="Environment", required=True, nullable=False, pattern=r"^(dev|staging|prod)$"),
            StructuredBooleanField(key="public", label="Public IP", default=False),
            StructuredEnumField(key="tier", label="Tier", options=["free", "pro", "enterprise"], required=True, nullable=False),
        ]

        # Valid submission.
        errors = validate_structured_form_submission(
            fields,
            {"replicas": 42, "env": "prod", "public": True, "tier": "enterprise"},
        )
        assert errors == []

    def test_field_level_error_locations_are_preserved(self) -> None:
        fields: list[StructuredFormField] = [
            StructuredIntegerField(key="count", label="Count", required=True, nullable=False, min=0, max=10),
            StructuredStringField(key="name", label="Name", required=True, nullable=False, min_length=3),
            StructuredEnumField(key="region", label="Region", options=["us", "eu"], required=True, nullable=False),
        ]

        errors = validate_structured_form_submission(
            fields,
            {"count": 999, "name": "ab", "region": "asia"},
        )
        # All three fields should report errors with correct keys.
        error_keys = {e.field for e in errors}
        assert error_keys == {"count", "name", "region"}
        # At least one error per field.
        assert sum(1 for e in errors if e.field == "count") >= 1
        assert sum(1 for e in errors if e.field == "name") >= 1
        assert sum(1 for e in errors if e.field == "region") >= 1

    def test_missing_required_fields_are_reported_per_field(self) -> None:
        fields: list[StructuredFormField] = [
            StructuredIntegerField(key="a", label="A", required=True, nullable=False),
            StructuredIntegerField(key="b", label="B", required=False),
            StructuredIntegerField(key="c", label="C", required=True, nullable=False),
        ]

        errors = validate_structured_form_submission(fields, {"b": 1})
        error_keys = {e.field for e in errors}
        assert error_keys == {"a", "c"}

    def test_null_semantics_are_per_field(self) -> None:
        fields: list[StructuredFormField] = [
            StructuredIntegerField(key="x", label="X", required=True, nullable=False),
            StructuredIntegerField(key="y", label="Y", required=False, nullable=True),
        ]
        # x=null should fail (required + non-nullable), y=null should pass.
        errors = validate_structured_form_submission(fields, {"x": None, "y": None})
        assert any(e.field == "x" for e in errors)
        assert not any(e.field == "y" for e in errors)

    def test_unknown_field_is_reported(self) -> None:
        fields: list[StructuredFormField] = [
            StructuredIntegerField(key="known", label="Known"),
        ]
        errors = validate_structured_form_submission(
            fields, {"known": 1, "secret": "oops"}
        )
        assert any(e.field == "secret" and "unknown" in e.message for e in errors)


# ── End-to-end: construction → args → headless ───────────────────────


class TestRequestArgsLifecycle:
    """Args schema is constructable and can represent the four field types."""

    def test_request_args_accept_all_four_field_types(self) -> None:
        args = StructuredFormRequestArgs(
            form_protocol=STRUCTURED_FORM_PROTOCOL,
            form_id="test-form",
            schema_version=STRUCTURED_FORM_SCHEMA_VERSION,
            title="Test",
            fields=[
                StructuredIntegerField(key="num", label="Number", required=True, nullable=False),
                StructuredStringField(key="str", label="String", required=True, nullable=False),
                StructuredBooleanField(key="bool", label="Toggle"),
                StructuredEnumField(key="choice", label="Choice", options=["a", "b"]),
            ],
        )
        assert len(args.fields) == 4

    def test_request_args_rejects_empty_fields(self) -> None:
        with pytest.raises(Exception):
            StructuredFormRequestArgs(
                form_protocol=STRUCTURED_FORM_PROTOCOL,
                form_id="test-form",
                schema_version=STRUCTURED_FORM_SCHEMA_VERSION,
                title="Test",
                fields=[],
            )
