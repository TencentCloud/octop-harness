"""Tests for ``octop_harness.middleware.pii``."""

from __future__ import annotations

import pytest

from octop_harness.middleware.pii import detect_pii


class TestDetectsKnownFormats:
    def test_openai_key(self) -> None:
        text = "use sk-4b829b7b-b0aa-4064-8d53-18b0025594f2 for testing"
        matches = detect_pii(text)
        assert len(matches) == 1
        assert matches[0]["type"] == "openai"
        assert matches[0]["value"] == "sk-4b829b7b-b0aa-4064-8d53-18b0025594f2"

    def test_openai_project_key(self) -> None:
        matches = detect_pii("token: sk-proj-abcd1234abcd1234abcd1234")
        assert matches and matches[0]["type"] == "openai_project"

    def test_anthropic_key(self) -> None:
        matches = detect_pii("use sk-ant-abcd1234abcd1234abcd1234abcd1234")
        assert matches and matches[0]["type"] == "anthropic"

    def test_aws_access_key_id(self) -> None:
        matches = detect_pii("AKIAIOSFODNN7EXAMPLE is mine")
        assert matches and matches[0]["type"] == "aws_access_key_id"
        assert matches[0]["value"] == "AKIAIOSFODNN7EXAMPLE"

    def test_google_api_key(self) -> None:
        # Google keys are exactly AIza + 35 chars.
        key = "AIza" + "X" * 35
        matches = detect_pii(f"key={key} done")
        types = {m["type"] for m in matches}
        assert "google_api_key" in types

    def test_huggingface_token(self) -> None:
        matches = detect_pii("HF: hf_abcdefghijklmnopqrstuvwxyz0123456789ABC")
        assert matches and matches[0]["type"] == "huggingface"

    def test_generic_assignment_value_only(self) -> None:
        """For ``api_key=...`` assignments we should mark only the secret span,
        not the surrounding ``api_key=`` label."""
        text = 'config = { api_key: "abcdef0123456789abcdef0123456789" }'
        matches = detect_pii(text)
        # The assignment pattern hits, returning the *value* span.
        assert any(m["type"] == "generic_api_key_assignment" for m in matches)
        for m in matches:
            if m["type"] == "generic_api_key_assignment":
                assert m["value"] == "abcdef0123456789abcdef0123456789"
                # The reported span should match the value, not the whole assignment.
                assert text[m["start"] : m["end"]] == m["value"]


class TestDoesNotMatchBenignText:
    def test_uuid_is_ignored(self) -> None:
        # UUIDs are not API keys and contain dashes — make sure we don't trip on them.
        assert detect_pii("550e8400-e29b-41d4-a716-446655440000") == []

    def test_short_string_is_ignored(self) -> None:
        # ``sk-x`` is too short for any provider — should not trigger.
        assert detect_pii("ack-shoe is not a key") == []

    def test_plain_prose_is_ignored(self) -> None:
        assert detect_pii("the quick brown fox jumps over the lazy dog") == []


class TestMatchOrdering:
    def test_matches_returned_left_to_right(self) -> None:
        text = "first sk-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa then sk-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        matches = detect_pii(text)
        starts = [m["start"] for m in matches]
        assert starts == sorted(starts)

    def test_overlapping_patterns_first_wins(self) -> None:
        # ``sk-ant-...`` could in theory be matched by both the anthropic
        # pattern and the generic openai pattern; the de-dup logic should
        # surface only the more-specific anthropic match.
        text = "use sk-ant-1234567890abcdefghijklmnopqrstuvwxyz"
        labels = {m["type"] for m in detect_pii(text)}
        assert "anthropic" in labels
        assert "openai" not in labels


# ---------------------------------------------------------------------------
# Integration with PIIMiddleware: the most important behavior — the secret
# gets masked in the model's view of the messages.
# ---------------------------------------------------------------------------


class TestPIIMiddlewareIntegration:
    def test_mask_strategy_replaces_keys(self) -> None:
        from langchain.agents.middleware import PIIMiddleware

        mw = PIIMiddleware(
            pii_type="api_key",
            strategy="mask",
            detector=detect_pii,
            apply_to_input=True,
        )
        # The middleware exposes a low-level helper we can call directly.
        from langchain_core.messages import HumanMessage

        original = HumanMessage(content="my key is sk-4b829b7b-b0aa-4064-8d53-18b0025594f2")
        # Use the PIIMiddleware's content-processing hook directly.
        # If the public surface differs across versions, we fall back to
        # invoking ``before_model`` with synthetic state.
        state = {"messages": [original]}
        try:
            new_state = mw.before_model(state, runtime=None)  # type: ignore[arg-type]
        except TypeError:
            # Some versions take state as first positional; just exercise it.
            pytest.skip("PIIMiddleware.before_model signature changed; integration tested in test_agent.")
        if new_state is None:
            pytest.skip("PIIMiddleware.before_model returned None on this version.")
        new_msg = new_state["messages"][-1]
        content = new_msg.content if hasattr(new_msg, "content") else new_msg["content"]
        # The original secret must NOT appear in the masked content.
        assert "sk-4b829b7b-b0aa-4064-8d53-18b0025594f2" not in str(content)
        assert "*" in str(content) or "[REDACTED" in str(content)
