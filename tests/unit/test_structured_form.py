"""Unit tests for the typed structured HITL form tool."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from octop_harness.builtin.tools.structured_form import (
    MAX_STRUCTURED_FORM_CONTEXT_KEYS,
    MAX_STRUCTURED_FORM_FIELDS,
    STRUCTURED_FORM_HEADLESS_FALLBACK,
    STRUCTURED_FORM_PROTOCOL,
    STRUCTURED_FORM_SCHEMA_VERSION,
    STRUCTURED_FORM_TOOL_NAME,
    StructuredBooleanField,
    StructuredEnumField,
    StructuredFormRequestArgs,
    StructuredFormResponse,
    StructuredIntegerField,
    StructuredStringField,
    request_structured_form,
    validate_structured_form_submission,
)


def _request(fields: list[dict[str, Any]], **overrides: Any) -> StructuredFormRequestArgs:
    payload: dict[str, Any] = {"form_id": "test_form", "title": "Test Form", "fields": fields}
    payload.update(overrides)
    return StructuredFormRequestArgs(**payload)


# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------


def test_protocol_constants_are_stable() -> None:
    assert STRUCTURED_FORM_PROTOCOL == "octop.structured_form.v1"
    assert STRUCTURED_FORM_SCHEMA_VERSION == 1
    assert STRUCTURED_FORM_TOOL_NAME == "request_structured_form"
    assert STRUCTURED_FORM_HEADLESS_FALLBACK


# ---------------------------------------------------------------------------
# Integer field
# ---------------------------------------------------------------------------


def test_integer_field_accepts_valid_definition() -> None:
    field = StructuredIntegerField(key="replicas", label="Replicas", min=1, max=50, default=3)
    assert field.type == "integer"
    assert field.step == 1
    assert field.default == 3


def test_integer_field_rejects_inverted_range() -> None:
    with pytest.raises(ValidationError):
        StructuredIntegerField(key="n", label="N", min=10, max=5)


def test_integer_field_rejects_negative_min() -> None:
    with pytest.raises(ValidationError):
        StructuredIntegerField(key="n", label="N", min=-1)


def test_integer_field_rejects_required_with_nullable() -> None:
    with pytest.raises(ValidationError):
        StructuredIntegerField(key="n", label="N", required=True)


def test_integer_field_rejects_default_below_min() -> None:
    with pytest.raises(ValidationError):
        StructuredIntegerField(key="n", label="N", min=5, default=1)


def test_integer_field_rejects_default_above_max() -> None:
    with pytest.raises(ValidationError):
        StructuredIntegerField(key="n", label="N", max=5, default=9)


def test_integer_field_rejects_bad_key() -> None:
    with pytest.raises(ValidationError):
        StructuredIntegerField(key="Bad-Key", label="N")


# ---------------------------------------------------------------------------
# String field
# ---------------------------------------------------------------------------


def test_string_field_accepts_valid_definition() -> None:
    field = StructuredStringField(
        key="name",
        label="Name",
        min_length=3,
        max_length=20,
        pattern=r"^[a-z][a-z0-9-]*$",
        default="web-1",
    )
    assert field.type == "string"
    assert field.default == "web-1"
    assert field.multiline is False


def test_string_field_rejects_inverted_length() -> None:
    with pytest.raises(ValidationError):
        StructuredStringField(key="s", label="S", min_length=10, max_length=5)


def test_string_field_rejects_invalid_pattern() -> None:
    with pytest.raises(ValidationError):
        StructuredStringField(key="s", label="S", pattern="[unclosed")


def test_string_field_rejects_default_too_short() -> None:
    with pytest.raises(ValidationError):
        StructuredStringField(key="s", label="S", min_length=5, default="ab")


def test_string_field_rejects_default_too_long() -> None:
    with pytest.raises(ValidationError):
        StructuredStringField(key="s", label="S", max_length=3, default="abcdef")


def test_string_field_rejects_default_not_matching_pattern() -> None:
    with pytest.raises(ValidationError):
        StructuredStringField(key="s", label="S", pattern=r"^\d+$", default="abc")


def test_string_field_rejects_oversized_max_length() -> None:
    with pytest.raises(ValidationError):
        StructuredStringField(key="s", label="S", max_length=100_000)


# ---------------------------------------------------------------------------
# Boolean field
# ---------------------------------------------------------------------------


def test_boolean_field_defaults_to_non_nullable() -> None:
    field = StructuredBooleanField(key="public", label="Public")
    assert field.type == "boolean"
    assert field.nullable is False


def test_boolean_field_rejects_explicit_nullable() -> None:
    with pytest.raises(ValidationError):
        StructuredBooleanField(key="flag", label="Flag", nullable=True)


def test_boolean_field_accepts_default() -> None:
    field = StructuredBooleanField(key="flag", label="Flag", default=True)
    assert field.default is True


# ---------------------------------------------------------------------------
# Enum field
# ---------------------------------------------------------------------------


def test_enum_field_accepts_valid_options() -> None:
    field = StructuredEnumField(key="channel", label="Channel", options=["sms", "email"], default="sms")
    assert field.type == "enum"
    assert field.options == ["sms", "email"]


def test_enum_field_rejects_empty_options() -> None:
    with pytest.raises(ValidationError):
        StructuredEnumField(key="e", label="E", options=[])


def test_enum_field_rejects_duplicate_options() -> None:
    with pytest.raises(ValidationError):
        StructuredEnumField(key="e", label="E", options=["a", "a"])


def test_enum_field_rejects_blank_option() -> None:
    with pytest.raises(ValidationError):
        StructuredEnumField(key="e", label="E", options=["a", "   "])


def test_enum_field_rejects_default_outside_options() -> None:
    with pytest.raises(ValidationError):
        StructuredEnumField(key="e", label="E", options=["a", "b"], default="c")


def test_enum_field_rejects_oversized_option() -> None:
    with pytest.raises(ValidationError):
        StructuredEnumField(key="e", label="E", options=["x" * 121])


# ---------------------------------------------------------------------------
# Request envelope
# ---------------------------------------------------------------------------


def test_request_accepts_all_four_field_types() -> None:
    request = _request(
        [
            {"key": "replicas", "label": "Replicas", "type": "integer", "min": 1, "max": 50},
            {"key": "name", "label": "Name", "type": "string"},
            {"key": "public", "label": "Public", "type": "boolean"},
            {"key": "channel", "label": "Channel", "type": "enum", "options": ["sms", "email"]},
        ],
        timeout_seconds=300,
    )
    assert [field.type for field in request.fields] == ["integer", "string", "boolean", "enum"]
    assert request.timeout_seconds == 300


def test_request_rejects_duplicate_field_keys() -> None:
    with pytest.raises(ValidationError):
        _request(
            [
                {"key": "a", "label": "A", "type": "integer"},
                {"key": "a", "label": "B", "type": "string"},
            ]
        )


def test_request_rejects_unknown_field_type() -> None:
    with pytest.raises(ValidationError):
        _request([{"key": "a", "label": "A", "type": "float"}])


def test_request_rejects_empty_field_list() -> None:
    with pytest.raises(ValidationError):
        _request([])


def test_request_rejects_too_many_fields() -> None:
    fields = [{"key": f"f{i}", "label": f"F{i}", "type": "integer"} for i in range(MAX_STRUCTURED_FORM_FIELDS + 1)]
    with pytest.raises(ValidationError):
        _request(fields)


def test_request_accepts_max_fields() -> None:
    fields = [{"key": f"f{i}", "label": f"F{i}", "type": "integer"} for i in range(MAX_STRUCTURED_FORM_FIELDS)]
    assert len(_request(fields).fields) == MAX_STRUCTURED_FORM_FIELDS


def test_request_rejects_too_many_context_keys() -> None:
    context = {f"k{i}": i for i in range(MAX_STRUCTURED_FORM_CONTEXT_KEYS + 1)}
    with pytest.raises(ValidationError):
        _request([{"key": "a", "label": "A", "type": "integer"}], context=context)


@pytest.mark.parametrize("timeout", [30, 60, 300, 3600])
def test_request_accepts_timeout_within_bounds(timeout: int) -> None:
    assert _request([{"key": "a", "label": "A", "type": "integer"}], timeout_seconds=timeout).timeout_seconds == timeout


@pytest.mark.parametrize("timeout", [0, 29, 3601, -5])
def test_request_rejects_timeout_out_of_bounds(timeout: int) -> None:
    with pytest.raises(ValidationError):
        _request([{"key": "a", "label": "A", "type": "integer"}], timeout_seconds=timeout)


def test_request_rejects_wrong_protocol() -> None:
    with pytest.raises(ValidationError):
        _request([{"key": "a", "label": "A", "type": "integer"}], form_protocol="octop.structured_form.v2")


def test_request_rejects_bad_form_id() -> None:
    with pytest.raises(ValidationError):
        _request([{"key": "a", "label": "A", "type": "integer"}], form_id="Bad Form")


def test_request_forbids_extra_keys() -> None:
    with pytest.raises(ValidationError):
        _request([{"key": "a", "label": "A", "type": "integer"}], unexpected=True)


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------


def test_response_accepts_mixed_value_types() -> None:
    response = StructuredFormResponse(
        form_id="test_form",
        submission_id="sub-1",
        values={"replicas": 3, "name": "web-1", "public": True, "channel": "sms", "note": None},
    )
    assert response.values["replicas"] == 3
    assert response.values["public"] is True


def test_response_rejects_empty_submission_id() -> None:
    with pytest.raises(ValidationError):
        StructuredFormResponse(form_id="test_form", submission_id="", values={})


def test_response_forbids_extra_keys() -> None:
    with pytest.raises(ValidationError):
        StructuredFormResponse(form_id="test_form", submission_id="s", values={}, extra=1)


# ---------------------------------------------------------------------------
# Submission validation
# ---------------------------------------------------------------------------


def _fields() -> list[Any]:
    return _request(
        [
            {
                "key": "replicas",
                "label": "Replicas",
                "type": "integer",
                "required": True,
                "nullable": False,
                "min": 1,
                "max": 50,
            },
            {"key": "name", "label": "Name", "type": "string", "min_length": 3, "max_length": 20},
            {"key": "public", "label": "Public", "type": "boolean"},
            {"key": "channel", "label": "Channel", "type": "enum", "options": ["sms", "email"]},
        ]
    ).fields


def test_submission_accepts_valid_values() -> None:
    values = {"replicas": 3, "name": "web-1", "public": True, "channel": "sms"}
    assert validate_structured_form_submission(_fields(), values) == []


def test_submission_reports_missing_required_field() -> None:
    errors = validate_structured_form_submission(_fields(), {"name": "web-1"})
    assert [(error.field, error.message) for error in errors] == [("replicas", "required value is missing")]


def test_submission_reports_null_required_field() -> None:
    errors = validate_structured_form_submission(_fields(), {"replicas": None})
    assert [error.field for error in errors] == ["replicas"]
    assert errors[0].message == "required value must not be null"


def test_submission_reports_null_for_non_nullable_nullable_field() -> None:
    errors = validate_structured_form_submission(_fields(), {"replicas": 1, "public": None})
    assert [(error.field, error.message) for error in errors] == [("public", "value must not be null")]


def test_submission_allows_omitted_optional_fields() -> None:
    assert validate_structured_form_submission(_fields(), {"replicas": 1}) == []


def test_submission_reports_unknown_field() -> None:
    errors = validate_structured_form_submission(_fields(), {"replicas": 1, "ghost": 2})
    assert [(error.field, error.message) for error in errors] == [("ghost", "unknown field")]


def test_submission_reports_integer_above_max() -> None:
    errors = validate_structured_form_submission(_fields(), {"replicas": 99})
    message = "value must be less than or equal to 50"
    assert [(error.field, error.message) for error in errors] == [("replicas", message)]


def test_submission_reports_integer_below_min() -> None:
    errors = validate_structured_form_submission(_fields(), {"replicas": 0})
    message = "value must be greater than or equal to 1"
    assert [(error.field, error.message) for error in errors] == [("replicas", message)]


def test_submission_rejects_boolean_for_integer() -> None:
    errors = validate_structured_form_submission(_fields(), {"replicas": True})
    assert [(error.field, error.message) for error in errors] == [("replicas", "value must be an integer")]


def test_submission_reports_string_too_short() -> None:
    errors = validate_structured_form_submission(_fields(), {"replicas": 1, "name": "ab"})
    assert [(error.field, error.message) for error in errors] == [("name", "value must be at least 3 characters")]


def test_submission_reports_string_too_long() -> None:
    errors = validate_structured_form_submission(_fields(), {"replicas": 1, "name": "x" * 21})
    assert [(error.field, error.message) for error in errors] == [("name", "value must be at most 20 characters")]


def test_submission_rejects_non_string_for_string_field() -> None:
    errors = validate_structured_form_submission(_fields(), {"replicas": 1, "name": 42})
    assert [(error.field, error.message) for error in errors] == [("name", "value must be a string")]


def test_submission_rejects_non_boolean_for_boolean_field() -> None:
    errors = validate_structured_form_submission(_fields(), {"replicas": 1, "public": "yes"})
    assert [(error.field, error.message) for error in errors] == [("public", "value must be a boolean")]


def test_submission_rejects_unknown_enum_option() -> None:
    errors = validate_structured_form_submission(_fields(), {"replicas": 1, "channel": "slack"})
    assert [(error.field, error.message) for error in errors] == [("channel", "value must be one of: sms, email")]


def test_submission_reports_every_invalid_field_at_once() -> None:
    values = {"replicas": 99, "name": "ab", "public": "yes", "channel": "slack", "ghost": 1}
    errors = validate_structured_form_submission(_fields(), values)
    assert {error.field for error in errors} == {"replicas", "name", "public", "channel", "ghost"}


def test_submission_applies_string_pattern() -> None:
    fields = _request([{"key": "name", "label": "Name", "type": "string", "pattern": r"^[a-z][a-z0-9-]*$"}]).fields
    assert validate_structured_form_submission(fields, {"name": "web-1"}) == []
    errors = validate_structured_form_submission(fields, {"name": "Web_1"})
    assert [(error.field, error.message) for error in errors] == [("name", "value does not match the required pattern")]


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------


def test_tool_name_matches_protocol_constant() -> None:
    assert request_structured_form.name == STRUCTURED_FORM_TOOL_NAME
    assert request_structured_form.args_schema is StructuredFormRequestArgs


def test_headless_invocation_returns_fallback() -> None:
    result = request_structured_form.invoke(
        {
            "form_protocol": STRUCTURED_FORM_PROTOCOL,
            "form_id": "test_form",
            "schema_version": STRUCTURED_FORM_SCHEMA_VERSION,
            "title": "Test Form",
            "fields": [{"key": "count", "label": "Count", "type": "integer"}],
        }
    )
    assert result == STRUCTURED_FORM_HEADLESS_FALLBACK


def test_headless_invocation_accepts_timeout_seconds() -> None:
    result = request_structured_form.invoke(
        {
            "form_protocol": STRUCTURED_FORM_PROTOCOL,
            "form_id": "test_form",
            "schema_version": STRUCTURED_FORM_SCHEMA_VERSION,
            "title": "Test Form",
            "fields": [{"key": "count", "label": "Count", "type": "integer"}],
            "timeout_seconds": 300,
        }
    )
    assert result == STRUCTURED_FORM_HEADLESS_FALLBACK


def test_headless_invocation_accepts_all_field_types() -> None:
    result = request_structured_form.invoke(
        {
            "form_protocol": STRUCTURED_FORM_PROTOCOL,
            "form_id": "test_form",
            "schema_version": STRUCTURED_FORM_SCHEMA_VERSION,
            "title": "Test Form",
            "fields": [
                {"key": "replicas", "label": "Replicas", "type": "integer"},
                {"key": "name", "label": "Name", "type": "string"},
                {"key": "public", "label": "Public", "type": "boolean"},
                {"key": "channel", "label": "Channel", "type": "enum", "options": ["sms", "email"]},
            ],
        }
    )
    assert result == STRUCTURED_FORM_HEADLESS_FALLBACK
