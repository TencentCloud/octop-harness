"""Built-in tool for typed, declarative Human-in-the-Loop forms.

The protocol supports a bounded list of typed fields -- ``integer``,
``string``, ``boolean`` and ``enum`` -- each carrying its own declarative
constraints and optional default. The host renders the request and sends a
structured response through the normal HITL ``respond`` decision; this
function is only the non-blocking headless fallback.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Self

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

STRUCTURED_FORM_TOOL_NAME = "request_structured_form"
STRUCTURED_FORM_PROTOCOL = "octop.structured_form.v1"
STRUCTURED_FORM_SCHEMA_VERSION = 1
MAX_STRUCTURED_FORM_FIELDS = 64
MAX_STRUCTURED_FORM_CONTEXT_KEYS = 16
MAX_STRUCTURED_FORM_OPTIONS = 64
MAX_STRUCTURED_FORM_TEXT_LENGTH = 4096
MAX_STRUCTURED_FORM_PATTERN_LENGTH = 256

STRUCTURED_FORM_HEADLESS_FALLBACK = (
    "No interactive user is available in this run. Do not call "
    "request_structured_form again. State that the structured form could not "
    "be collected and continue without inventing field values."
)

_FIELD_KEY_PATTERN = r"^[a-z][a-z0-9_]*$"


class StructuredFormFieldError(BaseModel):
    """Field-scoped validation error so a renderer can highlight the control."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str
    message: str


class StructuredFieldBase(BaseModel):
    """Declarative surface shared by every supported field type."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., min_length=1, max_length=64, pattern=_FIELD_KEY_PATTERN)
    label: str = Field(..., min_length=1, max_length=120)
    required: bool = False
    nullable: bool = True
    description: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def _validate_nullability(self) -> Self:
        if self.required and self.nullable:
            raise ValueError(f"field '{self.key}': required fields must set nullable=false")
        return self


class StructuredIntegerField(StructuredFieldBase):
    """Integer field with an inclusive range, unit step and optional default."""

    type: Literal["integer"] = "integer"
    min: int | None = Field(default=0, ge=0)
    max: int | None = Field(default=None, ge=0)
    step: Literal[1] = 1
    default: int | None = None

    @model_validator(mode="after")
    def _validate_integer(self) -> Self:
        if self.max is not None and self.min is not None and self.max < self.min:
            raise ValueError(f"field '{self.key}': max must be greater than or equal to min")
        if self.default is None:
            return self
        if self.min is not None and self.default < self.min:
            raise ValueError(f"field '{self.key}': default must be greater than or equal to min")
        if self.max is not None and self.default > self.max:
            raise ValueError(f"field '{self.key}': default must be less than or equal to max")
        return self


class StructuredStringField(StructuredFieldBase):
    """Free-text field with optional length and regular-expression constraints."""

    type: Literal["string"] = "string"
    min_length: int = Field(default=0, ge=0, le=MAX_STRUCTURED_FORM_TEXT_LENGTH)
    max_length: int | None = Field(default=None, ge=1, le=MAX_STRUCTURED_FORM_TEXT_LENGTH)
    pattern: str | None = Field(default=None, max_length=MAX_STRUCTURED_FORM_PATTERN_LENGTH)
    multiline: bool = False
    default: str | None = None

    @model_validator(mode="after")
    def _validate_string(self) -> Self:
        if self.max_length is not None and self.max_length < self.min_length:
            raise ValueError(f"field '{self.key}': max_length must be greater than or equal to min_length")
        compiled = self._compiled_pattern()
        if self.default is None:
            return self
        if len(self.default) < self.min_length:
            raise ValueError(f"field '{self.key}': default is shorter than min_length")
        if self.max_length is not None and len(self.default) > self.max_length:
            raise ValueError(f"field '{self.key}': default is longer than max_length")
        if compiled is not None and compiled.search(self.default) is None:
            raise ValueError(f"field '{self.key}': default does not match pattern")
        return self

    def _compiled_pattern(self) -> re.Pattern[str] | None:
        if self.pattern is None:
            return None
        try:
            return re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(f"field '{self.key}': pattern is not a valid regular expression ({exc})") from exc


class StructuredBooleanField(StructuredFieldBase):
    """Boolean toggle. ``null`` carries no boolean meaning, so nullable is false."""

    type: Literal["boolean"] = "boolean"
    nullable: bool = False
    default: bool | None = None

    @model_validator(mode="after")
    def _validate_boolean(self) -> Self:
        if self.nullable:
            raise ValueError(f"field '{self.key}': boolean fields do not support nullable=true")
        return self


class StructuredEnumField(StructuredFieldBase):
    """Constrained choice field backed by a fixed, unique option list."""

    type: Literal["enum"] = "enum"
    options: list[str] = Field(..., min_length=1, max_length=MAX_STRUCTURED_FORM_OPTIONS)
    default: str | None = None

    @model_validator(mode="after")
    def _validate_enum(self) -> Self:
        if any(not option.strip() for option in self.options):
            raise ValueError(f"field '{self.key}': options must not contain empty values")
        if any(len(option) > 120 for option in self.options):
            raise ValueError(f"field '{self.key}': options must be at most 120 characters")
        if len(set(self.options)) != len(self.options):
            raise ValueError(f"field '{self.key}': options must be unique")
        if self.default is not None and self.default not in self.options:
            raise ValueError(f"field '{self.key}': default must be one of options")
        return self


StructuredFormField = Annotated[
    StructuredIntegerField | StructuredStringField | StructuredBooleanField | StructuredEnumField,
    Field(discriminator="type"),
]


class StructuredFormRequestArgs(BaseModel):
    """Strict request envelope accepted by ``request_structured_form``."""

    model_config = ConfigDict(extra="forbid")

    form_protocol: Literal["octop.structured_form.v1"] = "octop.structured_form.v1"
    form_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_.-]*$")
    schema_version: Literal[1] = 1
    title: str = Field(..., min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=500)
    context: dict[str, str | int | bool | None] = Field(
        default_factory=dict,
        max_length=MAX_STRUCTURED_FORM_CONTEXT_KEYS,
    )
    fields: list[StructuredFormField] = Field(
        ...,
        min_length=1,
        max_length=MAX_STRUCTURED_FORM_FIELDS,
    )
    timeout_seconds: int | None = Field(
        default=None,
        ge=30,
        le=3600,
        description="Seconds before the form auto-expires; None means no timeout",
    )

    @model_validator(mode="after")
    def _validate_unique_keys(self) -> Self:
        keys = [field.key for field in self.fields]
        if len(keys) != len(set(keys)):
            raise ValueError("structured form field keys must be unique")
        return self


class StructuredFormResponse(BaseModel):
    """Structured response envelope sent back as a HITL ``respond`` message."""

    model_config = ConfigDict(extra="forbid")

    form_protocol: Literal["octop.structured_form.v1"] = "octop.structured_form.v1"
    form_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_.-]*$")
    schema_version: Literal[1] = 1
    submission_id: str = Field(..., min_length=1, max_length=128)
    context: dict[str, str | int | bool | None] = Field(
        default_factory=dict,
        max_length=MAX_STRUCTURED_FORM_CONTEXT_KEYS,
    )
    values: dict[str, int | str | bool | None] = Field(
        default_factory=dict,
        max_length=MAX_STRUCTURED_FORM_FIELDS,
    )


def _check_integer(field: StructuredIntegerField, value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return "value must be an integer"
    if field.min is not None and value < field.min:
        return f"value must be greater than or equal to {field.min}"
    if field.max is not None and value > field.max:
        return f"value must be less than or equal to {field.max}"
    return None


def _check_string(field: StructuredStringField, value: Any) -> str | None:
    if not isinstance(value, str):
        return "value must be a string"
    if len(value) < field.min_length:
        return f"value must be at least {field.min_length} characters"
    if field.max_length is not None and len(value) > field.max_length:
        return f"value must be at most {field.max_length} characters"
    if field.pattern is not None and re.search(field.pattern, value) is None:
        return "value does not match the required pattern"
    return None


def _check_boolean(value: Any) -> str | None:
    if not isinstance(value, bool):
        return "value must be a boolean"
    return None


def _check_enum(field: StructuredEnumField, value: Any) -> str | None:
    if not isinstance(value, str):
        return "value must be a string"
    if value not in field.options:
        return f"value must be one of: {', '.join(field.options)}"
    return None


def _check_value(field: StructuredFormField, value: Any) -> str | None:
    if isinstance(field, StructuredIntegerField):
        return _check_integer(field, value)
    if isinstance(field, StructuredStringField):
        return _check_string(field, value)
    if isinstance(field, StructuredEnumField):
        return _check_enum(field, value)
    if isinstance(field, StructuredBooleanField):
        return _check_boolean(value)
    return f"unsupported field type: {field.type}"


def validate_structured_form_submission(
    fields: list[StructuredFormField],
    values: dict[str, Any],
) -> list[StructuredFormFieldError]:
    """Validate submitted values against the declared fields.

    Every error is scoped to a single ``field`` key so the renderer can
    highlight the offending control in place instead of rejecting the whole
    form and making the user retype it.
    """
    declared = {field.key: field for field in fields}
    errors = [
        StructuredFormFieldError(field=key, message="unknown field")
        for key in values
        if key not in declared
    ]
    for field in fields:
        if field.key not in values:
            if field.required:
                errors.append(StructuredFormFieldError(field=field.key, message="required value is missing"))
            continue
        value = values[field.key]
        if value is None:
            if field.required:
                errors.append(StructuredFormFieldError(field=field.key, message="required value must not be null"))
            elif not field.nullable:
                errors.append(StructuredFormFieldError(field=field.key, message="value must not be null"))
            continue
        message = _check_value(field, value)
        if message is not None:
            errors.append(StructuredFormFieldError(field=field.key, message=message))
    return errors


@tool(STRUCTURED_FORM_TOOL_NAME, args_schema=StructuredFormRequestArgs)
def request_structured_form(
    *,
    form_protocol: str,
    form_id: str,
    schema_version: int,
    title: str,
    fields: list[Any],
    description: str | None = None,
    context: dict[str, Any] | None = None,
    timeout_seconds: int | None = None,
) -> str:
    """Request bounded typed values from an interactive user.

    In an interactive channel, HITL middleware intercepts this call before the
    function executes. In headless runs this fallback is returned immediately.
    """
    del form_protocol, form_id, schema_version, title, fields, description, context, timeout_seconds
    return STRUCTURED_FORM_HEADLESS_FALLBACK


__all__ = [
    "MAX_STRUCTURED_FORM_CONTEXT_KEYS",
    "MAX_STRUCTURED_FORM_FIELDS",
    "MAX_STRUCTURED_FORM_OPTIONS",
    "MAX_STRUCTURED_FORM_PATTERN_LENGTH",
    "MAX_STRUCTURED_FORM_TEXT_LENGTH",
    "STRUCTURED_FORM_HEADLESS_FALLBACK",
    "STRUCTURED_FORM_PROTOCOL",
    "STRUCTURED_FORM_SCHEMA_VERSION",
    "STRUCTURED_FORM_TOOL_NAME",
    "StructuredBooleanField",
    "StructuredEnumField",
    "StructuredFieldBase",
    "StructuredFormField",
    "StructuredFormFieldError",
    "StructuredFormRequestArgs",
    "StructuredFormResponse",
    "StructuredIntegerField",
    "StructuredStringField",
    "request_structured_form",
    "validate_structured_form_submission",
]
