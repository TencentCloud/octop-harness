"""Built-in tool for typed, declarative Human-in-the-Loop forms.

The protocol supports a bounded list of integer fields. The host renders the
request and sends a structured response through the normal HITL ``respond``
decision; this function is only the non-blocking headless fallback.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

STRUCTURED_FORM_TOOL_NAME = "request_structured_form"
STRUCTURED_FORM_PROTOCOL = "octop.structured_form.v1"
STRUCTURED_FORM_SCHEMA_VERSION = 1
MAX_STRUCTURED_FORM_FIELDS = 64
MAX_STRUCTURED_FORM_CONTEXT_KEYS = 16

STRUCTURED_FORM_HEADLESS_FALLBACK = (
    "No interactive user is available in this run. Do not call "
    "request_structured_form again. State that the structured form could not "
    "be collected and continue without inventing field values."
)


class StructuredIntegerField(BaseModel):
    """Allowlisted integer field definition for the MVP protocol."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    label: str = Field(..., min_length=1, max_length=120)
    type: Literal["integer"] = "integer"
    required: bool = False
    nullable: bool = True
    min: int | None = Field(default=0, ge=0)
    max: int | None = Field(default=None, ge=0)
    step: Literal[1] = 1

    @model_validator(mode="after")
    def _validate_range(self) -> StructuredIntegerField:
        if self.max is not None and self.min is not None and self.max < self.min:
            raise ValueError("field max must be greater than or equal to min")
        if self.required and self.nullable:
            raise ValueError("required integer fields must set nullable=false")
        return self


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
    fields: list[StructuredIntegerField] = Field(
        ...,
        min_length=1,
        max_length=MAX_STRUCTURED_FORM_FIELDS,
    )

    @model_validator(mode="after")
    def _validate_unique_keys(self) -> StructuredFormRequestArgs:
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
    values: dict[str, int | None] = Field(default_factory=dict, max_length=MAX_STRUCTURED_FORM_FIELDS)


@tool(STRUCTURED_FORM_TOOL_NAME, args_schema=StructuredFormRequestArgs)
def request_structured_form(
    form_protocol: str,
    form_id: str,
    schema_version: int,
    title: str,
    fields: list[Any],
    description: str | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    """Request bounded typed integer values from an interactive user.

    In an interactive channel, HITL middleware intercepts this call before the
    function executes. In headless runs this fallback is returned immediately.
    """
    del form_protocol, form_id, schema_version, title, fields, description, context
    return STRUCTURED_FORM_HEADLESS_FALLBACK


__all__ = [
    "MAX_STRUCTURED_FORM_CONTEXT_KEYS",
    "MAX_STRUCTURED_FORM_FIELDS",
    "STRUCTURED_FORM_HEADLESS_FALLBACK",
    "STRUCTURED_FORM_PROTOCOL",
    "STRUCTURED_FORM_SCHEMA_VERSION",
    "STRUCTURED_FORM_TOOL_NAME",
    "StructuredFormRequestArgs",
    "StructuredFormResponse",
    "StructuredIntegerField",
    "request_structured_form",
]
