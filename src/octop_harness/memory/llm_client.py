"""``HarnessAgentLLMClient`` — adapt the agent's models to octop-memory.

``octop-memory``'s extractor / promotion / page-regen paths talk to the
host model through the :class:`octop_memory.ports.llm.LLMClient` protocol. This
module wraps the agent's existing :class:`ChatModelFactory` so we can reuse
the same providers and credentials the rest of the agent already uses.

Model selection
---------------
Extract uses **one** model, not a light/heavy pair. Resolution order:

1. optional aux override (``aux_model``, or the legacy light/heavy aliases)
2. the live chat model (:meth:`set_current_model`)
3. the agent's ``default_model``

``tier`` only changes the timeout. If a ref fails (expired subscription,
400, missing model), ``call_llm`` tries the next one in that list.

Failure discipline
------------------
Any transport / model error is wrapped in :class:`LLMClientError` so the
plugin's extractor / promotion worker can degrade gracefully (write the
``failure_reason`` and move on) — never bubble into the user reply path.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Literal

from octop_memory.ports.llm import LLMClientError, LLMTier

from octop_harness.llm.invoke_helpers import bind_call_options, build_text_messages, stringify_content

if TYPE_CHECKING:
    from octop_harness.llm.factory import ChatModelFactory

logger = logging.getLogger(__name__)


class HarnessAgentLLMClient:
    """Adapter from :class:`ChatModelFactory` to :class:`LLMClient`.

    Args:
        factory: The agent's model factory; used to build / cache
            ``BaseChatModel`` instances by ``"<provider>/<model>"`` ref.
        aux_model: Optional extract override. When unset, extract follows
            the live chat model and then ``default_model``.
        light_model / heavy_model: Legacy aliases for ``aux_model``. The
            first non-empty value wins; both tiers share it.
        default_model: Last-resort model ref when no aux and no live chat
            model have been recorded yet.
    """

    # Default timeouts in seconds. Light covers per-session extraction and alias
    # disambiguation; heavy covers cross-session consolidation. Hard caps prevent
    # upstream gateway stalls or slow inference from blocking extract threads.
    DEFAULT_LIGHT_TIMEOUT_S: float = 120.0
    DEFAULT_HEAVY_TIMEOUT_S: float = 300.0

    def __init__(
        self,
        factory: ChatModelFactory,
        *,
        aux_model: str | None = None,
        light_model: str | None = None,
        heavy_model: str | None = None,
        default_model: str | None = None,
        light_timeout_s: float | None = None,
        heavy_timeout_s: float | None = None,
    ) -> None:
        configured = _first_ref(aux_model, light_model, heavy_model)
        if configured is None and default_model is None:
            raise ValueError(
                "HarnessAgentLLMClient requires aux_model or default_model",
            )
        self._factory = factory
        self._aux_model = configured
        self._default_model = default_model
        self._light_timeout_s = light_timeout_s if light_timeout_s is not None else self.DEFAULT_LIGHT_TIMEOUT_S
        self._heavy_timeout_s = heavy_timeout_s if heavy_timeout_s is not None else self.DEFAULT_HEAVY_TIMEOUT_S
        self._current_lock = threading.Lock()
        self._current_model: str | None = None

    def set_current_model(self, ref: str | None) -> None:
        """Record the model the user is actually chatting with.

        When no aux override is set, extract uses this ref. When aux is
        set and fails, extract retries here.
        """
        value = ref.strip() if isinstance(ref, str) and ref.strip() else None
        with self._current_lock:
            self._current_model = value

    # ------------------------------------------------------------------
    # LLMClient protocol
    # ------------------------------------------------------------------

    def call_llm(
        self,
        prompt: str,
        *,
        tier: LLMTier = "light",
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: Literal["text", "json"] = "text",
    ) -> str:
        refs = self._candidate_refs()
        if not refs:
            raise LLMClientError(f"no model configured for tier={tier!r}")

        last_exc: BaseException | None = None
        for index, ref in enumerate(refs):
            try:
                return self._invoke_ref(
                    ref,
                    prompt,
                    tier=tier,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                )
            except Exception as exc:  # provider error types vary; try the next ref; pylint: disable=broad-except
                last_exc = exc
                next_ref = refs[index + 1] if index + 1 < len(refs) else None
                if next_ref is None:
                    break
                logger.info(
                    "HarnessAgentLLMClient: %s failed for tier=%s; falling back to %s: %r (%s)",
                    ref,
                    tier,
                    next_ref,
                    exc,
                    type(exc).__name__,
                )

        assert last_exc is not None
        failed_ref = refs[-1]
        detail = str(last_exc).strip() or type(last_exc).__name__
        raise LLMClientError(
            f"chat model {failed_ref!r} failed for tier={tier!r}: {detail}",
        ) from last_exc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _candidate_refs(self) -> list[str]:
        """Return distinct refs: aux override, then live chat, then default."""
        with self._current_lock:
            current = self._current_model
        ordered: list[str] = []
        seen: set[str] = set()
        for ref in (self._aux_model, current, self._default_model):
            stripped = _first_ref(ref)
            if stripped is None or stripped in seen:
                continue
            seen.add(stripped)
            ordered.append(stripped)
        return ordered

    def _invoke_ref(
        self,
        ref: str,
        prompt: str,
        *,
        tier: LLMTier,
        system: str | None,
        max_tokens: int | None,
        temperature: float | None,
        response_format: Literal["text", "json"],
    ) -> str:
        try:
            model = self._factory.get_chat_model(ref)
        except Exception as exc:
            detail = str(exc).strip() or type(exc).__name__
            raise LLMClientError(
                f"failed to build chat model {ref!r} for tier={tier!r}: {detail}",
            ) from exc

        timeout_s = self._light_timeout_s if tier == "light" else self._heavy_timeout_s
        bound_model = bind_call_options(
            model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            timeout_s=timeout_s,
        )
        messages = build_text_messages(prompt, system=system)

        try:
            response = bound_model.invoke(messages)
        except Exception as exc:
            if temperature is not None and _is_temperature_rejection(exc):
                # Some models accept exactly one temperature value (e.g. Kimi
                # Coding requires 1.0) and reject every other with a
                # deterministic 400. Retry once with the provider default so
                # the fallback chain is not burned on a parameter mismatch.
                logger.info(
                    "HarnessAgentLLMClient: %s rejected temperature=%s (tier=%s); retrying with provider default",
                    ref,
                    temperature,
                    tier,
                )
                try:
                    default_bound = bind_call_options(
                        model,
                        max_tokens=max_tokens,
                        temperature=None,
                        response_format=response_format,
                        timeout_s=timeout_s,
                    )
                    return stringify_content(default_bound.invoke(messages).content)
                except Exception as retry_exc:  # pylint: disable=broad-except
                    exc = retry_exc
            # Provider 400s (expired subscription, bad request) are not
            # always RuntimeError/OSError subclasses. Treat every model
            # failure as degrade-able so extract never kills the user turn.
            logger.info(
                "HarnessAgentLLMClient.call_llm failed (model=%s, tier=%s): %r (%s)",
                ref,
                tier,
                exc,
                type(exc).__name__,
            )
            detail = str(exc).strip() or type(exc).__name__
            raise LLMClientError(
                f"chat model {ref!r} failed for tier={tier!r}: {detail}",
            ) from exc

        return stringify_content(response.content)


def _first_ref(*refs: str | None) -> str | None:
    for ref in refs:
        if isinstance(ref, str) and ref.strip():
            return ref.strip()
    return None


def _is_temperature_rejection(exc: BaseException) -> bool:
    """Whether a provider error rejects the requested temperature value.

    Providers word this 400 differently ("invalid temperature",
    "temperature only support 1"...); a broad substring match keeps us from
    maintaining a per-provider catalogue. A false positive only costs one
    extra call that fails the same way before the normal fallback.
    """
    return "temperature" in str(exc).lower()


__all__ = ["HarnessAgentLLMClient"]
