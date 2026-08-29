"""Provider-neutral model access.

Every model call in the system goes through here, which is what keeps prompts and
completions out of LangChain's automatic tracing: the telemetry channel only ever sees
what this module chooses to emit.

Temperature is pinned to zero at the gateway rather than left to callers, so a decision
is reproducible by construction.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .contracts import Contract, ErrorCode, PromptArtifact

T = TypeVar("T", bound=BaseModel)


class GatewayError(RuntimeError):
    def __init__(
        self,
        message: str,
        code: ErrorCode,
        usage: ModelUsage | None = None,
        *,
        truncated: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        #: The response hit the output ceiling. Distinct from every other failure
        #: because it says something the scheduler can act on: the batch was too big.
        self.truncated = truncated
        #: Present when the provider was reached and billed but the response was
        #: unusable. Without it a failed parse vanishes from the bill while still
        #: costing money.
        self.usage = usage


class ModelProfile(Contract):
    profile_id: str
    #: LiteLLM form, e.g. `openrouter/openai/gpt-5-mini`.  A moving alias would make a
    #: run irreproducible, so the profile must name a fixed model.
    model: str
    api_base: str | None = None
    #: Generous by default. A reasoning model spends part of its budget before emitting
    #: anything, and a domain that reports per-item values for a large set needs room —
    #: truncation there surfaced only as "malformed JSON".
    max_output_tokens: int = 16384
    timeout_seconds: float = 120.0


class ModelUsage(Contract):
    requested_model: str
    resolved_model: str | None = None
    provider: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Gateway(Protocol):
    def complete(
        self,
        profile: ModelProfile,
        messages: Sequence[dict[str, str]],
        response_model: type[T],
    ) -> tuple[T, ModelUsage]: ...


def profile_from_environment(profile_id: str = "default") -> ModelProfile:
    model = os.environ.get("SHAKESPEARE_MODEL")
    if not model:
        raise GatewayError("SHAKESPEARE_MODEL is required", ErrorCode.MODEL_PERMANENT)
    if model.endswith(("latest", "-auto")) or "/auto" in model:
        raise GatewayError(
            f"{model!r} is a moving alias; a run must pin a fixed model to stay reproducible",
            ErrorCode.MODEL_PERMANENT,
        )
    _refuse_a_model_whose_tokens_cannot_be_counted(model)
    return ModelProfile(
        profile_id=profile_id,
        model=model,
        api_base=os.environ.get("SHAKESPEARE_API_BASE") or None,
    )


def _refuse_a_model_whose_tokens_cannot_be_counted(model: str) -> None:
    """Refuse a model this interpreter has no tokenizer for.

    `tokenizers` has no free-threaded build, so `compat/tokenizers` stands in for it and
    refuses rather than approximate (ADR 0006). LiteLLM catches that refusal, logs it at
    debug, and counts with tiktoken anyway — so a Claude, Cohere or Llama model would be
    counted by the wrong tokenizer, and the wrong number would reach the bill and the
    batch sizer with nothing anywhere saying it was wrong. A run that cannot count its own
    tokens should not start.

    Which models those are is LiteLLM's question, not ours, so this asks LiteLLM's own
    selector instead of keeping a copy of its list to drift from.
    """
    try:
        from tokenizers import TokenizerUnavailable
    except ImportError:
        # The real tokenizers is installed, so every model LiteLLM supports can be
        # counted exactly and there is nothing to refuse. Returning here also keeps this
        # check off the network: with the real library, asking would download one.
        return

    from litellm.utils import _return_huggingface_tokenizer

    try:
        _return_huggingface_tokenizer(model)
    except TokenizerUnavailable as exc:
        raise GatewayError(
            f"{model!r} is counted by a HuggingFace tokenizer, and there is no "
            f"free-threaded build of `tokenizers` for this interpreter to count it with. "
            f"LiteLLM would silently fall back to tiktoken and bill the run against the "
            f"wrong tokenizer. Pin an OpenAI-shaped model — `openrouter/openai/...` — or "
            f"see docs/adr/0006-free-threaded-python-only.md to build the real one.",
            ErrorCode.MODEL_PERMANENT,
        ) from exc


class LiteLLMGateway:
    """Provider-neutral routing via LiteLLM."""

    def complete(
        self,
        profile: ModelProfile,
        messages: Sequence[dict[str, str]],
        response_model: type[T],
    ) -> tuple[T, ModelUsage]:
        import litellm

        try:
            response = litellm.completion(
                model=profile.model,
                messages=list(messages),
                temperature=0,
                max_tokens=profile.max_output_tokens,
                timeout=profile.timeout_seconds,
                api_base=profile.api_base,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            transient = type(exc).__name__ in {
                "RateLimitError",
                "Timeout",
                "APIConnectionError",
                "ServiceUnavailableError",
                "InternalServerError",
            }
            raise GatewayError(
                str(exc),
                ErrorCode.MODEL_TRANSIENT if transient else ErrorCode.MODEL_PERMANENT,
            ) from exc

        choice = response.choices[0]
        content = choice.message.content or ""
        if getattr(choice, "finish_reason", None) == "length":
            raise GatewayError(
                f"the response was cut off at the {profile.max_output_tokens}-token limit, "
                f"so it is incomplete rather than malformed. Ask for less in one response, "
                f"or raise max_output_tokens.",
                ErrorCode.MODEL_PERMANENT,
                usage=_usage_of(response, profile),
                truncated=True,
            )
        hidden = getattr(response, "_hidden_params", {}) or {}
        usage_data = getattr(response, "usage", None)
        usage = ModelUsage(
            requested_model=profile.model,
            resolved_model=getattr(response, "model", None),
            provider=hidden.get("custom_llm_provider") or hidden.get("llm_provider"),
            prompt_tokens=getattr(usage_data, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage_data, "completion_tokens", 0) or 0,
            cost_usd=hidden.get("response_cost") or 0.0,
        )
        try:
            return _parse(content, response_model), usage
        except GatewayError as exc:
            raise GatewayError(str(exc), exc.code, usage=usage) from exc


def _usage_of(response: Any, profile: ModelProfile) -> ModelUsage:
    hidden = getattr(response, "_hidden_params", {}) or {}
    usage_data = getattr(response, "usage", None)
    return ModelUsage(
        requested_model=profile.model,
        resolved_model=getattr(response, "model", None),
        provider=hidden.get("custom_llm_provider") or hidden.get("llm_provider"),
        prompt_tokens=getattr(usage_data, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage_data, "completion_tokens", 0) or 0,
        cost_usd=hidden.get("response_cost") or 0.0,
    )


def _parse[M: BaseModel](content: str, response_model: type[M]) -> M:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GatewayError(
            f"model did not return JSON: {exc}", ErrorCode.MODEL_PERMANENT
        ) from exc
    try:
        return response_model.model_validate(payload)
    except (ValidationError, TypeError, ValueError, KeyError) as exc:
        # A schema violation is permanent for this prompt: retrying the same call would
        # produce the same shape.  The planner decides what to do at the stage boundary.
        raise GatewayError(
            f"model response does not satisfy {response_model.__name__}: {exc}",
            ErrorCode.MODEL_PERMANENT,
        ) from exc


@dataclass
class FakeGateway:
    """Scripted gateway so the whole suite runs offline.

    Responses are keyed by response-model name and consumed in order, which lets a test
    script a planner that skips a domain or forces a rerun.
    """

    responses: dict[str, list[Any]] = field(default_factory=dict)
    calls: list[tuple[str, list[dict[str, str]]]] = field(default_factory=list)

    def queue(self, response_model: type[BaseModel], *values: Any) -> FakeGateway:
        self.responses.setdefault(response_model.__name__, []).extend(values)
        return self

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def complete(
        self,
        profile: ModelProfile,
        messages: Sequence[dict[str, str]],
        response_model: type[T],
    ) -> tuple[T, ModelUsage]:
        self.calls.append((response_model.__name__, list(messages)))
        queued = self.responses.get(response_model.__name__)
        if not queued:
            raise GatewayError(
                f"FakeGateway has no queued {response_model.__name__} response",
                ErrorCode.MODEL_PERMANENT,
            )
        value = queued.pop(0)
        # Validate exactly as LiteLLMGateway does, so a fake response that violates its
        # contract fails the same way a real one would.  A fake that is more forgiving
        # than production would hide precisely the bugs it exists to catch.
        parsed = (
            value
            if isinstance(value, response_model)
            else _parse(json.dumps(value, default=str), response_model)
        )
        return parsed, ModelUsage(
            requested_model=profile.model,
            resolved_model=profile.model,
            provider="fake",
            prompt_tokens=1,
            completion_tokens=1,
        )


def render_prompt(artifact: PromptArtifact, **variables: Any) -> list[dict[str, str]]:
    """Build messages from a pinned prompt artifact.

    Demonstrations come from the artifact rather than from the call site, so promoting a
    compiled prompt changes behaviour in exactly one versioned place.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": artifact.instructions}]
    for demonstration in artifact.demonstrations:
        messages.append({"role": "user", "content": json.dumps(demonstration.get("input", {}))})
        messages.append(
            {"role": "assistant", "content": json.dumps(demonstration.get("output", {}))}
        )
    messages.append({"role": "user", "content": json.dumps(variables, default=str)})
    return messages
