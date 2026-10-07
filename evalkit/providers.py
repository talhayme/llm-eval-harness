"""Provider abstraction with deterministic offline fallback.

The point of this module: evaluation code should never care *which* model
answered, and the test suite must run with no API key and no network.
Set ``LLM_PROVIDER=openai`` (plus ``OPENAI_API_KEY``) to hit a real model;
leave it unset and the harness uses a scripted provider instead.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol


@dataclass
class Completion:
    """One model response plus the metadata an eval needs to judge it."""

    text: str
    model: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Cost is computed by the provider because pricing is provider-specific.
    cost_usd: float = 0.0
    meta: Dict[str, str] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Provider(Protocol):
    name: str

    def complete(self, prompt: str, *, system: Optional[str] = None) -> Completion:
        ...


def _approx_tokens(text: str) -> int:
    """Rough token count. Good enough for cost *trends*, not for billing."""
    return max(1, len(text) // 4)


class ScriptedProvider:
    """Deterministic stand-in for a real model.

    Answers come from a fixture table keyed by an exact prompt match, so a
    golden-set run is byte-for-byte reproducible on any machine. Unknown
    prompts fall back to a hash-derived canned answer, which keeps the
    harness exercising its full code path instead of raising.
    """

    name = "scripted"

    def __init__(self, responses: Optional[Dict[str, str]] = None, *, model: str = "scripted-v1"):
        self.responses = responses or {}
        self.model = model

    def complete(self, prompt: str, *, system: Optional[str] = None) -> Completion:
        started = time.perf_counter()
        if prompt in self.responses:
            text = self.responses[prompt]
        else:
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
            text = f"[no fixture for this prompt: {digest}]"
        latency_ms = (time.perf_counter() - started) * 1000
        return Completion(
            text=text,
            model=self.model,
            latency_ms=latency_ms,
            prompt_tokens=_approx_tokens(prompt + (system or "")),
            completion_tokens=_approx_tokens(text),
            cost_usd=0.0,
            meta={"offline": "true"},
        )


class OpenAIProvider:
    """Thin real-API adapter.

    Deliberately minimal: this repo is about the evaluation layer, not about
    wrapping an SDK. It exists to show the seam where a real provider plugs in.
    """

    name = "openai"

    # USD per 1M tokens.
    PRICING = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
    }

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Unset LLM_PROVIDER to run the "
                "harness offline with the scripted provider."
            )

    def complete(self, prompt: str, *, system: Optional[str] = None) -> Completion:
        from openai import OpenAI  # imported lazily so offline runs need no dependency

        client = OpenAI(api_key=self.api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        started = time.perf_counter()
        resp = client.chat.completions.create(model=self.model, messages=messages)
        latency_ms = (time.perf_counter() - started) * 1000

        usage = resp.usage
        p_in, p_out = self.PRICING.get(self.model, (0.0, 0.0))
        cost = (usage.prompt_tokens * p_in + usage.completion_tokens * p_out) / 1_000_000

        return Completion(
            text=resp.choices[0].message.content or "",
            model=self.model,
            latency_ms=latency_ms,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=cost,
        )


class FallbackProvider:
    """Try providers in order; return the first that answers.

    Mirrors the production pattern from BookahTranslate: when the primary
    provider times out or rate-limits, the request should degrade to a
    secondary rather than fail the user. Errors are recorded so a run can
    report *that* a fallback happened, not just the final answer.
    """

    name = "fallback"

    def __init__(self, providers: List[Provider]):
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self.providers = providers
        self.failures: List[str] = []

    def complete(self, prompt: str, *, system: Optional[str] = None) -> Completion:
        last_error: Optional[Exception] = None
        for provider in self.providers:
            try:
                completion = provider.complete(prompt, system=system)
                if self.failures:
                    completion.meta["fell_back_from"] = ",".join(self.failures)
                return completion
            except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure should fall through
                self.failures.append(f"{provider.name}:{type(exc).__name__}")
                last_error = exc
        raise RuntimeError(f"all providers failed: {self.failures}") from last_error


def provider_from_env() -> Provider:
    """Pick a provider from ``LLM_PROVIDER``, defaulting to offline."""
    choice = os.environ.get("LLM_PROVIDER", "scripted").lower()
    if choice == "openai":
        return OpenAIProvider(model=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    return ScriptedProvider()
