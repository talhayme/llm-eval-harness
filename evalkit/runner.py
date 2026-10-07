"""Run a golden set against a provider and aggregate the results."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .dataset import Case
from .metrics import (
    contains_all,
    exact_match,
    groundedness,
    hallucination_rate,
    is_refusal,
    token_f1,
)
from .providers import Provider


@dataclass
class CaseResult:
    """Scores for a single case. Serialized into the run report."""

    id: str
    passed: bool
    accuracy: float
    groundedness: float
    latency_ms: float
    cost_usd: float
    output: str
    tags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class RunReport:
    """Aggregate view of one evaluation run."""

    provider: str
    model: str
    cases: List[CaseResult]

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def mean_accuracy(self) -> float:
        return statistics.fmean(c.accuracy for c in self.cases) if self.cases else 0.0

    @property
    def hallucination_rate(self) -> float:
        return hallucination_rate([c.groundedness for c in self.cases])

    @property
    def p95_latency_ms(self) -> float:
        """95th-percentile latency. Means hide the tail that users notice."""
        if not self.cases:
            return 0.0
        ordered = sorted(c.latency_ms for c in self.cases)
        # Nearest-rank: smallest value at or above 95% of the distribution.
        index = max(0, min(len(ordered) - 1, int(round(0.95 * len(ordered))) - 1))
        return ordered[index]

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.cases)

    def summary(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "mean_accuracy": round(self.mean_accuracy, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "total_cost_usd": round(self.total_cost_usd, 6),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary(), "cases": [asdict(c) for c in self.cases]}


def score_case(case: Case, output: str) -> Dict[str, Any]:
    """Score one output. Separated from I/O so it is directly testable."""
    notes: List[str] = []
    refused = is_refusal(output)

    if case.expect_refusal:
        # The whole point of the case is that the model must NOT answer.
        accuracy = 1.0 if refused else 0.0
        if not refused:
            notes.append("expected a refusal but the model answered")
    elif refused:
        accuracy = 0.0
        notes.append("model refused on an answerable case")
    else:
        scores: List[float] = []
        if case.expected:
            # Credit either a clean exact match or strong token overlap.
            scores.append(max(exact_match(output, case.expected), token_f1(output, case.expected)))
        if case.must_include:
            hit = contains_all(output, case.must_include)
            scores.append(hit)
            if hit < 1.0:
                missing = [s for s in case.must_include if s.casefold() not in output.casefold()]
                notes.append(f"missing required content: {', '.join(missing)}")
        accuracy = statistics.fmean(scores) if scores else 1.0

    ground = groundedness(output, case.context) if case.context else 1.0
    if case.context and ground < 0.8 and not refused:
        notes.append("answer contains content not supported by the context")

    return {"accuracy": accuracy, "groundedness": ground, "notes": notes}


def run_suite(
    cases: Sequence[Case],
    provider: Provider,
    *,
    pass_threshold: float = 0.8,
    model_label: Optional[str] = None,
) -> RunReport:
    """Execute every case and collect scores.

    A provider exception is caught and recorded as a failed case: one broken
    scenario should not abort the run and hide the other results.
    """
    results: List[CaseResult] = []
    model = model_label or getattr(provider, "model", provider.name)

    for case in cases:
        try:
            completion = provider.complete(case.prompt, system=case.system)
            scored = score_case(case, completion.text)
            accuracy = scored["accuracy"]
            results.append(
                CaseResult(
                    id=case.id,
                    passed=accuracy >= pass_threshold and scored["groundedness"] >= 0.8,
                    accuracy=accuracy,
                    groundedness=scored["groundedness"],
                    latency_ms=completion.latency_ms,
                    cost_usd=completion.cost_usd,
                    output=completion.text,
                    tags=case.tags,
                    notes=scored["notes"],
                )
            )
        except Exception as exc:  # noqa: BLE001 - a provider error is a test result, not a crash
            results.append(
                CaseResult(
                    id=case.id,
                    passed=False,
                    accuracy=0.0,
                    groundedness=0.0,
                    latency_ms=0.0,
                    cost_usd=0.0,
                    output="",
                    tags=case.tags,
                    notes=[f"provider raised {type(exc).__name__}: {exc}"],
                )
            )

    return RunReport(provider=provider.name, model=model, cases=results)
