"""Release gate: compare a run against thresholds and a stored baseline.

This is the piece that turns an eval suite into a quality control. A score
on its own is a number in a log; a gate is what stops a prompt change from
reaching users when it makes things worse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .runner import RunReport


@dataclass
class Thresholds:
    """Absolute floors a run must clear, independent of history."""

    min_pass_rate: float = 0.80
    min_mean_accuracy: float = 0.75
    max_hallucination_rate: float = 0.10
    max_p95_latency_ms: Optional[float] = None
    max_total_cost_usd: Optional[float] = None
    # How far a metric may slip against the baseline before it counts as a
    # regression. Small drifts are noise; a 5-point drop is a signal.
    max_pass_rate_drop: float = 0.05

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Thresholds":
        known = {f for f in cls.__dataclass_fields__}  # noqa: F821 - dataclass attr
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown threshold key(s): {', '.join(sorted(unknown))}")
        return cls(**raw)


@dataclass
class GateResult:
    """Verdict plus the reasoning, so CI output explains itself."""

    passed: bool
    violations: List[str] = field(default_factory=list)
    regressions: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)

    def render(self) -> str:
        lines: List[str] = []
        if self.violations:
            lines.append("Threshold violations:")
            lines += [f"  - {v}" for v in self.violations]
        if self.regressions:
            lines.append("Regressions vs baseline:")
            lines += [f"  - {r}" for r in self.regressions]
        if self.improvements:
            lines.append("Improvements vs baseline:")
            lines += [f"  + {i}" for i in self.improvements]
        if not lines:
            lines.append("All thresholds met; no regressions against baseline.")
        return "\n".join(lines)


def check_thresholds(report: RunReport, thresholds: Thresholds) -> List[str]:
    """Absolute checks. Returns a list of human-readable violations."""
    violations: List[str] = []

    if report.pass_rate < thresholds.min_pass_rate:
        violations.append(
            f"pass rate {report.pass_rate:.1%} is below the required {thresholds.min_pass_rate:.1%}"
        )
    if report.mean_accuracy < thresholds.min_mean_accuracy:
        violations.append(
            f"mean accuracy {report.mean_accuracy:.3f} is below the required "
            f"{thresholds.min_mean_accuracy:.3f}"
        )
    if report.hallucination_rate > thresholds.max_hallucination_rate:
        violations.append(
            f"hallucination rate {report.hallucination_rate:.1%} exceeds the allowed "
            f"{thresholds.max_hallucination_rate:.1%}"
        )
    if (
        thresholds.max_p95_latency_ms is not None
        and report.p95_latency_ms > thresholds.max_p95_latency_ms
    ):
        violations.append(
            f"p95 latency {report.p95_latency_ms:.0f}ms exceeds the allowed "
            f"{thresholds.max_p95_latency_ms:.0f}ms"
        )
    if (
        thresholds.max_total_cost_usd is not None
        and report.total_cost_usd > thresholds.max_total_cost_usd
    ):
        violations.append(
            f"run cost ${report.total_cost_usd:.4f} exceeds the allowed "
            f"${thresholds.max_total_cost_usd:.4f}"
        )

    return violations


def compare_to_baseline(report: RunReport, baseline: Dict[str, Any], thresholds: Thresholds):
    """Per-case and aggregate comparison against a previous run.

    Per-case comparison is what catches the change that lifts the average
    while quietly breaking three specific scenarios.
    """
    regressions: List[str] = []
    improvements: List[str] = []

    base_summary = baseline.get("summary", {})
    base_pass_rate = base_summary.get("pass_rate")
    if isinstance(base_pass_rate, (int, float)):
        drop = base_pass_rate - report.pass_rate
        if drop > thresholds.max_pass_rate_drop:
            regressions.append(
                f"pass rate fell {drop:.1%} (baseline {base_pass_rate:.1%} -> "
                f"current {report.pass_rate:.1%})"
            )
        elif report.pass_rate > base_pass_rate:
            improvements.append(
                f"pass rate rose from {base_pass_rate:.1%} to {report.pass_rate:.1%}"
            )

    base_cases = {c["id"]: c for c in baseline.get("cases", [])}
    for case in report.cases:
        previous = base_cases.get(case.id)
        if previous is None:
            continue
        if previous.get("passed") and not case.passed:
            note = f" ({case.notes[0]})" if case.notes else ""
            regressions.append(f"case {case.id!r} passed in the baseline and now fails{note}")
        elif not previous.get("passed") and case.passed:
            improvements.append(f"case {case.id!r} now passes")

    return regressions, improvements


def evaluate_gate(
    report: RunReport,
    thresholds: Thresholds,
    baseline_path: Optional[Path] = None,
) -> GateResult:
    """Full gate: absolute thresholds plus regression against a baseline."""
    violations = check_thresholds(report, thresholds)
    regressions: List[str] = []
    improvements: List[str] = []

    if baseline_path is not None:
        path = Path(baseline_path)
        if path.exists():
            try:
                baseline = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                # A corrupt baseline must not silently disable regression checks.
                violations.append(f"baseline {path.name} is not valid JSON: {exc}")
                baseline = None
            if baseline:
                regressions, improvements = compare_to_baseline(report, baseline, thresholds)

    return GateResult(
        passed=not violations and not regressions,
        violations=violations,
        regressions=regressions,
        improvements=improvements,
    )
