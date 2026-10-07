"""Command-line entry point: ``python -m evalkit.cli run ...``"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

from .dataset import load_cases
from .gate import Thresholds, evaluate_gate
from .providers import ScriptedProvider, provider_from_env
from .runner import run_suite


def _load_fixtures(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_provider(args: argparse.Namespace):
    """Offline runs read answers from a fixture file; online runs hit the API."""
    if args.provider == "scripted":
        return ScriptedProvider(_load_fixtures(args.fixtures))
    return provider_from_env()


def cmd_run(args: argparse.Namespace) -> int:
    cases = load_cases(args.suite)
    if args.tag:
        cases = [c for c in cases if args.tag in c.tags]
        if not cases:
            print(f"No cases carry the tag {args.tag!r}.", file=sys.stderr)
            return 2

    provider = _build_provider(args)
    thresholds = Thresholds.from_dict(
        json.loads(args.thresholds.read_text(encoding="utf-8")) if args.thresholds else {}
    )

    report = run_suite(cases, provider, pass_threshold=args.pass_threshold)
    summary = report.summary()

    print(f"\n  Suite: {args.suite.name}   Provider: {report.provider}   Model: {report.model}")
    print(f"  {'-' * 64}")
    for case in report.cases:
        mark = "PASS" if case.passed else "FAIL"
        print(f"  [{mark}] {case.id:<28} acc={case.accuracy:.2f}  grounded={case.groundedness:.2f}")
        for note in case.notes:
            print(f"         note: {note}")
    print(f"  {'-' * 64}")
    print(
        f"  {summary['passed']}/{summary['total']} passed"
        f"   pass_rate={summary['pass_rate']:.1%}"
        f"   accuracy={summary['mean_accuracy']:.3f}"
        f"   hallucination={summary['hallucination_rate']:.1%}"
    )
    print(f"  p95 latency={summary['p95_latency_ms']:.1f}ms   cost=${summary['total_cost_usd']:.6f}\n")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Report written to {args.out}")

    if args.update_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Baseline updated at {args.baseline}")
        return 0

    gate = evaluate_gate(report, thresholds, args.baseline)
    print(f"\n{gate.render()}\n")

    if not gate.passed:
        print("GATE FAILED — not safe to ship this change.\n", file=sys.stderr)
        return 1
    print("GATE PASSED\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalkit",
        description="Run an LLM golden set and gate releases on the result.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a golden set and apply the gate")
    run.add_argument("suite", type=Path, help="path to the golden set (.json/.yaml)")
    run.add_argument(
        "--provider", choices=("scripted", "env"), default="scripted",
        help="'scripted' runs offline from fixtures (default); 'env' reads LLM_PROVIDER",
    )
    run.add_argument("--fixtures", type=Path, help="JSON map of prompt -> canned answer")
    run.add_argument("--thresholds", type=Path, help="JSON file of gate thresholds")
    run.add_argument("--baseline", type=Path, default=Path("baseline.json"),
                     help="baseline report to compare against")
    run.add_argument("--out", type=Path, help="write the full JSON report here")
    run.add_argument("--tag", help="only run cases carrying this tag")
    run.add_argument("--pass-threshold", type=float, default=0.8,
                     help="minimum accuracy for a single case to count as passed")
    run.add_argument("--update-baseline", action="store_true",
                     help="overwrite the baseline with this run instead of gating")
    run.set_defaults(func=cmd_run)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
