import json

import pytest

from evalkit.dataset import Case, load_cases
from evalkit.gate import Thresholds, evaluate_gate
from evalkit.providers import FallbackProvider, ScriptedProvider
from evalkit.runner import run_suite, score_case


def _case(**kwargs):
    base = {"id": "c1", "prompt": "q"}
    base.update(kwargs)
    return Case.from_dict(base)


class TestScoring:
    def test_required_content_present_scores_full(self):
        result = score_case(_case(must_include=["14 days"]), "Refunds within 14 days.")
        assert result["accuracy"] == 1.0

    def test_missing_required_content_is_reported(self):
        result = score_case(_case(must_include=["14 days"]), "Refunds are available.")
        assert result["accuracy"] == 0.0
        assert "missing required content" in result["notes"][0]

    def test_expected_refusal_satisfied(self):
        result = score_case(_case(expect_refusal=True), "I don't know.")
        assert result["accuracy"] == 1.0

    def test_answering_when_refusal_expected_fails(self):
        result = score_case(_case(expect_refusal=True), "Yes, we ship there for $9.99.")
        assert result["accuracy"] == 0.0
        assert "expected a refusal" in result["notes"][0]

    def test_refusing_an_answerable_case_fails(self):
        result = score_case(_case(must_include=["14 days"]), "I don't know.")
        assert result["accuracy"] == 0.0
        assert "refused" in result["notes"][0]


class TestRunSuite:
    def test_all_cases_pass_with_matching_fixtures(self):
        cases = [_case(id="a", prompt="ping", must_include=["pong"])]
        provider = ScriptedProvider({"ping": "pong"})
        report = run_suite(cases, provider)
        assert report.pass_rate == 1.0
        assert report.passed == 1

    def test_provider_error_becomes_a_failed_case_not_a_crash(self):
        class Boom:
            name = "boom"

            def complete(self, prompt, *, system=None):
                raise RuntimeError("upstream exploded")

        report = run_suite([_case()], Boom())
        assert report.pass_rate == 0.0
        assert "upstream exploded" in report.cases[0].notes[0]

    def test_p95_latency_is_reported(self):
        report = run_suite([_case(id=f"c{i}", prompt="ping") for i in range(5)],
                           ScriptedProvider({"ping": "pong"}))
        assert report.p95_latency_ms >= 0.0


class TestFallbackProvider:
    def test_falls_through_to_the_working_provider(self):
        class Broken:
            name = "broken"

            def complete(self, prompt, *, system=None):
                raise ConnectionError("network unreachable")

        provider = FallbackProvider([Broken(), ScriptedProvider({"ping": "pong"})])
        completion = provider.complete("ping")
        assert completion.text == "pong"
        assert "broken:ConnectionError" in completion.meta["fell_back_from"]

    def test_raises_only_when_every_provider_fails(self):
        class Broken:
            name = "broken"

            def complete(self, prompt, *, system=None):
                raise ConnectionError("down")

        provider = FallbackProvider([Broken(), Broken()])
        with pytest.raises(RuntimeError, match="all providers failed"):
            provider.complete("ping")

    def test_rejects_an_empty_provider_list(self):
        with pytest.raises(ValueError):
            FallbackProvider([])


class TestGate:
    def _report(self, pass_rate_cases):
        cases = [_case(id=f"c{i}", prompt="ping", must_include=["pong"])
                 for i in range(len(pass_rate_cases))]
        answers = {"ping": "pong"}
        provider = ScriptedProvider(answers)
        report = run_suite(cases, provider)
        # Force the intended pass/fail shape for threshold testing.
        for case_result, should_pass in zip(report.cases, pass_rate_cases):
            case_result.passed = should_pass
            case_result.accuracy = 1.0 if should_pass else 0.0
        return report

    def test_gate_passes_when_thresholds_met(self):
        gate = evaluate_gate(self._report([True] * 5), Thresholds())
        assert gate.passed is True

    def test_gate_fails_on_low_pass_rate(self):
        gate = evaluate_gate(self._report([True, False, False, False]), Thresholds())
        assert gate.passed is False
        assert any("pass rate" in v for v in gate.violations)

    def test_gate_detects_per_case_regression(self, tmp_path):
        baseline = {
            "summary": {"pass_rate": 1.0},
            "cases": [{"id": "c0", "passed": True}, {"id": "c1", "passed": True}],
        }
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(baseline), encoding="utf-8")

        gate = evaluate_gate(self._report([True, False]), Thresholds(min_pass_rate=0.0,
                                                                     min_mean_accuracy=0.0),
                             baseline_path=path)
        assert gate.passed is False
        assert any("c1" in r for r in gate.regressions)

    def test_corrupt_baseline_is_surfaced_not_ignored(self, tmp_path):
        path = tmp_path / "baseline.json"
        path.write_text("{not json", encoding="utf-8")
        gate = evaluate_gate(self._report([True] * 5), Thresholds(), baseline_path=path)
        assert gate.passed is False
        assert any("not valid JSON" in v for v in gate.violations)

    def test_unknown_threshold_key_is_rejected(self):
        with pytest.raises(ValueError, match="unknown threshold key"):
            Thresholds.from_dict({"min_pass_rate": 0.5, "typo_key": 1})


class TestDataset:
    def test_duplicate_ids_are_rejected(self, tmp_path):
        path = tmp_path / "suite.json"
        path.write_text(json.dumps([
            {"id": "same", "prompt": "a"},
            {"id": "same", "prompt": "b"},
        ]), encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate case id"):
            load_cases(path)

    def test_unknown_field_is_rejected(self):
        with pytest.raises(ValueError, match="unknown field"):
            Case.from_dict({"id": "x", "prompt": "y", "expceted": "typo"})

    def test_missing_prompt_is_rejected(self):
        with pytest.raises(ValueError, match="missing required field"):
            Case.from_dict({"id": "x"})

    def test_loads_the_shipped_example_suite(self):
        cases = load_cases("examples/support_suite.json")
        assert len(cases) == 8
        assert any(c.expect_refusal for c in cases)
