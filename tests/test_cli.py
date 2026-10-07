"""End-to-end tests for the command line, including CI exit codes.

The exit code is the contract CI depends on, so it is asserted explicitly:
0 means safe to ship, 1 means the gate blocked the change.
"""

import json

from evalkit.cli import main

SUITE = "examples/support_suite.json"
GOOD = "examples/fixtures_good.json"
REGRESSED = "examples/fixtures_regressed.json"
THRESHOLDS = "examples/thresholds.json"


def test_clean_run_exits_zero(capsys, tmp_path):
    code = main([
        "run", SUITE,
        "--fixtures", GOOD,
        "--thresholds", THRESHOLDS,
        "--baseline", str(tmp_path / "absent.json"),
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "GATE PASSED" in out
    assert "8/8 passed" in out


def test_regressed_run_exits_one(capsys, tmp_path):
    code = main([
        "run", SUITE,
        "--fixtures", REGRESSED,
        "--thresholds", THRESHOLDS,
        "--baseline", str(tmp_path / "absent.json"),
    ])
    out = capsys.readouterr().out
    assert code == 1
    assert "hallucination rate" in out


def test_baseline_roundtrip_detects_per_case_regression(capsys, tmp_path):
    baseline = tmp_path / "baseline.json"

    main(["run", SUITE, "--fixtures", GOOD, "--baseline", str(baseline), "--update-baseline"])
    assert baseline.exists()
    capsys.readouterr()

    code = main([
        "run", SUITE,
        "--fixtures", REGRESSED,
        "--thresholds", THRESHOLDS,
        "--baseline", str(baseline),
    ])
    out = capsys.readouterr().out
    assert code == 1
    assert "passed in the baseline and now fails" in out


def test_report_is_written_as_json(tmp_path):
    out_path = tmp_path / "nested" / "report.json"
    main([
        "run", SUITE,
        "--fixtures", GOOD,
        "--baseline", str(tmp_path / "absent.json"),
        "--out", str(out_path),
    ])
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total"] == 8
    assert len(payload["cases"]) == 8


def test_tag_filter_selects_a_subset(capsys, tmp_path):
    main([
        "run", SUITE,
        "--fixtures", GOOD,
        "--baseline", str(tmp_path / "absent.json"),
        "--tag", "injection",
    ])
    out = capsys.readouterr().out
    assert "1/1 passed" in out


def test_unknown_tag_exits_two(capsys, tmp_path):
    code = main([
        "run", SUITE,
        "--fixtures", GOOD,
        "--baseline", str(tmp_path / "absent.json"),
        "--tag", "does-not-exist",
    ])
    assert code == 2
    assert "No cases carry the tag" in capsys.readouterr().err
