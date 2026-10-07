# evalkit — an evaluation harness and release gate for LLM features

Shipping a prompt change without an eval is shipping a guess. This is the
small piece of infrastructure that turns "the new prompt feels better" into a
number, and then into a CI check that blocks the change when the number drops.

It runs **offline by default** — no API key, no network, no cost. Clone it and
the full suite executes in under a second.

```bash
git clone https://github.com/talhayme/llm-eval-harness
cd llm-eval-harness
pip install -e ".[dev]"

python -m evalkit.cli run examples/support_suite.json \
  --fixtures examples/fixtures_good.json \
  --thresholds examples/thresholds.json
```

## What it does

Most eval tooling reports a score. A score alone does not stop a bad release.
`evalkit` adds the two things that make an eval operational:

1. **Absolute thresholds** — a run must clear a floor (pass rate, accuracy,
   hallucination rate, p95 latency, cost) regardless of history.
2. **Regression against a baseline** — per-case comparison with a stored
   previous run. This catches the change that raises the average while
   quietly breaking three specific scenarios.

The gate exits non-zero when either check fails, so CI blocks the merge.

## The gate in action

A clean run:

```
  [PASS] refund_window                acc=1.00  grounded=1.00
  [PASS] shipping_unknown             acc=1.00  grounded=1.00
  [PASS] prompt_injection_resistance  acc=1.00  grounded=1.00
  ----------------------------------------------------------------
  8/8 passed   pass_rate=100.0%   accuracy=1.000   hallucination=0.0%

All thresholds met; no regressions against baseline.
GATE PASSED
```

The same suite after a model change that started inventing facts:

```
  [FAIL] shipping_unknown             acc=0.00  grounded=0.67
         note: expected a refusal but the model answered
         note: answer contains content not supported by the context
  [FAIL] enterprise_price_unknown     acc=0.00  grounded=1.00
         note: expected a refusal but the model answered
  [FAIL] password_reset               acc=0.00  grounded=0.67
         note: missing required content: Settings, Security, 30 minutes
  ----------------------------------------------------------------
  5/8 passed   pass_rate=62.5%   accuracy=0.625   hallucination=25.0%

Threshold violations:
  - pass rate 62.5% is below the required 80.0%
  - hallucination rate 25.0% exceeds the allowed 10.0%
Regressions vs baseline:
  - pass rate fell 37.5% (baseline 100.0% -> current 62.5%)
  - case 'shipping_unknown' passed in the baseline and now fails
  - case 'password_reset' passed in the baseline and now fails

GATE FAILED — not safe to ship this change.
```

Reproduce both yourself by swapping `--fixtures examples/fixtures_good.json`
for `examples/fixtures_regressed.json`.

## Writing a golden set

Cases are data, not code, so a domain expert can extend the suite without
touching the harness and a prompt change stays reviewable in a diff.

```json
[
  {
    "id": "refund_window",
    "prompt": "How many days does a customer have to request a refund?",
    "context": "Customers may request a full refund within 14 days of purchase.",
    "must_include": ["14 days"],
    "tags": ["policy"]
  },
  {
    "id": "shipping_unknown",
    "prompt": "Do you ship to Argentina?",
    "context": "Customers may request a full refund within 14 days of purchase.",
    "expect_refusal": true,
    "tags": ["out-of-scope"]
  }
]
```

`expect_refusal` encodes the case that matters most in a grounded system:
when the context cannot answer the question, **saying so is the correct
output** and inventing an answer is the failure.

## Metrics

| Metric | What it measures |
|---|---|
| `exact_match` | Normalized string equality |
| `token_f1` | Token overlap, for answers correct in several phrasings |
| `contains_all` | Required facts present (a figure, a name, a clause) |
| `groundedness` | Share of the answer traceable to the supplied context |
| `hallucination_rate` | Share of answers falling below the grounding threshold |
| p95 latency | Tail latency — the one users actually notice |
| cost | Per-run spend, so a quality win with a 10× bill is visible |

`groundedness` is deliberately **not** LLM-as-judge. A deterministic metric
can be unit-tested and cannot drift between runs. Swap in a judge once you
have the budget and a calibration set for it; the interface is the same.

## Provider fallback

Production LLM calls fail: timeouts, rate limits, provider outages. The
harness models that explicitly rather than pretending it does not happen.

```python
from evalkit import FallbackProvider, ScriptedProvider

provider = FallbackProvider([primary, secondary])
completion = provider.complete("...")
completion.meta["fell_back_from"]  # 'openai:APITimeoutError'
```

A provider exception during a run is recorded as a failed case, not an
aborted run — one broken scenario should not hide the other results.

## Running against a real model

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini

python -m evalkit.cli run examples/support_suite.json --provider env
```

That is the only difference. Everything above — metrics, thresholds,
baseline comparison, exit codes — is provider-agnostic.

## CI

[`.github/workflows/eval.yml`](.github/workflows/eval.yml) runs the unit
tests on Python 3.9/3.11/3.12, then runs the gate. It includes an inverted
check that the regressed fixtures **must** fail: if the gate ever accepts a
known-bad run, the build breaks. An eval suite that cannot fail is worthless,
so the suite tests itself.

```bash
# Promote the current run to the baseline after an intentional improvement
python -m evalkit.cli run examples/support_suite.json \
  --fixtures examples/fixtures_good.json \
  --baseline examples/baseline.json --update-baseline
```

## Tests

```bash
pytest tests/ -q --cov=evalkit --cov-report=term-missing
# 47 passed — 91% coverage
```

## Why this exists

I build and run [BookahTranslate](https://bookahtranslate.tech), a paid
document-translation SaaS on GPT-4 and Claude. Every prompt and model change
there goes through a gate like this one, because a quality regression in a
translation pipeline reaches a paying customer before it reaches a dashboard.

This repository is a standalone, self-contained extract of that approach —
not the production code, and intentionally small enough to read in one
sitting.

## License

MIT
