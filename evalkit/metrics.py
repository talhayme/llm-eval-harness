"""Scoring functions for model output.

Each metric returns a float in [0.0, 1.0] where higher is better, so the
gate in :mod:`evalkit.gate` can treat them uniformly. Keeping them pure and
dependency-free means they are themselves unit-testable, which matters: a
silently broken metric is worse than no metric.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence


def _normalize(text: str) -> str:
    """Casefold, strip accents and collapse whitespace/punctuation."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> Sequence[str]:
    return _normalize(text).split()


def exact_match(output: str, expected: str) -> float:
    """1.0 only if the normalized strings are identical."""
    return 1.0 if _normalize(output) == _normalize(expected) else 0.0


def contains_all(output: str, required: Iterable[str]) -> float:
    """Fraction of required substrings present in the output.

    Useful when an answer is correct in several phrasings but must mention
    specific facts (a figure, a name, a clause number).
    """
    required = list(required)
    if not required:
        return 1.0
    norm = _normalize(output)
    hits = sum(1 for item in required if _normalize(item) in norm)
    return hits / len(required)


def token_f1(output: str, expected: str) -> float:
    """Token-overlap F1 — a tolerant alternative to exact match."""
    out_tokens = _tokens(output)
    exp_tokens = _tokens(expected)
    if not out_tokens or not exp_tokens:
        return 1.0 if out_tokens == exp_tokens else 0.0

    overlap = 0
    remaining = list(exp_tokens)
    for token in out_tokens:
        if token in remaining:
            remaining.remove(token)
            overlap += 1
    if overlap == 0:
        return 0.0

    precision = overlap / len(out_tokens)
    recall = overlap / len(exp_tokens)
    return 2 * precision * recall / (precision + recall)


# Phrases a grounded system should use when the context cannot answer the
# question. Scored as a *success*: refusing to invent is correct behaviour.
REFUSAL_MARKERS = (
    "i don't know",
    "i do not know",
    "not enough information",
    "cannot answer",
    "can't answer",
    "no information",
    "not stated",
    "not specified",
    "unable to determine",
)


def is_refusal(output: str) -> bool:
    norm = _normalize(output)
    return any(_normalize(marker) in norm for marker in REFUSAL_MARKERS)


def groundedness(output: str, context: str, *, min_overlap: float = 0.5) -> float:
    """How much of the answer is traceable to the supplied context.

    A crude but honest proxy for hallucination: content words in the answer
    that never appear in the context are treated as unsupported. An explicit
    refusal scores 1.0 — declining to answer is grounded behaviour.

    This is intentionally not an LLM-as-judge. A deterministic metric can be
    unit-tested and cannot drift between runs; swap in a judge when you have
    the budget and a calibration set for it.
    """
    if is_refusal(output):
        return 1.0

    answer_tokens = [t for t in _tokens(output) if len(t) > 3]
    if not answer_tokens:
        return 1.0

    context_tokens = set(_tokens(context))
    supported = sum(1 for t in answer_tokens if t in context_tokens)
    score = supported / len(answer_tokens)
    # Rescale so that `min_overlap` maps to 1.0: natural language needs
    # connective words that will never appear in a short context.
    return min(1.0, score / min_overlap) if min_overlap > 0 else score


def hallucination_rate(scores: Sequence[float], *, threshold: float = 0.8) -> float:
    """Share of answers whose groundedness fell below ``threshold``."""
    if not scores:
        return 0.0
    return sum(1 for s in scores if s < threshold) / len(scores)
