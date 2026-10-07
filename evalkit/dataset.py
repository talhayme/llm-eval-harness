"""Golden-set loading and validation.

A golden set is a YAML/JSON list of cases. Keeping it as data (not code)
means a domain expert can extend it without touching the harness, and the
diff of a prompt change stays readable in review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Case:
    """One scenario in the golden set."""

    id: str
    prompt: str
    expected: Optional[str] = None
    # Substrings the answer must contain; checked by `contains_all`.
    must_include: List[str] = field(default_factory=list)
    # Grounding context. When present, groundedness is scored against it.
    context: str = ""
    # True when the correct behaviour is to refuse rather than answer.
    expect_refusal: bool = False
    system: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Case":
        missing = [k for k in ("id", "prompt") if not raw.get(k)]
        if missing:
            raise ValueError(f"case is missing required field(s): {', '.join(missing)}")

        known = {
            "id", "prompt", "expected", "must_include",
            "context", "expect_refusal", "system", "tags",
        }
        unknown = set(raw) - known
        if unknown:
            # Fail loudly: a typo'd key silently weakens the suite otherwise.
            raise ValueError(f"case {raw['id']!r} has unknown field(s): {', '.join(sorted(unknown))}")

        return cls(
            id=str(raw["id"]),
            prompt=str(raw["prompt"]),
            expected=raw.get("expected"),
            must_include=list(raw.get("must_include") or []),
            context=str(raw.get("context") or ""),
            expect_refusal=bool(raw.get("expect_refusal", False)),
            system=raw.get("system"),
            tags=list(raw.get("tags") or []),
        )


def load_cases(path: Path) -> List[Case]:
    """Load cases from a .json or .yaml file.

    YAML support is optional: if PyYAML is absent we say so plainly rather
    than failing with an opaque ImportError deep in a run.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                f"{path.name} is YAML but PyYAML is not installed. "
                "Install it (`pip install pyyaml`) or use a .json golden set."
            ) from exc
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)

    if not isinstance(raw, list):
        raise ValueError(f"{path.name} must contain a list of cases, got {type(raw).__name__}")

    cases = [Case.from_dict(item) for item in raw]

    duplicates = _find_duplicates([c.id for c in cases])
    if duplicates:
        # Duplicate ids make per-case regression tracking meaningless.
        raise ValueError(f"duplicate case id(s) in {path.name}: {', '.join(duplicates)}")

    return cases


def _find_duplicates(ids: List[str]) -> List[str]:
    seen, dupes = set(), set()
    for item in ids:
        if item in seen:
            dupes.add(item)
        seen.add(item)
    return sorted(dupes)
