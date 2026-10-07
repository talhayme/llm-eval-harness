"""A small, dependency-light evaluation harness for LLM features."""

from .dataset import Case, load_cases
from .gate import GateResult, Thresholds, evaluate_gate
from .metrics import groundedness, hallucination_rate, token_f1
from .providers import Completion, FallbackProvider, ScriptedProvider, provider_from_env
from .runner import RunReport, run_suite, score_case

__version__ = "0.1.0"

__all__ = [
    "Case", "load_cases",
    "Thresholds", "GateResult", "evaluate_gate",
    "groundedness", "hallucination_rate", "token_f1",
    "Completion", "ScriptedProvider", "FallbackProvider", "provider_from_env",
    "RunReport", "run_suite", "score_case",
]
