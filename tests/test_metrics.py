import pytest

from evalkit.metrics import (
    contains_all,
    exact_match,
    groundedness,
    hallucination_rate,
    is_refusal,
    token_f1,
)


class TestExactMatch:
    def test_identical_strings_match(self):
        assert exact_match("Refund within 14 days", "Refund within 14 days") == 1.0

    def test_match_ignores_case_and_punctuation(self):
        assert exact_match("Refund within 14 days.", "refund within 14 days") == 1.0

    def test_different_strings_do_not_match(self):
        assert exact_match("14 days", "30 days") == 0.0


class TestContainsAll:
    def test_all_present(self):
        assert contains_all("Open Settings then Security", ["Settings", "Security"]) == 1.0

    def test_partial_returns_fraction(self):
        assert contains_all("Open Settings only", ["Settings", "Security"]) == 0.5

    def test_empty_requirement_is_vacuously_satisfied(self):
        assert contains_all("anything", []) == 1.0


class TestTokenF1:
    def test_identical_text_scores_one(self):
        assert token_f1("the plan costs 29 dollars", "the plan costs 29 dollars") == 1.0

    def test_disjoint_text_scores_zero(self):
        assert token_f1("alpha beta", "gamma delta") == 0.0

    def test_partial_overlap_is_between(self):
        score = token_f1("Pro costs 29 dollars monthly", "Pro costs 29 dollars")
        assert 0.0 < score < 1.0


class TestRefusalDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "I don't know.",
            "Not enough information in the provided context.",
            "I cannot answer that request.",
            "That is not specified in the policy.",
        ],
    )
    def test_recognises_refusal_phrasings(self, text):
        assert is_refusal(text) is True

    def test_a_real_answer_is_not_a_refusal(self):
        assert is_refusal("Pro is $29 per month.") is False


class TestGroundedness:
    CONTEXT = "Pro is $29 per month and includes priority support."

    def test_answer_drawn_from_context_scores_high(self):
        assert groundedness("Pro includes priority support", self.CONTEXT) >= 0.8

    def test_invented_answer_scores_low(self):
        score = groundedness(
            "Enterprise costs 199 dollars and ships worldwide overnight", self.CONTEXT
        )
        assert score < 0.8

    def test_refusal_counts_as_grounded(self):
        # Declining to answer is correct behaviour, not a hallucination.
        assert groundedness("I don't know.", self.CONTEXT) == 1.0

    def test_empty_answer_is_not_penalised(self):
        assert groundedness("", self.CONTEXT) == 1.0


class TestHallucinationRate:
    def test_counts_scores_below_threshold(self):
        assert hallucination_rate([1.0, 1.0, 0.2, 0.1], threshold=0.8) == 0.5

    def test_empty_input_is_zero(self):
        assert hallucination_rate([]) == 0.0

    def test_all_grounded_is_zero(self):
        assert hallucination_rate([1.0, 0.95, 0.9], threshold=0.8) == 0.0
