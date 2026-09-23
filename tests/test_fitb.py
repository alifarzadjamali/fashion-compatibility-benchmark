import pytest

from repbench.data.polyvore import FITBQuestion
from repbench.eval.fitb import evaluate_fitb


def test_fitb_inserts_each_candidate_and_selects_highest():
    question = FITBQuestion(("top", "shoe"), ("bad", "good", "other"), 1)
    result = evaluate_fitb([question], lambda outfit: float("good" in outfit))
    assert result["accuracy"] == 1.0
    assert result["predictions"] == [1]


def test_fitb_rejects_empty_questions():
    with pytest.raises(ValueError, match="at least one question"):
        evaluate_fitb([], lambda outfit: 0.0)
