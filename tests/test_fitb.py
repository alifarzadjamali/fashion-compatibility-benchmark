from repbench.data.polyvore import FITBQuestion
from repbench.eval.fitb import evaluate_fitb


def test_fitb_inserts_each_candidate_and_selects_highest():
    question = FITBQuestion(("top", "shoe"), ("bad", "good", "other"), 1)
    result = evaluate_fitb([question], lambda outfit: float("good" in outfit))
    assert result["accuracy"] == 1.0
    assert result["predictions"] == [1]
