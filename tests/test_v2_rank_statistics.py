import numpy as np
import pytest
from scipy.stats import friedmanchisquare

from repbench.v2.rank_statistics import friedman_nemenyi, plot_critical_difference

HAND = [[0.9, 0.8, 0.7], [0.9, 0.7, 0.8], [0.8, 0.9, 0.7], [0.9, 0.8, 0.7]]


def test_hand_computed_example():
    # Ranks: [1,2,3],[1,3,2],[2,1,3],[1,2,3] -> rank sums 5, 8, 11.
    # chi2 = 12/(4*3*4)*(25+64+121) - 3*4*4 = 4.5; W = 4.5/(4*2) = 0.5625.
    out = friedman_nemenyi(HAND, ["a", "b", "c"])
    assert out["label"] == "descriptive"
    assert out["average_ranks"] == {"a": 1.25, "b": 2.0, "c": 2.75}
    assert out["friedman_chi2"] == pytest.approx(4.5)
    assert out["kendall_w"] == pytest.approx(0.5625)
    assert out["nemenyi_q"] == pytest.approx(2.343, abs=1e-3)  # Demsar (2006) table, k=3
    assert out["critical_difference"] == pytest.approx(2.3437 * np.sqrt(0.5), abs=1e-3)
    assert out["nemenyi_significant"] == [[0, 0, 0]] * 3
    assert out["friedman_pvalue"] == pytest.approx(friedmanchisquare(*np.array(HAND).T).pvalue)


def test_ties_match_scipy_and_lower_is_better():
    rng = np.random.default_rng(0)
    x = np.round(rng.random((8, 5)), 1)
    out = friedman_nemenyi(x, list("abcde"))
    assert out["friedman_chi2"] == pytest.approx(friedmanchisquare(*x.T).statistic)
    flipped = friedman_nemenyi(-x, list("abcde"), higher_is_better=False)
    assert flipped["average_ranks"] == out["average_ranks"]


def test_perfect_agreement_and_plot(tmp_path):
    x = np.tile([0.9, 0.8, 0.7, 0.6], (10, 1))
    out = friedman_nemenyi(x, list("abcd"))
    assert out["kendall_w"] == pytest.approx(1.0)
    assert out["nemenyi_significant"][0][3] == 1
    path = tmp_path / "cd.png"
    plot_critical_difference(out, path)
    assert path.stat().st_size > 0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        friedman_nemenyi([[1, 2]], ["a", "b"])
    with pytest.raises(ValueError):
        friedman_nemenyi(HAND, ["a", "a", "c"])
    with pytest.raises(ValueError):
        friedman_nemenyi([[1, np.nan, 2], [1, 2, 3]], list("abc"))
