import numpy as np
import pytest

from repbench.eval.calibration import expected_calibration_error, reliability_bins


def test_perfect_binary_calibration_has_zero_ece():
    assert expected_calibration_error([0, 1], [0.0, 1.0], 10) == pytest.approx(0.0)


def test_boundary_probabilities_are_counted_once():
    rows = reliability_bins(np.array([0, 1, 1]), np.array([0.0, 0.5, 1.0]), 10)
    assert sum(row["count"] for row in rows) == 3
    assert rows[0]["count"] == 1
    assert rows[5]["count"] == 1
    assert rows[9]["count"] == 1


def test_invalid_probability_rejected():
    with pytest.raises(ValueError):
        expected_calibration_error([0], [1.1])


def test_reliability_bins_reject_invalid_probability():
    with pytest.raises(ValueError, match="Probabilities must lie"):
        reliability_bins([0], [-0.1])
