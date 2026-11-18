import numpy as np
import pytest

from repbench.features.native import TrainOnlyNative
from repbench.features.outfit_features import outfit_feature
from repbench.features.pair_operators import pair_outfit_feature


def test_native_uses_training_statistics_only_and_retains_width():
    train = np.array([[0.0, 2.0, 3.0], [2.0, 3.0, 6.0], [4.0, 4.0, 9.0]])
    transform = TrainOnlyNative().fit(train)
    means = transform.scaler_.mean_.copy()
    result = transform.transform(np.array([[1e6, -1e6, 1e6]]))
    np.testing.assert_array_equal(means, transform.scaler_.mean_)
    assert result.shape == (1, 3)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-6)
    assert transform.n_train_samples_ == 3
    with pytest.raises(RuntimeError):
        TrainOnlyNative().transform(train)


def test_pair_sensitivity_is_order_invariant_and_preserves_primary_blocks():
    vectors = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0]), "c": np.array([5.0, 6.0])}
    primary = pair_outfit_feature(["a", "b", "c"], vectors)
    np.testing.assert_allclose(primary, outfit_feature(["a", "b", "c"], vectors))
    extended = pair_outfit_feature(["a", "b", "c"], vectors, operator="abs_product_sum")
    np.testing.assert_allclose(extended[:4], primary)
    np.testing.assert_allclose(extended[4:], 2 * np.mean(list(vectors.values()), axis=0))
    np.testing.assert_allclose(
        extended, pair_outfit_feature(["c", "b", "a"], vectors, operator="abs_product_sum")
    )
