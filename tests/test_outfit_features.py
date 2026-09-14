import numpy as np

from repbench.features.outfit_features import outfit_feature


def test_outfit_feature_is_order_invariant():
    embeddings = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([0.0, 1.0], dtype=np.float32),
        "c": np.array([0.5, 0.5], dtype=np.float32),
    }
    expected = outfit_feature(("a", "b", "c"), embeddings)
    actual = outfit_feature(("c", "a", "b"), embeddings)
    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (4,)
