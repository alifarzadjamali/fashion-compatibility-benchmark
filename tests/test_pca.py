import numpy as np

from repbench.features.pca import TrainOnlyPCA


def test_pca_has_fixed_width_and_unit_norm():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(300, 300)).astype(np.float32)
    valid = rng.normal(size=(20, 300)).astype(np.float32)
    test = rng.normal(size=(20, 300)).astype(np.float32)
    pipeline = TrainOnlyPCA(256, seed=7)
    transformed = pipeline.fit_transform_splits(train, valid, test)
    assert [x.shape for x in transformed] == [(300, 256), (20, 256), (20, 256)]
    np.testing.assert_allclose(np.linalg.norm(transformed[1], axis=1), 1.0, atol=1e-5)
    assert pipeline.n_train_samples_ == 300
