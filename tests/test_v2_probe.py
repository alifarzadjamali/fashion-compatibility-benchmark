import numpy as np
import pytest

from repbench.data.polyvore import CompatibilityExample, FITBQuestion
from repbench.features import native
from repbench.v2.probe import run_probe

DIM = 16


def _world(seed=0, dim=DIM, n_items=40, eval_items=10):
    rng = np.random.default_rng(seed)
    cluster = np.arange(n_items) % 2
    centers = rng.normal(size=(2, dim)) * 1.5
    emb = {f"i{k}": centers[cluster[k]] + rng.normal(size=dim) for k in range(n_items)}
    pool_a = np.arange(n_items - eval_items)
    pool_b = np.arange(n_items - eval_items, n_items)

    def outfits(pool, n):
        out = []
        for j in range(n):
            label = j % 2
            if label:
                c = rng.integers(2)
                ids = rng.choice(pool[cluster[pool] == c], 3, replace=False)
            else:
                ids = np.r_[rng.choice(pool[cluster[pool] == 0], 2, replace=False),
                            rng.choice(pool[cluster[pool] == 1], 1)]
                rng.shuffle(ids)
            out.append(CompatibilityExample(label, tuple(f"i{k}" for k in ids)))
        return out

    fitb = []
    for _ in range(30):
        c = rng.integers(2)
        same = rng.choice(pool_b[cluster[pool_b] == c], 3, replace=False)
        wrong = rng.choice(pool_b[cluster[pool_b] != c], 3, replace=False)
        answer = int(rng.integers(4))
        cands = list(wrong)
        cands.insert(answer, same[2])
        fitb.append(FITBQuestion(tuple(f"i{k}" for k in same[:2]),
                                 tuple(f"i{k}" for k in cands), answer))
    return emb, outfits(pool_a, 200), outfits(pool_a, 80), outfits(pool_b, 80), fitb


def _strip(result):
    return {k: v for k, v in result.items() if k != "fit_seconds"}


@pytest.mark.parametrize("learner", ["logistic", "xgboost", "mlp"])
def test_probe_learns_signal_and_is_deterministic(learner):
    from sklearn.metrics import roc_auc_score

    emb, train, valid, test, fitb = _world()
    first = run_probe(emb, train, valid, test, fitb, setting="native", learner=learner)
    second = run_probe(emb, train, valid, test, fitb, setting="native", learner=learner)
    assert _strip(first) == _strip(second)
    assert roc_auc_score(first["cp"]["labels"], first["cp"]["scores"]) > 0.7
    assert np.mean(first["fitb"]["correct"]) > 0.4
    assert first["feature_dim"] == 2 * DIM and first["n_train_items"] == 30
    sel = first["selection"]
    assert sel["chosen"] and len(sel["validation_curve"]) >= 4
    assert max(r["validation_auc"] for r in sel["validation_curve"]) == sel["validation_auc"]
    rows = first["fitb"]["candidate_scores"]
    assert first["fitb"]["chosen_index"] == [int(np.argmax(r)) for r in rows]


def test_alternative_operator_widens_features():
    emb, train, valid, test, _ = _world()
    out = run_probe(emb, train, valid, test, None, setting="native", learner="logistic",
                    operator="abs_product_sum")
    assert out["feature_dim"] == 3 * DIM and out["fitb"] is None


def test_item_transform_is_fitted_on_training_items_only(monkeypatch):
    emb, train, valid, test, fitb = _world()
    seen = []
    original = native.TrainOnlyNative.fit
    monkeypatch.setattr(native.TrainOnlyNative, "fit",
                        lambda self, x: seen.append(x.copy()) or original(self, x))
    base = run_probe(emb, train, valid, test, fitb, setting="native", learner="logistic")
    train_ids = sorted({i for e in train for i in e.item_ids})
    np.testing.assert_array_equal(seen[0], np.stack([emb[i] for i in train_ids]))
    # Corrupting eval-only items must not change anything fitted or selected.
    shifted = {k: (v * 50 + 7 if int(k[1:]) >= 30 else v) for k, v in emb.items()}
    moved = run_probe(shifted, train, valid, test, fitb, setting="native", learner="logistic")
    assert moved["selection"] == base["selection"]
    assert moved["cp"]["scores"] != base["cp"]["scores"]


def test_pca_setting_and_width_guard():
    emb, train, valid, test, fitb = _world(dim=136, n_items=220, eval_items=20)
    out = run_probe(emb, train, valid, test, fitb, setting="pca128", learner="logistic")
    assert out["item_dim_out"] == 128 and out["feature_dim"] == 256
    assert 0 < out["pca_explained_variance"] <= 1
    small = _world()
    with pytest.raises(ValueError, match="pca128"):
        run_probe(*small, setting="pca128", learner="logistic")


def test_fitb_ties_resolve_to_lowest_index():
    emb, train, valid, _, _ = _world()
    emb = {**emb, "copy": emb["i31"].copy()}
    questions = [FITBQuestion(("i30", "i32"), ("i31", "copy"), 1),
                 FITBQuestion(("i30", "i32"), ("copy", "i31"), 0)]
    out = run_probe(emb, train, valid, None, questions, setting="native", learner="logistic")
    a, b = out["fitb"]["candidate_scores"][0]
    assert a == b
    assert out["fitb"]["chosen_index"] == [0, 0] and out["fitb"]["correct"] == [0, 1]


def test_fail_closed_inputs():
    emb, train, valid, test, fitb = _world()
    ghost = [*test, CompatibilityExample(1, ("i30", "ghost"))]
    with pytest.raises(KeyError, match="lack embeddings"):
        run_probe(emb, train, valid, ghost, None, setting="native", learner="logistic")
    with pytest.raises(ValueError):
        run_probe(emb, train, valid, test, fitb, setting="pca64", learner="logistic")
    with pytest.raises(ValueError):
        run_probe(emb, train, valid, test, fitb, setting="native", learner="svm")
    with pytest.raises(ValueError):
        run_probe(emb, [e for e in train if e.label], valid, test, None,
                  setting="native", learner="logistic")
    with pytest.raises(ValueError):
        run_probe(emb, train, valid, None, None, setting="native", learner="logistic")
    bad = [FITBQuestion(("i30", "i32"), ("i31",), 3)]
    with pytest.raises(ValueError):
        run_probe(emb, train, valid, None, bad, setting="native", learner="logistic")
