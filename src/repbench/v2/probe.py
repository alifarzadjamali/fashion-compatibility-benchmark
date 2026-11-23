"""V2 frozen-feature probe (HANDOFF_V2 §6.1): train-only item transform, pair features,
train-only outfit scaler, validation-selected learner, CP and FITB scoring."""

from __future__ import annotations

import time
from itertools import product

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from repbench.features.native import TrainOnlyNative
from repbench.features.pair_operators import pair_outfit_matrix
from repbench.features.pca import TrainOnlyPCA
from repbench.v2.logistic import V2LogisticCompatibility

SETTINGS = {"native": None, "pca128": 128, "pca256": 256, "pca512": 512}
LEARNERS = ("logistic", "xgboost", "mlp")
XGB_GRID = {"max_depth": (3, 6), "n_estimators": (200, 500), "learning_rate": (0.1,)}
MLP_GRID = {"hidden_layer_sizes": ((256,), (512, 256)), "alpha": (1e-4, 1e-3)}
FITB_TIE_RULE = "argmax of candidate probability; exact ties resolve to the lowest candidate index"


def _grid(spec):
    keys = list(spec)
    return [dict(zip(keys, values)) for values in product(*(spec[k] for k in keys))]


def _labels(examples, name):
    if not examples:
        raise ValueError(f"{name} compatibility split is empty")
    labels = np.asarray([e.label for e in examples], dtype=np.int64)
    if not np.isin(labels, (0, 1)).all() or len(np.unique(labels)) != 2:
        raise ValueError(f"{name} labels must be binary and contain both classes")
    return labels


def _check_embeddings(embeddings, item_ids):
    missing = sorted(set(item_ids) - set(embeddings))
    if missing:
        raise KeyError(f"{len(missing)} outfit items lack embeddings, e.g. {missing[:5]}")
    widths = {np.asarray(embeddings[i]).shape for i in item_ids}
    if len(widths) != 1 or len(next(iter(widths))) != 1:
        raise ValueError("Item embeddings must be 1-D vectors of a single width")
    return next(iter(widths))[0]


def _select(candidates, fit_one, x_train, y_train, x_valid, y_valid):
    """Fit every grid point; keep the first (grid-order) maximum of validation AUC."""
    curve, best = [], None
    for params in candidates:
        model = fit_one(params).fit(x_train, y_train)
        auc = float(roc_auc_score(y_valid, model.predict_proba(x_valid)[:, 1]))
        curve.append({**{k: list(v) if isinstance(v, tuple) else v for k, v in params.items()},
                      "validation_auc": auc})
        if best is None or auc > best[0]:
            best = (auc, params, model)
    return best, curve


def _fit_learner(learner, x_train, y_train, x_valid, y_valid, seed, n_jobs):
    if learner == "logistic":
        model = V2LogisticCompatibility(seed=seed).fit(x_train, y_train, x_valid, y_valid)
        return model.predict_proba, {
            "validation_curve": model.validation_curve_,
            "chosen": {"c": float(model.best_c_)},
            "validation_auc": float(model.validation_auc_),
            "grid_extensions": model.grid_extensions_,
        }
    if learner == "xgboost":
        from xgboost import XGBClassifier

        def fit_one(params):
            return XGBClassifier(tree_method="hist", device="cpu", n_jobs=n_jobs,
                                 random_state=seed, eval_metric="logloss", **params)

        candidates = _grid(XGB_GRID)
    elif learner == "mlp":
        def fit_one(params):
            return MLPClassifier(early_stopping=True, validation_fraction=0.1, max_iter=200,
                                 random_state=seed, **params)

        candidates = _grid(MLP_GRID)
    else:
        raise ValueError(f"Unknown learner {learner!r}; expected one of {LEARNERS}")
    (auc, params, model), curve = _select(candidates, fit_one, x_train, y_train, x_valid, y_valid)
    chosen = {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}
    return (lambda x: model.predict_proba(x)[:, 1]), {
        "validation_curve": curve, "chosen": chosen, "validation_auc": auc,
        "grid_extensions": [],
    }


def run_probe(embeddings, train_cp, fit_valid_cp, eval_cp, eval_fitb, *, setting, learner,
              operator="abs_product", seed=20260912, n_jobs=4) -> dict:
    """Run one probe configuration. ``eval_cp`` or ``eval_fitb`` may be None (not both)."""
    if setting not in SETTINGS:
        raise ValueError(f"Unknown setting {setting!r}; expected one of {tuple(SETTINGS)}")
    if learner not in LEARNERS:
        raise ValueError(f"Unknown learner {learner!r}; expected one of {LEARNERS}")
    if eval_cp is None and eval_fitb is None:
        raise ValueError("At least one of eval_cp / eval_fitb is required")
    y_train, y_valid = _labels(train_cp, "train"), _labels(fit_valid_cp, "fit_valid")
    eval_cp = list(eval_cp) if eval_cp is not None else None
    eval_fitb = list(eval_fitb) if eval_fitb is not None else None
    y_eval = _labels(eval_cp, "eval") if eval_cp is not None else None
    fitb_outfits = []
    for q in eval_fitb or []:
        if not q.candidate_item_ids or not 0 <= q.correct_index < len(q.candidate_item_ids):
            raise ValueError(f"Invalid FITB question {q}")
        fitb_outfits.extend(tuple(q.question_item_ids) + (c,) for c in q.candidate_item_ids)
    if eval_fitb is not None and not fitb_outfits:
        raise ValueError("eval_fitb is empty")

    all_outfits = [tuple(e.item_ids) for e in [*train_cp, *fit_valid_cp, *(eval_cp or [])]]
    all_outfits += fitb_outfits
    needed = sorted({i for outfit in all_outfits for i in outfit})
    in_dim = _check_embeddings(embeddings, needed)
    train_items = sorted({i for e in train_cp for i in e.item_ids})

    started = time.perf_counter()
    width = SETTINGS[setting]
    if width is None:
        transform = TrainOnlyNative()
    else:
        if width > in_dim or width > len(train_items):
            raise ValueError(f"{setting} needs width {width} <= raw dim ({in_dim}) and "
                             f"<= training items ({len(train_items)})")
        transform = TrainOnlyPCA(n_components=width, seed=seed)
    transform.fit(np.stack([np.asarray(embeddings[i], dtype=np.float64) for i in train_items]))
    projected = transform.transform(
        np.stack([np.asarray(embeddings[i], dtype=np.float64) for i in needed]))
    items = dict(zip(needed, projected))

    def features(examples):
        # A repeated item adds no pairwise information; 9/33,982 Polyvore-Clean train CP
        # outfits list an item twice. De-duplicate order-preservingly (logged decision).
        return pair_outfit_matrix([tuple(dict.fromkeys(e)) for e in examples], items,
                                  operator=operator)

    scaler = StandardScaler().fit(features(e.item_ids for e in train_cp))
    x_train = scaler.transform(features(e.item_ids for e in train_cp))
    x_valid = scaler.transform(features(e.item_ids for e in fit_valid_cp))
    predict, selection = _fit_learner(learner, x_train, y_train, x_valid, y_valid, seed, n_jobs)
    fit_seconds = time.perf_counter() - started

    result = {
        "setting": setting, "learner": learner, "operator": operator, "seed": seed,
        "item_dim_in": int(in_dim), "item_dim_out": int(projected.shape[1]),
        "feature_dim": int(x_train.shape[1]), "n_train_items": len(train_items),
        "n_train_outfits": len(train_cp), "n_fit_valid_outfits": len(fit_valid_cp),
        "pca_explained_variance": (transform.explained_variance_ratio if width else None),
        "selection": selection, "fit_seconds": float(fit_seconds),
        "cp": None, "fitb": None,
    }
    if eval_cp is not None:
        scores = predict(scaler.transform(features(e.item_ids for e in eval_cp)))
        result["cp"] = {"labels": y_eval.tolist(), "scores": [float(s) for s in scores]}
    if eval_fitb is not None:
        flat = predict(scaler.transform(features(fitb_outfits)))
        candidate_scores, chosen, offset = [], [], 0
        for q in eval_fitb:
            row = [float(s) for s in flat[offset: offset + len(q.candidate_item_ids)]]
            offset += len(row)
            candidate_scores.append(row)
            chosen.append(int(np.argmax(row)))  # np.argmax returns the first maximum
        correct_index = [int(q.correct_index) for q in eval_fitb]
        result["fitb"] = {
            "candidate_scores": candidate_scores, "chosen_index": chosen,
            "correct_index": correct_index,
            "correct": [int(c == k) for c, k in zip(chosen, correct_index)],
            "tie_rule": FITB_TIE_RULE,
        }
    return result
