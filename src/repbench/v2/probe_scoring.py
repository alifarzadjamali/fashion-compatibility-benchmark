"""Score extra question sets (A100) with the probe a corpus' pipeline run fits.

``run_probe`` does not expose its fitted model, so the probe is re-fitted through the
identical code path (same TRAIN outfits, the same seeded fit-valid carve-out as
``scripts/v2/pipeline.py``, same seed and grid) with the extra questions passed as
``eval_fitb``. Fitting is deterministic and never sees the extra questions: the item
transform and the outfit scaler are fitted on TRAIN only and the hyperparameters are
selected on the fit-valid carve-out; extra items are only transformed (row-wise).
The selection (chosen C, validation AUC) can be cross-checked against the pipeline's
prediction file for the same corpus/model/setting/learner.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def cp_pairs(examples):
    """Copy of ``scripts/v2/pipeline.py::cp_pairs`` (balanced, paired CP lines)."""
    positive = [e for e in examples if e.label == 1]
    negative = [e for e in examples if e.label == 0]
    if len(positive) != len(negative):
        raise ValueError("Balanced paired CP construction expected")
    return list(zip(positive, negative, strict=True))


def fit_valid_carveout(train_examples, dataset_name, seed):
    """Exactly the pipeline's split of TRAIN CP pairs into (train_cp, fit_valid_cp)."""
    train_pairs = cp_pairs(train_examples)
    rng = random.Random(f"{seed}:{dataset_name}:fitvalid")
    fit_idx = set(rng.sample(range(len(train_pairs)), max(2, len(train_pairs) // 5)))
    train_cp = [e for i, p in enumerate(train_pairs) if i not in fit_idx for e in p]
    fit_cp = [e for i, p in enumerate(train_pairs) if i in fit_idx for e in p]
    return train_cp, fit_cp, {"train_pairs": len(train_pairs) - len(fit_idx),
                              "fit_valid_pairs": len(fit_idx)}


def load_store_checked(root, dataset, model):
    """Same rule as the pipeline: one store or the merge of its parts (trainvalid/ + test/,
    all/); every part checksum-verified, conflicting vectors fail."""
    from repbench.v2.stores import load_embedding_store

    folder = Path(root) / dataset / model
    store, _ = load_embedding_store(folder)
    meta_files = [folder / "metadata.json"] if (folder / "metadata.json").exists() else sorted(
        folder.glob("*/metadata.json"))
    if not meta_files:
        raise FileNotFoundError(f"No embedding metadata under {folder}")
    metas = [json.loads(f.read_text()) for f in meta_files]
    for key in ("model", "revision"):
        if len({m.get(key) for m in metas}) > 1:
            raise ValueError(f"Store parts under {folder} disagree on {key}")
    return store, metas[0]


def merge_stores(corpus_store, extra_store, extra_items, *, corpus_meta=None, extra_meta=None):
    """Corpus store plus the extra (A100) items; ids must not collide."""
    missing = sorted(set(extra_items) - set(extra_store))
    if missing:
        raise KeyError(f"{len(missing)} A100 items lack embeddings, e.g. {missing[:5]}")
    collisions = sorted(set(extra_items) & set(corpus_store))
    if collisions:
        raise ValueError(f"A100 item ids collide with corpus ids, e.g. {collisions[:5]}")
    for key in ("model", "revision", "embedding_dim"):
        left = (corpus_meta or {}).get(key)
        right = (extra_meta or {}).get(key)
        if left is not None and right is not None and left != right:
            raise ValueError(f"Corpus and A100 stores differ in {key}: {left} vs {right}")
    merged = dict(corpus_store)
    merged.update({i: extra_store[i] for i in extra_items})
    return merged


def score_with_fitted_probe(store, train_cp, fit_cp, questions, *, setting, learner,
                            operator="abs_product", seed=20260912):
    """Fit on TRAIN (selection on the fit-valid carve-out) and score ``questions``
    (Polyvore ``FITBQuestion`` objects). Returns ``run_probe``'s result."""
    from repbench.v2.probe import run_probe

    return run_probe(store, train_cp, fit_cp, None, list(questions), setting=setting,
                     learner=learner, operator=operator, seed=seed)


def selection_matches(selection, pipeline_prediction_file):
    """Compare chosen hyperparameters and validation AUC with a pipeline output."""
    reference = json.loads(Path(pipeline_prediction_file).read_text())
    if "error" in reference:
        return False
    other = reference["selection"]
    return (other["chosen"] == selection["chosen"]
            and float(other["validation_auc"]) == float(selection["validation_auc"]))
