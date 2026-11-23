import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.v2.outfit_transformer_v2 import (
    EmbeddingTable,
    TrainConfig,
    score_cp,
    score_fitb,
    train,
)
from repbench.v2.stores import load_embedding_store

DIM, STYLES = 16, 4


@pytest.fixture(autouse=True)
def single_thread():
    # Tiny models: intra-op threading only adds overhead on many-core CPUs.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def build_protocol(root, seed=0, outfits_per_split=None):
    """Planted signal: positive outfits share a style direction; negatives mix styles."""
    outfits_per_split = outfits_per_split or {"train": 160, "valid": 80, "test": 60}
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(STYLES, DIM)) * 1.5
    protocol = root / "protocol"
    protocol.mkdir(parents=True)
    ids, vectors = [], []
    for split, count in outfits_per_split.items():
        outfits, styles = [], []
        for o in range(count):
            set_id = f"{split}{o}"
            style = o % STYLES
            items = []
            for k in range(1, 4):
                item = f"{set_id}_item{k}"
                ids.append(item)
                vectors.append(centres[style] + rng.normal(size=DIM))
                items.append({"item_id": item, "index": k})
            outfits.append({"set_id": set_id, "items": items})
            styles.append(style)
        (protocol / f"{split}.json").write_text(json.dumps(outfits))
        lines = []
        for o in range(count):
            lines.append("1 " + " ".join(f"{split}{o}_{k}" for k in range(1, 4)))
            others = rng.choice(count, size=3, replace=False)
            lines.append("0 " + " ".join(f"{split}{x}_{k + 1}" for k, x in enumerate(others)))
        (protocol / f"compatibility_{split}.txt").write_text("\n".join(lines) + "\n")
        fitb = []
        for o in range(count):
            wrong = [x for x in range(count) if styles[x] != styles[o]][:3]
            answers = [f"{split}{o}_3"] + [f"{split}{x}_3" for x in wrong]
            order = rng.permutation(4)
            fitb.append(
                {
                    "question": [f"{split}{o}_1", f"{split}{o}_2"],
                    "answers": [answers[i] for i in order],
                }
            )
        (protocol / f"fill_in_blank_{split}.json").write_text(json.dumps(fitb))
    store = root / "store.npz"
    norm = np.asarray(vectors, np.float32)
    norm /= np.linalg.norm(norm, axis=1, keepdims=True)
    np.savez(store, item_ids=np.asarray(ids), embeddings=norm)
    return protocol, store


def test_planted_signal_and_determinism(tmp_path):
    protocol, store_path = build_protocol(tmp_path)
    data = PolyvoreDisjoint(tmp_path, protocol, "synthetic")
    store, _ = load_embedding_store(store_path)
    table = EmbeddingTable(store)
    config = TrainConfig(lr=3e-4, seed=7, max_epochs=12, patience=4)
    tr, va, te = (data.compatibility(s) for s in ("train", "valid", "test"))
    first = train(config, table, tr, va, max_length=4)
    second = train(config, table, tr, va, max_length=4)
    assert first.history == second.history
    assert first.best_epoch == second.best_epoch
    s1 = score_cp(first.model, table, te, max_length=4)
    s2 = score_cp(second.model, table, te, max_length=4)
    np.testing.assert_array_equal(s1, s2)
    auc = roc_auc_score([e.label for e in te], s1)
    assert auc > 0.6, auc
    scores, correct = score_fitb(first.model, table, data.fitb("test"), max_length=4)
    assert scores.shape == (60, 4) and correct.mean() > 0.3
    assert 1 <= first.best_epoch <= first.epochs_run <= 12
    assert first.early_stopped == (
        first.epochs_run < 12 or first.epochs_run - first.best_epoch >= 4
    )
    different = train(TrainConfig(lr=3e-4, seed=8, max_epochs=2), table, tr, va, max_length=4)
    assert different.history != first.history[:2]


def test_missing_embedding_fails(tmp_path):
    table = EmbeddingTable({"a": np.ones(4), "b": np.zeros(4)})
    with pytest.raises(KeyError):
        table.encode([("a", "c")], 3)
    with pytest.raises(ValueError):
        table.encode([("a", "b", "a", "b")], 3)


def _script():
    spec = importlib.util.spec_from_file_location(
        "ot_script", Path("scripts/v2/run_outfit_transformer_v2.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_train_select_evaluate(tmp_path):
    protocol, store = build_protocol(
        tmp_path, outfits_per_split={"train": 40, "valid": 20, "test": 12}
    )
    out = tmp_path / "ot"
    script = _script()
    common = [
        "--protocol-root",
        str(protocol),
        "--dataset-name",
        "synthetic_clean",
        "--embeddings",
        str(store),
        "--output-root",
        str(out),
        "--device",
        "cpu",
    ]
    train_args = [
        "train",
        *common,
        "--encoder",
        "fake",
        "--lrs",
        "1e-4",
        "3e-4",
        "--seeds",
        "1",
        "2",
        "--max-epochs",
        "2",
    ]
    script.main(train_args)
    metrics = json.loads((out / "lr_0.0003" / "seed_2" / "metrics.json").read_text())
    for key in ("best_epoch", "early_stopping_fired", "valid_cp_auc", "valid_fitb_accuracy"):
        assert key in metrics
    cp = pd.read_csv(out / "lr_0.0003" / "seed_2" / "cp_scores_valid.csv")
    assert len(cp) == 40 and set(cp.columns) == {"example_index", "label", "score"}
    script.main(train_args)  # resume: finished runs are skipped, not overwritten
    script.main(["select", "--output-root", str(out), "--min-seeds", "2"])
    selection = json.loads((out / "selection.json").read_text())
    assert selection["selected_lr"] in (1e-4, 3e-4) and selection["runs_total"] == 4
    with pytest.raises(FileExistsError):
        script.main(["select", "--output-root", str(out), "--min-seeds", "2"])
    with pytest.raises(SystemExit, match="freeze"):
        script.main(["evaluate", *common, "--split", "test"])
    script.main(["evaluate", *common, "--split", "test", "--confirm-protocol-frozen"])
    fitb = pd.read_csv(out / "eval_test" / "seed_1" / "fitb_candidate_scores_test.csv")
    assert len(fitb) == 12 * 4 and fitb.is_answer.sum() == 12
    with pytest.raises(SystemExit, match="historical"):
        script.main(
            [
                "train",
                *common[:2],
                "--dataset-name",
                "historical_polyvore_d",
                *common[4:],
                "--encoder",
                "x",
            ]
        )
