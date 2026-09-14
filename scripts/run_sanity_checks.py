from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.data.validation import inspect_disjoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/polyvore_outfits")
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    dataset = PolyvoreDisjoint(args.root)
    report = inspect_disjoint(dataset)
    rng = np.random.default_rng(args.seed)
    cp = dataset.compatibility("test")
    cp_auc = roc_auc_score([x.label for x in cp], rng.random(len(cp)))
    fitb = dataset.fitb("test")
    guesses = [int(rng.integers(len(q.candidate_item_ids))) for q in fitb]
    fitb_acc = np.mean([guess == q.correct_index for guess, q in zip(guesses, fitb)])
    report["chance_checks"] = {"cp_auc": float(cp_auc), "fitb_accuracy": float(fitb_acc)}
    output = Path("artifacts/sanity_check_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
