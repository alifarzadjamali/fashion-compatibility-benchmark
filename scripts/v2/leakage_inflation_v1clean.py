"""Leakage-inflation analysis: v1-Clean vs v2-Clean test metrics (frozen protocol,
near_duplicate_policy.comparison: "v1-Clean splits evaluated only as leakage-inflation
analysis").

Runs the PRIMARY probe exactly as scripts/v2/pipeline.py does in Phase 2 (native
features, train-item scaler, abs_product operator, logistic, C selected on VALID,
test scored once) on the v1-Clean protocols, then compares each (dataset, encoder,
metric) with the recorded v2-Clean test value. Descriptive only: the test sets differ,
so no paired CI of the difference is computed.

Example:
  .venv/bin/python scripts/v2/leakage_inflation_v1clean.py \
    --spec polyvore_d_clean=<v1clean/polyvore>=<emb root>=polyvore_v2_clean=polyvore_v2_clean \
    --models resnet50,... --v2-metrics <results_r20k/metrics.csv> --results <out>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
_spec = importlib.util.spec_from_file_location("v2pipeline", REPO / "scripts/v2/pipeline.py")
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)  # reuse load_store / cp_pairs unchanged

SEED = 20260912  # pipeline default, used for the recorded v2 run
MIN_RANK_ENCODERS = 10  # rank statistics are Polyvore-only (15 encoders); IQON has 3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_inputs(protocol):
    from repbench.data.polyvore import PolyvoreDisjoint

    data = PolyvoreDisjoint(protocol.parent, protocol, protocol.name)
    flat = lambda pairs: [e for p in pairs for e in p]
    train_cp = flat(pipeline.cp_pairs(data.compatibility("train")))  # Phase 2: all train
    fit_cp = flat(pipeline.cp_pairs(data.compatibility("valid")))  # C selected on valid
    test = data.compatibility("test")
    eval_cp = [e for e in test if e.label == 1] + [e for e in test if e.label == 0]
    return train_cp, fit_cp, eval_cp, data.fitb("test")


_CTX = {}


def _run(job):
    from repbench.v2.logistic import GridEdgeError
    from repbench.v2.probe import run_probe

    name, protocol, emb_root, emb_name, model, out_path = job
    if out_path.exists():
        return str(out_path), 0.0
    started = time.time()
    train_cp, fit_cp, eval_cp, eval_fitb = _CTX[str(protocol)]
    store, meta = pipeline.load_store(Path(emb_root), emb_name, model)
    try:
        out = run_probe(store, train_cp, fit_cp, eval_cp, eval_fitb, setting="native",
                        learner="logistic", operator="abs_product", seed=SEED)
    except GridEdgeError as error:
        out = {"error": "grid_edge", "detail": str(error)}
    out["provenance"] = {"eval_split": "test", "selection_split": "valid",
                         "items_sha256": meta["items_sha256"], "store_parts": meta["parts"],
                         "protocol": str(protocol), "dataset": name}
    with open(out_path, "x") as stream:
        json.dump(out, stream)
    return str(out_path), time.time() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", action="append", required=True,
                        help="name=protocol=emb_root=emb_name=v2_dataset[=models]")
    parser.add_argument("--models", required=True, help="default model list")
    parser.add_argument("--v2-metrics", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    started = time.time()
    preds = args.results / "predictions"
    preds.mkdir(parents=True, exist_ok=True)

    jobs, specs, provenance = [], [], {"protocol_files": {}, "embedding_stores": {}}
    for spec in args.spec:
        parts = spec.split("=")
        name, protocol, emb_root, emb_name, v2_name = parts[:5]
        models = parts[5].split(",") if len(parts) > 5 else args.models.split(",")
        protocol = Path(protocol).expanduser()
        specs.append((name, v2_name, models))
        _CTX[str(protocol)] = protocol_inputs(protocol)
        provenance["protocol_files"][name] = {
            p.name: sha(p) for p in sorted(protocol.iterdir()) if p.is_file()}
        train_cp, fit_cp, eval_cp, eval_fitb = _CTX[str(protocol)]
        provenance[f"{name}_counts"] = {"train_pairs": len(train_cp) // 2,
                                        "valid_pairs": len(fit_cp) // 2,
                                        "test_cp": len(eval_cp), "test_fitb": len(eval_fitb)}
        for model in models:
            jobs.append((name, protocol, emb_root, emb_name, model,
                         preds / f"{name}__{model}__native__logistic.json"))
    seconds = {}
    if args.workers <= 1:  # in-process: avoids the forked-pool deadlock (deviation D2a)
        finished = map(_run, jobs)
    else:
        pool = mp.get_context("fork").Pool(args.workers)
        finished = pool.imap_unordered(_run, jobs)
    for path, sec in finished:
        seconds[Path(path).name] = sec
        print(f"done {Path(path).name} ({sec:.0f}s)", flush=True)
    if args.workers > 1:
        pool.close()
        pool.join()

    v2 = pd.read_csv(args.v2_metrics)
    v2 = v2[(v2.setting == "native") & (v2.learner == "logistic") & (v2.split == "test")]
    rows, metric_rows = [], []
    for name, v2_name, models in specs:
        for model in models:
            out = json.loads((preds / f"{name}__{model}__native__logistic.json").read_text())
            provenance["embedding_stores"][f"{name}/{model}"] = out["provenance"]["items_sha256"]
            if "error" in out:
                metric_rows.append(dict(dataset=name, model=model, error=out["error"]))
                continue
            values = {"cp_auc": roc_auc_score(out["cp"]["labels"], out["cp"]["scores"]),
                      "fitb_acc": float(np.mean(out["fitb"]["correct"]))}
            for metric, value in values.items():
                metric_rows.append(dict(dataset=name, model=model, setting="native",
                                        learner="logistic", split="test", metric=metric,
                                        value=value, c=out["selection"]["chosen"]["c"],
                                        validation_auc=out["selection"]["validation_auc"]))
                ref = v2[(v2.dataset == v2_name) & (v2.model == model) & (v2.metric == metric)]
                if len(ref) != 1:
                    raise ValueError(f"Expected one v2 reference row for {v2_name}/{model}/"
                                     f"{metric}, found {len(ref)}")
                v2_value = float(ref.value.iloc[0])
                rows.append(dict(dataset=name, v2_dataset=v2_name, model=model, metric=metric,
                                 v1clean_test=value, v2clean_test=v2_value,
                                 inflation=value - v2_value,
                                 inflation_pct=100 * (value - v2_value) / v2_value))
    summary = pd.DataFrame(rows)
    rankable = {d for d, n in summary.groupby("dataset").model.nunique().items()
                if n >= MIN_RANK_ENCODERS}  # Polyvore (15); IQON's 3 encoders are not ranked
    for (dataset, metric), block in summary.groupby(["dataset", "metric"]):
        if dataset not in rankable:
            continue
        for col in ("v1clean_test", "v2clean_test"):
            summary.loc[block.index, f"rank_{col}"] = block[col].rank(ascending=False).values
    pd.DataFrame(metric_rows).to_csv(args.results / "metrics_v1clean.csv", index=False)
    summary.to_csv(args.results / "summary.csv", index=False)

    aggregates = []
    for (dataset, metric), block in summary.groupby(["dataset", "metric"]):
        row = dict(
            dataset=dataset, metric=metric, n_encoders=len(block),
            mean_inflation=float(block.inflation.mean()),
            median_inflation=float(block.inflation.median()),
            min_inflation=float(block.inflation.min()), max_inflation=float(block.inflation.max()),
            n_positive=int((block.inflation > 0).sum()))
        if dataset in rankable:
            rho, rho_p = spearmanr(block.v1clean_test, block.v2clean_test)
            tau, tau_p = kendalltau(block.v1clean_test, block.v2clean_test)
            row.update(spearman_rho=float(rho), spearman_p=float(rho_p),
                       kendall_tau=float(tau), kendall_p=float(tau_p),
                       top_model_v1clean=block.loc[block.v1clean_test.idxmax(), "model"],
                       top_model_v2clean=block.loc[block.v2clean_test.idxmax(), "model"])
        else:
            row["rank_statistics"] = (f"not reported: {len(block)} encoders "
                                      f"(< {MIN_RANK_ENCODERS}); per-encoder differences only")
        aggregates.append(row)
    try:
        commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                         text=True).strip()
    except Exception:  # noqa: BLE001
        commit = None
    record = {
        "analysis": "v1-Clean vs v2-Clean leakage inflation (descriptive; different test sets)",
        "probe": {"setting": "native", "operator": "abs_product", "learner": "logistic",
                  "scaler": "train-item/outfit StandardScaler fit on train only",
                  "selection_split": "valid", "eval_split": "test", "seed": SEED},
        "aggregates": aggregates,
        "command": " ".join([".venv/bin/python", *sys.argv]),
        "git_commit": commit, "script_sha256": sha(__file__),
        "pipeline_sha256": sha(REPO / "scripts/v2/pipeline.py"),
        "probe_sha256": sha(REPO / "src/repbench/v2/probe.py"),
        "v2_metrics": {"path": str(args.v2_metrics), "sha256": sha(args.v2_metrics)},
        "provenance": provenance, "cell_seconds": seconds,
        "runtime_seconds": time.time() - started,
    }
    (args.results / "summary.json").write_text(json.dumps(record, indent=1, default=float))
    print(pd.DataFrame(aggregates).to_string())


if __name__ == "__main__":
    main()
