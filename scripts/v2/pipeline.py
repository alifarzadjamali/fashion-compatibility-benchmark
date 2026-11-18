"""v2 analysis pipeline over extracted embeddings: probes -> stats -> calibration -> CSVs.

Phase 1 (G5 dry run) evaluates on **valid** only: C/hyperparameters are selected
on a seeded 20% *fit-valid* carve-out of TRAIN outfits' CP pairs, so the valid
split plays the role the test split will play after the freeze. `--eval-split
test` is refused unless a frozen protocol hash file is supplied (Phase 2).

Writes, under --results (never overwriting a completed file):
  predictions/<dataset>__<model>__<setting>__<learner>.json  per-example outputs
  metrics.csv  pairwise.csv  rank_intervals.csv  equivalence.csv
  calibration.csv  friedman.json  selection.json  manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

MARGINS = {"cp_auc": (0.002, 0.005, 0.010), "fitb_acc": (0.005, 0.01, 0.02)}


def load_store(root, dataset, model):
    folder = root / dataset / model
    meta = json.loads((folder / "metadata.json").read_text())
    values = np.load(folder / "items.npy")
    if hashlib.sha256((folder / "items.npy").read_bytes()).hexdigest() != meta["items_sha256"]:
        raise ValueError(f"Embedding checksum mismatch for {dataset}/{model}")
    ids = json.loads((folder / "item_ids.json").read_text())
    return {i: values[k] for k, i in enumerate(ids)}, meta


def cp_pairs(examples):
    positive = [e for e in examples if e.label == 1]
    negative = [e for e in examples if e.label == 0]
    if len(positive) != len(negative):
        raise ValueError("Balanced paired CP construction expected")
    return list(zip(positive, negative, strict=True))


def item_components(outfits):
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for outfit in outfits:
        first = find(outfit.item_ids[0])
        for item in outfit.item_ids[1:]:
            parent[find(item)] = first
    return {o.set_id: find(o.item_ids[0]) for o in outfits}


def groups_for(protocol, data, split, raw_cp_lines, raw_fitb, grouping):
    if grouping == "two_way":  # user x item-component (pigeonhole) bootstrap
        first = groups_for(protocol, data, split, raw_cp_lines, raw_fitb, "user")
        second = groups_for(protocol, data, split, raw_cp_lines, raw_fitb, "component")
        return first, second
    if grouping == "user":
        mapping = json.loads((protocol / f"group_ids_{split}.json").read_text())
    elif grouping == "component":
        mapping = item_components(data.outfits(split))
    else:
        raise ValueError(f"Unknown grouping {grouping}")
    positives = [line.split()[1].split("_", 1)[0] for line in raw_cp_lines if line.startswith("1")]
    cp = [str(mapping[s]) for s in positives] * 2  # pair-aligned: positives then negatives
    fitb = [str(mapping[q["question"][0].split("_", 1)[0]]) for q in raw_fitb]
    return cp, fitb


def main():
    from repbench.data.polyvore import PolyvoreDisjoint
    from repbench.eval.paired_statistics import (
        cluster_bootstrap,
        delong_covariance,
        delong_pvalue,
        exact_mcnemar,
        pairwise_inference,
    )
    from repbench.v2.calibration_v2 import calibrate_and_evaluate
    from repbench.v2.logistic import GridEdgeError
    from repbench.v2.probe import run_probe
    from repbench.v2.rank_statistics import friedman_nemenyi, plot_critical_difference
    from repbench.v2.registry import V2_ENCODERS
    from repbench.v2.selection import select_models

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True,
                        help="comma list of name=protocol_dir=grouping (grouping: user|component)")
    parser.add_argument("--models", required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--settings", default="native,pca128,pca256,pca512")
    parser.add_argument("--learners", default="logistic,xgboost,mlp")
    parser.add_argument("--operators", default="abs_product,abs_product_sum")
    parser.add_argument("--eval-split", default="valid")
    parser.add_argument("--frozen-protocol-hash", type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    if args.eval_split == "test" and not args.frozen_protocol_hash:
        raise PermissionError("Test evaluation requires the frozen v2 protocol (G7) and Phase 2")
    models = args.models.split(",")
    results = args.results
    (results / "predictions").mkdir(parents=True, exist_ok=True)
    failures = []
    rows, pairwise_rows, rank_rows, equivalence_rows, calibration_rows = [], [], [], [], []
    selection, manifest = {}, {"started": time.time(), "inputs": {}}
    for spec in args.datasets.split(","):
        name, protocol, grouping = spec.split("=")
        protocol = Path(protocol)
        data = PolyvoreDisjoint(protocol.parent, protocol, name)
        train_pairs = cp_pairs(data.compatibility("train"))
        rng = random.Random(f"{args.seed}:{name}:fitvalid")
        fit_idx = set(rng.sample(range(len(train_pairs)), max(2, len(train_pairs) // 5)))
        flat = lambda pairs: [e for p in pairs for e in p]  # noqa: E731
        train_cp = flat(p for i, p in enumerate(train_pairs) if i not in fit_idx)
        fit_cp = flat(p for i, p in enumerate(train_pairs) if i in fit_idx)
        eval_examples = data.compatibility(args.eval_split)
        eval_cp = [e for e in eval_examples if e.label == 1] + [e for e in eval_examples if e.label == 0]
        eval_fitb = data.fitb(args.eval_split)
        raw_cp = [l for l in (protocol / f"compatibility_{args.eval_split}.txt").read_text().splitlines() if l.strip()]
        raw_fitb = json.loads((protocol / f"fill_in_blank_{args.eval_split}.json").read_text())
        grouped = groups_for(protocol, data, args.eval_split, raw_cp, raw_fitb, grouping)
        if grouping == "two_way":
            (cp_groups, fitb_groups), (cp_second, fitb_second) = grouped
        else:
            (cp_groups, fitb_groups), cp_second, fitb_second = grouped, None, None
        manifest["inputs"][name] = {
            "protocol": str(protocol), "grouping": grouping,
            "train_pairs": len(train_pairs) - len(fit_idx), "fit_valid_pairs": len(fit_idx),
            "eval_cp": len(eval_cp), "eval_fitb": len(eval_fitb),
            "cp_clusters": len(set(cp_groups)), "fitb_clusters": len(set(fitb_groups)),
        }
        outputs = {}
        for model in models:
            store, meta = load_store(args.embeddings, name, model)
            manifest["inputs"][f"{name}/{model}"] = meta["items_sha256"]
            for setting in args.settings.split(","):
                for learner in args.learners.split(","):
                    for operator in args.operators.split(","):
                        # Operator sensitivity only for the primary native/logistic setting.
                        if operator != "abs_product" and (setting, learner) != ("native", "logistic"):
                            continue
                        tag = setting if operator == "abs_product" else f"{setting}+{operator}"
                        path = results / "predictions" / f"{name}__{model}__{tag}__{learner}.json"
                        if path.exists():
                            out = json.loads(path.read_text())
                        else:
                            try:
                                out = run_probe(store, train_cp, fit_cp, eval_cp, eval_fitb,
                                                setting=setting, learner=learner,
                                                operator=operator, seed=args.seed)
                            except GridEdgeError as error:
                                # Recorded, never silently dropped: a pre-freeze grid decision.
                                out = {"error": "grid_edge", "detail": str(error)}
                            with path.open("x") as stream:
                                json.dump(out, stream)
                        if "error" in out:
                            failures.append(dict(dataset=name, model=model, setting=tag,
                                                 learner=learner, error=out["error"]))
                            continue
                        outputs[(model, tag, learner)] = out
                        for metric, value in (
                            ("cp_auc", None), ("fitb_acc", float(np.mean(out["fitb"]["correct"])))
                        ):
                            if metric == "cp_auc":
                                from sklearn.metrics import roc_auc_score

                                value = roc_auc_score(out["cp"]["labels"], out["cp"]["scores"])
                            rows.append(dict(dataset=name, model=model, setting=tag, learner=learner,
                                             split=args.eval_split, metric=metric, value=value,
                                             selection=json.dumps(out["selection"]["chosen"])))
                        if (tag, learner) == ("native", "logistic"):  # calibration on primary
                            fit_scores = run_probe(store, train_cp, fit_cp, fit_cp, None,
                                                   setting=setting, learner=learner,
                                                   seed=args.seed)["cp"]["scores"]
                            try:
                                calibration = calibrate_and_evaluate(
                                    [e.label for e in fit_cp], fit_scores, out["cp"]["labels"],
                                    out["cp"]["scores"], method="temperature",
                                    eval_groups=cp_groups,
                                    replicates=min(args.replicates, 1000), seed=args.seed)
                            except ValueError as error:
                                failures.append(dict(dataset=name, model=model, setting=tag,
                                                     learner=learner,
                                                     error=f"calibration: {error}"))
                                continue
                            for phase in ("before", "after"):
                                calibration_rows.append(dict(
                                    dataset=name, model=model, phase=phase,
                                    temperature=calibration["params"],
                                    **{k: v for k, v in calibration[phase].items()
                                       if isinstance(v, int | float)}))
        # Paired statistics per (setting, learner) across models.
        for tag, learner in sorted({(t, l) for (_, t, l) in outputs}):
            if any((m, tag, learner) not in outputs for m in models):
                continue  # incomplete cell (recorded in failures.csv)
            cp = np.column_stack([outputs[(m, tag, learner)]["cp"]["scores"] for m in models])
            labels = np.asarray(outputs[(models[0], tag, learner)]["cp"]["labels"])
            fitb = np.column_stack([outputs[(m, tag, learner)]["fitb"]["correct"] for m in models])
            for metric, matrix, groups, second in (("cp_auc", cp, cp_groups, cp_second),
                                                   ("fitb_acc", fitb, fitb_groups, fitb_second)):
                boot = cluster_bootstrap(matrix, metric="cp_auc" if metric == "cp_auc" else "accuracy",
                                         groups=groups, second_groups=second,
                                         labels=labels if metric == "cp_auc" else None,
                                         replicates=args.replicates, seed=args.seed)
                inference = pairwise_inference(boot, models, metric=metric, margins=MARGINS[metric])
                for estimate in inference["model_estimates"]:
                    for row in rows:
                        if (row["dataset"], row["model"], row["setting"], row["learner"], row["metric"]) == (
                                name, estimate["model"], tag, learner, metric):
                            row["ci_low"], row["ci_high"] = estimate["ci_low"], estimate["ci_high"]
                common = dict(dataset=name, setting=tag, learner=learner, metric=metric,
                              method=boot["method"], clusters=boot["cluster_count"])
                if metric == "cp_auc":
                    auc, cov = delong_covariance(labels, cp)
                for comparison in inference["comparisons"]:
                    i, j = models.index(comparison["left"]), models.index(comparison["right"])
                    extra = ({"delong_p": delong_pvalue(auc, cov, i, j)} if metric == "cp_auc"
                             else {"mcnemar_exact_p": exact_mcnemar(fitb[:, i], fitb[:, j])["pvalue"]})
                    pairwise_rows.append({**common, **comparison, **extra})
                rank_rows += [{**common, **r} for r in inference["rank_intervals"]]
                equivalence_rows += [{**common, **r} for r in inference["equivalence"]]
        native = [dict(dataset=name, split="valid", metric="cp_auc", setting="native",
                       model_key=r["model"], value=r["value"])
                  for r in rows if r["dataset"] == name and r["setting"] == "native"
                  and r["learner"] == "logistic" and r["metric"] == "cp_auc"]
        families = {m: V2_ENCODERS[m].family for m in models}
        try:
            selection[name] = select_models(
                [{**r, "dataset": name if name.endswith("_clean") else name + "_clean"} for r in native],
                dataset=name if name.endswith("_clean") else name + "_clean", families=families)
        except ValueError as error:  # e.g. dry-run roster lacks a fashion model
            selection[name] = {"status": "not selectable with this roster", "reason": str(error)}
    metrics = pd.DataFrame(rows)
    for frame, file in ((metrics, "metrics.csv"), (pd.DataFrame(pairwise_rows), "pairwise.csv"),
                        (pd.DataFrame(rank_rows), "rank_intervals.csv"),
                        (pd.DataFrame(equivalence_rows), "equivalence.csv"),
                        (pd.DataFrame(calibration_rows), "calibration.csv"),
                        (pd.DataFrame(failures, columns=["dataset", "model", "setting", "learner", "error"]),
                         "failures.csv")):
        frame.to_csv(results / file, index=False)
    blocks = metrics[metrics.metric == "cp_auc"].pivot_table(
        index=["dataset", "setting", "learner"], columns="model", values="value"
    ).reindex(columns=models).dropna()  # complete blocks only; gaps are in failures.csv
    if len(models) >= 3 and len(blocks) >= 2:
        friedman = friedman_nemenyi(blocks[models].to_numpy(), models,
                                    ["/".join(map(str, b)) for b in blocks.index])
        plot_critical_difference(friedman, results / "fig_critical_difference.pdf")
        (results / "friedman.json").write_text(json.dumps(friedman, indent=1, default=float))
    (results / "selection.json").write_text(json.dumps(selection, indent=1))
    manifest["seconds"] = time.time() - manifest.pop("started")
    manifest["eval_split"] = args.eval_split
    (results / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(rows)} metric rows; {len(pairwise_rows)} pairwise rows -> {results}")


if __name__ == "__main__":
    main()
