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


def _load_one(folder):
    meta = json.loads((folder / "metadata.json").read_text())
    if hashlib.sha256((folder / "items.npy").read_bytes()).hexdigest() != meta["items_sha256"]:
        raise ValueError(f"Embedding checksum mismatch: {folder}")
    values = np.load(folder / "items.npy")
    ids = json.loads((folder / "item_ids.json").read_text())
    return ids, values, meta


def load_store(root, dataset, model):
    """One store, or the merge of its parts (e.g. trainvalid/ + test/); conflicts fail."""
    folder = root / dataset / model
    parts = [folder] if (folder / "metadata.json").exists() else sorted(
        p for p in folder.iterdir() if (p / "metadata.json").exists())
    if not parts:
        raise FileNotFoundError(f"No embedding store under {folder}")
    store, digests = {}, {}
    for part in parts:
        ids, values, meta = _load_one(part)
        digests[part.name] = meta["items_sha256"]
        for k, item in enumerate(ids):
            if item in store and not np.array_equal(store[item], values[k]):
                raise ValueError(f"Conflicting vectors for {item} across store parts")
            store[item] = values[k]
    combined = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
    return store, {"items_sha256": combined, "parts": digests}


def cp_pairs(examples):
    positive = [e for e in examples if e.label == 1]
    negative = [e for e in examples if e.label == 0]
    if len(positive) != len(negative):
        raise ValueError("Balanced paired CP construction expected")
    return list(zip(positive, negative, strict=True))


EQUIVALENCE = {}  # dataset name -> groups file (from --equivalence)


def equivalence_map(protocol, name):
    """v2-Clean near-duplicate groups merged by the split builder (decision E). The file
    must match the SHA-256 the protocol manifest recorded; v1-format protocols have none."""
    manifest = protocol / "manifest.json"
    info = json.loads(manifest.read_text()).get("image_equivalence") if manifest.exists() else None
    if not info:
        return {}
    path = EQUIVALENCE.get(name)
    if path is None:
        raise ValueError(f"{name}: protocol merged near-duplicates; pass --equivalence {name}=<groups>")
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError(f"{name}: equivalence file does not match the protocol manifest")
    data = json.loads(Path(path).read_text())
    return data.get("groups", data)


def item_components(outfits, equivalence=None):
    parent = {}
    equivalence = equivalence or {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for outfit in outfits:
        first = find(outfit.item_ids[0])
        for item in outfit.item_ids[1:]:
            parent[find(item)] = first
    for item, group in equivalence.items():  # near-duplicates are one dependence unit
        parent[find(item)] = find("eq:" + str(group))
    return {o.set_id: find(o.item_ids[0]) for o in outfits}


def groups_for(protocol, data, split, raw_cp_lines, raw_fitb, grouping):
    if grouping == "two_way":  # user x item-component (pigeonhole) bootstrap
        first = groups_for(protocol, data, split, raw_cp_lines, raw_fitb, "user")
        second = groups_for(protocol, data, split, raw_cp_lines, raw_fitb, "component")
        return first, second
    if grouping == "user":
        mapping = json.loads((protocol / f"group_ids_{split}.json").read_text())
    elif grouping == "component":
        mapping = item_components(data.outfits(split), equivalence_map(protocol, data.dataset_name))
    else:
        raise ValueError(f"Unknown grouping {grouping}")
    positives = [line.split()[1].split("_", 1)[0] for line in raw_cp_lines if line.startswith("1")]
    cp = [str(mapping[s]) for s in positives] * 2  # pair-aligned: positives then negatives
    fitb = [str(mapping[q["question"][0].split("_", 1)[0]]) for q in raw_fitb]
    return cp, fitb


_CTX = {}


def _cell(job):
    """Fit one probe cell in a forked worker (store shared copy-on-write) and save it."""
    from repbench.v2.logistic import GridEdgeError
    from repbench.v2.probe import run_probe

    setting, learner, operator, path, fit_only = job
    c = _CTX
    try:
        if fit_only:  # calibration needs fit-valid scores of the primary probe
            out = run_probe(c["store"], c["train"], c["fit"], c["fit"], None, setting=setting,
                            learner=learner, operator=operator, seed=c["seed"])
        else:
            out = run_probe(c["store"], c["train"], c["fit"], c["eval_cp"], c["eval_fitb"],
                            setting=setting, learner=learner, operator=operator, seed=c["seed"])
    except GridEdgeError as error:
        out = {"error": "grid_edge", "detail": str(error)}
    out["provenance"] = c["provenance"]
    with open(path, "x") as stream:
        json.dump(out, stream)
    return str(path)


def precompute(results, name, model, store, train_cp, fit_cp, eval_cp, eval_fitb, args):
    import multiprocessing as mp

    jobs = []
    for setting in args.settings.split(","):
        for learner in args.learners.split(","):
            for operator in args.operators.split(","):
                if operator != "abs_product" and (setting, learner) != ("native", "logistic"):
                    continue
                tag = setting if operator == "abs_product" else f"{setting}+{operator}"
                path = results / "predictions" / f"{name}__{model}__{tag}__{learner}.json"
                if not path.exists():
                    jobs.append((setting, learner, operator, path, False))
    fit_path = results / "predictions" / f"{name}__{model}__native__logistic__fitvalid.json"
    if "native" in args.settings.split(",") and not fit_path.exists():
        jobs.append(("native", "logistic", "abs_product", fit_path, True))
    if not jobs:
        return
    _CTX.update(store=store, train=train_cp, fit=fit_cp, eval_cp=eval_cp, eval_fitb=eval_fitb,
                seed=args.seed, provenance=args.provenance)
    if args.workers <= 1:
        for job in jobs:
            _cell(job)
    else:
        with mp.get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
            for done in pool.imap_unordered(_cell, jobs):
                print(f"done {Path(done).name}", flush=True)
    _CTX.clear()


def main():
    from repbench.data.polyvore import PolyvoreDisjoint
    from repbench.eval.paired_statistics import (
        cluster_adjusted_mcnemar,
        cluster_bootstrap,
        delong_covariance,
        delong_pvalue,
        exact_mcnemar,
        pairwise_inference,
    )
    from repbench.v2.calibration_v2 import calibrate_and_evaluate
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
    parser.add_argument("--stage", choices=["probes", "stats", "all"], default="all",
                        help="probes: fit missing cells only; stats: no fitting (all cells must exist)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--equivalence", nargs="*", default=[],
                        help="name=equivalence_groups.json for v2-Clean protocols (verified)")
    parser.add_argument("--selection-split", choices=["carveout", "valid"],
                        help="default: valid when evaluating test (Phase 2), carve-out otherwise")
    args = parser.parse_args()
    if args.eval_split == "test":
        if not args.frozen_protocol_hash or not args.frozen_protocol_hash.exists():
            raise PermissionError("Test evaluation requires the frozen v2 protocol (G7) and Phase 2")
        record = json.loads(args.frozen_protocol_hash.read_text())
        if hashlib.sha256(Path(record["file"]).read_bytes()).hexdigest() != record["sha256"]:
            raise PermissionError("Frozen protocol file changed since freeze; refusing")
    EQUIVALENCE.update(dict(item.split("=", 1) for item in args.equivalence))
    selection_split = args.selection_split or ("valid" if args.eval_split == "test" else "carveout")
    if selection_split == "valid" and args.eval_split == "valid":
        raise ValueError("Cannot select on the split being evaluated")
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
        flat = lambda pairs: [e for p in pairs for e in p]
        if selection_split == "valid":  # Phase 2: fit train, select/calibrate valid, eval test
            fit_idx = set()
            train_cp = flat(train_pairs)
            fit_cp = flat(cp_pairs(data.compatibility("valid")))
        else:  # dry run: valid stands in for test, so select on a seeded train carve-out
            rng = random.Random(f"{args.seed}:{name}:fitvalid")
            fit_idx = set(rng.sample(range(len(train_pairs)), max(2, len(train_pairs) // 5)))
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
            "selection_split": selection_split,
            "train_pairs": len(train_cp) // 2, "fit_valid_pairs": len(fit_cp) // 2,
            "eval_cp": len(eval_cp), "eval_fitb": len(eval_fitb),
            "cp_clusters": len(set(cp_groups)), "fitb_clusters": len(set(fitb_groups)),
        }
        outputs = {}
        for model in models:
            store, meta = load_store(args.embeddings, name, model)
            manifest["inputs"][f"{name}/{model}"] = meta["items_sha256"]
            args.provenance = {"eval_split": args.eval_split, "selection_split": selection_split,
                               "items_sha256": meta["items_sha256"]}
            if args.stage != "stats":
                precompute(results, name, model, store, train_cp, fit_cp, eval_cp, eval_fitb, args)
            if args.stage == "probes":
                continue
            for setting in args.settings.split(","):
                for learner in args.learners.split(","):
                    for operator in args.operators.split(","):
                        # Operator sensitivity only for the primary native/logistic setting.
                        if operator != "abs_product" and (setting, learner) != ("native", "logistic"):
                            continue
                        tag = setting if operator == "abs_product" else f"{setting}+{operator}"
                        path = results / "predictions" / f"{name}__{model}__{tag}__{learner}.json"
                        if not path.exists():
                            raise FileNotFoundError(f"Missing probe cell {path.name}; run --stage probes")
                        out = json.loads(path.read_text())
                        if out.get("provenance") and out["provenance"] != args.provenance:
                            raise ValueError(f"{path.name} was fitted on other inputs: "
                                             f"{out['provenance']} != {args.provenance}")
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
                                             selection=json.dumps(out["selection"]["chosen"]),
                                             validation_auc=out["selection"]["validation_auc"]))
                        if (tag, learner) == ("native", "logistic"):  # calibration on primary
                            fit_out = json.loads((results / "predictions" /
                                                  f"{name}__{model}__native__logistic__fitvalid.json").read_text())
                            if "error" in fit_out:
                                continue
                            fit_scores = fit_out["cp"]["scores"]
                            for method in ("temperature", "platt"):
                                try:
                                    calibration = calibrate_and_evaluate(
                                        [e.label for e in fit_cp], fit_scores, out["cp"]["labels"],
                                        out["cp"]["scores"], method=method,
                                        eval_groups=cp_groups,
                                        replicates=min(args.replicates, 1000), seed=args.seed)
                                except ValueError as error:
                                    failures.append(dict(dataset=name, model=model, setting=tag,
                                                         learner=learner,
                                                         error=f"calibration {method}: {error}"))
                                    continue
                                for phase in ("before", "after"):
                                    calibration_rows.append(dict(
                                        dataset=name, model=model, method=method, phase=phase,
                                        params=json.dumps(calibration["params"], default=float),
                                        **{k: v for k, v in calibration[phase].items()
                                           if isinstance(v, int | float)}))
        if args.stage == "probes":
            continue
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
                inference = pairwise_inference(boot, models, metric="cp_auc" if metric == "cp_auc" else "accuracy",
                                               margins=MARGINS[metric])
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
                             else {"mcnemar_exact_p": exact_mcnemar(fitb[:, i], fitb[:, j])["pvalue"],
                                   "mcnemar_cluster_p": cluster_adjusted_mcnemar(
                                       fitb[:, i], fitb[:, j], groups,
                                       replicates=args.replicates, seed=args.seed)["pvalue"]})
                    pairwise_rows.append({**common, **comparison, **extra})
                rank_rows += [{**common, **r} for r in inference["rank_intervals"]]
                equivalence_rows += [{**common, **r} for r in inference["equivalence"]]
        native = [dict(dataset=name, split="valid", metric="cp_auc", setting="native",
                       model_key=r["model"], value=r["validation_auc"])  # never eval-split scores
                  for r in rows if r["dataset"] == name and r["setting"] == "native"
                  and r["learner"] == "logistic" and r["metric"] == "cp_auc"]
        families = {m: V2_ENCODERS[m].family for m in models}
        try:
            selection[name] = select_models(
                [{**r, "dataset": name if name.endswith("_clean") else name + "_clean"} for r in native],
                dataset=name if name.endswith("_clean") else name + "_clean", families=families)
        except ValueError as error:  # e.g. dry-run roster lacks a fashion model
            selection[name] = {"status": "not selectable with this roster", "reason": str(error)}
    if args.stage == "probes":
        print(f"probe cells complete for {models}")
        return
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
