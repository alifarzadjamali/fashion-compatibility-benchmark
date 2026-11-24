"""A100 v2 evaluation (Phase 2 only): probes trained on each corpus score A100 questions.

A100 is an external human/expert test set: evaluated unchanged and never fitted,
tuned, calibrated or used for selection. Every command refuses to run unless the
frozen-protocol hash file (G7) exists and still matches the file it names.

  score      For each corpus x model, fit the probe on the corpus TRAIN split with the
             pipeline's seeded fit-valid carve-out (identical code path to
             scripts/v2/pipeline.py) and score every A100 LAT/AAT question.
             Writes <output>/predictions/<corpus>__<model>__<setting>__<learner>.json
             (exclusive create; completed files are re-verified and reused).
  summarize  Question-level bootstrap CIs, all-pairwise Holm, exact McNemar and TOST
             (2/5/10 pp) per training corpus x setting x learner; optional zero-shot
             MLLM runs (scripts/v2/mllm_zeroshot.py outputs) as their own family.
             Writes <output>/<family>/{metrics,pairwise,rank_intervals,equivalence}.csv
             (each family directory is a valid scripts/v2/make_tables_v2.py --results),
             questions.csv, and combined *_all.csv at <output>.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HASH = ROOT / "configs" / "v2_protocol_hash.json"


def guard(args):
    from repbench.v2.a100 import require_frozen_protocol

    return require_frozen_protocol(args.frozen_protocol_hash, root=ROOT)


def score(args):
    from repbench.data.polyvore import PolyvoreDisjoint
    from repbench.v2.a100 import all_item_ids, fitb_questions, load_questions, question_records
    from repbench.v2.logistic import GridEdgeError
    from repbench.v2.probe_scoring import (
        load_store_checked,
        merge_stores,
        score_with_fitted_probe,
        selection_matches,
    )
    from repbench.v2.state import sha256, write_json

    frozen = guard(args)
    questions, a100_manifest = load_questions(args.a100)
    fitb = fitb_questions(questions)
    a100_items = all_item_ids(questions)
    out = args.output
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    started = time.time()
    run = {"command": "score", "frozen_protocol": frozen, "a100": str(args.a100),
           "a100_manifest_sha256": sha256(args.a100 / "manifest.json"),
           "a100_archive_sha256": a100_manifest["archive_sha256"],
           "seed": args.seed, "settings": args.settings, "learners": args.learners,
           "operator": args.operator, "models": args.models, "corpora": {}, "outputs": {},
           "failures": []}
    for spec in args.corpora.split(","):
        name, protocol = spec.split("=")
        protocol = Path(protocol)
        data = PolyvoreDisjoint(protocol.parent, protocol, name)
        # Identical to the pipeline's Phase 2 rule: all TRAIN fits, VALID selects.
        from repbench.v2.sensitivities import carve_train, flatten

        train_pairs, fit_cp = carve_train(data, name, args.seed, args.selection_split)
        train_cp = flatten(train_pairs)
        carve = {"selection_split": args.selection_split, "train_pairs": len(train_pairs),
                 "fit_valid_pairs": len(fit_cp) // 2}
        run["corpora"][name] = {"protocol": str(protocol), **carve,
                                "train_sha256": sha256(protocol / "train.json")}
        for model in args.models.split(","):
            store, meta = load_store_checked(args.embeddings, name, model)
            a100_store, a100_meta = load_store_checked(args.embeddings, args.a100_dataset, model)
            merged = merge_stores(store, a100_store, a100_items,
                                  corpus_meta=meta, extra_meta=a100_meta)
            inputs = {"corpus_items_sha256": meta["items_sha256"],
                      "a100_items_sha256": a100_meta["items_sha256"],
                      "a100_manifest_sha256": run["a100_manifest_sha256"],
                      "train_sha256": run["corpora"][name]["train_sha256"], "seed": args.seed}
            for setting in args.settings.split(","):
                for learner in args.learners.split(","):
                    op = args.operator
                    tag = setting if op == "abs_product" else f"{setting}+{op}"
                    stem = f"{name}__{model}__{tag}__{learner}"
                    path = out / "predictions" / f"{stem}.json"
                    if path.exists():
                        previous = json.loads(path.read_text())
                        if previous["meta"]["inputs"] != inputs:
                            raise ValueError(f"{path} was produced from different inputs")
                        print(f"reuse {stem}", flush=True)
                        run["outputs"][stem] = sha256(path)
                        continue
                    try:
                        result = score_with_fitted_probe(
                            merged, train_cp, fit_cp, fitb, setting=setting, learner=learner,
                            operator=args.operator, seed=args.seed)
                    except GridEdgeError as error:
                        run["failures"].append({"stem": stem, "error": "grid_edge",
                                                "detail": str(error)})
                        continue
                    verified = None
                    if args.verify_pipeline_results:
                        reference = args.verify_pipeline_results / "predictions" / f"{stem}.json"
                        if reference.exists():
                            verified = selection_matches(result["selection"], reference)
                            if not verified:
                                raise ValueError(
                                    f"{stem}: refit selection differs from {reference}")
                    context = {"training_corpus": name, "model": model, "setting": tag,
                               "learner": learner}
                    records = question_records(questions, result["fitb"]["candidate_scores"],
                                               result["fitb"]["chosen_index"], context)
                    payload = {
                        "meta": {"inputs": inputs, **context, "operator": args.operator,
                                 "pipeline_selection_verified": verified,
                                 "embedding_model": meta.get("model"),
                                 "embedding_revision": meta.get("revision"),
                                 "created_utc": datetime.now(UTC).isoformat()},
                        "selection": result["selection"],
                        "probe": {k: result[k] for k in (
                            "item_dim_in", "item_dim_out", "feature_dim", "n_train_items",
                            "n_train_outfits", "n_fit_valid_outfits", "fit_seconds")},
                        "tie_rule": result["fitb"]["tie_rule"],
                        "records": records,
                    }
                    with path.open("x") as stream:
                        json.dump(payload, stream, allow_nan=False)
                    run["outputs"][stem] = sha256(path)
                    print(f"{stem}: C={result['selection']['chosen']} "
                          f"(selection verified: {verified})", flush=True)
    run["seconds"] = time.time() - started
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(out / "manifests" / f"score_{stamp}.json", run)
    print(f"{len(run['outputs'])} prediction files, {len(run['failures'])} failures -> {out}")
    return 0 if not run["failures"] else 2


def mllm_records(run_dir, questions):
    """Per-question records from a finished scripts/v2/mllm_zeroshot.py A100 run."""
    from repbench.v2.a100 import question_records
    from repbench.v2.mllm_zeroshot import LETTERS, read_results

    config = json.loads((run_dir / "config.json").read_text())
    if config["source"] != "a100":
        raise ValueError(f"{run_dir} is not an A100 run")
    if not (run_dir / "summary.json").exists():
        raise ValueError(f"{run_dir} is not finished (no summary.json)")
    task = config["split"].upper()
    by_id = {q.question_id: q for q in questions if q.task == task}
    results = read_results(run_dir / "results.jsonl")
    if sorted(r["question_id"] for r in results) != sorted(by_id):
        raise ValueError(f"{run_dir} does not cover all A100 {task} questions exactly once")
    ordered = [by_id[r["question_id"]] for r in results]
    scores, chosen = [], []
    for record, q in zip(results, ordered, strict=True):
        if int(record["correct_index"]) != q.primary_gt_index:
            raise ValueError(f"{q.question_id}: MLLM run used a different ground truth")
        per_original = [0.0] * len(q.candidate_items)
        for position, original in enumerate(record["permutation"]):
            per_original[original] = record["letter_logprobs"][LETTERS[position]]
        scores.append(per_original)
        chosen.append(int(record["predicted_index"]))
    from repbench.v2.registry import ZERO_SHOT

    if config["model"] != ZERO_SHOT.checkpoint:
        raise ValueError(f"{run_dir}: unexpected zero-shot model {config['model']}")
    context = {"training_corpus": "zero_shot", "model": ZERO_SHOT.model_key,
               "setting": "zero_shot", "learner": "letter_logprob"}
    return question_records(ordered, scores, chosen, context)


def summarize(args):
    from repbench.v2.a100 import load_questions, summarize_block
    from repbench.v2.state import sha256, write_json

    frozen = guard(args)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    questions, _ = load_questions(args.a100)
    expected = {q.question_id: q for q in questions}
    families = {}
    inputs = {}
    for path in sorted((args.scores / "predictions").glob("*.json")):
        payload = json.loads(path.read_text())
        records = payload["records"]
        if sorted(r["question_id"] for r in records) != sorted(expected):
            raise ValueError(f"{path} does not cover every A100 question exactly once")
        for r in records:
            q = expected[r["question_id"]]
            if (r["archive_gt_index"], r["primary_gt_index"]) != (q.archive_gt_index,
                                                                  q.primary_gt_index):
                raise ValueError(f"{path}: ground truth differs from the prepared A100")
        meta = payload["meta"]
        key = (meta["training_corpus"], meta["setting"], meta["learner"])
        families.setdefault(key, {})[meta["model"]] = records
        inputs[str(path)] = sha256(path)
    zero_shot = {}
    for run_dir in args.mllm or []:
        records = mllm_records(run_dir, questions)
        model = records[0]["model"]
        zero_shot.setdefault(model, []).extend(records)
        inputs[str(run_dir / "results.jsonl")] = sha256(run_dir / "results.jsonl")
    for model, records in zero_shot.items():
        families.setdefault(("zero_shot", "zero_shot", "letter_logprob"), {})[model] = records
    if not families:
        raise ValueError("No A100 predictions found")
    args.output.mkdir(parents=True)
    combined = {name: [] for name in ("metrics", "pairwise", "rank_intervals", "equivalence")}
    for (corpus, setting, learner), by_model in sorted(families.items()):
        models = sorted(by_model)
        context = {"setting": setting, "learner": learner, "split": f"trained_on_{corpus}",
                   "training_corpus": corpus,
                   "selection": "probe: corpus TRAIN fit, selection as recorded in the score files (Phase 2: VALID)"
                   if corpus != "zero_shot" else "none (zero-shot)"}
        blocks = summarize_block(by_model, models, context, replicates=args.replicates,
                                 seed=args.seed)
        folder = args.output / (corpus if corpus == "zero_shot"
                                else f"{corpus}__{setting}__{learner}")
        folder.mkdir()
        for name, rows in zip(combined, blocks, strict=True):
            pd.DataFrame(rows).to_csv(folder / f"{name}.csv", index=False)
            combined[name] += rows
        pd.DataFrame([r for m in models for r in by_model[m]]).assign(
            candidate_item_ids=lambda f: f.candidate_item_ids.map(json.dumps),
            candidate_scores=lambda f: f.candidate_scores.map(json.dumps),
        ).to_csv(folder / "questions.csv", index=False)
    for name, rows in combined.items():
        pd.DataFrame(rows).to_csv(args.output / f"{name}_all.csv", index=False)
    write_json(args.output / "manifest.json", {
        "command": "summarize", "frozen_protocol": frozen, "inputs": inputs,
        "a100_manifest_sha256": sha256(args.a100 / "manifest.json"),
        "replicates": args.replicates, "seed": args.seed,
        "clusters": "question (A100, HANDOFF_V2 §6.3)",
        "holm_family": "all model pairs within training corpus x setting x learner x "
                       "dataset x metric",
        "tost_margins": [0.02, 0.05, 0.10],
        "primary": "a100_lat acc_majority",
        "families": ["/".join(k) for k in sorted(families)],
        "created_utc": datetime.now(UTC).isoformat()})
    primary = pd.DataFrame(combined["metrics"])
    primary = primary[primary.metric.isin(["acc_majority", "acc"])]
    with np.printoptions(precision=3):
        print(primary[["training_corpus", "dataset", "model", "metric", "value"]]
              .to_string(index=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--a100", type=Path, required=True, help="prepare_a100_v2.py output")
    s.add_argument("--embeddings", type=Path, required=True,
                   help="root with <corpus>/<model>/ and <a100-dataset>/<model>/ stores")
    s.add_argument("--a100-dataset", default="a100")
    s.add_argument("--corpora", required=True, help="comma list of name=protocol_dir")
    s.add_argument("--models", required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--settings", default="native")
    s.add_argument("--learners", default="logistic")
    s.add_argument("--operator", default="abs_product")
    s.add_argument("--seed", type=int, default=20260912)
    s.add_argument("--selection-split", choices=["valid", "carveout"], default="valid",
                   help="valid = Phase 2 rule (must match the pipeline run being verified)")
    s.add_argument("--verify-pipeline-results", type=Path,
                   help="pipeline --results dir; refit selection must equal its predictions")
    s.add_argument("--frozen-protocol-hash", type=Path, default=DEFAULT_HASH)
    m = sub.add_parser("summarize")
    m.add_argument("--a100", type=Path, required=True)
    m.add_argument("--scores", type=Path, required=True, help="score --output dir")
    m.add_argument("--mllm", type=Path, nargs="*", help="finished mllm_zeroshot A100 run dirs")
    m.add_argument("--output", type=Path, required=True)
    m.add_argument("--replicates", type=int, default=2000)
    m.add_argument("--seed", type=int, default=20260912)
    m.add_argument("--frozen-protocol-hash", type=Path, default=DEFAULT_HASH)
    args = parser.parse_args(argv)
    return score(args) if args.command == "score" else summarize(args)


if __name__ == "__main__":
    sys.exit(main())
