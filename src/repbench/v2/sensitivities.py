"""v2 versions of the v1 strength analyses (HANDOFF_V2 §2, §6.3, §6.8, §6.9).

Every analysis reads v2 pipeline outputs (embedding stores, Polyvore-format protocols,
``predictions/*.json`` and ``metrics.csv`` of ``scripts/v2/pipeline.py``):

* label fractions  -- primary probe on nested seeded subsets of TRAIN CP pairs;
* construction seeds -- CP negatives / FITB candidates regenerated on fixed partitions
  (seeds 42-46), collision audit, primary probe, total uncertainty;
* provenance        -- pairwise / rank statistics without the provenance-uncertain group;
* subgroups         -- outfit-length / category subgroups and failure consensus;
* near duplicates   -- primary metrics excluding near-duplicate-affected test outfits;
* LookBench         -- rank correlation of published LookBench v3 scores with CP AUC.

No function chooses the evaluation split: callers pass it explicitly (the CLI applies
the same frozen-protocol guard as the pipeline). Outputs never overwrite existing files.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from repbench.data.clean_protocol import SPLITS, file_sha256, generate_questions
from repbench.data.polyvore import PolyvoreDisjoint
from repbench.eval.paired_statistics import cluster_bootstrap, pairwise_inference

ROOT = Path(__file__).resolve().parents[3]
PRIMARY = ("native", "logistic")
LABEL_FRACTIONS = (0.1, 0.25, 0.5, 1.0)
FRACTION_SEED = 20260912
PROBE_SEED = 20260912
CONSTRUCTION_SEEDS = (42, 43, 44, 45, 46)
PROVENANCE_UNCERTAIN = ("marqo_fashionsiglip", "marqo_fashionclip", "gr_lite",
                        "qwen3_vl_emb_2b", "qwen3_vl_emb_8b")
METRIC_COLUMNS = ["dataset", "model", "setting", "learner", "split", "metric", "value",
                  "ci_low", "ci_high"]
BOOTSTRAP_METRIC = {"cp_auc": "cp_auc", "fitb_acc": "accuracy"}
CP_CORRECT_THRESHOLD = 0.5  # v1 failure-consensus rule for CP correctness


# --------------------------------------------------------------------------- utilities
@lru_cache(maxsize=1)
def pipeline():
    """``scripts/v2/pipeline.py`` as a module (helpers cp_pairs, groups_for, MARGINS)."""
    path = ROOT / "scripts" / "v2" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("repbench_v2_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_new_csv(path, frame):
    """Write a CSV, refusing to overwrite a completed output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        pd.DataFrame(frame).to_csv(stream, index=False)
    return path


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=1, default=_json_default)
    return path


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


def check_eval_split(split, frozen_protocol_hash=None):
    """Same guard as the pipeline: test evaluation needs the frozen protocol hash file."""
    if split not in SPLITS:
        raise ValueError(f"eval split must be one of {SPLITS}")
    if split != "test":
        return
    if not frozen_protocol_hash or not Path(frozen_protocol_hash).exists():
        raise PermissionError("Test evaluation requires the frozen v2 protocol (G7) and Phase 2")
    record = json.loads(Path(frozen_protocol_hash).read_text())
    if hashlib.sha256(Path(record["file"]).read_bytes()).hexdigest() != record["sha256"]:
        raise PermissionError("Frozen protocol file changed since freeze; refusing")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    protocol: Path
    grouping: str


def parse_dataset_specs(text):
    """``name=protocol_dir=grouping`` comma list (the pipeline's --datasets format)."""
    specs = []
    for spec in text.split(","):
        name, protocol, grouping = spec.split("=")
        specs.append(DatasetSpec(name, Path(protocol), grouping))
    return specs


def specs_from_manifest(results):
    manifest = json.loads((Path(results) / "manifest.json").read_text())
    return [DatasetSpec(name, Path(v["protocol"]), v["grouping"])
            for name, v in manifest["inputs"].items() if isinstance(v, dict) and "protocol" in v]


def manifest_split(results):
    path = Path(results) / "manifest.json"
    return json.loads(path.read_text()).get("eval_split") if path.exists() else None


# --------------------------------------------------------------------------- eval data
@dataclass
class EvalData:
    spec: DatasetSpec
    split: str
    data: PolyvoreDisjoint
    cp: list               # positives then negatives (pipeline order)
    fitb: list
    cp_set_ids: list       # pair-aligned positive set_id for every CP example
    fitb_set_ids: list
    cp_groups: list
    fitb_groups: list
    cp_second: list | None
    fitb_second: list | None
    raw_cp: list
    raw_fitb: list

    @property
    def labels(self):
        return np.asarray([e.label for e in self.cp])


def load_eval(spec, split):
    """Evaluation examples and bootstrap clusters exactly as scripts/v2/pipeline.py builds them."""
    pipe = pipeline()
    data = PolyvoreDisjoint(spec.protocol.parent, spec.protocol, spec.name)
    examples = data.compatibility(split)
    cp = [e for e in examples if e.label == 1] + [e for e in examples if e.label == 0]
    fitb = data.fitb(split)
    raw_cp = [line for line in (spec.protocol / f"compatibility_{split}.txt").read_text().splitlines()
              if line.strip()]
    raw_fitb = json.loads((spec.protocol / f"fill_in_blank_{split}.json").read_text())
    grouped = pipe.groups_for(spec.protocol, data, split, raw_cp, raw_fitb, spec.grouping)
    if spec.grouping == "two_way":
        (cp_groups, fitb_groups), (cp_second, fitb_second) = grouped
    else:
        (cp_groups, fitb_groups), cp_second, fitb_second = grouped, None, None
    positives = [line.split()[1].split("_", 1)[0] for line in raw_cp if line.startswith("1")]
    return EvalData(spec, split, data, cp, fitb, positives * 2,
                    [q["question"][0].split("_", 1)[0] for q in raw_fitb],
                    cp_groups, fitb_groups, cp_second, fitb_second, raw_cp, raw_fitb)


def default_selection_split(split):
    """The pipeline's rule: Phase 2 (test) selects on valid; otherwise on the carve-out."""
    return "valid" if split == "test" else "carveout"


def carve_train(data, name, seed=PROBE_SEED, selection_split="carveout"):
    """Replicates the pipeline's training / selection data.

    ``carveout`` (pre-freeze): seeded 20% of TRAIN CP pairs is the fit-valid set;
    ``valid`` (Phase 2): all TRAIN pairs train, the valid split's CP pairs select.
    Returns (training pairs in original order, fit-valid examples)."""
    pipe = pipeline()
    train_pairs = pipe.cp_pairs(data.compatibility("train"))
    if selection_split == "valid":
        return train_pairs, flatten(pipe.cp_pairs(data.compatibility("valid")))
    if selection_split != "carveout":
        raise ValueError(f"Unknown selection split {selection_split!r}")
    rng = random.Random(f"{seed}:{name}:fitvalid")
    fit_idx = set(rng.sample(range(len(train_pairs)), max(2, len(train_pairs) // 5)))
    remaining = [p for i, p in enumerate(train_pairs) if i not in fit_idx]
    fit_cp = [e for i, p in enumerate(train_pairs) if i in fit_idx for e in p]
    return remaining, fit_cp


def flatten(pairs):
    return [e for p in pairs for e in p]


def nested_fraction_indices(n_pairs, fractions, seed=FRACTION_SEED):
    """Nested subsets: one seeded permutation, each fraction takes a prefix (kept in the
    original pair order); fraction 1.0 is exactly the full TRAIN carve-out."""
    order = np.random.default_rng(seed).permutation(n_pairs)
    subsets = {}
    for fraction in sorted(fractions):
        if not 0 < fraction <= 1:
            raise ValueError(f"Label fraction must lie in (0, 1]: {fraction}")
        count = n_pairs if fraction == 1 else max(2, math.ceil(fraction * n_pairs))
        subsets[fraction] = sorted(int(i) for i in order[:count])
    return subsets


# ----------------------------------------------------------------------- predictions
def metric_values(out):
    return {"cp_auc": float(roc_auc_score(out["cp"]["labels"], out["cp"]["scores"])),
            "fitb_acc": float(np.mean(out["fitb"]["correct"]))}


def check_alignment(out, ev, where):
    if out.get("cp") is None or out.get("fitb") is None:
        raise ValueError(f"{where}: prediction lacks CP or FITB output")
    if not np.array_equal(np.asarray(out["cp"]["labels"]), ev.labels):
        raise ValueError(f"{where}: CP predictions do not align with {ev.spec.name}/{ev.split}")
    if [int(q.correct_index) for q in ev.fitb] != list(out["fitb"]["correct_index"]):
        raise ValueError(f"{where}: FITB predictions do not align with {ev.spec.name}/{ev.split}")


def load_predictions(results, dataset, tag="native", learner="logistic", models=None):
    """{model: run_probe dict} from ``predictions/<dataset>__<model>__<tag>__<learner>.json``."""
    found = {}
    for path in sorted((Path(results) / "predictions").glob(f"{dataset}__*__{tag}__{learner}.json")):
        parts = path.stem.split("__")
        if len(parts) != 4 or parts[0] != dataset or parts[2] != tag or parts[3] != learner:
            continue
        if models is not None and parts[1] not in models:
            continue
        out = json.loads(path.read_text())
        if "error" not in out:
            found[parts[1]] = out
    return found


def prediction_cells(results, dataset):
    """Sorted set of (tag, learner) cells present for ``dataset``."""
    cells = set()
    for path in (Path(results) / "predictions").glob(f"{dataset}__*.json"):
        parts = path.stem.split("__")
        if len(parts) == 4 and parts[0] == dataset:
            cells.add((parts[2], parts[3]))
    return sorted(cells)


def matrices(outputs, models):
    cp = np.column_stack([outputs[m]["cp"]["scores"] for m in models])
    fitb = np.column_stack([outputs[m]["fitb"]["correct"] for m in models])
    return cp, fitb


def _subset(values, mask):
    return None if values is None else [v for v, keep in zip(values, mask, strict=True) if keep]


def bootstrap(matrix, metric, labels, groups, second, replicates, seed):
    """cluster_bootstrap wrapper; returns (result or None, reason)."""
    if replicates < 2:
        return None, "bootstrap disabled"
    try:
        return cluster_bootstrap(matrix, metric=BOOTSTRAP_METRIC[metric], groups=groups,
                                 second_groups=second,
                                 labels=labels if metric == "cp_auc" else None,
                                 replicates=replicates, seed=seed), ""
    except ValueError as error:
        return None, str(error)


def metric_rows(outputs, models, ev, *, setting, learner, replicates, seed,
                cp_mask=None, fitb_mask=None, extra=None):
    """Point estimates + joint cluster-bootstrap CIs (pipeline schema) on optional subsets.

    Returns (rows, {metric: bootstrap or None})."""
    cp, fitb = matrices(outputs, models)
    labels = ev.labels
    cp_mask = np.ones(len(labels), bool) if cp_mask is None else np.asarray(cp_mask, bool)
    fitb_mask = np.ones(len(fitb), bool) if fitb_mask is None else np.asarray(fitb_mask, bool)
    rows, boots = [], {}
    blocks = (
        ("cp_auc", cp[cp_mask], labels[cp_mask], _subset(ev.cp_groups, cp_mask),
         _subset(ev.cp_second, cp_mask), int(cp_mask.sum())),
        ("fitb_acc", fitb[fitb_mask], None, _subset(ev.fitb_groups, fitb_mask),
         _subset(ev.fitb_second, fitb_mask), int(fitb_mask.sum())),
    )
    for metric, matrix, y, groups, second, n in blocks:
        valid = n > 0 and (metric != "cp_auc" or len(np.unique(y)) == 2)
        boot, reason = (bootstrap(matrix, metric, y, groups, second, replicates, seed)
                        if valid else (None, "empty or single-class subset"))
        boots[metric] = boot
        for k, model in enumerate(models):
            if not valid:
                value = float("nan")
            elif metric == "cp_auc":
                value = float(roc_auc_score(y, matrix[:, k]))
            else:
                value = float(matrix[:, k].mean())
            low = high = float("nan")
            if boot is not None:
                low, high = (float(q) for q in np.quantile(boot["samples"][:, k], [0.025, 0.975]))
            rows.append(dict(dataset=ev.spec.name, model=model, setting=setting, learner=learner,
                             split=ev.split, metric=metric, value=value, ci_low=low, ci_high=high,
                             n=n, clusters=None if boot is None else boot["cluster_count"],
                             bootstrap_note=reason or (boot["method"] if boot else ""),
                             **(extra or {})))
    return rows, boots


def inference_rows(boots, models, common):
    """pairwise / rank / equivalence rows from joint bootstraps (pipeline format)."""
    margins = pipeline().MARGINS
    pairwise, ranks, equivalence = [], [], []
    for metric, boot in boots.items():
        if boot is None or len(models) < 2:
            continue
        inference = pairwise_inference(boot, models, metric=BOOTSTRAP_METRIC[metric],
                                       margins=margins[metric])
        base = {**common, "metric": metric, "method": boot["method"],
                "clusters": boot["cluster_count"]}
        pairwise += [{**base, **r} for r in inference["comparisons"]]
        ranks += [{**base, **r} for r in inference["rank_intervals"]]
        equivalence += [{**base, **r} for r in inference["equivalence"]]
    return pairwise, ranks, equivalence


# --------------------------------------------------------------- 1. label fractions
def _probe(store, train_cp, fit_cp, ev, seed):
    from repbench.v2.logistic import GridEdgeError
    from repbench.v2.probe import run_probe

    try:
        return run_probe(store, train_cp, fit_cp, ev.cp, ev.fitb, setting=PRIMARY[0],
                         learner=PRIMARY[1], operator="abs_product", seed=seed)
    except GridEdgeError as error:
        return {"error": "grid_edge", "detail": str(error)}


def _cached_probe(path, compute):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    out = compute()
    write_new_json(path, out)
    return out


def run_label_fractions(specs, models, embeddings, out, *, split, fractions=LABEL_FRACTIONS,
                        fraction_seed=FRACTION_SEED, seed=PROBE_SEED, replicates=2000,
                        selection_split=None):
    """Primary probe (native/logistic) on nested TRAIN-pair fractions; the selection set
    (fit-valid carve-out, or valid in Phase 2) is unchanged across fractions."""
    selection_split = selection_split or default_selection_split(split)
    from repbench.v2.stores import load_model_store

    out = Path(out)
    rows, failures, manifest = [], [], {"split": split, "fractions": list(fractions),
                                        "fraction_seed": fraction_seed, "probe_seed": seed,
                                        "selection_split": selection_split, "inputs": {}}
    for spec in specs:
        ev = load_eval(spec, split)
        remaining, fit_cp = carve_train(ev.data, spec.name, seed, selection_split)
        subsets = nested_fraction_indices(len(remaining), fractions, fraction_seed)
        manifest["inputs"][spec.name] = {
            "protocol": str(spec.protocol), "grouping": spec.grouping,
            "train_pairs": len(remaining), "fit_valid_pairs": len(fit_cp) // 2,
            "subset_pairs": {str(f): len(v) for f, v in subsets.items()},
            "subset_sha256": {str(f): hashlib.sha256(json.dumps(v).encode()).hexdigest()
                              for f, v in subsets.items()},
        }
        stores = {}
        for fraction, indices in subsets.items():
            outputs = {}
            for model in models:
                path = out / "predictions" / f"{spec.name}__{model}__f{fraction:g}__logistic.json"

                def compute(model=model, indices=indices, stores=stores, spec=spec,
                            remaining=remaining, fit_cp=fit_cp, ev=ev):
                    if model not in stores:
                        stores.clear()  # keep one store in memory at a time
                        stores[model], provenance = load_model_store(embeddings, spec.name, model)
                        manifest["inputs"][f"{spec.name}/{model}"] = provenance
                    train_cp = flatten(remaining[i] for i in indices)
                    return _probe(stores[model], train_cp, fit_cp, ev, seed)

                result = _cached_probe(path, compute)
                if "error" in result:
                    failures.append(dict(dataset=spec.name, model=model, fraction=fraction,
                                         error=result["error"]))
                    continue
                check_alignment(result, ev, path.name)
                outputs[model] = result
            present = [m for m in models if m in outputs]
            if not present:
                continue
            frame, _ = metric_rows(outputs, present, ev, setting="native", learner="logistic",
                                   replicates=replicates, seed=seed,
                                   extra={"fraction": fraction, "train_pairs": len(indices)})
            for row in frame:
                row["selected_c"] = outputs[row["model"]]["selection"]["chosen"].get("c")
            rows += frame
    write_new_csv(out / "metrics.csv", rows)
    write_new_csv(out / "failures.csv",
                  pd.DataFrame(failures, columns=["dataset", "model", "fraction", "error"]))
    write_new_json(out / "manifest.json", manifest)
    return pd.DataFrame(rows)


# ------------------------------------------------------------- 2. construction seeds
def _protocol_rows(protocol):
    return {s: json.loads((Path(protocol) / f"{s}.json").read_text(encoding="utf-8"))
            for s in SPLITS if (Path(protocol) / f"{s}.json").is_file()}


def protocol_categories(protocol, config=None):
    """item_id -> semantic category from the rows, the protocol's
    ``polyvore_item_metadata.json`` or ``config['categories_file']`` / raw root metadata.
    Returns None if unavailable."""
    config = config or {}
    rows = _protocol_rows(protocol)
    items = [item for values in rows.values() for row in values for item in row["items"]]
    if items and all("semantic_category" in item for item in items):
        return {str(i["item_id"]): str(i["semantic_category"]).strip() for i in items}
    candidates = [Path(protocol) / "polyvore_item_metadata.json"]
    if config.get("categories_file"):
        candidates.insert(0, Path(config["categories_file"]))
    if config.get("raw_root"):
        candidates.append(Path(config["raw_root"]) / "polyvore_item_metadata.json")
    for path in candidates:
        if path.is_file():
            metadata = json.loads(path.read_text(encoding="utf-8"))
            ids = {str(i["item_id"]) for i in items}
            if ids <= set(metadata):
                return {i: str(metadata[i]["semantic_category"]).strip() for i in ids}
    return None


def _union_find_identity(items, groups):
    parent = {i: i for i in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for group in groups:
        members = [i for i in group if i in parent]
        for other in members[1:]:
            a, b = find(members[0]), find(other)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return {i: find(i) for i in items}


def question_identity(protocol, config=None):
    """item_id -> image identity used by generate_questions for candidate exclusion.

    ``config['identity']`` (default: ``metadata`` when the protocol ships
    ``polyvore_item_metadata.json`` with ``image_identity`` — POG — else required):
      metadata      protocol polyvore_item_metadata.json ``image_identity`` (POG builder);
      file          JSON item -> identity at ``config['identity_file']``;
      polyvore_raw  v2-Clean Polyvore builder rule: exact-byte duplicate groups under
                    ``raw_root/images`` joined with ``image_equivalence`` groups;
      iqon_catalog  IQON builder rule: decoded-RGB SHA-256 from
                    ``catalog/image_audit_resolved.parquet`` merged with
                    ``image_equivalence`` (merge over the catalog pool);
      item          every item is its own identity (v1-Clean Polyvore builder behaviour).
    Only the induced partition matters; regeneration is verified against the base build.
    """
    config = config or {}
    rows = _protocol_rows(protocol)
    items = sorted({str(i["item_id"]) for v in rows.values() for r in v for i in r["items"]})
    metadata_path = Path(protocol) / "polyvore_item_metadata.json"
    kind = config.get("identity")
    if kind is None:
        if metadata_path.is_file():
            sample = json.loads(metadata_path.read_text(encoding="utf-8"))
            if items and all("image_identity" in sample.get(i, {}) for i in items):
                kind = "metadata"
        if kind is None:
            raise ValueError(f"{protocol}: no persisted image identity; set an identity source "
                             "(file | polyvore_raw | iqon_catalog | item) in the dataset config")
    if kind == "metadata":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {i: str(metadata[i]["image_identity"]) for i in items}, kind
    if kind == "file":
        mapping = json.loads(Path(config["identity_file"]).read_text(encoding="utf-8"))
        return {i: str(mapping[i]) for i in items}, kind
    if kind == "item":
        return {i: i for i in items}, kind
    from repbench.data.image_equivalence import equivalence_groups, load_image_equivalence

    equivalence = (load_image_equivalence(Path(config["image_equivalence"]))[0]
                   if config.get("image_equivalence") else None)
    if kind == "polyvore_raw":
        from repbench.data.clean_protocol import exact_file_duplicate_groups

        if equivalence is None:
            raise ValueError("polyvore_raw identity reproduces the v2-Clean builder and needs "
                             "image_equivalence (v1-Clean used per-item identity: use 'item')")
        groups = exact_file_duplicate_groups(Path(config["raw_root"]), set(items))
        groups += equivalence_groups(equivalence, items)
        return _union_find_identity(items, groups), kind
    if kind == "iqon_catalog":
        from repbench.data.image_equivalence import merge_identities

        catalog = Path(config["catalog"])
        images = pd.read_parquet(catalog / "image_audit_resolved.parquet",
                                 columns=["item_id", "decoded_rgb_sha256"])
        images["item_id"] = images.item_id.astype(str)
        decoded = images.set_index("item_id").decoded_rgb_sha256.to_dict()
        pool = set(items)
        if (catalog / "outfits.parquet").is_file():
            outfits = pd.read_parquet(catalog / "outfits.parquet", columns=["item_ids"])
            pool |= {str(i) for values in outfits.item_ids for i in values if str(i) in decoded}
        identity = {i: str(decoded[i]) for i in pool}
        if equivalence is not None:
            identity = {**identity, **merge_identities(identity, equivalence)}
        return {i: identity[i] for i in items}, kind
    raise ValueError(f"Unknown identity source {kind!r}")


def base_question_seed(manifest, split, override=None):
    index = SPLITS.index(split)
    if override is not None:
        return int(override) + 10_000 * (index + 1)
    if "question_seeds" in manifest:
        return int(manifest["question_seeds"][split])
    for key in ("construction_seed", "seed"):
        if key in manifest:
            return int(manifest[key]) + 10_000 * (index + 1)
    return None


def _generate(rows, categories, identity, seed):
    split_items = {str(i["item_id"]) for r in rows for i in r["items"]}
    metadata = {i: {"semantic_category": categories[i]} for i in split_items}
    return generate_questions(rows, metadata, seed, {i: identity[i] for i in split_items})


def audit_questions(rows, cp_lines, fitb_rows, identity):
    """Model-free collision audit of one split's questions (v1 audit_question_construction
    with the v2 image identity). ``passes`` covers the rules the generator guarantees."""
    key_to_id = {f"{r['set_id']}_{int(i['index'])}": str(i["item_id"]) for r in rows for i in r["items"]}
    split_items = set(key_to_id.values())
    positives = {tuple(sorted(str(i["item_id"]) for i in r["items"])) for r in rows}
    positive_images = {tuple(sorted(identity[i] for i in p)) for p in positives}
    parsed = [line.split() for line in cp_lines if line.strip()]
    negatives = [tuple(sorted(key_to_id[k] for k in line[1:])) for line in parsed if line[0] == "0"]
    n_pos = sum(line[0] == "1" for line in parsed)
    off_split = sum(any(k not in key_to_id for k in line[1:]) for line in parsed)
    known, dup_ids, dup_images, distractor_positive, usage = 0, 0, 0, 0, Counter()
    for row in fitb_rows:
        question = [key_to_id[k] for k in row["question"]]
        answers = [key_to_id[k] for k in row["answers"]]
        completes = [tuple(sorted(question + [a])) in positives for a in answers]
        known += sum(completes) == 1
        dup_ids += len(set(answers)) != len(answers)
        truth = row["question"][0].split("_", 1)[0]
        # Distractors: distinct identities, none equal to the answer or a question item.
        distractor_ids = [identity[a] for a, k in zip(answers, row["answers"], strict=True)
                          if k.split("_", 1)[0] != truth]
        answer_ids = {identity[a] for a, k in zip(answers, row["answers"], strict=True)
                      if k.split("_", 1)[0] == truth}
        dup_images += (len(set(distractor_ids)) != len(distractor_ids)
                       or bool(set(distractor_ids) & (answer_ids | {identity[q] for q in question})))
        for a, key in zip(answers, row["answers"], strict=True):
            if key.split("_", 1)[0] != truth:
                usage[a] += 1
                distractor_positive += tuple(sorted(question + [a])) in positives
    report = {
        "positive_outfits": len(positives), "cp_positive_lines": n_pos,
        "cp_negatives": len(negatives), "cp_unique_negative_item_sets": len(set(negatives)),
        "cp_negative_item_sets_matching_positive": len(set(negatives) & positives),
        "cp_negative_identity_sets_matching_positive": sum(
            tuple(sorted(identity[i] for i in n)) in positive_images for n in negatives),
        "questions_referencing_other_splits": off_split,
        "fitb_questions": len(fitb_rows),
        "fitb_questions_with_one_known_completion": known,
        "fitb_questions_with_duplicate_candidate_ids": dup_ids,
        "fitb_questions_with_duplicate_image_identities": dup_images,
        "fitb_distractor_completions_matching_known_positive": distractor_positive,
        "fitb_max_distractor_reuse": max(usage.values(), default=0),
        "items_in_split": len(split_items),
    }
    report["passes"] = bool(
        n_pos == len(negatives) == len(rows) and off_split == 0
        and len(set(negatives)) == len(negatives)
        and report["cp_negative_item_sets_matching_positive"] == 0
        and known == len(fitb_rows) == len(rows) and dup_ids == 0 and dup_images == 0
        and distractor_positive == 0)
    return report


AUXILIARY = ("group_ids_{split}.json",)
COPIED = ("polyvore_item_metadata.json", "image_paths.json")


def prepare_regeneration(base, config=None, verify_base=True):
    """Categories, identity and (optionally) proof that they reproduce the base questions."""
    base = Path(base)
    config = config or {}
    categories = protocol_categories(base, config)
    if categories is None:
        raise ValueError(f"{base}: semantic categories unavailable; set categories_file")
    identity, identity_kind = question_identity(base, config)
    manifest_path = base / "manifest.json"
    base_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    rows = _protocol_rows(base)
    if set(rows) != set(SPLITS):
        raise ValueError(f"{base} must contain train/valid/test outfits")
    verification = {}
    if verify_base:
        for split in SPLITS:
            base_seed = base_question_seed(base_manifest, split, config.get("base_seed"))
            if base_seed is None:
                raise ValueError(f"{base}: base question seed unknown; set base_seed or disable "
                                 "verification (recorded)")
            cp, fitb, _ = _generate(rows[split], categories, identity, base_seed)
            stored_cp = [l for l in (base / f"compatibility_{split}.txt").read_text().splitlines()
                         if l.strip()]
            stored_fitb = json.loads((base / f"fill_in_blank_{split}.json").read_text())
            if cp != stored_cp or fitb != stored_fitb:
                raise ValueError(f"{base}/{split}: identity/category inputs do not reproduce the "
                                 f"base questions (seed {base_seed}); refusing to regenerate")
            verification[split] = {"base_question_seed": base_seed, "reproduced": True}
    return {"base": base, "rows": rows, "categories": categories, "identity": identity,
            "identity_kind": identity_kind, "verification": verification or "disabled",
            "base_manifest_sha256": file_sha256(manifest_path) if manifest_path.exists() else None}


def regenerate_protocol(base, output, seed, *, config=None, verify_base=True, prepared=None):
    """Regenerate CP negatives and FITB candidates of ``base`` with construction ``seed``
    (question seed = seed + 10,000 x (split index + 1), as v1), keeping partitions, the
    category/length matching and identity exclusion of ``generate_questions``.

    With ``verify_base`` the identity/category inputs must first reproduce the base
    protocol's own questions byte for byte (proof that the same rules are applied).
    Writes a new directory (refuses to overwrite) and returns its manifest.
    """
    base, output = Path(base), Path(output)
    prepared = prepared or prepare_regeneration(base, config, verify_base)
    rows, categories, identity = prepared["rows"], prepared["categories"], prepared["identity"]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "kind": "v2_construction_seed_protocol", "construction_seed": seed,
        "question_seeds": {s: seed + 10_000 * (i + 1) for i, s in enumerate(SPLITS)},
        "base_protocol": str(base),
        "base_manifest_sha256": prepared["base_manifest_sha256"],
        "identity_source": prepared["identity_kind"],
        "identity_partition_sha256": hashlib.sha256(json.dumps(
            sorted(sorted(g) for g in _classes(identity).values())).encode()).hexdigest(),
        "base_verification": prepared["verification"],
        "rules": "repbench.data.clean_protocol.generate_questions on unchanged partitions: "
                 "category/length-matched full-corruption CP negatives, one four-way "
                 "same-category FITB per outfit, identity-aware candidate exclusion, "
                 "within-split candidates only",
        "splits": {},
    }
    # D4 fix: partitions are unchanged, so the base protocol's near-duplicate groups still
    # apply; without this key pipeline.equivalence_map() silently returned no groups and the
    # construction-seed bootstrap ignored the v2-Clean merge (CIs too narrow).
    base_info = json.loads((base / "manifest.json").read_text()) if (base / "manifest.json").exists() else {}
    if base_info.get("image_equivalence"):
        manifest["image_equivalence"] = base_info["image_equivalence"]
    for i, split in enumerate(SPLITS):
        shutil.copy2(base / f"{split}.json", output / f"{split}.json")
        for name in (n.format(split=split) for n in AUXILIARY):
            if (base / name).is_file():
                shutil.copy2(base / name, output / name)
        cp, fitb, diagnostics = _generate(rows[split], categories, identity,
                                          manifest["question_seeds"][split])
        audit = audit_questions(rows[split], cp, fitb, identity)
        if not audit["passes"]:
            raise AssertionError(f"Collision audit failed for seed {seed}/{split}: {audit}")
        (output / f"compatibility_{split}.txt").write_text("\n".join(cp) + "\n", encoding="utf-8")
        (output / f"fill_in_blank_{split}.json").write_text(json.dumps(fitb, indent=2),
                                                            encoding="utf-8")
        manifest["splits"][split] = {
            "question_generation": diagnostics, "collision_audit": audit,
            "sha256": {n: file_sha256(output / n) for n in
                       (f"{split}.json", f"compatibility_{split}.txt", f"fill_in_blank_{split}.json")},
        }
    for name in COPIED:
        if (base / name).is_file():
            shutil.copy2(base / name, output / name)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _classes(identity):
    classes = defaultdict(list)
    for item, value in identity.items():
        classes[value].append(item)
    return classes


def _verify_existing(output, base):
    """Reuse an existing seed protocol only if its recorded hashes still hold."""
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("base_protocol") != str(base):
        raise ValueError(f"{output} was built from another base protocol")
    for record in manifest["splits"].values():
        for name, digest in record["sha256"].items():
            if file_sha256(output / name) != digest:
                raise ValueError(f"{output}/{name} changed since it was generated")
    return manifest


def total_uncertainty(frame):
    """Per (dataset, model, metric): mean, between-seed variance, mean bootstrap variance,
    total = within + between (HANDOFF §6.3) and Rubin's within + (1 + 1/m) between."""
    rows = []
    for (dataset, model, metric, split), group in frame.groupby(
            ["dataset", "model", "metric", "split"], sort=True):
        values = group.value.to_numpy(float)
        m = len(values)
        between = float(np.var(values, ddof=1)) if m > 1 else float("nan")
        within = float(np.nanmean(group.bootstrap_var.to_numpy(float)))
        total = within + between
        rubin = within + (1 + 1 / m) * between if m > 1 else float("nan")
        mean = float(values.mean())
        half = 1.96 * math.sqrt(total) if math.isfinite(total) else float("nan")
        rows.append(dict(
            dataset=dataset, model=model, setting="native", learner="logistic", split=split,
            metric=metric, value=mean, ci_low=mean - half, ci_high=mean + half,
            n_seeds=m, seed_sd=math.sqrt(between) if m > 1 else float("nan"),
            seed_min=float(values.min()), seed_max=float(values.max()),
            between_seed_var=between, mean_bootstrap_var=within, total_var=total,
            total_sd=math.sqrt(total) if math.isfinite(total) else float("nan"),
            rubin_total_var=rubin,
            rank_min=int(group["rank"].min()), rank_max=int(group["rank"].max()),
            ci_rule="mean +/- 1.96 sqrt(mean bootstrap var + between-seed var)"))
    return pd.DataFrame(rows)


def run_construction_seeds(specs, models, embeddings, out, *, split, seeds=CONSTRUCTION_SEEDS,
                           configs=None, seed=PROBE_SEED, replicates=2000, verify_base=True,
                           selection_split=None):
    """Regenerate questions per construction seed, run the primary probe on each, and
    summarise per-seed metrics, between-seed SD and total uncertainty."""
    selection_split = selection_split or default_selection_split(split)
    from repbench.v2.stores import load_model_store

    out = Path(out)
    configs = configs or {}
    rows, failures = [], []
    manifest = {"split": split, "construction_seeds": list(seeds), "probe_seed": seed,
                "selection_split": selection_split, "protocols": {}, "inputs": {}}
    for spec in specs:
        stores, prepared = {}, None
        for construction_seed in seeds:
            protocol = out / "protocols" / spec.name / f"seed_{construction_seed}"
            if (protocol / "manifest.json").exists():
                proto_manifest = _verify_existing(protocol, spec.protocol)
            else:
                prepared = prepared or prepare_regeneration(spec.protocol, configs.get(spec.name),
                                                            verify_base)
                proto_manifest = regenerate_protocol(spec.protocol, protocol, construction_seed,
                                                     prepared=prepared)
            manifest["protocols"][f"{spec.name}/seed_{construction_seed}"] = {
                "path": str(protocol), "manifest_sha256": file_sha256(protocol / "manifest.json"),
                "base_verification": proto_manifest["base_verification"]}
            seeded = DatasetSpec(spec.name, protocol, spec.grouping)
            ev = load_eval(seeded, split)
            remaining, fit_cp = carve_train(ev.data, spec.name, seed, selection_split)
            train_cp = flatten(remaining)
            outputs = {}
            for model in models:
                path = (out / "predictions" /
                        f"{spec.name}__{model}__seed{construction_seed}__logistic.json")

                def compute(model=model, stores=stores, spec=spec, train_cp=train_cp,
                            fit_cp=fit_cp, ev=ev):
                    if model not in stores:
                        stores.clear()
                        stores[model], provenance = load_model_store(embeddings, spec.name, model)
                        manifest["inputs"][f"{spec.name}/{model}"] = provenance
                    return _probe(stores[model], train_cp, fit_cp, ev, seed)

                result = _cached_probe(path, compute)
                if "error" in result:
                    failures.append(dict(dataset=spec.name, model=model,
                                         construction_seed=construction_seed, error=result["error"]))
                    continue
                check_alignment(result, ev, path.name)
                outputs[model] = result
            present = [m for m in models if m in outputs]
            if not present:
                continue
            frame, boots = metric_rows(outputs, present, ev, setting="native", learner="logistic",
                                       replicates=replicates, seed=seed,
                                       extra={"construction_seed": construction_seed})
            for row in frame:
                boot = boots[row["metric"]]
                k = present.index(row["model"])
                row["bootstrap_var"] = (float(np.var(boot["samples"][:, k], ddof=1))
                                        if boot is not None else float("nan"))
            for metric in ("cp_auc", "fitb_acc"):
                block = [r for r in frame if r["metric"] == metric]
                ranks = pd.Series([r["value"] for r in block]).rank(ascending=False, method="min")
                for r, rank in zip(block, ranks, strict=True):
                    r["rank"] = int(rank)
            rows += frame
    frame = pd.DataFrame(rows)
    write_new_csv(out / "metrics.csv", frame)
    write_new_csv(out / "summary.csv", total_uncertainty(frame) if len(frame) else frame)
    write_new_csv(out / "failures.csv", pd.DataFrame(
        failures, columns=["dataset", "model", "construction_seed", "error"]))
    write_new_json(out / "manifest.json", manifest)
    return frame


# -------------------------------------------------------------------- 3. provenance
def provenance_group(config_path=None):
    """The protocol's provenance-uncertain group (config key ``provenance_uncertain_group``)."""
    path = Path(config_path) if config_path else ROOT / "configs" / "v2_protocol_draft.yaml"
    if path.is_file():
        import yaml

        def find(node):
            if isinstance(node, dict):
                if "provenance_uncertain_group" in node:
                    return node["provenance_uncertain_group"]
                for value in node.values():
                    found = find(value)
                    if found is not None:
                        return found
            return None

        group = find(yaml.safe_load(path.read_text()))
        if group:
            return tuple(group)
    if config_path:
        raise ValueError(f"{config_path} lacks provenance_uncertain_group")
    return PROVENANCE_UNCERTAIN


def run_provenance(results, specs, out, *, split, exclude=PROVENANCE_UNCERTAIN, replicates=2000,
                   seed=PROBE_SEED):
    """Pairwise / rank / equivalence / Friedman statistics on existing predictions with the
    provenance-uncertain models removed. No refitting."""
    from repbench.v2.rank_statistics import friedman_nemenyi

    out = Path(out)
    metrics, pairwise, ranks, equivalence, friedman = [], [], [], [], {}
    manifest = {"split": split, "excluded": list(exclude), "datasets": {}}
    for spec in specs:
        ev = load_eval(spec, split)
        cells = prediction_cells(results, spec.name)
        kept_models = set()
        for tag, learner in cells:
            outputs = load_predictions(results, spec.name, tag, learner)
            models = sorted(m for m in outputs if m not in exclude)
            if len(models) < 2:
                continue
            for m in models:
                check_alignment(outputs[m], ev, f"{spec.name}/{m}/{tag}/{learner}")
            kept_models |= set(models)
            rows, boots = metric_rows(outputs, models, ev, setting=tag, learner=learner,
                                      replicates=replicates, seed=seed,
                                      extra={"analysis": "provenance_restricted"})
            metrics += rows
            p, r, e = inference_rows(boots, models, dict(dataset=spec.name, setting=tag,
                                                         learner=learner))
            pairwise += p
            ranks += r
            equivalence += e
        manifest["datasets"][spec.name] = {"models": sorted(kept_models), "cells": len(cells)}
        frame = pd.DataFrame([r for r in metrics if r["dataset"] == spec.name])
        if len(frame):
            cp = frame[frame.metric == "cp_auc"].pivot_table(
                index=["setting", "learner"], columns="model", values="value").dropna()
            if cp.shape[1] >= 3 and len(cp) >= 2:
                friedman[spec.name] = friedman_nemenyi(cp.to_numpy(), list(cp.columns),
                                                       ["/".join(b) for b in cp.index])
    write_new_csv(out / "metrics.csv", metrics)
    write_new_csv(out / "pairwise.csv", pairwise)
    write_new_csv(out / "rank_intervals.csv", ranks)
    write_new_csv(out / "equivalence.csv", equivalence)
    write_new_json(out / "friedman.json", friedman)
    write_new_json(out / "manifest.json", manifest)
    return pd.DataFrame(pairwise)


# --------------------------------------------------------------------- 4. subgroups
def length_group(length):
    return str(length) if length <= 5 else "6+"


def run_subgroups(results, specs, out, *, split, configs=None, tag="native", learner="logistic",
                  replicates=1000, seed=PROBE_SEED, min_cp=200, min_fitb=100):
    """Outfit-length / category subgroups (v1 final_subgroup_analysis) with joint cluster
    bootstrap CIs, and failure consensus across all models (overall and by question
    feature), from existing primary predictions."""
    out = Path(out)
    configs = configs or {}
    sub_rows, consensus_rows, unanimous = [], [], {}
    for spec in specs:
        ev = load_eval(spec, split)
        outputs = load_predictions(results, spec.name, tag, learner)
        models = sorted(outputs)
        if not models:
            continue
        for m in models:
            check_alignment(outputs[m], ev, f"{spec.name}/{m}")
        categories = protocol_categories(spec.protocol, configs.get(spec.name))
        cp_len = np.asarray([length_group(len(e.item_ids)) for e in ev.cp])
        fitb_len = np.asarray([length_group(len(q.question_item_ids) + 1) for q in ev.fitb])
        blank = (np.asarray([categories[q.candidate_item_ids[q.correct_index]] for q in ev.fitb])
                 if categories else None)
        groupings = [("CP", "outfit_length", g, cp_len == g) for g in sorted(set(cp_len))]
        groupings += [("FITB", "outfit_length", g, fitb_len == g) for g in sorted(set(fitb_len))]
        if categories:
            sets = [{categories[i] for i in e.item_ids} for e in ev.cp]
            for c in sorted(set().union(*sets)):
                mask = np.asarray([c in s for s in sets])
                if mask.sum() >= min_cp:
                    groupings.append(("CP", "contains_category", c, mask))
            for c in sorted(set(blank)):
                mask = blank == c
                if mask.sum() >= min_fitb:
                    groupings.append(("FITB", "blank_category", c, mask))
        for task, kind, group, mask in groupings:
            rows, _ = metric_rows(
                outputs, models, ev, setting=tag, learner=learner, replicates=replicates,
                seed=seed, cp_mask=mask if task == "CP" else np.zeros(len(ev.cp), bool),
                fitb_mask=mask if task == "FITB" else np.zeros(len(ev.fitb), bool),
                extra={"task": task, "subgroup_type": kind, "subgroup": group})
            sub_rows += [r for r in rows if r["metric"] == ("cp_auc" if task == "CP" else "fitb_acc")]
        # Failure consensus.
        cp_scores, fitb_correct = matrices(outputs, models)
        cp_correct = (cp_scores >= CP_CORRECT_THRESHOLD) == ev.labels[:, None].astype(bool)
        k = len(models)
        features = {"CP": [("all", np.full(len(ev.cp), "all")), ("outfit_length", cp_len)],
                    "FITB": [("all", np.full(len(ev.fitb), "all")), ("outfit_length", fitb_len)]}
        if blank is not None:
            features["FITB"].append(("blank_category", blank))
        for task, correct in (("CP", cp_correct), ("FITB", fitb_correct.astype(bool))):
            count = correct.sum(axis=1)
            for feature, values in features[task]:
                for group in sorted(set(values)):
                    mask = values == group
                    c = count[mask]
                    accuracy = correct[mask].mean(axis=0)
                    consensus_rows.append(dict(
                        dataset=spec.name, split=split, task=task, setting=tag, learner=learner,
                        feature=feature, group=group, models=k, n=int(mask.sum()),
                        unanimous_wrong_n=int((c == 0).sum()),
                        unanimous_wrong_rate=float((c == 0).mean()),
                        independence_expected_unanimous_wrong_rate=float(np.prod(1 - accuracy)),
                        majority_wrong_n=int((c < k / 2).sum()),
                        model_disagreement_n=int(((c > 0) & (c < k)).sum()),
                        unanimous_correct_n=int((c == k).sum())))
        chosen = np.column_stack([outputs[m]["fitb"]["chosen_index"] for m in models])
        failing = np.flatnonzero(fitb_correct.sum(axis=1) == 0)
        unanimous[spec.name] = {
            "split": split, "models": models,
            "cp_threshold": CP_CORRECT_THRESHOLD,
            "fitb_unanimous_failures": [
                {"question_index": int(i), "set_id": ev.fitb_set_ids[i],
                 "outfit_length": int(len(ev.fitb[i].question_item_ids) + 1),
                 "blank_category": None if blank is None else str(blank[i]),
                 "same_wrong_choice": bool(len(set(chosen[i])) == 1),
                 "chosen_index": chosen[i].tolist(), "correct_index": int(ev.fitb[i].correct_index)}
                for i in failing],
        }
    frame = pd.DataFrame(sub_rows)
    write_new_csv(out / "subgroups.csv", frame)
    if len(frame):
        summary = (frame.groupby(["dataset", "split", "task", "subgroup_type", "subgroup", "n"],
                                 as_index=False).value.agg(["min", "max", "mean"]))
    else:
        summary = frame
    write_new_csv(out / "subgroups_summary.csv", summary)
    write_new_csv(out / "failure_consensus.csv", consensus_rows)
    write_new_json(out / "unanimous_failures.json", unanimous)
    write_new_json(out / "manifest.json", {
        "split": split, "setting": tag, "learner": learner, "replicates": replicates,
        "seed": seed, "min_cp": min_cp, "min_fitb": min_fitb,
        "cp_correct_threshold": CP_CORRECT_THRESHOLD,
        "datasets": {name: value["models"] for name, value in unanimous.items()}})
    return frame


# --------------------------------------------------------------- 5. near duplicates
def near_duplicate_masks(ev, affected, key="any_method"):
    """Masks keeping CP pairs / FITB questions not touched by the audit.

    ``outfit`` variant: drop a CP pair (positive + its paired negative) or FITB question
    whose source outfit (``set_id``) is listed in ``test_outfits[key]``.
    ``item`` variant (when ``test_items`` covers ``key``'s sources): additionally drop any CP
    example or FITB question that shows a flagged item (negatives, distractors included).
    """
    outfits = set(affected["test_outfits"][key])
    known = {o.set_id for o in ev.data.outfits(ev.split)}
    unknown = outfits - known
    if unknown:
        raise ValueError(f"{len(unknown)} affected outfits are not in {ev.spec.name}/{ev.split} "
                         f"(e.g. {sorted(unknown)[:3]}); wrong split or protocol")
    masks = {"outfit": (np.asarray([s not in outfits for s in ev.cp_set_ids]),
                        np.asarray([s not in outfits for s in ev.fitb_set_ids]))}
    items_by_key = affected.get("test_items", {})
    sources = {"any_method": list(items_by_key),
               "phash_any": [k for k in items_by_key if k.startswith("phash_")],
               "embedding_any": [k for k in items_by_key if k.startswith("embedding_")]}.get(key, [key])
    if sources and all(s in items_by_key for s in sources):
        flagged = set().union(*(set(items_by_key[s]) for s in sources))
        n = len(ev.cp) // 2
        touched = np.asarray([bool(flagged.intersection(e.item_ids)) for e in ev.cp])
        pair = touched[:n] | touched[n:]
        cp_keep = masks["outfit"][0] & ~np.concatenate([pair, pair])
        fitb_keep = masks["outfit"][1] & np.asarray(
            [not flagged.intersection(q.question_item_ids + q.candidate_item_ids) for q in ev.fitb])
        masks["item"] = (cp_keep, fitb_keep)
    return masks


def run_near_duplicates(results, specs, affected_paths, out, *, split, key="any_method",
                        tag="native", learner="logistic", replicates=2000, seed=PROBE_SEED):
    out = Path(out)
    metrics, pairwise, ranks, manifest = [], [], [], {"split": split, "key": key, "datasets": {}}
    for spec in specs:
        if spec.name not in affected_paths:
            continue
        affected = json.loads(Path(affected_paths[spec.name]).read_text())
        ev = load_eval(spec, split)
        outputs = load_predictions(results, spec.name, tag, learner)
        models = sorted(outputs)
        if not models:
            continue
        for m in models:
            check_alignment(outputs[m], ev, f"{spec.name}/{m}")
        masks = near_duplicate_masks(ev, affected, key)
        full, _ = metric_rows(outputs, models, ev, setting=tag, learner=learner,
                              replicates=replicates, seed=seed, extra={"variant": "full"})
        reference = {(r["model"], r["metric"]): r["value"] for r in full}
        metrics += [{**r, "delta_vs_full": 0.0} for r in full]
        manifest["datasets"][spec.name] = {
            "affected_file": str(affected_paths[spec.name]),
            "affected_sha256": file_sha256(Path(affected_paths[spec.name])),
            "affected_outfits": len(affected["test_outfits"][key]), "variants": {}}
        for variant, (cp_keep, fitb_keep) in masks.items():
            name = f"exclude_{variant}:{key}"
            rows, boots = metric_rows(outputs, models, ev, setting=tag, learner=learner,
                                      replicates=replicates, seed=seed, cp_mask=cp_keep,
                                      fitb_mask=fitb_keep, extra={"variant": name})
            metrics += [{**r, "delta_vs_full": r["value"] - reference[(r["model"], r["metric"])]}
                        for r in rows]
            p, r, _ = inference_rows(boots, models, dict(dataset=spec.name, setting=tag,
                                                         learner=learner, variant=name))
            pairwise += p
            ranks += r
            manifest["datasets"][spec.name]["variants"][name] = {
                "cp_examples_kept": int(cp_keep.sum()), "cp_examples": len(cp_keep),
                "fitb_kept": int(fitb_keep.sum()), "fitb": len(fitb_keep)}
    write_new_csv(out / "metrics.csv", metrics)
    write_new_csv(out / "pairwise.csv", pairwise)
    write_new_csv(out / "rank_intervals.csv", ranks)
    write_new_json(out / "manifest.json", manifest)
    return pd.DataFrame(metrics)


# ---------------------------------------------------------------------- 6. LookBench
def _fisher_ci(r, se):
    if not np.isfinite(se) or abs(r) >= 1:
        return float("nan"), float("nan")
    z = np.arctanh(r)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def rank_correlation(x, y, *, replicates=2000, seed=PROBE_SEED):
    """Spearman and Kendall (tau-b) with Fisher-z CIs (Bonett-Wright / Fieller SEs) and
    model-resampling percentile bootstrap CIs. Descriptive: n is the model count."""
    from scipy.stats import kendalltau, spearmanr

    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    rho, tau = spearmanr(x, y), kendalltau(x, y)
    rng = np.random.default_rng(seed)
    boot_rho, boot_tau = [], []
    for _ in range(replicates):
        idx = rng.integers(n, size=n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        boot_rho.append(spearmanr(x[idx], y[idx]).statistic)
        boot_tau.append(kendalltau(x[idx], y[idx]).statistic)
    result = {"n_models": n, "spearman_rho": float(rho.statistic), "spearman_p": float(rho.pvalue),
              "kendall_tau": float(tau.statistic), "kendall_p": float(tau.pvalue)}
    result["spearman_fisher_low"], result["spearman_fisher_high"] = _fisher_ci(
        rho.statistic, math.sqrt((1 + rho.statistic ** 2 / 2) / (n - 3)) if n > 3 else float("nan"))
    result["kendall_fisher_low"], result["kendall_fisher_high"] = _fisher_ci(
        tau.statistic, math.sqrt(0.437 / (n - 4)) if n > 4 else float("nan"))
    for name, draws in (("spearman", boot_rho), ("kendall", boot_tau)):
        draws = np.asarray(draws, float)
        draws = draws[np.isfinite(draws)]
        low, high = (np.quantile(draws, [0.025, 0.975]) if len(draws) else (np.nan, np.nan))
        result[f"{name}_boot_low"], result[f"{name}_boot_high"] = float(low), float(high)
    result["bootstrap_valid_draws"] = len(boot_rho)
    return result


def run_lookbench(metrics_path, scores_path, out, *, replicates=2000, seed=PROBE_SEED,
                  min_models=3):
    """Correlate published LookBench v3 scores (config) with primary CP AUC (and FITB) per
    dataset. Models without a published score or without an exact-checkpoint match are
    listed in overlap.csv but excluded from the correlation."""
    out = Path(out)
    config = json.loads(Path(scores_path).read_text())
    published = config["models"]
    metrics = pd.read_csv(metrics_path)
    primary = metrics[(metrics.setting == "native") & (metrics.learner == "logistic")]
    overlap, correlations, reversals = [], [], []
    for (dataset, split), block in primary.groupby(["dataset", "split"], sort=True):
        table = block.pivot_table(index="model", columns="metric", values="value")
        for model in table.index:
            entry = published.get(model, {})
            usable = entry.get("score") is not None and entry.get("exact_match") is True
            overlap.append(dict(dataset=dataset, split=split, model=model,
                                lookbench_label=entry.get("lookbench_label"),
                                lookbench_score=entry.get("score"),
                                exact_match=entry.get("exact_match"), used=usable,
                                cp_auc=table.loc[model].get("cp_auc"),
                                fitb_acc=table.loc[model].get("fitb_acc")))
        used = [r for r in overlap if r["dataset"] == dataset and r["split"] == split and r["used"]]
        for metric in ("cp_auc", "fitb_acc"):
            pairs = [(r["lookbench_score"], r[metric]) for r in used if pd.notna(r[metric])]
            row = dict(dataset=dataset, split=split, compatibility_metric=metric,
                       lookbench_metric=config.get("metric"), n_models=len(pairs))
            if len(pairs) < min_models:
                row["status"] = f"insufficient: {len(pairs)} models with a published exact score"
            else:
                row.update(rank_correlation(*zip(*pairs), replicates=replicates, seed=seed))
                row["status"] = "ok"
            correlations.append(row)
            for a in range(len(used)):
                for b in range(a + 1, len(used)):
                    left, right = used[a], used[b]
                    if (np.sign(left["lookbench_score"] - right["lookbench_score"])
                            * np.sign(left[metric] - right[metric]) < 0):
                        reversals.append(dict(dataset=dataset, split=split, metric=metric,
                                              model_a=left["model"], model_b=right["model"]))
    write_new_csv(out / "overlap.csv", overlap)
    write_new_csv(out / "correlations.csv", pd.DataFrame(correlations))
    write_new_csv(out / "rank_reversals.csv",
                  pd.DataFrame(reversals, columns=["dataset", "split", "metric", "model_a", "model_b"]))
    write_new_json(out / "manifest.json", {
        "scores_file": str(scores_path), "scores_sha256": file_sha256(Path(scores_path)),
        "metrics_file": str(metrics_path), "metrics_sha256": file_sha256(Path(metrics_path)),
        "source": config.get("source"), "benchmark": config.get("benchmark")})
    return pd.DataFrame(correlations)
