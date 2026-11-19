"""A100 for v2: external human/expert test set, evaluated unchanged, never fitted on.

Nothing in this module fits, tunes, selects or calibrates anything on A100. It
(1) prepares the official archive into a question protocol directory that the v2
extractor reads like a Polyvore-format protocol, (2) loads the questions, (3) turns
candidate scores (probe or zero-shot MLLM) into per-question records under both LAT
label interpretations, and (4) computes question-level statistics (A100 clusters =
question, HANDOFF_V2 §6.3).

Label semantics are those of v1 (``repbench.data.a100.prepare_a100``):
- LAT primary ground truth = majority of ``gt_distribution`` (first maximum on a tie);
  sensitivity = the archive ``gt``; mLAT = mean human agreement
  ``gt_distribution[chosen]``.
- AAT ground truth = archive ``gt``; facet from ``AAT_DIMENSIONS`` by question number.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from repbench.data.a100 import AAT_DIMENSIONS

A100_ARCHIVE_SHA256 = "31773883ab6fcf9c0de8182b7561f7319a1d928d542906b8c3c4b8674196f829"
#: The prepared protocol exposes A100 only as split "test", so the v2 extractor's
#: frozen-protocol guard also covers A100 embedding extraction.
A100_SPLIT = "test"
TASKS = ("LAT", "AAT")
FACETS = tuple(AAT_DIMENSIONS)
#: Pre-specified TOST margins for A100 accuracies (2/5/10 percentage points).
A100_MARGINS = (0.02, 0.05, 0.10)
PREPARE_VERSION = "a100-v2-prepare-1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- guard


def require_frozen_protocol(hash_file, root=None):
    """Same rule as ``scripts/v2/pipeline.py --eval-split test``: the hash file written at
    the G7 freeze must exist and the protocol file it names must still match."""
    hash_file = Path(hash_file)
    if not hash_file.is_file():
        raise PermissionError(
            f"A100 evaluation requires the frozen v2 protocol (G7) and Phase 2; "
            f"missing {hash_file}"
        )
    record = json.loads(hash_file.read_text())
    target = Path(record["file"])
    if not target.is_absolute() and not target.is_file() and root is not None:
        target = Path(root) / target
    if not target.is_file():
        raise PermissionError(f"Frozen protocol file named by {hash_file} is missing: {target}")
    if sha256_file(target) != record["sha256"]:
        raise PermissionError("Frozen protocol file changed since freeze; refusing")
    return {"hash_file": str(hash_file), "file": str(target), "sha256": record["sha256"]}


# --------------------------------------------------------------------------- questions


@dataclass(frozen=True)
class A100Question:
    question_id: str
    task: str
    question_num: int
    question_items: tuple
    candidate_items: tuple
    primary_gt_index: int  # LAT: majority; AAT: archive
    archive_gt_index: int
    majority_gt_index: int | None = None  # LAT only
    gt_distribution: tuple | None = None  # LAT only
    facet: str | None = None  # AAT only

    def to_json(self):
        value = asdict(self)
        value["question_items"] = list(self.question_items)
        value["candidate_items"] = list(self.candidate_items)
        if self.gt_distribution is not None:
            value["gt_distribution"] = list(self.gt_distribution)
        return value


def questions_from_v1_rows(task, rows):
    """Convert v1 ``lat.json`` / ``aat.json`` rows (prepare_a100 schema)."""
    task = task.upper()
    out = []
    for row in rows:
        if task == "LAT":
            distribution = tuple(float(v) for v in row["gt_distribution"])
            majority = int(row["majority_gt_index"])
            if majority != int(np.argmax(distribution)):
                raise ValueError(f"LAT {row['question_num']}: majority index inconsistent")
            out.append(A100Question(
                f"LAT_{row['question_num']}", "LAT", int(row["question_num"]),
                tuple(row["question"]), tuple(row["answers"]), majority,
                int(row["archive_gt_index"]), majority, distribution, None))
        elif task == "AAT":
            out.append(A100Question(
                f"AAT_{row['question_num']}", "AAT", int(row["question_num"]),
                tuple(row["question"]), tuple(row["answers"]), int(row["archive_gt_index"]),
                int(row["archive_gt_index"]), None, None, row["dimension"]))
        else:
            raise ValueError(f"Unknown A100 task {task}")
    for q in out:
        if not 0 <= q.archive_gt_index < len(q.candidate_items):
            raise ValueError(f"{q.question_id}: archive gt out of range")
    return out


def load_questions(prepared):
    """Questions of a prepared directory, verified against its manifest."""
    prepared = Path(prepared)
    manifest = json.loads((prepared / "manifest.json").read_text())
    path = prepared / "questions.json"
    if sha256_file(path) != manifest["files"]["questions.json"]:
        raise ValueError("questions.json does not match the A100 manifest")
    rows = json.loads(path.read_text())
    questions = [
        A100Question(**{**r, "question_items": tuple(r["question_items"]),
                        "candidate_items": tuple(r["candidate_items"]),
                        "gt_distribution": (tuple(r["gt_distribution"])
                                            if r["gt_distribution"] is not None else None)})
        for r in rows
    ]
    if len({q.question_id for q in questions}) != len(questions):
        raise ValueError("Duplicate A100 question ids")
    return questions, manifest


def fitb_questions(questions):
    """Polyvore-style FITB objects (correct_index = primary gt) for the probe scorer."""
    from repbench.data.polyvore import FITBQuestion

    return [FITBQuestion(q.question_items, q.candidate_items, q.primary_gt_index)
            for q in questions]


def all_item_ids(questions):
    return sorted({i for q in questions for i in (*q.question_items, *q.candidate_items)})


# --------------------------------------------------------------------------- prepare


def _polyvore_files(questions):
    """Polyvore-format split 'test' (key = '<set_id>_<index>').

    Pseudo-sets exist only so official-format keys resolve: set ``<T><nnn>`` holds the
    partial outfit plus the primary-gt candidate, set ``<T><nnn>d`` the other candidates.
    They are NOT outfits; there are no CP examples (empty compatibility file).
    """
    outfits, fitb = [], []
    for q in questions:
        base = f"{q.task}{q.question_num:03d}"
        if "_" in base:
            raise ValueError("Set ids must not contain '_'")
        main = [{"item_id": i, "index": k + 1} for k, i in enumerate(q.question_items)]
        gt_key = f"{base}_{len(main) + 1}"
        main.append({"item_id": q.candidate_items[q.primary_gt_index], "index": len(main) + 1})
        others = [c for k, c in enumerate(q.candidate_items) if k != q.primary_gt_index]
        decoys = [{"item_id": i, "index": k + 1} for k, i in enumerate(others)]
        outfits += [{"set_id": base, "items": main, "a100_role": "partial_outfit_plus_gt"},
                    {"set_id": base + "d", "items": decoys, "a100_role": "other_candidates"}]
        answers, decoy = [], 0
        for k in range(len(q.candidate_items)):
            if k == q.primary_gt_index:
                answers.append(gt_key)
            else:
                decoy += 1
                answers.append(f"{base}d_{decoy}")
        entry = {
            "question": [f"{base}_{k + 1}" for k in range(len(q.question_items))],
            "answers": answers,
            "blank_position": len(q.question_items) + 1,
            "question_id": q.question_id,
            "task": q.task,
            "question_num": q.question_num,
            "primary_gt_index": q.primary_gt_index,
            "archive_gt_index": q.archive_gt_index,
        }
        if q.task == "LAT":
            entry["majority_gt_index"] = q.majority_gt_index
            entry["gt_distribution"] = list(q.gt_distribution)
        else:
            entry["facet"] = q.facet
        fitb.append(entry)
    return outfits, fitb


def _normalize_images(images_root, item_ids):
    """Decode every image with the v2 rule; composite non-JPEG/transparent ones on white.

    JPEG cannot carry alpha and the v2 encoders decode with ``load_rgb``, so JPEG bytes
    are kept unchanged (the compositing is a no-op and re-encoding would add loss).
    Anything else is composited onto white and stored losslessly as PNG.
    """
    from PIL import Image

    from repbench.v2.images import to_rgb_on_white

    mapping, counts, composited = {}, {}, []
    for item in item_ids:
        path = images_root / f"{item}.jpg"
        with Image.open(path) as image:
            image.load()
            kind = (image.format, image.mode, "transparency" in image.info)
            counts[str(kind)] = counts.get(str(kind), 0) + 1
            if image.format == "JPEG" and image.mode in ("RGB", "L", "CMYK"):
                to_rgb_on_white(image)  # decodes under the shared rule
                mapping[item] = path.name
                continue
            rgb = to_rgb_on_white(image)
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG")
        target = images_root / f"{item}.png"
        with target.open("xb") as stream:
            stream.write(buffer.getvalue())
        path.unlink()
        mapping[item] = target.name
        composited.append(item)
    return mapping, counts, composited


def prepare_a100_v2(archive, output, *, expected_sha256=A100_ARCHIVE_SHA256):
    """Prepare A100 into ``output`` (refuses to overwrite; builds in ``<output>.partial``)."""
    from repbench.data.a100 import prepare_a100
    from repbench.data.polyvore import PolyvoreDisjoint

    archive, output = Path(archive), Path(output)
    partial = output.with_name(output.name + ".partial")
    for path in (output, partial):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    archive_sha = sha256_file(archive)
    if expected_sha256 is not None and archive_sha != expected_sha256:
        raise ValueError(f"A100 archive sha256 {archive_sha} != expected {expected_sha256}")
    # v1 conversion (ids, label semantics, image copy) reused verbatim.
    v1_manifest = prepare_a100(archive, partial, partial / "images")
    (partial / "manifest.json").rename(partial / "v1_prepare_manifest.json")
    questions = []
    for task in TASKS:
        rows = json.loads((partial / f"{task.lower()}.json").read_text())
        questions += questions_from_v1_rows(task, rows)
    problems = []
    for q in questions:
        if len(set(q.candidate_items)) != len(q.candidate_items):
            problems.append(f"{q.question_id}: repeated candidate")
        if set(q.candidate_items) & set(q.question_items):
            problems.append(f"{q.question_id}: candidate also in partial outfit")
    if problems:
        raise ValueError("; ".join(problems))
    items = all_item_ids(questions)
    mapping, image_counts, composited = _normalize_images(partial / "images", items)
    on_disk = sorted(p.name for p in (partial / "images").iterdir())
    if on_disk != sorted(mapping.values()):
        raise ValueError("Prepared images do not match referenced A100 items")
    (partial / "image_paths.json").write_text(json.dumps(mapping, indent=1, sort_keys=True))
    (partial / "questions.json").write_text(
        json.dumps([q.to_json() for q in questions], indent=1))
    outfits, fitb = _polyvore_files(questions)
    (partial / f"{A100_SPLIT}.json").write_text(json.dumps(outfits, indent=1))
    (partial / f"fill_in_blank_{A100_SPLIT}.json").write_text(json.dumps(fitb, indent=1))
    (partial / f"compatibility_{A100_SPLIT}.txt").write_text("")
    # Round trip through the loader the extractor uses.
    data = PolyvoreDisjoint(partial.parent, partial, "a100")
    loaded = data.fitb(A100_SPLIT)
    for q, f in zip(questions, loaded, strict=True):
        if (f.question_item_ids, f.candidate_item_ids, f.correct_index) != (
                q.question_items, q.candidate_items, q.primary_gt_index):
            raise AssertionError(f"Polyvore-format round trip failed for {q.question_id}")
    if data.compatibility(A100_SPLIT):
        raise AssertionError("A100 must have no CP examples")
    lat = [q for q in questions if q.task == "LAT"]
    ties = [q.question_num for q in lat
            if sum(v == max(q.gt_distribution) for v in q.gt_distribution) > 1]
    files = sorted(p.name for p in partial.iterdir() if p.is_file())
    manifest = {
        "kind": "a100_v2_prepared",
        "version": PREPARE_VERSION,
        "role": "external human/expert test set; evaluated unchanged; never fitted, tuned, "
                "calibrated or used for selection",
        "archive": str(archive),
        "archive_sha256": archive_sha,
        "expected_archive_sha256": expected_sha256,
        "split": A100_SPLIT,
        "tasks": {task: {
            "questions": sum(q.task == task for q in questions),
            "items": len(all_item_ids([q for q in questions if q.task == task])),
            "candidates_per_question": sorted({len(q.candidate_items) for q in questions
                                               if q.task == task}),
            "partial_outfit_sizes": sorted({len(q.question_items) for q in questions
                                            if q.task == task}),
        } for task in TASKS},
        "items": len(items),
        "lat_primary_gt": "majority of gt_distribution (first maximum on ties)",
        "lat_sensitivity_gt": "archive gt",
        "mlat": "mean over LAT questions of gt_distribution[chosen candidate]",
        "aat_gt": "archive gt",
        "aat_facets": {k: list(v) for k, v in AAT_DIMENSIONS.items()},
        "lat_majority_ties": ties,
        "lat_archive_majority_disagreements": [q.question_num for q in lat
                                               if q.archive_gt_index != q.majority_gt_index],
        "image_decoding": "repbench.v2.images.to_rgb_on_white; JPEG bytes kept (no alpha, "
                          "compositing is a no-op); other formats composited on white -> PNG",
        "image_formats": image_counts,
        "composited_items": composited,
        "polyvore_format": "split 'test': pseudo-sets <T><nnn> (partial outfit + primary gt) "
                           "and <T><nnn>d (other candidates) exist only for key resolution; "
                           "no CP examples",
        "v1_prepare_manifest": v1_manifest,
        "files": {name: sha256_file(partial / name) for name in files},
    }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=1))
    partial.rename(output)
    return manifest


# --------------------------------------------------------------------------- records


def question_records(questions, candidate_scores, chosen, context):
    """Per-question record under both LAT interpretations (scores in archive order)."""
    if not (len(questions) == len(candidate_scores) == len(chosen)):
        raise ValueError("Questions, scores and choices must align")
    records = []
    for q, scores, pick in zip(questions, candidate_scores, chosen, strict=True):
        if len(scores) != len(q.candidate_items) or not 0 <= pick < len(scores):
            raise ValueError(f"{q.question_id}: invalid scores/choice")
        record = {
            **context,
            "question_id": q.question_id, "task": q.task, "question_num": q.question_num,
            "facet": q.facet, "n_candidates": len(q.candidate_items),
            "candidate_item_ids": list(q.candidate_items),
            "candidate_scores": [float(s) for s in scores],
            "chosen_index": int(pick),
            "primary_gt_index": q.primary_gt_index,
            "archive_gt_index": q.archive_gt_index,
            "majority_gt_index": q.majority_gt_index,
            "correct_primary": int(pick == q.primary_gt_index),
            "correct_archive": int(pick == q.archive_gt_index),
            "correct_majority": int(pick == q.majority_gt_index) if q.task == "LAT" else None,
            "human_agreement": float(q.gt_distribution[pick]) if q.task == "LAT" else None,
        }
        records.append(record)
    return records


#: (dataset, metric, task, facet or None, record field, binary?)
METRIC_DEFINITIONS = (
    ("a100_lat", "acc_majority", "LAT", None, "correct_majority", True),  # primary
    ("a100_lat", "acc_archive", "LAT", None, "correct_archive", True),
    ("a100_lat", "mlat", "LAT", None, "human_agreement", False),
    ("a100_aat", "acc", "AAT", None, "correct_archive", True),
    *(("a100_aat", f"acc_{facet.lower()}", "AAT", facet, "correct_archive", True)
      for facet in FACETS),
)


def mean_cluster_bootstrap(values, *, groups, replicates=2000, seed=20260912):
    """Same resampling as ``paired_statistics.cluster_bootstrap(metric='accuracy')`` but
    for bounded continuous per-question values (mLAT human agreement)."""
    from repbench.eval.paired_statistics import _cluster_codes

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.shape[0] or not np.isfinite(values).all():
        raise ValueError("Values must be a finite examples-by-models matrix")
    codes, n_groups = _cluster_codes(groups, len(values))
    rng = np.random.default_rng(seed)
    samples = []
    while len(samples) < replicates:
        counts = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)
        weights = counts[codes].astype(float)
        if weights.sum():
            samples.append(np.average(values, axis=0, weights=weights))
    return {"point": values.mean(0), "samples": np.asarray(samples), "cluster_count": n_groups,
            "second_cluster_count": None, "rejected_degenerate_draws": 0,
            "method": "paired cluster bootstrap", "replicates": replicates, "seed": seed}


def summarize_block(records_by_model, models, context, *, replicates=2000, seed=20260912):
    """Metrics, pairwise, rank-interval and equivalence rows for one family of models
    (same training corpus, setting, learner). Clusters = question."""
    from repbench.eval.paired_statistics import (
        cluster_bootstrap,
        exact_mcnemar,
        pairwise_inference,
    )

    metrics, pairwise, ranks, equivalence = [], [], [], []
    for dataset, metric, task, facet, field, binary in METRIC_DEFINITIONS:
        columns, order = [], None
        for model in models:
            rows = [r for r in records_by_model[model]
                    if r["task"] == task and (facet is None or r["facet"] == facet)]
            rows.sort(key=lambda r: r["question_num"])
            ids = [r["question_id"] for r in rows]
            if order is None:
                order = ids
            elif ids != order:
                raise ValueError(f"{model}: question set differs within the family")
            columns.append([r[field] for r in rows])
        if not order:
            continue
        matrix = np.column_stack(columns).astype(float)
        if binary:
            boot = cluster_bootstrap(matrix, metric="accuracy", groups=order,
                                     replicates=replicates, seed=seed)
        else:
            boot = mean_cluster_bootstrap(matrix, groups=order, replicates=replicates, seed=seed)
        inference = pairwise_inference(boot, list(models),
                                       metric="accuracy" if binary else metric,
                                       margins=A100_MARGINS)
        for estimate in inference["model_estimates"]:
            metrics.append({
                "dataset": dataset, "model": estimate["model"], **context, "metric": metric,
                "value": estimate["estimate"], "ci_low": estimate["ci_low"],
                "ci_high": estimate["ci_high"], "n_questions": len(order),
                "ci_method": f"{boot['method']} over questions, {replicates} reps",
            })
        common = {"dataset": dataset, **context, "metric": metric, "method": boot["method"],
                  "clusters": boot["cluster_count"]}
        for comparison in inference["comparisons"]:
            i, j = models.index(comparison["left"]), models.index(comparison["right"])
            extra = ({"mcnemar_exact_p": exact_mcnemar(matrix[:, i], matrix[:, j])["pvalue"]}
                     if binary else {"mcnemar_exact_p": None})
            pairwise.append({**common, **comparison, **extra})
        ranks += [{**common, **r} for r in inference["rank_intervals"]]
        equivalence += [{**common, **r} for r in inference["equivalence"]]
    return metrics, pairwise, ranks, equivalence
