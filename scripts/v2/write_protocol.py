"""Generate the frozen v2 protocol (G7) from code constants, the registry and evidence.

Nothing scientific is typed twice: revisions come from `repbench.v2.registry`, grids
and budgets from the modules that use them, split/question/archive hashes from the
evidence files. `--final` refuses to write while any field is PENDING and writes
`configs/v2_protocol.yaml`, a readable `.txt` twin and `v2_protocol_hash.json`
(exclusive create: a frozen protocol is never overwritten).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

PENDING = "PENDING"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    return json.loads(json.dumps(value, default=list))


def maybe_json(path):
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else None


def main():
    from repbench.v2 import probe
    from repbench.v2.calibration_v2 import calibrate_and_evaluate  # noqa: F401 (existence)
    from repbench.v2.images import MAX_PIXELS, QWEN_MAX_PIXELS, QWEN_MIN_PIXELS
    from repbench.v2.logistic import V2_C_GRID
    from repbench.v2.registry import V2_ENCODERS, ZERO_SHOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="*", default=[], help="name=build_report.json")
    parser.add_argument("--question-ids", nargs="*", default=[], help="name=frozen_ids.json")
    parser.add_argument("--equivalence", nargs="*", default=[], help="name=report.json")
    parser.add_argument("--archives", type=Path, default=Path.home() / "repbench-mirror/SHA256SUMS")
    parser.add_argument("--out", type=Path, default=Path("configs"))
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()

    roster = {
        key: {
            "checkpoint": s.checkpoint, "revision": s.revision, "family": s.family,
            "input_resolution": s.input_resolution or f"dynamic, max_pixels={QWEN_MAX_PIXELS}",
            "embedding_dim": s.embedding_dim, "backend": s.backend, "inference_dtype": "bfloat16",
        }
        for key, s in V2_ENCODERS.items()
    }
    archives = dict(line.split("  ")[::-1] for line in args.archives.read_text().splitlines())
    splits = {}
    for item in args.splits:
        name, path = item.split("=")
        report = maybe_json(path)
        splits[name] = PENDING if not report else {
            "report_sha256": sha(path), "status": report.get("status"),
            "counts": report.get("splits") or report.get("counts"),
            "file_sha256": report.get("file_sha256") or report.get("hashes"),
        }
    questions = {}
    for item in args.question_ids:
        name, path = item.split("=")
        questions[name] = {"ids_file_sha256": sha(path)} if Path(path).exists() else PENDING
    equivalence = {}
    for item in args.equivalence:
        name, path = item.split("=")
        equivalence[name] = {"report_sha256": sha(path)} if Path(path).exists() else PENDING

    protocol = {
        "protocol": "RepBench-Fashion v2",
        "phase": "frozen before any v2 test-split or POG-test outcome is viewed",
        "decisions_log": "docs/v2_decisions.md (owner decisions A-F, 2026-11-19)",
        "roster": {
            "encoders": roster,
            "zero_shot_mllm": {"checkpoint": ZERO_SHOT.checkpoint, "revision": ZERO_SHOT.revision,
                               "dtype": "bfloat16", "role": "baseline, not a probe"},
            "matched_parent_pairs": [["fashionclip2", "openclip_b32_laion2b"],
                                     ["marqo_fashionsiglip", "siglip_b16_224"],
                                     ["gr_lite", "dinov3_vitl16_336"]],
            "provenance_uncertain_group": ["marqo_fashionsiglip", "marqo_fashionclip", "gr_lite",
                                           "qwen3_vl_emb_2b", "qwen3_vl_emb_8b"],
            "provenance_note": "Qwen3-VL-Embedding added to the uncertain group (web-scale, undisclosed data)",
        },
        "inputs": {
            "image_decoding": "alpha composited onto white for every dataset (repbench.v2.images)",
            "pixel_budget": f"no model sees more than {MAX_PIXELS} pixels (384x384); fixed-resolution "
                            "encoders use official processors; Qwen min/max_pixels="
                            f"{QWEN_MIN_PIXELS}/{QWEN_MAX_PIXELS}",
            "precision": "uniform bfloat16 + PyTorch SDPA (GR-Lite: algebraically equivalent SDPA)",
            "determinism": {"seed": 20260912, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                            "cudnn_benchmark": False},
        },
        "datasets": {
            "archives_sha256": archives,
            "protocols": {
                "polyvore_v2_clean": {"grouping": "item_component"},
                "iqon3000_v2_clean": {"grouping": "two_way_user_x_item_component",
                                      "sensitivity": "user_only"},
                "pog_v2_clean": {"grouping": "two_way_user_x_item_component",
                                 "sensitivity": "user_only", "same_category": "raw cate_id",
                                 "min_category_images_per_split": 10},
                "a100": {"role": "external human/expert test; evaluated unchanged, never trained on",
                         "lat_primary": "majority label", "lat_sensitivity": "archive gt",
                         "also": ["mLAT", "AAT facets"]},
            },
            "near_duplicate_policy": {
                "rule": "model-free: pHash Hamming <= 2 AND pixel-level confirmation; equivalent "
                        "images form one indivisible unit across splits",
                "applies_to": ["polyvore", "iqon3000", "pog"],
                "equivalence_reports": equivalence or PENDING,
                "comparison": "v1-Clean splits evaluated only as leakage-inflation analysis",
            },
            "split_evidence": splits or PENDING,
            "dependency_note": "POG user clicks cover ~12.5% of outfits (decision B)",
        },
        "probe": {
            "primary": "native: train-item StandardScaler -> L2 -> mean pair [|zi-zj|; zi*zj] "
                       "-> train-outfit StandardScaler -> logistic regression",
            "logistic_c_grid": [float(c) for c in V2_C_GRID],
            "c_grid_edge_rule": "validation-only extension, up to 4 decades, recorded; "
                                "failure recorded in failures.csv",
            "fit_select_evaluate": "fit on TRAIN; hyperparameters, C and post-hoc calibration selected/"
                                   "fitted on VALID only; evaluated once on TEST (G5 dry run used a "
                                   "seeded 20% train carve-out because valid stood in for test)",
            "sensitivities": {
                "pca_widths": [128, 256, 512], "pca": "randomized, train-items only, seed 20260912 "
                "(Halko et al. 2011); PCA-256 = v1 continuity",
                "operator": "abs_product_sum ([|zi-zj|; zi*zj; zi+zj])",
                "learners": {
                    "xgboost": {"grid": plain(probe.XGB_GRID), "tree_method": "hist", "device": "cpu"},
                    "mlp": {"grid": plain(probe.MLP_GRID), "early_stopping": "internal 10% of train"},
                },
                "label_fractions": [0.10, 0.25, 0.50, 1.00],
                "construction_seeds": [42, 43, 44, 45, 46],
            },
            "fitb_rule": probe.FITB_TIE_RULE,
            "repeated_items": "de-duplicated within an outfit (order-preserving)",
        },
        "model_selection_rule": "highest VALID-split native CP AUC per dataset, read from probe "
                                "selection evidence, never from test (code: "
                                "repbench.v2.selection.select_models); no hard-coded keys",
        "statistics": {
            "paired_cluster_bootstrap": {"replicates": 2000, "seed": 20260912},
            "all_pairwise": "Holm within dataset x metric",
            "cp": ["bootstrap", "DeLong (ignores clustering; disagreements explained)"],
            "fitb_a100": ["exact McNemar", "cluster sign-flip McNemar where users cluster"],
            "equivalence_tost_margins": {"cp_auc": [0.002, 0.005, 0.010],
                                         "fitb_pp": [0.5, 1.0, 2.0], "a100_pp": [2.0, 5.0, 10.0]},
            "rank": ["bootstrap rank intervals", "Friedman + Nemenyi CD (descriptive)", "Kendall W"],
            "effect_sizes": ["delta with CI", "Cohen h for accuracies"],
            "p_value_reporting": "with floor, e.g. < 5e-4 at 2,000 replicates",
        },
        "calibration": {"metrics": ["brier", "log_loss", "ece_10", "ece_15", "ece_20"],
                        "posthoc": ["temperature", "platt"], "fit_on": "valid split",
                        "brier_decomposition": "Murphy, 10 equal-width bins",
                        "ece_ci": "clustered bootstrap"},
        "baselines": {
            "outfit_transformer": {"lr_grid": [1e-4, 3e-4], "max_epochs": 100, "patience": 10,
                                   "seeds": [42, 43, 44, 45, 46],
                                   "inputs": ["resnet50", "selected best generic", "selected best fashion"]},
            "zero_shot_mllm": {"scoring": "next-token log-prob of option letters; argmax over plain "
                                          "letter tokens", "option_order": "SHA-256(seed, question id) "
                                                                           "permutation, seed 20261118",
                               "questions": "all A100 + 2,000 seeded FITB per clean test set",
                               "frozen_question_ids": questions or PENDING},
        },
        "efficiency": "scripts/v2/profile_efficiency_v2.py on the DGX GB10, nothing else running: "
                      "params, FlopCounterMode GFLOPs, CUDA-event GPU-only latency (10 warm-up + 50 "
                      "timed, median+IQR, batch 1/64), allocator peak, CPU preprocessing separate, "
                      "energy (NVML or 10 Hz power sampling, method stated), storage",
        "audits": ["cross-split near-duplicate (pHash + embedding cosine >= 0.95) exclusion "
                   "sensitivity", "IQON/POG retention and selection effect",
                   "lee-research Polyvore-D overlap cross-check", "LookBench v3 comparison "
                   "(regenerated on clean data)"],
        "deviations_from_handoff": [
            "G3 revised (owner): original v1 split identity unverifiable",
            "POG acquisition success threshold 90% (handoff 95%)",
            "Qwen pixel cap 384x384 (provider default up to 1,800 visual tokens)",
            "v2-Clean splits merge near-duplicate images (new; v1-Clean kept as comparison)",
            "IQON primary grouping two-way user x item-component (handoff: user primary)",
        ],
    }
    pending = json.dumps(protocol).count(PENDING)
    if args.final and pending:
        raise SystemExit(f"{pending} PENDING fields; cannot freeze")
    protocol["generated_utc"] = datetime.now(UTC).isoformat()
    text = yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True, width=100)
    stem = args.out / ("v2_protocol" if args.final else "v2_protocol_draft")
    mode = "x" if args.final else "w"
    with open(f"{stem}.yaml", mode) as stream:
        stream.write(text)
    with open(f"{stem}.txt", mode) as stream:
        stream.write(text.replace(": ", " = "))
    digest = sha(f"{stem}.yaml")
    if args.final:
        with open(args.out / "v2_protocol_hash.json", "x") as stream:
            json.dump({"file": f"{stem}.yaml", "sha256": digest}, stream, indent=2)
    print(f"{stem}.yaml sha256={digest} pending={pending}")


if __name__ == "__main__":
    main()
