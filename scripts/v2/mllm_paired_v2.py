"""Post-run sweep B6: paired zero-shot MLLM vs linear-probe FITB comparison on the SAME frozen
2,000-question test subset per dataset (the figure in make_figures_v2.py compares the MLLM
subset with probe accuracy on the full test set, which is not paired).

For each dataset: correctness matrix [2,000 questions x (MLLM + 15 primary probes)], cluster
bootstrap with the dataset's primary grouping (Polyvore item component; IQON/POG two-way
user x item component, near-duplicate groups merged), pairwise_inference for MLLM-vs-probe
pairs only, Holm within dataset over those 15 comparisons, exact and cluster McNemar.
Reads frozen outputs only; writes a new directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

MODELS = ("resnet50 fashionclip2 openclip_b32_laion2b marqo_fashionclip marqo_fashionsiglip "
          "siglip_b16_224 siglip2_b16_384 dinov3_vitl16 gr_lite dinov3_vitl16_336 clip_vitl14_336 "
          "pe_core_l14_336 siglip2_so400m_384 qwen3_vl_emb_2b qwen3_vl_emb_8b").split()
GROUPING = {"polyvore_v2_clean": "component", "iqon3000_v2_clean": "two_way",
            "pog_v2_clean": "two_way"}
EQUIV = {"polyvore_v2_clean": "polyvore", "iqon3000_v2_clean": "iqon", "pog_v2_clean": "pog"}
MLLM = "qwen3_vl_8b_zeroshot"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import pipeline as P
    from repbench.data.polyvore import PolyvoreDisjoint
    from repbench.eval.paired_statistics import (cluster_adjusted_mcnemar, cluster_bootstrap,
                                                 exact_mcnemar, holm, pairwise_inference)

    ap = argparse.ArgumentParser()
    ap.add_argument("--ff", type=Path, required=True, help="final v2/full directory")
    ap.add_argument("--protocols", type=Path, required=True, help="dir with <dataset>/build_a and image_equivalence/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replicates", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260912)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows, summary, inputs = [], {}, {}
    for name, grouping in GROUPING.items():
        protocol = args.protocols / name / "build_a"
        eq = args.protocols / "image_equivalence" / EQUIV[name] / "equivalence_groups.json"
        P.EQUIVALENCE[name] = str(eq)
        data = PolyvoreDisjoint(protocol.parent, protocol, name)
        raw_cp = [l for l in (protocol / "compatibility_test.txt").read_text().splitlines() if l.strip()]
        raw_fitb = json.loads((protocol / "fill_in_blank_test.json").read_text())
        grouped = P.groups_for(protocol, data, "test", raw_cp, raw_fitb, grouping)
        if grouping == "two_way":
            (_, groups), (_, second) = grouped
        else:
            (_, groups), second = grouped, None
        mllm_path = args.ff / "mllm" / name / "results.jsonl"
        records = [json.loads(l) for l in mllm_path.read_text().splitlines() if l.strip()]
        qids = [int(r["question_id"]) for r in records]
        mllm_correct = np.array([float(r["correct"]) for r in records])
        columns, keys = [mllm_correct], [MLLM]
        inputs[name] = {"mllm_results_sha256": sha(mllm_path), "n_questions": len(qids),
                        "grouping": grouping, "equivalence_sha256": sha(eq)}
        for model in MODELS:
            path = args.ff / "results" / "predictions" / f"{name}__{model}__native__logistic.json"
            fitb = json.loads(path.read_text())["fitb"]
            correct_index = np.asarray(fitb["correct_index"])
            mismatch = [q for q, r in zip(qids, records) if correct_index[q] != r["correct_index"]]
            if mismatch:
                raise ValueError(f"{name}/{model}: question alignment failed for {len(mismatch)} ids")
            columns.append(np.asarray(fitb["correct"], dtype=float)[qids])
            keys.append(model)
            inputs[name][model] = sha(path)
        matrix = np.column_stack(columns)
        sub_groups = [groups[q] for q in qids]
        sub_second = None if second is None else [second[q] for q in qids]
        boot = cluster_bootstrap(matrix, metric="accuracy", groups=sub_groups,
                                 second_groups=sub_second, replicates=args.replicates, seed=args.seed)
        inference = pairwise_inference(boot, keys, metric="accuracy", margins=[0.005, 0.01, 0.02])
        pairs = [c for c in inference["comparisons"] if MLLM in (c["left"], c["right"])]
        adjusted = holm([c["bootstrap_p"] for c in pairs])
        for c, p_holm in zip(pairs, adjusted):
            probe = c["right"] if c["left"] == MLLM else c["left"]
            sign = 1.0 if c["left"] == probe else -1.0  # report probe minus MLLM
            k = keys.index(probe)
            lo, hi = sorted((sign * c["ci_low"], sign * c["ci_high"]))
            rows.append({
                "dataset": name, "probe": probe, "n_questions": len(qids),
                "clusters": boot["cluster_count"], "probe_acc": float(matrix[:, k].mean()),
                "mllm_acc": float(mllm_correct.mean()), "delta_probe_minus_mllm": sign * c["delta"],
                "ci_low": lo, "ci_high": hi, "bootstrap_p": c["bootstrap_p"], "holm_p": float(p_holm),
                "mcnemar_exact_p": exact_mcnemar(matrix[:, k], mllm_correct)["pvalue"],
                "mcnemar_cluster_p": cluster_adjusted_mcnemar(matrix[:, k], mllm_correct, sub_groups,
                                                              replicates=args.replicates,
                                                              seed=args.seed)["pvalue"],
                "replicates": args.replicates})
        mine = [r for r in rows if r["dataset"] == name]
        summary[name] = {"mllm_acc": float(mllm_correct.mean()),
                         "probes_above_mllm_holm": sum(r["holm_p"] < 0.05 and r["delta_probe_minus_mllm"] > 0 for r in mine),
                         "probes_below_mllm_holm": sum(r["holm_p"] < 0.05 and r["delta_probe_minus_mllm"] < 0 for r in mine),
                         "best_probe_on_subset": max(mine, key=lambda r: r["probe_acc"])["probe"]}
        print(name, json.dumps(summary[name]), flush=True)
    import csv
    with open(args.out / "paired_mllm_vs_probes.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps({
        "analysis": "paired MLLM zero-shot vs primary probe FITB on the frozen MLLM question subset",
        "holm_family": "per dataset, the 15 MLLM-vs-probe comparisons",
        "mcnemar_cluster_note": "cluster sign-flip McNemar uses the first grouping dimension only (Polyvore item component; IQON/POG user); the bootstrap CI and p use the full two-way grouping",
        "margins_tost": [0.005, 0.01, 0.02], "seed": args.seed, "replicates": args.replicates,
        "summary": summary, "inputs": inputs}, indent=2))


if __name__ == "__main__":
    main()
