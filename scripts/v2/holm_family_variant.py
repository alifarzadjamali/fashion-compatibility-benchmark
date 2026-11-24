"""Post-run sweep B8: the frozen protocol says "Holm within dataset x metric"; the pipeline
applies Holm per dataset x setting x learner x metric block (105 pairs each; 78 blocks).
This script re-applies Holm over the wider dataset x metric family (all settings and
learners pooled) to the same bootstrap p-values and reports both, unchanged inputs."""
import argparse
import json
from pathlib import Path

import pandas as pd

from repbench.eval.paired_statistics import holm

ap = argparse.ArgumentParser()
ap.add_argument("--pairwise", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
df = pd.read_csv(args.pairwise)
df["holm_dataset_metric"] = 0.0
for _, idx in df.groupby(["dataset", "metric"]).groups.items():
    df.loc[idx, "holm_dataset_metric"] = holm(df.loc[idx, "bootstrap_p"].tolist())
# analytic p-values have no resampling floor: DeLong for CP-AUC (ignores clustering),
# exact McNemar for FITB (ignores clustering); pooled dataset x metric Holm on those too
df["p_analytic"] = df["delong_p"].where(df.metric == "cp_auc", df["mcnemar_exact_p"])
df["holm_dataset_metric_analytic"] = 0.0
for _, idx in df.groupby(["dataset", "metric"]).groups.items():
    df.loc[idx, "holm_dataset_metric_analytic"] = holm(df.loc[idx, "p_analytic"].tolist())
prim = (df.setting == "native") & (df.learner == "logistic")
res = {
    "rows": int(len(df)),
    "blocks_as_run": int(df.groupby(["dataset", "setting", "learner", "metric"]).ngroups),
    "families_dataset_metric": int(df.groupby(["dataset", "metric"]).ngroups),
    "significant_as_run_block_holm": int((df.bootstrap_p_holm < 0.05).sum()),
    "significant_dataset_metric_holm": int((df.holm_dataset_metric < 0.05).sum()),
    "primary_block_as_run": int((df[prim].bootstrap_p_holm < 0.05).sum()),
    "primary_block_dataset_metric_holm": int((df[prim].holm_dataset_metric < 0.05).sum()),
    "primary_rows": int(prim.sum()),
    "bootstrap_p_floor": float(df.p_resolution_floor.min()),
    "pooled_family_size_max": int(df.groupby(["dataset", "metric"]).size().max()),
    "significant_dataset_metric_holm_analytic": int((df.holm_dataset_metric_analytic < 0.05).sum()),
    "primary_block_dataset_metric_holm_analytic": int((df[prim].holm_dataset_metric_analytic < 0.05).sum()),
    "primary_block_as_run_by_metric": {m: int(((df.metric == m) & prim & (df.bootstrap_p_holm < 0.05)).sum())
                                       for m in ("cp_auc", "fitb_acc")},
}
args.out.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out / "pairwise_holm_dataset_metric.csv", index=False)
(args.out / "summary.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
