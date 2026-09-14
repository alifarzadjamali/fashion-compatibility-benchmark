from __future__ import annotations

import pandas as pd
from scipy.stats import kendalltau, spearmanr


def main() -> None:
    primary = pd.read_csv("artifacts/primary_results.csv")
    native = pd.read_csv("artifacts/native_results.csv")
    columns = ["representation", "cp_auc", "pr_auc", "fitb_acc"]
    comparison = primary[columns].merge(native[columns], on="representation", suffixes=("_pca256", "_native"))
    for metric in ("cp_auc", "pr_auc", "fitb_acc"):
        comparison[f"{metric}_native_minus_pca256"] = (
            comparison[f"{metric}_native"] - comparison[f"{metric}_pca256"]
        )
    comparison["cp_rank_pca256"] = comparison.cp_auc_pca256.rank(
        method="min", ascending=False
    ).astype(int)
    comparison["cp_rank_native"] = comparison.cp_auc_native.rank(
        method="min", ascending=False
    ).astype(int)
    comparison["fitb_rank_pca256"] = comparison.fitb_acc_pca256.rank(
        method="min", ascending=False
    ).astype(int)
    comparison["fitb_rank_native"] = comparison.fitb_acc_native.rank(
        method="min", ascending=False
    ).astype(int)
    comparison.to_csv("artifacts/native_robustness.csv", index=False)
    rows = []
    for metric in ("cp", "fitb"):
        pca_rank = comparison[f"{metric}_rank_pca256"]
        native_rank = comparison[f"{metric}_rank_native"]
        rho = spearmanr(pca_rank, native_rank)
        tau = kendalltau(pca_rank, native_rank)
        rows.append(
            {
                "metric": metric,
                "n_models": len(comparison),
                "spearman_rho": rho.statistic,
                "spearman_p": rho.pvalue,
                "kendall_tau": tau.statistic,
                "kendall_p": tau.pvalue,
            }
        )
    pd.DataFrame(rows).to_csv("artifacts/native_rank_stability.csv", index=False)


if __name__ == "__main__":
    main()
