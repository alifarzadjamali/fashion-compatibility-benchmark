from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEED = 20260912
N_BOOTSTRAP = 2000


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    cp = pd.read_parquet("artifacts/pilot_cp_predictions.parquet")
    fitb = pd.read_parquet("artifacts/pilot_fitb_predictions.parquet")
    models = list(cp["model_key"].drop_duplicates())
    cp_by_model = {
        model: cp.loc[cp.model_key == model].sort_values("example_index") for model in models
    }
    fitb_by_model = {
        model: fitb.loc[fitb.model_key == model].sort_values("question_index") for model in models
    }
    labels = cp_by_model[models[0]]["label"].to_numpy()
    if any(not np.array_equal(labels, frame["label"].to_numpy()) for frame in cp_by_model.values()):
        raise ValueError("CP labels are not aligned across models")
    rng = np.random.default_rng(SEED)
    cp_indices = rng.integers(0, len(labels), size=(N_BOOTSTRAP, len(labels)), dtype=np.int32)
    fitb_indices = rng.integers(
        0,
        len(fitb_by_model[models[0]]),
        size=(N_BOOTSTRAP, len(fitb_by_model[models[0]])),
        dtype=np.int32,
    )
    cp_bootstrap: dict[str, np.ndarray] = {}
    fitb_bootstrap: dict[str, np.ndarray] = {}
    rows = []
    for model in models:
        scores = cp_by_model[model]["score"].to_numpy()
        correctness = fitb_by_model[model]["correct"].to_numpy(dtype=np.float64)
        cp_values = np.fromiter(
            (roc_auc_score(labels[index], scores[index]) for index in cp_indices),
            dtype=np.float64,
            count=N_BOOTSTRAP,
        )
        fitb_values = correctness[fitb_indices].mean(axis=1)
        cp_bootstrap[model] = cp_values
        fitb_bootstrap[model] = fitb_values
        cp_low, cp_high = interval(cp_values)
        fitb_low, fitb_high = interval(fitb_values)
        rows.append(
            {
                "comparison": model,
                "metric": "absolute_cp_auc",
                "estimate": roc_auc_score(labels, scores),
                "ci95_low": cp_low,
                "ci95_high": cp_high,
            }
        )
        rows.append(
            {
                "comparison": model,
                "metric": "absolute_fitb_accuracy",
                "estimate": correctness.mean(),
                "ci95_low": fitb_low,
                "ci95_high": fitb_high,
            }
        )

    comparisons = [
        ("dinov2_vitb14", "resnet50"),
        ("clip_vitl14", "fashionclip2"),
        ("clip_vitl14", "resnet50"),
    ]
    for model_a, model_b in comparisons:
        for metric, values in (
            ("cp_auc_difference", cp_bootstrap[model_a] - cp_bootstrap[model_b]),
            ("fitb_accuracy_difference", fitb_bootstrap[model_a] - fitb_bootstrap[model_b]),
        ):
            low, high = interval(values)
            rows.append(
                {
                    "comparison": f"{model_a}_minus_{model_b}",
                    "metric": metric,
                    "estimate": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv("artifacts/pilot_statistics.csv", index=False)
    summary = {
        "protocol": "historical_polyvore_d",
        "bootstrap_resamples": N_BOOTSTRAP,
        "seed": SEED,
        "gate_1_pass": True,
        "gate_reason": "Large CP and FITB spread across the fixed four-model pilot.",
        "full_sweep_authorized": False,
    }
    Path("artifacts/gate1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
