from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

os.environ.setdefault("HF_HOME", str(Path(".venv/cache/huggingface").resolve()))

from transformers import AutoImageProcessor, AutoModel

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.eval.compatibility import cp_metrics
from repbench.features.outfit_features import outfit_matrix
from repbench.features.pca import TrainOnlyPCA
from repbench.models.logistic import LogisticCompatibility
from repbench.utils.seeds import seed_everything

SEED = 20260912
MODEL_KEY = "dinov2_vitb14"
CHECKPOINT = "facebook/dinov2-base"
CACHE = Path("data/embeddings/historical_polyvore_d") / MODEL_KEY
EXTENDED_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)


def load_cache(split: str) -> tuple[list[str], np.ndarray]:
    ids = json.loads((CACHE / f"{split}_item_ids.json").read_text(encoding="utf-8"))
    return ids, np.load(CACHE / f"{split}.npy").astype(np.float32, copy=False)


def score_variant(
    dataset: PolyvoreDisjoint,
    name: str,
    split_embeddings: dict[str, np.ndarray],
    c_grid=(0.01, 0.1, 1.0, 10.0),
):
    mappings = {
        split: dict(zip(load_cache(split)[0], values)) for split, values in split_embeddings.items()
    }
    examples = {split: dataset.compatibility(split) for split in dataset.SPLITS}
    features = {
        split: outfit_matrix([row.item_ids for row in examples[split]], mappings[split])
        for split in dataset.SPLITS
    }
    labels = {
        split: np.asarray([row.label for row in examples[split]], dtype=np.int8)
        for split in dataset.SPLITS
    }
    model = LogisticCompatibility(c_grid=c_grid, seed=SEED).fit(
        features["train"], labels["train"], features["valid"], labels["valid"]
    )
    cp = cp_metrics(labels["test"], model.predict_proba(features["test"]))
    fitb = dataset.fitb("test")
    completed = [
        question.question_item_ids + (candidate,)
        for question in fitb
        for candidate in question.candidate_item_ids
    ]
    scores = model.predict_proba(outfit_matrix(completed, mappings["test"])).reshape(len(fitb), 4)
    predictions = scores.argmax(axis=1)
    return {
        "variant": name,
        "dimension": next(iter(split_embeddings.values())).shape[1],
        "cp_auc": cp["roc_auc"],
        "pr_auc": cp["pr_auc"],
        "fitb_acc": float(np.mean(predictions == [question.correct_index for question in fitb])),
        "validation_auc": model.validation_auc_,
        "best_c": model.best_c_,
        "validation_curve": model.validation_curve_,
    }


def pooling_check(dataset: PolyvoreDisjoint) -> dict:
    device = torch.device("cuda")
    processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
    model = AutoModel.from_pretrained(CHECKPOINT, dtype=torch.float16).eval().to(device)
    model.requires_grad_(False)
    ids = sorted(dataset.item_ids("train"))[:8]
    images = []
    for item_id in ids:
        with Image.open(dataset.image_path(item_id)) as image:
            images.append(image.convert("RGB"))
    inputs = processor(images=images, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**{key: value.to(device) for key, value in inputs.items()})
    cached_ids, cached = load_cache("train")
    cached_map = dict(zip(cached_ids, cached))
    direct = torch.nn.functional.normalize(outputs.pooler_output.float(), dim=1).cpu().numpy()
    expected = np.stack([cached_map[item_id] for item_id in ids])
    return {
        "processor_tensor_shape": list(inputs["pixel_values"].shape),
        "pooler_equals_cls_max_abs": float(
            (outputs.pooler_output - outputs.last_hidden_state[:, 0]).abs().max().item()
        ),
        "direct_vs_cache_max_abs": float(np.max(np.abs(direct - expected))),
        "direct_vs_cache_mean_cosine": float(np.mean(np.sum(direct * expected, axis=1))),
        "model_training_mode": bool(model.training),
        "parameter_requires_grad_after_load": bool(
            any(p.requires_grad for p in model.parameters())
        ),
    }


def main() -> None:
    seed_everything(SEED)
    dataset = PolyvoreDisjoint("data/raw/polyvore_outfits")
    raw = {split: load_cache(split)[1] for split in dataset.SPLITS}
    standardized = TrainOnlyPCA(256, seed=SEED, standardize=True)
    standardized_values = dict(
        zip(dataset.SPLITS, standardized.fit_transform_splits(*(raw[s] for s in dataset.SPLITS)))
    )
    unstandardized = TrainOnlyPCA(256, seed=SEED, standardize=False)
    unstandardized_values = dict(
        zip(dataset.SPLITS, unstandardized.fit_transform_splits(*(raw[s] for s in dataset.SPLITS)))
    )
    rows = [
        score_variant(dataset, "pca256_standardized_pilot", standardized_values),
        score_variant(
            dataset,
            "pca256_standardized_extended_c_diagnostic",
            standardized_values,
            c_grid=EXTENDED_C_GRID,
        ),
        score_variant(
            dataset,
            "pca256_without_standardization_extended_c",
            unstandardized_values,
            c_grid=EXTENDED_C_GRID,
        ),
        score_variant(dataset, "native_l2_768_extended_c", raw, c_grid=EXTENDED_C_GRID),
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv("artifacts/dinov2_sanity_variants.csv", index=False)
    report = {
        "protocol": "historical_polyvore_d_diagnostic_only",
        "checkpoint": CHECKPOINT,
        "revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        "pooling_check": pooling_check(dataset),
        "pca_explained_variance": {
            "standardized": standardized.explained_variance_ratio,
            "without_standardization": unstandardized.explained_variance_ratio,
        },
        "variants": rows,
    }
    Path("artifacts/dinov2_sanity_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
