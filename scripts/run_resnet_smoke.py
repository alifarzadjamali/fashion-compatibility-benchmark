from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.encoders.torchvision_encoder import ResNet50Encoder
from repbench.features.outfit_features import outfit_feature
from repbench.features.pca import TrainOnlyPCA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/polyvore_outfits"))
    parser.add_argument("--items", type=int, default=300)
    args = parser.parse_args()
    dataset = PolyvoreDisjoint(args.root)
    item_ids = sorted(dataset.item_ids("train"))[: args.items]
    paths = [dataset.image_path(item_id) for item_id in item_ids]
    output = ResNet50Encoder().encode(paths, item_ids, batch_size=64)
    transformed = TrainOnlyPCA(256).fit(output.embeddings).transform(output.embeddings)
    mapping = dict(zip(item_ids, transformed))
    feature = outfit_feature(tuple(item_ids[:3]), mapping)
    report = {
        "items": len(item_ids),
        "raw_shape": list(output.embeddings.shape),
        "pca_shape": list(transformed.shape),
        "outfit_feature_shape": list(feature.shape),
        "finite": bool(np.isfinite(feature).all()),
        "device": output.metadata["device"],
        "images_per_second": output.metadata["images_per_second"],
    }
    target = Path("artifacts/resnet_smoke_report.json")
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
