"""Add the prospective IQON3000 exposure audit without erasing the original audit."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

IQON = {
    "resnet50": ("no", "unlikely", "no", "clean_provenance"),
    "dinov3_vitl16": (
        "no documented use",
        "possible incidental public-social-image exposure",
        "no documented use",
        "clean_provenance_with_web_uncertainty",
    ),
    "clip_vitl14_336": (
        "no documented use",
        "possible incidental undisclosed web exposure",
        "no documented use",
        "clean_provenance_with_web_uncertainty",
    ),
    "siglip2_b16_384": (
        "no documented use",
        "possible incidental WebLI exposure",
        "no documented use",
        "clean_provenance_with_web_uncertainty",
    ),
    "fashionclip2": (
        "no documented use",
        "possible only through undisclosed LAION membership; Farfetch fine-tuning is a distinct source",
        "no documented use",
        "clean_provenance_with_web_uncertainty",
    ),
    "marqo_fashionsiglip": (
        "not disclosed",
        "possibly_exposed: fashion fine-tuning sources are insufficiently enumerated",
        "no documented use",
        "possibly_exposed",
    ),
    "gr_lite": (
        "not documented",
        "possibly_exposed: 1.3M open-source fashion images are not enumerated item-by-item",
        "no documented use",
        "possibly_exposed",
    ),
}


def main() -> None:
    source = Path("artifacts/model_audit.csv")
    archive = Path("artifacts/archive/model_audit_pre_iqon.csv")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        shutil.copy2(source, archive)
    frame = pd.read_csv(source)
    for column, index in (
        ("known_iqon_training_exposure", 0),
        ("possible_iqon_exposure", 1),
        ("known_iqon_evaluation_exposure", 2),
        ("iqon_provenance_status", 3),
    ):
        frame[column] = frame.model_key.map({key: value[index] for key, value in IQON.items()})
    if set(frame.model_key) != set(IQON) or frame.iqon_provenance_status.isna().any():
        raise ValueError("IQON provenance mapping is incomplete")
    frame.to_csv(source, index=False)


if __name__ == "__main__":
    main()
