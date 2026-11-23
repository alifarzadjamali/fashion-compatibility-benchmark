"""Extract G3's surviving numerical reference from the pinned public README."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

from repbench.v2.state import write_json

V1_COMMIT = "83b4979dd950bdca63bb82854bd9d64d340ad243"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = subprocess.run(
        ["git", "show", f"{V1_COMMIT}:README.md"], capture_output=True, check=True
    ).stdout
    text = raw.decode()
    models = {}
    for display, key in [("ResNet50", "resnet50"), ("CLIP", "clip_vitl14_336")]:
        match = re.search(
            r"^\| " + display + r" \| (\.[0-9]+) \| (\.[0-9]+) \| (\.[0-9]+) \| (\.[0-9]+) \|",
            text,
            re.MULTILINE,
        )
        if not match:
            raise ValueError(f"Missing published reference: {display}")
        values = [float(v) for v in match.groups()]
        models[key] = dict(
            zip(
                ["polyvore_cp_auc", "polyvore_fitb", "iqon_cp_auc", "iqon_fitb"],
                values,
                strict=True,
            )
        )
    match = re.search(r"for Polyvore-D-Clean and ([\d,]+) / ([\d,]+) / ([\d,]+) for IQON", text)
    poly = re.search(r"are ([\d,]+) / ([\d,]+) / ([\d,]+)\s+for Polyvore", text)
    if not match or not poly:
        raise ValueError("Published outfit counts not found")
    write_json(
        args.output,
        {
            "source_commit": V1_COMMIT,
            "source_path": "README.md",
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "models": models,
            "outfit_counts_train_valid_test": {
                "polyvore_d_clean": [int(v.replace(",", "")) for v in poly.groups()],
                "iqon3000_clean": [int(v.replace(",", "")) for v in match.groups()],
            },
            "tolerances": {"cp_auc_absolute": 0.002, "fitb_absolute": 0.005},
            "rounding": "Published metrics have four decimal places; exact pilot values unavailable.",
            "original_split_hashes": None,
            "g3_amendment": "Owner approved two independent matching reconstructions, published counts and published metric comparisons; original split identity is unverifiable.",
        },
    )


if __name__ == "__main__":
    main()
