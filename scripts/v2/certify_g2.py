"""Combine unchanged, validated G2 successes across immutable retry directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from g2_models import ALL_MODELS, signature

from repbench.v2.state import input_hashes, require_gate, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1", type=Path, required=True)
    parser.add_argument("--attempts", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_gate(args.g1, "G1")
    evidence = {}
    for key, spec in ALL_MODELS.items():
        for directory in args.attempts:
            path = directory / f"{key}.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            if (
                record.get("status") == "PASS"
                and record.get("signature") == signature(spec)
                and record.get("checks")
                and all(record["checks"].values())
            ):
                evidence[key] = {"path": str(path), "sha256": sha256(path)}
                break
    missing = sorted(set(ALL_MODELS) - evidence.keys())
    write_json(
        args.output,
        {
            "gate": "G2",
            "status": "FAIL" if missing else "PASS",
            "input_sha256": input_hashes(),
            "models": evidence,
            "missing_or_stale": missing,
            "g1_evidence": {"path": str(args.g1), "sha256": sha256(args.g1)},
        },
    )
    print(f"G2: {len(evidence)}/{len(ALL_MODELS)} validated; missing/stale={missing}")
    raise SystemExit(bool(missing))


if __name__ == "__main__":
    main()
