from __future__ import annotations

import argparse
import json
from pathlib import Path

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.data.validation import inspect_clean_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/polyvore_outfits"))
    parser.add_argument("--protocol", type=Path, default=Path("data/protocols/polyvore_d_clean"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/polyvore_d_clean_validation.json")
    )
    args = parser.parse_args()
    dataset = PolyvoreDisjoint(args.root, args.protocol, "polyvore_d_clean")
    report = inspect_clean_protocol(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passes_clean_protocol"]:
        raise SystemExit("Clean protocol failed validation")


if __name__ == "__main__":
    main()
