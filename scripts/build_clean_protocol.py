from __future__ import annotations

import argparse
import json
from pathlib import Path

from repbench.data.clean_protocol import construct_clean_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/polyvore_outfits"))
    parser.add_argument("--output", type=Path, default=Path("data/protocols/polyvore_d_clean"))
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    manifest = construct_clean_protocol(args.root, args.output, args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
