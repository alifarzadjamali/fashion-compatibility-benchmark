from __future__ import annotations

import argparse
import json
from pathlib import Path

from repbench.data.a100 import prepare_a100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path(".venv/a100_audit/A100.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/protocols/a100"))
    parser.add_argument("--images", type=Path, default=Path(".venv/a100/images"))
    args = parser.parse_args()
    print(json.dumps(prepare_a100(args.archive, args.output, args.images), indent=2))


if __name__ == "__main__":
    main()
