"""Historical Polyvore-D item-overlap counts, compared with the counts reported by
lee-research/outfit-completion-retrieval (train/valid 3,781; train/test 84;
valid/test 34). Reads the local raw archive only; the lee-research repository itself
is not available locally and is not downloaded, so this is a count-level check."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

REFERENCE = {"train&valid": 3781, "train&test": 84, "valid&test": 34}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = {}
    with zipfile.ZipFile(args.archive) as archive:
        for split in ("train", "valid", "test"):
            name = next(n for n in archive.namelist() if n.endswith(f"/disjoint/{split}.json"))
            outfits = json.loads(archive.read(name))
            items[split] = {str(i["item_id"]) for o in outfits for i in o["items"]}
    local = {f"{a}&{b}": len(items[a] & items[b])
             for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    record = {
        "archive": str(args.archive),
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "unique_items": {k: len(v) for k, v in items.items()},
        "local_overlap": local, "lee_research_reference_quoted_in_handoff": REFERENCE,
        "match": local == REFERENCE,
        "note": "lee-research item lists not available locally; count-level check only",
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
