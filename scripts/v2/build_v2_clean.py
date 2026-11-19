"""Build a v2-Clean protocol twice and audit it (owner decision E, 2026-11-19).

v2-Clean = the v1 clean construction of each dataset plus near-duplicate image
equivalence: items whose images are model-free near duplicates
(``scripts/v2/image_equivalence.py``: pHash Hamming <= 2 AND pixel-level colour/shape
agreement) form one indivisible unit and can never be split across train/valid/test.

Datasets and builders (each called with ``image_equivalence=<groups>``):
  polyvore_v2_clean  repbench.data.clean_protocol.construct_clean_protocol (--raw-root)
  iqon3000_v2_clean  repbench.data.iqon.construct_iqon_clean (--catalog/--source-audit/--archive)
  pog_v2_clean       repbench.v2.pog.construct_pog_clean (--items/--outfits/--download-manifest)

Steps: verify the equivalence file against its report, build into <output>/build_a and
<output>/build_b, require SHA-256-identical files, run the existing audits (Polyvore:
inspect_clean_protocol; IQON: manifest split_overlap; POG: audit_protocol) plus a
uniform audit (cross-split item IDs, exact image identities, outfit tuples, CP/FITB
collision rules) and the v2 audit that no equivalence group -- and no merged
exact+near-duplicate identity -- spans two splits. Writes <output>/build_report.json.
Phase 1 data construction only: no model is run and no test outcome is read.
Nothing is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))  # works under python -I

from repbench.data.clean_protocol import SPLITS, exact_file_duplicate_groups
from repbench.data.image_equivalence import (
    groups_spanning_splits,
    load_image_equivalence,
    merge_identities,
)
from repbench.data.polyvore import PolyvoreDisjoint

DATASETS = {
    "polyvore_v2_clean": "polyvore",
    "iqon3000_v2_clean": "iqon",
    "pog_v2_clean": "pog",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def folder_digest(folder: Path) -> dict[str, str]:
    return {p.name: file_sha256(p) for p in sorted(Path(folder).iterdir()) if p.is_file()}


# ---------------------------------------------------------------- equivalence input


def load_equivalence(groups_path: Path, report_path: Path, kind: str) -> tuple[dict, dict]:
    mapping, info = load_image_equivalence(groups_path)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("kind") != "v2_image_equivalence_groups":
        raise ValueError(f"{report_path} is not an image-equivalence report")
    if report.get("dataset") != kind:
        raise ValueError(f"Equivalence report is for {report.get('dataset')!r}, not {kind!r}")
    expected = report["outputs_sha256"]["equivalence_groups.json"]
    if expected != info["sha256"]:
        raise ValueError("Equivalence groups file does not match its report (sha256)")
    info.update(
        {
            "report_file": Path(report_path).name,
            "report_sha256": file_sha256(report_path),
            "rule": report["rule"],
            "thresholds": report["thresholds"],
            "signature_version": report["signature_version"],
            "input_sha256": report["input_sha256"],
            "input_items_hashed": report["input_items_hashed"],
            "unreadable_images": len(report["unreadable_images"]),
            "pairs": report["pairs"],
            "groups": report["groups"],
        }
    )
    return mapping, info


# ---------------------------------------------------------------- uniform audit


def uniform_audit(protocol: Path, category: dict, identity: dict, name: str) -> dict:
    """Cross-split item/identity/outfit disjointness and CP/FITB collision rules."""
    dataset = PolyvoreDisjoint(protocol, protocol, name)
    items = {split: dataset.item_ids(split) for split in SPLITS}
    tuples = {
        split: [tuple(sorted(o.item_ids)) for o in dataset.outfits(split)] for split in SPLITS
    }
    report: dict = {"cross_split": {}, "splits": {}}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1 :]:
            report["cross_split"][f"{left}_{right}"] = {
                "item_ids": len(items[left] & items[right]),
                "image_identities": len(
                    {identity[x] for x in items[left]} & {identity[x] for x in items[right]}
                ),
                "outfit_tuples": len(set(tuples[left]) & set(tuples[right])),
            }
    ok = all(v == 0 for row in report["cross_split"].values() for v in row.values())
    all_positive = set().union(*(set(v) for v in tuples.values()))
    for split in SPLITS:
        positives = set(tuples[split])
        cp = dataset.compatibility(split)
        pos = [row for row in cp if row.label == 1]
        neg = [row for row in cp if row.label == 0]
        neg_sets = [tuple(sorted(row.item_ids)) for row in neg]
        matched = sum(
            sorted(category[x] for x in p.item_ids) == sorted(category[x] for x in n.item_ids)
            for p, n in zip(pos, neg, strict=True)
        )
        invalid = 0
        for q in dataset.fitb(split):
            cands = q.candidate_item_ids
            correct = cands[q.correct_index]
            invalid += not (
                len(cands) == 4
                and len(set(cands)) == 4
                and tuple(sorted(q.question_item_ids + (correct,))) in positives
                and all(c in items[split] for c in cands)
                and all(category[c] == category[correct] for c in cands)
                and not any(
                    tuple(sorted(q.question_item_ids + (c,))) in all_positive
                    for k, c in enumerate(cands)
                    if k != q.correct_index
                )
            )
        checks = {
            "outfits": len(tuples[split]),
            "duplicate_outfit_tuples": len(tuples[split]) - len(positives),
            "cp_balanced": len(pos) == len(neg) == len(tuples[split]),
            "cp_category_and_length_matched": matched == len(neg),
            "cp_negative_duplicates": len(neg_sets) - len(set(neg_sets)),
            "cp_negative_equals_positive": len(set(neg_sets) & all_positive),
            "fitb_invalid": invalid,
        }
        report["splits"][split] = checks
        ok &= (
            checks["duplicate_outfit_tuples"] == 0
            and checks["cp_balanced"]
            and checks["cp_category_and_length_matched"]
            and checks["cp_negative_duplicates"] == 0
            and checks["cp_negative_equals_positive"] == 0
            and checks["fitb_invalid"] == 0
        )
    report["passes"] = bool(ok)
    return report


def equivalence_audit(protocol: Path, name: str, exact: dict, mapping: dict) -> dict:
    """No near-duplicate group and no merged exact+near identity spans two splits."""
    dataset = PolyvoreDisjoint(protocol, protocol, name)
    items = {split: dataset.item_ids(split) for split in SPLITS}
    every = set().union(*items.values())
    merged = merge_identities({x: exact[x] for x in every}, mapping)
    owners = defaultdict(set)
    for split, values in items.items():
        for x in values:
            owners[merged[x]].add(split)
    spanning = sorted(k for k, v in owners.items() if len(v) > 1)
    groups = groups_spanning_splits(items, mapping)
    return {
        **groups,
        "merged_identities_spanning_splits": len(spanning),
        "merged_examples": spanning[:20],
        "passes": groups["groups_spanning_splits"] == 0 and not spanning,
    }


# ---------------------------------------------------------------- per dataset


def build(args, destination: Path, mapping: dict, info: dict) -> dict:
    kind = DATASETS[args.dataset]
    if kind == "polyvore":
        from repbench.data.clean_protocol import construct_clean_protocol

        return construct_clean_protocol(
            args.raw_root,
            destination,
            seed=args.seed,
            image_equivalence=mapping,
            image_equivalence_info=info,
            protocol_name=args.dataset,
        )
    if kind == "iqon":
        from repbench.data.iqon import construct_iqon_clean

        return construct_iqon_clean(
            args.catalog,
            destination,
            args.source_audit,
            args.archive,
            seed=args.seed,
            image_equivalence=mapping,
            image_equivalence_info=info,
            protocol_name=args.dataset,
            # v2-Clean: merged near-duplicate units cannot fill v1's locked capacities;
            # fill to capacity and disclose counts (same principle as POG, decision A/E).
            allow_underfill=True,
        )
    from repbench.v2.pog import construct_pog_clean

    category_map = None
    if args.category_map is not None:
        category_map = json.loads(args.category_map.read_text(encoding="utf-8"))
    return construct_pog_clean(
        args.items,
        args.outfits,
        destination,
        download_manifest=args.download_manifest,
        user_clicks=args.user_clicks,
        category_map=category_map,
        category_map_name=args.category_map.name if args.category_map else "identity_cate_id",
        seed=args.seed,
        image_equivalence=mapping,
        image_equivalence_info=info,
        protocol_name=args.dataset,
    )


def dataset_audits(args, protocol: Path, manifest: dict, mapping: dict) -> dict:
    kind = DATASETS[args.dataset]
    audits: dict = {}
    if kind == "polyvore":
        from repbench.data.validation import inspect_clean_protocol

        existing = inspect_clean_protocol(PolyvoreDisjoint(args.raw_root, protocol, args.dataset))
        audits["existing_inspect_clean_protocol"] = {
            "passes": existing["passes_clean_protocol"],
            "item_overlap_counts": existing["item_overlap_counts"],
            "global_duplicate_outfits": existing["global_duplicate_outfits"],
            "missing_images": {s: r["missing_images"] for s, r in existing["splits"].items()},
        }
        metadata = json.loads(
            (args.raw_root / "polyvore_item_metadata.json").read_text(encoding="utf-8")
        )
        items = {
            str(item["item_id"])
            for split in SPLITS
            for row in json.loads((protocol / f"{split}.json").read_text(encoding="utf-8"))
            for item in row["items"]
        }
        category = {x: str(metadata[x]["semantic_category"]).strip() for x in items}
        exact = {x: "item:" + x for x in items}
        for group in exact_file_duplicate_groups(args.raw_root, items):
            for x in group:
                exact[x] = "sha_group:" + group[0]
    elif kind == "iqon":
        import pandas as pd

        overlap = manifest["split_overlap"]
        audits["existing_manifest_split_overlap"] = {
            "passes": all(v == 0 for row in overlap.values() for v in row.values()),
            "split_overlap": overlap,
        }
        rows = {
            split: json.loads((protocol / f"{split}.json").read_text(encoding="utf-8"))
            for split in SPLITS
        }
        category = {
            str(i["item_id"]): i["semantic_category"]
            for values in rows.values()
            for row in values
            for i in row["items"]
        }
        images = pd.read_parquet(
            args.catalog / "image_audit_resolved.parquet",
            columns=["item_id", "decoded_rgb_sha256"],
        )
        images["item_id"] = images.item_id.astype(str)
        decoded = (
            images.drop_duplicates("item_id").set_index("item_id").decoded_rgb_sha256.to_dict()
        )
        exact = {x: "decoded:" + str(decoded[x]) for x in category}
    else:
        from repbench.v2.pog import audit_protocol

        existing = audit_protocol(protocol)
        audits["existing_audit_protocol"] = {
            "passes": existing["passes"],
            "cross_split": existing["cross_split"],
        }
        metadata = json.loads(
            (protocol / "polyvore_item_metadata.json").read_text(encoding="utf-8")
        )
        category = {x: row["semantic_category"] for x, row in metadata.items()}
        exact = {x: row.get("exact_image_identity", row["image_identity"]) for x, row in metadata.items()}
    audits["uniform_exact_image"] = uniform_audit(protocol, category, exact, args.dataset)
    audits["image_equivalence_not_split"] = equivalence_audit(protocol, args.dataset, exact, mapping)
    audits["passes"] = all(
        value["passes"] for key, value in audits.items() if isinstance(value, dict)
    )
    return audits


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--image-equivalence", type=Path, required=True,
                        help="equivalence_groups.json from scripts/v2/image_equivalence.py")
    parser.add_argument("--equivalence-report", type=Path, required=True,
                        help="the matching equivalence_report.json")
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--raw-root", type=Path, help="polyvore: raw root (disjoint/, images/, metadata)")
    parser.add_argument("--catalog", type=Path, help="iqon: catalog dir")
    parser.add_argument("--source-audit", type=Path, help="iqon: source audit JSON")
    parser.add_argument("--archive", type=Path, help="iqon: source archive")
    parser.add_argument("--items", type=Path, help="pog: item_data.txt")
    parser.add_argument("--outfits", type=Path, help="pog: outfit_data.txt")
    parser.add_argument("--download-manifest", type=Path, help="pog: manifest.jsonl")
    parser.add_argument("--user-clicks", type=Path, help="pog: click summary JSON")
    parser.add_argument("--category-map", type=Path, help="pog: optional {cate_id: category}")
    args = parser.parse_args(argv)
    kind = DATASETS[args.dataset]
    need = {
        "polyvore": ["raw_root"],
        "iqon": ["catalog", "source_audit", "archive"],
        "pog": ["items", "outfits", "download_manifest"],
    }[kind]
    missing = [n for n in need if getattr(args, n) is None]
    if missing:
        parser.error(f"{args.dataset} needs " + ", ".join("--" + n.replace("_", "-") for n in missing))
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    mapping, info = load_equivalence(args.image_equivalence, args.equivalence_report, kind)
    args.output.mkdir(parents=True)
    manifests, timings, audits = {}, {}, {}
    for label in ("build_a", "build_b"):
        started = time.monotonic()
        manifests[label] = build(args, args.output / label, mapping, info)
        timings[f"{label}_seconds"] = round(time.monotonic() - started, 1)
        print(f"{label} built in {timings[f'{label}_seconds']} s", flush=True)
    digest_a = folder_digest(args.output / "build_a")
    digest_b = folder_digest(args.output / "build_b")
    identical = digest_a == digest_b
    started = time.monotonic()
    audits = dataset_audits(args, args.output / "build_a", manifests["build_a"], mapping)
    timings["audit_seconds"] = round(time.monotonic() - started, 1)
    manifest = manifests["build_a"]
    passed = identical and audits["passes"]
    report = {
        "stage": "v2-Clean protocol construction (two deterministic builds)",
        "dataset": args.dataset,
        "status": "PASS" if passed else "FAIL",
        "builds_hash_identical": identical,
        "mismatched_files": sorted(
            n for n in set(digest_a) | set(digest_b) if digest_a.get(n) != digest_b.get(n)
        ),
        "file_sha256": digest_a,
        "image_equivalence": manifest["image_equivalence"],
        "splits": {s: manifest["splits"][s]["outfits"] for s in SPLITS},
        "audits": audits,
        "timings": timings,
        "note": "No model run; no test outcome read. v1-Clean splits are kept for the "
        "leakage-inflation comparison only.",
    }
    with (args.output / "build_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps({k: report[k] for k in ("status", "splits", "timings")}, indent=1))
    if not passed:
        raise SystemExit("v2-Clean builds differ or audits failed")


if __name__ == "__main__":
    main()
