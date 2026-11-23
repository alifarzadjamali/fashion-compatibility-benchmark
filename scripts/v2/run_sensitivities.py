"""v2 sensitivity and v1-strength analyses over v2 pipeline outputs.

Subcommands (each writes a new directory under --out; completed files are never
overwritten; all randomness is seeded):

  label-fractions     primary probe on nested 10/25/50/100% TRAIN CP pairs
  construction-seeds  regenerate CP negatives / FITB candidates (seeds 42-46) on fixed
                      partitions, collision audit, primary probe, total uncertainty
  provenance          pairwise / rank statistics without the provenance-uncertain group
  subgroups           outfit-length / category subgroups and failure consensus
  near-duplicates     primary metrics excluding near-duplicate-affected test outfits
  lookbench           LookBench v3 rank correlation with primary CP AUC / FITB
  all                 every subcommand above for one results directory

Datasets use the pipeline's ``name=protocol_dir=grouping`` list; prediction-based
subcommands default to the datasets recorded in <results>/manifest.json. The evaluation
split defaults to the results manifest's eval split (else valid); ``test`` requires
--frozen-protocol-hash exactly as scripts/v2/pipeline.py.

``--dataset-config`` JSON (optional), keyed by dataset name:
  {"polyvore_v2_clean": {"identity": "polyvore_raw", "raw_root": "...",
                         "image_equivalence": ".../equivalence_groups.json",
                         "categories_file": ".../polyvore_item_metadata.json",
                         "affected_test_outfits": ".../affected_test_outfits.json"},
   "iqon3000_v2_clean": {"identity": "iqon_catalog", "catalog": "...",
                         "image_equivalence": "..."},
   "pog_v2_clean": {}}
(identity: metadata | file | polyvore_raw | iqon_catalog | item; see
repbench.v2.sensitivities.question_identity; ``base_seed`` overrides the manifest seed.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repbench.v2 import sensitivities as S


def _common(parser):
    parser.add_argument("--out", type=Path, help="output root (default <results>/sensitivities)")
    parser.add_argument("--results", type=Path, help="pipeline results directory")
    parser.add_argument("--datasets", help="name=protocol_dir=grouping,... (default: results manifest)")
    parser.add_argument("--eval-split", help="default: results manifest eval_split, else valid")
    parser.add_argument("--frozen-protocol-hash", type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=S.PROBE_SEED)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--equivalence", nargs="*", default=[],
                        help="name=equivalence_groups.json (same as scripts/v2/pipeline.py)")


def _probe_args(parser):
    parser.add_argument("--models", help="comma list (default: models with primary predictions)")
    parser.add_argument("--embeddings", type=Path, help="embedding root <root>/<dataset>/<model>/")
    parser.add_argument("--selection-split", choices=["carveout", "valid"],
                        help="as the pipeline: default valid when evaluating test, else carve-out")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {}
    for name in ("label-fractions", "construction-seeds", "provenance", "subgroups",
                 "near-duplicates", "lookbench", "all"):
        commands[name] = sub.add_parser(name)
        if name != "lookbench":
            _common(commands[name])
    for name in ("label-fractions", "construction-seeds", "all"):
        _probe_args(commands[name])
    for name in ("label-fractions", "all"):
        commands[name].add_argument("--fractions", default=",".join(map(str, S.LABEL_FRACTIONS)))
        commands[name].add_argument("--fraction-seed", type=int, default=S.FRACTION_SEED)
    for name in ("construction-seeds", "all"):
        commands[name].add_argument("--construction-seeds",
                                    default=",".join(map(str, S.CONSTRUCTION_SEEDS)))
        commands[name].add_argument("--no-verify-base", action="store_true",
                                    help="skip proof that inputs reproduce the base questions "
                                         "(recorded in the manifest)")
    for name in ("provenance", "all"):
        commands[name].add_argument("--exclude", help="comma list (default: protocol config group)")
        commands[name].add_argument("--protocol-config", type=Path)
    for name in ("subgroups", "all"):
        commands[name].add_argument("--subgroup-replicates", type=int, default=1000)
        commands[name].add_argument("--min-cp", type=int, default=200)
        commands[name].add_argument("--min-fitb", type=int, default=100)
    for name in ("near-duplicates", "all"):
        commands[name].add_argument("--affected", action="append", default=[],
                                    help="name=affected_test_outfits.json (repeatable)")
        commands[name].add_argument("--affected-key", default="any_method")
    lookbench = commands["lookbench"]
    lookbench.add_argument("--metrics", type=Path, required=True)
    lookbench.add_argument("--out", type=Path, required=True)
    lookbench.add_argument("--replicates", type=int, default=2000)
    lookbench.add_argument("--seed", type=int, default=S.PROBE_SEED)
    for parser_ in (lookbench, commands["all"]):
        parser_.add_argument("--lookbench-scores", type=Path,
                             default=S.ROOT / "configs" / "lookbench_v3_scores.json")
    return parser


def _resolve(args):
    if args.out is None:
        if args.results is None:
            raise SystemExit("--out or --results is required")
        args.out = args.results / "sensitivities"
    if args.datasets:
        specs = S.parse_dataset_specs(args.datasets)
    elif args.results:
        specs = S.specs_from_manifest(args.results)
    else:
        raise SystemExit("--datasets or --results (with manifest.json) is required")
    split = args.eval_split or (S.manifest_split(args.results) if args.results else None) or "valid"
    S.check_eval_split(split, args.frozen_protocol_hash)
    if getattr(args, "selection_split", None) == "valid" and split == "valid":
        raise SystemExit("Cannot select on the split being evaluated")
    configs = json.loads(args.dataset_config.read_text()) if args.dataset_config else {}
    return specs, split, configs


def _models(args, specs):
    if args.models:
        return args.models.split(",")
    if not args.results:
        raise SystemExit("--models or --results is required")
    found = set()
    for spec in specs:
        found |= set(S.load_predictions(args.results, spec.name, *S.PRIMARY))
    if not found:
        raise SystemExit("No primary predictions found; pass --models")
    return sorted(found)


def _need(args, *names):
    missing = [n for n in names if getattr(args, n) is None]
    if missing:
        raise SystemExit(f"{args.command} requires " + ", ".join("--" + m.replace("_", "-")
                                                                 for m in missing))


def run(args):
    command = args.command
    if command == "lookbench":
        S.run_lookbench(args.metrics, args.lookbench_scores, args.out,
                        replicates=args.replicates, seed=args.seed)
        return
    specs, split, configs = _resolve(args)
    todo = (["label-fractions", "construction-seeds", "provenance", "subgroups",
             "near-duplicates", "lookbench"] if command == "all" else [command])
    if command == "all":
        _need(args, "results", "embeddings")
    done = {}
    for step in todo:
        out = args.out / step.replace("-", "_")
        if (out / "manifest.json").exists():
            print(f"[skip] {step}: {out} already complete", flush=True)
            continue
        print(f"[run] {step} -> {out} (split={split})", flush=True)
        if step == "label-fractions":
            _need(args, "embeddings")
            S.run_label_fractions(specs, _models(args, specs), args.embeddings, out, split=split,
                                  fractions=[float(f) for f in args.fractions.split(",")],
                                  fraction_seed=args.fraction_seed, seed=args.seed,
                                  replicates=args.replicates, selection_split=args.selection_split)
        elif step == "construction-seeds":
            _need(args, "embeddings")
            S.run_construction_seeds(specs, _models(args, specs), args.embeddings, out, split=split,
                                     seeds=[int(s) for s in args.construction_seeds.split(",")],
                                     configs=configs, seed=args.seed, replicates=args.replicates,
                                     verify_base=not args.no_verify_base,
                                     selection_split=args.selection_split)
        elif step == "provenance":
            _need(args, "results")
            exclude = (tuple(args.exclude.split(",")) if args.exclude
                       else S.provenance_group(args.protocol_config))
            S.run_provenance(args.results, specs, out, split=split, exclude=exclude,
                             replicates=args.replicates, seed=args.seed)
        elif step == "subgroups":
            _need(args, "results")
            S.run_subgroups(args.results, specs, out, split=split, configs=configs,
                            replicates=args.subgroup_replicates, seed=args.seed,
                            min_cp=args.min_cp, min_fitb=args.min_fitb)
        elif step == "near-duplicates":
            _need(args, "results")
            affected = {name: cfg["affected_test_outfits"] for name, cfg in configs.items()
                        if cfg.get("affected_test_outfits")}
            affected.update(dict(a.split("=", 1) for a in args.affected))
            if not affected:
                print("[skip] near-duplicates: no affected_test_outfits.json given", flush=True)
                continue
            S.run_near_duplicates(args.results, specs, affected, out, split=split,
                                  key=args.affected_key, replicates=args.replicates, seed=args.seed)
        elif step == "lookbench":
            _need(args, "results")
            S.run_lookbench(args.results / "metrics.csv", args.lookbench_scores, out,
                            replicates=args.replicates, seed=args.seed)
        done[step] = str(out)
    print(json.dumps({"completed": done}), flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    from repbench.v2.sensitivities import pipeline

    pipeline().EQUIVALENCE.update(dict(item.split("=", 1) for item in args.equivalence))
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
