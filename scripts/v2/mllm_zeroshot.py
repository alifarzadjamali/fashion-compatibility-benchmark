"""Zero-shot FITB with Qwen3-VL-8B-Instruct (bf16, SDPA), HANDOFF_V2 §6.2.

Two steps:

  freeze-ids  Draw a deterministic seeded subset of question ids (default 2,000) from a
              FITB split, or all A100 questions, and write the frozen list (exclusive
              create). Reads question structure only; no model runs.
  score       Score the frozen questions by next-token letter log-probabilities.
              Append-only, fsync'd JSONL; resumable; refuses to run once summary.json
              exists. Scoring a test split or A100 requires --confirm-protocol-frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from repbench.utils.paths import configure_cache_environment

configure_cache_environment()

from repbench.v2.mllm_zeroshot import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    MCQuestion,
    append_result,
    config_record,
    read_results,
    sample_question_ids,
    summarize,
)
from repbench.v2.state import sha256, write_json
from repbench.v2.stores import image_resolver

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_HASH = ROOT / "configs" / "v2_protocol_hash.json"


def fitb_source_file(protocol, split):
    for name in (f"fill_in_blank_{split}.json", f"fill_in_the_blank_{split}.json"):
        if (protocol / name).is_file():
            return protocol / name
    raise FileNotFoundError(f"No FITB file for {split} in {protocol}")


def load_all_questions(args):
    """Return (source_file, list of MCQuestion) in canonical order."""
    if args.source == "fitb":
        from repbench.data.polyvore import PolyvoreDisjoint

        dataset = PolyvoreDisjoint(args.protocol_root.parent, args.protocol_root, args.dataset_name)
        resolve = image_resolver(args.images_root, args.protocol_root)
        questions = [
            MCQuestion(
                str(index),
                tuple(resolve(i) for i in q.question_item_ids),
                tuple(resolve(i) for i in q.candidate_item_ids),
                q.correct_index,
            )
            for index, q in enumerate(dataset.fitb(args.split))
        ]
        return fitb_source_file(args.protocol_root, args.split), questions
    source = args.a100_root / f"{args.split.lower()}.json"
    rows = json.loads(source.read_text())
    # Prepared by scripts/v2/prepare_a100_v2.py: image_paths.json (else {id}.jpg).
    resolve = image_resolver(args.images_root or args.a100_root / "images", args.a100_root)
    questions = []
    for row in rows:
        lat = args.split.upper() == "LAT"
        correct = row["majority_gt_index"] if lat else row["archive_gt_index"]
        extra = {"archive_gt_index": row["archive_gt_index"]}
        if lat:
            extra["gt_distribution"] = row["gt_distribution"]
        else:
            extra["dimension"] = row["dimension"]
        questions.append(
            MCQuestion(
                f"{args.split.upper()}_{row['question_num']}",
                tuple(resolve(i) for i in row["question"]),
                tuple(resolve(i) for i in row["answers"]),
                int(correct),
                extra,
            )
        )
    return source, questions


def freeze_ids(args):
    source, questions = load_all_questions(args)
    n = len(questions) if args.source == "a100" else args.n
    positions = sample_question_ids(len(questions), n, args.seed)
    record = {
        "kind": "frozen_mllm_question_ids",
        "source": args.source,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "seed": args.seed,
        "n_requested": n,
        "total_questions": len(questions),
        "sampling": "numpy default_rng(seed).choice(total, n, replace=False), sorted; "
        "all questions if n >= total",
        "source_file": str(source),
        "source_sha256": sha256(source),
        "question_ids": [questions[p].question_id for p in positions],
        "created_utc": datetime.now(UTC).isoformat(),
    }
    write_json(args.output, record)
    print(f"Froze {len(positions)} of {len(questions)} question ids -> {args.output}")


def score(args):
    import torch

    from repbench.utils.seeds import seed_everything

    frozen = json.loads(args.ids.read_text())
    args.source, args.split = frozen["source"], frozen["split"]
    args.dataset_name = frozen["dataset_name"]
    if args.source == "fitb" and args.images_root is None:
        raise SystemExit("--images-root is required to score FITB questions")
    is_test = args.source == "a100" or args.split == "test"
    if is_test and not args.confirm_protocol_frozen:
        raise SystemExit(
            "Test/A100 scoring is blocked before the protocol freeze; pass "
            "--confirm-protocol-frozen only after the owner has frozen configs/v2_protocol.yaml"
        )
    if is_test:
        from repbench.v2.a100 import require_frozen_protocol

        try:
            require_frozen_protocol(args.frozen_protocol_hash, root=ROOT)
        except PermissionError as error:
            raise SystemExit(str(error)) from error
    source, questions = load_all_questions(args)
    if sha256(source) != frozen["source_sha256"]:
        raise ValueError("Question source changed since the ids were frozen")
    by_id = {q.question_id: q for q in questions}
    selected = [by_id[i] for i in frozen["question_ids"]]
    if args.limit:
        selected = selected[: args.limit]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"{summary_path} exists; finished runs are immutable")
    from repbench.v2.registry import ZERO_SHOT

    config = config_record(
        model=ZERO_SHOT.checkpoint,
        revision=ZERO_SHOT.revision,
        dtype="bfloat16",
        attention="sdpa",
        ids_file=str(args.ids),
        ids_sha256=sha256(args.ids),
        source=args.source,
        dataset_name=args.dataset_name,
        split=args.split,
        option_seed=args.option_seed,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_visual_tokens_per_image=args.max_pixels // (32 * 32),
        max_input_tokens=args.max_input_tokens,
        limit=args.limit,
        scoring="argmax next-token log-prob over bare letter tokens at the final position",
        a100_correct="LAT majority_gt_index (primary); AAT archive_gt_index",
    )
    config_path = out / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Resume configuration differs from the recorded run; use a new dir")
    else:
        write_json(config_path, config)
    results_path = out / "results.jsonl"
    done = {r["question_id"] for r in read_results(results_path)}
    remaining = [q for q in selected if q.question_id not in done]
    print(f"{len(done)} done, {len(remaining)} remaining", flush=True)
    if remaining:
        from repbench.v2.mllm_zeroshot import ZeroShotScorer
        from repbench.v2.qwen import load_qwen
        from repbench.v2.state import require_budget

        if not torch.cuda.is_available():
            raise SystemExit("Scoring requires CUDA")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        seed_everything(args.option_seed)
        torch.backends.cudnn.benchmark = False
        if not args.skip_budget_check:
            require_budget(1800)
        started = time.perf_counter()
        model, processor = load_qwen(ZERO_SHOT, torch.bfloat16)
        model = model.eval().requires_grad_(False).to("cuda")
        print(f"Loaded in {time.perf_counter() - started:.1f}s", flush=True)
        scorer = ZeroShotScorer(
            model,
            processor,
            device="cuda",
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            max_input_tokens=args.max_input_tokens,
        )
        session = datetime.now(UTC).isoformat()
        for count, question in enumerate(remaining, 1):
            if (out / "STOP").exists():
                print("STOP requested; partial results retained", flush=True)
                return
            if count % 100 == 0 and not args.skip_budget_check:
                require_budget(600)
            record = scorer.score(question, args.option_seed)
            record["session_utc"] = session
            append_result(results_path, record)
            if count % 50 == 0:
                print(f"{len(done) + count}/{len(selected)}", flush=True)
    records = read_results(results_path)
    wanted = {q.question_id for q in selected}
    records = [r for r in records if r["question_id"] in wanted]
    if {r["question_id"] for r in records} != wanted or len(records) != len(wanted):
        raise ValueError("Results do not cover the frozen ids exactly once")
    summary = summarize(records, config)
    summary["results_sha256"] = sha256(results_path)
    summary["device"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    summary["finished_utc"] = datetime.now(UTC).isoformat()
    write_json(summary_path, summary)
    print(json.dumps({k: summary[k] for k in ("questions", "accuracy")}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-ids")
    freeze.add_argument("--source", choices=["fitb", "a100"], default="fitb")
    freeze.add_argument("--protocol-root", type=Path, help="Clean protocol directory (fitb)")
    freeze.add_argument("--dataset-name", default="")
    freeze.add_argument("--images-root", type=Path, default=None,
                        help="FITB image root; A100 default <a100-root>/images")
    freeze.add_argument("--a100-root", type=Path, help="Prepared A100 dir with lat.json/aat.json")
    freeze.add_argument("--split", required=True, help="FITB split, or LAT/AAT for A100")
    freeze.add_argument("--n", type=int, default=2000)
    freeze.add_argument("--seed", type=int, default=20261118)
    freeze.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("score")
    run.add_argument("--ids", type=Path, required=True)
    run.add_argument("--protocol-root", type=Path)
    run.add_argument("--a100-root", type=Path)
    run.add_argument("--images-root", type=Path, default=None,
                     help="required for fitb; A100 default <a100-root>/images")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--option-seed", type=int, default=20261118)
    run.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    run.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    run.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    run.add_argument("--limit", type=int, default=0, help="Score only the first N frozen ids")
    run.add_argument("--confirm-protocol-frozen", action="store_true")
    run.add_argument("--skip-budget-check", action="store_true")
    run.add_argument("--frozen-protocol-hash", type=Path, default=DEFAULT_PROTOCOL_HASH,
                     help="G7 hash file; required (and verified) for test/A100 scoring")
    args = parser.parse_args(argv)
    if args.command == "freeze-ids":
        if args.source == "fitb" and not args.protocol_root:
            parser.error("--protocol-root is required for fitb")
        if args.source == "a100" and not args.a100_root:
            parser.error("--a100-root is required for a100")
        if args.source == "fitb" and args.images_root is None:
            args.images_root = Path(".")
        freeze_ids(args)
    else:
        score(args)


if __name__ == "__main__":
    sys.exit(main())
