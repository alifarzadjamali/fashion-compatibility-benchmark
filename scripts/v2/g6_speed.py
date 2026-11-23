"""G6 speed test: real-image extraction throughput on the DGX (HANDOFF §8).

`measure` times one encoder on a fixed 5,000-image sample of Polyvore *train*
images (development data only; no labels or test outcomes are touched), using
the v2 settings (uniform bf16 + SDPA). CPU preprocessing runs in a thread pool
that overlaps with GPU forward passes, as in a real extraction job. It records:
end-to-end wall time, GPU-only forward time (CUDA-synchronized), CPU preprocessing
time, load time and allocator peak memory.

`summarize` combines per-model records into a projected full-run extraction time.
This is a planning measurement, not the §6.6 efficiency profiler.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REQUIRED = ["resnet50", "clip_vitl14_336", "siglip2_so400m_384", "qwen3_vl_emb_2b"]
DATASET_IMAGES = {  # images to embed per dataset in the full run
    "polyvore_d_clean": 152_785,
    "iqon3000_clean": 434_089,
    "pog_target": 184_339,
}


def sample_images(root, train_json, count, seed):
    names = sorted(p.name for p in (root / "images").iterdir() if p.suffix == ".jpg")
    train = json.loads(train_json.read_text())
    train_items = {item["item_id"] for outfit in train for item in outfit["items"]}
    pool = [n for n in names if n.removesuffix(".jpg") in train_items]
    return [root / "images" / n for n in sorted(random.Random(seed).sample(pool, count))]


def measure(args):
    import numpy as np
    import torch

    from repbench.utils.seeds import seed_everything
    from repbench.v2.encoders import V2Encoder
    from repbench.v2.registry import V2_ENCODERS
    from repbench.v2.state import require_budget, require_gate, write_json

    require_gate(args.g2, "G2")
    require_budget(1200)
    if args.output.exists():
        raise FileExistsError(args.output)
    others = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout.split()
    others = [p for p in others if p.strip().isdigit() and int(p) != os.getpid()]
    if others and not args.allow_concurrent_gpu:
        raise RuntimeError(f"Other GPU processes running {others}; timings would be biased")
    seed_everything(20260912)
    torch.backends.cudnn.benchmark = False
    spec = V2_ENCODERS[args.model]
    paths = sample_images(args.polyvore_root, args.train_json, args.images, args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    encoder = V2Encoder(spec)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    batches = [paths[i : i + args.batch_size] for i in range(0, len(paths), args.batch_size)]

    def prepare(batch):
        t0 = time.perf_counter()
        inputs = encoder.prepare(batch)
        return inputs, time.perf_counter() - t0

    # Warm-up on one batch (excluded from timing; triggers kernel selection).
    with torch.inference_mode():
        encoder.forward(encoder.to_device(prepare(batches[0])[0]))
    torch.cuda.synchronize()

    gpu_seconds = cpu_seconds = 0.0
    finite = True
    dim = None
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool, torch.inference_mode():
        for inputs, prep in pool.map(prepare, batches):
            cpu_seconds += prep
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            output = encoder.forward(encoder.to_device(inputs))
            end.record()
            end.synchronize()
            gpu_seconds += start.elapsed_time(end) / 1000
            values = output.float().cpu().numpy()
            finite &= bool(np.isfinite(values).all())
            dim = values.shape[-1]
    wall_seconds = time.perf_counter() - wall_start
    if not finite:
        raise ValueError("Non-finite embeddings in speed test")
    record = {
        "gate_part": "G6 speed test (planning measurement)",
        "utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "checkpoint": spec.checkpoint,
        "revision": spec.revision,
        "images": len(paths),
        "image_source": "Polyvore-D-Clean train images, seeded sample (development data)",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "cpu_prep_threads": args.workers,
        "dtype": "bfloat16",
        "embedding_dim": dim,
        "load_seconds": round(load_seconds, 2),
        "end_to_end_seconds": round(wall_seconds, 2),
        "end_to_end_images_per_second": round(len(paths) / wall_seconds, 2),
        "gpu_forward_seconds": round(gpu_seconds, 2),
        "gpu_images_per_second": round(len(paths) / gpu_seconds, 2),
        "cpu_prep_seconds_summed_over_threads": round(cpu_seconds, 2),
        "bottleneck": "gpu" if gpu_seconds > 0.8 * wall_seconds else "cpu/io",
        "allocator_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "concurrent_load_note": args.note,
    }
    write_json(args.output, record)
    print(json.dumps({k: record[k] for k in ("model", "end_to_end_images_per_second",
                                              "gpu_images_per_second", "bottleneck")}))


def summarize(args):
    from repbench.v2.state import write_json

    records = {}
    for path in sorted(args.input.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("gate_part", "").startswith("G6"):
            records[record["model"]] = record
    missing = [m for m in REQUIRED if m not in records]
    total_images = sum(DATASET_IMAGES.values())
    per_model = {
        model: round(total_images / r["end_to_end_images_per_second"] / 3600, 2)
        for model, r in records.items()
    }
    summary = {
        "gate": "G6",
        "status": "PASS" if not missing else "INCOMPLETE",
        "missing_required": missing,
        "utc": datetime.now(UTC).isoformat(),
        "dataset_images": DATASET_IMAGES,
        "projected_extraction_hours_per_model": per_model,
        "projected_extraction_hours_measured_models": round(sum(per_model.values()), 1),
        "models_measured": sorted(records),
        "caveats": [
            "A100 images excluded (small); MLLM zero-shot cost measured separately.",
            "End-to-end rates include decode/preprocess; full-run I/O may differ.",
            "Projection covers extraction only; probes, baselines and stats are extra.",
        ],
    }
    write_json(args.output, summary)
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--g2", type=Path, required=True)
    m.add_argument("--model", required=True)
    m.add_argument("--polyvore-root", type=Path, required=True)
    m.add_argument("--train-json", type=Path, required=True, help="clean rebuild train.json")
    m.add_argument("--output", type=Path, required=True)
    m.add_argument("--images", type=int, default=5000)
    m.add_argument("--seed", type=int, default=20261118)
    m.add_argument("--batch-size", type=int, default=32)
    m.add_argument("--workers", type=int, default=8)
    m.add_argument("--allow-concurrent-gpu", action="store_true")
    m.add_argument("--note", default="")
    s = sub.add_parser("summarize")
    s.add_argument("--input", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measure(args) if args.command == "measure" else summarize(args)


if __name__ == "__main__":
    main()
