"""G2 synthetic smoke checks. No dataset labels or test outcomes are read."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from repbench.utils.paths import configure_cache_environment
from repbench.v2.registry import V2_ENCODERS, ZERO_SHOT
from repbench.v2.state import input_hashes, require_budget, require_gate, sha256, write_json

configure_cache_environment()
ALL_MODELS = {**V2_ENCODERS, ZERO_SHOT.model_key: ZERO_SHOT}


def signature(spec):
    # Include implementation and lock: passing results cannot survive code changes silently.
    files = sorted(Path("src/repbench/v2").glob("*.py")) + [
        Path("scripts/v2/g2_models.py"),
        Path("uv.lock"),
    ]
    payload = {"spec": asdict(spec), "files": {str(p): sha256(p) for p in files}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_bytes():
    from huggingface_hub.constants import HF_HUB_CACHE

    root = Path(HF_HUB_CACHE)
    return sum(p.stat().st_size for p in root.glob("models--*/blobs/*") if p.is_file())


def check_model(key, output):
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image

    from repbench.utils.seeds import seed_everything
    from repbench.v2.encoders import V2Encoder

    spec = ALL_MODELS[key]
    record = {
        "model_key": key,
        "spec": asdict(spec),
        "signature": signature(spec),
        "utc": datetime.now(UTC).isoformat(),
        "input": "64 synthetic RGB images; seed 20260912",
        "status": "FAIL",
    }
    seed_everything(20260912)
    started = time.perf_counter()
    try:
        before = cache_bytes()
        download_started = time.perf_counter()
        if spec.backend != "torchvision":
            if spec.backend == "open_clip_hub":
                patterns = ["open_clip_config.json", "open_clip_model.safetensors"]
            else:
                patterns = ["*.json", "*.jinja", "*.txt", "*.model", "*.py"]
                patterns += (
                    ["pytorch_model.bin"] if spec.use_safetensors is False else ["*.safetensors"]
                )
            snapshot_download(
                spec.checkpoint, revision=spec.revision, allow_patterns=patterns, max_workers=4
            )
        record["download_seconds"] = time.perf_counter() - download_started
        record["new_hf_cache_bytes"] = max(0, cache_bytes() - before)
        record["hf_effective_mib_per_second"] = (
            record["new_hf_cache_bytes"] / 2**20 / record["download_seconds"]
        )
        record["download_measurement"] = (
            "new HF blob bytes / snapshot wall time; cache hits included; not wire bytes"
        )
        load_started = time.perf_counter()
        encoder = V2Encoder(spec)
        torch.cuda.synchronize()
        record["load_seconds"] = time.perf_counter() - load_started
        if spec.backend == "torchvision":
            record["download_measurement"] = "torchvision weight transfer included in load_seconds"
        record["attention_method"] = encoder.attention_method
        record["image_tower_parameters"] = encoder.image_parameter_count()
        record["total_parameters"] = sum(p.numel() for p in encoder.model.parameters())
        rng = np.random.default_rng(20260912)
        images = [
            Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)) for _ in range(64)
        ]
        prep_started = time.perf_counter()
        inputs = encoder.to_device(encoder.prepare(images))
        torch.cuda.synchronize()
        record["preprocess_and_transfer_seconds"] = time.perf_counter() - prep_started
        torch.cuda.reset_peak_memory_stats()
        inference_started = time.perf_counter()
        with torch.inference_mode():
            first = encoder.forward(inputs)
            second = encoder.forward(inputs)
        torch.cuda.synchronize()
        record["two_smoke_passes_seconds"] = time.perf_counter() - inference_started
        record["peak_allocator_bytes"] = torch.cuda.max_memory_allocated()
        record["shape"] = list(first.shape)
        finite = bool(torch.isfinite(first).all() and torch.isfinite(second).all())
        if spec.backend == "qwen_generation":
            drift = float((first - second).abs().max())
            dimensions = first.ndim == 2 and first.shape[0] == 64
            record["smoke_kind"] = "next-token log probabilities, not embeddings"
            record["max_logprob_drift"] = drift
        else:
            drift = float((1 - torch.nn.functional.cosine_similarity(first, second)).abs().max())
            dimensions = tuple(first.shape) == (64, spec.embedding_dim)
            record["max_cosine_drift"] = drift
        record["checks"] = {
            "finite": finite,
            "expected_shape": dimensions,
            "deterministic_within_1e_3": drift <= 1e-3,
            "bf16": next(encoder.model.parameters()).dtype == torch.bfloat16,
        }
        record["status"] = "PASS" if all(record["checks"].values()) else "FAIL"
    except Exception as exc:  # noqa: BLE001 - preserve failure evidence for each independent model
        # Full trace remains in the local stage log; never include credential-bearing URLs here.
        record["error_type"] = type(exc).__name__
        print(f"{key}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    record["total_seconds"] = time.perf_counter() - started
    write_json(output, record)
    print(json.dumps(record, indent=2), flush=True)
    return record["status"] == "PASS"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--g1", type=Path, required=True)
    parser.add_argument("--model", choices=ALL_MODELS)
    args = parser.parse_args()
    require_gate(args.g1, "G1")
    if args.model:
        require_budget(900)
        sys.exit(0 if check_model(args.model, args.output) else 1)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "gate.json").exists():
        raise FileExistsError("Use a fresh attempt directory; completed G2 record is immutable")
    records = []
    for key, spec in ALL_MODELS.items():
        if (args.output / "STOP").exists():
            print("STOP requested; no new model started", flush=True)
            break
        try:
            require_budget(900)
        except RuntimeError as exc:
            print(str(exc), flush=True)
            break
        output = args.output / (key + ".json")
        if output.exists():
            record = json.loads(output.read_text())
            if record.get("signature") != signature(spec):
                raise RuntimeError(f"Stale evidence for {key}; use a fresh attempt directory")
            records.append(record)
            continue
        with (args.output / (key + ".log")).open("x") as log:
            try:
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--g1",
                        str(args.g1),
                        "--output",
                        str(output),
                        "--model",
                        key,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=900,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                print(f"{key}: exceeded 15 minute stage budget", flush=True)
        if output.exists():
            records.append(json.loads(output.read_text()))
        print(f"Completed attempt: {key}", flush=True)
    passed = len(records) == len(ALL_MODELS) and all(r["status"] == "PASS" for r in records)
    write_json(
        args.output / "gate.json",
        {
            "gate": "G2",
            "status": "PASS" if passed else "FAIL",
            "expected_models": list(ALL_MODELS),
            "input_sha256": input_hashes(),
            "completed": len(records),
            "passing": [r["model_key"] for r in records if r["status"] == "PASS"],
            "note": "G3 and later gates must not run until G2 passes.",
        },
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
