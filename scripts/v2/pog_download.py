"""Download the POG images needed by the frozen-seed target outfits (HANDOFF_V2 §4).

Target selection is identical to `pog_probe.py` (same seed, same 124k outfits).
Resumable: every verified image is appended to `manifest.jsonl` (item id, path,
bytes, kind, SHA-256); a restart skips items already recorded. Files are written
atomically. Images are research-only and never redistributed; only the summary
report goes to Git. Final success on required items must be >= 95% (skip rule).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # works under python -I
from pog_probe import USER_AGENT, RateLimiter, image_kind


def load_target(items_path, outfits_path, seed, target_outfits):
    outfits = []
    with outfits_path.open(encoding="utf-8") as stream:
        for line in stream:
            outfit_id, _, items = line.rstrip("\n").partition(",")
            if items:
                outfits.append((outfit_id, items.split(";")))
    outfits.sort()
    rng = random.Random(seed)
    target = rng.sample(outfits, min(target_outfits, len(outfits)))
    needed = {item for _, items in target for item in items}
    urls = {}
    with items_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = line.rstrip("\n").split(",", 3)
            if len(parts) >= 3 and parts[0] in needed:
                url = parts[2]
                urls[parts[0]] = "http:" + url if url.startswith("//") else url
    return sorted(o for o, _ in target), urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--outfits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="image mirror root")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--target-outfits", type=int, default=124000)
    parser.add_argument("--rate", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--min-success", type=float, default=0.95)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    started = time.monotonic()
    outfit_ids, urls = load_target(args.items, args.outfits, args.seed, args.target_outfits)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "target_outfits.txt").write_text("\n".join(outfit_ids) + "\n")
    manifest_path = args.output / "manifest.jsonl"
    done = set()
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["item_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # a torn final line from an interrupted run
    lock = threading.Lock()
    limiter = RateLimiter(args.rate)
    failures = {}
    progress = {"n": 0}
    pending = sorted(set(urls) - done)
    print(f"target items {len(urls)}, already verified {len(done)}, pending {len(pending)}", flush=True)

    def fetch(item_id):
        url = urls[item_id]
        error = None
        for attempt in range(args.retries):
            if attempt:
                time.sleep(2**attempt)
            limiter.wait()
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    payload = response.read()
                kind = image_kind(payload)
                if not kind:
                    error = "not_image"
                    continue
                relative = Path("images") / item_id[:2] / f"{item_id}.{kind}"
                path = args.output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_bytes(payload)
                temporary.replace(path)
                record = {
                    "item_id": item_id,
                    "path": str(relative),
                    "bytes": len(payload),
                    "kind": kind,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                with lock:
                    with manifest_path.open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    done.add(item_id)
                break
            except urllib.error.HTTPError as exc:
                error = f"http_{exc.code}"
                if exc.code in (400, 403, 404, 410):
                    break
            except Exception as exc:
                error = type(exc).__name__
        with lock:
            if item_id not in done:
                failures[item_id] = error
            progress["n"] += 1
            if progress["n"] % 5000 == 0:
                elapsed = time.monotonic() - started
                print(f"{progress['n']}/{len(pending)} processed, {len(failures)} failed, "
                      f"{elapsed / 3600:.2f} h", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(fetch, pending))

    records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    success = len(done) / len(urls)
    reasons = {}
    for reason in failures.values():
        reasons[reason] = reasons.get(reason, 0) + 1
    report = {
        "stage": "G4 POG required-image acquisition",
        "utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if success >= args.min_success else "FAIL_SKIP_POG",
        "rule_min_success": args.min_success,
        "seed": args.seed,
        "target_outfits": len(outfit_ids),
        "target_outfits_sha256": hashlib.sha256(
            (args.output / "target_outfits.txt").read_bytes()
        ).hexdigest(),
        "target_items": len(urls),
        "verified_items": len(done),
        "success_rate": round(success, 5),
        "failure_reasons": reasons,
        "failed_item_ids": sorted(failures),
        "total_bytes": sum(r["bytes"] for r in records),
        "kinds": {k: sum(r["kind"] == k for r in records) for k in {r["kind"] for r in records}},
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "rate_cap_per_second": args.rate,
        "user_agent": USER_AGENT,
        "redistribution": "none; local private mirror only",
        "elapsed_hours": round((time.monotonic() - started) / 3600, 3),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=1)
    print(f"{report['status']}: {len(done)}/{len(urls)} verified ({success:.4f})", flush=True)


if __name__ == "__main__":
    main()
