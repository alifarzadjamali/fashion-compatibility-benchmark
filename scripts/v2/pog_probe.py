"""POG go/skip availability probe (HANDOFF_V2 §4). Phase 1 only; no model is run.

Samples 1,000 items, with a frozen seed, from items used by POG outfits, then fetches
each pic_url once, rate-limited, with a generic user-agent. Images are validated
by magic bytes and hashed in memory; no image is stored or redistributed.

Go rule: success >= 97% AND projected download of an IQON-sized target
(~124k outfits) fits in 48 h at the probed rate. The report is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

USER_AGENT = "Mozilla/5.0 (research availability probe)"
MAGIC = {b"\xff\xd8\xff": "jpeg", b"\x89PNG": "png", b"GIF8": "gif", b"RIFF": "webp"}


def image_kind(payload):
    for prefix, kind in MAGIC.items():
        if payload.startswith(prefix):
            return kind
    return None


class RateLimiter:
    def __init__(self, per_second):
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_slot = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_slot)
            self.next_slot = slot + self.interval
        time.sleep(max(0.0, slot - now))


def fetch(item_id, url, limiter, timeout):
    limiter.wait()
    started = time.monotonic()
    record = {"item_id": item_id, "url": url}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            record["http"] = response.status
        kind = image_kind(payload)
        record.update(
            bytes=len(payload),
            kind=kind,
            sha256=hashlib.sha256(payload).hexdigest(),
            ok=bool(kind) and len(payload) > 0,
        )
    except urllib.error.HTTPError as error:
        record.update(http=error.code, ok=False, error="HTTPError")
    except Exception as error:  # network failures are outcomes, not crashes
        record.update(ok=False, error=type(error).__name__)
    record["seconds"] = round(time.monotonic() - started, 4)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True, help="item_data.txt")
    parser.add_argument("--outfits", type=Path, required=True, help="outfit_data.txt")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--target-outfits", type=int, default=124000)
    parser.add_argument("--rate", type=float, default=8.0, help="requests per second cap")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--go-success", type=float, default=0.97)
    parser.add_argument("--go-hours", type=float, default=48.0)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    t0 = time.monotonic()

    outfits = []
    with args.outfits.open(encoding="utf-8") as stream:
        for line in stream:
            outfit_id, _, items = line.rstrip("\n").partition(",")
            if items:
                outfits.append((outfit_id, items.split(";")))
    outfits.sort()
    rng = random.Random(args.seed)
    target = rng.sample(outfits, min(args.target_outfits, len(outfits)))
    target_items = sorted({item for _, items in target for item in items})
    used_items = {item for _, items in outfits for item in items}

    urls, categories = {}, {}
    with args.items.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = line.rstrip("\n").split(",", 3)
            if len(parts) >= 3 and parts[0] in used_items:
                # ~19% of pic_urls are scheme-relative ("//img.alicdn.com/..."); the
                # rest use http:, so resolve them the same way a browser would.
                url = parts[2]
                urls[parts[0]] = "http:" + url if url.startswith("//") else url
                categories[parts[1]] = categories.get(parts[1], 0) + 1
    missing_url = sorted(used_items - urls.keys())
    probe_ids = sorted(rng.sample(sorted(urls), args.sample))

    limiter = RateLimiter(args.rate)
    probe_start = time.monotonic()
    with ThreadPoolExecutor(args.workers) as pool:
        records = list(pool.map(lambda i: fetch(i, urls[i], limiter, args.timeout), probe_ids))
    probe_seconds = time.monotonic() - probe_start

    ok = [r for r in records if r["ok"]]
    success = len(ok) / len(records)
    throughput = len(records) / probe_seconds
    projected_hours = len(target_items) / throughput / 3600
    mean_bytes = statistics.mean(r["bytes"] for r in ok) if ok else 0
    failures = {}
    for r in records:
        if not r["ok"]:
            key = f"{r.get('http')}/{r.get('error') or r.get('kind')}"
            failures[key] = failures.get(key, 0) + 1
    decision = "GO" if success >= args.go_success and projected_hours <= args.go_hours else "SKIP"

    report = {
        "stage": "G4 POG go/skip availability probe",
        "utc": datetime.now(UTC).isoformat(),
        "decision": decision,
        "rule": {"min_success": args.go_success, "max_projected_hours": args.go_hours},
        "seed": args.seed,
        "sample": len(records),
        "success_rate": round(success, 4),
        "failures": failures,
        "probe_seconds": round(probe_seconds, 1),
        "rate_cap_per_second": args.rate,
        "workers": args.workers,
        "observed_requests_per_second": round(throughput, 3),
        "median_fetch_seconds": statistics.median(r["seconds"] for r in records),
        "mean_image_bytes": round(mean_bytes),
        "outfits_total": len(outfits),
        "items_used_by_outfits": len(used_items),
        "items_used_without_url": len(missing_url),
        "categories_used": len(categories),
        "target_outfits": len(target),
        "target_unique_items": len(target_items),
        "projected_download_hours_at_probe_rate": round(projected_hours, 2),
        "projected_storage_gb": round(len(target_items) * mean_bytes / 1e9, 2),
        "input_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.items, args.outfits)
        },
        "user_agent": USER_AGENT,
        "images_stored": False,
        "records": records,
        "total_seconds": round(time.monotonic() - t0, 1),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=1, ensure_ascii=False)
    print(
        f"{decision}: success {success:.4f}, {throughput:.2f} req/s, "
        f"{len(target_items)} target items -> {projected_hours:.1f} h"
    )


if __name__ == "__main__":
    main()
