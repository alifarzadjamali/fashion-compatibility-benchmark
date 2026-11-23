"""Checksum-verified acquisition with resumable curl and compact transfer evidence.

Downloads archives only: no extraction, labels, model predictions or test evaluation.
Public provider links can expire; HTTP success alone never establishes a valid source.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from repbench.utils.paths import data_path
from repbench.v2.state import require_budget, sha256, write_json


def run_transfer(command, partial, budget):
    started = time.monotonic()
    with subprocess.Popen(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        while True:
            try:
                stdout, stderr = process.communicate(timeout=30)
                return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                size = partial.stat().st_size if partial.exists() else 0
                print(
                    f"Transfer progress: {size / 2**20:.1f} MiB on disk; {elapsed:.0f}s elapsed",
                    flush=True,
                )
                if elapsed >= budget:
                    process.kill()
                    stdout, stderr = process.communicate()
                    return subprocess.CompletedProcess(command, 124, stdout, stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["iqon3000", "a100", "polyvore", "pog"], required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=1200)
    parser.add_argument(
        "--establish-checksum",
        action="store_true",
        help="Polyvore only: record first checksum, pending content audits",
    )
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    source = json.loads(Path("configs/v2_sources.json").read_text())[args.source]
    print(f"Acquiring {args.source}; budget={args.budget_seconds}s", flush=True)
    establish = (
        args.establish_checksum and args.source == "polyvore" and source.get("provider_version")
    )
    if not source["url"] or (not source["expected_sha256"] and not establish):
        raise RuntimeError("Source identity/checksum unresolved; refusing unverified substitute")
    destination = data_path(source["relative_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    record = {
        "source": args.source,
        "source_page": source["source_page"],
        "expected_sha256": source["expected_sha256"],
        "provider_version": source.get("provider_version"),
        "checksum_established_now": bool(establish),
        "utc": datetime.now(UTC).isoformat(),
    }
    if destination.exists():
        record["sha256"] = sha256(destination)
        record["reused"] = True
    else:
        require_budget(args.budget_seconds)
        partial = destination.with_suffix(destination.suffix + ".part")
        record["existing_partial_bytes"] = partial.stat().st_size if partial.exists() else 0
        transfer_start = time.perf_counter()
        # curl JSON includes effective signed URLs; only allowlisted numeric fields
        # below are preserved. stderr does not print request/auth headers.
        result = run_transfer(
            [
                "curl",
                "--location",
                "--fail",
                "--silent",
                "--show-error",
                "--user-agent",
                "Mozilla/5.0",
                "--retry",
                "2",
                "--connect-timeout",
                "30",
                "--max-time",
                str(args.budget_seconds),
                "--continue-at",
                "-",
                "--output",
                str(partial),
                "--write-out",
                "%{json}",
                source["url"],
            ],
            partial,
            args.budget_seconds,
        )
        record["transfer_wall_seconds"] = time.perf_counter() - transfer_start
        record["curl_returncode"] = result.returncode
        try:
            transfer = json.loads(result.stdout)
            record["transfer"] = {
                k: transfer.get(k)
                for k in [
                    "http_code",
                    "size_download",
                    "speed_download",
                    "time_total",
                    "num_redirects",
                ]
            }
        except json.JSONDecodeError:
            record["transfer"] = None
        if result.returncode:
            record["status"] = "FAIL"
            record["reason"] = "transfer_failed; partial retained; inspect provider access"
            write_json(args.report, record)
            print(json.dumps(record, indent=2), flush=True)
            raise SystemExit(1)
        hash_start = time.perf_counter()
        record["sha256"] = sha256(partial)
        record["checksum_seconds"] = time.perf_counter() - hash_start
        record["bytes"] = partial.stat().st_size
        if (record["sha256"] == source["expected_sha256"] or establish) and zipfile.is_zipfile(
            partial
        ):
            # Same-filesystem publication only after complete validation.
            partial.rename(destination)
    record["status"] = (
        "PASS" if destination.exists() and record["sha256"] == source["expected_sha256"] else "FAIL"
    )
    if establish and destination.exists() and zipfile.is_zipfile(destination):
        record["status"] = "ACQUIRED_PENDING_CONTENT_AUDIT"
    record["total_seconds"] = time.perf_counter() - started
    if destination.exists():
        record["bytes"] = destination.stat().st_size
    write_json(args.report, record)
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if record["status"] in {"PASS", "ACQUIRED_PENDING_CONTENT_AUDIT"} else 1)


if __name__ == "__main__":
    main()
