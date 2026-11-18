"""Verify compact tracked evidence against bytes fetched back from GitHub.

Does not stage files or upload secrets. Call only after reviewing and pushing the
intended evidence. A fresh clone avoids mistaking local objects for remote proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def git(*args, cwd=None):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=180
    ).stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default="v2-readiness")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    commit = git("rev-parse", "HEAD")
    paths = git("ls-files", "research/v2").splitlines()
    if not paths:
        raise RuntimeError("No tracked evidence to verify")
    if git("status", "--porcelain", "--", "research/v2"):
        raise RuntimeError("Commit all intended evidence before verifying")
    checks = {}
    with tempfile.TemporaryDirectory(prefix="repbench-restore-") as folder:
        git(
            "clone",
            "--depth",
            "1",
            "--branch",
            args.branch,
            "https://github.com/alifarzadjamali/fashion-compatibility-benchmark.git",
            folder,
        )
        fetched_commit = git("rev-parse", "HEAD", cwd=folder)
        if fetched_commit != commit:
            raise RuntimeError("Remote branch does not match the intended local commit")
        for name in paths:
            path = Path(name)
            if path.stat().st_size > 10 * 2**20:
                raise RuntimeError(f"Evidence exceeds conservative ordinary-Git limit: {name}")
            local_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            restored_hash = hashlib.sha256((Path(folder) / name).read_bytes()).hexdigest()
            if local_hash != restored_hash:
                raise RuntimeError(f"Restored bytes differ: {name}")
            checks[name] = local_hash
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(
            {
                "verified_utc": datetime.now(UTC).isoformat(),
                "commit": commit,
                "branch": args.branch,
                "method": "fresh HTTPS shallow clone and SHA-256 comparison",
                "status": "PASS",
                "files": checks,
            },
            stream,
            indent=2,
        )
    print(f"Verified {len(checks)} files restored from GitHub commit {commit}")


if __name__ == "__main__":
    main()
