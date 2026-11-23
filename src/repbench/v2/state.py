"""Fail-closed booking checks and immutable, hashed readiness records."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: a completed record is never overwritten.
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def remaining_seconds():
    result = subprocess.run(["timeleft"], text=True, capture_output=True, timeout=10, check=False)
    return parse_timeleft(result.stdout, result.returncode)


def parse_timeleft(text, returncode=0):
    """Parse `timeleft` output, including the Slurm day form `D-HH:MM:SS`."""
    match = re.search(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})\s+left", text)
    if returncode or not match:
        raise RuntimeError("Cannot establish booking deadline from timeleft; refuse GPU job")
    days, hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def require_budget(job_seconds, reserve_seconds=1800):
    remaining = remaining_seconds()
    if remaining < job_seconds + reserve_seconds:
        raise RuntimeError(
            f"Insufficient booking time: {remaining}s left; "
            f"need {job_seconds}s + {reserve_seconds}s upload reserve"
        )
    return remaining


def require_gate(path, expected):
    record = json.loads(Path(path).read_text())
    if record.get("gate") != expected or record.get("status") != "PASS":
        raise RuntimeError(f"{expected} has not passed")
    if not record.get("input_sha256"):
        raise RuntimeError(f"{expected} has no code/lock binding; rerun on current inputs")
    for name, digest in record["input_sha256"].items():
        if not Path(name).is_file() or sha256(name) != digest:
            raise RuntimeError(f"{expected} evidence is stale: {name}")
    return record


def input_hashes():
    files = [Path("pyproject.toml"), Path("uv.lock")]
    for directory in ("src", "scripts", "tests"):
        files.extend(Path(directory).rglob("*.py"))
    return {p.as_posix(): sha256(p) for p in sorted(files)}
