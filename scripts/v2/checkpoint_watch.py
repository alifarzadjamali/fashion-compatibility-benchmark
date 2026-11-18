"""Booking-local gate checkpoint uploader; no agent or laptop is needed once started.

Requires a repository-scoped SSH deploy key with write access. Only allowlisted
code and compact reports are copied; data, weights and embedding caches stay local.
Each new gate status produces at most one commit, with a repository-wide daily cap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPOSITORY = "alifarzadjamali/fashion-compatibility-benchmark"
SECRET = re.compile(
    rb"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)
MAX_FILE_BYTES = 10 * 2**20


def command(args, *, cwd=None, env=None):
    return subprocess.run(
        args, cwd=cwd, env=env, check=True, capture_output=True, text=True, timeout=180
    ).stdout.strip()


def compact_files(source, data_root, booking):
    """Yield reviewed-format files, never arbitrary cache/dataset directory trees."""
    for folder in ("src", "scripts", "tests", "configs", "docs"):
        for path in (source / folder).rglob("*"):
            if path.is_file() and path.suffix in {
                ".py",
                ".sh",
                ".json",
                ".yaml",
                ".toml",
                ".lock",
                ".md",
                ".txt",
            }:
                yield path, path.relative_to(source)
    for name in ("README.md", "AGENTS.md", "pyproject.toml", "uv.lock"):
        path = source / name
        if path.is_file():
            yield path, Path(name)
    for path in (source / "research/v2").rglob("*"):
        if path.is_file() and path.suffix in {
            ".json",
            ".txt",
            ".md",
            ".csv",
            ".png",
            ".pdf",
            ".svg",
        }:
            yield path, path.relative_to(source)
    target = Path("research/v2/readiness") / booking / "g3"
    for dataset in ("polyvore", "iqon"):
        path = data_root / f"v2/g3_{dataset}/reconstruction_check.json"
        if path.is_file():
            yield path, target / f"{dataset}_reconstruction.json"
        for model in ("resnet50", "clip_vitl14_336"):
            path = data_root / f"v2/g3_metrics/{dataset}/{model}/reproduction_check.json"
            if path.is_file():
                yield path, target / f"{dataset}_{model}.json"
    for name in ("source_audit.json", "image_audit.json"):
        path = data_root / "v2/iqon_preparation" / name
        if path.is_file():
            yield path, target / ("iqon_" + name)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def gate_status(source, data_root):
    gates = {f"G{i}": "NOT_RUN" for i in range(1, 8)}
    evidence = {}
    # Prefer the latest report per gate, including explicit failed attempts.
    for path in sorted(
        (source / "research/v2/readiness").rglob("gate.json"), key=lambda p: p.stat().st_mtime_ns
    ):
        record = read_json(path)
        if record and record.get("gate") in gates:
            gates[record["gate"]] = record["status"]
            evidence[record["gate"]] = str(path.relative_to(source))
    required = [
        data_root / f"v2/g3_{dataset}/reconstruction_check.json" for dataset in ("polyvore", "iqon")
    ]
    required += [
        data_root / f"v2/g3_metrics/{dataset}/{model}/reproduction_check.json"
        for dataset in ("polyvore", "iqon")
        for model in ("resnet50", "clip_vitl14_336")
    ]
    reports = [read_json(path) for path in required]
    if any(record and record.get("status") == "FAIL" for record in reports):
        gates["G3"] = "FAIL"
    elif all(record and record.get("status") == "PASS" for record in reports):
        gates["G3"] = "PASS_REVISED_OWNER_APPROVED"
    elif any(record for record in reports):
        gates["G3"] = "IN_PROGRESS"
    return gates, evidence


def publish_and_verify(checkout, branch, environment, hashes):
    command(["git", "push", "origin", f"HEAD:refs/heads/{branch}"], cwd=checkout, env=environment)
    commit = command(["git", "rev-parse", "HEAD"], cwd=checkout)
    with tempfile.TemporaryDirectory(prefix="repbench-upload-verify-") as folder:
        command(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                branch,
                f"https://github.com/{REPOSITORY}.git",
                folder,
            ]
        )
        if command(["git", "rev-parse", "HEAD"], cwd=folder) != commit:
            raise RuntimeError("Readback commit differs")
        for name, digest in hashes.items():
            if hashlib.sha256((Path(folder) / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Readback hash differs: {name}")
    return commit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--upload-root", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--booking", required=True)
    parser.add_argument("--branch", default="v2-readiness-checkpoints")
    parser.add_argument("--base-branch", default="v2-readiness")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.upload_root.mkdir(parents=True, exist_ok=True)
    state_path = args.upload_root / "uploader_state.json"
    state = read_json(state_path) or {}
    known_hosts = args.upload_root / "github_known_hosts"
    if not known_hosts.exists():
        with urllib.request.urlopen("https://api.github.com/meta", timeout=20) as response:
            keys = json.load(response)["ssh_keys"]
        if not keys:
            raise RuntimeError("GitHub's HTTPS metadata returned no SSH host keys")
        known_hosts.write_text("".join(f"[ssh.github.com]:443 {key}\n" for key in keys))
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["GIT_SSH_COMMAND"] = (
        f"ssh -i {shlex.quote(str(args.key))} -o IdentitiesOnly=yes -o BatchMode=yes "
        f"-o StrictHostKeyChecking=yes -o UserKnownHostsFile={shlex.quote(str(known_hosts))}"
    )
    checkout = args.upload_root / "checkout"
    public = f"https://github.com/{REPOSITORY}.git"
    if not checkout.exists():
        existing = command(["git", "ls-remote", public, f"refs/heads/{args.branch}"])
        command(
            [
                "git",
                "clone",
                "--branch",
                args.branch if existing else args.base_branch,
                public,
                str(checkout),
            ]
        )
        if not existing:
            command(["git", "switch", "-c", args.branch], cwd=checkout)
        command(
            [
                "git",
                "remote",
                "set-url",
                "--push",
                "origin",
                f"ssh://git@ssh.github.com:443/{REPOSITORY}.git",
            ],
            cwd=checkout,
        )
        command(["git", "config", "user.name", "RepBench checkpoint"], cwd=checkout)
        command(
            ["git", "config", "user.email", "repbench-checkpoint@users.noreply.github.com"],
            cwd=checkout,
        )
    while True:
        if (args.upload_root / "STOP").exists():
            return
        try:
            if state.get("pending_commit"):
                snapshot = read_json(checkout / "research/v2/checkpoint_state.json")
                commit = publish_and_verify(
                    checkout, args.branch, environment, snapshot["files_sha256"]
                )
                state = {
                    "uploaded_gate_status": state["pending_gate_status"],
                    "commit": commit,
                    "final_uploaded": state["pending_final"],
                    "verified_files": len(snapshot["files_sha256"]),
                }
                state_path.write_text(json.dumps(state, indent=2))
                print(f"Verified retried GitHub checkpoint {commit}", flush=True)
                if args.once:
                    return
            timeleft = command(["timeleft"])
            match = re.search(r"(\d+):(\d{2}):(\d{2})\s+left", timeleft)
            if not match:
                raise RuntimeError("Cannot determine booking deadline")
            hours, minutes, seconds = map(int, match.groups())
            remaining = hours * 3600 + minutes * 60 + seconds
            gates, evidence = gate_status(args.source, args.data_root)
            terminal = {k: v for k, v in gates.items() if v not in {"NOT_RUN", "IN_PROGRESS"}}
            changed = terminal != state.get("uploaded_gate_status")
            final = remaining <= 1800 and not state.get("final_uploaded")
            if changed or final or args.once:
                command(["git", "fetch", "origin"], cwd=checkout)
                today = datetime.now(ZoneInfo("Europe/London")).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                daily = command(
                    ["git", "log", "--all", f"--since={today.isoformat()}", "--format=%H"],
                    cwd=checkout,
                )
                if len(set(daily.splitlines())) >= 14:
                    raise RuntimeError("Daily commit cap reached (14); no new commit created")
                # Test write authentication before creating an unpushable local checkpoint.
                command(
                    ["git", "push", "--dry-run", "origin", f"HEAD:refs/heads/{args.branch}"],
                    cwd=checkout,
                    env=environment,
                )
                hashes = {}
                for original, relative in compact_files(args.source, args.data_root, args.booking):
                    if original.stat().st_size > MAX_FILE_BYTES:
                        raise RuntimeError(
                            f"Oversized compact artifact requires separate preservation: {relative}"
                        )
                    payload = original.read_bytes()
                    if SECRET.search(payload):
                        raise RuntimeError(f"Possible credential in {relative}; upload stopped")
                    if original.suffix == ".json":
                        try:
                            json.loads(payload)
                        except json.JSONDecodeError:
                            continue  # A producer may still be writing; retry next checkpoint.
                    destination = checkout / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(payload)
                    hashes[str(relative)] = hashlib.sha256(payload).hexdigest()
                snapshot = {
                    "captured_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
                    "booking": args.booking,
                    "gate_status": gates,
                    "evidence": evidence,
                    "timeleft": timeleft,
                    "scope": "Phase 1 only; tested snapshots, no protocol freeze",
                    "large_artifacts": "Not uploaded: source archives, images, weights, embeddings, fitted estimators. Regenerate after booking wipe.",
                    "files_sha256": hashes,
                }
                path = checkout / "research/v2/checkpoint_state.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(snapshot, indent=2))
                command(
                    [
                        "git",
                        "add",
                        "src",
                        "scripts",
                        "tests",
                        "configs",
                        "docs",
                        "research",
                        "README.md",
                        "pyproject.toml",
                        "uv.lock",
                    ],
                    cwd=checkout,
                )
                if (checkout / "AGENTS.md").exists():
                    command(["git", "add", "AGENTS.md"], cwd=checkout)
                command(
                    [
                        "git",
                        "commit",
                        "-m",
                        f"Checkpoint {args.booking}: "
                        + ", ".join(f"{k} {v}" for k, v in terminal.items()),
                    ],
                    cwd=checkout,
                )
                state.update(
                    pending_commit=command(["git", "rev-parse", "HEAD"], cwd=checkout),
                    pending_gate_status=terminal,
                    pending_final=final or state.get("final_uploaded", False),
                )
                state_path.write_text(json.dumps(state, indent=2))
                commit = publish_and_verify(checkout, args.branch, environment, hashes)
                state = {
                    "uploaded_gate_status": terminal,
                    "commit": commit,
                    "final_uploaded": final or state.get("final_uploaded", False),
                    "verified_files": len(hashes),
                }
                state_path.write_text(json.dumps(state, indent=2))
                print(f"Verified GitHub checkpoint {commit}: {terminal}", flush=True)
        except (
            OSError,
            RuntimeError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            # Do not log subprocess stdout/stderr, which could contain auth details.
            print(
                f"Checkpoint pending: {type(exc).__name__}; inspect authentication, budget, or local report validation",
                flush=True,
            )
            if args.once:
                raise SystemExit(1) from exc
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
