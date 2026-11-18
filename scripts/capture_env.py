"""Capture reproducibility evidence without dumping environment variables/secrets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch

    evidence = {
        "captured_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "uv_lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "nvidia_smi": command(["nvidia-smi", "-q"]),
        "timeleft": command(["timeleft"]),
        "torch_config": torch.__config__.show(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "packages": sorted(
            (
                {"name": d.metadata["Name"], "version": d.version}
                for d in importlib.metadata.distributions()
            ),
            key=lambda d: d["name"].lower(),
        ),
        "container_digest": None,
        "environment_method": "uv locked host environment",
        "model_revisions": "Not loaded yet; recorded separately by G2",
        "dataset_checksums": "Not acquired yet; recorded separately by G4",
    }
    if torch.cuda.is_available():
        evidence["gpu"] = torch.cuda.get_device_name()
        evidence["compute_capability"] = torch.cuda.get_device_capability()
        evidence["bf16_supported"] = torch.cuda.is_bf16_supported()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(evidence, stream, indent=2)


if __name__ == "__main__":
    main()
