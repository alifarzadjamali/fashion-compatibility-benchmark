"""G1 checks: execute only environment tests, never dataset/model evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checks = {}
    import torch

    from repbench.utils.seeds import seed_everything
    from repbench.v2.state import input_hashes

    seed_everything(20260912)
    checks["python312"] = sys.version_info[:2] == (3, 12)
    checks["gpu_visible"] = torch.cuda.is_available()
    checks["bf16_supported"] = checks["gpu_visible"] and torch.cuda.is_bf16_supported()
    if checks["bf16_supported"]:
        q = torch.randn(2, 4, 32, 64, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            output = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            matmul = q @ q.transpose(-1, -2)
        torch.cuda.synchronize()
        checks["bf16_sdpa_finite"] = bool(torch.isfinite(output).all())
        checks["bf16_matmul_finite"] = bool(torch.isfinite(matmul).all())
    with (args.output / "pytest.txt").open("x") as stream:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-ra"],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    checks["pytest"] = result.returncode == 0
    result = subprocess.run(
        [
            sys.executable,
            "scripts/capture_env.py",
            "--output",
            str(args.output / "environment.json"),
        ],
        check=False,
    )
    checks["environment_capture"] = result.returncode == 0
    record = {
        "gate": "G1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "utc": datetime.now(UTC).isoformat(),
        "checks": checks,
        "input_sha256": input_hashes(),
        "check_seconds": time.perf_counter() - started,
        "note": "Build timing is in setup log; skipped dataset tests do not certify data readiness.",
    }
    (args.output / "gate.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)
    sys.exit(0 if record["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
