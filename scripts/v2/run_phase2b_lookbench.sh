#!/usr/bin/env bash
# Phase 2b: LookBench sensitivity. The standalone `run_sensitivities.py lookbench` CLI crashes in
# main() (its subparser has no --equivalence). This calls the same function with the same
# arguments that `run_sensitivities.py all` uses (results/metrics.csv, default replicates/seed),
# writing to the same directory. No .py change; disclosed in docs/v2_decisions.md.
set -uo pipefail
cd "$(dirname "$0")/../.."
F=${REPBENCH_DATA_ROOT:-/data/$USER/repbench/data}/v2/full
export PYTHONHASHSEED=20260912 OMP_NUM_THREADS=2
echo "=== lookbench (library call) start $(date '+%F %T')"
.venv/bin/python - "$F" <<'PY'
import sys
from pathlib import Path
from repbench.v2 import sensitivities as S
F = Path(sys.argv[1])
S.run_lookbench(F / "results/metrics.csv", Path("configs/lookbench_v3_scores.json"),
                F / "results/sensitivities/lookbench", replicates=2000, seed=S.PROBE_SEED)
PY
rc=$?; echo "=== lookbench end rc=$rc $(date '+%F %T')"; exit $rc
