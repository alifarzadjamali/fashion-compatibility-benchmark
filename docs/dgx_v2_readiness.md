# DGX v2 readiness operations

Phase 1 only. No full experiment command is enabled. The initial booking on
2026-11-18 had about three hours remaining, not 48 hours. Gates must pass in
order. A green unit-test suite alone does not certify model, data or scientific
readiness. See the [decision log](v2_decisions.md).

## Rebuild and start

Clone the readiness branch into the booking's home directory. Python 3.12 is
required. Data, model and package caches live in `/data/$USER/repbench`, never
inside `.venv`. The original v1 environment files are archived in
`configs/v1_environment/`; v1 scientific defaults and protocol hashes are unchanged.

```bash
git clone --branch v2-readiness https://github.com/alifarzadjamali/fashion-compatibility-benchmark.git
cd fashion-compatibility-benchmark
mkdir -p /data/$USER/repbench/logs
timeleft
tmux new-session -d -s repbench-readiness \
  'bash scripts/v2/run_readiness.sh booking01 > /data/$USER/repbench/logs/readiness.log 2>&1'
```

Use a unique attempt name. The script rebuilds from the lock, runs G1, then
attempts G2. It does not proceed to G3 or later gates. G2 uses 64 generated
images, with a separate process for each encoder and the MLLM. The latter
returns next-token log probabilities, not embeddings. Each model has a 15-minute
attempt budget; no attempt starts with less than that plus a 30-minute upload
reserve remaining. Timed-out downloads retain the HF cache for later retries.

G1 records source/lock hashes. G2 rejects stale G1 evidence. Completed reports
are immutable; a fresh attempt reuses downloaded resources but repeats checks.
An interrupted G2 can resume into its directory provided its gate report was
not finalized and code/lock signatures are unchanged. This is **within-booking**
resumption only.

For gated models, authenticate on the remote machine after environment creation:

```bash
HF_HOME=/data/$USER/repbench/cache/huggingface uv run --frozen hf auth login
```

Do not put access tokens in commands, config files, logs or Git. Login is a
one-time interactive prerequisite; research jobs run non-interactively.

## Reconnect, monitor and stop

```bash
ssh a_jamali2@spark.salford.cloud
tmux attach -t repbench-readiness
# Detach: Ctrl+B, then D
tail -f /data/$USER/repbench/logs/readiness.log
timeleft
```

For a graceful stop after the current model:

```bash
touch ~/fashion-compatibility-benchmark/research/v2/readiness/booking01/g2/STOP
```

For an urgent stop, attach and press Ctrl+C. An incomplete model record is not
accepted as a pass. Preserve the logs and use a fresh attempt directory after
an interrupted write. Never use `tmux kill-server` when unrelated jobs may exist.
tmux does not survive the end of a booking.

## Preservation boundaries

Track code, locks, configuration, decisions, compact gate evidence and environment
manifests in `research/v2/`. Inspect logs before staging; credentials and signed
download URLs must never be committed. Ignore raw data, weights, embeddings,
partial files and arbitrary binary artifacts. The default repository ignores
`artifacts/` and `reports/`, so files placed there are **not** preserved merely
by pushing the code.

Before booking expiry, copy the compact evidence, commit/push it, and verify
the remote commit and file hashes against a fresh fetch/clone. Remote upload
authentication must be verified separately; local GitHub write access does not
establish that a disconnected DGX job can upload on its own.

G3 additionally requires the original v1 split hashes/files and reference results.
Do not manufacture a reference from a reconstruction. Archive v1 inputs without
overwriting them. Missing original inputs block G3.

## Provisional multi-booking design (not yet a runtime estimate)

G6 must measure real 5,000-image extraction for the four required encoders.
Synthetic G2 timings are not substitutes. Include downloads, checksum/audit
passes, extraction, native and PCA fits, all learners/seeds/fractions, both
baseline families, analysis, upload and a conservative reserve in the estimate.

If the full study does not fit, freeze the **entire** scientific protocol before
any new test outcomes, then use these unchanged scientific units:

1. For each assigned encoder × dataset, extract features and complete all probe
   settings that need those features within one booking. Save per-example
   predictions, question IDs/order, labels, cluster assignments, validation
   selection evidence, seeds and settings plus hashes. Summary metrics alone
   cannot support paired statistics or failure-consensus analysis.
2. Preserve those compact bundles on GitHub (ordinary Git only for genuinely
   small files; release assets for larger bundles if limits and upload access
   are verified). Download and hash-check each bundle before declaring it safe.
3. Once complete clean-validation results select best generic/fashion models,
   re-extract their inputs if necessary in the contextual-baseline booking.
   Do not claim large feature caches survive. Include this repeated work in
   the budget. Keep all required seeds, learning rates and datasets.
4. Run the frozen zero-shot question sets in a separate booking if needed;
   retain scores and option-order mappings. Perform isolated efficiency profiling
   without concurrent CPU/GPU workloads. Combine preserved predictions for all
   paired statistics and generated paper artifacts.

This requires exact source data and pinned weights to remain obtainable each
booking. POG's CDN makes that uncertain: a local mirror is not durable storage.
POG cannot be called reproducible across bookings until its required images
can be restored byte-for-byte. Do not redistribute those images through GitHub.
If its preservation requirement cannot be met, raise that scientific decision
before freeze rather than silently changing the POG go/skip criterion.

No final Phase 2 schedule is approved or claimed feasible until measured G6
timings and all artifact restoration requirements are available.
