# Research continuation instructions

For v2 research work, first read `HANDOFF_V2.md` completely, then
`docs/RESUME_V2.md` and `docs/v2_decisions.md`. Inspect live DGX jobs and recorded
evidence before starting or repeating a stage.

- Phase 1 only unless the owner explicitly authorizes Phase 2.
- No new-setting or POG test evaluation before the protocol freeze. The owner
  authorizes the narrowly scoped v1-only G3 reproduction described in the log.
- Original v1 split hashes are unavailable. The owner approved revised G3:
  two matching deterministic rebuilds, published counts and published metric
  comparisons, with explicit disclosure that original split identity is unknown.
- Preserve v1 artifacts. A100 is evaluation-only. Never commit credentials,
  raw datasets, weights or large embedding caches.
- **Fewer than 15 commits per calendar day**: maximum 14, including automated
  checkpoint commits. Batch changes; checkpoint completed gates and verify upload.
- SSH/tmux jobs can continue without this chat. DGX storage and tmux are wiped
  at booking expiry. GitHub code/evidence does not preserve large caches.
- Keep the recovery document and decision log current; do not rely on chat history.
