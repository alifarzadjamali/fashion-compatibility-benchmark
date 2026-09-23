# Maintenance checklist

Use this checklist for small post-release repository updates:

1. Keep protocol manifests, frozen configurations, and reported results
   unchanged unless the change is an explicitly reviewed experiment revision.
2. Keep datasets, embeddings, checkpoints, raw predictions, credentials, and
   local caches outside the public repository.
3. Preserve the distinction between historical, clean, and external evaluation
   protocols when editing documentation.
4. Run the repository tests and the release-artifact audit before publishing a
   change that affects code, manifests, or package contents.
5. Explain any changed assumption in the decision log or the relevant report.
