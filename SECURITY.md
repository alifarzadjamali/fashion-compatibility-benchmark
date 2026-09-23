# Security and data-handling policy

Do not commit dataset images, embeddings, checkpoints, prediction dumps, access-controlled archives,
API keys, Hugging Face tokens, credentials, or local environment files.

Report a suspected exposed secret privately to the repository owner before opening a public issue.
If a credential is ever committed, revoke it first and then rewrite every reachable Git ref; deleting
the working-tree file is insufficient.

The release process scans all reachable blobs for credential patterns, dataset/image signatures, and
oversized files. `.gitignore` is defense in depth, not a substitute for this audit.

Before sharing a branch or release candidate, inspect the staged file list and
run the same scans against the candidate revision. Pay particular attention to
new notebooks, logs, report attachments, and exported model artifacts, since
these are common paths for accidental data or credential disclosure.
