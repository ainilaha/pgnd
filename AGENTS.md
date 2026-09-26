# PGND repository rules

This is a small research foundation, not a framework. Read `README.md` and
`src/AGENTS.md` before editing. Inspect the worktree and preserve unrelated edits.

- Keep the active code flat and manually auditable. No factories, registries,
  configuration frameworks, generic trainers, empty modules, CI or Docker.
- The active scope is complete simulated recordings, verified conventional
  baselines, normalization, metrics and plotting. Do not add PGND/Neural ODEs,
  irregular sampling, imputation or experimental modes without a new request.
- Git history preserves old exploratory implementations. Do not recreate a
  `legacy/` directory or copy historical code into the active tree.
- Preserve raw files and manuscript sources. Never change scientific behavior
  silently: report changes to splits, inputs/targets, normalization, alignment,
  model architecture, initializers, optimization, checkpoint selection or metrics.
- Before a requested experiment, declare the hypothesis, matched controls,
  validation criterion and bounded run budget. Negative results count as evidence.
- Generated results belong in ignored `results/`. Do not overwrite scientific
  artifacts or delete them without explicit authorization. Figure regeneration
  may overwrite matching PDFs, not unrelated files or source data.
- Do not train or launch dataset experiments locally. Synthetic tests and
  explicitly requested read-only checkpoint reproduction checks are allowed.
- Remote sync/training needs a current explicit request. Old credentials are
  not standing authorization. Never store credentials in files, commands or logs.
- Do not change `manuscript/` merely to match code or scores. Its proposals are
  not measured results or permission to implement a new method.
- Review the diff and run focused tests before handoff. State what changed,
  what was verified, limitations, and whether training/remote actions occurred.
