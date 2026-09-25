# PGND repository rules

This is a small research codebase, not a software framework. The priority is
readable code, reproducible comparisons and preserving scientific evidence.
Read `README.md` and, before editing Python, `src/AGENTS.md`.

## Keep the scope small

- Change only what the user requests. Cleanup does not authorize new models,
  changed experiments, manuscript rewrites, remote syncs or training runs.
- Keep the six flat source files: `data.py`, `model.py`, `baselines.py`,
  `train.py`, `evaluate.py`, `plot.py`. Prefer editing them to adding modules.
- Do not add factories, registries, generic trainer classes, configuration
  frameworks, experiment managers, packages, CI or Docker. A small explicit
  conditional is preferable to infrastructure for hypothetical future models.
- Use existing dependencies. Explain the concrete need before adding a file,
  dependency, CLI option or architectural layer. Do not create placeholders.
- Keep `README.md` an operating guide, not a chronological experiment diary.
  Historical findings belong in `legacy/experiment_notes.md`; run-specific
  scripts, settings, logs and outputs belong in `results/<run>/`.

## Protect the science

- The normal training choices are PGND, LSTM, GRU, RNN and CNN-GRU. Existing
  experimental checkpoints remain evaluable; this does not promote their
  architectures into the active training interface.
- Do not silently change inputs/targets, units, data splits, normalization,
  window alignment, initializers, architecture, loss, optimizer, solver,
  checkpoint selection or metrics. State any scientific change explicitly.
- Before a requested method experiment, declare one hypothesis, fair controls,
  validation criterion and bounded run budget. Do not keep adding variants
  until one wins. Promotion into `src/` requires the user's agreement and
  documented evidence; a single-seed pilot is not statistical significance.
- Preserve failed experiments and scientifically useful baselines. Archive
  their implementation and provenance rather than erasing negative evidence.
- Raw data, saved checkpoints, histories and predictions are immutable unless
  the user explicitly requests otherwise. Never regenerate or retouch scores
  to match manuscript text. The draft is in `manuscript/main.tex`; proposals
  and placeholders there are not measured results or permission to implement.

## Execution and handoff

- Do not train or run real-data experiments locally. Small synthetic tests,
  syntax checks and checkpoint compatibility checks are allowed.
- Remote training/sync requires an explicit request covering that action;
  old credentials or prior experiments are not standing authorization. Never
  put passwords, keys or tokens in files, commands, logs or instructions.
- If model runs are authorized in parallel, use independent model CLI jobs.
  Do not add multiprocessing or threading machinery to the training code.
- Inspect the worktree first, preserve unrelated edits, and review the diff.
  Use the lightweight checks in `src/AGENTS.md`; report unavailable checks.
- At handoff, state what changed, what was verified, any compatibility or
  scientific changes, and whether training or remote actions occurred.
