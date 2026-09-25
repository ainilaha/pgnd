# Source-code boundaries

Apply the root `AGENTS.md` too. Keep these responsibilities separate:

- `data.py`: one shared loader, recording-local splits/windows, train-only
  statistics. Do not create model-specific preprocessing.
- `model.py`: PGND dynamics, state initialization and readout. No file I/O,
  dataset loading, training, metrics or plotting. The optional observation
  readout is retained for old-checkpoint compatibility, not a new default.
- `baselines.py`: the four established PyTorch baseline definitions only.
- `train.py`: explicit model selection, shared Adam/MSE loop, history and
  best/final checkpoints. No separate pipelines per model.
- `evaluate.py`: checkpoint reconstruction, comparison guards, predictions
  and metrics. Only the old-checkpoint compatibility branch may import
  `legacy.pgnd_experiments`; training must not depend on archived experiments.
- `plot.py`: saved CSVs only, no model execution. PDF output only; separate
  training/validation figures, comparable models on common axes. Overwrite
  matching PDFs without deleting unrelated files or changing recorded data.

## Invariants to preserve during cleanup

- Windows use `[acceleration, displacement]` and predict the following force,
  excluding target-time observations and force-history inputs.
- Split rows before windowing, never cross recordings, and fit scalers on
  training rows only. Keep original row provenance and dataset/source hashes.
- Compare identical target rows, split, scaling, seed and training budget.
  Select by validation prediction MSE; do not mix best and final checkpoints.
- Keep checkpoint tensor names/shapes and historical model identifiers
  loadable. Do not refactor numerical operations without equivalence checks.
- Save each epoch's history and preserve complete checkpoint metadata.
  Training/evaluation output directories must not overwrite earlier runs.
- Use plain functions and `nn.Module` classes, with short mathematical or
  shape comments where needed. Do not replace clear repetition with a framework.

## Lightweight checks (from repository root)

```bash
python -m unittest discover -s tests -v
python -m src.train --help
python -m src.evaluate --help
python -m src.plot --help
git diff --check
```

Tests use tiny synthetic arrays only: no dataset files, optimizer steps,
downloads, remote access or training. Add a focused test when behavior changes;
do not add a new test framework. For a pure move, also compare the old/new
definitions and predictions, including archived checkpoint reconstruction.
