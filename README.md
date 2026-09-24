# Physics-Guided Neural Dynamics for Pantograph–Catenary Contact Force Estimation

**PGND** is a research project for estimating pantograph–catenary contact forces from panhead acceleration and displacement using physics-guided continuous-time neural dynamics. The repository includes shared legacy data preparation and PyTorch RNN, LSTM, GRU, and CNN–GRU baselines. PGND itself remains unimplemented pending the choices identified in the [migration audit](legacy/README.md).

The current formulation is in [main.tex](main.tex); the previous paper is in `papers/`. Read the [migration audit](legacy/README.md) before treating a run as a reproduction of the paper: its experiment descriptions and code are not fully consistent.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The active code uses PyTorch only. `pandas` and `xlrd` read the original `.xls` files; `openpyxl` supports inspection of the older `.xlsx` collection. TensorFlow/Keras and notebook plotting dependencies are not required or installed.

## Data

The original spreadsheets were moved unchanged to `data/Data/` (120 `.xls` files) and `data/data_old/` (3 `.xlsx` files). They are local and ignored by Git. See [data/README.md](data/README.md) for their observed layout and provenance.

`src/data.py` retains trimming, raw feature values, concatenation, and the mapping from the preceding 16 observations to the next force sample. It deliberately preserves the legacy header-reading and window/split behavior, with warnings. Filenames are now sorted, which can change the validation tail compared with the old filesystem order. No additional filtering or normalization is applied.

## Training

Run from the repository root. This example selects the notebook's 20 Hz case split; it is a compatibility experiment, not a finalized PGND protocol:

```bash
python -m src.train --model lstm \
  --train-pattern 'V(300|350|380)_Case[1-6]_CutFre20\.xls$' \
  --test-pattern 'V(300|350|380)_Case[7-8]_CutFre20\.xls$' \
  --output results/lstm_20hz.pt
```

Use `rnn`, `lstm`, `gru`, or `cnn_gru` with the same arguments for comparison. Defaults recovered from the notebooks/configurations are 20 epochs, 16 observations per window, 10% trimming at **each** end of each file, a final 15% validation tail, batch size 32, shuffled training windows, Adam (`lr=0.001`, `betas=(0.9, 0.999)`, `eps=1e-7`), and force MSE. There is no early stopping, scheduler, dropout, weight decay, or test-based model selection.

File patterns are required so the script does not silently choose between conflicting paper/code protocols. Identical train/test files are rejected; shared cases/speeds at different cutoffs produce a warning. Checkpoints contain final-epoch weights, ordered file lists, original-file hashes, preprocessing settings, split index, seed, and actual training/validation histories. Existing checkpoint paths are not overwritten. Seed 0 and sorted filenames are new reproducibility choices; they do not reproduce the original random draws.

## Evaluation

```bash
python -m src.evaluate results/lstm_20hz.pt
```

Evaluation reads the test files recorded in the checkpoint, verifies their hashes, applies the same preprocessing, and reports scalar-force MSE in N². Test data are not used during optimization. Results and checkpoints belong under `results/` and are ignored by Git.

## Source layout

- `src/data.py`: shared spreadsheet loading, trimming, windows, and legacy validation split.
- `src/baselines.py`: four PyTorch baseline architectures.
- `src/model.py`: PGND placeholder referencing the manuscript equations.
- `src/train.py`, `src/evaluate.py`: shared baseline training and force-MSE evaluation.
- `legacy/`: reference implementations and the audit of unresolved scientific issues.

The legacy validation split shares observations between adjacent windows, and concatenation creates windows across simulation boundaries. These are retained and reported for compatibility, not endorsed as the protocol for new PGND claims. Resolve recording-level splits, normalization, causal timing, and target alignment before comparing PGND with newly trained baselines.
