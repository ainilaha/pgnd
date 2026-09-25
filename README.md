# Physics-Guided Neural Dynamics for Pantograph–Catenary Contact Force Estimation

**PGND** is a PyTorch research project for predicting contact force from past
panhead acceleration and displacement. It compares a physics-guided latent
ODE with LSTM, GRU, RNN and CNN–GRU under one data and evaluation pipeline.
The current experiments do not establish a significant PGND advantage.

## Structure

```text
src/
├── data.py          # shared preprocessing and windows
├── model.py         # original PGND (PGND-0)
├── baselines.py     # LSTM, GRU, RNN, CNN–GRU
├── train.py         # common training and checkpoint saving
├── evaluate.py      # common metrics and prediction export
└── plot.py          # PDF figures from saved CSVs
```

Repository rules are in [AGENTS.md](AGENTS.md) and [src/AGENTS.md](src/AGENTS.md).
The working draft is [manuscript/main.tex](manuscript/main.tex).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch and `torchdiffeq` provide learning/ODE integration; pandas and the Excel
readers load the original data; Matplotlib produces figures. No TensorFlow,
configuration framework or experiment-management service is needed.

## Data and protocol

Original simulation files are in `data/Data/` (120 XLS files). The three older
XLSX files in `data/data_old/` are preserved but not mixed into this loader.
Datasets and generated results are ignored by Git; back them up separately.
See [data/README.md](data/README.md) for provenance and columns.

- Read headerless files, trim 10% at each end, and reject non-finite values.
- Reserve the final 15% of each training recording's retained rows for
  validation. Test recordings are separate. Split **before** building windows.
- Standardize both inputs and force with training-row mean/std only.
- Use `[acceleration, displacement]` windows to predict the following force:
  `x[i:i+L] -> force[i+L]`. No force history, target-time input or cross-recording
  windows. Every model resets its state per window.
- Infer 1 ms sampling from distance and speed, assuming metres and km/h.
  Physical timestamp units and upstream filter causality still need confirmation.

The commands below retain the existing 20-Hz protocol: all three speeds,
cases 1–6 for training/validation, cases 7–8 for test, L=16, and training target
stride 16. Counts are 12,126 training, 33,983 validation and 69,909 test windows.
Stride does not skip observations inside a window. Validation/test stride is 1.
Changing context or stride requires a new, matched comparison for every model.
Validation shares recording identities with training, so it is not an
independent recording-level generalization estimate.

These leakage-aware choices differ from the old paper's code; scores are not
directly comparable to its published results. The discrepancies are preserved
in the [legacy audit](legacy/README.md).

## Training (remote server only)

Run from the repository root in an environment with the requirements installed.
Choose an unused `RUN` directory. These commands do not overwrite earlier runs.

```bash
set -e
RUN=results/20hz_seed0_clean_run
for model in lstm gru rnn cnn_gru pgnd; do
  python -m src.train --model "$model" \
    --train-pattern 'V(300|350|380)_Case[1-6]_CutFre20\.xls$' \
    --test-pattern 'V(300|350|380)_Case[7-8]_CutFre20\.xls$' \
    --epochs 100 --batch-size 128 --train-stride 16 --sequence-length 16 \
    --cut-percent 0.1 --validation-fraction 0.15 --learning-rate 0.001 \
    --seed 0 --device cuda --threads 1 --preload-data \
    --output "$RUN/$model/$model.pt"
done
```

All models use Adam (betas 0.9/0.999, epsilon 1e-7) and standardized-force MSE.
Baselines retain their two 32-unit recurrent layers; CNN–GRU adds a width-1
64-channel ReLU convolution. PGND retains q/v dimensions 16, encoding 16,
hidden width 32, PSD damping/stiffness, and a neural residual. Its loss adds
`0.001 * mean(sum(residual**2))`. RK4 uses a 1 ms step and millisecond solver
time. These latent coefficients are not calibrated mechanical parameters.
There is no scheduler, early stopping, dropout, clipping or weight decay.

Each model saves `.run.json` (settings, data/source hashes), `.history.csv`
(epoch losses/timings), `.best.pt` (minimum validation prediction MSE), and
`.pt` (final epoch). Checkpoints include complete history and normalization;
they are prediction checkpoints, not optimizer-resume snapshots.

## Evaluation and figures

Use validation during development. For a frozen final comparison, change both
`--split validation` to `--split test` and the output directory to a new name.
Never mix best-validation and final-epoch checkpoints.

```bash
python -m src.evaluate \
  "$RUN/lstm/lstm.best.pt" "$RUN/gru/gru.best.pt" "$RUN/rnn/rnn.best.pt" \
  "$RUN/cnn_gru/cnn_gru.best.pt" "$RUN/pgnd/pgnd.best.pt" \
  --split validation --device cuda --threads 1 --preload-data \
  --output-dir "$RUN/validation"

python -m src.plot "$RUN/validation" --output-dir "$RUN/figures"
```

Evaluation verifies matching data, protocol, seed, budget and checkpoint
selection. It saves pooled/per-recording MAE, RMSE, MSE and R², actual histories,
and ordered predictions/ground truth in newtons with file/row provenance.
Evaluation directories must be new; figure directories may already exist.

Figures are PDF only: separate training and validation loss comparisons,
metric comparison, and a common predicted/ground-truth force curve. Matching
PDFs are overwritten; unrelated files remain untouched. Force plots default
to the first recording; `--recording`, `--start` and `--points` select a segment
without smoothing or changing the underlying results.

## Archived experiments and checks

The former long README, all measured results and proposed methods are retained
in [experiment notes](legacy/experiment_notes.md), not presented as current
instructions. Unsuccessful pilots did not become the default PGND.

Normal training now accepts only the five models above. Existing `pgnd_obs`,
`direct`, `pgnd_modal`, `history_linear` and `history_mlp` checkpoints remain
evaluable with the same command. Their frozen model definitions are in
[pgnd_experiments.py](legacy/pgnd_experiments.py), except the observation-readout
flag retained in `PGNDModel` for compatibility. For historical retraining, the
exact executed sources remain in the corresponding `results/<run>/source/`
snapshots; old pilot launch scripts are not commands for the cleaned trainer.
No data, checkpoints, scientific results or manuscript content were removed.

```bash
python -m unittest discover -s tests -v
git diff --check
```

These small regression checks use synthetic data only, with no optimizer steps
or real-data experiments. Keep future changes scoped; do not add another model
or infrastructure layer without an agreed scientific need and validation plan.
