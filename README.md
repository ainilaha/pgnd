# Physics-Guided Neural Dynamics for Pantograph–Catenary Contact Force Estimation

**PGND** is a PyTorch research project for predicting contact force from past
panhead acceleration and displacement. It compares a physics-guided latent
ODE with LSTM, GRU, RNN and CNN–GRU under one data and evaluation pipeline.
The current experiments do not establish a significant PGND advantage.

## Structure

```text
src/
├── data.py          # shared preprocessing and windows
├── model.py         # current PGND; historical ablations are archived
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

## Persistent-state clean PGND (opt-in experiment)

`src.train --model pgnd --persistent-state` replaces independent windows with
ordered recording streams. Set `--sequence-length 64 --target-start 64` for the
current comparison. The existing initializer runs once at each partition's
first observation; the first 64 observations warm the ODE before the first
scored next-sample target. State is never carried across files, splits or epochs.

The assimilated state uses the existing linear encoded-observation interpolation.
Each next-sample forecast branches from the preceding state with a held final
encoding, so target-time observations cannot enter that prediction. Once the
next observation arrives, the assimilated trajectory continues from the preceding
state using the now-known interpolation endpoints. There is no force feedback,
new initializer, decoder, residual, loss term or solver. The old `forward()` and
all window-trained checkpoints remain unchanged.

Recording mode necessarily extends the information history and keeps a global
recording-relative clock in the time-dependent residual. Training uses ordered
chunks with at most L intervals (including this L-sample warm-up), detaches
gradient history without resetting state values, and accumulates the original
batch-size count of targets before each Adam update. The residual penalty still
averages the last L evolved grid points per target, with detached earlier-chunk
states. This is truncated backpropagation, not full-recording backpropagation;
gradient order/history differs from the shuffled window control.

This first pilot is clean PGND/RK4 only: no missing-block augmentation or
window-based `--diagnostics`. Checkpoints record the state protocol. The ordinary
evaluation command automatically uses persistent propagation for those weights
and window evaluation for old controls, verifying identical target provenance.
Comparisons report `state_mode` and `history_scope`; persistent PGND has more
historical information than windowed baselines, so a win would not by itself
establish a mechanical-structure benefit. The manuscript still describes the
window-reset method; a streaming equation does not describe those older runs.

## Missing-observation pilot (opt-in; not yet trained)

Hypothesis: training with short sensor outages reduces force error when recent
measurements are unavailable. This is shared augmentation, not a PINN, new
architecture or demonstrated PGND improvement.

- `--max-missing-block N` works for all five models. Each training window is
  clean with probability 50%; otherwise one contiguous two-channel block has
  uniform length 1..N and a uniform valid start after sample zero. Fill it from
  the immediately preceding observation; subsequent observations resume normally.
  First observations, targets, timestamps and original data are untouched.
- A dedicated NumPy generator uses the run seed. Matched windows, minibatches
  and epoch budgets give all models matching masks, independent of model RNGs.
  N=0 is the default and leaves original numerical behavior unchanged.
- Train-only scalers use original training rows. Validation/checkpoint selection
  remain **clean force MSE**. Augmented training curves measure corrupted-input
  MSE, not clean training error. The objective, solver and architectures do not change.
- Evaluation `--missing-block G` hides the last G input samples in every window,
  with the same past-value fill and identical targets for all models. G=0 is
  clean. The last real observation is target row minus G minus 1.
- Checkpoints save augmentation settings. Every evaluation saves `evaluation.json`
  with the condition and hashes, gap fields in metrics, and last-observed-row
  provenance in predictions. Comparing clean/augmented training is intentional;
  existing split, scaling, seed, budget and checkpoint-selection guards remain.
- This is a **fixed-grid, filled outage per window**, not a recording-wide
  missingness mask, native irregular sampling, single-channel failures or noise.
  No mask/elapsed-time channels are added. Raw non-finite/irregular files still
  fail loudly; field-data ingestion needs the actual sensor/timestamp specification.
  PGND's existing interpolation is retrospective within the available history,
  not a claim of zero-delay streaming estimation or recovery of missing truth.

### Fixed first-screen protocol

Four runs only: PGND and CNN–GRU, each clean/augmented, seed 0, 100 epochs.
Keep all speeds, CutFre20, L=64, target anchor 64, stride 16, batch 128 and Adam
.001. Train cases 1–4, validate on 5–6; this explicitly differs from the older
tail-split commands above. Cases 7–8 were previously inspected and are not an
untouched confirmation set; the screen does not score them.

Train maximum gap 16. Evaluate gaps 0, 4, 16 and 32 samples (approximately
milliseconds, pending timestamp confirmation); 32 is secondary extrapolation.
Proposed success criterion against each model's clean-trained control: >=5%
reduction in mean RMSE over gaps 4 and 16, improvement in both validation cases
using equal-recording case MSE, and <=1% degradation in clean pooled RMSE.
These are practical screening thresholds, not significance. Compare PGND against
**equally augmented** CNN–GRU too. Shared augmentation gains do not establish a
mechanical-structure benefit. Report every condition, including negative results.
No extra seeds, variants or tuning are authorized automatically.

### Remote commands (not executed)

Run in Bash from `/root/pgnd` with a new RUN. Remote synchronization/training is
a separate action. LSTM/GRU/RNN support the same option but are not additional
runs in this bounded first screen.

```bash
set -euo pipefail
RUN=results/missing_blocks_seed0_v1
COMMON=(
  --train-pattern 'V(300|350|380)_Case[1-4]_CutFre20\.xls$'
  --validation-pattern 'V(300|350|380)_Case[5-6]_CutFre20\.xls$'
  --test-pattern 'V(300|350|380)_Case[7-8]_CutFre20\.xls$'
  --sequence-length 64 --target-start 64 --train-stride 16
  --epochs 100 --batch-size 128 --learning-rate 0.001
  --seed 0 --device cuda --threads 1 --preload-data
)
for model in pgnd cnn_gru; do
  for maximum in 0 16; do
    name="${model}_gap${maximum}"
    python -m src.train --model "$model" "${COMMON[@]}" \
      --max-missing-block "$maximum" --output "$RUN/$name/$name.pt"
  done
done

CHECKPOINTS=(
  "$RUN/pgnd_gap0/pgnd_gap0.best.pt"
  "$RUN/pgnd_gap16/pgnd_gap16.best.pt"
  "$RUN/cnn_gru_gap0/cnn_gru_gap0.best.pt"
  "$RUN/cnn_gru_gap16/cnn_gru_gap16.best.pt"
)
for gap in 0 4 16 32; do
  python -m src.evaluate "${CHECKPOINTS[@]}" \
    --split validation --missing-block "$gap" --device cuda --threads 1 \
    --preload-data --output-dir "$RUN/validation_gap$gap"
  python -m src.plot "$RUN/validation_gap$gap"
done
```

## Diagnostics

### Irregular-observation pilot

`data.observation_mask` makes nested recording-level sensor masks;
`compact_observations` retains only available values and their relative times.
`PGNDModel.forward_events` integrates those events without reconstructing raw
missing sensors. It is an opt-in **window-based** path, not a persistent-state
observer. CNN-GRU uses prefix-local interpolation through
`interpolate_observations`, never target/future sensor values. The ordinary
training/evaluation commands and existing checkpoints are unchanged.

The bounded seed-0 pilot, its checks, exact source snapshots and launchers are
in `results/irregular_seed0_20260926_gyHiJX/`. Read `PROTOCOL.md` and
`MATCHED_PROTOCOL.md` before reproducing it. The elapsed-time GRU definition
is pilot-local in `matched.py`, not an additional normal training option.
Use a **new** output directory for any rerun; original histories, masks,
predictions and checkpoints must not be overwritten. `summarize.py` reads
saved outputs only and regenerates PDF figures without executing models.

### Saved-model probes

`src.train --diagnostics` retains fixed clean training probes without updates.
`src.evaluate --split validation --diagnostics` retains gradients/state probes,
frozen solver checks, residual-rescaling checks and case comparisons. These
remain useful numerical checks, not experimental model menus. Evaluation probes
use the requested gap; histories keep their original clean validation criterion.
Energy terms are latent, not identified physical energy.

Priority 0–3 protocols and launcher instructions are archived in
[experiment notes](legacy/experiment_notes.md). Old ablations remain evaluable,
but historical retraining requires the matching run's source snapshot, not
the simplified trainer.

## Archived experiments and checks

The former long README, all measured results and proposed methods are retained
in [experiment notes](legacy/experiment_notes.md), not presented as current
instructions. Unsuccessful pilots did not become the default PGND.

Normal training accepts only the five models above. P1–P3 variants and `pgnd_obs`,
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
