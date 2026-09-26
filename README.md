# Physics-Guided Neural Dynamics for Pantograph–Catenary Contact Force Estimation

**PGND** is a research project for contact-force estimation from acceleration
and uplift. This repository currently contains only complete simulated-data
infrastructure and verified conventional baselines. Continuous-time models
will be rebuilt separately; no PGND, Neural ODE, irregular-sampling generator
or training entry point is currently implemented.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Dependencies are NumPy, pandas, xlrd, PyTorch and Matplotlib. No TensorFlow,
ODE library, configuration framework or experiment manager is required.

## Structure

```text
src/
├── AGENTS.md
├── data.py           # complete recordings and train-only normalization
├── baseline_data.py  # explicit baseline trim and regular windows
├── baselines.py      # unchanged CNN–GRU, GRU, LSTM, RNN
├── metrics.py        # scalar MAE, RMSE, MSE, R²
├── evaluate.py       # clean baseline inference, metrics and target provenance
└── plot.py           # small PDF plotting functions
tests/
├── test_core.py      # synthetic checks, no training
└── BASELINE_VERIFICATION.md
```

Raw local data are in `data/`; generated artifacts belong in ignored `results/`.
The manuscript in `manuscript/` and reference PDFs in `papers/` are preserved,
but the draft does not describe the current baseline-only software. Historical
code is in Git, not an active `legacy/` tree. There are no empty configs/scripts
directories or new model packages.

## Canonical data

`load_recordings(cutoff=20)` returns separate `Recording` objects, never a
concatenated stream or window dataset. Each has a filename, split, and a table
indexed by the original zero-based source row, with these aligned columns:

`distance_m, acceleration, uplift, force_N, time_s`

The loader reads all four headerless source columns without trimming, filtering,
normalizing, removing rows or imputing values. `uplift` is the source displacement
column, renamed without conversion. Invalid/non-finite rows and non-increasing
time fail explicitly. See [data/README.md](data/README.md) for raw-data provenance.

The **established whole-case split** is fixed across speeds and cutoff variants:
cases 1–4 train, 5–6 validation, 7–8 test. There is no random row split. The older
within-recording 15% validation-tail protocol is not an active option; its saved
LSTM/RNN test predictions were checked separately during cleanup.

Files supply distance rather than timestamps. `time_s = distance_m / (speed/3.6)`
uses the speed in the filename, assumes constant speed, metres and km/h, and
preserves the original origin. The inferred interval is approximately 1 ms.
`CutFre20` is a filter label, **not** a 20 Hz sampling rate. Other cutoff files
also change force values and must not be treated as identical ground truth.

## Baseline adapter and normalization

The verified alignment is **`X[k-L:k] -> force[k]`**: L preceding sensor rows,
excluding target-time sensors. This is one-step-ahead prediction, not the
inclusive-current-sample reconstruction sometimes used in notation. Windows
are constructed separately inside each recording; recurrent state resets per
window. The adapter checks regular time spacing and consecutive source rows.

For the established clean CNN–GRU reference, explicitly trim `floor(0.1*N)`
rows from each end, use L=64 and training target stride 16. Validation/test use
stride 1. These are **reference settings**, not canonical-data defaults.
The adapter gives 9,401 / 76,989 / 69,621 train/validation/test windows.

Fit population means/stds on the trimmed **training rows before windowing**,
including training context rows. Force and the two inputs are standardized
separately, with the existing 1e-12 std floor. Fitting rejects held-out records
and duplicate identities. Transforming never refits the scaler or mutates raw
recordings. Window tensors are float32; statistics and raw rows use float64.

```python
from src.data import load_recordings, fit_normalization
from src.baseline_data import trim_recording, make_windows

recordings = load_recordings(cutoff=20)
training = [trim_recording(r, 0.1) for r in recordings if r.split == "train"]
statistics = fit_normalization(training)
x, y, targets = make_windows(training[0], statistics, sequence_length=64, stride=16)
```

## Models and evaluation

All four definitions and initializers are unchanged from commit `bf4c1857`.
Each accepts `(batch, L, 2)` and outputs `(batch, 1)`. All have two 32-unit
recurrent layers, a linear scalar readout and no dropout. CNN–GRU first applies
a width-1, 64-channel convolution and ReLU. Their only dependency is PyTorch;
no model loads data or constructs windows. State-dict keys/shapes are unchanged.

Use the class corresponding to your checkpoint and its **saved normalization**.
For example, with a trusted, recovered whole-case CNN–GRU reference checkpoint:

```python
import torch
from src.baselines import CNNGRU
from src.evaluate import evaluate

# Full historical checkpoints contain metadata as well as tensors.
# weights_only=False is for your own trusted files only.
checkpoint = torch.load("/path/to/cnn_gru_l64.best.pt", map_location="cpu", weights_only=False)
assert checkpoint["model"] == "cnn_gru"
assert checkpoint["protocol"] == "case-validation-training-standardization-v1"
assert checkpoint["normalization"] == statistics
model = CNNGRU()
model.load_state_dict(checkpoint["state_dict"], strict=True)
scores, predictions = evaluate(
    model, [r for r in recordings if r.split == "test"], statistics,
    sequence_length=64, cut_percent=0.1,
)
```

`scores` contains `mae`, `rmse`, `mse`, `r2`; errors are in N (MSE in N²).
The generic metric function uses float64, with NaN R² for constant targets.
`predictions` retains file, split, source row, time, distance, force and prediction.
Use identical target rows and protocols for comparisons. The old experiment
CLI/checkpoint-reconstruction machinery was deliberately removed; there is no
automatic support for experimental model identifiers or tail-split validation.

## Plots and verification

`src.plot` exposes `plot_losses`, `plot_metrics` and `plot_force`, accepting
tables, labels, colors and an output directory. Loss tables use `epoch`,
`train_mse_scaled`, `val_mse_scaled`. Metric tables use the four score keys
above; force tables are those returned by `evaluate`. Force comparisons reject
different ordered target rows/ground truth. PDF output only; matching figures
are overwritten. Plotting performs no model execution or data processing.

The cleanup checks and clean-checkpoint reproduction results are recorded in
[tests/BASELINE_VERIFICATION.md](tests/BASELINE_VERIFICATION.md). No models were
trained and no remote actions were performed. Checkpoints and generated outputs
are intentionally absent from the active tree; Git does not back up ignored
datasets or results. Keep those separately if long-term recovery is needed.
