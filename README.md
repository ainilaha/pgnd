# Physics-Guided Neural Dynamics for Pantograph–Catenary Contact Force Estimation

**PGND** is a research project for contact-force estimation from acceleration
and uplift. This repository contains complete simulated-data infrastructure,
shared irregular-observation masks and verified conventional baselines.
Continuous-coordinate models will be rebuilt separately; no PGND, Neural ODE or
training entry point is currently implemented.

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
├── sampling.py       # shared masks and retained distances/sensors
├── baseline_data.py  # explicit trim, linear interpolation and regular windows
├── baselines.py      # unchanged CNN–GRU, GRU, LSTM, RNN
├── metrics.py        # regression and removed-only interpolation errors
├── evaluate.py       # clean baseline inference, metrics and target provenance
└── plot.py           # small PDF plotting functions
tests/
├── test_core.py      # synthetic checks, no training
├── test_sampling.py  # shared masks, interpolation, clean-path equivalence
└── BASELINE_VERIFICATION.md
```

Raw local data are in `data/`; result records belong in version-controlled
`results/`. Figures, metrics, verification records, reproduction scripts, run logs
and histories are tracked; checkpoints, caches and bulk predictions remain ignored.
The manuscript in `manuscript/` and reference PDFs in `papers/` are preserved,
but the draft does not describe the current baseline-only software. Historical
code is in Git, not an active `legacy/` tree. There are no empty configs/scripts
directories or new model packages.

## Canonical data

`load_recordings(cutoff=20)` returns separate `Recording` objects, never a
concatenated stream or window dataset. Each has a filename, split, and a table
indexed by the original zero-based source row, with these aligned columns:

`distance, acceleration, uplift, force_N`

The loader reads all four headerless source columns without trimming, filtering,
normalizing, removing rows or imputing values. `uplift` is the source displacement
column, renamed without conversion. `distance` is exactly the original first
XLS column in **meters**, with its original origin and precision. Units are
documented in `samples.attrs["units"]`. Invalid/non-finite rows and non-increasing
distance fail explicitly. Nonuniform coordinates are preserved, not repaired.
See [data/README.md](data/README.md) for raw-data provenance.

The **established whole-case split** is fixed across speeds and cutoff variants:
cases 1–4 train, 5–6 validation, 7–8 test. There is no random row split. The older
within-recording 15% validation-tail protocol is not an active option; its saved
LSTM/RNN test predictions were checked separately during cleanup.

Files supply distance, not a clock. **No synthetic timestamps are created.**
Nominal speed stays filename metadata, not a coordinate-conversion rule.
`distance_spacing(recording.samples.distance)` returns all N−1 positive
coordinate differences in meters, without adding an interval before the first row.
`CutFre20` is a filter label, **not** a 20 Hz sampling rate. Other cutoff files
also change force values and must not be treated as identical ground truth.

## Baseline adapter and normalization

The verified clean alignment is **`X[k-L:k] -> force[k]`**: L preceding sensor rows,
excluding target-time sensors. On clean data this is one-step-ahead prediction, not the
inclusive-current-sample reconstruction sometimes used in notation. Windows
are constructed separately inside each recording; recurrent state resets per
window. The adapter checks regular distance spacing (`rtol=1e-4`, `atol=1e-9 m`)
and consecutive source rows; it never replaces the original coordinate values.
For irregular inputs, interpolation can introduce future sensor information
into those same preceding rows, as detailed below.

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

## Irregular observations (data pipeline only)

The study concerns dynamics over a continuous distance coordinate, not imputation. `sampling.py` is
the **only mask generator**. Create a mask once per recording/condition and pass
that same object to each model's adapter. Acceleration and uplift share the mask.
The generator never reads sensor or force values. Masks preserve file identity,
split and source-row indexing; adapters reject mismatched masks.

- Supported retention ratios: `1.0`, `0.8`, `0.6`. Remove exactly
  `round(N*(1-retention))` samples, so the achieved fraction is within `0.5/N`.
- **First and last samples always retained**, including the endpoints of a common
  trimmed segment. Every missing interval has observed left/right endpoints;
  there is no boundary extrapolation. Infeasible retention/burst counts on tiny
  segments raise an error rather than silently changing the requested counts.
- `random`: uniformly remove individual rows without replacement, excluding
  both endpoint anchors. Masks are nested across retention levels for a fixed seed.
- `bursty`: randomly place nonoverlapping **8-sample bursts** by default, with
  at least one observed row between bursts. One shorter burst supplies any
  remainder needed for the exact count. Eight removed samples create nine
  original distance intervals between the retained endpoints. The actual gap
  is their coordinate difference in **meters**, not a synthetic time duration.
  The same missing-row count need not span the same distance across speeds. Burst length
  is an explicit argument; burst masks at different retentions are not nested.
- NumPy PCG64 is seeded by a stable SHA-256 digest of filename, starting row,
  segment length, seed and pattern. Results do not depend on model RNGs or the
  order recordings are processed. Record the NumPy version with diagnostics.

The endpoint constraint changes the generated 80%/60% masks relative to the
earlier first-anchor-only implementation, even with the same seed. Regenerate
shared masks for all models; do not mix artifacts from the two protocols.

Apply any **common segment selection first** (e.g. the existing 10% per-end
trim), then generate one mask for that entire segment, never per sliding window.
Do not use a mask from the untrimmed recording on a trimmed segment. The canonical
raw recording remains unchanged. Use the same chosen segment for future ODEs.

Fit normalization once on the complete, unmasked training segments, then freeze
it across patterns, retention levels and models. Do not fit on retained-only or
interpolated rows. The baseline adapter uses standard linear interpolation only,
on the original distance grid and before the original windows are formed. Each
missing value is weighted between its nearest retained left/right observations.
Retained values remain untouched; force values are never used in interpolation.
There is no alternative imputation, mask/time input channel, learned imputation
or change to baseline models.

**This is offline, non-causal preprocessing:** a right-hand sensor observation
may occur at or beyond the force target distance. The window indices remain unchanged,
but this pipeline must not be described as strictly causal prediction. A future
comparison with a causal continuous-coordinate model must explicitly disclose this
difference in available information; identical masks alone do not remove it.

```python
from src.sampling import observation_mask, retained_observations

segment = training[0]  # same complete segment for every compared model
mask = observation_mask(segment, 0.8, "bursty", seed=0, burst_length=8)
x, y, targets = make_windows(segment, statistics, 64, stride=16, mask=mask)
observations = retained_observations(segment, mask, statistics)
```

`x` is the interpolated, normalized two-channel baseline input. `observations` contains
**only** retained `distance, acceleration, uplift`, carrying
original row IDs. Only the two sensors are normalized with the same training
statistics; distance remains in meters. Intervals are computed from consecutive
retained distances when needed, not stored as an additional field or feature.
No distance or gap feature is added to the existing RNN-family inputs.
Force is never an observation input.

Complete distance and force remain in `segment.samples`; `y` and `targets` and
window warm-up exclusions are identical across mask conditions. A future ODE
must score those same rows and consume retained sensors and their actual
distances without filling. Its independent coordinate will be **distance**,
`dz(s)/ds = f(z(s), s)`, not synthetic time. No continuous-coordinate model or
distance-aware recurrent baseline is implemented here.

API correction: `distance_m` is now named `distance`, and `time_s` is removed
from recordings and target/prediction metadata. There is no compatibility alias
that silently recreates a clock. Clean baseline feature/target arrays are
unchanged. Distance-based and previous constant-speed time-based linear
interpolation are mathematically equivalent; small floating-point differences
on irregular inputs are checked and reported, not treated as a new method.

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
`predictions` retains file, split, source row, distance, force and prediction.
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

`plot_sampling(recording, mask, filled, output_dir, ...)` shows three aligned
panels: acceleration [m/s²], uplift [m], and complete contact-force truth [N].
Units are documented from the reference paper in `data/README.md`. The x-axis
uses **Distance [m]**, the original recorded coordinate used by the data and
interpolation interfaces. Input panels show the original signal, retained observations, lightly
shaded missing cells, and the output of `baseline_data.linear_interpolate`.
Only markers are thinned to at most 80 per input panel for legibility; no signal
rows or masks change. Force is plotted in full without masking, interpolation,
or missing-region shading. Use the same recording/start/points across conditions;
axis limits depend only on the complete displayed signals, not the masks.
The function does not create masks or interpolate itself. Seed-0 diagnostics, checks and their
reproduction script are kept in `results/sampling_linear_seed0/`:

```bash
python results/sampling_linear_seed0/generate.py
```

The script audits all 120 complete XLS recordings against the native cell values,
then checks the 24 CutFre20 recordings at all six pattern/retention
combinations, with the established 10% end trimming, without training or model
inference. It overwrites the six overview PDFs with the full trimmed example
recording and adds four `_zoom.pdf` views for random/bursty 80%/60%. All zooms
use one common interval around overlapping eight-row bursts in bursty 60%/80%,
nearest the midpoint and selected without looking at signal values or
interpolation errors. The legend labels
are `Original`, `Retained`, and `Interpolated`; only the two sensors are filled.

The same script writes lightweight sensor-distortion tables in that directory:

- `interpolation_by_recording.csv`: MAE/RMSE in physical units, **only at removed
  rows**, separately for each recording, condition and input channel.
- `interpolation_summary.csv`: pooled errors by condition/channel and split.
  `split=all` is a descriptive summary of all 24 recordings, not a new split.
  Pooling weights individual removed samples, not per-recording RMSEs.
- `interpolation_by_gap.csv`: the same errors grouped by consecutive missing
  row count, with removed-sample and gap counts. No new burst lengths are tested;
  the default bursty protocol has eight-row gaps plus an occasional remainder.

These measure interpolation distortion, **not contact-force prediction error**.
At 100% retention there are no removed rows: input identity is checked exactly,
and missing-only MAE/RMSE are undefined (empty CSV cells), not reported as zero.
The earlier `verification.json`, `interpolation_checks.json` and interpolation
error CSVs are preserved as historical evidence; the script verifies that new
interpolation errors agree to numerical precision instead of overwriting them.
The current `distance_checks.json` records checks and hashes. The pre-change
`baseline_input_reference.json` freezes clean-window hashes; all features,
standardized targets, target row IDs, force and distance values must match exactly.
This checks numeric compatibility independently of the coordinate-column rename.
`coordinate_compatibility.json` records measured interpolation round-off against
the pre-change adapter; these are sensor-input differences, not force-prediction errors.

Spatial reports (all distances in meters):

- `raw_distance_spacing.csv`: one row per complete XLS recording, with exact
  source alignment, strict monotonicity and mean/std/min/max/median Δs.
- `observation_distance_spacing.csv`: one row per trimmed recording/condition,
  describing Δs between consecutive retained observations and maximum missing-row count.
- `observation_distance_summary.csv`: pooled actual intervals by nominal speed,
  pattern and retention across all splits. Population std and median are computed
  from intervals, not averages of per-recording summaries. No boundary intervals
  join recordings; no artificial first-row interval is added. `max_m` is the
  maximum **retained-endpoint separation**, including ordinary steps at 100%.

The generator verifies all existing mask hashes, unchanged complete force and
distance, exact retained values, and clean-window identity. It reports any
unavailable trained-checkpoint recheck explicitly; synthetic predictions and
identical input hashes do not constitute a new saved-checkpoint evaluation.
`src.evaluate.evaluate` remains the clean-checkpoint utility; no new experimental
runner or irregular-model evaluation mode has been added in this step.

The cleanup checks and clean-checkpoint reproduction results are recorded in
[tests/BASELINE_VERIFICATION.md](tests/BASELINE_VERIFICATION.md). No models were
trained and no remote actions were performed. Result records are kept in Git.
Git does not back up ignored raw datasets, checkpoints or bulk predictions;
keep those separately if long-term recovery is needed.
