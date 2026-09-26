# Source boundaries

Apply the root `AGENTS.md`. Use plain functions and the existing `nn.Module`s.

- `data.py`: complete recordings, explicit identity/whole-case membership,
  aligned time/acceleration/uplift/force, train-only normalization. No windows,
  trimming, interpolation, missing-data simulation or model-specific logic.
- `sampling.py`: the sole deterministic mask generator for every model. Joint
  sensor masks, first/last observations retained, complete force truth unchanged.
  Retained-only sensor/time tables for future ODEs, with no filling or force input.
- `baseline_data.py`: explicit end trimming, linear interpolation at original
  times with the shared mask, and regular recording-local windows. No alternative
  imputation or mask/time channels. Preserve `X[k-L:k] -> force[k]` row alignment.
  Interpolation uses right-hand observations, possibly at/after the target time:
  this is offline preprocessing, not a causal irregular-input pipeline.
- `baselines.py`: unchanged CNN–GRU, GRU, LSTM and RNN architectures/initializers.
  Preserve checkpoint tensor names/shapes. No dataset or experiment dependencies.
- `metrics.py`: scalar MAE, RMSE, MSE and R²; removed-only sensor interpolation
  errors. Undefined metrics (constant-target R², no removed rows) are NaN.
- `evaluate.py`: ordered clean baseline inference and metrics, no fitting scalers
  on held-out data, training, architecture registry or experimental branches.
- `plot.py`: supplied tables only, no data/model execution. PDF only; separate
  training/validation curves. Force comparisons must use identical target rows.

Keep raw rows immutable in practice: transformations return copies. Split before
windowing, never cross recordings, retain original source-row IDs. Do not promote
the historical L=64 reference setting into a canonical-data default.
Choose the same recording segment for all models before masking. Fit normalization
on complete unmasked training segments once; never refit by mask or model. Keep
complete force targets, scoring rows and original times identical across conditions.

Checks from repository root:

```bash
python -m unittest discover -s tests -v
git diff --check
```

Unit tests use tiny synthetic arrays, no raw datasets, optimizer steps, network
or training. Checkpoint reproduction is a separate, explicitly reported check.
