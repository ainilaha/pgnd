# Source boundaries

Apply the root `AGENTS.md`. Use plain functions and the existing `nn.Module`s.

- `data.py`: complete recordings, explicit identity/whole-case membership,
  aligned time/acceleration/uplift/force, train-only normalization. No windows,
  trimming, interpolation, missing-data simulation or model-specific logic.
- `baseline_data.py`: explicit end trimming and regular, recording-local windows.
  Preserve `X[k-L:k] -> force[k]`: no target-time sensors or force-history inputs.
- `baselines.py`: unchanged CNN–GRU, GRU, LSTM and RNN architectures/initializers.
  Preserve checkpoint tensor names/shapes. No dataset or experiment dependencies.
- `metrics.py`: scalar MAE, RMSE, MSE and R². Undefined R² is NaN, not zero.
- `evaluate.py`: ordered clean baseline inference and metrics, no fitting scalers
  on held-out data, training, architecture registry or experimental branches.
- `plot.py`: supplied tables only, no data/model execution. PDF only; separate
  training/validation curves. Force comparisons must use identical target rows.

Keep raw rows immutable in practice: transformations return copies. Split before
windowing, never cross recordings, retain original source-row IDs. Do not promote
the historical L=64 reference setting into a canonical-data default.

Checks from repository root:

```bash
python -m unittest discover -s tests -v
git diff --check
```

Unit tests use tiny synthetic arrays, no raw datasets, optimizer steps, network
or training. Checkpoint reproduction is a separate, explicitly reported check.
