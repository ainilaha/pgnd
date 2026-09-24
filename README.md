# Physics-Guided Neural Dynamics for Pantograph–Catenary Contact Force Estimation

**PGND** is a research proof-of-concept for estimating contact force from panhead acceleration and displacement using physics-guided continuous-time latent dynamics. It implements a quadratic-potential version of the formulation in [main.tex](main.tex), alongside PyTorch RNN, LSTM, GRU, and CNN–GRU baselines. The first experiment is a shared finite-window, next-sample prediction comparison, not yet a validation of the draft's streaming or robustness claims.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Only PyTorch is used for learning; `torchdiffeq` supplies the ODE solver. `pandas`/`xlrd` read the original XLS data; `openpyxl` supports the older XLSX collection. Matplotlib generates headless PDF figures. No TensorFlow/Keras installation is needed.

## Data and common experimental protocol

Original files remain unchanged in `data/Data/` (120 XLS files) and `data/data_old/` (3 older XLSX files). They are local and ignored by Git. Their provenance and column layouts are described in [data/README.md](data/README.md); the previous-paper/code discrepancies are recorded in [legacy/README.md](legacy/README.md).

All five models use exactly the same `src/data.py` pipeline:

1. Read each four-column XLS recording with `header=None`; keep distance, acceleration, displacement, force and original zero-based row indices. No additional filtering is applied. Non-finite data are rejected rather than closing time gaps.
2. Retain the legacy 10% trim at **each** end. Reserve the final 15% of the remaining rows of **each training recording** for validation. Test recordings are held out entirely.
3. Fit feature-wise population mean/std and force mean/std on training rows only. Freeze those statistics for validation/test. Both inputs and force targets are standardized; metrics are inverse-transformed to newtons.
4. Construct windows separately inside each recording and partition: `x[i:i+16] -> force[i+16]`. No input or target row crosses a train/validation boundary. No force history is an input.
5. Sort filenames; shuffle only training windows, with a generator seeded independently of model initialization. Validation/test retain sample order.

These are **explicit changes from the earlier compatibility loader**, which discarded the first numeric row, concatenated recordings before windowing, returned unscaled values despite fitting a scaler, and split overlapping windows. The new scores are not directly comparable to the published/legacy scores. The original helper remains in `legacy/`; baseline architectures are unchanged. Validation shares recording identities with training but no rows; temporal correlation can remain, so it is not a whole-recording generalization estimate.

The initial experiment selects all three speeds (300/350/380 km/h), cases 1–6 for the training pool, cases 7–8 for test, and **20 Hz only**. It uses training target stride 16 to limit CPU cost; all 16 contiguous observations remain in each input. Validation/test stride is 1. Counts are 12,126 training, 33,983 validation, and 69,909 test windows. This is a single-seed, reduced-training-target proof-of-concept, not the full benchmark. `--train-stride 1` enables all eligible training targets, but requires retraining every model for a comparable experiment.

Distance increments divided by filename speed imply 0.001 s sampling, assuming distance in metres and constant speed in km/h. The loader checks that this inferred interval is uniform. Timestamp units and upstream filter causality are not independently established by the supplied files.

## PGND correspondence to the manuscript

Equation labels below refer to `main.tex`, avoiding unstable equation numbering.

| Manuscript component | Implementation in `src/model.py` |
| --- | --- |
| `eq:observation_encoder`, `eq:linear_interpolation` | 2→32→16 tanh MLP; linear interpolation of encoded observations |
| `eq:initial_state`, `eq:latent_state` | 2→32→32 tanh MLP from the first observation; `q,v` each have dimension 16; initialization length K0=0 |
| `eq:final_pgnd`, `eq:damping_constraint` | `q_dot=v`, `v_dot=-Dv-Kq+Bh+r`; learned `D=L_D L_Dᵀ`; bias-free 16→16 drive B |
| `eq:quadratic_potential`, `eq:linear_restoring` | `V(q)=qᵀKq/2`, `K=L_K L_Kᵀ`; exact analytic `grad V=Kq`, differentiable with respect to q and K |
| `eq:neural_residual` | `[q,v,h,t]` (49 inputs) →32→16 tanh MLP; final layer initialized to zero |
| `eq:force_decoder` | `[q,v]`→32→1 tanh MLP, linear standardized-force output, then fixed inverse scaling for reporting |
| `eq:ode_solver`, `eq:total_loss` | `torchdiffeq.odeint` with direct autograd; standardized force MSE plus residual penalty |

These dimensions and MLP widths are initial implementation choices, not recovered paper hyperparameters. The model has 5,761 trainable parameters. Both factor matrices start at `0.1 I`, giving `D=K=0.01 I`; other layers use PyTorch's default initialization except the zero residual output. A symmetric K is needed for the manuscript's `grad V=Kq` identity; the PSD factorization is an explicitly chosen, bounded-below quadratic special case, not a learned nonlinear potential or an identified physical pantograph stiffness.

ODE time is relative to the window start and measured in milliseconds (`time_unit=0.001` seconds). Consequently latent v is `dq/d(milliseconds)`, and learned coefficients are in these latent/time coordinates, not SI mechanical parameters. RK4 uses step 1 in those units (one nominal sample interval). `dopri5` is also available with rtol=1e-5, atol=1e-7; tolerances do not control fixed-step RK4 accuracy. No custom integrator or adjoint approximation is used.

**Information budget and loss:** each model resets on the same 16 historical observations and predicts the following force. PGND linearly interpolates only the observed history and holds the last encoding constant for the final, unobserved interval. It never uses the target-time observation. This differs from the draft's contemporaneous reconstruction and long continuous trajectory: only the last decoded force is supervised per window, while the residual penalty is the mean, over windows and 16 evolved sample times, of the **sum** of squared residual components. The whole latent/decoded trajectory is available via `return_details=True`, but it is not trajectory-supervised in this experiment.

The loss is `MSE(standardized force) + 0.001 * mean(||r||²)`. The residual weight is a stated initial assumption, with no test-set tuning. Because force is standardized, this weight is not interchangeable with a weight multiplying raw-N² MSE. Unforced dynamics with B and r removed satisfy `dE/dt=-vᵀDv <= 0`; driving/residual terms can inject energy, and PSD damping alone does not guarantee stability or physically calibrated latent states in the trained model. No extra physics penalties were invented.

## Training

Run the following commands on the **remote server**, from the repository root after installation. Copy the ignored `data/Data/` files to that server first; cloning Git does not include them. The commands preserve the first comparison's experimental settings and write a **new** experiment directory with one subdirectory per model. Set `DEVICE=cpu` if CUDA is unavailable; changing device may change numerical results. No local training/evaluation was run for this logging/plotting update.

```bash
set -e
RUN=results/20hz_seed0_run1
DEVICE=cuda

# 1. Train each baseline.
for model in lstm gru rnn cnn_gru; do
  python -m src.train --model "$model" \
    --train-pattern 'V(300|350|380)_Case[1-6]_CutFre20\.xls$' \
    --test-pattern 'V(300|350|380)_Case[7-8]_CutFre20\.xls$' \
    --epochs 20 --batch-size 128 --train-stride 16 \
    --sequence-length 16 --cut-percent 0.1 --validation-fraction 0.15 \
    --learning-rate 0.001 --seed 0 --device "$DEVICE" --threads 1 \
    --output "$RUN/$model/$model.pt"
done

# 2. Train PGND.
python -m src.train --model pgnd \
  --train-pattern 'V(300|350|380)_Case[1-6]_CutFre20\.xls$' \
  --test-pattern 'V(300|350|380)_Case[7-8]_CutFre20\.xls$' \
  --epochs 20 --batch-size 128 --train-stride 16 \
  --sequence-length 16 --cut-percent 0.1 --validation-fraction 0.15 \
  --learning-rate 0.001 --seed 0 --device "$DEVICE" --threads 1 \
  --latent-dim 16 --encoding-dim 16 --hidden-dim 32 --residual-weight 0.001 \
  --time-unit 0.001 --ode-method rk4 --ode-step 1 --rtol 1e-5 --atol 1e-7 \
  --output "$RUN/pgnd/pgnd.pt"
```

All models use Adam (betas 0.9/0.999, epsilon 1e-7), MSE and final-epoch weights. There is no early stopping, scheduler, clipping, weight decay, dropout or test-based selection. Defaults remain batch size 32 and training stride 1; the commands deliberately override them. Use stride 1 for full training targets only if changed consistently for all models in a new run.

Before optimization, `<model>.run.json` records the run settings: ordered files/hashes, preprocessing/normalization, counts, seed, optimizer, device, and PGND architecture/solver settings. Each completed epoch is appended to `<model>.history.csv`, so an interrupted run keeps its completed history. Columns are `epoch`, `train_mse_scaled`, `val_mse_scaled`, `residual_loss`, `train_total_loss`, and `seconds`. MSE is in standardized-force units, not N²; PGND's total includes its residual penalty, whereas validation MSE does not. Training MSE averages minibatches during updates; validation MSE uses the end-of-epoch weights. The final `.pt` also contains the settings and full history. Interrupted histories are not resume checkpoints; choose a new output name to rerun. Existing checkpoint/history/settings files are never overwritten.

The baselines retain two 32-unit recurrent layers and a linear output; CNN–GRU adds a width-1 64-channel ReLU convolution. Trainable parameter counts are RNN 3,233; LSTM 12,833; GRU 9,825; CNN–GRU 15,969. The preserved Keras initializer families, LSTM forget bias and reset-after GRU are documented in the audit. Architecture equivalence does not imply bitwise Keras equivalence. PGND has fewer parameters and a different computation budget; equal epochs are not equal runtime or architecture-specific hyperparameter optimization.

## Evaluation

```bash
# 3. Evaluate all models with the common pipeline (same shell as above).
python -m src.evaluate \
  "$RUN/lstm/lstm.pt" "$RUN/gru/gru.pt" "$RUN/rnn/rnn.pt" \
  "$RUN/cnn_gru/cnn_gru.pt" "$RUN/pgnd/pgnd.pt" \
  --device "$DEVICE" --threads 1 --output-dir "$RUN/evaluation"
```

The evaluator checks file hashes and matching protocol, splits, statistics, training budget and seed before comparing. It pools exactly the same test targets for each model and reports MAE (N), RMSE (N), the legacy MSE (N²), and R² (undefined for constant targets). The new evaluation directory contains:

- `comparison.csv`: all models, including run name, model, seed, sample/parameter counts and metrics.
- `<model>.metrics.csv`: final metrics for that model/run.
- `<model>.predictions.csv`: filename, original zero-based source row, distance, ground-truth `force_N`, and `prediction_N` in newtons.
- `<model>.history.csv`: the actual training history embedded in that checkpoint, bundled for plotting without loading weights.

The run name is the checkpoint filename stem. Stems must be distinct in a comparison, even when checkpoints live in separate directories. Evaluation refuses an existing output directory; use a new name to reevaluate. The existing compatibility checks deliberately compare one matched seed/budget at a time, not aggregate multiple seeds into error bars. Outputs remain ignored by Git.

## Visualization

```bash
# 4. Generate all comparison figures from saved CSVs only.
python -m src.plot "$RUN/evaluation" \
  --recording V300_Case7_CutFre20.xls --start 0 --points 1000 \
  --output-dir "$RUN/figures"
```

`src/plot.py` is separate from the model, training, and evaluation code. It imports no PyTorch model or raw-data loader and uses Matplotlib's headless backend. It saves vector PDF versions only of:

- `training_loss_curves`: all models' training **MSE** on one axis, distinguished by colors and markers.
- `validation_loss_curves`: all models' validation **MSE** on a separate figure, using the same model colors and markers. Training and validation are not mixed; PGND's regularized total is not mislabeled as MSE.
- `metric_comparison`: separate panels for MAE, RMSE, MSE and R², with units and no unsupported uncertainty bars. Negative R² is retained; undefined R² is labeled.
- `force_predictions`: one black ground-truth curve and all model predictions on one axis, distinguished by colors and sparse markers, for exactly the same consecutive targets in one recording. Marker spacing does not subsample the plotted lines.

`--start` is an offset into that recording's eligible test targets, not an original XLS row number. `--points` is capped at the recording's remaining targets. Original source row numbers appear in the figure title; distance is the horizontal axis. No smoothing, resampling, best-segment search, or joining of unrelated recordings is performed. If `--recording` is omitted, the first evaluated recording is used. Prediction files must have identical ordered target provenance and ground truth. Use a new figure directory for another selection; existing figures are not overwritten.

New layout (repeat the model subdirectory for all five models):

```text
results/20hz_seed0_run1/
├── lstm/
│   ├── lstm.run.json
│   ├── lstm.history.csv
│   └── lstm.pt
├── gru/ ...
├── rnn/ ...
├── cnn_gru/ ...
├── pgnd/ ...
├── evaluation/
│   ├── comparison.csv
│   ├── lstm.metrics.csv
│   ├── lstm.history.csv
│   ├── lstm.predictions.csv
│   └── ...                  # equivalent files for each model
└── figures/
    ├── training_loss_curves.pdf
    ├── validation_loss_curves.pdf
    ├── metric_comparison.pdf
    └── force_predictions.pdf
```

Previously saved checkpoints from the shared protocol remain evaluable. Older evaluation folders lack the bundled histories/run column: reevaluate those checkpoints into a new directory on the server before using this plot command. This update does not change preprocessing, model definitions, optimization, metric formulas, or prior result artifacts.

This update was checked with Python/shell syntax validation and synthetic CSV fixtures only: per-epoch history writing, PNG/PDF exports, single-model panels, negative/undefined R², recording selection, matching-target checks and overwrite protection. Synthetic figures were inspected outside the repository. No training or real-data evaluation was run for this update.

## Previously measured first comparison

These are previously obtained final-epoch results, not new runs from the logging/visualization update and not the prior paper's results. Local artifacts are in `results/poc_20hz_seed0/`; the remote commands use a separate directory to avoid overwriting them. The prior run used CPU; the remote example selects CUDA.

| Model | MAE (N) | RMSE (N) | MSE (N²) | R² |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 12.779977 | 16.298784 | 265.650361 | 0.808066 |
| GRU | 12.725487 | 16.088461 | 258.838564 | 0.812987 |
| RNN | 13.374754 | 17.071518 | 291.436724 | 0.789435 |
| CNN–GRU | 11.280560 | 14.662261 | 214.981911 | 0.844674 |
| PGND | 9.790036 | 12.932532 | 167.250380 | 0.879161 |

PGND has the lowest errors in this run. This is encouraging evidence for the implemented finite-window quadratic variant, **not** evidence of statistical superiority, unseen-speed/bandwidth robustness, streaming performance, or a causal benefit from the physics terms without ablations. No hyperparameters or epoch choices were changed after inspecting test results.

Executed on CPU with one PyTorch thread, Python 3.12.14, PyTorch 2.14.0, torchdiffeq 0.2.5, NumPy 2.3.5 and pandas 2.2.3. The local interpreter was `/tmp/pgnd-audit-C5c0lj/venv/bin/python`; it is not a repository dependency. Dependency versions are reported, not pinned; bitwise agreement across environments is not promised.

## Verification and remaining scientific questions

Lightweight checks verified window/target alignment, row-disjoint train/validation partitions, training-only statistics, common model input/output shapes, finite ODE trajectories and gradients, interpolation/last-input holding, the potential gradient, PSD damping/stiffness, and numerical unforced energy decrease. A double-precision ODE gradient agreed with a central finite difference (-0.00459402674095 versus -0.00459402674235). Sixty optimization steps on 64 real training windows reduced total PGND loss from 0.378305 to 0.072909. These establish implementation sanity, not generalization or physical identification.

The trained model also has finite, nonzero gradients in every parameter tensor. On 256 evenly spaced validation windows, changing RK4 from a 1 ms step to 0.5 ms changed predictions by 0.045455 N RMS (maximum 0.121011 N); 0.5→0.25 ms changed them by 0.001948 N RMS. Dopri5 (rtol=1e-6, atol=1e-8) differed from 0.25 ms RK4 by 0.000552 N RMS. This small numerical-resolution check is recorded in `results/poc_20hz_seed0/solver_check.csv`; it did not change the trained checkpoint or reported 1 ms test scores. Adaptive-solver forward/backward and nonuniform-time smoke checks passed as well. All 123 original dataset hashes were reverified unchanged, and all five checkpoint/evaluation round trips succeeded.

Before paper-level claims, resolve/validate physical timestamps and filter causality, streaming versus finite-window operation, contemporaneous reconstruction versus next-step prediction, nonlinear versus quadratic potential, latent dimensions and initialization length, loss/solver choices, and the intended speed/bandwidth protocols. Run multiple seeds, full training targets, broader numerical convergence checks, and physics/residual ablations. Temporal interpolation on an available history is compatible with this next-step task; it does not establish zero-delay online reconstruction under unknown upstream filtering.

The draft's duplicate `eq:residual_loss` label and unrelated keywords/CIFAR-10/edge-pruning statements remain flagged, not edited. Its abstract promises improvements and robustness before supplying evidence. The previous paper also disagrees with its code about counts and several splits/model labels; consult the retained audit before comparing against its numbers. No baseline dominance, robustness or streaming claim should follow from one reduced single-bandwidth run.

## Repository layout

```text
pgnd/
├── README.md
├── requirements.txt
├── .gitignore
├── main.tex
├── data/
│   ├── README.md
│   ├── Data/                 # 120 unchanged local XLS files
│   └── data_old/             # 3 unchanged local XLSX files
├── papers/                   # previous paper and Neural ODE reference PDFs
├── src/
│   ├── data.py
│   ├── model.py
│   ├── baselines.py
│   ├── train.py
│   ├── evaluate.py
│   └── plot.py
├── results/
│   ├── .gitkeep
│   └── poc_20hz_seed0/       # local checkpoints, histories and evaluation CSVs
└── legacy/                  # scientific references and migration audit
```

No datasets, legacy scientific references, baseline definitions or manuscript content were removed or rewritten during PGND implementation.
