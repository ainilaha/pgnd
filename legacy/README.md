# Legacy inspection and migration audit

**Historical scope:** this audit records the initial migration and its compatibility loader. The active implementation has since added PGND and a revised shared protocol (correct header handling, recording-local windows, row-disjoint validation, training-only standardization). Statements below about the "active"/"new" compatibility code or unimplemented PGND describe that earlier stage, not today's `src/`. See the [root README](../README.md) for current choices, commands and experiment results. Legacy sources and scientific findings below remain preserved.

The import was actually named `old_project/` (not `old_code/`). All 355 files were inventoried, all notebook source variants were inspected, all 123 spreadsheets were parsed, and the previous paper and current manuscript were read before repository changes. The folder is now `legacy/`. Original data bytes and the root `main.tex` were preserved. The identical `papers/main.tex` copy was removed.

## Sources and classification

The previous paper is `papers/Real-time prediction of high-speed rail pantograph-catenary contact force using deep learning framework.pdf` (Ale et al., published 13 January 2026, DOI 10.1080/00423114.2025.2606345). Page references below are the printed article pages, excluding its publisher cover. The new method specification is the repository-root `main.tex`.

| Area | Legacy source | Migration decision |
| --- | --- | --- |
| Loading, trim, missing values, concatenation | `data_util.py`; earlier `read_data.ipynb` | Adapted into `src/data.py`; original references retained |
| Scaling and windows | `data_util.py`, `rnn2_old.ipynb`, `rnn_old.ipynb` | Main unscaled, next-sample logic reused; incompatible older variants retained only as references |
| Case, speed, bandwidth splits | `rnn_fre20-20.ipynb`, `rnn_fre200-200.ipynb`, `rnn_vilocity.ipynb`, `rnn.ipynb`, `rnn_fre20.ipynb` | Explicit file regexes in the new CLI; no automatic strategy labels |
| Main RNN/LSTM/GRU/CNN–GRU | Frequency notebooks and saved `.keras` configurations | Reimplemented in PyTorch in `src/baselines.py` |
| Other models | `rnn.ipynb`, `rnn_vilocity.ipynb`, `rnn2_old.ipynb`, `rnn_old.ipynb`, `data_explore_old.ipynb` | Attention-GRU, bidirectional recurrent variants, dense models, and SVR preserved in their notebooks |
| Training | Notebook `compile`/`fit` cells, saved model optimizer configurations | Shared Adam/MSE loop in `src/train.py` |
| Evaluation | Notebook `plot_error` functions, `sklearn.metrics.mean_squared_error` | Same scalar-force MSE in `src/evaluate.py`, independent of plotting |
| Exploration and plotting | `Data_Explore.ipynb`, `load_model.ipynb`, frequency/speed notebooks, two `.pptx` files | Sources retained; generated standalone figures removed |
| Signal filtering | Cutoff-specific spreadsheets; paper §4 | No corresponding scientific filter implementation was found; no new filter added |

The `gaussian_filter1d` calls in `image5.ipynb` were for unrelated mobile-edge/RL plots, not pantograph signal preprocessing. That notebook and `image.ipynb` were removed.

## What the shared data code preserves

The active loader adapts `get_files`, `read_data`, and `create_sequences` from `data_util.py`, without importing TensorFlow or scikit-learn. It reads the original spreadsheets, assigns the four legacy column names, trims `int(N * cut_percent)` samples at **each** end of each recording, concatenates recordings, drops NaNs, and resets row indices. The old helper default was 5%; the main experiment notebooks explicitly used 10%, which remains the training default.

Features are acceleration and displacement, in that order. Distance is retained for target alignment, not used as an input feature. Windows are `X[i:i+L]` and predict `force[i+L]`, with `distance[i+L]` as the index; L defaults to 16. No target scaling or extra signal processing is applied. The three older five-column `.xlsx` files are not silently combined with the four-column dataset.

Known limitations retained deliberately:

1. All 120 `.xls` files are headerless. Original `pd.read_excel` uses `header=0`, consuming their first numeric row. The new reader explicitly retains `header=0`; fixing this later will alter trimming and alignment.
2. The main `create_sequences` fits a MinMaxScaler but uses the **unscaled** features in its windows. The unused fit was removed, preserving returned values. There are no normalization statistics in the active protocol. If scaling is introduced later, fit only on training recordings and freeze it for validation/test.
3. Recordings are concatenated before window construction. A length-16 window can therefore cross unrelated cases/speeds or a distance reset. For K sufficiently long recordings, `(K-1)*16` windows involve a recording boundary, counting the target. The old helper even contains a FIXME about this. This behavior is warned about rather than silently corrected.
4. Keras `validation_split=0.15` holds out the final 15% of windows before shuffling training batches. This is reproduced with `int(N * 0.85)` and no gap. The last training and first validation windows share 15 of 16 observations; the held-out tail also depends strongly on file order. It is not an independent recording-level validation set.
5. Dropping NaNs before windowing would close time gaps without accounting for them. No missing/non-finite values were found in the supplied parsed datasets, but this matters for future measurements.

Explicit changes from the legacy helper: filenames are sorted instead of using unspecified `os.listdir` order; input lengths and four-column layout are validated; empty selections fail clearly; the `cut_precent` argument is renamed `cut_percent`; dead scaler computation is removed; file paths resolve to repository-level `data/Data/`. **Sorting changes concatenation boundaries and potentially training/validation membership.** This is a reproducible new order, not a reconstruction of the unknown order used for each historical run. `read_data` still accepts an explicitly ordered list, and checkpoints record the exact selected order and split index.

## Baseline equivalence

The main notebooks, paper §5, and eight saved model configurations agree on these architectures:

| Model | Architecture | Original / new trainable parameters |
| --- | --- | --- |
| RNN | 2 inputs → tanh RNN(32) → tanh RNN(32) → linear(1) | 3,233 / 3,233 |
| LSTM | 2 inputs → LSTM(32) → LSTM(32) → linear(1) | 12,833 / 12,833 |
| GRU | 2 inputs → GRU(32) → GRU(32) → linear(1) | 9,825 / 9,825 |
| CNN–GRU | Conv1D(2→64, kernel=1, stride=1, ReLU) → GRU(32) → GRU(32) → linear(1) | 15,969 / 15,969 |

All are unidirectional, use the last recurrent output, start each window from zero state, and have no dropout or recurrent dropout. GRU uses the reset-after formulation with two bias vectors, matching saved Keras `reset_after=True`. LSTM uses a unit forget bias. PyTorch RNN/LSTM normally have two biases per layer; the extra recurrent bias is fixed at zero to match the single trainable Keras bias. Total stored parameters are consequently 3,297 and 13,089 for RNN/LSTM, but trainable counts match the paper. Input/output weights use Xavier uniform initialization and recurrent weights orthogonal initialization, matching the original initializer families.

The new loop retains MSE, Adam learning rate 0.001, betas (0.9, 0.999), epsilon 1e-7, 20 epochs, batch size 32, shuffled training windows, and final-epoch weights. Epoch/window/validation settings are explicit in the notebooks; batch size 32 and shuffling are inferred from their omitted Keras `fit` arguments (the saved 6,067 steps per epoch agree with 194,140 optimization windows at batch size 32). Adam values are confirmed by the saved model configs. No checkpoint-selection rule, early stopping, learning-rate schedule, clipping, or regularization is introduced.

This is architectural equivalence, not bitwise Keras reproduction: framework random draws, gate storage order (Keras GRU z/r/h versus PyTorch r/z/n), kernels, floating-point reductions, and Adam epsilon/bias-correction details can differ. Seed 0 and independently seeded minibatch shuffling are new reproducibility choices. Checkpoints store the actual settings and losses. No TensorFlow/Keras dependency or old-weight conversion is included. Other exploratory model variants remain in legacy; they are not falsely presented as one of the paper's four validated baselines.

## Paper/code discrepancies and questionable results

- **Dataset size/bandwidth:** paper §4 describes 96 groups, using 20/50/150/200 Hz. The import has 120 groups including 100 Hz, plus three older `.xlsx` files. All are preserved; 100 Hz is not silently added to a paper experiment.
- **Sample counts:** the paper reports approximately 124,200 training and 41,400 test samples. The supplied files with the main 20 Hz notebook selections and 10% trim produce 228,400 training-pool windows (194,140 optimization / 34,260 validation) and 69,983 test windows. The exact data/subsampling behind the published counts cannot be established from the import.
- **Strategy 2:** Figure 8(b) and §5.2 train on cases 1–6 at 300/350 km/h, testing the remainder at 200 Hz. `rnn_fre200-200.ipynb` instead trains cases 1–6 at **all three speeds**. `rnn_vilocity.ipynb` trains 300/350 km/h but **all eight cases**. Neither is the stated case-and-speed holdout.
- **Strategy 4 and paper wording:** Figure 8(d) shows 20 Hz training on cases 1–6, and testing other cutoffs across all cases. The prose also says the full 20 Hz dataset is used, while elsewhere saying the first six cases. References to “three additional cases” conflict with eight total cases and a six-case training set. Figure 8 and conflicting passages remain references rather than being collapsed into a single assumed protocol.
- **Cross-bandwidth dependence:** several notebooks test all cases at another cutoff, including the same speed/case combinations used for training. That is bandwidth transfer on related trajectories, not an independent unseen-trajectory test. The new CLI warns when these identities overlap; exact train/test file overlap is rejected.
- **Wrong cutoff:** both `rnn_fre200-200.ipynb` and its `Copy1` variant use `CutFre20` for their V380 test while naming the result `trained_Fre200_test_Fre200_V380`.
- **Wrong loaded model:** `load_model.ipynb` and frequency notebook reload cells assign the RNN checkpoint to `gru_model`. Comparisons from these cells may label an RNN result as GRU. The new evaluator constructs the architecture recorded in its own checkpoint.
- **Mislabeled histories/plots:** aggregate `history/val_loss.csv` mixes GRU/CNN–GRU training losses with RNN validation loss and labels RNN values as LSTM; aggregate `loss.csv` also copies RNN into LSTM. Some loss plots use CNN–GRU validation loss alongside other models' training losses. `rnn_vilocity.ipynb` explicitly builds a supposed LSTM curve as `data['GRU'][2:] - 20`. These curves must not be treated as measured LSTM validation performance. Original files are retained as evidence; new histories are collected directly from each run.
- **Model labels:** some 20 Hz error tables assign `['RNN','CNN-GRU','GRU','LSTM']` to values returned in RNN/GRU/CNN–GRU/LSTM order, swapping GRU and CNN–GRU labels.
- **Old exploratory leakage:** `rnn2_old.ipynb` actually scales the concatenated data before a random 80/20 window split. Test information affects the scaler, and highly overlapping windows appear on both sides. Its target is `Raw_Force`, unlike the main four-column cutoff-specific target. `rnn_old.ipynb` uses a five-sample window with an in-window/end-of-window target, another distinct protocol. These are not migrated as the common data path.
- **Stale execution state:** several notebooks call older three-argument sequence helpers or unpack two results, while `data_util.py` now requires four arguments and returns three arrays. The saved helper snapshot returns force rather than distance as its third array. `read_data.ipynb` contains an obsolete concatenation trim using the accumulated frame length, filenames missing underscores, and an invalid regex. Saved notebook outputs do not prove a clean rerun succeeds.
- **Filtering:** the paper describes low-pass filtering, but filter design/order/phase/causality and generating code are absent. For V300/Case1, both input signals **and force** differ between the supplied 20/200 Hz files. The CNN kernel in the code is width 1, so it mixes channels per time step without temporal smoothing/downsampling; broad paper claims about noise reduction or reducing parameter cost should not be treated as a different implemented convolution.
- **Internal paper ambiguities:** the validation-curve discussion refers to both Strategy 1 and Strategy 2; Strategy 1 includes 380 km/h in training yet discusses higher-speed data as unseen. MSE has units N², although one results passage uses N. No new metric interpretation is inferred from these passages.

These findings identify provenance and reproducibility problems; they do not establish which saved notebook state or checkpoint generated any particular published figure. Historical results need independent recomputation under a confirmed protocol.

## Retention and cleanup

Retained for scientific reference:

- `data_util.py`, the main frequency/speed notebooks, `rnn_fre200-200-Copy1.ipynb` (different stored run/output state), `load_model.ipynb`, and `Data_Explore.ipynb`.
- `rnn_old.ipynb`, `rnn2_old.ipynb`, `data_explore_old.ipynb`, `read_data.ipynb`: different targets, architectures, or preprocessing history that may matter for reproduction.
- `data_util_snapshot.py`, `data_explore_old_snapshot.ipynb`, `rnn_fre20-20_snapshot.ipynb`, `rnn_fre200-200_snapshot.ipynb`: nonredundant historical source extracted from autosaves before removing cache directories.
- Both sets of four `.keras` checkpoints and twelve history CSVs. These are distinct historical artifacts, not duplicate bytes; their exact experiment provenance is uncertain. They remain local and ignored by Git. Faulty aggregate histories are retained as evidence, not accepted metrics.
- `diagram plot.pptx` and `plots.pptx`: editable architecture/figure authoring sources.

Removed from the repository (195 files, including two files outside the import):

- `images/`, `images 2/`, `images 3/`, `images 2.zip`, root `output.png`, `output2.png`, and `val_loss_plot.pdf`: generated graphics and their copies/autosaves. The ZIP includes one different `Strategy4.pdf`, but it is also a generated figure; it is covered by the recovery archive.
- `.ipynb_checkpoints/`, `.virtual_documents/`, `__pycache__/`, the history checkpoint directory, and `.DS_Store` files: caches/autosaves, after retaining meaningful source snapshots. Virtual-document differences were comments or plotting presentation, not additional scientific logic.
- `rnn_fre20-speed.ipynb`: byte-identical to `rnn_fre20-20.ipynb`.
- `image.ipynb` and `image5.ipynb`: unrelated mobile-edge/RL plotting code; `plots_rev.ipynb`: empty.
- Duplicate `papers/main.tex`: byte-identical to root `main.tex`, which was left unchanged; repository-root `.DS_Store`: OS metadata.

Cleanup is recoverable during this local session: the full original import plus duplicate manuscript is in `/tmp/pgnd-audit-C5c0lj/legacy-before-cleanup.tar.gz`; removed files and their exact list are in `/tmp/pgnd-audit-C5c0lj/removed/`. This temporary location is not a durable backup or a repository dependency.

Only data-path references were mechanically edited in retained legacy source. Run reference notebooks from `legacy/` if inspecting them; they still require their original optional libraries and contain the known stale calls. No attempt was made to repair their experimental logic or regenerate their outputs. The active PyTorch code does not import anything from this directory.

## Choices to resolve before implementing/training PGND

The draft specifies observations `[acceleration, uplift]`, latent state `z=(q,v)`, `dq/dt=v`, and `dv/dt=-(L_D L_D^T)v - grad V(q) + B h(t) + r(q,v,h,t)`. It decodes contact force from the integrated state and trains with force MSE plus a residual-magnitude penalty. `src/model.py` records this structure but intentionally does not invent its missing implementation details.

Resolve the latent/encoding dimensions; encoder, potential, residual and decoder networks; potential constraints; initialization interval K0; residual weight; ODE solver, tolerances and differentiation method; batching and state resets at recording boundaries; and physical-time coordinates. Distance increments and filename speeds imply 1 ms sampling under constant-speed/unit assumptions, but the files provide no explicit timestamps or simulation/filter implementation to confirm this.

Also resolve observation availability: the old baseline estimates F[k] using x[k-16:k], excluding x[k]. The draft's linear interpolation over [t[k],t[k+1]] uses x[k+1], and its force decoder reconstructs the state at that endpoint. Whether contemporaneous observations are allowed changes the prediction task and latency; compare models under the same causal information budget and score exactly the same target samples. Streaming PGND state also gives a potentially longer history than reset-per-window baselines.

Choose a common recording-level split/validation policy, treatment of cross-bandwidth related trajectories, header handling, window boundaries, trimming, and whether to introduce train-only normalization. These choices are deliberately left unresolved rather than silently substituted for the historical pipeline. PGND's trajectory/residual loss will require an extension of the current scalar-window training interface once those decisions are made.

The current draft also contains unrelated inherited IEEE keywords, CIFAR-10 data availability, an `edge_pruning` code-availability URL, and a repeated `eq:residual_loss` label. These were flagged, not edited. The full bibliography/template dependencies are not present, so this migration makes no manuscript-build claim.

## Migration verification

All 123 dataset hashes, eight historical checkpoint hashes, twelve retained CSV hashes, and the manuscript hash were verified unchanged. On two original recordings, the new loader and windows matched the original helper's numerical output exactly for the same file order, including next-sample alignment and its known cross-file windows. The validation-tail boundary and overlapping inputs were also checked.

All four PyTorch models passed published trainable-parameter counts, forward comparisons against an independent NumPy implementation of the Keras-style cell equations, finite-gradient checks, and short optimization checks. A CPU LSTM command-line training/checkpoint/evaluation round trip passed on a small slice of the original data. Syntax, notebook JSON, Git ignore rules, and whitespace checks passed. These were smoke tests, not full training runs or reproduced paper results; no model accuracy claim is made. Temporary tests and outputs were kept outside the repository, and no TensorFlow/Keras was installed.
