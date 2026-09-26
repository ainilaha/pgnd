# Cleanup verification — 2026-09-26

Reference source: Git commit `bf4c1857eb896f904dd71e8d25050b80bc9af211`
(`pgnd backup`). This is a verification record, not a new model experiment.
No training, optimizer steps, tuning, remote synchronization or remote runs.

## Infrastructure checks

- All 120 headerless XLS recordings were read: **1,865,145 rows**. Original
  distance/acceleration/uplift/force values and source-row IDs matched exactly.
  Derived times were strictly increasing. No rows were trimmed in the raw loader.
- All **123 raw XLS/XLSX files** were hashed before/after verification; bytes
  were unchanged. The three older XLSX files are preserved, not active inputs.
- Whole-case split: cases 1–4 train, 5–6 validation, 7–8 test, all three speeds,
  CutFre20. The same case assignment applies to other cutoff variants.
- For the established reference adapter (10% trim at each end, L=64, train
  stride 16, evaluation stride 1), **every input tensor, standardized target and
  original target-row provenance matched the old implementation exactly**.
  Counts: 9,401 / 76,989 / 69,621 train/validation/test windows.
- Training-only mean/std matched the checkpoint exactly: input mean
  `[0.0013599267869852706, 0.07942407069803016]`, input std
  `[5.327149721546923, 0.019491810393643165]`, force mean
  `183.58088002838994`, force std `39.71405445306965`.
- Executable syntax trees of **all four baseline definitions and initializers**
  were identical to the pre-cleanup file. Only its introductory reference
  docstring changed. All original state dictionaries loaded strictly.
- The old and new CNN–GRU pipelines, using the same saved weights on the same
  CPU, gave **exactly identical predictions on all 76,989 validation targets**,
  including the change from pooled to recording-by-recording evaluation batches.

## Saved-checkpoint reproduction

All checks used the original complete clean sensor inputs and original force
truth, seed-0 best-validation weights, saved scaling, trim and window length.
Filename/target-row order matched saved predictions exactly; physical truth and
distance agreed to CSV round-trip precision (absolute tolerance 1e-10).

| Model / split | L | Targets | Selected epoch | Saved RMSE (N) | Current CPU RMSE (N) | Max prediction difference (N) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN–GRU / validation | 64 | 76,989 | 81 | 12.799996644 | 12.800010126 | 0.027336 |
| CNN–GRU / test | 64 | 69,621 | 81 | 11.267562154 | 11.267463605 | 0.029808 |
| GRU / validation | 64 | 76,989 | 76 | 13.134946701 | 13.134947481 | 0.001322 |
| LSTM / historical test | 16 | 69,909 | 69 | 12.146377998 | 12.146361571 | 0.026282 |
| RNN / historical test | 16 | 69,909 | 89 | 12.211522492 | 12.211566519 | 0.044118 |

CNN–GRU/GRU use whole-case validation. LSTM/RNN checkpoints use the **older**
training-cases-1–6 / within-recording-15%-validation-tail protocol. Their saved
normalization was independently reconstructed with the original code before
removal. Their unchanged case-7/8 test inputs were verified with the new adapter.
These rows verify compatibility, **not a fair cross-model ranking**. Tail-split
validation/training is deliberately not an option in the new canonical layer.

Saved references used PyTorch 2.8.0+cu128; local verification used PyTorch 2.14.0
CPU, NumPy 2.3.5 and pandas 2.2.3. Every RMSE difference is below the initial
1e-4 N check threshold. The initial 1e-3 N *pointwise* cross-device threshold
failed; it was not silently relaxed. The table reports the observed differences.
Exact same-CPU old/new CNN–GRU agreement isolates cleanup from this discrepancy;
hardware/library numerical variation is consistent with the remaining small
differences, but its specific kernel cause was not established. No claim of
bitwise GPU reproduction or reproduction by retraining is made.

## Reference identities

Checkpoint SHA-256 values (not weights embedded in this repository):

```text
cnn_gru_l64.best.pt ddff5556cf475027b3cf2704612ac50b94294fe44659502a32bab87fb42c957f
gru_l64.best.pt     cef98a8b147aaa18bd3866e916082f2bf1517b50770915926a8bb2035c4f790c
lstm.best.pt        475e4f2f55797bb0b71fb3fdddc761ef1cee5f2e28fc62b8a6bdb6891d91d750
rnn.best.pt         f38d5212bb31c32240a53a89bdefaafceba1c11722a8a27f0f74239b127f3472
```

CNN–GRU/GRU weights and validation CSVs came from
`results/priority123_seed0_20260925_p1_snapshot_O9KXeL/`; the clean CNN–GRU test
CSV came from `results/irregular_seed0_20260926_gyHiJX/frozen_test/clean/`.
LSTM/RNN weights and `evaluation_best/` CSVs came from
`results/20hz_seed0_readout_20260925/`. These paths were removed after checking.

The SHA-256 of the sorted JSON mapping of all 123 repository-relative raw-data
filenames to their SHA-256 hashes is
`04ec650dcf95ec2011795f32017baca1e6de056c16075422a19ee115ab72d344`.
Serialization used Python `json.dumps(mapping, sort_keys=True)`.

## Focused tests and limits

`python -m unittest discover -s tests -v`: **18 passed**. Tests cover raw
dimensions/alignment/time, deterministic disjoint cases, boundaries, train-only
scaling, exact next-sample window targets, target/future-input exclusion, metric
correctness, model dimensions/initializers, state-dict round trips, independent
window inference, and comparable plotting inputs. Tests are synthetic and
perform no training or real-data inference. `git diff --check` also passed.

Remaining scientific limits:

- Time is inferred from distance and nominal constant speed, not a supplied
  sensor clock. Upstream filtering phase/causality and sensor units need evidence.
- Source cutoffs alter contact-force truth as well as sensors. They are not
  missingness/sampling variants with identical targets.
- Baselines predict the next sample from preceding observations. Inclusive
  target-time reconstruction would be a different task, not a harmless adapter change.
- Cases 7/8 were already inspected in prior work; they are not untouched
  confirmation data. This cleanup creates no new generalization evidence.
- Checkpoints/predictions were ignored by Git. Removed outputs were moved to
  macOS Trash for recovery; source history alone cannot restore these weights.
