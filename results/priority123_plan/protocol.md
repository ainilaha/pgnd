# Priority 1–3: fixed diagnostic ablations (not a replacement PGND)

## Gate and budget

First inspect the frozen Priority 0 `pgnd.best.probe.csv`,
`pgnd.best.solver.csv` and `paired_cases.csv`. The printed accuracy table does
not certify these checks. Investigate non-finite values, absent prediction
gradients or probe solver changes above 0.1 N or 1% of saved probe RMSE before
training. These thresholds are screening flags, not a convergence proof.

The script defines **12 runs maximum, seed 0, 100 epochs each**, in stages
8 + 2 + 2. This replaces, rather than automatically adds to, the unlaunched
25-run reference plan. No automatic tuning, continuation, early stopping or
winner promotion. No training has been launched as part of preparing it.
Negative results and all completed runs must be reported. Replication across
seeds requires a separately agreed budget; a seed-0 screen is not significance.

## Common experimental controls

- 20-Hz-filtered simulation files, all three speeds. This is NOT 20-Hz sampling.
  Training cases 1–4; validation cases 5–6; cases 7–8 remain the previously
  examined development test and are not scored by this script.
- Whole-case validation replaces the earlier within-recording tail split;
  scalers are refitted on retained rows of cases 1–4 only. Trim 10% per end.
- Target anchor 64 in **every recording partition and every split**, including
  L=16 runs. Training target indices are 64, 80, 96, …; validation/test targets
  are 64, 65, 66, …. Every input is the last L observations before its target.
  Thus train targets, minibatch ordering, update counts, validation/test targets
  and scalers match across L. Short contexts deliberately discard targets that
  longer contexts cannot predict. No concatenation across files.
- Consequently these are NEW controlled runs, not directly comparable to the
  supplied old seed-0 scores. Old checkpoints are not reused as trained controls.
- Acceleration/displacement only; no force history or target-time observation.
  Same observation encoder, piecewise-linear conditioning, last-input hold,
  1-ms sample/solver unit, RK4 step 1, readout and standardized-force MSE.
  Linear conditioning is retrospective within an observed window; it does not
  make intermediate states online forecasts. Only the following force is scored.
- Batch 128, Adam LR .001, betas .9/.999, epsilon 1e-7, train stride 16, seed 0;
  no scheduler, clipping or new losses. Best checkpoint by validation MSE only.
  Model and DataLoader seeds match; shared PGND modules keep their seeded values.
  Baseline initializer conventions stay unchanged. Runs execute sequentially.
- Assess pooled validation RMSE/MAE plus each case/recording; retain histories,
  training gradient probes, validation solver checks, NFE, parameters and time.
  The primary score is pooled validation RMSE (same ranking as selection MSE).
  A useful screening signal is >=1% relative RMSE reduction versus the named
  control with lower case-balanced error in BOTH validation cases; otherwise
  report it as weak, mixed or negative evidence. This is not a significance test.
  Investigate any solver/stability flag even if RMSE improves. With only two
  validation cases, paired-case confidence intervals are deliberately omitted.
  Each comparison also exports `primary_contrasts.csv` for the predeclared pairs
  below, using the same case-balanced calculation (candidate minus control;
  negative favors the candidate). The complete `per_recording.csv` remains the
  source for checking both validation cases rather than hiding mixed effects.

## P1: context and causal initialization (8 runs)

Hypothesis A: 16 samples provide too little history for latent-state inference.
Run PGND-0, CNN–GRU and GRU at L=16 and L=64. Compare L64 vs L16 **within each
model**, not just PGND-L64 vs baseline-L16. CNN–GRU and GRU are the two focused
sequence controls; this screen does not rerun every LSTM/RNN combination.
An improvement shared by all three suggests a context bottleneck, not a
PGND-specific advantage. Longer context incurs more ODE work; updates, not wall
time, are matched.

Hypothesis B: one observation poorly initializes a partially observed state.
At fixed L=16 compare:

| Initializer | Observations used to initialize | ODE start | Parameters |
|---|---|---|---:|
| first (PGND-0) | x[0] | t[0] | 5,761 |
| delayed control | x[7] | t[7] | 5,761 |
| history | flattened x[0:8] | t[7] | 6,209 |

Primary contrast: history vs delayed. Both integrate the same 9 intervals;
history does not look beyond its start instant. Delayed vs first separately
reveals the change in start time/integration horizon. History adds 448 input
weights, so a gain is evidence for this history-initialization design, not a
capacity-matched proof. The initializer output layer, encoder, dynamics and
readout retain the original seeded weights. No GRU is inserted into PGND.

## P2: interpretable residual experiment (2 additional runs)

Hypothesis: the correction provides useful flexibility, while its penalty may
be weak or merely alter an arbitrary latent scale. Keep L=16 and first-sample
initialization fixed, regardless of P1 outcomes:

- Full PGND, lambda=.001: reuse the NEW `pgnd_l16` reference (5,761 parameters).
- No residual: remove its network/parameters entirely, r=0 (3,633 parameters).
  Compare against unpenalized full PGND to test residual capacity.
- Unpenalized residual: retain the full network but set lambda=0 (5,761).
  Compare against lambda=.001 to test the regularizer, not residual presence.

Validation diagnostics also write `*.residual_scaling.csv`: frozen copies with
z'=c*z, c in {0.1, 1, 10}, rescaled initializer/drive/residual and inverse-scaled
decoder. In exact arithmetic predictions are invariant but the residual loss
scales by c². Report measured prediction changes (solver numerics may matter).
This demonstrates a non-identifiability, NOT a remedy or a new training model.
Do not call small residual norms identified physical adherence; gradient and
term magnitudes are parameterization-dependent too. No new arbitrary norm
constraint or lambda sweep is introduced here.

## P3: mechanical-structure controls (2 additional runs)

Hypothesis: the mechanical organization itself helps beyond generic latent
continuous-time processing. Keep L=16 and first initialization; do not inherit
the winning P1 setting. All four contrasts here have lambda=0:

| Dynamics | Definition | Parameters |
|---|---|---:|
| Full, unpenalized PGND | dq=v; dv=-Dv-Kq+Bh+r(z,h,t) | 5,761 |
| Structured only | dq=v; dv=-Dv-Kq+Bh | 3,633 |
| Neural second order | dq=v; dv=f(z,h,t) | 4,993 |
| Neural first order | dz=f(z,h,t) | 5,521 |

Generic f uses the same concatenated (z,h,t), a 32-wide tanh hidden layer and
linear output; all states have 32 components. Second-order f outputs 16
accelerations; first-order f outputs all 32 derivatives. It has normal PyTorch
Linear initialization, not a zero-output initialization (f is the entire field,
unlike PGND's correction). Encoder/initializer/readout values are seed-matched.
No unused D/K/B or residual-penalty parameters remain in the generic controls.
The unrestricted first-order state is NOT labeled mechanical q/v in diagnostics.

Primary contrasts: full unpenalized PGND vs neural second order (explicit
PSD restoring/damping plus linear drive); second vs first order (dq=v).
Structured-only vs full also uses the P2 residual-capacity contrast. Counts and
initial vector fields differ, so these isolate architectural families, not a
perfect parameter-count-matched physical causal effect. A gain requires later
capacity/seed checks; a loss argues against promoting mechanical claims under
this task. Driven latent energy growth alone is not an implementation error.
Separate energy/potential/dissipation removals would require a follow-up design;
they are not silently bundled into this 12-run screen.

## Commands (after transferring the updated code and this plan to the server)

```bash
cd /root/pgnd
RUN=results/priority123_seed0_20260925
bash results/priority123_plan/run.sh p1 "$RUN"
# Inspect context_validation/ and initialization_validation/, including solver CSVs.
bash results/priority123_plan/run.sh p2 "$RUN"
# Inspect residual_validation/ before continuing.
bash results/priority123_plan/run.sh p3 "$RUN"
```

Each stage trains, evaluates only validation, and saves PDF figures. Stop and
diagnose a failing stage; do not change a setting selectively for one candidate.
The script snapshots sources and refuses source changes between stages and
overwriting completed runs. It does not resume interrupted training. Choose a
new run directory for a changed protocol. No automatically scheduled runs.
Only these two plan files are exempt from the `results/` Git ignore. Run
artifacts remain ignored; transfer the plan with the code and back up outputs.
