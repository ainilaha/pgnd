"""Compare PGND and baselines on identical held-out targets, in physical force units."""

import argparse
import copy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import torch

from src.baselines import CNNGRU, GRU, LSTM, RNN
from src.data import DATA_DIR, hold_missing_block, prepare_data, recording_batches
from src.model import PGNDModel


def mean_squared_error(target, prediction):
    """Scalar-force MSE, the original paper's primary prediction metric."""
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.size == 0 or target.shape != prediction.shape:
        raise ValueError("Require nonempty, equally sized targets and predictions.")
    if not (np.isfinite(target).all() and np.isfinite(prediction).all()):
        raise ValueError("Targets and predictions must be finite.")
    return float(np.mean((target - prediction) ** 2))


def force_metrics(target, prediction):
    target, prediction = np.asarray(target).reshape(-1), np.asarray(prediction).reshape(-1)
    mse = mean_squared_error(target, prediction)
    variance = float(np.var(target.astype(np.float64)))
    return {"mae_N": float(np.mean(np.abs(target - prediction))),
            "rmse_N": float(np.sqrt(mse)), "mse_N2": mse,
            "r2": 1 - mse / variance if variance > 0 else float("nan")}


def predict(model, X, batch_size=1024, device="cpu"):
    """Predict in sample order without carrying state between windows."""
    if len(X) == 0 or batch_size < 1:
        raise ValueError("Require nonempty inputs and a positive batch size.")
    model.to(device).eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            predictions.append(model(batch).reshape(-1))
    # Transfer once, rather than synchronizing after every prediction batch.
    return torch.cat(predictions).cpu().numpy()


def predict_recordings(model, recordings, context_length, target_start, stride=1,
                       batch_size=1024, device="cpu"):
    """One warm-up per partition; return the same recording-major target order."""
    model.to(device).eval()
    count = sum(len(range(target_start, len(x), stride)) for x in recordings)
    parameter = next(model.parameters())
    predictions = torch.empty(count, dtype=parameter.dtype, device=device)
    batches = recording_batches(recordings, target_start, stride, batch_size, context_length)
    with torch.no_grad():
        for indices, prediction, _ in model.recording_stream(recordings, batches, context_length):
            positions = torch.as_tensor(indices, dtype=torch.long, device=device)
            predictions.index_copy_(0, positions, prediction)
    return predictions.cpu().numpy()


def model_probe(model, X, target, residual_weight=0.001):
    """One fixed batch, no optimizer step: loss gradients and PGND state diagnostics.

    Gradients are with respect to trainable parameters, including the encoder
    and initializer. autograd.grad does not replace existing parameter .grad.
    Term RMS averages over samples, evolved sample times and latent components;
    powers sum components before averaging. These are LATENT, not physical units.
    """
    parameter = next(model.parameters())
    X = torch.as_tensor(X, dtype=parameter.dtype, device=parameter.device)
    target = torch.as_tensor(target, dtype=parameter.dtype, device=parameter.device).reshape(-1, 1)
    if len(X) == 0 or len(X) != len(target) or residual_weight < 0:
        raise ValueError("Require nonempty aligned probe data and nonnegative residual weight.")
    was_training, nfe = model.training, 0

    def count_call(*_):
        nonlocal nfe
        nfe += 1

    hook = model.dynamics.register_forward_hook(count_call) if isinstance(model, PGNDModel) else None
    # cuDNN recurrent backward requires training mode. The supported models
    # have no dropout/batchnorm, so this adds no stochastic training behavior.
    model.train()
    try:
        if isinstance(model, PGNDModel):
            prediction, detail = model(X, return_details=True)
            residual = detail["residual_loss"]
        else:
            prediction, residual = model(X), X.new_zeros(())
        mse = (prediction - target).square().mean()
        parameters = tuple(p for p in model.parameters() if p.requires_grad)

        def gradient(loss, retain_graph=False):
            values = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=retain_graph)
            return torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten()
                              for p, g in zip(parameters, values)])

        mse_gradient = gradient(mse, retain_graph=residual.requires_grad)
        residual_gradient = (gradient(residual_weight * residual) if residual.requires_grad
                             else torch.zeros_like(mse_gradient))
        if not (torch.isfinite(mse_gradient).all() and torch.isfinite(residual_gradient).all()):
            raise ValueError("Non-finite diagnostic gradients.")
        norm_mse, norm_residual = mse_gradient.norm(), residual_gradient.norm()
        row = {"samples": len(X), "mse_scaled": mse.item(), "residual_loss": residual.item(),
               "weighted_residual": (residual_weight * residual).item(),
               "weighted_residual_to_mse": (residual_weight * residual / mse).item() if mse > 0 else None,
               "mse_gradient_norm": norm_mse.item(),
               "weighted_residual_gradient_norm": norm_residual.item(),
               "gradient_norm_ratio": (norm_residual / norm_mse).item() if norm_mse > 0 else None,
               "gradient_cosine": ((mse_gradient @ residual_gradient) / (norm_mse * norm_residual)).item()
               if norm_mse > 0 and norm_residual > 0 else None,
               "forward_nfe": nfe}
        # Parameter-group totals expose missing/imbalanced learning signals.
        offset, norms = 0, {}
        total_gradient = mse_gradient + residual_gradient
        for name, p in model.named_parameters():
            if p.requires_grad:
                group = "residual" if name.startswith("dynamics.residual.") else name.split(".")[0]
                squared = total_gradient[offset:offset + p.numel()].square().sum().item()
                norms[group] = norms.get(group, 0.) + squared
                offset += p.numel()
        row.update({f"gradient_norm_{name}": value ** 0.5 for name, value in norms.items()})
        if isinstance(model, PGNDModel):
            with torch.no_grad():
                states = detail["states"][:, 1:]
                row["state_max_abs"] = detail["states"].abs().max().item()
                row["state_rms"] = states.square().mean().sqrt().item()
                row["integration_intervals"] = states.shape[1]
                # No mechanical energy identity exists for the generic controls.
                if model.dynamics_kind != "structured":
                    return row
                q, v = states.chunk(2, dim=-1)
                encoded = model.encoder(X)
                h = torch.cat((encoded, encoded[:, -1:]), dim=1)[:, model.initialization_index + 1:]
                times = detail["solver_times"][None, 1:, None]
                D, K = model.dynamics.damping(), model.dynamics.stiffness()
                terms = {"damping": -v @ D, "restoring": -q @ K,
                         "drive": model.dynamics.drive(h),
                         "residual": model.dynamics.residual_force(times, states, h)}
                for name, values in {"q": q, "v": v, **terms}.items():
                    row[f"{name}_rms"] = values.square().mean().sqrt().item()
                for name, values in terms.items():
                    row[f"{name}_power_mean"] = (v * values).sum(-1).mean().item()
                energy_rate = (v * (terms["damping"] + terms["drive"] + terms["residual"])).sum(-1)
                row["energy_rate_positive_fraction"] = (energy_rate > 0).float().mean().item()
                for name, matrix in (("damping", D), ("stiffness", K)):
                    eigenvalues = torch.linalg.eigvalsh(matrix.detach().cpu().double())
                    row[f"{name}_eigenvalue_min"] = eigenvalues[0].item()
                    row[f"{name}_eigenvalue_max"] = eigenvalues[-1].item()
        return row
    finally:
        if hook is not None:
            hook.remove()
        model.train(was_training)


def solver_check(model, X, target, force_std):
    """Frozen PGND, same observations/targets; restore every solver option.

    Report a single-batch forward time (not an isolated latency benchmark).
    Differences are from the saved solver, not a claim that it is exact.
    """
    saved = (model.method, model.step_size, model.rtol, model.atol)
    was_training = model.training
    step = model.step_size if model.method == "rk4" else model.sample_interval / model.time_unit
    variants = [("saved", *saved), ("rk4_half", "rk4", step / 2, saved[2], saved[3]),
                ("rk4_quarter", "rk4", step / 4, saved[2], saved[3]),
                ("dopri5_tight", "dopri5", step, min(saved[2], 1e-7), min(saved[3], 1e-9))]
    parameter = next(model.parameters())
    X = torch.as_tensor(X, dtype=parameter.dtype, device=parameter.device)
    target = np.asarray(target).reshape(-1)
    rows, reference, previous = [], None, None
    nfe = 0

    def count_call(*_):
        nonlocal nfe
        nfe += 1

    hook = model.dynamics.register_forward_hook(count_call)
    model.eval()
    try:
        with torch.no_grad():
            for label, method, step_size, rtol, atol in variants:
                model.method, model.step_size, model.rtol, model.atol = method, step_size, rtol, atol
                nfe = 0
                if X.is_cuda:
                    torch.cuda.synchronize(X.device)
                start = time.perf_counter()
                prediction = model(X).reshape(-1)
                if X.is_cuda:
                    torch.cuda.synchronize(X.device)
                seconds = time.perf_counter() - start
                prediction = prediction.cpu().numpy().astype(float)
                if reference is None:
                    reference = prediction
                    previous = prediction
                rows.append({"solver": label, "method": method,
                             "step_seconds": step_size * model.time_unit if method == "rk4" else None,
                             "rtol": rtol, "atol": atol, "samples": len(X), "forward_nfe": nfe,
                             "single_batch_seconds": seconds,
                             "rmse_N": mean_squared_error(target, prediction) ** 0.5 * force_std,
                             "rms_change_N": mean_squared_error(reference, prediction) ** 0.5 * force_std,
                             "rms_change_from_previous_N": mean_squared_error(previous, prediction) ** 0.5 * force_std,
                             "max_change_N": float(np.max(np.abs(prediction - reference))) * force_std})
                previous = prediction
    finally:
        hook.remove()
        model.method, model.step_size, model.rtol, model.atol = saved
        model.train(was_training)
    return rows


def residual_scaling_check(model, X, force_std):
    """Frozen reparameterization z'=c*z: same predictor, penalty scales by c².

    Diagnostic of the LATENT penalty's non-identifiability, not a new model or
    optimizer step. Work on a copy; adaptive-solver tolerances may break exact
    numerical invariance, so report prediction changes as well as loss ratios.
    """
    if not isinstance(model, PGNDModel) or not model.use_residual:
        raise ValueError("Residual scaling requires structured PGND with a residual.")
    parameter = next(model.parameters())
    X = torch.as_tensor(X, dtype=parameter.dtype, device=parameter.device)
    rows = []
    with torch.no_grad():
        frozen = copy.deepcopy(model).eval()
        reference, reference_loss = frozen(X, return_residual=True)
        for scale in (0.1, 1., 10.):
            candidate = copy.deepcopy(frozen)
            candidate.initial_state[-1].weight.mul_(scale)
            candidate.initial_state[-1].bias.mul_(scale)
            candidate.decoder[0].weight.div_(scale)
            candidate.dynamics.drive.weight.mul_(scale)
            d = candidate.dynamics.latent_dim
            candidate.dynamics.residual[0].weight[:, :2 * d].div_(scale)
            candidate.dynamics.residual[-1].weight.mul_(scale)
            candidate.dynamics.residual[-1].bias.mul_(scale)
            prediction, residual = candidate(X, return_residual=True)
            difference = (prediction - reference) * force_std
            rows.append({"latent_scale": scale, "residual_loss": residual.item(),
                         "loss_ratio": (residual / reference_loss).item() if reference_loss > 0 else None,
                         "expected_loss_ratio": scale ** 2,
                         "rms_prediction_change_N": difference.square().mean().sqrt().item(),
                         "max_prediction_change_N": difference.abs().max().item()})
    return rows


def paired_case_comparison(recording_results, reference):
    """Paired bootstrap of whole cases: keep speeds/cutoffs together.

    Equal weight per recording within each case, then per case. This differs
    from pooled-window RMSE. Intervals are conditional on these fitted models;
    they do not include seed uncertainty or correct validation selection bias.
    Fewer than five cases: report differences, not misleading confidence limits.
    """
    frame = pd.DataFrame(recording_results).copy()
    frame["case"] = frame["file"].str.extract(r"_Case(\d+)_", expand=False)
    if frame.case.isna().any():
        raise ValueError("Case diagnostics require filenames containing _Case<number>_.")
    cases = frame.groupby(["run", "case"], sort=True).mse_N2.mean().unstack("run")
    if cases.isna().any().any():
        raise ValueError("Paired diagnostics require identical cases for every model.")
    rng = np.random.default_rng(0)
    draws = rng.integers(len(cases), size=(2000, len(cases)))
    ref = cases[reference].to_numpy()
    rows = []
    for run in cases.columns:
        if run == reference:
            continue
        other = cases[run].to_numpy()
        difference = np.sqrt(ref[draws].mean(1)) - np.sqrt(other[draws].mean(1))
        low, high = np.quantile(difference, [0.025, 0.975]) if len(cases) >= 5 else (None, None)
        rows.append({"reference": reference, "comparison": run, "cases": len(cases),
                     "reference_case_balanced_rmse_N": np.sqrt(ref.mean()),
                     "comparison_case_balanced_rmse_N": np.sqrt(other.mean()),
                     "difference_rmse_N": np.sqrt(ref.mean()) - np.sqrt(other.mean()),
                     "case_bootstrap_low_N": low, "case_bootstrap_high_N": high,
                     "interval_status": "exploratory_conditional_on_fit" if len(cases) >= 5
                     else "too_few_cases"})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for CSV results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--split", choices=["validation", "test"], default="test",
                        help="Use validation during method development; test remains the default")
    parser.add_argument("--preload-data", action="store_true",
                        help="Keep inputs for the evaluated split on the selected device")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Validation only: fixed probes, PGND solver checks and paired case intervals")
    parser.add_argument("--missing-block", type=int, default=0,
                        help="Hide the final N input samples in every window; causal hold, same targets")
    args = parser.parse_args()
    if args.missing_block < 0:
        parser.error("--missing-block must be nonnegative.")
    if args.diagnostics and args.split != "validation":
        parser.error("--diagnostics requires --split validation; do not tune using test diagnostics")
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new results directory: {args.output_dir}")
    if len({path.stem for path in args.checkpoints}) != len(args.checkpoints):
        raise ValueError("Checkpoint filenames need distinct stems for prediction CSVs.")
    torch.set_num_threads(args.threads)
    checkpoints = [torch.load(path, map_location="cpu", weights_only=True) for path in args.checkpoints]
    reference = checkpoints[0]
    persistent = [c.get("state_mode", "window") == "persistent" for c in checkpoints]
    if any(c.get("state_mode", "window") not in ("window", "persistent") for c in checkpoints):
        raise ValueError("Unknown checkpoint state mode.")
    if any(persistent) and (args.missing_block or args.diagnostics):
        raise ValueError("Persistent-state pilot is clean only; window outage/probe diagnostics do not apply.")
    if reference["protocol"] not in ("recording-split-training-standardization-v1",
                                     "case-validation-training-standardization-v1"):
        raise ValueError("Retrain historical checkpoints under the shared revised protocol.")
    for checkpoint in checkpoints:
        # Legacy checkpoints have final weights and no selection metadata.
        kind = checkpoint.get("checkpoint_kind", "final_epoch")
        if kind not in ("final_epoch", "best_validation"):
            raise ValueError(f"Unknown checkpoint selection rule: {kind}")
        if kind != reference.get("checkpoint_kind", "final_epoch"):
            raise ValueError("Do not mix best-validation and final-epoch checkpoints.")
        if checkpoint.get("completed_epochs", checkpoint["epochs"]) != checkpoint["epochs"]:
            raise ValueError("Training budget is incomplete; wait for the run to finish.")
    # Reject incompatible comparisons, including different training subsets/scales.
    for checkpoint in checkpoints[1:]:
        for field in ("protocol", "train_files", "test_files", "normalization",
                      "data_sha256", "sample_interval", "seed", "epochs", "batch_size", "optimizer"):
            if checkpoint[field] != reference[field]:
                raise ValueError(f"Checkpoints differ in {field}; not the same experiment.")
        current, saved = checkpoint["data_settings"], reference["data_settings"]
        if current != saved:
            # Different context lengths are comparable ONLY with an explicit
            # common target anchor used during training as well as evaluation.
            anchor = saved.get("target_start")
            if (anchor is None or anchor != current.get("target_start")
                    or anchor < max(saved.get("sequence_length", 16), current.get("sequence_length", 16))
                    or {k: v for k, v in current.items() if k != "sequence_length"}
                    != {k: v for k, v in saved.items() if k != "sequence_length"}):
                raise ValueError("Checkpoints differ in data_settings; context comparisons need matched target_start.")
        if checkpoint.get("window_counts") != reference.get("window_counts"):
            raise ValueError("Checkpoints have different window counts; targets/update budgets are not matched.")
    for filename, expected in reference["data_sha256"].items():
        if hashlib.sha256((args.data_dir / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Data changed since training: {filename}")
    settings = dict(reference["data_settings"])
    lengths = [c["data_settings"].get("sequence_length", 16) for c in checkpoints]
    if args.missing_block >= min(lengths):
        raise ValueError("Missing block must leave at least one observation in every model's history.")
    if len(set(lengths)) > 1:
        settings["sequence_length"] = max(lengths)
    arrays, metadata, normalization, _ = prepare_data(
        reference["train_files"], reference["test_files"], args.data_dir,
        normalization=reference["normalization"], **settings)
    recording_inputs = None
    if any(persistent):
        recording_arrays, recording_metadata, _, _ = prepare_data(
            reference["train_files"], reference["test_files"], args.data_dir,
            normalization=reference["normalization"], recordings=True, **settings)
        pd.testing.assert_frame_equal(metadata[args.split], recording_metadata[args.split])
        recording_inputs = recording_arrays[args.split][0]
        if args.preload_data:
            recording_inputs = [torch.as_tensor(x, dtype=torch.float32, device=args.device)
                                for x in recording_inputs]
    full_X, _ = arrays[args.split]
    if args.missing_block:
        full_X = hold_missing_block(torch.as_tensor(full_X), full_X.shape[1] - args.missing_block,
                                    args.missing_block)
    if args.preload_data:
        full_X = torch.as_tensor(full_X, dtype=torch.float32, device=args.device)
    target = metadata[args.split]["force_N"].to_numpy()
    args.output_dir.mkdir(parents=True)
    # Always record the evaluation condition; a stress score is not a clean score.
    evaluation = {"split": args.split, "missing_block": args.missing_block,
                  "missing_seconds": args.missing_block * reference["sample_interval"],
                  "policy": "both channels hidden at history end; hold last available value; first sample kept",
                  "scope": ("clean recording-prefix versus window comparison; information histories differ"
                            if any(persistent) else
                            "fixed-grid filled outage per forecast window, not a recording-wide irregular mask"),
                  "normalization": normalization, "data_settings": settings,
                  "data_sha256": reference["data_sha256"],
                  "checkpoints": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.checkpoints},
                  "input_augmentation": {p.stem: c.get("input_augmentation", {"max_length": 0})
                                         for p, c in zip(args.checkpoints, checkpoints)},
                  "state_modes": {p.stem: c.get("state_mode", "window")
                                  for p, c in zip(args.checkpoints, checkpoints)},
                  "state_protocols": {p.stem: c.get("state_protocol")
                                      for p, c in zip(args.checkpoints, checkpoints)},
                  "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [*sorted(Path(__file__).parent.glob("*.py")),
                                              Path(__file__).resolve().parents[1] / "legacy" / "pgnd_experiments.py"]}}
    (args.output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    if args.diagnostics:
        # Deterministic 16 targets per recording, selected without looking at errors.
        indices = np.concatenate([group.index.to_numpy()[np.linspace(0, len(group) - 1,
                                  min(16, len(group)), dtype=int)]
                                  for _, group in metadata[args.split].groupby("file", sort=True)])
        metadata[args.split].iloc[indices].to_csv(args.output_dir / "probe_targets.csv", index=False)
        provenance = {"split": args.split, "missing_block": args.missing_block, "samples_per_recording": 16,
                      "selection": "evenly spaced eligible targets per recording",
                      "device": args.device, "torch_version": str(torch.__version__),
                      "torchdiffeq_version": version("torchdiffeq"),
                      "cuda_version": torch.version.cuda,
                      "device_name": torch.cuda.get_device_name(args.device)
                      if torch.device(args.device).type == "cuda" else args.device,
                      "dtype": "float32",
                      "checkpoints": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in args.checkpoints},
                      "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted(Path(__file__).parent.glob("*.py"))},
                      "data_sha256": reference["data_sha256"],
                      "data_settings": reference["data_settings"], "normalization": normalization,
                      "runs": {p.stem: {"data_settings": c["data_settings"],
                                        "model_options": c["model_options"],
                                        "residual_weight": c.get("residual_weight", 0.)}
                               for p, c in zip(args.checkpoints, checkpoints)},
                      "case_bootstrap": {"seed": 0, "replicates": 2000, "minimum_cases": 5},
                      "note": "Probes are not full-split metrics or physical parameter estimates. "
                              "Case intervals are exploratory and conditional on selected fits."}
        (args.output_dir / "diagnostics.json").write_text(json.dumps(provenance, indent=2) + "\n")
    results, recording_results = [], []
    for path, checkpoint in zip(args.checkpoints, checkpoints):
        name = checkpoint["model"]
        length = checkpoint["data_settings"].get("sequence_length", full_X.shape[1])
        X = full_X[:, -length:]
        if name == "rnn":
            model = RNN()
        elif name == "lstm":
            model = LSTM()
        elif name == "gru":
            model = GRU()
        elif name == "cnn_gru":
            model = CNNGRU()
        elif name in ("pgnd", "pgnd_obs"):
            options = checkpoint["model_options"]
            if any(key in options for key in ("initialization", "warmup_steps", "dynamics_kind", "use_residual")):
                from legacy.pgnd_experiments import AblationPGND
                model = AblationPGND(**options)
            else:
                model = PGNDModel(**options)
        elif name in ("direct", "history_linear", "history_mlp", "pgnd_modal"):
            # Archived pilots: load old weights without expanding normal training.
            from legacy.pgnd_experiments import DirectReadout, HistoryReadout, ModalPGND
            if name == "direct":
                model = DirectReadout(**checkpoint["model_options"])
            elif name == "pgnd_modal":
                model = ModalPGND(**checkpoint["model_options"])
            else:
                model = HistoryReadout(**checkpoint["model_options"])
        else:
            raise ValueError(f"Unsupported model: {name}")
        model.load_state_dict(checkpoint["state_dict"])
        stateful = checkpoint.get("state_mode", "window") == "persistent"
        if stateful:
            if type(model) is not PGNDModel:
                raise ValueError("Persistent-state checkpoints require the original PGND architecture.")
            prediction = predict_recordings(model, recording_inputs, length,
                                            settings.get("target_start", length), device=args.device)
        else:
            prediction = predict(model, X, device=args.device)
        prediction = prediction.astype(np.float64)
        prediction = prediction * normalization["force_std"] + normalization["force_mean"]
        row = {"run": path.stem, "model": name, "seed": checkpoint["seed"],
               "split": args.split, "samples": len(target),
               "missing_block": args.missing_block,
               "train_max_missing_block": checkpoint.get("input_augmentation", {}).get("max_length", 0),
               "sequence_length": length, "target_start": settings.get("target_start", length),
               "state_mode": "persistent" if stateful else "window",
               "history_scope": "whole_recording_prefix" if stateful else "fixed_window",
               "recording_warmup_observations": length if stateful else None,
               "initialization": getattr(model, "initialization", None),
               "warmup_steps": model.initialization_index + 1 if isinstance(model, PGNDModel) else None,
               "dynamics": getattr(model, "dynamics_kind", None),
               "residual_enabled": getattr(model, "use_residual", False),
               "residual_weight": checkpoint.get("residual_weight", 0.),
               "checkpoint_kind": checkpoint.get("checkpoint_kind", "final_epoch"),
               "selected_epoch": checkpoint.get("selected_epoch", checkpoint["epochs"]),
               "validation_mse_scaled": checkpoint.get(
                   "selected_val_mse", checkpoint["history"][-1]["val_mse_scaled"]),
               "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "budget_updates": checkpoint["epochs"] * (
                   (checkpoint["window_counts"]["train"] + checkpoint["batch_size"] - 1)
                   // checkpoint["batch_size"]) if "window_counts" in checkpoint else None,
               "selected_checkpoint_updates": checkpoint.get("selected_epoch", checkpoint["epochs"]) * (
                   (checkpoint["window_counts"]["train"] + checkpoint["batch_size"] - 1)
                   // checkpoint["batch_size"]) if "window_counts" in checkpoint else None,
               "training_validation_seconds": sum(r["seconds"] for r in checkpoint["history"])
               if all("seconds" in r for r in checkpoint["history"]) else None,
               "diagnostic_seconds": sum(r.get("probe_seconds", 0.) for r in checkpoint["history"]),
               **force_metrics(target, prediction)}
        results.append(row)
        if args.diagnostics:
            probe_X, probe_y = X[indices], arrays[args.split][1][indices]
            probe = model_probe(model, probe_X, probe_y, checkpoint.get("residual_weight", 0.))
            pd.DataFrame([probe]).to_csv(args.output_dir / f"{path.stem}.probe.csv", index=False)
            if isinstance(model, PGNDModel):
                pd.DataFrame(solver_check(model, probe_X, probe_y, normalization["force_std"])).to_csv(
                    args.output_dir / f"{path.stem}.solver.csv", index=False)
                if model.use_residual:
                    pd.DataFrame(residual_scaling_check(model, probe_X, normalization["force_std"])).to_csv(
                        args.output_dir / f"{path.stem}.residual_scaling.csv", index=False)
        pd.DataFrame([row]).to_csv(args.output_dir / f"{path.stem}.metrics.csv", index=False)
        # Bundle the checkpoint's actual history, not an unrelated sidecar file.
        pd.DataFrame(checkpoint["history"]).to_csv(
            args.output_dir / f"{path.stem}.history.csv", index=False)
        predictions = metadata[args.split].copy()
        predictions["missing_block"] = args.missing_block
        predictions["last_observed_source_row"] = predictions["source_row"] - 1 - args.missing_block
        predictions["state_origin_source_row"] = (
            predictions.groupby("file").source_row.transform("min") - settings.get("target_start", length)
            if stateful else predictions["source_row"] - length)
        predictions["prediction_N"] = prediction
        predictions.to_csv(args.output_dir / f"{path.stem}.predictions.csv", index=False)
        for filename, recording in predictions.groupby("file", sort=False):
            recording_results.append({
                "run": path.stem, "model": name, "split": args.split,
                "missing_block": args.missing_block,
                "train_max_missing_block": checkpoint.get("input_augmentation", {}).get("max_length", 0),
                "file": filename, "samples": len(recording),
                **force_metrics(recording["force_N"], recording["prediction_N"]),
            })
    table = pd.DataFrame(results)
    table.to_csv(args.output_dir / "comparison.csv", index=False)
    pd.DataFrame(recording_results).to_csv(args.output_dir / "per_recording.csv", index=False)
    if args.diagnostics and len(checkpoints) > 1:
        paired_case_comparison(recording_results, args.checkpoints[0].stem).to_csv(
            args.output_dir / "paired_cases.csv", index=False)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
