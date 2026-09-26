"""Train PGND or a baseline on the same recording-safe, standardized windows."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines import CNNGRU, GRU, LSTM, RNN
from src.data import DATA_DIR, augment_missing_blocks, get_files, prepare_data, recording_batches
from src.evaluate import mean_squared_error, model_probe, predict, predict_recordings
from src.model import PGNDModel


def save_checkpoint(checkpoint, path):
    """Replace only this run's checkpoint after a complete temporary write."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train_recording_epoch(model, recordings, target, optimizer, batch_size,
                          context_length, target_start, stride, residual_weight):
    """Truncated gradients, persistent values, exact original target/update count.

    Accumulate microchunk gradients into batches of supervised targets. Earlier
    states are carried unchanged across optimizer updates, not recomputed or
    reset; validation uses frozen parameters throughout each recording.
    """
    model.train()
    optimizer.zero_grad()
    totals = torch.zeros(2, dtype=torch.float64, device=target.device)
    pending, updates = 0, 0
    batches = recording_batches(recordings, target_start, stride, batch_size, context_length)

    def update(count):
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if count != batch_size:
            for gradient in gradients:
                gradient.mul_(batch_size / count)
        if not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError("A persistent-state training gradient became non-finite.")
        optimizer.step()
        optimizer.zero_grad()

    for indices, prediction, residual in model.recording_stream(
            recordings, batches, context_length, return_residual=True):
        positions = torch.as_tensor(indices, dtype=torch.long, device=target.device)
        squared_error = (prediction - target.index_select(0, positions).reshape(-1)).square()
        loss = (squared_error.sum() + residual_weight * residual.sum()) / batch_size
        if not torch.isfinite(loss):
            raise ValueError("Persistent-state training loss became non-finite.")
        loss.backward()
        totals += torch.stack((squared_error.detach().sum(), residual.detach().sum())).double()
        pending += len(indices)
        if pending == batch_size:
            update(pending)
            pending = 0
            updates += 1
    if pending:
        update(pending)
        updates += 1
    return *totals.cpu().tolist(), updates


def fit(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32,
        seed=0, device="cpu", learning_rate=0.001, residual_weight=0.001,
        history_path=None, preload_data=False, checkpoint=None, best_path=None,
        diagnostics=False, max_missing_block=0, persistent_state=False,
        context_length=64, target_start=64, train_stride=16):
    """Adam with a fixed epoch budget and best-validation prediction-MSE saving.

    MSE is in standardized-force units. PGND adds the manuscript's residual
    penalty; baselines have no residual term. Minibatch order has its own seed.
    Write each completed epoch immediately, preserving history if interrupted.
    Optional diagnostics use the same up-to-64 evenly spaced training windows
    at every epoch end, with no update. Probe time is logged separately.
    Optional missing-block augmentation affects inputs only; validation and
    checkpoint selection stay clean. A separate RNG gives all models matched
    augmentations when the seed, windows and minibatch budget match.
    """
    if epochs < 1 or batch_size < 1 or residual_weight < 0:
        raise ValueError("Require positive epochs/batch size and nonnegative residual weight.")
    if (checkpoint is None) != (best_path is None):
        raise ValueError("Provide both checkpoint metadata and best_path, or neither.")
    if persistent_state and (type(model) is not PGNDModel or max_missing_block or diagnostics):
        raise ValueError("Persistent-state pilot requires original PGND, clean data and no window probes.")
    if not persistent_state and not 0 <= max_missing_block < X_train.shape[1]:
        raise ValueError("max_missing_block must be >= 0 and smaller than the history.")
    missing_rng = np.random.default_rng(seed)
    device = torch.device(device)
    model.to(device)
    train_targets = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    if persistent_state:
        train_inputs = [torch.as_tensor(x, dtype=torch.float32,
                        device=device if preload_data else "cpu") for x in X_train]
        X_val = [torch.as_tensor(x, dtype=torch.float32,
                 device=device if preload_data else "cpu") for x in X_val]
        train_targets = train_targets.to(device)
    else:
        train_inputs = torch.as_tensor(X_train, dtype=torch.float32)
    if preload_data and not persistent_state:
        train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)
        X_val = torch.as_tensor(X_val, dtype=torch.float32, device=device)
        # Shuffle CPU indices with the same DataLoader generator/order as before.
        # Gather a whole GPU batch at once, not one GPU operation per sample.
        dataset = torch.arange(len(train_targets))
    elif not persistent_state:
        dataset = TensorDataset(train_inputs, train_targets)
    if not persistent_state:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed),
                            pin_memory=device.type == "cuda")
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=learning_rate, betas=(0.9, 0.999), eps=1e-7)
    loss_fn = nn.MSELoss()
    history = []
    best_mse, best_checkpoint = float("inf"), None
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        model.train()
        # Match the old Python-double accumulation without per-batch .item().
        totals = torch.zeros(2, dtype=torch.float64,
                             device="cpu" if device.type == "mps" else device)
        for batch in (() if persistent_state else loader):
            if preload_data:
                indices = batch.to(device, non_blocking=True)
                X_batch = train_inputs.index_select(0, indices)
                y_batch = train_targets.index_select(0, indices)
            else:
                X_batch, y_batch = (tensor.to(device, non_blocking=True) for tensor in batch)
            X_batch = augment_missing_blocks(X_batch, max_missing_block, missing_rng)
            optimizer.zero_grad()
            if isinstance(model, PGNDModel):
                prediction, residual = model(X_batch, return_residual=True)
            else:
                prediction = model(X_batch)
                residual = prediction.new_zeros(())
            mse = loss_fn(prediction, y_batch)
            loss = mse + residual_weight * residual
            if not torch.isfinite(loss):
                raise ValueError("Training loss became non-finite.")
            loss.backward()
            gradients = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
            if not torch.isfinite(gradients).all():
                raise ValueError("A training gradient became non-finite.")
            optimizer.step()
            totals += torch.stack((mse.detach(), residual.detach())).to(
                device=totals.device, dtype=torch.float64) * len(X_batch)
        if persistent_state:
            squared_error, residual_total, updates = train_recording_epoch(
                model, train_inputs, train_targets, optimizer, batch_size,
                context_length, target_start, train_stride, residual_weight)
            validation_prediction = predict_recordings(model, X_val, context_length, target_start, device=device)
        else:
            squared_error, residual_total = totals.cpu().tolist()
            updates = len(loader)
            validation_prediction = predict(model, X_val, device=device)
        val_mse = mean_squared_error(y_val, validation_prediction)
        row = {"epoch": epoch, "train_mse_scaled": squared_error / len(train_targets),
               "residual_loss": residual_total / len(train_targets), "val_mse_scaled": val_mse,
               "seconds": time.perf_counter() - start}
        row["train_total_loss"] = row["train_mse_scaled"] + residual_weight * row["residual_loss"]
        if persistent_state:
            row["optimizer_updates"] = updates
        if diagnostics:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            probe_start = time.perf_counter()
            indices = torch.as_tensor(np.linspace(0, len(train_targets) - 1,
                                      min(64, len(train_targets)), dtype=int), device=train_inputs.device)
            probe = model_probe(model, train_inputs[indices], train_targets[indices], residual_weight)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row.update({f"probe_{name}": value for name, value in probe.items()})
            row["probe_seconds"] = time.perf_counter() - probe_start
        history.append(row)
        if history_path is not None:
            with Path(history_path).open("x" if epoch == 1 else "a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=row.keys())
                if epoch == 1:
                    writer.writeheader()
                writer.writerow(row)
        if best_path is not None and val_mse < best_mse:
            best_mse = val_mse
            best_checkpoint = {
                **checkpoint, "checkpoint_kind": "best_validation",
                "selected_epoch": epoch, "completed_epochs": epoch,
                "selection_metric": "val_mse_scaled", "selected_val_mse": val_mse,
                "state_dict": {name: value.detach().cpu().clone()
                               for name, value in model.state_dict().items()},
                "history": list(history),
            }
            save_checkpoint(best_checkpoint, best_path)
        print(f"Epoch {epoch}/{epochs}: mse={row['train_mse_scaled']:.6g}, "
              f"residual={row['residual_loss']:.6g}, val_mse={val_mse:.6g}, "
              f"seconds={row['seconds']:.2f}", flush=True)
    if best_checkpoint is not None:
        # Keep the selected weights, but retain the full completed history for plots.
        best_checkpoint.update(history=history, completed_epochs=epochs)
        save_checkpoint(best_checkpoint, best_path)
        print(f"Best validation epoch: {best_checkpoint['selected_epoch']}, "
              f"mse={best_mse:.6g}; saved {best_path}", flush=True)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=["rnn", "lstm", "gru", "cnn_gru", "pgnd"])
    parser.add_argument("--train-pattern", required=True, help="Regex for training-pool filenames")
    parser.add_argument("--test-pattern", required=True, help="Regex for held-out test filenames")
    parser.add_argument("--validation-pattern", help="Optional whole-case validation regex; replaces tail split")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, required=True,
                        help="New final .pt checkpoint path; also writes <stem>.best.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--target-start", type=int,
                        help="First target row within EVERY split partition; >= L, shared across context runs")
    parser.add_argument("--cut-percent", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--preload-data", action="store_true",
                        help="Keep prepared training/validation inputs on the selected device")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Log fixed training-probe loss gradients, states/terms and forward NFE per epoch")
    parser.add_argument("--max-missing-block", type=int, default=0,
                        help="Opt-in input augmentation: 50%% clean, otherwise one held block of 1..N samples")
    parser.add_argument("--persistent-state", action="store_true",
                        help="Clean PGND pilot: one warm-up per recording; carry state, truncate gradients at L steps")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--encoding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--residual-weight", type=float, default=0.001)
    parser.add_argument("--ode-method", choices=["rk4", "dopri5"], default="rk4")
    parser.add_argument("--ode-step", type=float, default=1.0, help="Step in time-unit units (RK4 only)")
    parser.add_argument("--time-unit", type=float, default=0.001, help="Seconds per ODE time unit")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-7)
    args = parser.parse_args()
    if args.persistent_state and (args.model != "pgnd" or args.max_missing_block or args.diagnostics
                                 or args.ode_method != "rk4" or args.target_start not in (None, args.sequence_length)):
        parser.error("--persistent-state requires clean PGND/RK4, target_start=L and no window --diagnostics.")
    if not 0 <= args.max_missing_block < args.sequence_length:
        parser.error("Require 0 <= --max-missing-block < --sequence-length.")
    if args.target_start is not None and args.target_start < args.sequence_length:
        parser.error("--target-start must be >= --sequence-length.")
    if args.residual_weight < 0:
        parser.error("--residual-weight must be nonnegative.")
    history_path = args.output.with_suffix(".history.csv")
    settings_path = args.output.with_suffix(".run.json")
    best_path = args.output.with_suffix(".best.pt")
    if args.output.suffix != ".pt":
        parser.error("--output must end in .pt")
    if any(path.exists() for path in (args.output, best_path, history_path, settings_path)):
        raise FileExistsError(f"Choose a new output path: {args.output}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_files = get_files(args.train_pattern, args.data_dir)
    test_files = get_files(args.test_pattern, args.data_dir)
    validation_files = get_files(args.validation_pattern, args.data_dir) if args.validation_pattern else None
    train_runs = {re.sub(r"_CutFre\d+", "", name) for name in train_files}
    test_runs = {re.sub(r"_CutFre\d+", "", name) for name in test_files}
    if train_runs & test_runs:
        warnings.warn("Train/test share speed/case at different cutoffs: bandwidth transfer, "
                      "not unseen-trajectory generalization (identical files are rejected).")
    settings = {"sequence_length": args.sequence_length, "cut_percent": args.cut_percent,
                "validation_fraction": args.validation_fraction, "train_stride": args.train_stride}
    if args.target_start is not None:
        settings["target_start"] = args.target_start
    if validation_files is not None:
        settings.update(validation_files=validation_files, validation_fraction=None)
    arrays, _, normalization, dt = prepare_data(train_files, test_files, args.data_dir,
                                               recordings=args.persistent_state, **settings)
    print("Training pool:", train_files, flush=True)
    print("Test files:", test_files, flush=True)
    if validation_files is not None:
        print("Whole-case validation files:", validation_files, flush=True)
    counts = {name: len(y) for name, (_, y) in arrays.items()}
    print("Windows:", counts, "sample interval (s):", dt, flush=True)
    model_options = {}
    if args.model == "rnn":
        model = RNN()
    elif args.model == "lstm":
        model = LSTM()
    elif args.model == "gru":
        model = GRU()
    elif args.model == "cnn_gru":
        model = CNNGRU()
    else:
        model_options = {"latent_dim": args.latent_dim, "encoding_dim": args.encoding_dim,
                         "hidden_dim": args.hidden_dim, "sample_interval": dt,
                         "time_unit": args.time_unit, "method": args.ode_method,
                         "step_size": args.ode_step, "rtol": args.rtol, "atol": args.atol}
        model = PGNDModel(**model_options)
    residual_weight = args.residual_weight if isinstance(model, PGNDModel) and model.use_residual else 0.0
    checkpoint = {
        "model": args.model, "model_options": model_options,
        "protocol": ("case-validation-training-standardization-v1" if validation_files is not None
                     else "recording-split-training-standardization-v1"),
        "train_files": train_files, "test_files": test_files,
        "data_sha256": {name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
                        for name in train_files + (validation_files or []) + test_files},
        "data_settings": settings, "normalization": normalization, "sample_interval": dt,
        "window_counts": counts, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "threads": args.threads, "device": args.device,
        "preload_data": args.preload_data, "implementation": "core-models-missing-block-v1",
        "input_augmentation": {"kind": "causal-held-block-v1", "max_length": args.max_missing_block,
                               "clean_probability": 0.5 if args.max_missing_block else 1.0,
                               "seed": args.seed},
        "diagnostics": args.diagnostics,
        "diagnostic_probe": "64 evenly spaced training windows at each epoch end" if args.diagnostics else None,
        "checkpoint_selection": "minimum_validation_force_mse",
        "residual_weight": residual_weight,
        "optimizer": {"name": "Adam", "lr": args.learning_rate, "betas": (0.9, 0.999), "eps": 1e-7},
        "torch_version": str(torch.__version__),
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(Path(__file__).parent.glob("*.py"))},
    }
    if args.persistent_state:
        checkpoint["state_mode"] = "persistent"
        checkpoint["state_protocol"] = {
            "version": "recording-linear-history-held-forecast-v1",
            "warmup_observations": args.sequence_length,
            "gradient_chunk_intervals": args.sequence_length,
            "initialization": "original MLP once per recording partition",
            "time_origin": "recording partition start; never reset inside recording",
            "training_order": "ordered independent streams; exact target-count gradient accumulation",
            "residual_grid": "last L-1 assimilated states plus held-input forecast endpoint",
            "information_budget": "all earlier observations, not the window control's finite history",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Metadata is readable without PyTorch. Best weights survive interruption,
    # but these prediction checkpoints do not contain optimizer/resume state.
    with settings_path.open("x") as stream:
        json.dump(checkpoint, stream, indent=2)
        stream.write("\n")
    history = fit(model, *arrays["train"], *arrays["validation"], args.epochs,
                  args.batch_size, args.seed, args.device, args.learning_rate,
                  residual_weight, history_path=history_path, preload_data=args.preload_data,
                  checkpoint=checkpoint, best_path=best_path, diagnostics=args.diagnostics,
                  max_missing_block=args.max_missing_block, persistent_state=args.persistent_state,
                  context_length=args.sequence_length,
                  target_start=args.target_start or args.sequence_length, train_stride=args.train_stride)
    checkpoint.update(state_dict=model.cpu().state_dict(), history=history,
                      checkpoint_kind="final_epoch", selected_epoch=args.epochs,
                      completed_epochs=args.epochs, selection_metric=None,
                      selected_val_mse=history[-1]["val_mse_scaled"])
    save_checkpoint(checkpoint, args.output)
    print(f"Saved final-epoch weights and settings to {args.output}", flush=True)


if __name__ == "__main__":
    main()
