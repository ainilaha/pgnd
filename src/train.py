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

from src.baselines import CNNGRU, GRU, LSTM, RNN, DirectReadout
from src.data import DATA_DIR, get_files, prepare_data
from src.evaluate import mean_squared_error, predict
from src.model import PGNDModel


def save_checkpoint(checkpoint, path):
    """Replace only this run's checkpoint after a complete temporary write."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def fit(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32,
        seed=0, device="cpu", learning_rate=0.001, residual_weight=0.001,
        history_path=None, preload_data=False, checkpoint=None, best_path=None):
    """Adam with a fixed epoch budget and best-validation prediction-MSE saving.

    MSE is in standardized-force units. PGND adds the manuscript's residual
    penalty; baselines have no residual term. Minibatch order has its own seed.
    Write each completed epoch immediately, preserving history if interrupted.
    """
    if epochs < 1 or batch_size < 1 or residual_weight < 0:
        raise ValueError("Require positive epochs/batch size and nonnegative residual weight.")
    if (checkpoint is None) != (best_path is None):
        raise ValueError("Provide both checkpoint metadata and best_path, or neither.")
    device = torch.device(device)
    model.to(device)
    train_inputs = torch.as_tensor(X_train, dtype=torch.float32)
    train_targets = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    if preload_data:
        train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)
        X_val = torch.as_tensor(X_val, dtype=torch.float32, device=device)
        # Shuffle CPU indices with the same DataLoader generator/order as before.
        # Gather a whole GPU batch at once, not one GPU operation per sample.
        dataset = torch.arange(len(train_targets))
    else:
        dataset = TensorDataset(train_inputs, train_targets)
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
        for batch in loader:
            if preload_data:
                indices = batch.to(device, non_blocking=True)
                X_batch = train_inputs.index_select(0, indices)
                y_batch = train_targets.index_select(0, indices)
            else:
                X_batch, y_batch = (tensor.to(device, non_blocking=True) for tensor in batch)
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
        val_mse = mean_squared_error(y_val, predict(model, X_val, device=device))
        squared_error, residual_total = totals.cpu().tolist()
        row = {"epoch": epoch, "train_mse_scaled": squared_error / len(dataset),
               "residual_loss": residual_total / len(dataset), "val_mse_scaled": val_mse,
               "seconds": time.perf_counter() - start}
        row["train_total_loss"] = row["train_mse_scaled"] + residual_weight * row["residual_loss"]
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
                        choices=["rnn", "lstm", "gru", "cnn_gru", "pgnd", "pgnd_obs", "direct"])
    parser.add_argument("--train-pattern", required=True, help="Regex for training-pool filenames")
    parser.add_argument("--test-pattern", required=True, help="Regex for held-out test filenames")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, required=True,
                        help="New final .pt checkpoint path; also writes <stem>.best.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--cut-percent", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--preload-data", action="store_true",
                        help="Keep prepared training/validation inputs on the selected device")
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
    train_runs = {re.sub(r"_CutFre\d+", "", name) for name in train_files}
    test_runs = {re.sub(r"_CutFre\d+", "", name) for name in test_files}
    if train_runs & test_runs:
        warnings.warn("Train/test share speed/case at different cutoffs: bandwidth transfer, "
                      "not unseen-trajectory generalization (identical files are rejected).")
    settings = {"sequence_length": args.sequence_length, "cut_percent": args.cut_percent,
                "validation_fraction": args.validation_fraction, "train_stride": args.train_stride}
    arrays, _, normalization, dt = prepare_data(train_files, test_files, args.data_dir, **settings)
    print("Training pool:", train_files, flush=True)
    print("Test files:", test_files, flush=True)
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
    elif args.model == "direct":
        model_options = {"encoding_dim": args.encoding_dim, "hidden_dim": args.hidden_dim}
        model = DirectReadout(**model_options)
    else:
        model_options = {"latent_dim": args.latent_dim, "encoding_dim": args.encoding_dim,
                         "hidden_dim": args.hidden_dim, "sample_interval": dt,
                         "time_unit": args.time_unit, "method": args.ode_method,
                         "step_size": args.ode_step, "rtol": args.rtol, "atol": args.atol}
        if args.model == "pgnd_obs":
            model_options["observation_readout"] = True
        model = PGNDModel(**model_options)
    checkpoint = {
        "model": args.model, "model_options": model_options,
        "protocol": "recording-split-training-standardization-v1",
        "train_files": train_files, "test_files": test_files,
        "data_sha256": {name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
                        for name in train_files + test_files},
        "data_settings": settings, "normalization": normalization, "sample_interval": dt,
        "window_counts": counts, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "threads": args.threads, "device": args.device,
        "preload_data": args.preload_data, "implementation": "observation-readout-best-v1",
        "checkpoint_selection": "minimum_validation_force_mse",
        "residual_weight": args.residual_weight if isinstance(model, PGNDModel) else 0.0,
        "optimizer": {"name": "Adam", "lr": args.learning_rate, "betas": (0.9, 0.999), "eps": 1e-7},
        "torch_version": str(torch.__version__),
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(Path(__file__).parent.glob("*.py"))},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Metadata is readable without PyTorch. Best weights survive interruption,
    # but these prediction checkpoints do not contain optimizer/resume state.
    with settings_path.open("x") as stream:
        json.dump(checkpoint, stream, indent=2)
        stream.write("\n")
    history = fit(model, *arrays["train"], *arrays["validation"], args.epochs,
                  args.batch_size, args.seed, args.device, args.learning_rate,
                  args.residual_weight, history_path=history_path, preload_data=args.preload_data,
                  checkpoint=checkpoint, best_path=best_path)
    checkpoint.update(state_dict=model.cpu().state_dict(), history=history,
                      checkpoint_kind="final_epoch", selected_epoch=args.epochs,
                      completed_epochs=args.epochs, selection_metric=None,
                      selected_val_mse=history[-1]["val_mse_scaled"])
    save_checkpoint(checkpoint, args.output)
    print(f"Saved final-epoch weights and settings to {args.output}", flush=True)


if __name__ == "__main__":
    main()
