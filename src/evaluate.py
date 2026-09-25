"""Compare PGND and baselines on identical held-out targets, in physical force units."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.baselines import CNNGRU, GRU, LSTM, RNN
from src.data import DATA_DIR, prepare_data
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
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new results directory: {args.output_dir}")
    if len({path.stem for path in args.checkpoints}) != len(args.checkpoints):
        raise ValueError("Checkpoint filenames need distinct stems for prediction CSVs.")
    torch.set_num_threads(args.threads)
    checkpoints = [torch.load(path, map_location="cpu", weights_only=True) for path in args.checkpoints]
    reference = checkpoints[0]
    if reference["protocol"] != "recording-split-training-standardization-v1":
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
        for field in ("protocol", "train_files", "test_files", "data_settings", "normalization",
                      "data_sha256", "sample_interval", "seed", "epochs", "batch_size", "optimizer"):
            if checkpoint[field] != reference[field]:
                raise ValueError(f"Checkpoints differ in {field}; not the same experiment.")
    for filename, expected in reference["data_sha256"].items():
        if hashlib.sha256((args.data_dir / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Data changed since training: {filename}")
    arrays, metadata, normalization, _ = prepare_data(
        reference["train_files"], reference["test_files"], args.data_dir,
        normalization=reference["normalization"], **reference["data_settings"])
    X, _ = arrays[args.split]
    if args.preload_data:
        X = torch.as_tensor(X, dtype=torch.float32, device=args.device)
    target = metadata[args.split]["force_N"].to_numpy()
    args.output_dir.mkdir(parents=True)
    results, recording_results = [], []
    for path, checkpoint in zip(args.checkpoints, checkpoints):
        name = checkpoint["model"]
        if name == "rnn":
            model = RNN()
        elif name == "lstm":
            model = LSTM()
        elif name == "gru":
            model = GRU()
        elif name == "cnn_gru":
            model = CNNGRU()
        elif name in ("pgnd", "pgnd_obs"):
            model = PGNDModel(**checkpoint["model_options"])
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
        prediction = predict(model, X, device=args.device).astype(np.float64)
        prediction = prediction * normalization["force_std"] + normalization["force_mean"]
        row = {"run": path.stem, "model": name, "seed": checkpoint["seed"],
               "split": args.split, "samples": len(target),
               "checkpoint_kind": checkpoint.get("checkpoint_kind", "final_epoch"),
               "selected_epoch": checkpoint.get("selected_epoch", checkpoint["epochs"]),
               "validation_mse_scaled": checkpoint.get(
                   "selected_val_mse", checkpoint["history"][-1]["val_mse_scaled"]),
               "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               **force_metrics(target, prediction)}
        results.append(row)
        pd.DataFrame([row]).to_csv(args.output_dir / f"{path.stem}.metrics.csv", index=False)
        # Bundle the checkpoint's actual history, not an unrelated sidecar file.
        pd.DataFrame(checkpoint["history"]).to_csv(
            args.output_dir / f"{path.stem}.history.csv", index=False)
        predictions = metadata[args.split].copy()
        predictions["prediction_N"] = prediction
        predictions.to_csv(args.output_dir / f"{path.stem}.predictions.csv", index=False)
        for filename, recording in predictions.groupby("file", sort=False):
            recording_results.append({
                "run": path.stem, "model": name, "split": args.split,
                "file": filename, "samples": len(recording),
                **force_metrics(recording["force_N"], recording["prediction_N"]),
            })
    table = pd.DataFrame(results)
    table.to_csv(args.output_dir / "comparison.csv", index=False)
    pd.DataFrame(recording_results).to_csv(args.output_dir / "per_recording.csv", index=False)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
