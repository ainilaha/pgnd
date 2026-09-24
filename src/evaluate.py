"""Evaluate PyTorch baselines using the legacy force MSE in N^2."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.baselines import CNNGRU, GRU, LSTM, RNN
from src.data import DATA_DIR, load_sequences


def mean_squared_error(target, prediction):
    """Same scalar-force metric as the legacy sklearn call; no rescaling."""
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.size == 0 or target.shape != prediction.shape:
        raise ValueError("Require nonempty, equally sized targets and predictions.")
    if not (np.isfinite(target).all() and np.isfinite(prediction).all()):
        raise ValueError("Targets and predictions must be finite.")
    return float(np.mean((target - prediction) ** 2))


def predict(model, X, batch_size=1024, device="cpu"):
    """Predict in sample order without carrying state between windows."""
    if len(X) == 0 or batch_size < 1:
        raise ValueError("Require nonempty inputs and a positive batch size.")
    model.to(device).eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start:start + batch_size],
                                    dtype=torch.float32, device=device)
            predictions.append(model(batch).cpu().numpy().reshape(-1))
    return np.concatenate(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    name = checkpoint["model"]
    if name == "rnn":
        model = RNN()
    elif name == "lstm":
        model = LSTM()
    elif name == "gru":
        model = GRU()
    elif name == "cnn_gru":
        model = CNNGRU()
    else:
        raise ValueError(f"Unsupported baseline: {name}")
    model.load_state_dict(checkpoint["state_dict"])
    for filename in checkpoint["test_files"]:
        actual = hashlib.sha256((args.data_dir / filename).read_bytes()).hexdigest()
        if actual != checkpoint["data_sha256"][filename]:
            raise ValueError(f"Data changed since training: {filename}")
    X, y, _ = load_sequences(checkpoint["test_files"], args.data_dir,
                             checkpoint["cut_percent"], checkpoint["sequence_length"])
    prediction = predict(model, X, device=args.device)
    print(json.dumps({"model": name, "samples": len(y),
                      "mse_N2": mean_squared_error(y, prediction),
                      "test_files": checkpoint["test_files"]}, indent=2))


if __name__ == "__main__":
    main()
