"""Train the paper baselines with a shared, explicitly documented legacy protocol.

PGND remains unimplemented. File selections are required because the paper
and notebooks disagree about some experiments; no strategy is guessed here.
"""

import argparse
import hashlib
from pathlib import Path
import re
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines import CNNGRU, GRU, LSTM, RNN
from src.data import DATA_DIR, get_files, load_sequences, split_sequences
from src.evaluate import mean_squared_error, predict


def fit(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32,
        seed=0, device="cpu"):
    """Shared window-to-force training loop: Adam, MSE, final-epoch weights.

    The seed controls minibatch order independently of architecture. Seed
    PyTorch before constructing the model to also seed its initialization.
    """
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if len(X_train) != len(y_train) or len(X_train) == 0:
        raise ValueError("Require nonempty, aligned training inputs and targets.")
    model.to(device)
    dataset = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32),
                            torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=0.001, betas=(0.9, 0.999), eps=1e-7)
    loss_fn = nn.MSELoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        squared_error = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            if not torch.isfinite(loss):
                raise ValueError("Training loss became non-finite.")
            loss.backward()
            optimizer.step()
            squared_error += loss.item() * len(X_batch)
        val_mse = mean_squared_error(y_val, predict(model, X_val, device=device))
        row = {"epoch": epoch, "loss": squared_error / len(dataset), "val_loss": val_mse}
        history.append(row)
        print(f"Epoch {epoch}/{epochs}: loss={row['loss']:.6g}, val_loss={val_mse:.6g}")
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["rnn", "lstm", "gru", "cnn_gru"])
    parser.add_argument("--train-pattern", required=True, help="Regex for training-pool filenames")
    parser.add_argument("--test-pattern", required=True, help="Regex for held-out test filenames")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, required=True, help="New .pt checkpoint path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--cut-percent", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output path: {args.output}")
    train_files = get_files(args.train_pattern, args.data_dir)
    test_files = get_files(args.test_pattern, args.data_dir)
    if set(train_files) & set(test_files):
        raise ValueError("The training pool and test set contain the same files.")
    train_runs = {re.sub(r"_CutFre\d+", "", name) for name in train_files}
    test_runs = {re.sub(r"_CutFre\d+", "", name) for name in test_files}
    if train_runs & test_runs:
        warnings.warn("Train/test include the same speed and case at different cutoffs. "
                      "This tests bandwidth transfer, not unseen-trajectory generalization.")
    print("Training pool (ordered):", train_files)
    print("Held-out test files (ordered):", test_files)
    X, y, _ = load_sequences(train_files, args.data_dir, args.cut_percent, args.sequence_length)
    X_train, y_train, X_val, y_val = split_sequences(X, y, args.validation_fraction)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.model == "rnn":
        model = RNN()
    elif args.model == "lstm":
        model = LSTM()
    elif args.model == "gru":
        model = GRU()
    else:
        model = CNNGRU()
    history = fit(model, X_train, y_train, X_val, y_val, args.epochs,
                  args.batch_size, args.seed, args.device)
    checkpoint = {
        "model": args.model, "state_dict": model.cpu().state_dict(),
        "protocol": "legacy-concatenated-unscaled-header0-sorted",
        "train_files": train_files, "test_files": test_files,
        "data_sha256": {name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
                        for name in train_files + test_files},
        "sequence_length": args.sequence_length, "cut_percent": args.cut_percent,
        "validation_fraction": args.validation_fraction, "split_index": len(X_train),
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "optimizer": {"name": "Adam", "lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-7},
        "torch_version": str(torch.__version__), "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Saved final-epoch baseline and experiment metadata to {args.output}")


if __name__ == "__main__":
    main()
