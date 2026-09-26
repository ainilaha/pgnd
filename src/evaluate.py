"""Clean baseline inference and force metrics. No training or experimental modes."""

import pandas as pd
import torch

from src.baseline_data import make_windows, trim_recording
from src.metrics import regression_metrics


def predict(model, x, batch_size=1024, device="cpu"):
    """Preserve input order; the established models reset state per window."""
    if len(x) == 0 or batch_size < 1:
        raise ValueError("Require nonempty inputs and a positive batch size.")
    model.to(device).eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            predictions.append(model(batch).reshape(-1).cpu())
    return torch.cat(predictions).numpy()


def evaluate(model, recordings, statistics, sequence_length, *, cut_percent,
             batch_size=1024, device="cpu"):
    """Score every eligible target (stride 1) in one split, in physical newtons.

    Statistics must come from training, not these held-out recordings. Explicit
    trimming belongs to the baseline adapter; it never alters loaded recordings.
    Returns pooled metrics and a prediction table with original target provenance.
    """
    if (not recordings or len({r.split for r in recordings}) != 1
            or len({r.name for r in recordings}) != len(recordings)):
        raise ValueError("Require distinct recordings from one split.")
    predictions = []
    for recording in recordings:
        x, _, targets = make_windows(trim_recording(recording, cut_percent), statistics, sequence_length)
        scaled = predict(model, x, batch_size, device)
        targets["prediction_N"] = scaled * statistics["force_std"] + statistics["force_mean"]
        predictions.append(targets)
    predictions = pd.concat(predictions, ignore_index=True)
    scores = regression_metrics(predictions["force_N"], predictions["prediction_N"])
    return scores, predictions
