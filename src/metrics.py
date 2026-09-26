"""Scalar regression metrics in the units supplied by the caller."""

import numpy as np


def regression_metrics(target, prediction):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.size == 0 or target.shape != prediction.shape:
        raise ValueError("Require nonempty, equally sized targets and predictions.")
    if not (np.isfinite(target).all() and np.isfinite(prediction).all()):
        raise ValueError("Targets and predictions must be finite.")
    error = target - prediction
    mse = float(np.mean(error ** 2))
    variance = float(np.var(target))
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(mse)),
            "mse": mse, "r2": 1 - mse / variance if variance > 0 else float("nan")}
