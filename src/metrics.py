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


def interpolation_metrics(original, interpolated, observed):
    """One sensor channel, physical units, scored only at removed observations.

    Retained values must match exactly. With no removed rows, errors are
    undefined (NaN), not zero. This is not a contact-force prediction metric.
    """
    original = np.asarray(original, dtype=np.float64)
    interpolated = np.asarray(interpolated, dtype=np.float64)
    observed = np.asarray(observed)
    if (original.ndim != 1 or original.size == 0
            or interpolated.shape != original.shape or observed.shape != original.shape
            or observed.dtype != np.bool_):
        raise ValueError("Require aligned, nonempty 1D signals and a boolean observation mask.")
    if not (np.isfinite(original).all() and np.isfinite(interpolated).all()):
        raise ValueError("Sensor values must be finite.")
    if not np.array_equal(original[observed], interpolated[observed]):
        raise ValueError("Interpolation must not change retained observations.")
    removed = int((~observed).sum())
    if not removed:
        return {"removed": 0, "mae": float("nan"), "rmse": float("nan")}
    scores = regression_metrics(original[~observed], interpolated[~observed])
    return {"removed": removed, "mae": scores["mae"], "rmse": scores["rmse"]}
