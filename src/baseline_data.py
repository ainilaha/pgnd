"""Thin regular-window adapter. Raw recordings remain complete and model-neutral."""

import numpy as np

from src.data import INPUTS, Recording, normalize
from src.sampling import validate_mask


def trim_recording(recording, fraction):
    """Explicit historical baseline trim: floor(N*fraction) rows at EACH end."""
    if not 0 <= fraction < 0.5:
        raise ValueError("Trim fraction must be in [0, 0.5).")
    n = len(recording.samples)
    cut = int(n * fraction)
    return Recording(recording.name, recording.split, recording.samples.iloc[cut:n - cut].copy())


def linear_interpolate(recording, mask):
    """Interpolate raw sensors between retained observations at original times.

    Both endpoints must be retained; no extrapolation or force values are used.
    This recording-level preprocessing uses right-hand (future) observations,
    so the irregular baseline input is non-causal even with preceding windows.
    """
    validate_mask(recording, mask)
    t = recording.samples["time_s"].to_numpy(dtype=float)
    if not np.isfinite(t).all() or not (np.diff(t) > 0).all():
        raise ValueError("Interpolation requires finite, strictly increasing timestamps.")
    keep = mask.to_numpy()
    filled = recording.samples[INPUTS].copy()
    for column in INPUTS:
        values = filled.loc[mask, column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Retained sensor observations must be finite.")
        if not keep.all():
            filled.loc[~mask, column] = np.interp(t[~keep], t[keep], values)
    return filled


def make_windows(recording, statistics, sequence_length, stride=1, *, mask=None):
    """X[k-L:k] -> F[k], excluding target-time input rows and all force history.

    Call separately for each already-split recording. L and stride are explicit
    adapter settings, not properties of the canonical dataset. Returns float32
    X/y and unscaled target rows with recording/split identity and true row IDs.
    An optional shared mask linearly interpolates sensors BEFORE windowing.
    Interpolation can use retained sensors at/after the force target time.
    Statistics are fixed from complete training data, never refit after masking.
    """
    frame = recording.samples
    if (not isinstance(sequence_length, int) or not isinstance(stride, int)
            or not 2 <= sequence_length < len(frame) or stride < 1):
        raise ValueError("Require integer 2 <= L < recording rows and stride >= 1.")
    dt = np.diff(frame["time_s"].to_numpy())
    if not (dt > 0).all() or not np.allclose(dt, dt[0], rtol=1e-4, atol=1e-9):
        raise ValueError("Conventional baseline windows require a regular time grid.")
    if not (np.diff(frame.index.to_numpy()) == 1).all():
        raise ValueError("Baseline windows must not bridge omitted source rows.")
    x, y = normalize(recording, statistics)
    if mask is not None:
        filled = linear_interpolate(recording, mask).to_numpy(dtype=float)
        x = (filled - statistics["input_mean"]) / statistics["input_std"]
    positions = np.arange(sequence_length, len(frame), stride)
    windows = np.stack([x[k - sequence_length:k] for k in positions]).astype(np.float32)
    targets = frame.iloc[positions][["time_s", "distance_m", "force_N"]].reset_index()
    targets.insert(0, "file", recording.name)
    targets.insert(1, "split", recording.split)
    return windows, y[positions].astype(np.float32), targets
