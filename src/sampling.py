"""Shared, signal-independent observation masks; never modify complete recordings."""

import hashlib

import numpy as np
import pandas as pd

from src.data import INPUTS, UNITS, normalize


def observation_mask(recording, retention, pattern, *, seed=0, burst_length=8):
    """One indexed boolean mask for BOTH sensors and ALL compared models.

    Retain both endpoints, then remove round(N*(1-retention)) interior rows. Random
    thinning samples without replacement. Bursty dropout places nonoverlapping
    bursts separated by >=1 observed row: burst_length rows each, with one
    shorter remainder if necessary. Reject infeasible counts on tiny segments.
    Only identity, row span, seed and pattern seed the RNG, never signal values.
    """
    rows = recording.samples.index
    n = len(rows)
    if (n < 2 or not rows.is_unique or not (np.diff(rows.to_numpy()) == 1).all()
            or retention not in (1., .8, .6) or pattern not in ("random", "bursty")
            or not isinstance(seed, (int, np.integer)) or seed < 0
            or not isinstance(burst_length, (int, np.integer)) or burst_length < 1):
        raise ValueError("Need consecutive recording rows, retention 1/.8/.6, random/bursty, "
                         "nonnegative integer seed and positive integer burst_length.")
    identity = f"{recording.name}:{rows[0]}:{n}:{seed}:{pattern}"
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(
        hashlib.sha256(identity.encode()).digest()[:8], "little")))
    mask = np.ones(n, dtype=bool)
    removed = round(n * (1 - retention))
    if removed > n - 2:
        raise ValueError("Segment too short for this retention with both endpoints retained.")
    if pattern == "random":
        mask[1 + rng.permutation(n - 2)[:removed]] = False
    elif removed:
        lengths = np.full((removed + burst_length - 1) // burst_length, burst_length)
        lengths[-1] = removed - burst_length * (len(lengths) - 1)
        rng.shuffle(lengths)
        # Attach bursts after distinct retained rows, excluding the final one.
        # This preserves both endpoints and a retained separator between bursts.
        if len(lengths) > n - removed - 1:
            raise ValueError("Segment too short for separated bursts and retained endpoints.")
        anchors = np.sort(rng.choice(n - removed - 1, size=len(lengths), replace=False))
        starts = anchors + 1 + np.r_[0, np.cumsum(lengths[:-1])]
        for start, length in zip(starts, lengths):
            mask[start:start + length] = False
    result = pd.Series(mask, index=rows.copy(), name=recording.name)
    result.attrs = {"split": recording.split, "retention": retention, "pattern": pattern,
                    "seed": int(seed), "burst_length": int(burst_length)}
    return result


def validate_mask(recording, mask):
    """Reject mismatched recording/segment/split masks or missing endpoints."""
    if (not isinstance(mask, pd.Series) or mask.dtype != bool or mask.empty
            or mask.name != recording.name or mask.attrs.get("split") != recording.split
            or not mask.index.equals(recording.samples.index)
            or not mask.iloc[0] or not mask.iloc[-1]):
        raise ValueError("Mask must match this recording's identity, split and rows, "
                         "with the first and last observations retained.")


def retained_observations(recording, mask, statistics):
    """Retained distance, acceleration and uplift only; no filling or force input.

    Only acceleration/uplift are normalized, using the SAME training statistics
    as the baseline. Original distances remain exact integration coordinates;
    compute intervals from them when needed, never store gaps as model features.
    Complete distance/force rows stay in recording.
    """
    validate_mask(recording, mask)
    x, _ = normalize(recording, statistics)
    observed = recording.samples.loc[mask, ["distance", *INPUTS]].copy()
    observed[INPUTS] = x[mask.to_numpy()]
    observed.attrs = {"file": recording.name, "split": recording.split,
                      "units": {"distance": UNITS["distance"],
                                "acceleration": "standardized", "uplift": "standardized"}}
    return observed
