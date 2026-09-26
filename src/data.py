"""Complete simulated recordings and training-only standardization. No windows."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "Data"
INPUTS = ["acceleration", "uplift"]
FILENAME = re.compile(r"V(300|350|380)_Case([1-8])_CutFre(20|50|100|150|200)\.xls")


@dataclass(frozen=True)
class Recording:
    """One recording; samples retain original zero-based Excel rows as their index.

    Columns: distance_m, acceleration, uplift, force_N, time_s. Treat samples
    as read-only. Adapters return separate copies instead of changing raw data.
    """

    name: str
    split: str
    samples: pd.DataFrame


def recording_info(name):
    """Established whole-case split, shared across speeds and cutoff variants."""
    match = FILENAME.fullmatch(name)
    if match is None:
        raise ValueError(f"Unrecognized simulation filename: {name}")
    speed, case, cutoff = map(int, match.groups())
    split = "train" if case <= 4 else "validation" if case <= 6 else "test"
    return speed, case, cutoff, split


def load_recording(path):
    """Read ALL rows without trimming, filtering, resampling or dropping values.

    Files contain distance, not a clock. time_s = distance_m / (speed_kmh/3.6)
    assumes constant speed and the filename's units. Keep the original origin.
    """
    path = Path(path)
    speed, _, _, split = recording_info(path.name)
    frame = pd.read_excel(path, header=None)
    if frame.shape[1] != 4 or len(frame) < 2:
        raise ValueError(f"{path.name}: expected at least two rows and four headerless columns.")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{path.name}: non-finite values; raw rows must not be discarded.")
    frame.columns = ["distance_m", *INPUTS, "force_N"]
    frame.index.name = "source_row"
    frame["time_s"] = frame["distance_m"].to_numpy(dtype=float) / (speed / 3.6)
    if not (np.diff(frame["time_s"]) > 0).all():
        raise ValueError(f"{path.name}: timestamps must be strictly increasing.")
    return Recording(path.name, split, frame)


def load_recordings(path=DATA_DIR, *, cutoff):
    """Choose one supplied cutoff explicitly; never concatenate recording rows."""
    files = sorted(Path(path).glob(f"*_CutFre{cutoff}.xls"))
    if not files:
        raise ValueError(f"No CutFre{cutoff} recordings in {path}")
    return [load_recording(file) for file in files]


def fit_normalization(recordings):
    """Population mean/std over supplied TRAIN rows, before any window sampling.

    To reproduce a trimmed baseline, pass its explicitly trimmed training
    recordings. Validation/test records and duplicate identities are rejected.
    """
    if not recordings or any(r.split != "train" for r in recordings):
        raise ValueError("Fit normalization on nonempty training recordings only.")
    if len({r.name for r in recordings}) != len(recordings):
        raise ValueError("Duplicate training recording identities.")
    rows = pd.concat([r.samples for r in recordings])
    x, y = rows[INPUTS].to_numpy(dtype=float), rows["force_N"].to_numpy(dtype=float)
    if not len(y) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Training rows must be nonempty and finite.")
    return {"input_mean": x.mean(0).tolist(),
            "input_std": np.maximum(x.std(0), 1e-12).tolist(),
            "force_mean": float(y.mean()), "force_std": max(float(y.std()), 1e-12)}


def normalize(recording, statistics):
    """Return aligned float64 input/force arrays; never fit on evaluation rows."""
    mean, std = (np.asarray(statistics[key], dtype=float) for key in ("input_mean", "input_std"))
    force_mean, force_std = statistics["force_mean"], statistics["force_std"]
    if (mean.shape != (2,) or std.shape != (2,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)
            or not np.isfinite(force_mean) or not np.isfinite(force_std) or force_std <= 0):
        raise ValueError("Expected finite means and positive standard deviations.")
    x = (recording.samples[INPUTS].to_numpy(dtype=float) - mean) / std
    y = (recording.samples["force_N"].to_numpy(dtype=float) - force_mean) / force_std
    return x, y
