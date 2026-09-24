"""Shared, recording-safe preprocessing for PGND and the PyTorch baselines.

Retains legacy per-end trimming and next-sample targets. Unlike the legacy
protocol, reads headerless data correctly, splits before windowing, never
joins recordings, and standardizes using training rows only. See README.md.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "Data"
COLUMNS = ["distance", "acceleration", "displacement", "force"]


def get_files(pattern, path=DATA_DIR):
    """Select legacy filenames by regex, with reproducible ordering."""
    files = sorted(p.name for p in Path(path).iterdir()
                   if p.is_file() and re.match(pattern, p.name))
    if not files:
        raise ValueError(f"No files match {pattern!r} in {path}")
    return files


def read_recording(filename, path=DATA_DIR, cut_percent=0.1):
    """Read original values, retaining zero-based source rows as the index."""
    if not 0 <= cut_percent < 0.5:
        raise ValueError("cut_percent must be in [0, 0.5).")
    frame = pd.read_excel(Path(path) / filename, header=None)
    if frame.shape[1] != 4:
        raise ValueError(f"{filename}: expected four headerless columns.")
    frame.columns = COLUMNS
    cut = int(len(frame) * cut_percent)
    frame = frame.iloc[cut:len(frame) - cut]
    # Reject gaps rather than silently dropping rows and closing time intervals.
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{filename}: empty or non-finite data.")
    return frame


def create_sequences(features, target, index, sequence_length=16, stride=1):
    """X[i:i+L] -> force[i+L]; stride subsamples targets, not observations."""
    features, target, index = map(np.asarray, (features, target, index))
    if not len(features) == len(target) == len(index):
        raise ValueError("Features, targets and indices must have equal lengths.")
    if sequence_length < 2 or stride < 1 or len(features) <= sequence_length:
        raise ValueError("Require L >= 2, stride >= 1 and more than L rows.")
    starts = np.arange(0, len(features) - sequence_length, stride)
    windows = np.stack([features[i:i + sequence_length] for i in starts])
    return windows, target[starts + sequence_length], index[starts + sequence_length]


def prepare_data(train_files, test_files, path=DATA_DIR, sequence_length=16,
                 cut_percent=0.1, validation_fraction=0.15, train_stride=1,
                 normalization=None):
    """Return split arrays, target provenance, training statistics and dt.

    Each trimmed training recording reserves its final fraction of ROWS for
    validation. Windows are built separately inside each partition. Test
    recordings are held out entirely. Validation/test always use stride 1.
    Physical dt is inferred from distance[m]/speed[m/s] and must be uniform;
    constant speed and the filename units are explicit dataset assumptions.
    """
    if not train_files or not test_files or set(train_files) & set(test_files):
        raise ValueError("Require nonempty, disjoint train/test file lists.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1).")
    segments = {"train": [], "validation": [], "test": []}
    intervals = []
    for filename in train_files + test_files:
        frame = read_recording(filename, path, cut_percent)
        speed = float(re.match(r"V(\d+)_", filename).group(1)) / 3.6
        dt = np.diff(frame["distance"].to_numpy()) / speed
        if not (dt > 0).all() or not np.allclose(dt, dt[0], rtol=1e-4, atol=1e-9):
            raise ValueError(f"{filename}: the common fixed-grid experiment needs uniform dt.")
        intervals.append(float(np.median(dt)))
        if filename in train_files:
            split = int(len(frame) * (1 - validation_fraction))
            segments["train"].append((filename, frame.iloc[:split]))
            segments["validation"].append((filename, frame.iloc[split:]))
        else:
            segments["test"].append((filename, frame))
    if not np.allclose(intervals, intervals[0], rtol=1e-4, atol=1e-9):
        raise ValueError("Recordings have different sample intervals.")
    sample_interval = round(float(np.median(intervals)), 9)
    if normalization is None:
        training_rows = pd.concat([frame for _, frame in segments["train"]])
        x = training_rows[["acceleration", "displacement"]].to_numpy(dtype=float)
        y = training_rows["force"].to_numpy(dtype=float)
        normalization = {"input_mean": x.mean(0).tolist(),
                         "input_std": np.maximum(x.std(0), 1e-12).tolist(),
                         "force_mean": float(y.mean()),
                         "force_std": max(float(y.std()), 1e-12)}
    arrays, metadata = {}, {}
    for split_name, recordings in segments.items():
        windows, targets, provenance = [], [], []
        stride = train_stride if split_name == "train" else 1
        for filename, frame in recordings:
            features = frame[["acceleration", "displacement"]].to_numpy(dtype=float)
            features = (features - normalization["input_mean"]) / normalization["input_std"]
            force = frame["force"].to_numpy(dtype=float)
            scaled_force = (force - normalization["force_mean"]) / normalization["force_std"]
            X, y, rows = create_sequences(features, scaled_force, frame.index,
                                          sequence_length, stride)
            windows.append(X)
            targets.append(y)
            provenance.append(pd.DataFrame({
                "file": filename, "source_row": rows,
                "distance_m": frame["distance"].to_numpy()[sequence_length::stride],
                "force_N": force[sequence_length::stride],
            }))
        arrays[split_name] = (np.concatenate(windows).astype(np.float32),
                             np.concatenate(targets).astype(np.float32))
        metadata[split_name] = pd.concat(provenance, ignore_index=True)
    return arrays, metadata, normalization, sample_interval
