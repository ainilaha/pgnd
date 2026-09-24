"""Shared data preparation adapted from legacy/data_util.py.

This is the legacy numerical protocol, including its known limitations:
header=0 on headerless XLS files, concatenation before windowing, no scaling,
and next-sample targets. See legacy/README.md before using it for new results.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "Data"
COLUMNS = ["distance", "acceleration", "displacement", "force"]


def get_files(pattern, path=DATA_DIR):
    """Match a legacy filename regex; sort for a reproducible file order.

    Sorting differs from the original os.listdir order and can change the
    validation tail. Pass an explicit ordered list to read_data to replay it.
    """
    files = sorted(p.name for p in Path(path).iterdir()
                   if p.is_file() and re.match(pattern, p.name))
    if not files:
        raise ValueError(f"No files match {pattern!r} in {path}")
    return files


def read_data(file_list, path=DATA_DIR, cut_percent=0.05):
    """Read four-column XLS files, trim each end, concatenate and drop NaNs.

    cut_percent retains the meaning of the legacy misspelling cut_precent.
    The notebooks use 0.1; the original helper default is 0.05.
    """
    if not file_list:
        raise ValueError("Provide at least one data file.")
    if not 0 <= cut_percent < 0.5:
        raise ValueError("cut_percent must be in [0, 0.5).")
    frames = []
    for filename in file_list:
        # Deliberate compatibility: the old reader consumes row 1 as a header.
        frame = pd.read_excel(Path(path) / filename, header=0)
        if frame.shape[1] != 4:
            raise ValueError(f"{filename}: expected the four-column legacy XLS data.")
        frame.columns = COLUMNS
        cut = int(len(frame) * cut_percent)
        frames.append(frame.iloc[cut:len(frame) - cut])
    frame = pd.concat(frames, ignore_index=True).dropna().reset_index(drop=True)
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("The selected data are empty or contain non-finite values.")
    return frame


def create_sequences(features, target, index, sequence_length=16):
    """Return X[i:i+L], force[i+L], distance[i+L], without normalization.

    The old MinMaxScaler result was unused; removing that dead calculation
    leaves the returned values unchanged. Inputs have shape (N, L, 2).
    """
    features, target, index = map(np.asarray, (features, target, index))
    if features.ndim != 2 or target.ndim != 1 or index.ndim != 1:
        raise ValueError("Expected 2-D features and 1-D targets and indices.")
    if not len(features) == len(target) == len(index):
        raise ValueError("Features, targets and indices must have equal lengths.")
    if sequence_length < 1 or len(features) <= sequence_length:
        raise ValueError("sequence_length must be positive and shorter than the data.")
    windows = [features[i:i + sequence_length]
               for i in range(len(features) - sequence_length)]
    return (np.asarray(windows), target[sequence_length:].copy(),
            index[sequence_length:].copy())


def load_sequences(file_list, path=DATA_DIR, cut_percent=0.1, sequence_length=16):
    """Shared loader for every baseline; retain legacy cross-recording windows."""
    warnings.warn(
        "Legacy protocol: header=0 discards the first XLS sample; no scaling is applied. "
        "Files are concatenated before windowing, so windows can cross recording "
        "boundaries. See legacy/README.md.", UserWarning, stacklevel=2,
    )
    frame = read_data(file_list, path, cut_percent)
    return create_sequences(frame[["acceleration", "displacement"]],
                            frame["force"], frame["distance"], sequence_length)


def split_sequences(X, y, validation_fraction=0.15):
    """Reproduce Keras validation_split: reserve the final fraction of windows.

    No random split or temporal gap is introduced. Neighboring training and
    validation windows overlap in their inputs; this is not a clean holdout.
    """
    if len(X) != len(y) or not 0 < validation_fraction < 1:
        raise ValueError("Require aligned arrays and validation_fraction in (0, 1).")
    split = int(len(X) * (1 - validation_fraction))
    if not 0 < split < len(X):
        raise ValueError("The split must leave nonempty training and validation sets.")
    warnings.warn("Legacy validation tail has overlapping input windows across the "
                  "split boundary; validation MSE is not an independent holdout estimate.",
                  UserWarning, stacklevel=2)
    return X[:split], y[:split], X[split:], y[split:]
