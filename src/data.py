"""Shared, recording-safe preprocessing for PGND and the PyTorch baselines.

Retains legacy per-end trimming and next-sample targets. Unlike the legacy
protocol, reads headerless data correctly, splits before windowing, never
joins recordings, and standardizes using training rows only. See README.md.
"""

from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd
import torch


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "Data"
COLUMNS = ["distance", "acceleration", "displacement", "force"]


def observation_mask(rows, retention, removal="random", seed=0, recording=""):
    """Signal-independent, nested recording-level sensor masks for the pilot.

    Random removal ranks individual rows. Bursty removal ranks 8-row blocks,
    one randomly positioned candidate per 16-row cell (adjacent bursts can
    merge). The last block can be partial to give the exact rounded count.
    Neither force nor sensor values enter this function. Both channels share
    the mask; 60% retained rows are a subset of 80% for the same seed/file.
    """
    if rows < 2 or not 0.6 <= retention <= 1 or removal not in ("random", "bursty"):
        raise ValueError("Pilot requires >= 2 rows, retention in [0.6, 1], random/bursty.")
    identity = int.from_bytes(hashlib.sha256(recording.encode()).digest()[:8], "little")
    rng = np.random.default_rng(np.random.SeedSequence([seed, identity]))
    count = round(rows * (1 - retention))
    if removal == "random":
        order = rng.permutation(rows)
    else:
        blocks = [np.arange(start + offset, start + offset + 8)
                  for start, offset in zip(range(0, rows - 15, 16),
                                           rng.integers(0, 9, size=rows // 16))]
        order = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))]) if blocks else np.array([], dtype=int)
        if count > len(order):
            raise ValueError("Recording too short for the bounded-burst pilot.")
    mask = np.ones(rows, dtype=bool)
    mask[order[:count]] = False
    return mask


def compact_observations(x, mask, sample_interval):
    """Pack retained sensor rows and true relative seconds; padding is not data.

    x contains historical sensors only, never target-time sensors or force.
    The returned lengths exclude padding. Missing sensor values are never
    returned, interpolated or passed to the PGND encoder.
    """
    x, mask = np.asarray(x), np.asarray(mask)
    if (x.ndim != 3 or x.shape[-1] != 2 or mask.shape != x.shape[:2]
            or mask.dtype != bool or sample_interval <= 0 or np.any(mask.sum(1) < 2)):
        raise ValueError("Need aligned boolean masks, positive dt and >=2 retained observations.")
    lengths = mask.sum(1)
    positions = np.broadcast_to(np.arange(x.shape[1]), mask.shape)
    order = np.sort(np.where(mask, positions, x.shape[1]), axis=1)[:, :lengths.max()]
    valid = np.arange(order.shape[1])[None, :] < lengths[:, None]
    values = np.take_along_axis(x, order.clip(max=x.shape[1] - 1)[..., None], axis=1)
    values = np.where(valid[..., None], values, 0)
    times = np.where(valid, order * sample_interval, np.inf).astype(x.dtype)
    return values, times, lengths


def interpolate_observations(x, mask):
    """CNN input: prefix-local linear interpolation, endpoint hold at boundaries.

    All endpoints must already be available BEFORE the force target. Do not
    call this on a complete recording and then window it: that could borrow
    target/future sensors. Masked values have no influence on the result.
    """
    x, mask = np.asarray(x), np.asarray(mask)
    if (x.ndim != 3 or x.shape[-1] != 2 or mask.shape != x.shape[:2]
            or mask.dtype != bool or np.any(mask.sum(1) < 2)):
        raise ValueError("Need historical windows with >=2 retained observations.")
    positions = np.broadcast_to(np.arange(x.shape[1]), mask.shape)
    left = np.maximum.accumulate(np.where(mask, positions, -1), axis=1)
    right = np.minimum.accumulate(np.where(mask, positions, x.shape[1])[:, ::-1], axis=1)[:, ::-1]
    left = np.where(left < 0, right, left)
    right = np.where(right == x.shape[1], left, right)
    lo, hi = (np.take_along_axis(x, i[..., None], axis=1) for i in (left, right))
    weight = ((positions - left) / np.maximum(right - left, 1))[..., None]
    return (lo + weight * (hi - lo)).astype(x.dtype)


def hold_missing_block(x, starts, lengths):
    """Hide one contiguous two-channel block per window; hold its preceding value.

    x is a standardized (batch, time, 2) tensor. Starts/lengths are CPU arrays
    or scalars. Keep the first sample observed, never borrow future values,
    never change targets/time, and never mutate x. Zero lengths are a no-op.
    This models filled fixed-grid outages, NOT native irregular sampling.
    """
    if x.ndim != 3 or x.shape[-1] != 2 or x.shape[1] < 2:
        raise ValueError("Expected (batch, at least 2 observed steps, 2).")
    starts, lengths = (np.broadcast_to(np.asarray(v), (len(x),)) for v in (starts, lengths))
    if (not np.issubdtype(starts.dtype, np.integer)
            or not np.issubdtype(lengths.dtype, np.integer)
            or np.any(starts < 1) or np.any(lengths < 0)
            or np.any(starts + lengths > x.shape[1])):
        raise ValueError("Blocks need integer start >= 1, length >= 0, and must fit the history.")
    if not np.any(lengths):
        return x
    starts = torch.tensor(starts.copy(), dtype=torch.long, device=x.device)
    lengths = torch.tensor(lengths.copy(), dtype=torch.long, device=x.device)
    positions = torch.arange(x.shape[1], device=x.device)[None, :]
    missing = (positions >= starts[:, None]) & (positions < (starts + lengths)[:, None])
    last = x.gather(1, (starts - 1)[:, None, None].expand(-1, 1, 2))
    return torch.where(missing[:, :, None], last, x)


def augment_missing_blocks(x, max_length, rng):
    """Optional training augmentation: 50% clean; otherwise one random block.

    Positive lengths are uniform in [1, max_length], and starts are uniform
    among positions that fit after sample zero. Use a dedicated seeded NumPy
    generator so model initialization and batch shuffling cannot change masks.
    """
    if not 0 <= max_length < x.shape[1]:
        raise ValueError("max_length must be nonnegative and smaller than the history.")
    if max_length == 0:
        return x
    lengths = rng.integers(1, max_length + 1, size=len(x))
    lengths[rng.random(len(x)) < 0.5] = 0
    starts = rng.integers(1, x.shape[1] - lengths + 1)
    return hold_missing_block(x, starts, lengths)


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


def create_sequences(features, target, index, sequence_length=16, stride=1, target_start=None):
    """X[j-L:j] -> force[j]; optionally anchor targets across different L.

    target_start counts rows within each already-split recording partition.
    Use the same anchor >= max(L) in ALL splits for a matched context ablation.
    """
    features, target, index = map(np.asarray, (features, target, index))
    if not len(features) == len(target) == len(index):
        raise ValueError("Features, targets and indices must have equal lengths.")
    target_start = sequence_length if target_start is None else target_start
    if sequence_length < 2 or stride < 1 or not sequence_length <= target_start < len(features):
        raise ValueError("Require L >= 2, stride >= 1 and L <= target_start < partition rows.")
    starts = np.arange(target_start - sequence_length, len(features) - sequence_length, stride)
    windows = np.stack([features[i:i + sequence_length] for i in starts])
    return windows, target[starts + sequence_length], index[starts + sequence_length]


def prepare_data(train_files, test_files, path=DATA_DIR, sequence_length=16,
                 cut_percent=0.1, validation_fraction=0.15, train_stride=1,
                 normalization=None, validation_files=None, target_start=None,
                 recordings=False):
    """Return split arrays, target provenance, training statistics and dt.

    Each trimmed training recording reserves its final fraction of ROWS for
    validation. Windows are built separately inside each partition. Test
    recordings are held out entirely. Validation/test always use stride 1.
    Explicit validation_files instead hold out entire CASES (all speeds and
    cutoffs); validation_fraction is then unused. Old checkpoints use the tail
    split unchanged. Fit statistics on the actual training partition only.
    Physical dt is inferred from distance[m]/speed[m/s] and must be uniform;
    constant speed and the filename units are explicit dataset assumptions.
    recordings=True returns full standardized input partitions as a list,
    with exactly the same selected targets/provenance as the window path.
    """
    if not train_files or not test_files or set(train_files) & set(test_files):
        raise ValueError("Require nonempty, disjoint train/test file lists.")
    if validation_files is not None:
        if not validation_files:
            raise ValueError("Explicit validation files must be nonempty.")
        groups = [{re.search(r"_Case(\d+)_", name).group(1) for name in files}
                  for files in (train_files, validation_files, test_files)]
        if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Explicit validation requires disjoint train/validation/test CASES.")
    elif not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1).")
    segments = {"train": [], "validation": [], "test": []}
    intervals = []
    for filename in train_files + (validation_files or []) + test_files:
        frame = read_recording(filename, path, cut_percent)
        speed = float(re.match(r"V(\d+)_", filename).group(1)) / 3.6
        dt = np.diff(frame["distance"].to_numpy()) / speed
        if not (dt > 0).all() or not np.allclose(dt, dt[0], rtol=1e-4, atol=1e-9):
            raise ValueError(f"{filename}: the common fixed-grid experiment needs uniform dt.")
        intervals.append(float(np.median(dt)))
        if validation_files is not None:
            split_name = ("train" if filename in train_files else
                          "validation" if filename in validation_files else "test")
            segments[split_name].append((filename, frame))
        elif filename in train_files:
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
    for split_name, split_recordings in segments.items():
        windows, targets, provenance = [], [], []
        stride = train_stride if split_name == "train" else 1
        for filename, frame in split_recordings:
            features = frame[["acceleration", "displacement"]].to_numpy(dtype=float)
            features = (features - normalization["input_mean"]) / normalization["input_std"]
            force = frame["force"].to_numpy(dtype=float)
            scaled_force = (force - normalization["force_mean"]) / normalization["force_std"]
            if recordings:
                anchor = sequence_length if target_start is None else target_start
                if sequence_length < 2 or stride < 1 or not sequence_length <= anchor < len(frame):
                    raise ValueError("Require L >= 2, stride >= 1 and L <= target_start < partition rows.")
                positions = np.arange(anchor, len(frame), stride)
                X, y, rows = features, scaled_force[positions], frame.index[positions]
            else:
                X, y, rows = create_sequences(features, scaled_force, frame.index,
                                              sequence_length, stride, target_start)
            windows.append(X)
            targets.append(y)
            provenance.append(pd.DataFrame({
                "file": filename, "source_row": rows,
                "distance_m": frame.loc[rows, "distance"].to_numpy(),
                "force_N": frame.loc[rows, "force"].to_numpy(),
            }))
        arrays[split_name] = ([x.astype(np.float32) for x in windows] if recordings
                             else np.concatenate(windows).astype(np.float32),
                             np.concatenate(targets).astype(np.float32))
        metadata[split_name] = pd.concat(provenance, ignore_index=True)
    return arrays, metadata, normalization, sample_interval


def recording_batches(recordings, target_start, stride, batch_size, steps):
    """Ordered independent streams; microchunks never cross an Adam batch.

    Yield original target offsets, recording ids, observation starts/ends,
    and (batch-record, target-offset-from-start) pairs. Each stream advances
    at most steps intervals (except a longer initial warm-up). Accumulating
    these microchunks gives exactly batch_size targets per update, including
    across recording boundaries. No observations are skipped inside a stream.
    """
    if not recordings or min(target_start, stride, batch_size, steps) < 1:
        raise ValueError("Require recordings and positive streaming settings.")
    targets = [np.arange(target_start, len(x), stride) for x in recordings]
    if any(len(t) == 0 for t in targets):
        raise ValueError("Every recording must have targets after warm-up.")
    offsets = np.cumsum([0] + [len(t) for t in targets[:-1]])
    progress = np.zeros(len(targets), dtype=int)
    starts = np.zeros(len(targets), dtype=int)
    pending = 0
    while any(progress[i] < len(t) for i, t in enumerate(targets)):
        limits = [min(len(t) - progress[i], max(1, (starts[i] + steps - t[progress[i]]) // stride + 1))
                  if progress[i] < len(t) else 0 for i, t in enumerate(targets)]
        quotas = np.zeros(len(targets), dtype=int)
        room = min(batch_size - pending, sum(limits))
        while room:
            for i, limit in enumerate(limits):
                if quotas[i] < limit:
                    quotas[i] += 1
                    room -= 1
                    if room == 0:
                        break
        ids = np.flatnonzero(quotas)
        ends = [int(targets[i][progress[i] + quotas[i] - 1]) for i in ids]
        indices, pairs = [], []
        for row, i in enumerate(ids):
            positions = np.arange(progress[i], progress[i] + quotas[i])
            indices.extend((offsets[i] + positions).tolist())
            pairs.extend((row, int(targets[i][p] - starts[i])) for p in positions)
        yield np.asarray(indices), ids.tolist(), starts[ids].tolist(), ends, pairs
        pending = (pending + len(indices)) % batch_size
        progress += quotas
        starts[ids] = ends
