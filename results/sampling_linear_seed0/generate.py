"""Read-only data-pipeline verification and PDFs. No model imports or training.

Run from the repository root: python results/sampling_linear_seed0/generate.py
The original complete data are never written. Generated diagnostics stay here.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import INPUTS, load_recordings, fit_normalization, normalize
from src.baseline_data import linear_interpolate, make_windows, trim_recording
from src.sampling import observation_mask, retained_observations
from src.metrics import interpolation_metrics
from src.plot import plot_sampling


def pool_errors(rows, *, by_gap=False):
    """Pool removed samples, not per-recording RMSEs; keep splits identifiable."""
    frame = pd.DataFrame(rows)
    frame = pd.concat([frame, frame.assign(split="all")], ignore_index=True)
    groups = ["pattern", "retention", "seed", "split", "channel", "unit"]
    if by_gap:
        groups.append("gap_length_samples")
    totals = frame.groupby(groups)[["removed", "gaps", "absolute_error_sum",
                                   "squared_error_sum"]].sum().reset_index()
    denominator = totals.removed.replace(0, np.nan)
    totals["mae"] = totals.absolute_error_sum / denominator
    totals["rmse"] = np.sqrt(totals.squared_error_sum / denominator)
    return totals


output = Path(__file__).resolve().parent
previous_path = output / "verification.json"
previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
previous_masks = {(c["file"], c["pattern"], c["retention_requested"]): c["mask_sha256"]
                  for c in previous.get("checks", [])}
raw_files = sorted((ROOT / "data").rglob("*.xls")) + sorted((ROOT / "data").rglob("*.xlsx"))
hash_file = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
before = {str(p.relative_to(ROOT)): hash_file(p) for p in raw_files}
raw = load_recordings(cutoff=20)
records = [trim_recording(r, .1) for r in raw]
stats = fit_normalization([r for r in records if r.split == "train"])
original_stats = json.dumps(stats, sort_keys=True)
summary = {"purpose": "data/adapter validation only; no training or model inference",
           "seed": 0, "cutoff": 20, "cut_percent_each_end": .1, "sequence_length": 64,
           "train_stride": 16, "evaluation_stride": 1, "burst_length": 8,
           "baseline_adapter": "linear_interpolation", "retained_endpoints": "first_and_last",
           "causal_baseline_inputs": False,
           "numpy_version": np.__version__, "pandas_version": pd.__version__,
           "normalization": stats, "raw_sha256": before, "checks": [],
           "source_sha256": {p.name: hash_file(p) for p in sorted((ROOT / "src").glob("*.py"))}}
counts = {s: 0 for s in ("train", "validation", "test")}
sensor_scores, gap_scores = [], []
for record, original in zip(records, raw):
    original_frame = original.samples.copy(deep=True)
    segment_frame = record.samples.copy(deep=True)
    stride = 16 if record.split == "train" else 1
    clean_x, clean_y, clean_targets = make_windows(record, stats, 64, stride)
    counts[record.split] += len(clean_y)
    scaled, _ = normalize(record, stats)
    for pattern in ("random", "bursty"):
        for retention in (1., .8, .6):
            mask = observation_mask(record, retention, pattern, seed=0, burst_length=8)
            mask_hash = hashlib.sha256(mask.to_numpy().tobytes()).hexdigest()
            if previous_masks:
                assert mask_hash == previous_masks[(record.name, pattern, retention)]
            pd.testing.assert_series_equal(mask, observation_mask(record, retention, pattern, seed=0))
            assert mask.iloc[0] and mask.iloc[-1] and (~mask).sum() == round(len(mask) * (1 - retention))
            filled = linear_interpolate(record, mask)
            observed = retained_observations(record, mask, stats)
            x, y, targets = make_windows(record, stats, 64, stride, mask=mask)
            np.testing.assert_array_equal(y, clean_y)
            pd.testing.assert_frame_equal(targets, clean_targets)
            np.testing.assert_array_equal(observed.time_s, record.samples.loc[mask, "time_s"])
            np.testing.assert_array_equal(observed[INPUTS], scaled[mask])
            # Independent check of the stated endpoint-weighted formula.
            missing = np.flatnonzero(~mask.to_numpy())
            retained = np.flatnonzero(mask.to_numpy())
            right = np.searchsorted(retained, missing)
            a, b = retained[right - 1], retained[right]
            time = record.samples.time_s.to_numpy()
            sensors = record.samples[INPUTS].to_numpy()
            weight = (time[missing] - time[a]) / (time[b] - time[a])
            expected = sensors[a] + weight[:, None] * (sensors[b] - sensors[a])
            np.testing.assert_allclose(filled.iloc[missing], expected, rtol=1e-12, atol=1e-12)
            pd.testing.assert_frame_equal(filled.loc[mask], record.samples.loc[mask, INPUTS])
            if retention == 1.:
                pd.testing.assert_frame_equal(filled, record.samples[INPUTS])
                np.testing.assert_array_equal(x, clean_x)
            edges = np.diff(np.r_[False, ~mask.to_numpy(), False].astype(int))
            lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
            if pattern == "bursty" and len(lengths):
                assert lengths.max() <= 8 and (lengths != 8).sum() <= 1
            # Physical-unit sensor distortion only; force is never scored here.
            errors = sensors[missing] - filled.to_numpy()[missing]
            length_at_missing = np.repeat(lengths, lengths)
            assert len(length_at_missing) == len(missing)
            for channel_index, (channel, unit) in enumerate(zip(INPUTS, ("m/s^2", "m"))):
                metrics = interpolation_metrics(sensors[:, channel_index], filled[channel], mask)
                error = errors[:, channel_index]
                info = {"file": record.name, "split": record.split, "pattern": pattern,
                        "retention": retention, "seed": 0, "channel": channel, "unit": unit}
                sensor_scores.append({**info, **metrics, "gaps": len(lengths),
                                      "absolute_error_sum": float(np.abs(error).sum()),
                                      "squared_error_sum": float((error ** 2).sum())})
                for length in np.unique(lengths):
                    gap_error = error[length_at_missing == length]
                    gap_scores.append({**info, "gap_length_samples": int(length),
                                       "removed": len(gap_error), "gaps": int((lengths == length).sum()),
                                       "absolute_error_sum": float(np.abs(gap_error).sum()),
                                       "squared_error_sum": float((gap_error ** 2).sum())})
            summary["checks"].append({
                "file": record.name, "split": record.split, "pattern": pattern,
                "retention_requested": retention, "retention_actual": float(mask.mean()),
                "rows": len(mask), "observed": int(mask.sum()), "targets": len(y),
                "first_and_last_retained": bool(mask.iloc[0] and mask.iloc[-1]),
                "longest_missing_run_samples": int(lengths.max()) if len(lengths) else 0,
                "mask_sha256": mask_hash})
            if record is records[0]:
                plot_sampling(record, mask, filled, output, start=0, points=len(record.samples),
                              name=f"{pattern}_retention{round(retention * 100)}")
            pd.testing.assert_frame_equal(record.samples, segment_frame)
            pd.testing.assert_frame_equal(original.samples, original_frame)
    print(f"Verified {record.name}: six conditions, unchanged {len(clean_y)} targets", flush=True)
assert counts == {"train": 9401, "validation": 76989, "test": 69621}
assert before == {str(p.relative_to(ROOT)): hash_file(p) for p in raw_files}
assert json.dumps(stats, sort_keys=True) == original_stats

# One common zoom for all four conditions. Use overlapping 8-row bursts nearest
# the midpoint so both bursty conditions show a gap in a short readable window.
example = records[0]
masks = {(pattern, retention): observation_mask(example, retention, pattern, seed=0)
         for pattern in ("random", "bursty") for retention in (.8, .6)}
edges = np.diff(np.r_[False, ~masks[("bursty", .6)].to_numpy(), False].astype(int))
starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
full_bursts = np.flatnonzero(stops - starts == 8)
edges80 = np.diff(np.r_[False, ~masks[("bursty", .8)].to_numpy(), False].astype(int))
starts80, stops80 = np.flatnonzero(edges80 == 1), np.flatnonzero(edges80 == -1)
full80 = np.flatnonzero(stops80 - starts80 == 8)
for chosen in sorted(full_bursts, key=lambda i: abs((starts[i] + stops[i]) / 2 - len(example.samples) / 2)):
    overlap = full80[(starts80[full80] < stops[chosen]) & (stops80[full80] > starts[chosen])]
    if len(overlap):
        gap_left = min(starts[chosen], starts80[overlap[0]])
        gap_right = max(stops[chosen], stops80[overlap[0]])
        break
else:
    raise ValueError("No overlapping eight-row bursts for a common zoom.")
# Include eight context rows on either side, then extend to common retained
# endpoints so interpolation brackets remain visible in every zoom.
common = np.flatnonzero(np.logical_and.reduce([m.to_numpy() for m in masks.values()]))
left = int(common[common <= max(0, gap_left - 8)][-1])
right = int(common[common >= min(len(example.samples) - 1, gap_right + 8)][0])
for (pattern, retention), mask in masks.items():
    assert (~mask.iloc[left:right + 1]).any()
    plot_sampling(example, mask, linear_interpolate(example, mask), output,
                  start=left, points=right - left + 1,
                  name=f"{pattern}_retention{round(retention * 100)}_zoom")

per_recording = pd.DataFrame(sensor_scores)
pooled = pool_errors(sensor_scores)
by_gap = pool_errors(gap_scores, by_gap=True)
# The gap-length partition must account for every removed sample exactly once.
groups = ["pattern", "retention", "seed", "split", "channel", "unit"]
sums = ["removed", "gaps", "absolute_error_sum", "squared_error_sum"]
expected = pooled[pooled.retention < 1].set_index(groups)[sums].sort_index()
actual = by_gap.groupby(groups)[sums].sum().sort_index()
pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)
per_recording.to_csv(output / "interpolation_by_recording.csv", index=False)
pooled.to_csv(output / "interpolation_summary.csv", index=False)
by_gap.to_csv(output / "interpolation_by_gap.csv", index=False)

summary.update({"passed": True, "window_counts_per_condition": counts,
                "raw_files_unchanged": len(raw_files), "recordings_checked": len(records),
                "condition_checks": len(summary["checks"]),
                "previous_masks_checked": len(previous_masks),
                "interpolation_scoring": "removed sensor rows only, before normalization and windowing",
                "aggregation": "pooled absolute/squared errors weighted by removed sample count",
                "zero_missing_errors": "undefined; empty CSV cells, not zero",
                "retained_values_exact": True, "clean_inputs_exact": True,
                "figure_recording": example.name,
                "overview_source_rows": [int(example.samples.index[0]), int(example.samples.index[-1])],
                "zoom_source_rows": [int(example.samples.index[left]), int(example.samples.index[right])],
                "zoom_selection": "overlapping 8-sample bursts nearest midpoint in bursty 60% and 80%; common window for all conditions, no signal/error-based selection"})
# Preserve the previous verification evidence, including its original hashes.
(output / "interpolation_checks.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"PASS: {len(records)} recordings, {len(summary['checks'])} condition checks; {counts}")
print(pooled[pooled.split == "all"][["pattern", "retention", "channel", "unit", "removed", "mae", "rmse"]]
      .to_string(index=False))
