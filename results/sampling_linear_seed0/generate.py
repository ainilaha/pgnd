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
import xlrd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import INPUTS, load_recording, load_recordings, distance_spacing, fit_normalization, normalize, recording_info
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


def spacing_statistics(intervals):
    """Distribution of actual positive intervals; population std, all in meters."""
    return {"intervals": len(intervals), "mean_m": float(np.mean(intervals)),
            "std_m": float(np.std(intervals)), "min_m": float(np.min(intervals)),
            "max_m": float(np.max(intervals)), "median_m": float(np.median(intervals))}


output = Path(__file__).resolve().parent
previous_path = output / "verification.json"
previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
previous_masks = {(c["file"], c["pattern"], c["retention_requested"]): c["mask_sha256"]
                  for c in previous.get("checks", [])}
raw_files = sorted((ROOT / "data").rglob("*.xls")) + sorted((ROOT / "data").rglob("*.xlsx"))
hash_file = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
before = {str(p.relative_to(ROOT)): hash_file(p) for p in raw_files}
assert before.keys() == previous["raw_sha256"].keys()
changed_since_previous = [name for name in before if before[name] != previous["raw_sha256"][name]]
if changed_since_previous:
    print("Raw file hashes differ from the previous audit (files preserved):", changed_since_previous, flush=True)
reference = json.loads((output / "baseline_input_reference.json").read_text())
hash_array = lambda values: hashlib.sha256(np.asarray(values).tobytes()).hexdigest()
for filename in ("baselines.py", "evaluate.py"):
    assert hash_file(ROOT / "src" / filename) == reference["source_sha256"][filename]
# Audit ALL 120 XLS recordings independently against their native cell values.
# The older three XLSX files are preserved but are not canonical loader inputs.
raw_spacing = []
for path in sorted((ROOT / "data" / "Data").glob("*.xls")):
    record = load_recording(path)
    book = xlrd.open_workbook(path)
    sheet = book.sheet_by_index(0)
    for i, column in enumerate(["distance", *INPUTS, "force_N"]):
        np.testing.assert_array_equal(record.samples[column], sheet.col_values(i))
    book.release_resources()
    speed, case, cutoff, _ = recording_info(record.name)
    raw_spacing.append({"file": record.name, "split": record.split, "speed_kmh": speed,
                        "case": case, "cutoff": cutoff, "rows": len(record.samples),
                        "distance_first_m": float(record.samples.distance.iloc[0]),
                        "distance_last_m": float(record.samples.distance.iloc[-1]),
                        "raw_columns_exact": True, "strictly_increasing": True,
                        **spacing_statistics(distance_spacing(record.samples.distance))})
raw = load_recordings(cutoff=20)
records = [trim_recording(r, .1) for r in raw]
stats = fit_normalization([r for r in records if r.split == "train"])
assert stats == reference["normalization"]
assert stats == previous["normalization"]
original_stats = json.dumps(stats, sort_keys=True)
summary = {"purpose": "data/adapter validation only; no training or model inference",
           "seed": 0, "cutoff": 20, "cut_percent_each_end": .1, "sequence_length": 64,
           "train_stride": 16, "evaluation_stride": 1, "burst_length": 8,
           "baseline_adapter": "linear_interpolation", "retained_endpoints": "first_and_last",
           "causal_baseline_inputs": False,
           "coordinate": "distance", "coordinate_unit": "m", "synthetic_timestamps": False,
           "numpy_version": np.__version__, "pandas_version": pd.__version__,
           "normalization": stats, "raw_sha256": before, "checks": [],
           "raw_hash_changes_since_previous_audit": changed_since_previous,
           "source_sha256": {p.name: hash_file(p) for p in sorted((ROOT / "src").glob("*.py"))}}
counts = {s: 0 for s in ("train", "validation", "test")}
sensor_scores, gap_scores = [], []
observation_spacing, pooled_intervals = [], {}
for record, original in zip(records, raw):
    original_frame = original.samples.copy(deep=True)
    segment_frame = record.samples.copy(deep=True)
    stride = 16 if record.split == "train" else 1
    clean_x, clean_y, clean_targets = make_windows(record, stats, 64, stride)
    saved = next(row for row in reference["records"] if row["file"] == record.name)
    assert list(clean_x.shape) == saved["shape"] and record.split == saved["split"]
    for label, values in (("x", clean_x), ("y", clean_y), ("rows", clean_targets.source_row),
                          ("distance", clean_targets.distance), ("force", clean_targets.force_N)):
        assert hash_array(values) == saved[f"{label}_sha256"]
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
            assert list(observed.columns) == ["distance", *INPUTS]
            np.testing.assert_array_equal(observed.distance, record.samples.loc[mask, "distance"])
            ds_observed = distance_spacing(observed.distance)
            np.testing.assert_array_equal(observed[INPUTS], scaled[mask])
            # Independent check of the stated endpoint-weighted formula.
            missing = np.flatnonzero(~mask.to_numpy())
            retained = np.flatnonzero(mask.to_numpy())
            right = np.searchsorted(retained, missing)
            a, b = retained[right - 1], retained[right]
            distance = record.samples.distance.to_numpy()
            sensors = record.samples[INPUTS].to_numpy()
            weight = (distance[missing] - distance[a]) / (distance[b] - distance[a])
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
            speed = recording_info(record.name)[0]
            longest = int(lengths.max()) if len(lengths) else 0
            assert longest == int(np.diff(retained).max() - 1)
            observation_spacing.append({"file": record.name, "split": record.split,
                "speed_kmh": speed, "pattern": pattern, "retention": retention, "seed": 0,
                "source_row_first": int(record.samples.index[0]), "source_row_last": int(record.samples.index[-1]),
                "retained_rows": len(observed), "longest_missing_run_samples": longest,
                **spacing_statistics(ds_observed)})
            pooled_intervals.setdefault((speed, pattern, retention), []).append(ds_observed)
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
                "max_observation_gap_m": float(ds_observed.max()),
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
# Keep previous interpolation-error evidence unchanged; verify agreement to
# numerical precision rather than overwriting time-coordinate reference scores.
for name, table in (("interpolation_by_recording", per_recording),
                    ("interpolation_summary", pooled), ("interpolation_by_gap", by_gap)):
    path = output / f"{name}.csv"
    if path.exists():
        pd.testing.assert_frame_equal(table, pd.read_csv(path), check_exact=False, rtol=1e-9, atol=1e-12)
    else:
        table.to_csv(path, index=False)
pd.DataFrame(raw_spacing).to_csv(output / "raw_distance_spacing.csv", index=False)
pd.DataFrame(observation_spacing).to_csv(output / "observation_distance_spacing.csv", index=False)
spatial_summary = pd.DataFrame([
    {"speed_kmh": speed, "pattern": pattern, "retention": retention, "seed": 0,
     "scope": "all_splits_CutFre20_trim10percent", **spacing_statistics(np.concatenate(parts))}
    for (speed, pattern, retention), parts in pooled_intervals.items()])
spatial_summary.to_csv(output / "observation_distance_summary.csv", index=False)

summary.update({"passed": True, "window_counts_per_condition": counts,
                "raw_files_unchanged": len(raw_files), "recordings_checked": len(records),
                "condition_checks": len(summary["checks"]),
                "previous_masks_checked": len(previous_masks),
                "interpolation_scoring": "removed sensor rows only, before normalization and windowing",
                "aggregation": "pooled absolute/squared errors weighted by removed sample count",
                "zero_missing_errors": "undefined; empty CSV cells, not zero",
                "retained_values_exact": True, "clean_inputs_exact": True,
                "raw_xls_recordings_checked": len(raw_spacing),
                "prechange_clean_tensor_hashes_match": True,
                "baseline_and_inference_source_unchanged": True,
                "trained_checkpoint_recheck": "not run: archived checkpoint inaccessible in macOS Trash",
                "figure_recording": example.name,
                "overview_source_rows": [int(example.samples.index[0]), int(example.samples.index[-1])],
                "zoom_source_rows": [int(example.samples.index[left]), int(example.samples.index[right])],
                "zoom_selection": "overlapping 8-sample bursts nearest midpoint in bursty 60% and 80%; common window for all conditions, no signal/error-based selection"})
# Preserve previous verification/interpolation evidence, including its hashes.
(output / "distance_checks.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"PASS: {len(records)} recordings, {len(summary['checks'])} condition checks; {counts}")
print(pooled[pooled.split == "all"][["pattern", "retention", "channel", "unit", "removed", "mae", "rmse"]]
      .to_string(index=False))
print(spatial_summary.to_string(index=False))
