"""Read-only data-pipeline verification and PDFs. No model imports or training.

Run from the repository root: python results/sampling_diagnostics_seed0/generate.py
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
from src.baseline_data import forward_fill, make_windows, trim_recording
from src.sampling import observation_mask, retained_observations
from src.plot import plot_sampling


output = Path(__file__).resolve().parent
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
           "numpy_version": np.__version__, "pandas_version": pd.__version__,
           "normalization": stats, "raw_sha256": before, "checks": [],
           "source_sha256": {p.name: hash_file(p) for p in sorted((ROOT / "src").glob("*.py"))}}
counts = {s: 0 for s in ("train", "validation", "test")}
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
            pd.testing.assert_series_equal(mask, observation_mask(record, retention, pattern, seed=0))
            assert mask.iloc[0] and (~mask).sum() == round(len(mask) * (1 - retention))
            filled = forward_fill(record, mask)
            observed = retained_observations(record, mask, stats)
            x, y, targets = make_windows(record, stats, 64, stride, mask=mask)
            np.testing.assert_array_equal(y, clean_y)
            pd.testing.assert_frame_equal(targets, clean_targets)
            np.testing.assert_array_equal(observed.time_s, record.samples.loc[mask, "time_s"])
            np.testing.assert_array_equal(observed[INPUTS], scaled[mask])
            last = np.maximum.accumulate(np.where(mask, np.arange(len(mask)), 0))
            np.testing.assert_array_equal(filled, record.samples[INPUTS].to_numpy()[last])
            pd.testing.assert_frame_equal(filled.loc[mask], record.samples.loc[mask, INPUTS])
            if retention == 1.:
                np.testing.assert_array_equal(x, clean_x)
            edges = np.diff(np.r_[False, ~mask.to_numpy(), False].astype(int))
            lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
            if pattern == "bursty" and len(lengths):
                assert lengths.max() <= 8 and (lengths != 8).sum() <= 1
            summary["checks"].append({
                "file": record.name, "split": record.split, "pattern": pattern,
                "retention_requested": retention, "retention_actual": float(mask.mean()),
                "rows": len(mask), "observed": int(mask.sum()), "targets": len(y),
                "longest_missing_run_samples": int(lengths.max()) if len(lengths) else 0,
                "mask_sha256": hashlib.sha256(mask.to_numpy().tobytes()).hexdigest()})
            if record is records[0]:
                plot_sampling(record, mask, filled, output, start=0, points=200,
                              name=f"{pattern}_retention{round(retention * 100)}")
            pd.testing.assert_frame_equal(record.samples, segment_frame)
            pd.testing.assert_frame_equal(original.samples, original_frame)
    print(f"Verified {record.name}: six conditions, unchanged {len(clean_y)} targets", flush=True)
assert counts == {"train": 9401, "validation": 76989, "test": 69621}
assert before == {str(p.relative_to(ROOT)): hash_file(p) for p in raw_files}
assert json.dumps(stats, sort_keys=True) == original_stats
summary.update({"passed": True, "window_counts_per_condition": counts,
                "raw_files_unchanged": len(raw_files), "recordings_checked": len(records),
                "condition_checks": len(summary["checks"])})
(output / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"PASS: {len(records)} recordings, {len(summary['checks'])} condition checks; {counts}")
