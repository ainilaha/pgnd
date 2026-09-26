"""Synthetic sampling/adapter invariants; no training or raw dataset access."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.data import INPUTS, Recording, fit_normalization, normalize, recording_info
from src.sampling import observation_mask, retained_observations, validate_mask
from src.baseline_data import linear_interpolate, make_windows, trim_recording
from src.baselines import CNNGRU, GRU, LSTM, RNN
from src.metrics import interpolation_metrics
from src import plot


def recording(n=103, case=1, offset=0):
    k = np.arange(n)
    name = f"V300_Case{case}_CutFre20.xls"
    frame = pd.DataFrame({"time_s": 3. + k * .001, "distance_m": 250. + k / 12,
                          "acceleration": np.sin(k / 5) + offset,
                          "uplift": np.cos(k / 7) - offset,
                          "force_N": 180. + k * 3.},
                         index=pd.Index(k + 97, name="source_row"))
    return Recording(name, recording_info(name)[-1], frame)


def burst_lengths(mask):
    edges = np.diff(np.r_[False, ~mask.to_numpy(), False].astype(int))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


class SamplingTests(unittest.TestCase):
    def test_deterministic_signal_independent_and_order_independent(self):
        r = recording(501)
        changed = Recording(r.name, r.split, r.samples.copy())
        changed.samples.loc[:, [*INPUTS, "force_N"]] = -10000.
        for pattern in ("random", "bursty"):
            expected = observation_mask(r, .6, pattern, seed=2)
            np.random.seed(928)
            state = np.random.get_state()
            observation_mask(recording(501, case=8), .8, pattern, seed=999)
            pd.testing.assert_series_equal(expected, observation_mask(r, .6, pattern, seed=2))
            pd.testing.assert_series_equal(expected, observation_mask(changed, .6, pattern, seed=2))
            after = np.random.get_state()
            np.testing.assert_array_equal(state[1], after[1])
            self.assertEqual(state[2:], after[2:])
            self.assertFalse(expected.equals(observation_mask(r, .6, pattern, seed=3)))
            self.assertFalse(np.array_equal(expected, observation_mask(recording(501, case=2), .6, pattern, seed=2)))

    def test_exact_retention_including_small_and_remainder_cases(self):
        for n in (2, 3, 7, 20, 103, 1001):
            r = recording(n)
            for pattern in ("random", "bursty"):
                for retention in (1., .8, .6):
                    for seed in (0, 11):
                        if round(n * (1 - retention)) > n - 2:
                            with self.assertRaises(ValueError):
                                observation_mask(r, retention, pattern, seed=seed)
                            continue
                        mask = observation_mask(r, retention, pattern, seed=seed)
                        self.assertEqual(int((~mask).sum()), round(n * (1 - retention)))
                        self.assertLessEqual(abs(mask.mean() - retention), .5 / n + 1e-15)
                        self.assertTrue(mask.iloc[0])
                        self.assertTrue(mask.iloc[-1])
                        self.assertEqual(mask.dtype, bool)
                        self.assertTrue(mask.index.equals(r.samples.index))

    def test_burst_lengths_are_controlled_and_do_not_merge(self):
        for n in (7, 103, 1001):
            for length in (1, 4, 8, 16):
                for retention in (.8, .6):
                    mask = observation_mask(recording(n), retention, "bursty", burst_length=length)
                    removed = round(n * (1 - retention))
                    expected = [length] * (removed // length)
                    if removed % length:
                        expected.append(removed % length)
                    self.assertEqual(sorted(burst_lengths(mask)), sorted(expected))
        with self.assertRaises(ValueError):
            observation_mask(recording(4), .6, "bursty", burst_length=1)

    def test_random_thinning_is_uniform_away_from_endpoint_anchors(self):
        r = recording(100)
        frequency = np.stack([observation_mask(r, .6, "random", seed=s) for s in range(250)]).mean(0)
        self.assertEqual(frequency[0], 1.)
        self.assertEqual(frequency[-1], 1.)
        self.assertLess(float(np.abs(frequency[1:-1] - 58 / 98).max()), .14)
        # The permutation is shared across retention levels for random thinning.
        self.assertTrue(np.all(~observation_mask(r, .6, "random") | observation_mask(r, .8, "random")))

    def test_invalid_conditions_and_wrong_recording_masks_fail(self):
        r = recording()
        for kwargs in ({"retention": .5}, {"pattern": "linear"}, {"seed": -1},
                       {"seed": 1.5}, {"burst_length": 0}):
            options = {"retention": .8, "pattern": "random", **kwargs}
            with self.assertRaises(ValueError):
                observation_mask(r, **options)
        mask = observation_mask(r, .8, "random")
        for bad in (mask.to_numpy(), mask.iloc[::-1], mask.astype(int),
                    observation_mask(recording(case=7), .8, "random")):
            with self.assertRaises(ValueError):
                validate_mask(r, bad)
        with self.assertRaises(ValueError):
            validate_mask(trim_recording(r, .1), mask)
        for endpoint in (0, -1):
            bad = mask.copy()
            bad.iloc[endpoint] = False
            with self.assertRaises(ValueError):
                linear_interpolate(r, bad)

    def test_original_timestamps_and_complete_targets_are_unchanged(self):
        r = recording(case=7)
        original = r.samples.copy(deep=True)
        stats = fit_normalization([recording()])
        clean_x, clean_y, clean_targets = make_windows(r, stats, 8)
        for pattern in ("random", "bursty"):
            for retention in (1., .8, .6):
                mask = observation_mask(r, retention, pattern)
                observed = retained_observations(r, mask, stats)
                self.assertEqual(list(observed.columns), ["time_s", *INPUTS])
                np.testing.assert_array_equal(observed.time_s, original.loc[mask, "time_s"])
                np.testing.assert_array_equal(observed.index, original.index[mask])
                self.assertEqual(len(observed), mask.sum())
                self.assertEqual(observed.attrs, {"file": r.name, "split": "test"})
                x, y, targets = make_windows(r, stats, 8, mask=mask)
                np.testing.assert_array_equal(y, clean_y)
                pd.testing.assert_frame_equal(targets, clean_targets)
                pd.testing.assert_frame_equal(r.samples, original)
                self.assertEqual(x.shape, clean_x.shape)

    def test_same_fixed_normalization_for_retained_and_baseline_inputs(self):
        stats = fit_normalization([recording()])
        before = deepcopy(stats)
        for case in (1, 5, 7):
            r = recording(case=case, offset=1000)
            for pattern in ("random", "bursty"):
                mask = observation_mask(r, .6, pattern)
                filled = linear_interpolate(r, mask)
                observed = retained_observations(r, mask, stats)
                scaled = (filled.to_numpy() - stats["input_mean"]) / stats["input_std"]
                np.testing.assert_array_equal(observed[INPUTS], scaled[mask])
                x, _, targets = make_windows(r, stats, 8, mask=mask)
                np.testing.assert_array_equal(x[0], scaled[:8].astype(np.float32))
                self.assertEqual(targets.split.unique().tolist(), [r.split])
        self.assertEqual(stats, before)

    def test_linear_interpolation_uses_nearest_surrounding_observations(self):
        r = recording(25)
        mask = observation_mask(r, 1., "random")
        mask.iloc[1:12] = False
        mask.iloc[14:-1] = False
        filled = linear_interpolate(r, mask)
        t = r.samples.time_s.to_numpy()
        for left, right in ((0, 12), (13, 24)):
            weight = (t[left:right + 1] - t[left]) / (t[right] - t[left])
            a, b = r.samples[INPUTS].iloc[[left, right]].to_numpy()
            expected = a + weight[:, None] * (b - a)
            np.testing.assert_allclose(filled.iloc[left:right + 1], expected, rtol=1e-12)
        pd.testing.assert_frame_equal(filled.loc[mask], r.samples.loc[mask, INPUTS])
        stats = fit_normalization([r])
        x, _, _ = make_windows(r, stats, 4, mask=mask)
        # Interpolate the recording once, not each window. Both endpoints here
        # lie outside the window; the right endpoint is after its target time.
        expected = (filled.iloc[5:9].to_numpy() - stats["input_mean"]) / stats["input_std"]
        np.testing.assert_array_equal(x[5], expected.astype(np.float32))

    def test_interpolation_weights_use_timestamps_not_row_numbers(self):
        r = recording(5)
        r.samples["time_s"] = [10., 10.1, 10.5, 11.5, 12.]
        mask = observation_mask(r, 1., "random")
        mask.iloc[1:-1] = False
        a, b = r.samples[INPUTS].iloc[[0, -1]].to_numpy()
        expected = a + np.array([0., .05, .25, .75, 1.])[:, None] * (b - a)
        np.testing.assert_allclose(linear_interpolate(r, mask), expected, rtol=1e-12)
        for bad_times in ([10., 11., 11., 12., 13.], [10., 11., np.nan, 12., 13.],
                          [10., 9., 11., 12., 13.]):
            r.samples["time_s"] = bad_times
            with self.assertRaises(ValueError):
                linear_interpolate(r, mask)

    def test_hidden_sensor_and_force_values_cannot_influence_inputs(self):
        r = recording()
        stats = fit_normalization([r])
        mask = observation_mask(r, .6, "bursty")
        changed = Recording(r.name, r.split, r.samples.copy())
        changed.samples.loc[~mask, INPUTS] = np.nan
        changed.samples.loc[:, "force_N"] = -1e10
        pd.testing.assert_frame_equal(linear_interpolate(r, mask), linear_interpolate(changed, mask))
        pd.testing.assert_frame_equal(retained_observations(r, mask, stats),
                                      retained_observations(changed, mask, stats))
        original_x = make_windows(r, stats, 8, mask=mask)[0]
        changed_x = make_windows(changed, stats, 8, mask=mask)[0]
        np.testing.assert_array_equal(original_x, changed_x)

    def test_right_endpoint_changes_earlier_inputs_as_expected_for_offline_interpolation(self):
        r = recording(12)
        stats = fit_normalization([r])
        mask = observation_mask(r, 1., "random")
        mask.iloc[1:8] = False
        changed = Recording(r.name, r.split, r.samples.copy())
        changed.samples.loc[changed.samples.index[8], INPUTS] += 10.
        original_x, original_y, original_targets = make_windows(r, stats, 4, mask=mask)
        changed_x, changed_y, changed_targets = make_windows(changed, stats, 4, mask=mask)
        self.assertFalse(np.array_equal(original_x[0], changed_x[0]))  # target row 4
        np.testing.assert_array_equal(original_y, changed_y)
        pd.testing.assert_frame_equal(original_targets, changed_targets)

    def test_interpolation_never_crosses_recording_boundaries(self):
        a, b = recording(case=1), recording(case=7, offset=1e6)
        for r in (a, b, a):
            mask = observation_mask(r, .6, "bursty")
            np.testing.assert_array_equal(linear_interpolate(r, mask).iloc[[0, -1]],
                                          r.samples[INPUTS].iloc[[0, -1]])
            with self.assertRaises(ValueError):
                linear_interpolate(b if r is a else a, mask)

    def test_100_percent_is_exact_original_clean_path_for_every_baseline(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        r = trim_recording(recording(), .1)
        stats = fit_normalization([r])
        clean_x, clean_y, clean_targets = make_windows(r, stats, 8, stride=3)
        for pattern in ("random", "bursty"):
            mask = observation_mask(r, 1., pattern)
            with patch("src.baseline_data.np.interp", side_effect=AssertionError("No filling needed")):
                pd.testing.assert_frame_equal(linear_interpolate(r, mask), r.samples[INPUTS])
            x, y, targets = make_windows(r, stats, 8, stride=3, mask=mask)
            np.testing.assert_array_equal(x, clean_x)
            np.testing.assert_array_equal(y, clean_y)
            pd.testing.assert_frame_equal(targets, clean_targets)
            with torch.no_grad():
                for cls in (CNNGRU, GRU, LSTM, RNN):
                    model = cls().eval()
                    self.assertTrue(torch.equal(model(torch.from_numpy(x)), model(torch.from_numpy(clean_x))))

    def test_diagnostic_plot_shows_inputs_and_does_not_change_them(self):
        r = recording()
        original = r.samples.copy()
        shown = original.iloc[3:103]
        reference_limits = None
        for pattern in ("random", "bursty"):
            for retention in (1., .8, .6):
                mask = observation_mask(r, retention, pattern)
                filled = linear_interpolate(r, mask)
                filled_before = filled.copy()
                keep = mask.loc[shown.index]
                with patch("src.plot.save_figure") as save:
                    plot.plot_sampling(r, mask, filled, "/unused", start=3, points=100)
                    fig = save.call_args.args[0]
                    self.assertEqual(len(fig.axes), 3)
                    for ax, column in zip(fig.axes[:2], INPUTS):
                        self.assertEqual(len(ax.lines), 3)
                        np.testing.assert_array_equal(ax.lines[0].get_xdata(), shown.distance_m)
                        np.testing.assert_array_equal(ax.lines[0].get_ydata(), shown[column])
                        np.testing.assert_array_equal(ax.lines[1].get_ydata(), filled.loc[shown.index, column])
                        np.testing.assert_array_equal(ax.lines[2].get_xdata(), shown.loc[keep, "distance_m"])
                        np.testing.assert_array_equal(ax.lines[2].get_ydata(), shown.loc[keep, column])
                        self.assertEqual([line.get_label() for line in ax.lines],
                                         ["Original", "Interpolated", "Retained"])
                        self.assertEqual(ax.lines[1].get_drawstyle(), "default")
                        self.assertEqual(bool(ax.patches), bool((~keep).any()))
                        self.assertLessEqual(int(np.ceil(keep.sum() / ax.lines[2].get_markevery())), 80)
                    force_ax = fig.axes[2]
                    self.assertEqual(len(force_ax.lines), 1)
                    self.assertEqual(len(force_ax.patches), 0)
                    np.testing.assert_array_equal(force_ax.lines[0].get_xdata(), shown.distance_m)
                    np.testing.assert_array_equal(force_ax.lines[0].get_ydata(), shown.force_N)
                    self.assertEqual([ax.get_ylabel() for ax in fig.axes],
                                     [r"Acceleration [m/s$^2$]", "Uplift [m]", "Contact force [N]"])
                    self.assertEqual(force_ax.get_xlabel(), "Distance [m]")
                    self.assertTrue(all(force_ax.get_shared_x_axes().joined(ax, force_ax) for ax in fig.axes[:2]))
                    limits = [(ax.get_xlim(), ax.get_ylim()) for ax in fig.axes]
                    if reference_limits is None:
                        reference_limits = limits
                    self.assertEqual(limits, reference_limits)
                    plot.plt.close(fig)
                pd.testing.assert_frame_equal(filled, filled_before)
        pd.testing.assert_frame_equal(r.samples, original)

    def test_interpolation_metrics_score_only_removed_rows(self):
        original = np.array([0., 10., 20., 30.])
        filled = np.array([0., 12., 17., 30.])
        keep = np.array([True, False, False, True])
        scores = interpolation_metrics(original, filled, keep)
        self.assertEqual(scores["removed"], 2)
        self.assertEqual(scores["mae"], 2.5)
        self.assertAlmostEqual(scores["rmse"], np.sqrt(6.5))
        np.testing.assert_array_equal(original, [0., 10., 20., 30.])
        np.testing.assert_array_equal(filled, [0., 12., 17., 30.])

    def test_interpolation_metrics_reject_changed_retained_values_and_bad_inputs(self):
        for original, filled, keep in (
            ([1., 2.], [1., 3.], [True, True]),
            ([1., 2.], [1., 2.], [1, 0]),
            ([1., 2.], [1.], [True, False]),
            ([[1., 2.]], [[1., 2.]], [[True, False]]),
            ([1., np.nan], [1., 2.], [True, False]),
            ([], [], np.array([], dtype=bool)),
        ):
            with self.assertRaises(ValueError):
                interpolation_metrics(original, filled, keep)

    def test_interpolation_metrics_at_100_percent_are_not_applicable(self):
        scores = interpolation_metrics([1., 2.], [1., 2.], [True, True])
        self.assertEqual(scores["removed"], 0)
        self.assertTrue(np.isnan(scores["mae"]))
        self.assertTrue(np.isnan(scores["rmse"]))


if __name__ == "__main__":
    unittest.main()
