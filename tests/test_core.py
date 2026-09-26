"""Focused synthetic tests; no real dataset, optimizer, training or network."""

import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.baseline_data import make_windows, trim_recording
from src.baselines import CNNGRU, GRU, LSTM, RNN
from src.data import Recording, distance_spacing, fit_normalization, load_recording, load_recordings, normalize, recording_info
from src.evaluate import evaluate, predict
from src.metrics import regression_metrics
from src import plot


def recording(case=1, rows=20, offset=0):
    k = np.arange(rows, dtype=float)
    raw = pd.DataFrame({0: k / 12, 1: k + offset, 2: 2 * k - offset, 3: 100 + 3 * k})
    with patch("src.data.pd.read_excel", return_value=raw):
        return load_recording(f"V300_Case{case}_CutFre20.xls")


class DataTests(unittest.TestCase):
    def test_all_raw_rows_columns_and_distance_are_aligned(self):
        r = recording()
        f = r.samples
        self.assertEqual(f.shape, (20, 4))
        self.assertEqual(list(f.columns), ["distance", "acceleration", "uplift", "force_N"])
        self.assertEqual(list(f.index), list(range(20)))
        self.assertEqual(f.index.name, "source_row")
        np.testing.assert_array_equal(f.acceleration, np.arange(20))
        np.testing.assert_array_equal(f.uplift, 2 * np.arange(20))
        np.testing.assert_array_equal(f.force_N, 100 + 3 * np.arange(20))
        np.testing.assert_array_equal(f.distance, np.arange(20) / 12)
        self.assertTrue((distance_spacing(f.distance) > 0).all())
        self.assertEqual(f.attrs["units"]["distance"], "m")

    def test_headerless_read_and_original_distance_origin(self):
        raw = pd.DataFrame([[100, 1, 2, 3], [101, 4, 5, 6]])
        with patch("src.data.pd.read_excel", return_value=raw) as read:
            r = load_recording("V300_Case1_CutFre20.xls")
        read.assert_called_once_with(Path(r.name), header=None)
        np.testing.assert_array_equal(r.samples.distance, [100, 101])
        np.testing.assert_array_equal(r.samples.force_N, [3, 6])

    def test_distance_spacing_is_exact_and_nonuniform_coordinates_are_preserved(self):
        distance = np.array([100., 100.25, 101., 103.])
        np.testing.assert_array_equal(distance_spacing(distance), [.25, .75, 2.])
        raw = pd.DataFrame({0: distance, 1: np.ones(4), 2: np.ones(4), 3: np.ones(4)})
        with patch("src.data.pd.read_excel", return_value=raw):
            r = load_recording("V380_Case1_CutFre20.xls")
        np.testing.assert_array_equal(r.samples.distance, distance)
        for invalid in ([1.], [1., 1.], [2., 1.], [1., np.nan], [[1., 2.]]):
            with self.assertRaises(ValueError):
                distance_spacing(invalid)
        with self.assertRaises(ValueError):
            make_windows(r, fit_normalization([r]), 2)

    def test_invalid_raw_data_is_rejected_not_repaired(self):
        for values in ([[1, 2, 3]], [[1, 2, 3, 4]], [[1, 2, np.nan, 4], [2, 3, 4, 5]],
                       [[2, 2, 3, 4], [1, 3, 4, 5]], [[1, 2, 3, 4], [1, 3, 4, 5]]):
            with self.subTest(values=values), patch("src.data.pd.read_excel", return_value=pd.DataFrame(values)):
                with self.assertRaises(ValueError):
                    load_recording("V300_Case1_CutFre20.xls")

    def test_split_determinism_and_no_case_overlap(self):
        groups = {s: set() for s in ("train", "validation", "test")}
        for speed in (300, 350, 380):
            for case in range(1, 9):
                for cutoff in (20, 50, 100, 150, 200):
                    name = f"V{speed}_Case{case}_CutFre{cutoff}.xls"
                    info = recording_info(name)
                    self.assertEqual(info, recording_info(name))
                    groups[info[-1]].add(case)
        self.assertEqual(groups, {"train": {1, 2, 3, 4}, "validation": {5, 6}, "test": {7, 8}})
        with self.assertRaises(ValueError):
            recording_info("unidentified.xls")

    def test_loader_orders_recordings_without_merging(self):
        files = [Path("V300_Case7_CutFre20.xls"), Path("V300_Case1_CutFre20.xls")]
        with patch("src.data.Path.glob", return_value=files), patch("src.data.load_recording", side_effect=lambda p: p.name):
            self.assertEqual(load_recordings(cutoff=20), sorted(p.name for p in files))
        with patch("src.data.Path.glob", return_value=[]), self.assertRaises(ValueError):
            load_recordings(cutoff=20)

    def test_normalization_training_only_population_statistics(self):
        train = recording()
        stats = fit_normalization([train])
        x, y = normalize(train, stats)
        np.testing.assert_allclose(x.mean(0), 0, atol=1e-15)
        np.testing.assert_allclose(x.std(0), 1)
        self.assertAlmostEqual(float(y.std()), 1)
        before = dict(stats)
        normalize(recording(7, offset=10000), stats)
        self.assertEqual(stats, before)
        for records in ([], [train, train], [recording(5)], [train, recording(7)]):
            with self.assertRaises(ValueError):
                fit_normalization(records)

    def test_constant_training_channels_and_invalid_statistics(self):
        r = recording()
        r.samples.loc[:, ["acceleration", "uplift", "force_N"]] = 2.
        stats = fit_normalization([r])
        self.assertEqual(stats["force_std"], 1e-12)
        np.testing.assert_array_equal(normalize(r, stats)[0], 0)
        for std in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                normalize(r, {**stats, "force_std": std})

    def test_trim_and_windows_preserve_next_sample_alignment(self):
        raw = recording()
        original = raw.samples.copy(deep=True)
        trimmed = trim_recording(raw, .1)
        stats = fit_normalization([trimmed])
        x, y, targets = make_windows(trimmed, stats, 4, stride=3)
        self.assertEqual(x.shape, (4, 4, 2))
        self.assertEqual(x.dtype, np.float32)
        np.testing.assert_array_equal(targets.source_row, [6, 9, 12, 15])
        np.testing.assert_array_equal(x[0], normalize(trimmed, stats)[0][:4].astype(np.float32))
        np.testing.assert_array_equal(y, normalize(trimmed, stats)[1][[4, 7, 10, 13]].astype(np.float32))
        np.testing.assert_array_equal(targets.force_N, original.loc[[6, 9, 12, 15], "force_N"])
        pd.testing.assert_frame_equal(raw.samples, original)

    def test_future_sensors_and_force_do_not_enter_inputs(self):
        r = recording()
        stats = fit_normalization([r])
        original_x = make_windows(r, stats, 4)[0][0]
        changed = Recording(r.name, r.split, r.samples.copy())
        changed.samples.loc[4:, ["acceleration", "uplift"]] = 1e8
        changed.samples.loc[:, "force_N"] = -1e8
        np.testing.assert_array_equal(make_windows(changed, stats, 4)[0][0], original_x)

    def test_recording_boundaries_and_invalid_windows(self):
        stats = fit_normalization([recording()])
        for case in (1, 5, 7):
            r = recording(case, offset=case * 100)
            x, _, targets = make_windows(r, stats, 3)
            self.assertEqual(len(x), 17)
            self.assertEqual(targets.file.unique().tolist(), [r.name])
            self.assertEqual(targets.split.unique().tolist(), [r.split])
            self.assertEqual(targets.source_row.iloc[0], 3)
        r = recording()
        for length, stride in ((1, 1), (20, 1), (3, 0), (3.5, 1)):
            with self.assertRaises(ValueError):
                make_windows(r, stats, length, stride)
        omitted = Recording(r.name, r.split, r.samples.drop(index=5))
        with self.assertRaises(ValueError):
            make_windows(omitted, stats, 3)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_architecture_dimensions_and_trainable_parameter_counts(self):
        for cls, count in ((CNNGRU, 15969), (GRU, 9825), (LSTM, 12833), (RNN, 3233)):
            model = cls()
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), count)
            self.assertEqual(model.recurrent.num_layers, 2)
            self.assertEqual(model.recurrent.hidden_size, 32)
            self.assertEqual(model.recurrent.dropout, 0)
            self.assertFalse(model.recurrent.bidirectional)
            for length in (4, 16, 64):
                self.assertEqual(model(torch.zeros(3, length, 2)).shape, (3, 1))
        self.assertEqual(CNNGRU().convolution.kernel_size, (1,))
        self.assertEqual(CNNGRU().convolution.out_channels, 64)

    def test_initializers_and_single_lstm_rnn_bias(self):
        for cls in (RNN, LSTM):
            model = cls()
            self.assertFalse(model.recurrent.bias_hh_l0.requires_grad)
            self.assertFalse(model.recurrent.bias_hh_l1.requires_grad)
            torch.testing.assert_close(model.recurrent.bias_hh_l0, torch.zeros_like(model.recurrent.bias_hh_l0))
        lstm = LSTM()
        torch.testing.assert_close(lstm.recurrent.bias_ih_l0[32:64], torch.ones(32))

    def test_checkpoint_roundtrip_and_independent_windows(self):
        x = torch.randn(7, 8, 2)
        for cls in (CNNGRU, GRU, LSTM, RNN):
            model = cls().eval()
            reference = predict(model, x, batch_size=7)
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            restored = cls()
            restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
            np.testing.assert_allclose(predict(restored, x, batch_size=2), reference, atol=1e-6)
            np.testing.assert_allclose(predict(restored, x.flip(0), batch_size=2)[::-1], reference, atol=1e-6)

    def test_common_evaluation_metadata_and_metrics(self):
        train, heldout = recording(), [recording(7), recording(8)]
        stats = fit_normalization([trim_recording(train, .1)])
        scores, frame = evaluate(CNNGRU(), heldout, stats, 4, cut_percent=.1, batch_size=3)
        self.assertEqual(len(frame), 24)
        self.assertEqual(frame.split.unique().tolist(), ["test"])
        self.assertEqual(frame.groupby("file").source_row.min().tolist(), [6, 6])
        self.assertEqual(scores, regression_metrics(frame.force_N, frame.prediction_N))
        for records in ([], [train, heldout[0]], [heldout[0], heldout[0]]):
            with self.assertRaises(ValueError):
                evaluate(CNNGRU(), records, stats, 4, cut_percent=.1)

    def test_clean_baseline_predictions_do_not_use_distance_as_a_feature(self):
        r = recording()
        stats = fit_normalization([r])
        expected, y, targets = make_windows(r, stats, 4)
        changed = Recording(r.name, r.split, r.samples.copy())
        changed.samples["distance"] = 100. + 2 * changed.samples.distance
        x, changed_y, changed_targets = make_windows(changed, stats, 4)
        np.testing.assert_array_equal(x, expected)
        np.testing.assert_array_equal(changed_y, y)
        np.testing.assert_array_equal(changed_targets.source_row, targets.source_row)
        for cls in (CNNGRU, GRU, LSTM, RNN):
            model = cls().eval()
            np.testing.assert_array_equal(predict(model, x), predict(model, expected))


class MetricTests(unittest.TestCase):
    def test_known_values(self):
        scores = regression_metrics([1, 2, 3], [2, 2, 2])
        self.assertAlmostEqual(scores["mae"], 2 / 3)
        self.assertAlmostEqual(scores["mse"], 2 / 3)
        self.assertAlmostEqual(scores["rmse"], np.sqrt(2 / 3))
        self.assertAlmostEqual(scores["r2"], 0)
        self.assertEqual(regression_metrics([1, 2], [1, 2])["r2"], 1)

    def test_undefined_and_invalid_inputs(self):
        self.assertTrue(np.isnan(regression_metrics([2, 2], [2, 2])["r2"]))
        for target, prediction in (([], []), ([1, 2], [1]), ([np.nan], [1]), ([1], [np.inf])):
            with self.assertRaises(ValueError):
                regression_metrics(target, prediction)


class PlotTests(unittest.TestCase):
    def tearDown(self):
        plot.plt.close("all")

    def test_loss_splits_remain_separate_and_metrics_use_common_keys(self):
        histories = [pd.DataFrame({"epoch": [1, 2], "train_mse_scaled": [.4, .3],
                                   "val_mse_scaled": [.5, .4]})] * 2
        with patch("src.plot.save_figure") as save:
            plot.plot_losses(histories, ["A", "B"], ["red", "blue"], "/unused")
            self.assertEqual([c.args[2] for c in save.call_args_list],
                             ["training_loss_curves", "validation_loss_curves"])
            for call in save.call_args_list:
                self.assertEqual(len(call.args[0].axes[0].lines), 2)
            plot.plot_metrics(pd.DataFrame([regression_metrics([1, 2], [1, 2])]),
                              ["A"], ["red"], "/unused")
            self.assertEqual(save.call_args.args[2], "metric_comparison")

    def test_force_comparison_rejects_misaligned_truth(self):
        r = recording(7)
        stats = fit_normalization([recording()])
        _, _, frame = make_windows(r, stats, 4)
        frame["prediction_N"] = frame.force_N
        changed = frame.copy()
        changed.loc[0, "force_N"] += 1
        with self.assertRaises(ValueError):
            plot.plot_force([frame, changed], ["A", "B"], ["red", "blue"], r.name, 0, 4, "/unused")
        with patch("src.plot.save_figure") as save:
            plot.plot_force([frame, frame], ["A", "B"], ["red", "blue"], r.name, 0, 4, "/unused")
            self.assertEqual(len(save.call_args.args[0].axes[0].lines), 3)


if __name__ == "__main__":
    unittest.main()
