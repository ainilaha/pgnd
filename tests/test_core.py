"""Small synthetic regression tests: no datasets, optimizer steps or remote jobs."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from legacy.pgnd_experiments import DirectReadout, HistoryReadout, ModalPGND
from src.baselines import CNNGRU, GRU, LSTM, RNN
from src.data import create_sequences, prepare_data
from src import evaluate
from src.model import PGNDModel
from src.train import save_checkpoint


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_next_sample_alignment(self):
        x = np.arange(24).reshape(12, 2)
        y = np.arange(12) + 100
        windows, targets, rows = create_sequences(x, y, np.arange(12), 4, 3)
        np.testing.assert_array_equal(rows, [4, 7, 10])
        np.testing.assert_array_equal(targets, y[rows])
        for window, row in zip(windows, rows):
            np.testing.assert_array_equal(window, x[row - 4:row])

    def test_recording_boundaries_and_training_only_scaling(self):
        frames = {}
        names = [f"V300_Case{i}_CutFre20.xls" for i in (1, 2, 7)]
        for i, name in enumerate(names):
            values = np.arange(48, dtype=float) + i * 1000
            values[36:] += 100  # Validation has a deliberately different mean.
            frames[name] = pd.DataFrame({
                "distance": np.arange(48) * (300 / 3.6) * 0.001,
                "acceleration": values, "displacement": values * 2,
                "force": values + 50,
            }, index=np.arange(100, 148))
        with patch("src.data.read_recording", side_effect=lambda name, *_: frames[name].copy()):
            arrays, metadata, stats, dt = prepare_data(
                names[:2], names[2:], sequence_length=4, validation_fraction=0.25, train_stride=3)
        train = pd.concat([frames[name].iloc[:36] for name in names[:2]])
        np.testing.assert_allclose(stats["input_mean"], train[["acceleration", "displacement"]].mean())
        np.testing.assert_allclose(stats["input_std"], train[["acceleration", "displacement"]].std(ddof=0))
        self.assertAlmostEqual(stats["force_mean"], train.force.mean())
        self.assertAlmostEqual(stats["force_std"], train.force.std(ddof=0))
        self.assertEqual(dt, 0.001)
        self.assertTrue((metadata["train"].source_row < 136).all())
        self.assertTrue((metadata["validation"].source_row >= 140).all())
        self.assertEqual(set(metadata["test"].file), {names[2]})
        for split, (X, y) in arrays.items():
            for window, target, row in zip(X, y, metadata[split].itertuples()):
                original = frames[row.file]
                expected = original.loc[row.source_row - 4:row.source_row - 1,
                                        ["acceleration", "displacement"]].to_numpy()
                np.testing.assert_allclose(window * stats["input_std"] + stats["input_mean"],
                                           expected, atol=1e-4)
                self.assertAlmostEqual(float(target) * stats["force_std"] + stats["force_mean"],
                                       original.loc[row.source_row, "force"], places=3)

    def test_baseline_shapes_counts_and_gradients(self):
        for model, count in ((RNN(), 3233), (LSTM(), 12833), (GRU(), 9825), (CNNGRU(), 15969)):
            with self.subTest(model=type(model).__name__):
                self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), count)
                output = model(torch.randn(2, 4, 2))
                self.assertEqual(output.shape, (2, 1))
                output.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters() if p.requires_grad))

    def test_pgnd_readout_residual_and_gradients(self):
        x = torch.randn(2, 4, 2)
        for readout, count in ((False, 5761), (True, 6338)):
            model = PGNDModel(observation_readout=readout)
            if readout:  # Test a trained-like nonzero compatibility head too.
                torch.nn.init.normal_(model.direct_readout[-1].weight, std=0.1)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            prediction, detail = model(x, return_details=True)
            fast, residual = model(x, return_residual=True)
            torch.testing.assert_close(fast, prediction)
            torch.testing.assert_close(model(x), prediction)
            torch.testing.assert_close(residual, detail["residual_loss"])
            self.assertEqual(detail["states"].shape, (2, 5, 32))
            (prediction.square().mean() + 0.001 * residual).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))

    def test_metrics_and_invalid_inputs(self):
        metrics = evaluate.force_metrics([1, 2, 3], [1, 2, 4])
        self.assertAlmostEqual(metrics["mae_N"], 1 / 3)
        self.assertAlmostEqual(metrics["rmse_N"], np.sqrt(1 / 3))
        self.assertAlmostEqual(metrics["r2"], 0.5)
        self.assertTrue(np.isnan(evaluate.force_metrics([1, 1], [1, 2])["r2"]))
        with self.assertRaises(ValueError):
            evaluate.mean_squared_error([1, 2], [1, np.nan])

    def test_current_and_archived_checkpoint_evaluation(self):
        X = np.zeros((3, 4, 2), dtype=np.float32)
        normalization = {"input_mean": [0., 0.], "input_std": [1., 1.],
                         "force_mean": 100., "force_std": 10.}
        cases = [("rnn", RNN(), {}), ("lstm", LSTM(), {}), ("gru", GRU(), {}),
                 ("cnn_gru", CNNGRU(), {}), ("pgnd", PGNDModel(), {}),
                 ("pgnd_obs", PGNDModel(observation_readout=True), {"observation_readout": True}),
                 ("direct", DirectReadout(), {}), ("pgnd_modal", ModalPGND(), {}),
                 ("history_linear", HistoryReadout(4), {"sequence_length": 4}),
                 ("history_mlp", HistoryReadout(4, nonlinear=True),
                  {"sequence_length": 4, "nonlinear": True})]
        with tempfile.TemporaryDirectory() as directory:
            paths, expected = [], {}
            for name, model, options in cases:
                path = Path(directory) / f"{name}.pt"
                checkpoint = {
                    "model": name, "model_options": options, "state_dict": model.state_dict(),
                    "protocol": "recording-split-training-standardization-v1", "data_settings": {},
                    "train_files": ["train.xls"], "test_files": ["test.xls"], "data_sha256": {},
                    "normalization": normalization, "sample_interval": 0.001, "seed": 0,
                    "epochs": 1, "completed_epochs": 1, "batch_size": 2, "optimizer": {},
                    "checkpoint_kind": "best_validation", "selected_epoch": 1,
                    "history": [{"epoch": 1, "val_mse_scaled": 1.}],
                }
                save_checkpoint(checkpoint, path)
                self.assertFalse(path.with_suffix(".pt.tmp").exists())
                paths.append(str(path))
                expected[name] = evaluate.predict(model, X).astype(float) * 10 + 100
            for split in ("validation", "test"):
                targets = np.array([95., 100., 110.]) + (10 if split == "test" else 0)
                meta = pd.DataFrame({"file": [f"{split}.xls"] * 3, "source_row": [4, 5, 6],
                                     "distance_m": [0.4, 0.5, 0.6], "force_N": targets})
                prepared = ({split: (X, targets)}, {split: meta}, normalization, 0.001)
                output = Path(directory) / split
                with patch.object(sys, "argv", ["evaluate", *paths, "--split", split,
                                                "--output-dir", str(output)]), \
                     patch("src.evaluate.prepare_data", return_value=prepared), \
                     contextlib.redirect_stdout(io.StringIO()):
                    evaluate.main()
                for name in expected:
                    frame = pd.read_csv(output / f"{name}.predictions.csv")
                    np.testing.assert_allclose(frame.prediction_N, expected[name])
                    np.testing.assert_array_equal(frame.force_N, targets)
                self.assertEqual(set(pd.read_csv(output / "comparison.csv").split), {split})


if __name__ == "__main__":
    unittest.main()
