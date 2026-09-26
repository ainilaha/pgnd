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

from legacy.pgnd_experiments import AblationPGND, DirectReadout, HistoryReadout, ModalPGND
from src.baselines import CNNGRU, GRU, LSTM, RNN
from src.data import (augment_missing_blocks, compact_observations, create_sequences,
                      hold_missing_block, interpolate_observations, observation_mask,
                      prepare_data, recording_batches)
from src import evaluate, train
from src.model import PGNDModel
from src.train import save_checkpoint


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_irregular_masks_are_nested_exact_and_recording_local(self):
        for removal in ("random", "bursty"):
            masks = [observation_mask(1000, r, removal, 0, "recording-a") for r in (1., .8, .6)]
            self.assertEqual([m.sum() for m in masks], [1000, 800, 600])
            self.assertTrue(np.all(~masks[2] | masks[1]))
            np.testing.assert_array_equal(masks[2], observation_mask(1000, .6, removal, 0, "recording-a"))
            self.assertFalse(np.array_equal(masks[2], observation_mask(1000, .6, removal, 0, "recording-b")))
            if removal == "bursty":
                positions = np.flatnonzero(masks[2])
                self.assertLessEqual(np.diff(positions).max(), 17)
            # Overlapping windows see the SAME outage, not independent masks.
            windows = np.lib.stride_tricks.sliding_window_view(masks[2], 64)
            np.testing.assert_array_equal(windows[:-1, 1:], windows[1:, :-1])

    def test_event_compaction_and_interpolation_ignore_removed_sensors(self):
        x = np.arange(16, dtype=np.float32).reshape(1, 8, 2)
        mask = np.array([[False, True, False, True, False, False, True, False]])
        values, times, lengths = compact_observations(x, mask, .001)
        np.testing.assert_array_equal(values, x[:, [1, 3, 6]])
        np.testing.assert_allclose(times, [[.001, .003, .006]])
        np.testing.assert_array_equal(lengths, [3])
        filled = interpolate_observations(x, mask)
        np.testing.assert_array_equal(filled[:, 1:7], x[:, 1:7])
        np.testing.assert_array_equal(filled[:, 0], x[:, 1])
        np.testing.assert_array_equal(filled[:, -1], x[:, 6])
        x[~mask] = np.nan
        np.testing.assert_array_equal(filled, interpolate_observations(x, mask))
        np.testing.assert_array_equal(values, compact_observations(x, mask, .001)[0])

    def test_pgnd_events_regular_equivalence_and_gradients(self):
        model = PGNDModel(latent_dim=2, encoding_dim=3, hidden_dim=4).double()
        with torch.no_grad():
            model.dynamics.residual[-1].weight.fill_(.01)
        x = torch.randn(3, 8, 2, dtype=torch.float64)
        times = torch.arange(8, dtype=x.dtype)[None, :].expand(3, -1) * .001
        regular = model(x)
        irregular = model.forward_events(x, times, [8, 8, 8], .008)
        torch.testing.assert_close(regular, irregular, rtol=1e-10, atol=1e-10)
        old_grad = torch.autograd.grad(regular.sum(), tuple(model.parameters()))
        new_grad = torch.autograd.grad(irregular.sum(), tuple(model.parameters()))
        for old, new in zip(old_grad, new_grad):
            torch.testing.assert_close(old, new, rtol=1e-9, atol=1e-9)
        regular, old_penalty = model(x, return_residual=True)
        irregular, new_penalty = model.forward_events(x, times, [8, 8, 8], .008, return_residual=True)
        torch.testing.assert_close(regular, irregular, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(old_penalty, new_penalty, rtol=1e-10, atol=1e-10)

    def test_pgnd_events_ragged_matches_individual_odes_and_ignores_padding(self):
        from torchdiffeq import odeint
        model = PGNDModel(latent_dim=2, encoding_dim=3, hidden_dim=4).double()
        with torch.no_grad():
            model.dynamics.residual[-1].weight.fill_(.01)
        x = np.arange(32, dtype=np.float64).reshape(2, 8, 2) / 50
        mask = np.array([[False, True, False, True, False, False, True, False],
                         [True, False, True, True, False, False, True, True]])
        values, times, lengths = compact_observations(x, mask, .001)
        prediction = model.forward_events(torch.tensor(values), times, lengths, .008)
        expected = []
        for i, count in enumerate(lengths):
            events = torch.tensor(values[i:i + 1, :count])
            grid = torch.tensor(np.append(times[i, :count], .008)) / .001
            encoded = model.encoder(events)

            def field(t, state):
                return model.dynamics(t, state, model.interpolate(t, grid[:-1], encoded))

            states = odeint(field, model.initial_state(events[:, 0]), grid,
                            method="rk4", options={"step_size": 1.})
            expected.append(model.decoder(states[-1]))
        torch.testing.assert_close(prediction, torch.cat(expected), rtol=1e-9, atol=1e-9)
        values[np.arange(values.shape[1])[None, :] >= lengths[:, None]] = np.nan
        torch.testing.assert_close(prediction, model.forward_events(torch.tensor(values), times, lengths, .008),
                                   rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model.forward_events(torch.tensor(values), times, lengths, .005)
        bad_times = times.copy()
        bad_times[:, 1] = bad_times[:, 0]
        with self.assertRaises(ValueError):
            model.forward_events(torch.tensor(values), bad_times, lengths, .008)

    def test_cleanup_matches_frozen_pgnd_weights_outputs_and_gradients(self):
        x = torch.arange(16, dtype=torch.float64).reshape(2, 4, 2) / 20
        for readout in (False, True):
            options = dict(latent_dim=2, encoding_dim=3, hidden_dim=4, observation_readout=readout)
            torch.manual_seed(42)
            original = AblationPGND(**options).double()
            original_rng = torch.random.get_rng_state()
            torch.manual_seed(42)
            cleaned = PGNDModel(**options).double()
            self.assertTrue(torch.equal(original_rng, torch.random.get_rng_state()))
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, cleaned.state_dict()[name], rtol=0, atol=0)
            # Nonzero learned-like residual/readout weights, not only zero-init behavior.
            with torch.no_grad():
                original.dynamics.residual[-1].weight.fill_(0.01)
                if readout:
                    original.direct_readout[-1].weight.fill_(0.02)
            cleaned.load_state_dict(original.state_dict())
            expected, expected_penalty = original(x, return_residual=True)
            actual, actual_penalty = cleaned(x, return_residual=True)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            torch.testing.assert_close(expected_penalty, actual_penalty, rtol=0, atol=0)
            (expected.square().mean() + 0.001 * expected_penalty).backward()
            (actual.square().mean() + 0.001 * actual_penalty).backward()
            for before, after in zip(original.parameters(), cleaned.parameters()):
                torch.testing.assert_close(before.grad, after.grad, rtol=0, atol=0)

    def test_recording_loader_preserves_partitions_scalers_and_targets(self):
        names = [f"V300_Case{i}_CutFre20.xls" for i in (1, 2, 5, 7)]
        frames = {name: pd.DataFrame({
            "distance": np.arange(40 + i) * (300 / 3.6) * .001,
            "acceleration": np.arange(40 + i) + i * 100.,
            "displacement": np.arange(40 + i) * 2 + i * 100.,
            "force": np.arange(40 + i) * 3 + i * 100.,
        }, index=np.arange(100, 140 + i)) for i, name in enumerate(names)}
        for validation in (None, names[2:3]):
            settings = dict(sequence_length=4, target_start=4, train_stride=3,
                            validation_fraction=.25, validation_files=validation)
            with patch("src.data.read_recording", side_effect=lambda name, *_: frames[name].copy()):
                windowed, meta, stats, dt = prepare_data(names[:2], names[3:], **settings)
                streams, stream_meta, stream_stats, stream_dt = prepare_data(
                    names[:2], names[3:], recordings=True, **settings)
            self.assertEqual(stats, stream_stats)
            self.assertEqual(dt, stream_dt)
            for split in streams:
                pd.testing.assert_frame_equal(meta[split], stream_meta[split])
                np.testing.assert_array_equal(windowed[split][1], streams[split][1])
                stride = 3 if split == "train" else 1
                for x, (_, rows) in zip(streams[split][0], meta[split].groupby("file", sort=False)):
                    expected = np.stack([x[k-4:k] for k in range(4, len(x), stride)])
                    np.testing.assert_array_equal(expected, windowed[split][0][rows.index])

    def test_recording_batches_preserve_continuity_and_exact_update_budget(self):
        recordings = [torch.zeros(n, 2) for n in (19, 24, 31)]
        previous, all_indices, pending, updates = {}, [], 0, 0
        for indices, ids, starts, ends, pairs in recording_batches(recordings, 4, 3, 5, 4):
            self.assertLessEqual(pending + len(indices), 5)
            for i, start, end in zip(ids, starts, ends):
                self.assertEqual(start, previous.get(i, 0))
                self.assertLessEqual(end - start, 4)
                previous[i] = end
            self.assertEqual(len(pairs), len(indices))
            all_indices.extend(indices.tolist())
            pending += len(indices)
            if pending == 5:
                pending = 0
                updates += 1
        count = sum(len(range(4, len(x), 3)) for x in recordings)
        self.assertEqual(sorted(all_indices), list(range(count)))
        self.assertEqual(updates + bool(pending), (count + 4) // 5)

    def test_recording_propagation_matches_one_solve_and_keeps_global_clock(self):
        model = PGNDModel(latent_dim=2, encoding_dim=3, hidden_dim=4).double()
        torch.nn.init.normal_(model.dynamics.residual[-1].weight, std=.01)
        x = torch.randn(2, 12, 2, dtype=torch.float64) / 10
        whole = model.propagate(x)
        first = model.propagate(x[:, :5])
        rest = model.propagate(x[:, 4:], first[:, -1].detach(), start_indices=[4, 4])
        torch.testing.assert_close(torch.cat((first[:, :-1], rest), 1), whole, rtol=1e-10, atol=1e-10)

    def test_recording_stream_matches_full_prefix_without_target_input(self):
        model = PGNDModel(latent_dim=2, encoding_dim=3, hidden_dim=4).double()
        torch.nn.init.normal_(model.dynamics.residual[-1].weight, std=.02)
        x = torch.randn(11, 2, dtype=torch.float64, requires_grad=True) / 10
        batches = recording_batches([x], 4, 1, 5, 4)
        calls = []
        hook = model.initial_state.register_forward_hook(lambda *_: calls.append(1))
        saved = list(model.recording_stream([x], batches, 4, return_residual=True))
        hook.remove()
        self.assertEqual(len(calls), 1)
        for indices, predictions, penalties in saved:
            for j, index in enumerate(indices):
                target = int(index) + 4
                expected, details = model(x[None, :target], return_details=True)
                torch.testing.assert_close(predictions[j], expected.squeeze(), rtol=1e-9, atol=1e-9)
                states = details["states"][:, -4:]
                h = model.encoder(torch.cat((x[target-3:target], x[target-1:target])))[None]
                times = torch.arange(target-3, target+1, dtype=x.dtype)[None, :, None]
                residual = model.dynamics.residual_force(times, states, h).square().sum(-1).mean()
                torch.testing.assert_close(penalties[j], residual, rtol=1e-9, atol=1e-9)
                gradient = torch.autograd.grad(predictions[j], x, retain_graph=True)[0]
                self.assertEqual(gradient[target:].abs().sum().item(), 0.)
        changed = x.detach().clone()
        changed[0] += 5
        before = evaluate.predict_recordings(model, [x.detach()], 4, 4)
        after = evaluate.predict_recordings(model, [changed], 4, 4)
        self.assertGreater(abs(before[-1] - after[-1]), 1e-8)

    def test_recording_prediction_batch_invariance_and_file_isolation(self):
        model = PGNDModel(latent_dim=2, encoding_dim=3, hidden_dim=4).double()
        torch.nn.init.normal_(model.dynamics.residual[-1].weight, std=.01)
        recordings = [torch.randn(n, 2, dtype=torch.float64) / 10 for n in (11, 9, 13)]
        individual = np.concatenate([evaluate.predict_recordings(model, [x], 4, 4, batch_size=3)
                                     for x in recordings])
        for batch_size in (1, 5, 1024):
            calls = []
            hook = model.initial_state.register_forward_hook(lambda *_: calls.append(1))
            actual = evaluate.predict_recordings(model, recordings, 4, 4, batch_size=batch_size)
            hook.remove()
            self.assertEqual(len(calls), len(recordings))
            np.testing.assert_allclose(actual, individual, rtol=1e-10, atol=1e-10)

    def test_recording_gradient_accumulation_without_optimizer_steps(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        model = PGNDModel(latent_dim=2, encoding_dim=3, hidden_dim=4)
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        recordings = [torch.randn(n, 2) / 10 for n in (11, 9)]
        y = torch.zeros(sum(len(range(4, len(x), 2)) for x in recordings), 1)
        optimizer = SimpleNamespace(step=Mock(), zero_grad=lambda: model.zero_grad(set_to_none=True))
        mse, residual, updates = train.train_recording_epoch(model, recordings, y, optimizer, 5, 4, 4, 2, .001)
        self.assertEqual(updates, (len(y) + 4) // 5)
        self.assertEqual(optimizer.step.call_count, updates)
        self.assertTrue(np.isfinite([mse, residual]).all())
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, saved[k], rtol=0, atol=0)

    def test_recording_cli_evaluation_dispatch_and_provenance(self):
        normalization = {"input_mean": [0., 0.], "input_std": [1., 1.],
                         "force_mean": 100., "force_std": 10.}
        raw = np.arange(18, dtype=np.float32).reshape(9, 2) / 100
        X, y, rows = create_sequences(raw, np.zeros(9), np.arange(9), 4)
        meta = pd.DataFrame({"file": ["V300_Case5_CutFre20.xls"] * len(y),
                             "source_row": rows, "distance_m": rows / 10, "force_N": y * 10 + 100})
        model = PGNDModel(latent_dim=2, encoding_dim=3, hidden_dim=4)
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for state_mode in ("window", "persistent"):
                path = Path(directory) / f"{state_mode}.best.pt"
                save_checkpoint({
                    "model": "pgnd", "model_options": dict(latent_dim=2, encoding_dim=3, hidden_dim=4),
                    "state_dict": model.state_dict(), "state_mode": state_mode,
                    "protocol": "case-validation-training-standardization-v1",
                    "data_settings": {"sequence_length": 4, "target_start": 4},
                    "normalization": normalization, "train_files": ["train.xls"], "test_files": ["test.xls"],
                    "data_sha256": {}, "sample_interval": .001, "seed": 0, "epochs": 1,
                    "completed_epochs": 1, "batch_size": 5, "optimizer": {}, "selected_epoch": 1,
                    "checkpoint_kind": "best_validation", "history": [{"epoch": 1, "val_mse_scaled": 1.}],
                }, path)
                paths.append(str(path))
            output = Path(directory) / "validation"
            prepared = ({"validation": (X, y)}, {"validation": meta}, normalization, .001)
            streamed = ({"validation": ([raw], y)}, {"validation": meta}, normalization, .001)
            argv = ["evaluate", *paths, "--split", "validation", "--output-dir", str(output)]
            with patch.object(sys, "argv", argv), \
                 patch("src.evaluate.prepare_data", side_effect=[prepared, streamed]), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            expected = evaluate.predict_recordings(model, [raw], 4, 4).astype(float) * 10 + 100
            predictions = pd.read_csv(output / "persistent.best.predictions.csv")
            np.testing.assert_allclose(predictions.prediction_N, expected, atol=1e-5)
            self.assertTrue((predictions.state_origin_source_row == 0).all())
            np.testing.assert_array_equal(predictions.last_observed_source_row, rows - 1)
            table = pd.read_csv(output / "comparison.csv")
            self.assertEqual(table.state_mode.tolist(), ["window", "persistent"])
            for flag in (["--missing-block", "1"], ["--diagnostics"]):
                with patch.object(sys, "argv", [*argv[:-1], str(Path(directory) / "invalid"), *flag]), \
                     patch("src.evaluate.prepare_data") as loader, \
                     self.assertRaisesRegex(ValueError, "clean only"):
                    evaluate.main()
                loader.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available on this machine")
    def test_recording_cuda_prediction_and_gradients(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        model = PGNDModel(latent_dim=2, encoding_dim=3, hidden_dim=4)
        recordings = [torch.randn(n, 2) / 10 for n in (11, 9)]
        expected = evaluate.predict_recordings(model, recordings, 4, 4)
        actual = evaluate.predict_recordings(model, recordings, 4, 4, device="cuda")
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        recordings = [x.cuda() for x in recordings]
        optimizer = SimpleNamespace(step=Mock(), zero_grad=lambda: model.zero_grad(set_to_none=True))
        y = torch.zeros(sum(len(range(4, len(x), 2)) for x in recordings), 1, device="cuda")
        mse, residual, updates = train.train_recording_epoch(model, recordings, y, optimizer, 5, 4, 4, 2, .001)
        self.assertTrue(np.isfinite([mse, residual]).all())
        self.assertEqual(updates, (len(y) + 4) // 5)

    def test_missing_blocks_hold_only_the_past_and_do_not_mutate(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 6, 2)
        saved = x.clone()
        masked = hold_missing_block(x, [2, 3], [2, 3])
        expected = x.clone()
        expected[0, 2:4] = x[0, 1]
        expected[1, 3:] = x[1, 2]
        torch.testing.assert_close(masked, expected, rtol=0, atol=0)
        torch.testing.assert_close(x, saved, rtol=0, atol=0)
        changed = x.clone()
        changed[0, 2:4] = 10000
        changed[1, 3:] = -10000
        torch.testing.assert_close(hold_missing_block(changed, [2, 3], [2, 3]), masked)
        changed[0, 4:] = -20000  # Later observed data cannot change the held block.
        torch.testing.assert_close(hold_missing_block(changed, [2, 3], [2, 3])[0, 2:4], masked[0, 2:4])
        self.assertIs(hold_missing_block(x, 6, 0), x)
        for starts, lengths in ((0, 1), (2, -1), (3, 4), (1.5, 1)):
            with self.assertRaises(ValueError):
                hold_missing_block(x, starts, lengths)

    def test_augmentation_is_reproducible_clean_mixture_and_rng_independent(self):
        x = torch.arange(256 * 8 * 2, dtype=torch.float32).reshape(256, 8, 2)
        generator = np.random.default_rng(7)
        before = repr(generator.bit_generator.state)
        self.assertIs(augment_missing_blocks(x, 0, generator), x)
        self.assertEqual(before, repr(generator.bit_generator.state))
        rng = torch.random.get_rng_state()
        augmented = augment_missing_blocks(x, 3, generator)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        torch.randn(25)  # A model's RNG consumption must not affect mask generation.
        torch.testing.assert_close(augmented, augment_missing_blocks(x, 3, np.random.default_rng(7)))
        changed = (augmented != x).any(-1)
        self.assertFalse(changed[:, 0].any())
        self.assertTrue((changed.sum(-1) <= 3).all())
        self.assertTrue((changed.sum(-1) == 0).any())
        self.assertTrue((changed.sum(-1) > 0).any())
        for row in changed:
            positions = row.nonzero().flatten()
            if len(positions) > 1:
                self.assertTrue((positions.diff() == 1).all())
        for model in (PGNDModel(), RNN(), LSTM(), GRU(), CNNGRU()):
            self.assertTrue(torch.isfinite(model(augmented[:2] / 100)).all())
        with self.assertRaises(ValueError):
            augment_missing_blocks(x, 8, generator)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available on this machine")
    def test_missing_blocks_cuda_matches_cpu(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 6, 2)
        expected = augment_missing_blocks(x, 3, np.random.default_rng(7))
        actual = augment_missing_blocks(x.cuda(), 3, np.random.default_rng(7))
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def test_next_sample_alignment(self):
        x = np.arange(24).reshape(12, 2)
        y = np.arange(12) + 100
        windows, targets, rows = create_sequences(x, y, np.arange(12), 4, 3)
        np.testing.assert_array_equal(rows, [4, 7, 10])
        np.testing.assert_array_equal(targets, y[rows])
        for window, row in zip(windows, rows):
            np.testing.assert_array_equal(window, x[row - 4:row])

    def test_matched_context_targets_in_every_partition(self):
        names = [f"V300_Case{i}_CutFre20.xls" for i in (1, 7)]
        frame = pd.DataFrame({"distance": np.arange(48) * (300 / 3.6) * 0.001,
                              "acceleration": np.arange(48), "displacement": np.arange(48) * 2,
                              "force": np.arange(48) + 100}, index=np.arange(100, 148))
        with patch("src.data.read_recording", return_value=frame):
            short = prepare_data(names[:1], names[1:], sequence_length=4, target_start=8,
                                 train_stride=3, validation_fraction=0.25)
            long = prepare_data(names[:1], names[1:], sequence_length=8, target_start=8,
                                train_stride=3, validation_fraction=0.25)
        self.assertEqual(short[2], long[2])
        for split in ("train", "validation", "test"):
            np.testing.assert_array_equal(short[0][split][0], long[0][split][0][:, -4:])
            np.testing.assert_array_equal(short[0][split][1], long[0][split][1])
            pd.testing.assert_frame_equal(short[1][split], long[1][split])
        for anchor in (3, 48):
            with self.assertRaises(ValueError):
                create_sequences(np.zeros((48, 2)), np.zeros(48), np.arange(48), 4, target_start=anchor)

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

    def test_causal_initialization_and_same_start_control(self):
        X = torch.randn(2, 6, 2, requires_grad=True)
        shared = None
        for kind in ("first", "delayed", "history"):
            torch.manual_seed(10)
            model = AblationPGND(latent_dim=2, encoding_dim=3, hidden_dim=4,
                              initialization=kind, warmup_steps=3)
            core = {k: v for k, v in model.state_dict().items() if not k.startswith("initial_state.0.")}
            if shared is None:
                shared = core
            for name in core:
                torch.testing.assert_close(core[name], shared[name], rtol=0, atol=0)
            prediction, details = model(X, return_details=True)
            start = 0 if kind == "first" else 2
            self.assertEqual(details["states"].shape, (2, 7 - start, 4))
            self.assertAlmostEqual(details["times"][0].item(), start * 0.001)
            torch.testing.assert_close(details["solver_times"], torch.arange(start, 7, dtype=X.dtype))
            initial_input = X[:, :3].flatten(1) if kind == "history" else X[:, start]
            torch.testing.assert_close(details["states"][:, 0], model.initial_state(initial_input))
            initial_gradient = torch.autograd.grad(details["states"][:, 0].sum(), X, retain_graph=True)[0]
            self.assertEqual(initial_gradient[:, start + 1:].abs().sum().item(), 0.)
            if kind == "history":
                self.assertGreater(initial_gradient[:, :2].abs().sum().item(), 0.)
            else:
                self.assertEqual(initial_gradient[:, :start].abs().sum().item(), 0.)
            fast, residual = model(X, return_residual=True)
            torch.testing.assert_close(prediction, fast)
            torch.testing.assert_close(details["residual_loss"], residual)
            probe = evaluate.model_probe(model, X.detach(), torch.zeros(2))
            self.assertEqual(probe["forward_nfe"], 4 * (6 - start))
            self.assertEqual(probe["integration_intervals"], 6 - start)
        with self.assertRaises(ValueError):
            AblationPGND(initialization="history", warmup_steps=7)(X)

    def test_residual_removal_and_dynamics_controls(self):
        X, y = torch.randn(2, 4, 2), torch.randn(2)
        torch.manual_seed(11)
        reference = AblationPGND(latent_dim=2, encoding_dim=3, hidden_dim=4)
        for kind in ("structured", "second_order", "first_order"):
            torch.manual_seed(11)
            model = AblationPGND(latent_dim=2, encoding_dim=3, hidden_dim=4,
                              dynamics_kind=kind, use_residual=False)
            for name, value in reference.state_dict().items():
                if not name.startswith("dynamics."):
                    torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
            self.assertFalse(any(name.startswith("dynamics.residual.") for name, _ in model.named_parameters()))
            prediction, residual = model(X, return_residual=True)
            if kind == "structured":
                torch.testing.assert_close(prediction, reference(X), rtol=0, atol=0)
            self.assertEqual(residual.item(), 0.)
            self.assertFalse(residual.requires_grad)
            prediction.sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            probe = evaluate.model_probe(model, X, y)
            self.assertEqual(probe["weighted_residual_gradient_norm"], 0.)
            self.assertEqual("energy_rate_positive_fraction" in probe, kind == "structured")
            z, h = torch.randn(2, 4), torch.randn(2, 3)
            field = model.dynamics(torch.tensor(0.), z, h)
            self.assertEqual(field.shape, z.shape)
            if kind != "first_order":
                torch.testing.assert_close(field[:, :2], z[:, 2:])
        model = AblationPGND(latent_dim=2, encoding_dim=3, hidden_dim=4)
        torch.nn.init.normal_(model.dynamics.residual[-1].weight, std=0.02)
        probe = evaluate.model_probe(model, X, y, residual_weight=0.)
        self.assertGreater(probe["residual_loss"], 0.)
        self.assertEqual(probe["weighted_residual_gradient_norm"], 0.)
        self.assertGreater(probe["gradient_norm_residual"], 0.)  # Learns through prediction, despite lambda=0.

    def test_frozen_residual_rescaling_invariance(self):
        for initialization in ("first", "history"):
            model = AblationPGND(latent_dim=2, encoding_dim=3, hidden_dim=4,
                              initialization=initialization, warmup_steps=3).double()
            torch.nn.init.normal_(model.dynamics.residual[-1].weight, std=0.02)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            rng = torch.random.get_rng_state()
            X = torch.zeros(2, 4, 2, dtype=torch.float64)
            for row in evaluate.residual_scaling_check(model, X, 10.):
                self.assertAlmostEqual(row["loss_ratio"], row["expected_loss_ratio"], places=9)
                self.assertLess(row["max_prediction_change_N"], 1e-10)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_whole_case_validation_and_scaling(self):
        names = ["V300_Case1_CutFre20.xls", "V350_Case2_CutFre20.xls", "V380_Case7_CutFre20.xls"]
        frames = {}
        for i, name in enumerate(names):
            frames[name] = pd.DataFrame({
                "distance": np.arange(16) * (int(name[1:4]) / 3.6) * 0.001,
                "acceleration": np.arange(16) + i * 1000.,
                "displacement": np.arange(16) * 2 + i * 1000.,
                "force": np.arange(16) + i * 1000.,
            })
        with patch("src.data.read_recording", side_effect=lambda name, *_: frames[name]):
            arrays, meta, stats, _ = prepare_data(
                names[:1], names[2:], validation_files=names[1:2],
                sequence_length=4, validation_fraction=None)
        self.assertEqual(stats["force_mean"], 7.5)
        self.assertEqual(stats["input_mean"], [7.5, 15.])
        for split, name in zip(("train", "validation", "test"), names):
            self.assertEqual(set(meta[split].file), {name})
            self.assertEqual(len(arrays[split][1]), 12)
            np.testing.assert_array_equal(meta[split].source_row, np.arange(4, 16))
        for validation in ([], ["V350_Case1_CutFre200.xls"], [names[2]]):
            with self.assertRaises(ValueError):
                prepare_data(names[:1], names[2:], validation_files=validation)

    def test_probe_does_not_change_weights_gradients_mode_or_rng(self):
        X, target = torch.randn(3, 4, 2), torch.randn(3)
        pgnd = PGNDModel(latent_dim=2, encoding_dim=3, hidden_dim=4)
        torch.nn.init.normal_(pgnd.dynamics.residual[-1].weight, std=0.02)
        for model in (pgnd, RNN(), LSTM(), GRU(), CNNGRU()):
            model.eval()
            before = {k: v.clone() for k, v in model.state_dict().items()}
            for p in model.parameters():
                if p.requires_grad:
                    p.grad = torch.ones_like(p)
            rng = torch.random.get_rng_state()
            probe = evaluate.model_probe(model, X, target, 0.001)
            self.assertFalse(model.training)
            self.assertGreater(probe["mse_gradient_norm"], 0)
            self.assertEqual(probe["forward_nfe"], 16 if isinstance(model, PGNDModel) else 0)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(before[name], value))
            for p in model.parameters():
                if p.requires_grad:
                    self.assertTrue(torch.equal(p.grad, torch.ones_like(p)))
            if isinstance(model, PGNDModel):
                self.assertGreater(probe["weighted_residual_gradient_norm"], 0)
                self.assertLessEqual(probe["damping_power_mean"], 0)
                self.assertEqual(len(model.dynamics._forward_hooks), 0)

    def test_solver_check_restores_model(self):
        model = PGNDModel(latent_dim=2, encoding_dim=3, hidden_dim=4).double()
        X, target = np.zeros((2, 4, 2)), np.zeros(2)
        saved = (model.method, model.step_size, model.rtol, model.atol)
        before = model(torch.as_tensor(X)).detach()
        rows = evaluate.solver_check(model, X, target, 10.)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["rms_change_N"], 0)
        self.assertEqual(rows[0]["forward_nfe"], 16)
        self.assertEqual(rows[1]["forward_nfe"], 32)
        self.assertEqual(rows[2]["forward_nfe"], 64)
        self.assertEqual(saved, (model.method, model.step_size, model.rtol, model.atol))
        self.assertTrue(model.training)
        torch.testing.assert_close(before, model(torch.as_tensor(X)), rtol=0, atol=0)
        with patch.object(model, "forward", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaises(RuntimeError):
                evaluate.solver_check(model, X, target, 10.)
        self.assertEqual(saved, (model.method, model.step_size, model.rtol, model.atol))
        self.assertEqual(len(model.dynamics._forward_hooks), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available on this machine")
    def test_cuda_recurrent_probe_backward(self):
        X, y = torch.zeros(3, 4, 2, device="cuda"), torch.ones(3, device="cuda")
        for model in (RNN(), LSTM(), GRU(), CNNGRU(), PGNDModel()):
            model.cuda().eval()
            probe = evaluate.model_probe(model, X, y)
            self.assertGreater(probe["mse_gradient_norm"], 0)
            self.assertFalse(model.training)

    def test_paired_case_intervals_group_speeds(self):
        records = [{"run": run, "file": f"V{speed}_Case{case}_CutFre20.xls", "mse_N2": mse}
                   for run, mse in (("pgnd", 4.), ("gru", 9.))
                   for case in range(1, 7) for speed in (300, 350, 380)]
        paired = evaluate.paired_case_comparison(records, "pgnd").iloc[0]
        self.assertEqual(paired["cases"], 6)  # Not 18 independent recordings.
        self.assertEqual(paired.difference_rmse_N, -1.)
        self.assertEqual(paired.case_bootstrap_low_N, -1.)
        self.assertEqual(paired.case_bootstrap_high_N, -1.)
        small = [r for r in records if "_Case1_" in r["file"] or "_Case2_" in r["file"]]
        paired = evaluate.paired_case_comparison(small, "pgnd").iloc[0]
        self.assertIsNone(paired.case_bootstrap_low_N)
        self.assertEqual(paired.interval_status, "too_few_cases")

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

    def test_missing_block_evaluation_has_common_inputs_targets_and_provenance(self):
        import hashlib
        import json

        X = np.arange(32, dtype=np.float32).reshape(4, 4, 2) / 10
        original = X.copy()
        normalization = {"input_mean": [0., 0.], "input_std": [1., 1.],
                         "force_mean": 100., "force_std": 10.}
        y = np.arange(4, dtype=np.float32)
        meta = pd.DataFrame({"file": ["V300_Case5_CutFre20.xls"] * 4,
                             "source_row": np.arange(4, 8), "force_N": y * 10 + 100})
        prepared = ({"validation": (X, y)}, {"validation": meta}, normalization, 0.001)
        with tempfile.TemporaryDirectory() as directory:
            models, paths = [], []
            for name, model, augmented in (("pgnd", PGNDModel(), 2), ("cnn_gru", CNNGRU(), 0)):
                path = Path(directory) / f"{name}.best.pt"
                save_checkpoint({
                    "model": name, "model_options": {}, "state_dict": model.state_dict(),
                    "protocol": "case-validation-training-standardization-v1",
                    "data_settings": {"sequence_length": 4}, "normalization": normalization,
                    "train_files": ["train.xls"], "test_files": ["test.xls"], "data_sha256": {},
                    "sample_interval": 0.001, "seed": 0, "epochs": 1, "batch_size": 2, "optimizer": {},
                    "checkpoint_kind": "best_validation", "selected_epoch": 1,
                    "history": [{"epoch": 1, "train_mse_scaled": 1., "val_mse_scaled": 2.}],
                    "input_augmentation": {"kind": "causal-held-block-v1", "max_length": augmented},
                }, path)
                paths.append(path)
                models.append(model)
            hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            for gap in (0, 2):
                output = Path(directory) / f"gap{gap}"
                argv = ["evaluate", *map(str, paths), "--split", "validation", "--missing-block", str(gap),
                        "--preload-data", "--output-dir", str(output)]
                with patch.object(sys, "argv", argv), patch("src.evaluate.prepare_data", return_value=prepared), \
                     contextlib.redirect_stdout(io.StringIO()):
                    evaluate.main()
                inputs = hold_missing_block(torch.from_numpy(original), 4 - gap, gap)
                for model, path in zip(models, paths):
                    prediction = evaluate.predict(model, inputs).astype(float) * 10 + 100
                    frame = pd.read_csv(output / f"{path.stem}.predictions.csv")
                    np.testing.assert_allclose(frame.prediction_N, prediction)
                    np.testing.assert_array_equal(frame.force_N, meta.force_N)
                    np.testing.assert_array_equal(frame.source_row, meta.source_row)
                    np.testing.assert_array_equal(frame.last_observed_source_row, meta.source_row - 1 - gap)
                condition = json.loads((output / "evaluation.json").read_text())
                self.assertEqual(condition["missing_block"], gap)
                self.assertEqual(condition["checkpoints"], hashes)
                table = pd.read_csv(output / "comparison.csv")
                self.assertEqual(table.train_max_missing_block.tolist(), [2, 0])
                self.assertTrue((table.missing_block == gap).all())
            np.testing.assert_array_equal(X, original)
            self.assertEqual(hashes, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
            with patch.object(sys, "argv", ["evaluate", *map(str, paths), "--missing-block", "4",
                                            "--output-dir", str(Path(directory) / "invalid")]), \
                 patch("src.evaluate.prepare_data") as loader, self.assertRaisesRegex(ValueError, "leave at least"):
                evaluate.main()
            loader.assert_not_called()

    def test_diagnostics_cli_is_validation_only(self):
        with patch.object(sys, "argv", ["evaluate", "unused.pt", "--diagnostics", "--output-dir", "unused"]), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluate.main()

    def test_plot_does_not_label_an_ablation_as_original_pgnd(self):
        import os
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"MPLCONFIGDIR": directory}):
            from src import plot
            path = Path(directory)
            pd.DataFrame({"epoch": [1], "train_mse_scaled": [1.], "val_mse_scaled": [1.]}).to_csv(
                path / "control.history.csv", index=False)
            pd.DataFrame({"file": ["synthetic.xls"], "source_row": [4], "distance_m": [1.],
                          "force_N": [100.], "prediction_N": [101.]}).to_csv(path / "control.predictions.csv", index=False)
            for dynamics, residual, expected in (("first_order", False, "control"),
                                                 ("structured", False, "control"),
                                                 ("structured", True, "PGND-0")):
                pd.DataFrame([{"run": "control", "model": "pgnd", "dynamics": dynamics,
                               "initialization": "first", "residual_enabled": residual,
                               "residual_weight": 0.001 if residual else 0.}]).to_csv(path / "comparison.csv", index=False)
                with patch.object(sys, "argv", ["plot", directory]), patch("src.plot.plot_losses") as losses, \
                     patch("src.plot.plot_metrics"), patch("src.plot.plot_force"), \
                     contextlib.redirect_stdout(io.StringIO()):
                    plot.main()
                self.assertEqual(losses.call_args.args[1], [expected])

    def test_train_cli_saves_augmentation_settings_without_training(self):
        X, y = np.zeros((2, 8, 2), dtype=np.float32), np.zeros(2, dtype=np.float32)
        normalization = {"input_mean": [0., 0.], "input_std": [1., 1.],
                         "force_mean": 100., "force_std": 10.}
        prepared = ({s: (X, y) for s in ("train", "validation", "test")}, {}, normalization, 0.001)
        names = [f"V300_Case{i}_CutFre20.xls" for i in (1, 7, 5)]
        with tempfile.TemporaryDirectory() as directory:
            for name in names:
                (Path(directory) / name).write_bytes(b"synthetic hash fixture; never parsed")
            for label, flags in (("original", []), ("augmented", ["--max-missing-block", "3"]),
                                 ("persistent", ["--persistent-state"]),
                                 ("unpenalized", ["--residual-weight", "0"])):
                output = Path(directory) / f"{label}.pt"
                argv = ["train", "--model", "pgnd", "--train-pattern", "train", "--validation-pattern", "val",
                        "--test-pattern", "test", "--data-dir", directory, "--output", str(output),
                        "--sequence-length", "8", "--target-start", "8", "--epochs", "1", *flags]
                with patch.object(sys, "argv", argv), patch("src.train.get_files", side_effect=[[n] for n in names]), \
                     patch("src.train.prepare_data", return_value=prepared), \
                     patch("src.train.fit", return_value=[{"epoch": 1, "val_mse_scaled": 1.}]) as fit, \
                     contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                saved = torch.load(output, weights_only=True)
                self.assertEqual(saved["data_settings"]["target_start"], 8)
                self.assertEqual(saved["data_settings"]["validation_files"], names[2:])
                expected_weight = 0. if label == "unpenalized" else 0.001
                self.assertEqual(saved["residual_weight"], expected_weight)
                self.assertEqual(fit.call_args.args[10], expected_weight)
                expected_gap = 3 if label == "augmented" else 0
                self.assertEqual(saved["input_augmentation"]["max_length"], expected_gap)
                self.assertEqual(fit.call_args.kwargs["max_missing_block"], expected_gap)
                self.assertEqual(saved.get("state_mode", "window"), "persistent" if label == "persistent" else "window")
                self.assertEqual(fit.call_args.kwargs["persistent_state"], label == "persistent")
                restored = PGNDModel(**saved["model_options"])
                restored.load_state_dict(saved["state_dict"])
                torch.testing.assert_close(restored(torch.from_numpy(X)), fit.call_args.args[0](torch.from_numpy(X)))
            with patch.object(sys, "argv", ["train", "--model", "gru", "--train-pattern", "train",
                                            "--test-pattern", "test", "--output", str(output), "--no-residual"]), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.main()

    def test_ablation_checkpoint_evaluation_and_context_guards(self):
        X = np.arange(6 * 8 * 2, dtype=np.float32).reshape(6, 8, 2) / 100
        y = np.arange(6, dtype=np.float32) / 10
        normalization = {"input_mean": [0., 0.], "input_std": [1., 1.],
                         "force_mean": 100., "force_std": 10.}
        meta = pd.DataFrame({"file": ["V300_Case2_CutFre20.xls"] * 6,
                             "source_row": np.arange(8, 14), "force_N": y * 10 + 100})
        prepared = ({"validation": (X, y)}, {"validation": meta}, normalization, 0.001)
        with tempfile.TemporaryDirectory() as directory:
            paths, checkpoints, expected = [], [], {}
            cases = [("short", 4, {}), ("history", 8, {"initialization": "history", "warmup_steps": 3}),
                     ("r0", 8, {"use_residual": False}),
                     ("second", 8, {"dynamics_kind": "second_order"}),
                     ("first", 8, {"dynamics_kind": "first_order"})]
            for label, length, extra in cases:
                options = {"latent_dim": 2, "encoding_dim": 3, "hidden_dim": 4, **extra}
                model = AblationPGND(**options)
                checkpoint = {
                    "model": "pgnd", "model_options": options, "state_dict": model.state_dict(),
                    "protocol": "case-validation-training-standardization-v1",
                    "data_settings": {"sequence_length": length, "target_start": 8, "train_stride": 2},
                    "train_files": ["train.xls"], "test_files": ["test.xls"], "data_sha256": {},
                    "normalization": normalization, "sample_interval": 0.001, "seed": 0,
                    "epochs": 1, "completed_epochs": 1, "batch_size": 2, "optimizer": {},
                    "window_counts": {"train": 6}, "checkpoint_kind": "best_validation",
                    "selected_epoch": 1, "history": [{"epoch": 1, "val_mse_scaled": 1.}],
                    "residual_weight": 0.001 if model.use_residual else 0.,
                }
                path = Path(directory) / f"{label}.pt"
                save_checkpoint(checkpoint, path)
                paths.append(str(path))
                checkpoints.append(checkpoint)
                expected[label] = evaluate.predict(model, X[:, -length:]).astype(float) * 10 + 100
            output = Path(directory) / "comparison"
            argv = ["evaluate", *paths, "--split", "validation", "--diagnostics", "--preload-data",
                    "--output-dir", str(output)]
            with patch.object(sys, "argv", argv), patch("src.evaluate.prepare_data", return_value=prepared) as loader, \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            self.assertEqual(loader.call_args.kwargs["sequence_length"], 8)
            self.assertEqual(loader.call_args.kwargs["target_start"], 8)
            for label in expected:
                saved = pd.read_csv(output / f"{label}.predictions.csv")
                np.testing.assert_allclose(saved.prediction_N, expected[label])
                np.testing.assert_array_equal(saved.source_row, meta.source_row)
                self.assertTrue((output / f"{label}.solver.csv").is_file())
            self.assertTrue((output / "history.residual_scaling.csv").is_file())
            self.assertFalse((output / "r0.residual_scaling.csv").exists())
            self.assertNotIn("damping_power_mean", pd.read_csv(output / "first.probe.csv"))
            self.assertEqual(pd.read_csv(output / "history.probe.csv").integration_intervals.iloc[0], 6)
            for change in ({"target_start": None}, {"train_stride": 1}):
                checkpoints[1]["data_settings"] = {"sequence_length": 8, "target_start": 8,
                                                    "train_stride": 2, **change}
                save_checkpoint(checkpoints[1], Path(paths[1]))
                with patch.object(sys, "argv", [*argv[:-1], str(output / "invalid")]), \
                     patch("src.evaluate.prepare_data") as loader, self.assertRaisesRegex(ValueError, "data_settings"):
                    evaluate.main()
                loader.assert_not_called()

    def test_validation_diagnostics_cli_exports_and_keeps_checkpoints(self):
        import hashlib
        import json

        X, y = np.zeros((6, 4, 2), dtype=np.float32), np.arange(6, dtype=np.float32) / 10
        normalization = {"input_mean": [0., 0.], "input_std": [1., 1.],
                         "force_mean": 100., "force_std": 10.}
        meta = pd.DataFrame({"file": ["V300_Case2_CutFre20.xls"] * 3 + ["V350_Case3_CutFre20.xls"] * 3,
                             "source_row": [4, 5, 6] * 2, "force_N": y * 10 + 100})
        prepared = ({"validation": (X, y)}, {"validation": meta}, normalization, 0.001)
        options = {"latent_dim": 2, "encoding_dim": 3, "hidden_dim": 4}
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, model, model_options in (("pgnd", PGNDModel(**options), options), ("gru", GRU(), {})):
                path = Path(directory) / f"{name}.best.pt"
                save_checkpoint({
                    "model": name, "model_options": model_options, "state_dict": model.state_dict(),
                    "protocol": "case-validation-training-standardization-v1",
                    "data_settings": {"validation_files": sorted(meta.file.unique())},
                    "train_files": ["V300_Case1_CutFre20.xls"], "test_files": ["V300_Case7_CutFre20.xls"],
                    "normalization": normalization, "data_sha256": {}, "sample_interval": 0.001,
                    "seed": 0, "epochs": 1, "batch_size": 2, "optimizer": {},
                    "window_counts": {"train": 6}, "checkpoint_kind": "best_validation",
                    "selected_epoch": 1, "history": [{"epoch": 1, "val_mse_scaled": 1., "seconds": 2.}],
                    "residual_weight": 0.001 if name == "pgnd" else 0.,
                }, path)
                paths.append(path)
            hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            output = Path(directory) / "diagnostics"
            argv = ["evaluate", *map(str, paths), "--split", "validation", "--diagnostics",
                    "--preload-data", "--output-dir", str(output)]
            with patch.object(sys, "argv", argv), patch("src.evaluate.prepare_data", return_value=prepared), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            self.assertEqual(json.loads((output / "diagnostics.json").read_text())["checkpoints"], hashes)
            self.assertEqual(hashes, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
            self.assertEqual(len(pd.read_csv(output / "probe_targets.csv")), 6)
            self.assertEqual(len(pd.read_csv(output / "pgnd.best.solver.csv")), 4)
            self.assertEqual(pd.read_csv(output / "gru.best.probe.csv").forward_nfe.iloc[0], 0)
            self.assertEqual(pd.read_csv(output / "paired_cases.csv").interval_status.iloc[0], "too_few_cases")
            self.assertTrue((pd.read_csv(output / "comparison.csv").budget_updates == 3).all())
            with patch.object(sys, "argv", argv), self.assertRaises(FileExistsError):
                evaluate.main()


if __name__ == "__main__":
    unittest.main()
