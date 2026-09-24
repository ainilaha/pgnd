"""Publication figures from saved evaluation CSVs; no models, datasets or training.

Loss curves compare the same standardized-force MSE, NOT PGND's regularized
total objective against validation MSE. Force curves share one recording and
the same target rows; unrelated recordings are never joined into one curve.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless remote servers; set before importing pyplot.
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


LABELS = {"rnn": "RNN", "lstm": "LSTM", "gru": "GRU", "cnn_gru": "CNN–GRU",
          "pgnd": "PGND-0", "pgnd_obs": "PGND + direct", "direct": "Direct only"}
COLORS = {"rnn": "#9467bd", "lstm": "#0072B2", "gru": "#009E73",
          "cnn_gru": "#E69F00", "pgnd": "#D55E00", "pgnd_obs": "#CC79A7",
          "direct": "#666666"}
MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def save_figure(fig, output_dir, name):
    fig.savefig(output_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_losses(histories, labels, colors, output_dir):
    for split, column in (("Training", "train_mse_scaled"),
                          ("Validation", "val_mse_scaled")):
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        for i, (history, label, color) in enumerate(zip(histories, labels, colors)):
            ax.plot(history["epoch"], history[column], label=label, color=color,
                    marker=MARKERS[i % len(MARKERS)], markersize=4,
                    markevery=max(1, len(history) // 12))
        ax.set_title(f"{split} loss")
        ax.set_ylabel("Standardized MSE")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
        save_figure(fig, output_dir, f"{split.lower()}_loss_curves")


def plot_metrics(table, labels, colors, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")
    metrics = [("mae_N", "MAE (N) ↓"), ("rmse_N", "RMSE (N) ↓"),
               ("mse_N2", "MSE (N²) ↓"), ("r2", "R² ↑")]
    for ax, (column, title) in zip(axes.flat, metrics):
        values = table[column].to_numpy(dtype=float)
        # Undefined R² (constant targets) is labeled, never silently plotted as zero.
        bars = ax.bar(np.arange(len(table)), np.nan_to_num(values, nan=0.0), color=colors)
        ax.bar_label(bars, labels=[f"{v:.3f}" if np.isfinite(v) else "undefined" for v in values],
                     padding=3, fontsize=8)
        ax.set_xticks(np.arange(len(table)), labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.margins(y=0.2)
    save_figure(fig, output_dir, "metric_comparison")


def plot_force(predictions, labels, colors, recording, start, points, output_dir):
    reference = predictions[0]
    selected = reference[reference["file"] == recording].iloc[start:start + points]
    if selected.empty:
        raise ValueError("No target rows in the requested recording/window.")
    # No sorting, smoothing, resampling, or concatenation across recordings.
    if not (selected["source_row"].diff().dropna() == 1).all():
        raise ValueError("Force curves require consecutive source rows.")
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.plot(selected["distance_m"], selected["force_N"], color="black",
            linewidth=2, label="Ground truth")
    for i, (frame, label, color) in enumerate(zip(predictions, labels, colors)):
        ax.plot(selected["distance_m"], frame.loc[selected.index, "prediction_N"],
                color=color, linestyle="--", label=label, marker=MARKERS[i % len(MARKERS)],
                markersize=4, markerfacecolor="none", markevery=max(1, len(selected) // 15))
    ax.set_ylabel("Contact force (N)")
    ax.set_xlabel("Distance (m)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.set_title(f"{recording}\nSource rows {selected.source_row.iloc[0]}–{selected.source_row.iloc[-1]}")
    save_figure(fig, output_dir, "force_predictions")
    print(f"Force plot: {recording}, {len(selected)} targets, offset {start}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path, help="Directory produced by src.evaluate")
    parser.add_argument("--output-dir", type=Path, help="Figure directory (matching PDFs overwritten); default: evaluation_dir/figures")
    parser.add_argument("--recording", help="Exact data filename; default: first evaluated recording")
    parser.add_argument("--start", type=int, default=0, help="Zero-based offset among the recording's test targets")
    parser.add_argument("--points", type=int, default=1000, help="Maximum consecutive targets in force plot")
    args = parser.parse_args()
    if args.start < 0 or args.points < 1:
        parser.error("--start must be nonnegative and --points must be positive.")
    output_dir = args.output_dir or args.evaluation_dir / "figures"
    table = pd.read_csv(args.evaluation_dir / "comparison.csv")
    if table.empty or "run" not in table or not table["run"].is_unique:
        raise ValueError("Expected distinct run names; regenerate evaluation with the current evaluator.")
    labels = [LABELS.get(row.model, row.model) if table.model.is_unique else row.run
              for row in table.itertuples()]
    colors = [COLORS.get(model, "#0072B2") for model in table.model]
    histories, predictions = [], []
    for run in table["run"]:
        histories.append(pd.read_csv(args.evaluation_dir / f"{run}.history.csv"))
        predictions.append(pd.read_csv(args.evaluation_dir / f"{run}.predictions.csv"))
    columns = ["file", "source_row", "distance_m", "force_N"]
    reference = predictions[0][columns]
    for frame in predictions[1:]:
        if not reference.equals(frame[columns]):
            raise ValueError("Prediction files must have identical ordered test rows and ground truth.")
    recording = args.recording or reference["file"].iloc[0]
    recording_count = len(reference[reference["file"] == recording])
    if recording_count == 0:
        parser.error(f"Recording not found: {recording}")
    if args.start >= recording_count:
        parser.error("--start lies beyond the selected recording's targets.")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_losses(histories, labels, colors, output_dir)
    plot_metrics(table, labels, colors, output_dir)
    plot_force(predictions, labels, colors, recording, args.start, args.points, output_dir)
    print(f"Saved training_loss_curves, validation_loss_curves, metric_comparison "
          f"and force_predictions (PDF) to {output_dir}")


if __name__ == "__main__":
    main()
