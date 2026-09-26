"""Publication figures from supplied tables; no models, datasets or training.

Loss curves compare standardized-force prediction MSE. Force curves share one recording and
the same target rows; unrelated recordings are never joined into one curve.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless remote servers; set before importing pyplot.
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np


MARKERS = ("o", "s", "^", "D", "v", "P", "X")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def save_figure(fig, output_dir, name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_losses(histories, labels, colors, output_dir):
    if not histories or not len(histories) == len(labels) == len(colors):
        raise ValueError("Provide a label and color for every history.")
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
    if table.empty or not len(table) == len(labels) == len(colors):
        raise ValueError("Provide a label and color for every metric row.")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")
    metrics = [("mae", "MAE (N) ↓"), ("rmse", "RMSE (N) ↓"),
               ("mse", "MSE (N²) ↓"), ("r2", "R² ↑")]
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
    if (not predictions or not len(predictions) == len(labels) == len(colors)
            or start < 0 or points < 1):
        raise ValueError("Require labeled predictions, nonnegative start and positive points.")
    columns = ["file", "split", "source_row", "time_s", "distance_m", "force_N"]
    reference = predictions[0].reset_index(drop=True)
    predictions = [frame.reset_index(drop=True) for frame in predictions]
    for frame in predictions[1:]:
        if not reference[columns].equals(frame[columns]):
            raise ValueError("Compare identical ordered target rows and ground truth.")
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


def plot_sampling(recording, mask, filled, output_dir, *, start=0, points=200,
                  name="sampling"):
    """Plot supplied raw signals/mask/interpolated sensors without transforming them.

    Three aligned panels: acceleration, uplift, and complete force ground truth.
    Units follow the reference paper, Figs. 7 and 11; see data/README.md.
    Display original distance, not time inferred from nominal speed. Shading
    marks missing sensor cells only. Neither force nor source data are changed.
    """
    from src.sampling import validate_mask

    validate_mask(recording, mask)
    if not filled.index.equals(recording.samples.index) or start < 0 or points < 2:
        raise ValueError("Need aligned filled rows, nonnegative start and at least two points.")
    frame = recording.samples.iloc[start:start + points]
    if len(frame) < 2:
        raise ValueError("Not enough rows in the diagnostic segment.")
    keep = mask.loc[frame.index].to_numpy()
    distance = frame.distance_m.to_numpy()
    edges = np.r_[distance[0] - (distance[1] - distance[0]) / 2,
                  (distance[:-1] + distance[1:]) / 2,
                  distance[-1] + (distance[-1] - distance[-2]) / 2]
    transitions = np.diff(np.r_[False, ~keep, False].astype(int))
    gaps = list(zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)))
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    # Fixed margins and limits give all six conditions the same plotting area.
    fig.subplots_adjust(left=.12, right=.985, bottom=.08, top=.84, hspace=.2)
    for ax, column in zip(axes[:2], ("acceleration", "uplift")):
        for left, right in gaps:
            ax.axvspan(edges[left], edges[right], color="0.9", linewidth=0, zorder=0)
        ax.plot(distance, frame[column], color="0.25", linewidth=1.5, label="Original")
        ax.plot(distance, filled.loc[frame.index, column], color="#D55E00",
                linestyle="--", linewidth=1.2, label="Interpolated")
        ax.plot(distance[keep], frame[column].to_numpy()[keep], linestyle="none", marker="o",
                markersize=2.5, markeredgewidth=.7, markerfacecolor="none", color="#0072B2",
                markevery=max(1, int(np.ceil(keep.sum() / 80))), label="Retained")
    axes[2].plot(distance, frame.force_N, color="0.25", linewidth=1.5,
                 label="Complete ground truth")
    axes[2].set_title("Complete contact-force ground truth", loc="left", fontsize=9)
    labels = (r"Acceleration [m/s$^2$]", "Uplift [m]", "Contact force [N]")
    for ax, column, label in zip(axes, ("acceleration", "uplift", "force_N"), labels):
        # Use the complete displayed signal, never a mask-dependent autoscale.
        low, high = frame[column].min(), frame[column].max()
        padding = .08 * (high - low) if high > low else .08 * max(abs(low), 1.)
        ax.set_ylim(low - padding, high + padding)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylabel(label)
        ax.grid(alpha=.15)
    fig.align_ylabels(axes)
    handles = [*axes[0].lines, Patch(facecolor="0.9", label="Missing sensor intervals")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .935),
               ncol=2, frameon=False, fontsize=9)
    axes[-1].set_xlabel("Distance [m]")
    fig.suptitle(f"{recording.name} | {mask.attrs['pattern']} | requested retention "
                 f"{mask.attrs['retention']:.0%}\n"
                 f"Whole segment: {mask.mean():.2%} retained; displayed: {keep.mean():.1%}", fontsize=11)
    save_figure(fig, output_dir, name)
