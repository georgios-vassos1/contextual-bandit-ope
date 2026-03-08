from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pcmabinf.metrics import ESTIMATOR_FIELDS

# Human-readable display labels for each estimator field name.
_DISPLAY: dict[str, str] = {
    "truth": "Truth",
    "dm": "DM",
    "ips": "IPS",
    "dr": "DR",
    "adr": "ADR",
    "cadr": "CADR",
    "mrdr": "MRDR",
    "camrdr": "CAMRDR",
}


def bar_plot(
    data: NDArray[np.float64],
    metric: str,
    labels: list[str],
    truth: float | None = None,
) -> matplotlib.axes.Axes:
    """Bar chart of *metric* with error bars.

    Parameters
    ----------
    data:
        Array of shape (n_estimators, n_simulations).
    metric:
        Y-axis label.
    labels:
        Estimator names (length n_estimators).
    truth:
        If given, draw a horizontal dashed line at this value.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object so the caller can further customise or save the figure.
    """
    means = np.mean(data, axis=1)
    stderrs = np.sqrt(np.var(data, axis=1) / data.shape[1])

    print(f"\n{metric}")
    for i, label in enumerate(labels):
        print(f"{label.upper()}: {means[i]:.4f} ({stderrs[i]:.4f})")

    cmap = matplotlib.colormaps["tab10"]
    colors = [cmap(i) for i in range(len(data))]
    fig, ax = plt.subplots()
    ax.bar(x=range(len(data)), color=colors, height=means, yerr=stderrs)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([lbl.upper() for lbl in labels], rotation=90)
    if truth is not None:
        ax.axhline(y=truth, color="black", linestyle="--")
    ax.set_ylabel(metric)
    return ax


def visualize_coverage(
    results_path: Path,
    output_path: Path | None = None,
    target_policies: list[str] | None = None,
    competitors: list[str] | None = None,
    main: str = "cadr",
) -> matplotlib.figure.Figure:
    """Scatter-plot grid: *main* estimator vs each *competitor* for each target policy.

    Reads a pickled DataFrame produced by ``pcmabinf run``.  Each row in the
    DataFrame corresponds to one task; columns are named
    ``{policy}_{estimator}_mean`` and ``{policy}_{estimator}_stderr``.

    Parameters
    ----------
    results_path:
        Path to the pickled results DataFrame.
    output_path:
        If given, save the figure to this path instead of returning it.
    target_policies:
        Policy names to plot (default: ``["linear", "tree", "arm0", "arm1"]``).
    competitors:
        Estimator field names to plot on the x-axis
        (default: ``["dm", "ips", "dr", "adr", "mrdr"]``).
    main:
        Estimator field name for the y-axis (default: ``"cadr"``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if target_policies is None:
        target_policies = ["linear", "tree", "arm0", "arm1"]
    if competitors is None:
        competitors = ["dm", "ips", "dr", "adr", "mrdr"]

    target_policy_names = {
        "linear": "linear contextual\ntarget policy",
        "tree": "tree contextual\ntarget policy",
        "arm0": "arm 0 non-contextual\ntarget policy",
        "arm1": "arm 1 non-contextual\ntarget policy",
    }

    data = pd.read_pickle(results_path)

    font_size = 16
    size = 3
    n_rows = len(target_policies)
    n_cols = len(competitors)
    # squeeze=False ensures axs is always 2-D, even for n_rows=1 or n_cols=1.
    fig, axs = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(size * (n_cols + 3.5), size * (n_rows + 1)),
        squeeze=False,
    )
    fig.subplots_adjust(top=0.99, bottom=0.01, hspace=0.5, wspace=0.5)

    # Precompute once — ci_pct is a pure function of a constant.
    ci_label = int(ci_pct(0.95))
    x_label_suffix = f"% CI Coverage"
    main_label = f"{_DISPLAY.get(main, main.upper())} {ci_label}{x_label_suffix}"

    for t, tp in enumerate(target_policies):
        # Each column is a scalar per task; collect as arrays over tasks.
        y_bar = data[f"{tp}_{main}_mean"].to_numpy(dtype=float)
        y_err = data[f"{tp}_{main}_stderr"].to_numpy(dtype=float)

        for i1, comp in enumerate(competitors):
            ax = axs[t][i1]

            x_bar = data[f"{tp}_{comp}_mean"].to_numpy(dtype=float)
            x_err = data[f"{tp}_{comp}_stderr"].to_numpy(dtype=float)

            # Color: black = overlap, red = competitor better, blue = main better.
            diff = x_bar - y_bar
            combined_err = x_err + y_err
            colors_arr = np.where(
                np.abs(diff)[:, None] <= combined_err[:, None],
                [[0, 0, 0]],
                np.where(diff[:, None] > 0, [[1, 0, 0]], [[0, 0, 1]]),
            ).astype(float)

            ax.scatter(x=x_bar, y=y_bar, zorder=3, c=colors_arr, s=50, alpha=0.5)
            ax.errorbar(
                x=x_bar,
                y=y_bar,
                xerr=x_err,
                yerr=y_err,
                c="black",
                zorder=0,
                fmt="none",
            )
            ax.set_xlabel(
                f"{_DISPLAY.get(comp, comp.upper())} {ci_label}{x_label_suffix}",
                fontsize=font_size,
            )
            ax.set_ylabel(main_label, fontsize=font_size)
            ax.tick_params(axis="both", which="major", labelsize=font_size - 4)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xlim([-0.1, 1.1])
            ax.set_ylim([-0.1, 1.1])
            ax.axhline(y=0.95, linestyle="--", color="black")
            ax.axvline(x=0.95, linestyle="--", color="black")
            ax.axline(xy1=(0, 0), xy2=(10, 10), linestyle="-", color="black", alpha=0.3)
            ax.set_title(target_policy_names.get(tp, tp), fontsize=font_size)

    if output_path is not None:
        fig.savefig(output_path)
    return fig


def ci_pct(ci_level: float) -> float:
    """Convert a coverage fraction (e.g. 0.95) to a percentage (95.0)."""
    return ci_level * 100.0
