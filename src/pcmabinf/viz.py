from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def bar_plot(
    data: NDArray[np.float64],
    metric: str,
    labels: list[str],
    truth: float | None = None,
) -> None:
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
    """
    mean_metric = np.mean(data, axis=1)
    stderr_metric = np.sqrt(np.var(data, axis=1) / data.shape[1])

    print(f"\n{metric}")
    for i, label in enumerate(labels):
        print(f"{label.upper()}: {mean_metric[i]:.4f} ({stderr_metric[i]:.4f})")

    cmap = matplotlib.colormaps["tab10"]
    colors = [cmap(i) for i in range(len(data))]
    plt.bar(x=range(len(data)), color=colors, height=mean_metric, yerr=stderr_metric)
    plt.xticks(range(len(data)), [l.upper() for l in labels], rotation=90)
    if truth is not None:
        plt.axhline(y=truth, color="black", linestyle="--")
    plt.ylabel(metric)
    plt.show()


def visualize_coverage(
    results_path: Path,
    output_path: Path | None = None,
) -> None:
    """Scatter-plot grid: CADR vs DM/IPW/DR/ADR/MRDR for each target policy.

    Parameters
    ----------
    results_path:
        Path to a pickled DataFrame produced by the ``run`` CLI command (the
        ``all_tasks_df_data`` file).
    output_path:
        If given, save the figure to this path instead of displaying it.
    """
    data = pd.read_pickle(results_path)

    estimators = ["Truth", "DM", "IPW", "DR", "ADR", "CADR", "MRDR", "CAMRDR"]
    target_policies = ["linear", "tree", "arm0", "arm1"]
    target_policy_names = [
        "linear contextual\ntarget policy",
        "tree contextual\ntarget policy",
        "arm 0 non-contextual\ntarget policy",
        "arm 1 non-contextual\ntarget policy",
    ]
    est_dict = {name: i for i, name in enumerate(estimators)}

    competitors = ["DM", "IPW", "DR", "ADR", "MRDR"]
    main = "CADR"

    fontsize = 16
    size = 3
    nrows = len(target_policies)
    ncols = len(competitors)
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(size * (ncols + 3.5), size * (nrows + 1))
    )
    fig.subplots_adjust(top=0.99, bottom=0.01, hspace=0.5, wspace=0.5)

    for t, target_policy in enumerate(target_policies):
        mean = np.array(data[f"{target_policy}_coverage_95_mean"].to_list())
        stderr = np.array(data[f"{target_policy}_coverage_95_stderr"].to_list())

        e2 = est_dict[main]
        for i1, e1 in enumerate(est_dict[c] for c in competitors):
            ax = axs[t][i1]
            colors_list = []
            for i in range(len(mean[:, e1])):
                x = mean[i, e1]
                y = mean[i, e2]
                xerr = stderr[i, e1]
                yerr = stderr[i, e2]
                if abs(x - y) <= xerr + yerr:
                    colors_list.append([0, 0, 0])
                elif x > y:
                    colors_list.append([1, 0, 0])
                else:
                    colors_list.append([0, 0, 1])
            colors_arr = np.array(colors_list, dtype=float)
            ax.scatter(x=mean[:, e1], y=mean[:, e2], zorder=3, c=colors_arr, s=50, alpha=0.5)
            ax.errorbar(
                x=mean[:, e1],
                y=mean[:, e2],
                xerr=stderr[:, e1],
                yerr=stderr[:, e2],
                c="black",
                zorder=0,
                fmt="none",
            )
            ax.set_xlabel(f"{estimators[e1]} 95% CI Coverage", fontsize=fontsize)
            ax.set_ylabel(f"{estimators[e2]} 95% CI Coverage", fontsize=fontsize)
            ax.tick_params(axis="both", which="major", labelsize=fontsize - 4)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xlim([-0.1, 1.1])
            ax.set_ylim([-0.1, 1.1])
            ax.axhline(y=0.95, linestyle="--", color="black")
            ax.axvline(x=0.95, linestyle="--", color="black")
            ax.axline(xy1=(0, 0), xy2=(10, 10), linestyle="-", color="black", alpha=0.3)
            ax.set_title(target_policy_names[t], fontsize=fontsize)

    if output_path is not None:
        fig.savefig(output_path)
    else:
        plt.show()
