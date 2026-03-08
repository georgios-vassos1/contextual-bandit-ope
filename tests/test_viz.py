"""Smoke tests for viz.py — verify no crashes under a headless backend."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # must be set before any other matplotlib import

from pcmabinf.metrics import ESTIMATOR_FIELDS
from pcmabinf.viz import bar_plot, visualize_coverage


def _make_results_df(tmp_path: Path) -> Path:
    """Construct a minimal DataFrame matching the format written by the CLI.

    Columns: {policy}_{estimator}_mean and {policy}_{estimator}_stderr,
    one scalar per task row.
    """
    rng = np.random.default_rng(0)
    n_tasks = 5
    rows = []
    for _ in range(n_tasks):
        row: dict = {}
        for tp in ["linear", "tree", "arm0", "arm1"]:
            for est in ESTIMATOR_FIELDS:
                row[f"{tp}_{est}_mean"] = float(rng.uniform(0, 1))
                row[f"{tp}_{est}_stderr"] = float(rng.uniform(0, 0.1))
        rows.append(row)

    df = pd.DataFrame(rows)
    path = tmp_path / "fake_results"
    df.to_pickle(path)
    return path


def test_visualize_coverage_saves_file(tmp_path: Path) -> None:
    results_path = _make_results_df(tmp_path)
    output_path = tmp_path / "coverage.png"
    fig = visualize_coverage(results_path=results_path, output_path=output_path)
    assert output_path.exists()
    import matplotlib.pyplot as plt
    plt.close("all")


def test_visualize_coverage_returns_figure(tmp_path: Path) -> None:
    import matplotlib.figure
    results_path = _make_results_df(tmp_path)
    fig = visualize_coverage(results_path=results_path)
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close("all")


def test_bar_plot_runs_without_error() -> None:
    import matplotlib.pyplot as plt

    data = np.random.default_rng(1).uniform(0, 1, (4, 20))
    ax = bar_plot(data=data, metric="95% CI Coverage", labels=["dm", "ips", "dr", "adr"], truth=0.95)
    assert ax is not None
    plt.close("all")
