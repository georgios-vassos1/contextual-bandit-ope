"""Smoke tests for viz.py — verify no crashes under a headless backend."""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # must be set before any other matplotlib import

from pcmabinf.viz import bar_plot, visualize_coverage


def _make_results_df(tmp_path: Path) -> Path:
    """Construct a minimal DataFrame matching the shape expected by visualize_coverage."""
    estimators = ["Truth", "DM", "IPW", "DR", "ADR", "CADR", "MRDR", "CAMRDR"]
    n_tasks = 5
    rng = np.random.default_rng(0)

    rows = []
    for _ in range(n_tasks):
        row: dict = {}
        for tp in ["linear", "tree", "arm0", "arm1"]:
            row[f"{tp}_coverage_95_mean"] = list(rng.uniform(0, 1, len(estimators)))
            row[f"{tp}_coverage_95_stderr"] = list(rng.uniform(0, 0.1, len(estimators)))
        rows.append(row)

    df = pd.DataFrame(rows)
    path = tmp_path / "fake_results"
    df.to_pickle(path)
    return path


def test_visualize_coverage_saves_file(tmp_path: Path) -> None:
    results_path = _make_results_df(tmp_path)
    output_path = tmp_path / "coverage.png"
    visualize_coverage(results_path=results_path, output_path=output_path)
    assert output_path.exists()


def test_bar_plot_runs_without_error() -> None:
    import matplotlib.pyplot as plt

    data = np.random.default_rng(1).uniform(0, 1, (4, 20))
    bar_plot(data=data, metric="95% CI Coverage", labels=["dm", "ips", "dr", "adr"], truth=0.95)
    plt.close("all")
