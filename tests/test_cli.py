"""Tests for the click CLI."""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from pcmabinf.cli import main


def test_run_single_task(tmp_data_dir: Path, tmp_path: Path) -> None:
    """``pcmabinf run`` should complete without error and write a results file."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--data-dir", str(tmp_data_dir),
            "--task-id", "999",
            "--batch-count", "3",
            "--batch-size", "20",
            "--simulations", "2",
            "--n-jobs", "1",
            "--output-dir", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "all_tasks_df_data").exists()


def test_run_missing_task_is_skipped(tmp_data_dir: Path, tmp_path: Path) -> None:
    """An unknown task ID should be skipped gracefully, not raise."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--data-dir", str(tmp_data_dir),
            "--task-id", "does_not_exist",
            "--batch-count", "2",
            "--batch-size", "10",
            "--simulations", "1",
            "--n-jobs", "1",
            "--output-dir", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0
    assert "Skipping" in result.output


def test_run_produces_results_with_expected_columns(tmp_data_dir: Path, tmp_path: Path) -> None:
    import pandas as pd

    runner = CliRunner()
    runner.invoke(
        main,
        [
            "run",
            "--data-dir", str(tmp_data_dir),
            "--task-id", "999",
            "--batch-count", "3",
            "--batch-size", "20",
            "--simulations", "2",
            "--n-jobs", "1",
            "--output-dir", str(tmp_path / "out"),
        ],
    )
    df = pd.read_pickle(tmp_path / "out" / "all_tasks_df_data")
    assert "task_id" in df.columns
    # Columns follow the pattern {policy}_{estimator}_{mean|stderr}
    assert "linear_dm_mean" in df.columns
    assert "linear_dm_stderr" in df.columns
    assert "arm0_truth_mean" in df.columns
