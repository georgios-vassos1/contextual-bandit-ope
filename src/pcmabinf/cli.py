from __future__ import annotations

import pickle
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from pcmabinf.logging_policy import LoggingConfig
from pcmabinf.metrics import compute_coverage_metrics
from pcmabinf.policy import make_target_policy
from pcmabinf.simulate import run_bandit_simulations, run_ope_simulations
from pcmabinf.world import OpenMLCC18World


@click.group()
def main() -> None:
    """pcmabinf — Post Contextual Multi-Armed Bandit Inference."""


@main.command("run")
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=Path.home() / "pcmabinf" / "OpenML-CC18",
    show_default=True,
)
@click.option("--task-id", type=str, default=None, help="Single task ID; omit to run all.")
@click.option("--batch-count", type=int, default=100, show_default=True)
@click.option("--batch-size", type=int, default=100, show_default=True)
@click.option("--simulations", type=int, default=64, show_default=True)
@click.option("--reward-variance", type=float, default=1.0, show_default=True)
@click.option("--epsilon-multiplier", type=float, default=0.01, show_default=True)
@click.option("--n-jobs", type=int, default=-1, show_default=True)
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility.")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("results"),
    show_default=True,
)
def run_cmd(
    data_dir: Path,
    task_id: str | None,
    batch_count: int,
    batch_size: int,
    simulations: int,
    reward_variance: float,
    epsilon_multiplier: float,
    n_jobs: int,
    seed: int | None,
    output_dir: Path,
) -> None:
    """Run bandit simulations and OPE for one or all OpenML-CC18 tasks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine task list
    if task_id is not None:
        task_ids = [task_id]
    else:
        task_ids = [p.name for p in sorted(data_dir.iterdir()) if not p.name.startswith(".")]

    click.echo(f"Running {len(task_ids)} task(s).")

    logging_config = LoggingConfig(
        batch_count=batch_count,
        batch_size=batch_size,
        strategy="greedy",
        epsilon_multiplier=epsilon_multiplier,
        outcome_model=DecisionTreeRegressor(),
    )

    all_tasks_rows: list[dict] = []

    for j, tid in enumerate(task_ids):
        t0 = time.time()
        click.echo(f"\n[{j + 1}/{len(task_ids)}] task_id={tid}")

        try:
            world = OpenMLCC18World(task_id=tid, data_dir=data_dir, reward_variance=reward_variance)
        except FileNotFoundError:
            click.echo(f"  Skipping — data file not found for task {tid}")
            continue

        # Bandit simulations
        bandit_data_list = run_bandit_simulations(
            world, logging_config, n_simulations=simulations, n_jobs=n_jobs, seed=seed
        )

        # Target policies
        train_size = len(bandit_data_list[0].X)
        target_policies = {
            "linear": make_target_policy(
                "contextual",
                world,
                outcome_model=LinearRegression(),
                train_sample_size=train_size,
            ),
            "tree": make_target_policy(
                "contextual",
                world,
                outcome_model=DecisionTreeRegressor(),
                train_sample_size=train_size,
            ),
            "arm0": make_target_policy("non_contextual_constant_0", world),
            "arm1": make_target_policy("non_contextual_constant_1", world),
        }

        row: dict = {"task_id": tid}
        for name, policy in target_policies.items():
            ope_results = run_ope_simulations(
                bandit_data_list,
                world,
                policy,
                outcome_model=LinearRegression(),
                n_jobs=n_jobs,
            )
            metrics = compute_coverage_metrics(ope_results)
            for key, val in metrics.items():
                row[f"{name}_{key}"] = float(val)

        all_tasks_rows.append(row)

        t1 = time.time()
        click.echo(f"  done in {t1 - t0:.1f}s")

    df = pd.DataFrame(all_tasks_rows)
    out_path = output_dir / "all_tasks_df_data"
    df.to_pickle(out_path)
    click.echo(f"\nResults saved to {out_path}")


@main.command("visualize")
@click.argument("results_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", type=click.Path(path_type=Path), default=None)
def visualize_cmd(results_path: Path, output: Path | None) -> None:
    """Generate coverage scatter-plot grid from a saved results file."""
    from pcmabinf.viz import visualize_coverage

    visualize_coverage(results_path=results_path, output_path=output)
    if output:
        click.echo(f"Figure saved to {output}")
