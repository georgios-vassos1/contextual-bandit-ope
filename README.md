# pcmabinf

**Post Contextual Multi-Armed Bandit Inference** — off-policy evaluation (OPE) estimators for contextual bandits, evaluated on the [OpenML-CC18](https://www.openml.org/s/99) benchmark suite.

Implements DM, IPS, DR, ADR, CADR, MRDR, and CAMRDR estimators with cross-fitting, parallel simulation, and CI coverage diagnostics.

This package is a clean-room Python implementation based on the paper:

> Bibaut, A., Chambaz, A., Dimakopoulou, M., Kallus, N., & van der Laan, M. (2021).
> **Post-Contextual-Bandit Inference.**
> *arXiv:2106.00418* — <https://arxiv.org/abs/2106.00418>

and the accompanying reference notebook at
<https://github.com/mdimakopoulou/post-contextual-bandit-inference>.

---

## Background

In a contextual bandit, at each step a learner observes a context *x*, selects an arm *a*, and observes a reward. The goal of **off-policy evaluation** is to estimate how well a *target policy* π\* would have performed, using only data collected by a different *logging policy*.

This package evaluates estimators by measuring **confidence interval (CI) coverage**: across many simulation replicates, how often does the estimator's 95% CI contain the true policy value? A well-calibrated estimator should achieve ≈ 95% coverage.

The benchmark repurposes OpenML-CC18 classification tasks as bandit environments: each class label is an arm, and the reward is 1 if the chosen arm matches the true label and 0 otherwise.

---

## Installation

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> pcmabinf-pkg
cd pcmabinf-pkg
uv sync
```

For development dependencies (pytest, ruff, mypy):

```bash
uv sync --group dev
```

---

## Data

### Downloading the OpenML-CC18 benchmark

The package expects OpenML-CC18 task files in a local directory. Each file is a
pickle containing a `(features, targets)` tuple and is named by its OpenML task
ID (e.g. `10093`).

Download the data using the [OpenML Python client](https://openml.github.io/openml-python/):

```python
import pickle
from pathlib import Path
import openml

save_dir = Path.home() / "pcmabinf" / "OpenML-CC18"
save_dir.mkdir(parents=True, exist_ok=True)

benchmark = openml.study.get_suite("OpenML-CC18")
for task_id in benchmark.tasks:
    try:
        task = openml.tasks.get_task(task_id)
        X, y = task.get_X_and_y()
        path = save_dir / str(task_id)
        with open(path, "wb") as f:
            pickle.dump((X, y), f)
        print(f"Saved task {task_id}")
    except Exception as e:
        print(f"Skipped task {task_id}: {e}")
```

This saves roughly 70 task files totalling a few hundred MB. By default the CLI
looks for data at `~/pcmabinf/OpenML-CC18/`. Override with `--data-dir`.

---

## Reproducing the paper results

The key experiment in the paper evaluates all OPE estimators across the
OpenML-CC18 suite and compares their 95% CI coverage. The settings below match
those used in the reference notebook.

### Step 1 — run the experiments

```bash
make reproduce
```

This runs 64 simulations per task, with 100 batches of 100 observations each,
and skips tasks with more than 100 features (matching the notebook's filter).
Results are saved to `results/all_tasks_df_data`.

The full run covers ~70 tasks and uses all available CPU cores. To limit
parallelism or run a subset:

```bash
make reproduce N_JOBS=4
make reproduce TASK_MAX_FEATURES=50   # fewer, smaller tasks
```

### Step 2 — visualize

```bash
uv run pcmabinf visualize results/all_tasks_df_data
```

### Interpreting the plot

The output is a **4 × 5 grid of scatter plots**:

- **Rows** — four target policies: linear contextual, tree contextual, arm 0
  (always pick arm 0), arm 1 (always pick arm 1).
- **Columns** — five competitor estimators: DM, IPS, DR, ADR, MRDR.
- **Axes** — each point is one OpenML task. The x-axis shows the competitor's
  95% CI coverage; the y-axis shows CADR's coverage for the same task.
- **Dashed lines** — mark the nominal 95% coverage level on both axes.
- **Diagonal line** — the line of equal performance (x = y).
- **Point color**:
  - Black — the two estimators are within each other's standard error (tied).
  - Blue — CADR has higher coverage (CADR better).
  - Red — the competitor has higher coverage (competitor better).

A well-performing estimator should have most points near the dashed lines
(close to 95% coverage). CADR is expected to have more blue points than red
across the grid, particularly for the contextual target policies.

---

## CLI

### Run experiments

```bash
uv run pcmabinf run [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--data-dir PATH` | `~/pcmabinf/OpenML-CC18` | Directory containing task pickle files |
| `--task-id TEXT` | *(all tasks)* | Run a single task by ID |
| `--batch-count INT` | 100 | Number of batches per simulation |
| `--batch-size INT` | 100 | Observations per batch |
| `--simulations INT` | 64 | Independent simulation replicates |
| `--reward-variance FLOAT` | 1.0 | Gaussian noise variance on rewards |
| `--epsilon-multiplier FLOAT` | 0.01 | Scales the ε-greedy exploration rate |
| `--n-jobs INT` | -1 | Parallel workers (-1 = all cores) |
| `--seed INT` | *(random)* | Random seed for reproducibility |
| `--task-max-features INT` | *(no limit)* | Skip tasks with more features than this |
| `--output-dir PATH` | `results/` | Where to write results |

Results are saved as a pickled pandas DataFrame at `{output-dir}/all_tasks_df_data`.

**Example — single task:**

```bash
uv run pcmabinf run --task-id 10093 --simulations 16 --batch-count 20 --batch-size 50
```

**Example — all tasks, notebook settings:**

```bash
uv run pcmabinf run \
  --simulations 64 \
  --batch-count 100 \
  --batch-size 100 \
  --epsilon-multiplier 0.01 \
  --task-max-features 100
```

Or use the Makefile shortcut:

```bash
make reproduce                         # default settings
make reproduce SIMULATIONS=16 N_JOBS=4
```

### Visualize results

```bash
uv run pcmabinf visualize results/all_tasks_df_data
uv run pcmabinf visualize results/all_tasks_df_data --output results/coverage.png
```

---

## Python API

### Quick start

```python
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from pcmabinf import (
    OpenMLCC18World,
    LoggingConfig,
    run_bandit_simulations,
    run_ope_simulations_multipolicy,
    make_target_policy,
    compute_coverage_metrics,
)

# Load a task
world = OpenMLCC18World(
    task_id="10093",
    data_dir=Path.home() / "pcmabinf" / "OpenML-CC18",
    reward_variance=1.0,
)

# Configure the logging policy
# strategy="greedy" uses epsilon-greedy with epsilon_multiplier / (n+1)^(1/3)
config = LoggingConfig(
    batch_count=100,
    batch_size=100,
    strategy="greedy",
    epsilon_multiplier=0.01,
)

# Run independent simulation replicates in parallel
bandit_data_list = run_bandit_simulations(world, config, n_simulations=64, n_jobs=-1)

# Define target policies to evaluate
# train_size matches the amount of data the logging policy collected
train_size = len(bandit_data_list[0].X)
target_policies = {
    "linear": make_target_policy("contextual", world,
                                 outcome_model=LinearRegression(),
                                 train_sample_size=train_size),
    "tree":   make_target_policy("contextual", world,
                                 outcome_model=DecisionTreeRegressor(),
                                 train_sample_size=train_size),
    "arm0":   make_target_policy("non_contextual_constant_0", world),
    "arm1":   make_target_policy("non_contextual_constant_1", world),
}

# Evaluate all policies in one parallel pass
# Returns dict[policy_name, list[OPEResult]]
all_ope = run_ope_simulations_multipolicy(
    bandit_data_list, world, target_policies,
    outcome_model=LinearRegression(), n_jobs=-1,
)

# Compute 95% CI coverage for each estimator
# coverage = fraction of simulation replicates where the CI contains the truth
for policy_name, ope_results in all_ope.items():
    metrics = compute_coverage_metrics(ope_results)
    print(f"\n{policy_name}")
    for estimator in ["dm", "ips", "dr", "cadr", "mrdr"]:
        cov = metrics[f"{estimator}_mean"]
        print(f"  {estimator.upper()}: {cov:.3f}")
```

### What the output means

`compute_coverage_metrics` returns, for each estimator, two scalars:

- `{estimator}_mean` — fraction of simulation replicates where the 95% CI
  contained the true policy value. A value close to 0.95 means the estimator
  is well-calibrated. Values below 0.95 indicate under-coverage (CIs too
  narrow); values above 0.95 indicate over-coverage (CIs too wide).
- `{estimator}_stderr` — standard error of the coverage fraction across
  replicates, giving a sense of Monte Carlo uncertainty.

### Adding a new estimator

1. Add a `new_estimator: tuple[float, float] | None = None` field to
   `OPEResult` in `data.py`.
2. Implement `_new_estimator(self) -> tuple[float, float]` in `OPEEstimator`
   in `estimators.py`, following the pattern of the existing methods. Return
   `self._summarise(phi)` where `phi` is the per-observation influence
   function.
3. Add `"new_estimator"` to `ESTIMATOR_FIELDS` in `metrics.py`.
4. Add the field to `compute_all()` in `OPEEstimator`.

### Adding a new target policy

Implement the `TargetPolicyProtocol` — a single method `pi(X)` that returns
an `(N, K)` matrix of action probabilities for a batch of contexts:

```python
import numpy as np
from numpy.typing import NDArray

class MyPolicy:
    def pi(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        # Return (N, K) action probability matrix.
        # Rows must sum to 1.
        ...
```

Pass an instance directly to `run_ope_simulations_multipolicy` without going
through `make_target_policy`.

### Key classes and functions

| Symbol | Module | Description |
|---|---|---|
| `OpenMLCC18World` | `pcmabinf.world` | Bandit environment backed by an OpenML task |
| `LoggingConfig` | `pcmabinf.logging_policy` | Configuration for the data-collection policy |
| `run_logging_policy` | `pcmabinf.logging_policy` | Collect one `BanditData` run |
| `BanditData` | `pcmabinf.data` | Collected observations (X, A, Y, P, …) |
| `OPEResult` | `pcmabinf.data` | Per-estimator (mean, variance) tuples |
| `OPEEstimator` | `pcmabinf.estimators` | Computes all OPE estimators for one run |
| `run_bandit_simulations` | `pcmabinf.simulate` | Parallel bandit data collection |
| `run_ope_simulations_multipolicy` | `pcmabinf.simulate` | Parallel OPE for multiple target policies |
| `run_ope_simulations` | `pcmabinf.simulate` | Parallel OPE for a single target policy |
| `make_target_policy` | `pcmabinf.policy` | Factory for target policies by name |
| `compute_coverage_metrics` | `pcmabinf.metrics` | 95% CI coverage statistics |

### Target policy names

Pass to `make_target_policy(name, world, ...)`:

| Name | Description |
|---|---|
| `non_contextual_constant_0` | Always select arm 0 |
| `non_contextual_constant_1` | Always select arm 1 (etc.) |
| `non_contextual_uniform_random` | Uniform random over all arms |
| `non_contextual_most_frequent` | Always select the most frequent arm |
| `non_contextual_frequency_proportional` | Select arms proportional to frequency |
| `contextual` | Greedy policy trained on a fresh sample (requires `outcome_model` and `train_sample_size`) |

### OPE estimators

All estimators use 4-fold cross-fitting by default.

| Field | Estimator |
|---|---|
| `truth` | Oracle value (ground truth, variance = 0) |
| `dm` | Direct Method |
| `ips` | Inverse Propensity Score |
| `dr` | Doubly Robust |
| `adr` | Adaptive Doubly Robust |
| `cadr` | Contextual Adaptive DR |
| `mrdr` | Marginalized Robust DR |
| `camrdr` | Contextual Adaptive MRDR |

---

## Development

```bash
make install        # uv sync --group dev
make test           # uv run pytest --tb=short
make clean-cache    # remove __pycache__, .pytest_cache, dist, etc.
make build          # uv build (wheel + sdist)
```

---

## Project layout

```
src/pcmabinf/
├── __init__.py        # public re-exports
├── _utils.py          # score() helper (classifier/regressor unified predict)
├── config.py          # ExperimentConfig dataclass
├── data.py            # BanditData, OPEResult dataclasses
├── world.py           # OpenMLCC18World
├── policy.py          # TargetPolicyProtocol + concrete policy classes
├── logging_policy.py  # run_logging_policy() → BanditData
├── cross_fitting.py   # estimate_outcome_models() → (Q, Q_MRDR, Q_CAMRDR)
├── estimators.py      # OPEEstimator
├── simulate.py        # run_bandit_simulations(), run_ope_simulations_multipolicy()
├── metrics.py         # compute_coverage_metrics()
├── viz.py             # visualize_coverage(), bar_plot()
└── cli.py             # pcmabinf CLI entry point
```
