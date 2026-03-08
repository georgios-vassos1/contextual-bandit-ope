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

The package expects OpenML-CC18 task files in a local directory. Each file is a pickle containing `(features, targets)` arrays and is named by its task ID (e.g. `10093`).

By default the CLI looks for data at `~/pcmabinf/OpenML-CC18/`. Override with `--data-dir`.

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

Produces a scatter-plot grid (4 target policies × 5 competitors) showing 95% CI coverage for CADR vs each competitor estimator. Points are colored by which estimator has better coverage (black = tied, red = competitor better, blue = CADR better).

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
config = LoggingConfig(
    batch_count=100,
    batch_size=100,
    strategy="greedy",
    epsilon_multiplier=0.01,
)

# Run bandit simulations
bandit_data_list = run_bandit_simulations(world, config, n_simulations=64, n_jobs=-1)

# Define target policies to evaluate
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
all_ope = run_ope_simulations_multipolicy(
    bandit_data_list, world, target_policies,
    outcome_model=LinearRegression(), n_jobs=-1,
)

# Compute 95% CI coverage for each estimator
for policy_name, ope_results in all_ope.items():
    metrics = compute_coverage_metrics(ope_results)
    cadr_cov = metrics["cadr_mean"]
    print(f"{policy_name}: CADR coverage = {cadr_cov:.3f}")
```

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
