from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator

from pcmabinf.data import BanditData, OPEResult
from pcmabinf.estimators import OPEEstimator
from pcmabinf.logging_policy import LoggingConfig, run_logging_policy
from pcmabinf.policy import TargetPolicyProtocol
from pcmabinf.world import OpenMLCC18World


def run_bandit_simulations(
    world: OpenMLCC18World,
    config: LoggingConfig,
    n_simulations: int,
    n_jobs: int = -1,
    seed: int | None = None,
) -> list[BanditData]:
    """Run *n_simulations* independent bandit data-collection runs in parallel.

    Uses ``joblib.Parallel`` (loky backend by default) so there is no global
    state sharing — each worker receives the arguments by value.

    Parameters
    ----------
    seed:
        Optional integer seed for reproducibility.  A separate seed is derived
        for each simulation using ``numpy.random.default_rng(seed)``, so every
        run is independent while the full set of results is reproducible.
        When ``None`` (default) seeds are drawn from OS entropy.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_simulations).tolist()
    results: list[BanditData] = Parallel(n_jobs=n_jobs)(  # type: ignore[assignment]
        delayed(run_logging_policy)(world, config, s) for s in seeds
    )
    return results


def run_ope_simulations(
    bandit_data_list: list[BanditData],
    world: OpenMLCC18World,
    target_policy: TargetPolicyProtocol,
    outcome_model: BaseEstimator,
    n_folds: int = 4,
    n_jobs: int = -1,
) -> list[OPEResult]:
    """Run :class:`OPEEstimator` on each :class:`BanditData` in parallel.

    Parameters
    ----------
    bandit_data_list:
        One :class:`BanditData` per simulation replicate.
    world:
        The bandit environment (used for ground-truth computation).
    target_policy:
        The policy being evaluated.
    outcome_model:
        Sklearn-compatible estimator (cloned inside each worker).
    n_folds:
        Cross-fitting folds.
    n_jobs:
        Joblib parallelism (-1 = all cores).
    """

    def _single(data: BanditData) -> OPEResult:
        return OPEEstimator(data, world, target_policy, outcome_model, n_folds).compute_all()

    results: list[OPEResult] = Parallel(n_jobs=n_jobs)(  # type: ignore[assignment]
        delayed(_single)(d) for d in bandit_data_list
    )
    return results
