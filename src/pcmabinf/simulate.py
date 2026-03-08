from __future__ import annotations

import warnings

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
    state sharing — each worker receives the arguments by value.  Failed
    workers emit a :class:`UserWarning` and are excluded from the returned
    list rather than crashing the entire batch.

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

    def _single_sim(s: int) -> BanditData | None:
        try:
            return run_logging_policy(world, config, s)
        except Exception:  # noqa: BLE001
            return None

    raw: list[BanditData | None] = Parallel(n_jobs=n_jobs)(  # type: ignore[assignment]
        delayed(_single_sim)(s) for s in seeds
    )
    n_failed = sum(r is None for r in raw)
    if n_failed:
        warnings.warn(
            f"{n_failed}/{n_simulations} bandit simulations failed and were excluded.",
            stacklevel=2,
        )
    return [r for r in raw if r is not None]


def run_ope_simulations(
    bandit_data_list: list[BanditData],
    world: OpenMLCC18World,
    target_policy: TargetPolicyProtocol,
    outcome_model: BaseEstimator,
    n_folds: int = 4,
    n_jobs: int = -1,
) -> list[OPEResult]:
    """Run :class:`OPEEstimator` on each :class:`BanditData` in parallel.

    Failed workers emit a :class:`UserWarning` and are excluded from the
    returned list rather than crashing the entire batch.

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
    n_total = len(bandit_data_list)

    def _single(data: BanditData) -> OPEResult | None:
        try:
            return OPEEstimator(data, world, target_policy, outcome_model, n_folds).compute_all()
        except Exception:  # noqa: BLE001
            return None

    raw: list[OPEResult | None] = Parallel(n_jobs=n_jobs)(  # type: ignore[assignment]
        delayed(_single)(d) for d in bandit_data_list
    )
    n_failed = sum(r is None for r in raw)
    if n_failed:
        warnings.warn(
            f"{n_failed}/{n_total} OPE estimations failed and were excluded.",
            stacklevel=2,
        )
    return [r for r in raw if r is not None]
