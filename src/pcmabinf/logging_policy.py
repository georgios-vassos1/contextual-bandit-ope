from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from pcmabinf._utils import score
from pcmabinf.data import BanditData
from pcmabinf.world import OpenMLCC18World


@dataclass
class LoggingConfig:
    batch_count: int
    batch_size: int
    strategy: Literal["uniform", "greedy"] = "greedy"
    epsilon_multiplier: float = 1.0
    outcome_model: BaseEstimator = field(default_factory=DecisionTreeRegressor)

    def __post_init__(self) -> None:
        if self.batch_count <= 0:
            raise ValueError(f"batch_count must be > 0, got {self.batch_count}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.epsilon_multiplier <= 0.0:
            raise ValueError(
                f"epsilon_multiplier must be > 0, got {self.epsilon_multiplier}"
            )


def run_logging_policy(
    world: OpenMLCC18World,
    config: LoggingConfig,
    seed: int | None = None,
) -> BanditData:
    """Collect bandit data by running the logging policy defined in *config*.

    Parameters
    ----------
    world:
        The bandit environment.
    config:
        Logging policy configuration.
    seed:
        Optional integer seed for NumPy's global random state.  When provided
        the entire run — arm selection, reward sampling, context resampling —
        is fully reproducible.  Each parallel worker should receive a distinct
        seed; see :func:`run_bandit_simulations`.

    Returns
    -------
    BanditData
        All collected observations together with per-batch retroactive
        propensities (``propensity_history``) required by the CADR estimator.

    Notes
    -----
    Exploration rate: ``ε_t = epsilon_multiplier / (n_t + 1)^(1/3)``
    where ``n_t`` is the number of observations collected before batch *t*.
    This decaying schedule achieves sub-linear regret in the tabular case.
    """
    if seed is not None:
        np.random.seed(seed)

    K = world.arm_count
    N_total = config.batch_count * config.batch_size
    bs = config.batch_size

    # Pre-allocate output buffers to avoid repeated array concatenation.
    X_buf = np.empty((N_total, world.feature_count), dtype=np.float64)
    A_buf = np.empty(N_total, dtype=np.intp)
    Y_buf = np.empty(N_total, dtype=np.float64)
    P_buf = np.empty(N_total, dtype=np.float64)
    eps_buf = np.empty(N_total, dtype=np.float64)
    regret_buf = np.empty(N_total, dtype=np.intp)
    propensity_history: list[NDArray[np.float64]] = []

    n = 0  # observations collected so far

    with threadpool_limits(limits=1, user_api="blas"):
        for batch_id in range(config.batch_count):
            X_batch = world.sample_contexts(bs)
            arm_models: list[BaseEstimator] = []

            # ------------------------------------------------------------------
            # Arm selection
            # ------------------------------------------------------------------
            if config.strategy == "uniform":
                epsilon = 1.0
                A_batch = np.random.choice(K, size=bs).astype(np.intp)
                P_batch = np.full(bs, 1.0 / K, dtype=np.float64)

            elif config.strategy == "greedy":
                if n == 0:
                    has_all_arms = False
                else:
                    observed_arms, per_arm_counts = np.unique(A_buf[:n], return_counts=True)
                    has_all_arms = len(observed_arms) == K and int(per_arm_counts.min()) >= 1

                if has_all_arms:
                    epsilon = config.epsilon_multiplier * (n + 1) ** (-1.0 / 3.0)

                    q_hat = np.zeros((bs, K), dtype=np.float64)
                    for a in range(K):
                        arm_rows = np.where(A_buf[:n] == a)[0]
                        mod_a = clone(config.outcome_model)
                        mod_a.fit(X_buf[:n][arm_rows], Y_buf[:n][arm_rows])
                        arm_models.append(mod_a)
                        q_hat[:, a] = score(mod_a, X_batch)

                    explore_action = np.random.choice(K, size=bs).astype(np.intp)
                    greedy_action = np.argmax(q_hat, axis=1).astype(np.intp)
                    do_explore = np.random.random(bs) < epsilon
                    A_batch = np.where(do_explore, explore_action, greedy_action).astype(np.intp)
                    P_batch = np.where(
                        A_batch == greedy_action,
                        1.0 - epsilon + epsilon / K,
                        epsilon / K,
                    ).astype(np.float64)
                else:
                    epsilon = 1.0
                    A_batch = np.random.choice(K, size=bs).astype(np.intp)
                    P_batch = np.full(bs, 1.0 / K, dtype=np.float64)
            else:
                raise ValueError(f"Unknown strategy: {config.strategy!r}")

            # ------------------------------------------------------------------
            # Observe rewards and regrets  (vectorised — no Python loop)
            # ------------------------------------------------------------------
            Y_batch = world.rewards_batch(X_batch, A_batch)
            reg_batch = world.regrets_batch(X_batch, A_batch)

            # Write into pre-allocated buffers.
            X_buf[n : n + bs] = X_batch
            A_buf[n : n + bs] = A_batch
            Y_buf[n : n + bs] = Y_batch
            P_buf[n : n + bs] = P_batch
            eps_buf[n : n + bs] = epsilon
            regret_buf[n : n + bs] = reg_batch
            n += bs

            # ------------------------------------------------------------------
            # Retroactive propensity re-evaluation.
            # propensity_history[b] stores P(A_i | X_i) re-evaluated under the
            # outcome model trained after batch b, for all i in 0..n-1.
            # The CADR estimator uses these to correct for temporal distributional
            # shift in the logging policy.
            # ------------------------------------------------------------------
            if len(arm_models) == 0:
                pi_t = np.full(n, 1.0 / K, dtype=np.float64)
            else:
                q_hat_full = np.column_stack([score(m, X_buf[:n]) for m in arm_models])
                greedy_full = np.argmax(q_hat_full, axis=1).astype(np.intp)
                pi_t = np.where(
                    A_buf[:n] == greedy_full,
                    1.0 - epsilon + epsilon / K,
                    epsilon / K,
                ).astype(np.float64)

            propensity_history.append(pi_t)

    return BanditData(
        X=X_buf,
        A=A_buf,
        Y=Y_buf,
        P=P_buf,
        propensity_history=propensity_history,
        epsilon=eps_buf,
        regret=regret_buf,
        batch_size=bs,
    )
