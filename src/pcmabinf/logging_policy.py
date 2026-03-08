from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from pcmabinf._utils import predict
from pcmabinf.data import BanditData
from pcmabinf.world import OpenMLCC18World


@dataclass
class LoggingConfig:
    batch_count: int
    batch_size: int
    strategy: Literal["uniform_random", "contextual_epsilon_greedy"] = "contextual_epsilon_greedy"
    epsilon_multiplier: float = 1.0
    outcome_model: BaseEstimator = field(default_factory=DecisionTreeRegressor)


def run_logging_policy(world: OpenMLCC18World, config: LoggingConfig) -> BanditData:
    """Collect bandit data by running the logging policy defined in *config*.

    Returns
    -------
    BanditData
        All collected observations together with per-batch re-evaluated
        propensities (``propensity_history``) needed by the CADR estimator.

    Notes
    -----
    Epsilon schedule: ``ε = epsilon_multiplier * (n + 1)^(-1/3)``
    where *n* is the total number of observations collected before the current
    batch (i.e. ``batch_id * batch_size``).
    """
    K = world.arm_count
    N_total = config.batch_count * config.batch_size
    bs = config.batch_size

    # Pre-allocate output buffers — avoids O(B²) repeated concatenation.
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
            outcome_models: list[BaseEstimator] = []

            # ----------------------------------------------------------------
            # Arm-selection logic
            # ----------------------------------------------------------------
            if config.strategy == "uniform_random":
                epsilon = 1.0
                A_batch = np.random.choice(K, size=bs).astype(np.intp)
                P_batch = np.full(bs, 1.0 / K, dtype=np.float64)

            elif config.strategy == "contextual_epsilon_greedy":
                if n == 0:
                    has_all_arms = False
                else:
                    unique_arms, counts = np.unique(A_buf[:n], return_counts=True)
                    has_all_arms = len(unique_arms) == K and int(counts.min()) >= 1

                if has_all_arms:
                    epsilon = config.epsilon_multiplier * (n + 1) ** (-1.0 / 3.0)

                    Y_hat = np.zeros((bs, K), dtype=np.float64)
                    for a in range(K):
                        idx = np.where(A_buf[:n] == a)[0]
                        model_a = clone(config.outcome_model)
                        model_a.fit(X_buf[:n][idx], Y_buf[:n][idx])
                        outcome_models.append(model_a)
                        Y_hat[:, a] = predict(model_a, X_batch)

                    A_random = np.random.choice(K, size=bs).astype(np.intp)
                    A_best = np.argmax(Y_hat, axis=1).astype(np.intp)
                    explore = np.random.random(bs) < epsilon
                    A_batch = np.where(explore, A_random, A_best).astype(np.intp)
                    P_batch = np.where(
                        A_batch == A_best,
                        1.0 - epsilon + epsilon / K,
                        epsilon / K,
                    ).astype(np.float64)
                else:
                    epsilon = 1.0
                    A_batch = np.random.choice(K, size=bs).astype(np.intp)
                    P_batch = np.full(bs, 1.0 / K, dtype=np.float64)
            else:
                raise ValueError(f"Unknown strategy: {config.strategy!r}")

            # ----------------------------------------------------------------
            # Observe rewards and regrets
            # ----------------------------------------------------------------
            Y_batch = np.array(
                [world.reward(x, int(a)) for x, a in zip(X_batch, A_batch)],
                dtype=np.float64,
            )
            reg_batch = np.array(
                [world.regret(x, int(a)) for x, a in zip(X_batch, A_batch)],
                dtype=np.intp,
            )

            # Write into pre-allocated buffers.
            X_buf[n : n + bs] = X_batch
            A_buf[n : n + bs] = A_batch
            Y_buf[n : n + bs] = Y_batch
            P_buf[n : n + bs] = P_batch
            eps_buf[n : n + bs] = epsilon
            regret_buf[n : n + bs] = reg_batch
            n += bs

            # ----------------------------------------------------------------
            # Re-evaluate propensities for ALL collected data under current model.
            # `propensity_history[b]` stores P(A_i | X_i) re-evaluated under
            # the model trained after batch b, for all i = 0..n-1.
            # CADR uses this to correct for distributional shift over time.
            # ----------------------------------------------------------------
            if len(outcome_models) == 0:
                prev_P = np.full(n, 1.0 / K, dtype=np.float64)
            else:
                Y_hat_all = np.column_stack([predict(m, X_buf[:n]) for m in outcome_models])
                A_best_all = np.argmax(Y_hat_all, axis=1).astype(np.intp)
                prev_P = np.where(
                    A_buf[:n] == A_best_all,
                    1.0 - epsilon + epsilon / K,
                    epsilon / K,
                ).astype(np.float64)

            propensity_history.append(prev_P)

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
