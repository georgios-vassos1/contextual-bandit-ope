from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
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
    Epsilon schedule: ``ε_t = epsilon_multiplier * (t + 1)^(-1/3)``
    where *t* is the batch index (0-based).
    """
    K = world.arm_count

    X_list: list[NDArray[np.float64]] = []
    A_list: list[NDArray[np.intp]] = []
    Y_list: list[NDArray[np.float64]] = []
    P_list: list[NDArray[np.float64]] = []
    eps_list: list[NDArray[np.float64]] = []
    regret_list: list[NDArray[np.intp]] = []
    propensity_history: list[NDArray[np.float64]] = []

    with threadpool_limits(limits=1, user_api="blas"):
        for batch_id in range(config.batch_count):
            X_batch = world.sample_contexts(config.batch_size)
            outcome_models: list[BaseEstimator] = []

            # ----------------------------------------------------------------
            # Arm-selection logic
            # ----------------------------------------------------------------
            if config.strategy == "uniform_random":
                epsilon = 1.0
                A_batch = np.random.choice(K, size=len(X_batch)).astype(np.intp)
                P_batch = np.full(len(X_batch), 1.0 / K, dtype=np.float64)

            elif config.strategy == "contextual_epsilon_greedy":
                A_all = np.concatenate(A_list) if A_list else np.array([], dtype=np.intp)
                X_all = np.vstack(X_list) if X_list else np.empty((0, world.feature_count))
                Y_all = np.concatenate(Y_list) if Y_list else np.array([], dtype=np.float64)

                unique_arms, counts = np.unique(A_all, return_counts=True)
                has_all_arms = len(unique_arms) == K and counts.min() >= 1

                if has_all_arms:
                    epsilon = config.epsilon_multiplier * (batch_id * config.batch_size + 1) ** (-1.0 / 3.0)

                    Y_hat = np.zeros((len(X_batch), K), dtype=np.float64)
                    for a in range(K):
                        idx = np.where(A_all == a)[0]
                        model_a = deepcopy(config.outcome_model)
                        model_a.fit(X_all[idx], Y_all[idx])
                        outcome_models.append(model_a)
                        Y_hat[:, a] = predict(model_a, X_batch)

                    A_random = np.random.choice(K, size=len(X_batch)).astype(np.intp)
                    A_best = np.argmax(Y_hat, axis=1).astype(np.intp)
                    explore = np.random.random(len(X_batch)) < epsilon
                    A_batch = np.where(explore, A_random, A_best).astype(np.intp)
                    P_batch = np.where(
                        A_batch == A_best,
                        1.0 - epsilon + epsilon / K,
                        epsilon / K,
                    ).astype(np.float64)
                else:
                    epsilon = 1.0
                    A_batch = np.random.choice(K, size=len(X_batch)).astype(np.intp)
                    P_batch = np.full(len(X_batch), 1.0 / K, dtype=np.float64)
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

            X_list.append(X_batch)
            A_list.append(A_batch)
            Y_list.append(Y_batch)
            P_list.append(P_batch)
            eps_list.append(np.full(len(X_batch), epsilon, dtype=np.float64))
            regret_list.append(reg_batch)

            # ----------------------------------------------------------------
            # Re-evaluate propensities for ALL collected data under current model
            # ----------------------------------------------------------------
            A_all_new = np.concatenate(A_list).astype(np.intp)
            X_all_new = np.vstack(X_list)

            if len(outcome_models) == 0:
                prev_P = np.full(len(A_all_new), 1.0 / K, dtype=np.float64)
            else:
                Y_hat_all = np.column_stack([predict(m, X_all_new) for m in outcome_models])
                A_best_all = np.argmax(Y_hat_all, axis=1).astype(np.intp)
                prev_P = np.where(
                    A_all_new == A_best_all,
                    1.0 - epsilon + epsilon / K,
                    epsilon / K,
                ).astype(np.float64)

            propensity_history.append(prev_P)

    return BanditData(
        X=np.vstack(X_list),
        A=np.concatenate(A_list).astype(np.intp),
        Y=np.concatenate(Y_list),
        P=np.concatenate(P_list),
        propensity_history=propensity_history,
        epsilon=np.concatenate(eps_list),
        regret=np.concatenate(regret_list).astype(np.intp),
        batch_size=config.batch_size,
    )
