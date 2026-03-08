from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone

from pcmabinf._utils import score
from pcmabinf.data import BanditData
from pcmabinf.policy import TargetPolicyProtocol


def estimate_outcome_models(
    data: BanditData,
    target_policy: TargetPolicyProtocol,
    outcome_model: BaseEstimator,
    arm_count: int,
    n_folds: int = 4,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Fit outcome models using K-fold cross-fitting.

    Parameters
    ----------
    data:
        Bandit data collected by the logging policy.
    target_policy:
        Policy whose probability matrix drives the MRDR/CAMRDR sample weights.
    outcome_model:
        Sklearn-compatible estimator (cloned per arm per fold).
    arm_count:
        Number of arms K.
    n_folds:
        Number of cross-fitting folds (must be >= 3; default 4).

    Returns
    -------
    Q : (N, K)
        Unweighted outcome model predictions.
    Q_MRDR : (N, K)
        Predictions from model trained with MRDR importance weights
        ``g*(a|x) * (1 - g(a|x)) / g(a|x)**2``.
    Q_CAMRDR : (N, K)
        Predictions from model trained with CAMRDR importance weights
        ``g*(a|x) / g(a|x)``.

    Raises
    ------
    ValueError
        If ``n_folds < 3``.  The fold-exclusion scheme (skips fold k and k+1)
        leaves fold 0 with an empty training set when ``n_folds == 2``.
    """
    if n_folds < 3:
        raise ValueError(
            f"n_folds must be >= 3 (the fold-exclusion scheme produces an empty "
            f"training set for fold 0 when n_folds == 2), got {n_folds}"
        )

    N = len(data.X)
    fold_size = N // n_folds

    Q = np.zeros((N, arm_count), dtype=np.float64)
    Q_MRDR = np.zeros((N, arm_count), dtype=np.float64)
    Q_CAMRDR = np.zeros((N, arm_count), dtype=np.float64)

    for k in range(n_folds):
        # Training folds: all folds except fold k (evaluation) and fold k+1 (hold-out).
        # Out-of-range indices are silently ignored by the set membership test,
        # so the final fold trains on all remaining folds except itself.
        skip = frozenset((k, k + 1))
        train_folds = [f for f in range(n_folds) if f not in skip]
        tr_idx = (
            np.concatenate([np.arange(f * fold_size, (f + 1) * fold_size) for f in train_folds])
            if train_folds
            else np.array([], dtype=np.intp)
        )
        # The last evaluation fold absorbs any remainder rows.
        ev_idx = (
            np.arange(k * fold_size, N)
            if k == n_folds - 1
            else np.arange(k * fold_size, (k + 1) * fold_size)
        )

        A_train = data.A[tr_idx]
        X_train = data.X[tr_idx]
        Y_train = data.Y[tr_idx]
        P_train = data.P[tr_idx]
        X_eval = data.X[ev_idx]

        # Compute target-policy probabilities once for the entire training fold.
        pi_train = target_policy.pi(X_train)  # (n_train, K)

        for a in range(arm_count):
            arm_rows = np.where(A_train == a)[0]
            if len(arm_rows) == 0:
                continue

            X_a = X_train[arm_rows]
            Y_a = Y_train[arm_rows]
            g_a = P_train[arm_rows]
            g_star_a = pi_train[arm_rows, a]

            # Unweighted direct-method model.
            dm_mod = clone(outcome_model)
            dm_mod.fit(X_a, Y_a)
            Q[ev_idx, a] = score(dm_mod, X_eval)

            # When the target policy assigns zero probability to arm *a*, the
            # importance-weighted objectives are degenerate; use unit weights.
            degenerate = np.all(g_star_a == 0)

            # MRDR model: minimises a variance-penalised weighted squared loss.
            sw_mrdr = (
                np.ones_like(Y_a) if degenerate
                else g_star_a * (1.0 - g_a) / (g_a ** 2)
            )
            mrdr_mod = clone(outcome_model)
            mrdr_mod.fit(X_a, Y_a, sample_weight=sw_mrdr)
            Q_MRDR[ev_idx, a] = score(mrdr_mod, X_eval)

            # CAMRDR model: minimises a density-ratio weighted squared loss.
            sw_camrdr = (
                np.ones_like(Y_a) if degenerate
                else g_star_a / g_a
            )
            camrdr_mod = clone(outcome_model)
            camrdr_mod.fit(X_a, Y_a, sample_weight=sw_camrdr)
            Q_CAMRDR[ev_idx, a] = score(camrdr_mod, X_eval)

    return Q, Q_MRDR, Q_CAMRDR
