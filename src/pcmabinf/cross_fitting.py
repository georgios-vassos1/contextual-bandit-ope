from __future__ import annotations

import itertools
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from pcmabinf._utils import predict
from pcmabinf.data import BanditData
from pcmabinf.policy import TargetPolicyProtocol


def cross_fitting(
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
        Policy whose probability matrix is used to compute MRDR/CAMRDR weights.
    outcome_model:
        Sklearn-compatible estimator (cloned per arm per fold).
    arm_count:
        Number of arms K.
    n_folds:
        Number of cross-fitting folds (default 4).

    Returns
    -------
    Q : (N, K)
        Unweighted outcome model predictions.
    Q_MRDR : (N, K)
        Predictions from model weighted by ``g*(a|x)(1-g(a|x)) / g(a|x)^2``.
    Q_CAMRDR : (N, K)
        Predictions from model weighted by ``g*(a|x) / g(a|x)``.
    """
    N = len(data.X)
    fold_size = N // n_folds

    Q = np.zeros((N, arm_count), dtype=np.float64)
    Q_MRDR = np.zeros((N, arm_count), dtype=np.float64)
    Q_CAMRDR = np.zeros((N, arm_count), dtype=np.float64)

    for k in range(n_folds):
        # All folds except k (and k+1 per original notebook — note: original
        # uses "k+2" which skips adjacent fold; we replicate that behaviour).
        valid_folds = [v for v in range(n_folds) if v != k and v != k + 1]
        train_idx = list(
            itertools.chain.from_iterable(
                range(v * fold_size, (v + 1) * fold_size) for v in valid_folds
            )
        )
        # Last fold absorbs remainder
        eval_idx = (
            list(range(k * fold_size, N))
            if k == n_folds - 1
            else list(range(k * fold_size, (k + 1) * fold_size))
        )

        A_train = data.A[train_idx]
        X_train = data.X[train_idx]
        Y_train = data.Y[train_idx]
        P_train = data.P[train_idx]
        X_eval = data.X[eval_idx]

        for a in range(arm_count):
            idx_a = np.where(A_train == a)[0]
            if len(idx_a) == 0:
                continue

            X_a = X_train[idx_a]
            Y_a = Y_train[idx_a]
            g_a = P_train[idx_a]
            g_star_a = target_policy.pi(X_train[idx_a])[:, a]

            # --- Unweighted DM model ---
            m = deepcopy(outcome_model)
            m.fit(X_a, Y_a)
            Q[eval_idx, a] = predict(m, X_eval)

            # --- MRDR model: w = g*(a|x)(1-g(a|x)) / g(a|x)^2 ---
            if np.all(g_star_a == 0):
                w_mrdr = np.ones_like(Y_a)
            else:
                w_mrdr = g_star_a * (1.0 - g_a) / (g_a**2)
            m_mrdr = deepcopy(outcome_model)
            m_mrdr.fit(X_a, Y_a, sample_weight=w_mrdr)
            Q_MRDR[eval_idx, a] = predict(m_mrdr, X_eval)

            # --- CAMRDR model: w = g*(a|x) / g(a|x) ---
            if np.all(g_star_a == 0):
                w_camrdr = np.ones_like(Y_a)
            else:
                w_camrdr = g_star_a / g_a
            m_camrdr = deepcopy(outcome_model)
            m_camrdr.fit(X_a, Y_a, sample_weight=w_camrdr)
            Q_CAMRDR[eval_idx, a] = predict(m_camrdr, X_eval)

    return Q, Q_MRDR, Q_CAMRDR
