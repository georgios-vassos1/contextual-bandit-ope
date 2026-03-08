"""Tests for cross_fitting."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from pcmabinf.cross_fitting import cross_fitting
from pcmabinf.data import BanditData
from pcmabinf.policy import UniformPolicy


def test_output_shapes(simple_bandit_data: BanditData) -> None:
    N = len(simple_bandit_data.X)
    K = 3
    policy = UniformPolicy(arm_count=K)
    Q, Q_MRDR, Q_CAMRDR = cross_fitting(
        simple_bandit_data, policy, LinearRegression(), arm_count=K, n_folds=4
    )
    assert Q.shape == (N, K)
    assert Q_MRDR.shape == (N, K)
    assert Q_CAMRDR.shape == (N, K)


def test_cross_fitting_no_nan(simple_bandit_data: BanditData) -> None:
    K = 3
    policy = UniformPolicy(arm_count=K)
    Q, Q_MRDR, Q_CAMRDR = cross_fitting(
        simple_bandit_data, policy, LinearRegression(), arm_count=K, n_folds=4
    )
    assert not np.any(np.isnan(Q))
    assert not np.any(np.isnan(Q_MRDR))
    assert not np.any(np.isnan(Q_CAMRDR))


def test_cross_fitting_deterministic_reward(simple_bandit_data: BanditData) -> None:
    """With perfect labels (Y=1 for arm 0, Y=0 otherwise), DM Q should be ~1 for arm 0."""
    data = simple_bandit_data
    K = 3
    # Override rewards: arm 0 always gives 1, others 0
    Y = (data.A == 0).astype(np.float64)
    from dataclasses import replace

    data2 = BanditData(
        X=data.X,
        A=data.A,
        Y=Y,
        P=data.P,
        propensity_history=data.propensity_history,
        epsilon=data.epsilon,
        regret=data.regret,
        batch_size=data.batch_size,
    )
    policy = UniformPolicy(arm_count=K)
    Q, _, _ = cross_fitting(data2, policy, LinearRegression(), arm_count=K, n_folds=4)
    # Q[:, 0] should be higher than Q[:, 1] on average
    assert np.mean(Q[:, 0]) > np.mean(Q[:, 1])
