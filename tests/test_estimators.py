"""Tests for OPEEstimator."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from pcmabinf.estimators import OPEEstimator
from pcmabinf.policy import UniformPolicy
from pcmabinf.world import OpenMLCC18World


def test_compute_all_returns_ope_result(world: OpenMLCC18World, simple_bandit_data) -> None:
    policy = UniformPolicy(arm_count=world.arm_count)
    # Adapt simple_bandit_data to match world dimensions
    from pcmabinf.data import BanditData
    import numpy as np

    N, K = len(simple_bandit_data.X), world.arm_count
    rng = np.random.default_rng(1)
    data = BanditData(
        X=world.sample_contexts(N),
        A=rng.integers(0, K, size=N).astype(np.intp),
        Y=rng.standard_normal(N),
        P=np.full(N, 1.0 / K),
        propensity_history=[np.full(min((b + 1) * 10, N), 1.0 / K) for b in range(N // 10)],
        epsilon=np.full(N, 1.0),
        regret=rng.integers(0, 2, size=N).astype(np.intp),
        batch_size=10,
    )

    result = OPEEstimator(data, world, policy, LinearRegression(), n_folds=4).compute_all()
    assert result.truth is not None
    assert result.dm is not None
    assert result.ips is not None
    assert result.dr is not None
    assert result.adr is not None
    assert result.cadr is not None
    assert result.mrdr is not None
    assert result.camrdr is not None


def test_truth_variance_is_zero(world: OpenMLCC18World) -> None:
    """Truth always has variance=0 by construction."""
    from pcmabinf.data import BanditData

    N, K = 80, world.arm_count
    rng = np.random.default_rng(2)
    data = BanditData(
        X=world.sample_contexts(N),
        A=rng.integers(0, K, size=N).astype(np.intp),
        Y=rng.standard_normal(N),
        P=np.full(N, 1.0 / K),
        propensity_history=[np.full(min((b + 1) * 10, N), 1.0 / K) for b in range(N // 10)],
        epsilon=np.full(N, 1.0),
        regret=rng.integers(0, 2, size=N).astype(np.intp),
        batch_size=10,
    )
    policy = UniformPolicy(arm_count=K)
    result = OPEEstimator(data, world, policy, LinearRegression()).compute_all()
    assert result.truth is not None
    assert result.truth[1] == pytest.approx(0.0)


def test_mean_and_variance_uniform() -> None:
    """Unweighted mean/variance should match numpy."""
    D = np.array([1.0, 2.0, 3.0, 4.0])
    mean, var = OPEEstimator._mean_and_variance(D)
    assert mean == pytest.approx(np.mean(D))
    # variance = mean of (D - mean)^2 / N  (formula uses 1/N^2 * sum(D-mean)^2)
    expected_var = np.mean((D - np.mean(D)) ** 2) / len(D)
    assert var == pytest.approx(expected_var)
