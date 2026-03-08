"""Tests for BanditData and OPEResult dataclasses."""
from __future__ import annotations

import numpy as np
import pytest

from pcmabinf.data import BanditData


def _make_data(N: int = 20, d: int = 3, K: int = 3, bs: int = 10) -> BanditData:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, d)).astype(np.float64)
    A = rng.integers(0, K, N).astype(np.intp)
    Y = rng.standard_normal(N).astype(np.float64)
    P = np.full(N, 1.0 / K)
    return BanditData(
        X=X, A=A, Y=Y, P=P,
        propensity_history=[P.copy()],
        epsilon=np.ones(N),
        regret=np.zeros(N, dtype=np.intp),
        batch_size=bs,
    )


def test_valid_bandit_data_does_not_raise() -> None:
    _make_data()  # should not raise


def test_bandit_data_bad_A_shape() -> None:
    N, d, K = 20, 3, 3
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, d)).astype(np.float64)
    with pytest.raises(ValueError, match="A"):
        BanditData(
            X=X,
            A=np.zeros(5, dtype=np.intp),  # wrong shape
            Y=np.zeros(N),
            P=np.full(N, 1.0 / K),
            propensity_history=[np.full(N, 1.0 / K)],
            epsilon=np.ones(N),
            regret=np.zeros(N, dtype=np.intp),
            batch_size=10,
        )


def test_bandit_data_bad_Y_shape() -> None:
    N, d, K = 20, 3, 3
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, d)).astype(np.float64)
    with pytest.raises(ValueError, match="Y"):
        BanditData(
            X=X,
            A=np.zeros(N, dtype=np.intp),
            Y=np.zeros(5),  # wrong shape
            P=np.full(N, 1.0 / K),
            propensity_history=[np.full(N, 1.0 / K)],
            epsilon=np.ones(N),
            regret=np.zeros(N, dtype=np.intp),
            batch_size=10,
        )


def test_bandit_data_bad_batch_size() -> None:
    N, d, K = 20, 3, 3
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, d)).astype(np.float64)
    with pytest.raises(ValueError, match="batch_size"):
        BanditData(
            X=X,
            A=np.zeros(N, dtype=np.intp),
            Y=np.zeros(N),
            P=np.full(N, 1.0 / K),
            propensity_history=[np.full(N, 1.0 / K)],
            epsilon=np.ones(N),
            regret=np.zeros(N, dtype=np.intp),
            batch_size=0,  # invalid
        )
