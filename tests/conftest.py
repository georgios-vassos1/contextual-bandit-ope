"""Shared pytest fixtures."""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pcmabinf.data import BanditData, OPEResult
from pcmabinf.world import OpenMLCC18World


# ---------------------------------------------------------------------------
# Synthetic world fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a minimal synthetic OpenML-CC18 pickle for task '999'."""
    rng = np.random.default_rng(42)
    n, d, k = 200, 4, 3
    X = rng.standard_normal((n, d)).astype(np.float64)
    # Labels cycle 0, 1, 2, 0, 1, 2, ...
    y = (np.arange(n) % k).astype(np.intp)

    with open(tmp_path / "999", "wb") as fh:
        pickle.dump((X, y), fh)
    return tmp_path


@pytest.fixture()
def world(tmp_data_dir: Path) -> OpenMLCC18World:
    return OpenMLCC18World(task_id="999", data_dir=tmp_data_dir, reward_variance=0.0)


# ---------------------------------------------------------------------------
# Minimal BanditData fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_bandit_data() -> BanditData:
    """100 observations, 3 arms, batch_size=10."""
    rng = np.random.default_rng(0)
    N, d, K, bs = 100, 4, 3, 10
    X = rng.standard_normal((N, d)).astype(np.float64)
    A = rng.integers(0, K, size=N).astype(np.intp)
    Y = rng.standard_normal(N).astype(np.float64)
    P = np.full(N, 1.0 / K)

    n_batches = N // bs
    propensity_history = [np.full(min((b + 1) * bs, N), 1.0 / K) for b in range(n_batches)]

    return BanditData(
        X=X,
        A=A,
        Y=Y,
        P=P,
        propensity_history=propensity_history,
        epsilon=np.full(N, 1.0),
        regret=rng.integers(0, 2, size=N).astype(np.intp),
        batch_size=bs,
    )
