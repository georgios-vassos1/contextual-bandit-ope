"""Tests for run_bandit_simulations and run_ope_simulations."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from pcmabinf.data import BanditData, OPEResult
from pcmabinf.estimators import OPEEstimator
from pcmabinf.logging_policy import LoggingConfig, run_logging_policy
from pcmabinf.policy import UniformPolicy
from pcmabinf.simulate import run_bandit_simulations, run_ope_simulations
from pcmabinf.world import OpenMLCC18World


@pytest.fixture()
def small_config() -> LoggingConfig:
    return LoggingConfig(batch_count=3, batch_size=20, strategy="uniform")


def test_run_bandit_simulations_count(world: OpenMLCC18World, small_config: LoggingConfig) -> None:
    results = run_bandit_simulations(world, small_config, n_simulations=4, n_jobs=1)
    assert len(results) == 4
    assert all(isinstance(r, BanditData) for r in results)


def test_run_bandit_simulations_shapes(world: OpenMLCC18World, small_config: LoggingConfig) -> None:
    results = run_bandit_simulations(world, small_config, n_simulations=2, n_jobs=1)
    N = small_config.batch_count * small_config.batch_size
    for bd in results:
        assert bd.X.shape == (N, world.feature_count)
        assert bd.A.shape == (N,)


def test_run_ope_simulations_count(world: OpenMLCC18World, small_config: LoggingConfig) -> None:
    bandit_data_list = run_bandit_simulations(world, small_config, n_simulations=3, n_jobs=1)
    policy = UniformPolicy(arm_count=world.arm_count)
    results = run_ope_simulations(
        bandit_data_list, world, policy, LinearRegression(), n_folds=3, n_jobs=1
    )
    assert len(results) == 3
    assert all(isinstance(r, OPEResult) for r in results)


def test_run_ope_simulations_all_fields_set(
    world: OpenMLCC18World, small_config: LoggingConfig
) -> None:
    bandit_data_list = run_bandit_simulations(world, small_config, n_simulations=2, n_jobs=1)
    policy = UniformPolicy(arm_count=world.arm_count)
    results = run_ope_simulations(
        bandit_data_list, world, policy, LinearRegression(), n_folds=3, n_jobs=1
    )
    for r in results:
        assert r.truth is not None
        assert r.dm is not None
        assert r.ips is not None
        assert r.dr is not None


def test_seed_reproducibility(world: OpenMLCC18World, small_config: LoggingConfig) -> None:
    """Same seed must produce identical BanditData across two calls."""
    r1 = run_bandit_simulations(world, small_config, n_simulations=3, n_jobs=1, seed=42)
    r2 = run_bandit_simulations(world, small_config, n_simulations=3, n_jobs=1, seed=42)
    for bd1, bd2 in zip(r1, r2):
        np.testing.assert_array_equal(bd1.A, bd2.A)
        np.testing.assert_array_equal(bd1.Y, bd2.Y)


def test_different_seeds_differ(world: OpenMLCC18World, small_config: LoggingConfig) -> None:
    """Different seeds should (with overwhelming probability) produce different data."""
    r1 = run_bandit_simulations(world, small_config, n_simulations=2, n_jobs=1, seed=1)
    r2 = run_bandit_simulations(world, small_config, n_simulations=2, n_jobs=1, seed=2)
    assert not np.array_equal(r1[0].A, r2[0].A) or not np.array_equal(r1[0].Y, r2[0].Y)


def test_simulations_are_independent(world: OpenMLCC18World, small_config: LoggingConfig) -> None:
    """Each simulation within a seeded run should receive a distinct seed."""
    results = run_bandit_simulations(world, small_config, n_simulations=3, n_jobs=1, seed=99)
    # All three arm sequences must not all be identical.
    assert not (
        np.array_equal(results[0].A, results[1].A)
        and np.array_equal(results[1].A, results[2].A)
    )


# ---------------------------------------------------------------------------
# Error-handling tests
# ---------------------------------------------------------------------------


def test_bandit_simulation_failure_is_skipped(
    world: OpenMLCC18World, small_config: LoggingConfig
) -> None:
    """A failing worker must be excluded with a warning, not crash the batch."""
    from unittest.mock import patch

    original = run_logging_policy
    call_count = 0

    def sometimes_fail(w, c, s=None):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated failure")
        return original(w, c, s)

    with patch("pcmabinf.simulate.run_logging_policy", sometimes_fail):
        with pytest.warns(UserWarning, match="1/3"):
            results = run_bandit_simulations(world, small_config, n_simulations=3, n_jobs=1)

    assert len(results) == 2
    assert all(isinstance(r, BanditData) for r in results)


def test_ope_simulation_failure_is_skipped(
    world: OpenMLCC18World, small_config: LoggingConfig
) -> None:
    """A failing OPE worker must be excluded with a warning, not crash the batch."""
    from unittest.mock import patch

    bandit_data_list = run_bandit_simulations(world, small_config, n_simulations=3, n_jobs=1)
    policy = UniformPolicy(arm_count=world.arm_count)

    call_count = 0
    original_init = OPEEstimator.__init__

    def sometimes_fail(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated OPE failure")
        return original_init(self, *args, **kwargs)

    with patch.object(OPEEstimator, "__init__", sometimes_fail):
        with pytest.warns(UserWarning, match="1/3"):
            results = run_ope_simulations(
                bandit_data_list, world, policy, LinearRegression(), n_folds=3, n_jobs=1
            )

    assert len(results) == 2
    assert all(isinstance(r, OPEResult) for r in results)
