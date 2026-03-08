"""Tests for run_bandit_simulations and run_ope_simulations."""
from __future__ import annotations

import pytest
from sklearn.linear_model import LinearRegression

from pcmabinf.data import BanditData, OPEResult
from pcmabinf.logging_policy import LoggingConfig
from pcmabinf.policy import UniformPolicy
from pcmabinf.simulate import run_bandit_simulations, run_ope_simulations
from pcmabinf.world import OpenMLCC18World


@pytest.fixture()
def small_config() -> LoggingConfig:
    return LoggingConfig(batch_count=3, batch_size=20, strategy="uniform_random")


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
        bandit_data_list, world, policy, LinearRegression(), n_folds=2, n_jobs=1
    )
    assert len(results) == 3
    assert all(isinstance(r, OPEResult) for r in results)


def test_run_ope_simulations_all_fields_set(
    world: OpenMLCC18World, small_config: LoggingConfig
) -> None:
    bandit_data_list = run_bandit_simulations(world, small_config, n_simulations=2, n_jobs=1)
    policy = UniformPolicy(arm_count=world.arm_count)
    results = run_ope_simulations(
        bandit_data_list, world, policy, LinearRegression(), n_folds=2, n_jobs=1
    )
    for r in results:
        assert r.truth is not None
        assert r.dm is not None
        assert r.ips is not None
        assert r.dr is not None
