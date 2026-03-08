"""Tests for run_logging_policy."""
from __future__ import annotations

import numpy as np
import pytest

from pcmabinf.logging_policy import LoggingConfig, run_logging_policy
from pcmabinf.world import OpenMLCC18World


@pytest.mark.parametrize("strategy", ["uniform", "greedy"])
def test_shapes(world: OpenMLCC18World, strategy: str) -> None:
    config = LoggingConfig(
        batch_count=5,
        batch_size=20,
        strategy=strategy,  # type: ignore[arg-type]
        epsilon_multiplier=1.0,
    )
    data = run_logging_policy(world, config)
    N = config.batch_count * config.batch_size
    assert data.X.shape == (N, world.feature_count)
    assert data.A.shape == (N,)
    assert data.Y.shape == (N,)
    assert data.P.shape == (N,)
    assert data.epsilon.shape == (N,)
    assert data.regret.shape == (N,)


def test_propensity_history_length(world: OpenMLCC18World) -> None:
    config = LoggingConfig(batch_count=6, batch_size=10, strategy="uniform")
    data = run_logging_policy(world, config)
    assert len(data.propensity_history) == config.batch_count


def test_propensities_valid(world: OpenMLCC18World) -> None:
    config = LoggingConfig(batch_count=5, batch_size=20, strategy="uniform")
    data = run_logging_policy(world, config)
    assert np.all(data.P > 0)
    assert np.all(data.P <= 1)


def test_propensity_history_sizes(world: OpenMLCC18World) -> None:
    """Each entry in propensity_history covers all observations up to that batch."""
    config = LoggingConfig(batch_count=4, batch_size=10, strategy="uniform")
    data = run_logging_policy(world, config)
    for b, ph in enumerate(data.propensity_history):
        expected_n = (b + 1) * config.batch_size
        assert len(ph) == expected_n, f"batch {b}: expected {expected_n}, got {len(ph)}"


def test_arms_in_valid_range(world: OpenMLCC18World) -> None:
    config = LoggingConfig(batch_count=5, batch_size=20, strategy="uniform")
    data = run_logging_policy(world, config)
    assert np.all(data.A >= 0)
    assert np.all(data.A < world.arm_count)


def test_unknown_strategy_raises(world: OpenMLCC18World) -> None:
    import pytest
    config = LoggingConfig(batch_count=1, batch_size=10, strategy="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_logging_policy(world, config)


def test_logging_config_invalid_batch_count() -> None:
    with pytest.raises(ValueError, match="batch_count"):
        LoggingConfig(batch_count=0, batch_size=10)


def test_logging_config_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        LoggingConfig(batch_count=5, batch_size=0)


def test_logging_config_invalid_epsilon_multiplier() -> None:
    with pytest.raises(ValueError, match="epsilon_multiplier"):
        LoggingConfig(batch_count=5, batch_size=10, epsilon_multiplier=-1.0)
