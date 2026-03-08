"""Tests for OpenMLCC18World."""
from __future__ import annotations

import numpy as np
import pytest

from pcmabinf.world import OpenMLCC18World


def test_world_properties(world: OpenMLCC18World) -> None:
    assert world.arm_count == 3
    assert world.feature_count == 4
    assert world.observation_count == 200


def test_sample_contexts_shape(world: OpenMLCC18World) -> None:
    X = world.sample_contexts(50)
    assert X.shape == (50, world.feature_count)


def test_sample_labeled_shapes(world: OpenMLCC18World) -> None:
    X, y = world.sample_labeled(30)
    assert X.shape == (30, world.feature_count)
    assert y.shape == (30,)


def test_reward_zero_variance(world: OpenMLCC18World) -> None:
    """With reward_variance=0, reward equals 0 or 1 deterministically."""
    x = world.contexts[0]
    optimal = int(world.arms[0])
    assert world.reward(x, optimal) == pytest.approx(1.0)
    # any non-optimal arm (if it exists)
    for a in range(world.arm_count):
        if a != optimal:
            assert world.reward(x, a) == pytest.approx(0.0)
            break


def test_regret(world: OpenMLCC18World) -> None:
    x = world.contexts[0]
    optimal = int(world.arms[0])
    assert world.regret(x, optimal) == 0
    for a in range(world.arm_count):
        if a != optimal:
            assert world.regret(x, a) == 1
            break


def test_nan_imputation(tmp_path: "Path") -> None:
    """World should impute NaN values in the feature matrix without error."""
    import pickle
    from pathlib import Path

    rng = np.random.default_rng(99)
    X = rng.standard_normal((50, 4))
    X[0, 0] = np.nan  # inject a NaN
    X[10, 2] = np.nan
    y = (np.arange(50) % 2).astype(np.intp)
    with open(tmp_path / "nan_task", "wb") as fh:
        pickle.dump((X, y), fh)

    w = OpenMLCC18World(task_id="nan_task", data_dir=tmp_path, reward_variance=0.0)
    assert not np.any(np.isnan(w.contexts))


def test_reward_mean_per_arm(world: OpenMLCC18World) -> None:
    x = world.contexts[0]
    optimal = int(world.arms[0])
    means = world.reward_mean_per_arm(x)
    assert means.shape == (world.arm_count,)
    assert means[optimal] == pytest.approx(1.0)
    assert means.sum() == pytest.approx(1.0)
