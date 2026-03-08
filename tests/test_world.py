"""Tests for OpenMLCC18World."""
from __future__ import annotations

import numpy as np
import pytest

from pcmabinf.world import OpenMLCC18World


def test_negative_reward_variance_raises(tmp_data_dir: "Path") -> None:
    from pathlib import Path
    with pytest.raises(ValueError, match="reward_variance"):
        OpenMLCC18World(task_id="999", data_dir=tmp_data_dir, reward_variance=-0.1)


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


def test_optimal_arms_batch(world: OpenMLCC18World) -> None:
    """optimal_arms_batch should match scalar _optimal_arm for every context."""
    n = 20
    X = world.sample_contexts(n)
    batch = world.optimal_arms_batch(X)
    assert batch.shape == (n,)
    for i, x in enumerate(X):
        assert batch[i] == world._optimal_arm(x)


def test_rewards_batch_shape_and_range(world: OpenMLCC18World) -> None:
    n = 30
    X = world.sample_contexts(n)
    A = np.random.choice(world.arm_count, size=n).astype(np.intp)
    R = world.rewards_batch(X, A)
    assert R.shape == (n,)
    # With reward_variance=0 (fixture default) rewards are 0 or 1 exactly.
    assert np.all((R == 0.0) | (R == 1.0))


def test_regrets_batch_shape_and_range(world: OpenMLCC18World) -> None:
    n = 30
    X = world.sample_contexts(n)
    A = np.random.choice(world.arm_count, size=n).astype(np.intp)
    reg = world.regrets_batch(X, A)
    assert reg.shape == (n,)
    assert set(reg.tolist()).issubset({0, 1})


def test_rewards_batch_optimal_arm_gives_one(world: OpenMLCC18World) -> None:
    """Choosing the optimal arm should always give reward=1 with variance=0."""
    X = world.sample_contexts(20)
    A = world.optimal_arms_batch(X)
    R = world.rewards_batch(X, A)
    assert np.all(R == pytest.approx(1.0))
