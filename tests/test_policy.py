"""Tests for all target policies and make_target_policy factory."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from pcmabinf.policy import (
    ConstantPolicy,
    ContextualPolicy,
    FrequencyPolicy,
    MostFrequentPolicy,
    UniformPolicy,
    make_target_policy,
)
from pcmabinf.world import OpenMLCC18World


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _X(n: int = 10, d: int = 4) -> np.ndarray:
    return np.random.default_rng(0).standard_normal((n, d))


# ---------------------------------------------------------------------------
# ConstantPolicy
# ---------------------------------------------------------------------------


def test_constant_policy_shape(world: OpenMLCC18World) -> None:
    policy = ConstantPolicy(arm=0, arm_count=world.arm_count)
    probs = policy.pi(_X())
    assert probs.shape == (10, world.arm_count)


def test_constant_policy_all_mass_on_arm(world: OpenMLCC18World) -> None:
    K = world.arm_count
    for arm in range(K):
        policy = ConstantPolicy(arm=arm, arm_count=K)
        probs = policy.pi(_X(5))
        assert np.all(probs[:, arm] == 1.0)
        assert np.all(np.delete(probs, arm, axis=1) == 0.0)


# ---------------------------------------------------------------------------
# UniformPolicy
# ---------------------------------------------------------------------------


def test_uniform_policy_shape(world: OpenMLCC18World) -> None:
    policy = UniformPolicy(arm_count=world.arm_count)
    probs = policy.pi(_X(8))
    assert probs.shape == (8, world.arm_count)


def test_uniform_policy_sums_to_one(world: OpenMLCC18World) -> None:
    policy = UniformPolicy(arm_count=world.arm_count)
    probs = policy.pi(_X())
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# MostFrequentPolicy
# ---------------------------------------------------------------------------


def test_most_frequent_policy_valid(world: OpenMLCC18World) -> None:
    policy = MostFrequentPolicy(world=world)
    probs = policy.pi(_X())
    assert probs.shape == (10, world.arm_count)
    # Exactly one arm has probability 1 per row
    assert np.all(probs.sum(axis=1) == pytest.approx(1.0))
    assert np.all((probs == 0) | (probs == 1))


# ---------------------------------------------------------------------------
# FrequencyPolicy
# ---------------------------------------------------------------------------


def test_frequency_policy_sums_to_one(world: OpenMLCC18World) -> None:
    policy = FrequencyPolicy(world=world)
    probs = policy.pi(_X())
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_frequency_policy_nonnegative(world: OpenMLCC18World) -> None:
    policy = FrequencyPolicy(world=world)
    probs = policy.pi(_X())
    assert np.all(probs >= 0.0)


# ---------------------------------------------------------------------------
# ContextualPolicy
# ---------------------------------------------------------------------------


def test_contextual_policy_shape(world: OpenMLCC18World) -> None:
    policy = ContextualPolicy(world=world, outcome_model=LinearRegression(), train_sample_size=50)
    probs = policy.pi(world.sample_contexts(10))
    assert probs.shape == (10, world.arm_count)


def test_contextual_policy_one_hot(world: OpenMLCC18World) -> None:
    """Greedy policy puts all mass on a single arm."""
    policy = ContextualPolicy(world=world, outcome_model=LinearRegression(), train_sample_size=50)
    probs = policy.pi(world.sample_contexts(20))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs == 0) | (probs == 1))


# ---------------------------------------------------------------------------
# make_target_policy factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "non_contextual_constant_0",
    "non_contextual_uniform_random",
    "non_contextual_most_frequent",
    "non_contextual_frequency_proportional",
])
def test_factory_non_contextual(world: OpenMLCC18World, name: str) -> None:
    policy = make_target_policy(name, world)
    probs = policy.pi(_X())
    assert probs.shape == (10, world.arm_count)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_factory_contextual(world: OpenMLCC18World) -> None:
    policy = make_target_policy(
        "contextual", world,
        outcome_model=DecisionTreeRegressor(),
        train_sample_size=40,
    )
    probs = policy.pi(world.sample_contexts(5))
    assert probs.shape == (5, world.arm_count)


def test_factory_contextual_missing_args(world: OpenMLCC18World) -> None:
    with pytest.raises(ValueError, match="requires"):
        make_target_policy("contextual", world)


def test_factory_unknown_name(world: OpenMLCC18World) -> None:
    with pytest.raises(ValueError, match="Unknown policy"):
        make_target_policy("bogus_policy", world)
