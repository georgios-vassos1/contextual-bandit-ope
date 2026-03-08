from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyRegressor

from pcmabinf._utils import predict
from pcmabinf.world import OpenMLCC18World


class TargetPolicyProtocol(Protocol):
    def pi(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return (N, K) action probability matrix."""
        ...


# ---------------------------------------------------------------------------
# Non-contextual policies
# ---------------------------------------------------------------------------


class ConstantPolicy:
    """Always selects a fixed arm with probability 1."""

    def __init__(self, arm: int, arm_count: int) -> None:
        self._assignment = np.zeros(arm_count, dtype=np.float64)
        self._assignment[arm] = 1.0

    def pi(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.tile(self._assignment, (len(X), 1))


class UniformPolicy:
    """Selects each arm with equal probability."""

    def __init__(self, arm_count: int) -> None:
        self._assignment = np.full(arm_count, 1.0 / arm_count, dtype=np.float64)

    def pi(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.tile(self._assignment, (len(X), 1))


class MostFrequentPolicy:
    """Always selects the most frequent arm (by label frequency) with probability 1."""

    def __init__(self, world: OpenMLCC18World) -> None:
        labels, counts = np.unique(world.arms, return_counts=True)
        self._assignment = np.zeros(world.arm_count, dtype=np.float64)
        # Use labels[argmax] so the correct arm slot is set even for non-contiguous labels.
        most_frequent_arm = int(labels[np.argmax(counts)])
        self._assignment[most_frequent_arm] = 1.0

    def pi(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.tile(self._assignment, (len(X), 1))


class FrequencyPolicy:
    """Selects arms proportional to their frequency in the dataset labels."""

    def __init__(self, world: OpenMLCC18World) -> None:
        labels, counts = np.unique(world.arms, return_counts=True)
        freq = counts.astype(np.float64) / counts.sum()
        self._assignment = np.zeros(world.arm_count, dtype=np.float64)
        # Index by actual label values to handle non-contiguous arm labels correctly.
        self._assignment[labels.astype(int)] = freq

    def pi(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.tile(self._assignment, (len(X), 1))


class ContextualPolicy:
    """Greedy contextual policy that places all probability on the best-predicted arm."""

    def __init__(
        self,
        world: OpenMLCC18World,
        outcome_model: BaseEstimator,
        train_sample_size: int,
    ) -> None:
        self._arm_count = world.arm_count
        contexts = world.sample_contexts(train_sample_size)
        arms = np.random.choice(self._arm_count, size=len(contexts))
        rewards = world.rewards_batch(contexts, arms.astype(np.intp))

        self._models: list[BaseEstimator] = []
        for a in range(self._arm_count):
            idx = np.where(arms == a)[0]
            if len(idx) == 0:
                # No training data for this arm; predict 0.0 so it is never preferred.
                m: BaseEstimator = DummyRegressor(strategy="constant", constant=0.0)
                m.fit(contexts[:1], np.array([0.0]))
            else:
                m = clone(outcome_model)
                m.fit(contexts[idx], rewards[idx])
            self._models.append(m)

    def pi(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        Y_hat = np.column_stack([predict(m, X) for m in self._models])
        best = np.argmax(Y_hat, axis=1)
        probs = np.zeros((len(X), self._arm_count), dtype=np.float64)
        probs[np.arange(len(X)), best] = 1.0  # vectorised scatter; no Python loop
        return probs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_target_policy(
    name: str,
    world: OpenMLCC18World,
    outcome_model: BaseEstimator | None = None,
    train_sample_size: int | None = None,
) -> TargetPolicyProtocol:
    """Instantiate a target policy by name.

    Accepted names
    --------------
    ``non_contextual_constant_{arm}``  e.g. ``non_contextual_constant_0``
    ``non_contextual_uniform_random``
    ``non_contextual_most_frequent``
    ``non_contextual_frequency_proportional``
    ``contextual``  (requires *outcome_model* and *train_sample_size*)
    """
    if name.startswith("non_contextual_constant_"):
        arm = int(name.removeprefix("non_contextual_constant_"))
        return ConstantPolicy(arm=arm, arm_count=world.arm_count)
    if name == "non_contextual_uniform_random":
        return UniformPolicy(arm_count=world.arm_count)
    if name == "non_contextual_most_frequent":
        return MostFrequentPolicy(world=world)
    if name == "non_contextual_frequency_proportional":
        return FrequencyPolicy(world=world)
    if name == "contextual":
        if outcome_model is None or train_sample_size is None:
            raise ValueError("contextual policy requires outcome_model and train_sample_size")
        return ContextualPolicy(
            world=world,
            outcome_model=outcome_model,
            train_sample_size=train_sample_size,
        )
    raise ValueError(f"Unknown policy name: {name!r}")
