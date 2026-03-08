from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.utils import resample


class OpenMLCC18World:
    """Bandit environment wrapping an OpenML-CC18 classification task.

    The dataset is loaded from a pickle file (features, targets), NaN values are
    imputed, and the rows are shuffled.  The optimal arm for each context is the
    original class label.
    """

    def __init__(
        self,
        task_id: int | str,
        data_dir: Path,
        reward_variance: float = 0.0,
    ) -> None:
        if reward_variance < 0:
            raise ValueError(f"reward_variance must be >= 0, got {reward_variance}")

        self.task_id = task_id
        self.reward_variance = reward_variance

        path = Path(data_dir) / str(task_id)
        with open(path, "rb") as fh:
            contexts, arms = pickle.load(fh)

        if np.sum(np.isnan(contexts)) > 0:
            contexts = SimpleImputer().fit_transform(contexts)

        self._arm_count: int = int(len(np.unique(arms)))
        self._feature_count: int = int(contexts.shape[1])
        self._observation_count: int = int(contexts.shape[0])

        shuffle = np.random.permutation(self._observation_count)
        self.contexts: NDArray[np.float64] = contexts[shuffle].astype(np.float64)
        self.arms: NDArray[np.intp] = arms[shuffle].astype(np.intp)

        self._context_to_optimal_arm: dict[tuple[float, ...], int] = {
            tuple(map(float, c)): int(a) for c, a in zip(self.contexts, self.arms)
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def arm_count(self) -> int:
        return self._arm_count

    @property
    def feature_count(self) -> int:
        return self._feature_count

    @property
    def observation_count(self) -> int:
        return self._observation_count

    # ------------------------------------------------------------------
    # World interface
    # ------------------------------------------------------------------

    def _optimal_arm(self, x: NDArray[np.float64]) -> int:
        return self._context_to_optimal_arm[tuple(map(float, x))]

    def reward(self, x: NDArray[np.float64], arm: int) -> float:
        reward_mean = float(arm == self._optimal_arm(x))
        return reward_mean + np.random.normal(loc=0.0, scale=np.sqrt(self.reward_variance))

    def regret(self, x: NDArray[np.float64], arm: int) -> int:
        return int(arm != self._optimal_arm(x))

    def reward_mean_per_arm(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        means = np.zeros(self._arm_count, dtype=np.float64)
        means[self._optimal_arm(x)] = 1.0
        return means

    def sample_contexts(self, n: int) -> NDArray[np.float64]:
        """Return *n* bootstrap-sampled contexts (with replacement)."""
        return resample(self.contexts, n_samples=n)  # type: ignore[return-value]

    def sample_labeled(self, n: int) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
        """Return *n* bootstrap-sampled (contexts, labels) pairs (with replacement)."""
        ctx, lbl = resample(self.contexts, self.arms, n_samples=n)
        return ctx, lbl  # type: ignore[return-value]
