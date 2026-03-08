from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from pcmabinf.cross_fitting import cross_fitting
from pcmabinf.data import BanditData, OPEResult
from pcmabinf.policy import TargetPolicyProtocol
from pcmabinf.world import OpenMLCC18World


class OPEEstimator:
    """Compute all OPE estimators (DM, IPS, DR, ADR, CADR, MRDR, CAMRDR) for a
    single simulation run.

    Parameters
    ----------
    data:
        Bandit data collected by the logging policy.
    world:
        The bandit environment (used only to compute the ground-truth value).
    target_policy:
        The policy being evaluated.
    outcome_model:
        Sklearn-compatible estimator used for cross-fitting.
    n_folds:
        Number of cross-fitting folds.
    """

    def __init__(
        self,
        data: BanditData,
        world: OpenMLCC18World,
        target_policy: TargetPolicyProtocol,
        outcome_model: BaseEstimator,
        n_folds: int = 4,
    ) -> None:
        self.data = data
        self.world = world
        self.target_policy = target_policy
        self.n_folds = n_folds

        arm_count = world.arm_count

        # Cross-fitting — no duplication of this logic elsewhere
        Q, Q_MRDR, Q_CAMRDR = cross_fitting(
            data, target_policy, outcome_model, arm_count, n_folds
        )

        self._Q = Q
        self._Q_MRDR = Q_MRDR
        self._Q_CAMRDR = Q_CAMRDR

        # Target policy action probabilities
        self._g_star: NDArray[np.float64] = target_policy.pi(data.X)

        # True expected reward E[Y | A=a, X=x] under the world
        self._Y_true: NDArray[np.float64] = np.array(
            [world.reward_mean_per_arm(x) for x in data.X], dtype=np.float64
        )

        # Convenience scalars per observation
        self._Q_star = np.einsum("ij,ij->i", self._g_star, Q)
        self._Q_MRDR_star = np.einsum("ij,ij->i", self._g_star, Q_MRDR)
        self._Q_CAMRDR_star = np.einsum("ij,ij->i", self._g_star, Q_CAMRDR)
        self._Q0_star = np.einsum("ij,ij->i", self._g_star, self._Y_true)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all(self) -> OPEResult:
        return OPEResult(
            truth=self._truth(),
            dm=self._dm(),
            ips=self._ips(),
            dr=self._dr(),
            adr=self._adr(),
            cadr=self._cadr(),
            mrdr=self._mrdr(),
            camrdr=self._camrdr(),
        )

    # ------------------------------------------------------------------
    # Private estimators
    # ------------------------------------------------------------------

    def _truth(self) -> tuple[float, float]:
        mean, _ = self._mean_and_variance(self._Q0_star)
        return mean, 0.0

    def _dm(self) -> tuple[float, float]:
        return self._mean_and_variance(self._Q_star)

    def _ips(self) -> tuple[float, float]:
        A, P, Y = self.data.A, self.data.P, self.data.Y
        D = (self._g_star[np.arange(len(A)), A] / P) * Y
        return self._mean_and_variance(D)

    def _dr(self) -> tuple[float, float]:
        A, P, Y = self.data.A, self.data.P, self.data.Y
        g_star_a = self._g_star[np.arange(len(A)), A]
        D = self._Q_star + (g_star_a / P) * (Y - self._Q[np.arange(len(A)), A])
        return self._mean_and_variance(D)

    def _adr(self) -> tuple[float, float]:
        A, P, Y = self.data.A, self.data.P, self.data.Y
        g_star_a = self._g_star[np.arange(len(A)), A]
        D = self._Q_star + (g_star_a / P) * (Y - self._Q[np.arange(len(A)), A])
        w = np.sqrt(P)
        return self._mean_and_variance(D, w)

    def _cadr(self) -> tuple[float, float]:
        return self._adaptive_dr(self._Q_star, self._Q)

    def _mrdr(self) -> tuple[float, float]:
        A, P, Y = self.data.A, self.data.P, self.data.Y
        g_star_a = self._g_star[np.arange(len(A)), A]
        D = self._Q_MRDR_star + (g_star_a / P) * (Y - self._Q_MRDR[np.arange(len(A)), A])
        return self._mean_and_variance(D)

    def _camrdr(self) -> tuple[float, float]:
        return self._adaptive_dr(self._Q_CAMRDR_star, self._Q_CAMRDR)

    # ------------------------------------------------------------------
    # Helper: adaptive doubly-robust (CADR / CAMRDR)
    # ------------------------------------------------------------------

    def _adaptive_dr(
        self,
        Q_star: NDArray[np.float64],
        Q: NDArray[np.float64],
    ) -> tuple[float, float]:
        """Generic CADR computation.

        w_i = 0                      if i == 0
        w_i = 1/sqrt(sigma2_i)       otherwise, where sigma2_i is estimated
              from previous observations using the per-batch propensity history.
        """
        data = self.data
        A, P, Y = data.A, data.P, data.Y
        N = len(A)
        bs = data.batch_size

        D = np.empty(N, dtype=np.float64)
        w = np.zeros(N, dtype=np.float64)

        g_star_a = self._g_star[np.arange(N), A]

        for i in range(N):
            D[i] = Q_star[i] + (g_star_a[i] / P[i]) * (Y[i] - Q[i, A[i]])

            if i == 0:
                continue

            s = np.arange(i)
            gs_s = P[s]
            batch_idx = i // bs
            gt_s = data.propensity_history[batch_idx][s]

            D1_gt = Q_star[s] + (g_star_a[s] / gt_s) * (Y[s] - Q[s, A[s]])

            ratio = gt_s / gs_s
            sigma2 = np.mean(ratio * D1_gt**2) - np.mean(ratio * D1_gt) ** 2
            w[i] = 0.0 if sigma2 <= 0.0 else 1.0 / np.sqrt(sigma2)

        return self._mean_and_variance(D, w)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_and_variance(
        D: NDArray[np.float64],
        w: NDArray[np.float64] | None = None,
    ) -> tuple[float, float]:
        if w is None:
            w = np.ones_like(D)
        w_sum = w.sum()
        mean = float(D.dot(w) / w_sum)
        variance = float((w**2).dot((D - mean) ** 2) / w_sum**2)
        return mean, variance
