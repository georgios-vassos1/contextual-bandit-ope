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
        N = len(data.A)

        # Cross-fitting — no duplication of this logic elsewhere
        Q, Q_MRDR, Q_CAMRDR = cross_fitting(
            data, target_policy, outcome_model, arm_count, n_folds
        )

        self._Q = Q
        self._Q_MRDR = Q_MRDR
        self._Q_CAMRDR = Q_CAMRDR

        # Target policy action probabilities: (N, K)
        self._g_star: NDArray[np.float64] = target_policy.pi(data.X)

        # True expected reward E[Y | A=a, X=x] under the world: (N, K)
        self._Y_true: NDArray[np.float64] = np.array(
            [world.reward_mean_per_arm(x) for x in data.X], dtype=np.float64
        )

        # Row indices — computed once, reused by every estimator method.
        self._row_idx = np.arange(N)

        # Per-observation selected-arm quantities — computed once, reused everywhere.
        # g*(a_i | x_i): probability the target policy assigns to the arm that was taken.
        self._g_star_a: NDArray[np.float64] = self._g_star[self._row_idx, data.A]
        # Q(x_i, a_i) for each outcome model variant.
        self._Q_sel: NDArray[np.float64] = Q[self._row_idx, data.A]
        self._Q_MRDR_sel: NDArray[np.float64] = Q_MRDR[self._row_idx, data.A]
        self._Q_CAMRDR_sel: NDArray[np.float64] = Q_CAMRDR[self._row_idx, data.A]

        # Weighted sums over arms: V^(a)(x_i) = sum_a g*(a|x_i) Q(x_i,a)
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
        # Variance is 0 by definition: truth is the deterministic expected value.
        mean, _ = self._mean_and_variance(self._Q0_star)
        return mean, 0.0

    def _dm(self) -> tuple[float, float]:
        return self._mean_and_variance(self._Q_star)

    def _ips(self) -> tuple[float, float]:
        D = (self._g_star_a / self.data.P) * self.data.Y
        return self._mean_and_variance(D)

    def _dr(self) -> tuple[float, float]:
        D = self._Q_star + (self._g_star_a / self.data.P) * (self.data.Y - self._Q_sel)
        return self._mean_and_variance(D)

    def _adr(self) -> tuple[float, float]:
        # Weight w_i = sqrt(g(a_i|x_i)) — stabilises variance under low-propensity arms.
        D = self._Q_star + (self._g_star_a / self.data.P) * (self.data.Y - self._Q_sel)
        w = np.sqrt(self.data.P)
        return self._mean_and_variance(D, w)

    def _cadr(self) -> tuple[float, float]:
        return self._adaptive_dr(self._Q_star, self._Q_sel)

    def _mrdr(self) -> tuple[float, float]:
        D = self._Q_MRDR_star + (self._g_star_a / self.data.P) * (self.data.Y - self._Q_MRDR_sel)
        return self._mean_and_variance(D)

    def _camrdr(self) -> tuple[float, float]:
        return self._adaptive_dr(self._Q_CAMRDR_star, self._Q_CAMRDR_sel)

    # ------------------------------------------------------------------
    # Helper: adaptive doubly-robust (CADR / CAMRDR)
    # ------------------------------------------------------------------

    def _adaptive_dr(
        self,
        Q_star: NDArray[np.float64],
        Q_sel: NDArray[np.float64],  # pre-computed Q(x_i, a_i)
    ) -> tuple[float, float]:
        """Generic CADR computation.

        w_i = 0                      if i == 0
        w_i = 1/sqrt(sigma2_i)       otherwise, where sigma2_i is estimated
              from previous observations using the per-batch propensity history.

        Vectorized implementation: O(B) numpy passes instead of O(N²).

        Key observation: batch_idx = i // bs is constant for all i in
        [b*bs, (b+1)*bs), so ratio_s and D1_gt_s are constant within a batch.
        cumsum lets us read off sigma2_i for every i in the batch at once.
        """
        data = self.data
        P, Y = data.P, data.Y
        N = len(P)
        bs = data.batch_size
        B = len(data.propensity_history)

        # D is fully vectorized — no loop needed.
        D = Q_star + (self._g_star_a / P) * (Y - Q_sel)
        w = np.zeros(N, dtype=np.float64)  # w[0] stays 0 by definition

        for b in range(B):
            gt = data.propensity_history[b]   # shape: ((b+1)*bs,)
            n_b = len(gt)

            ratio = gt / P[:n_b]
            D1 = Q_star[:n_b] + (self._g_star_a[:n_b] / gt) * (Y[:n_b] - Q_sel[:n_b])

            cs1 = np.cumsum(ratio * D1 ** 2)  # cumsum of psi1
            cs2 = np.cumsum(ratio * D1)       # cumsum of psi2

            i_start = max(1, b * bs)
            i_end = min((b + 1) * bs, N)
            if i_start >= i_end:
                continue

            idx = np.arange(i_start, i_end)
            sigma2 = cs1[idx - 1] / idx - (cs2[idx - 1] / idx) ** 2
            safe = np.where(sigma2 > 0.0, sigma2, 1.0)  # avoid sqrt(0) in inactive branch
            w[idx] = np.where(sigma2 > 0.0, 1.0 / np.sqrt(safe), 0.0)

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
        if w_sum == 0.0:
            # Degenerate case (e.g. only one observation, or all adaptive weights
            # are zero because variance could not be estimated).  Fall back to the
            # unweighted mean so downstream code always receives a finite value.
            w = np.ones_like(D)
            w_sum = float(len(D))
        mean = float(D.dot(w) / w_sum)
        variance = float((w**2).dot((D - mean) ** 2) / w_sum**2)
        return mean, variance
