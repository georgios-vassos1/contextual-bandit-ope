from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from pcmabinf.cross_fitting import estimate_outcome_models
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

        arm_count = world.arm_count
        N = len(data.A)

        # Cross-fitted outcome model matrices: shape (N, K) each.
        Q, Q_MRDR, Q_CAMRDR = estimate_outcome_models(
            data, target_policy, outcome_model, arm_count, n_folds
        )

        self._Q = Q
        self._Q_MRDR = Q_MRDR
        self._Q_CAMRDR = Q_CAMRDR

        # Row indices — computed once, reused by every estimator method.
        self._row_idx = np.arange(N)

        # Target policy action probabilities: (N, K)
        self._g_star: NDArray[np.float64] = target_policy.pi(data.X)

        # Oracle expected reward under the world: (N, K)
        # One scatter write sets the optimal arm's column to 1; all others remain 0.
        optimal_arms = world.optimal_arms_batch(data.X)
        self._Y_true = np.zeros((N, arm_count), dtype=np.float64)
        self._Y_true[self._row_idx, optimal_arms] = 1.0

        # Per-observation quantities indexed at the chosen arm — cached once.
        self._g_star_a: NDArray[np.float64] = self._g_star[self._row_idx, data.A]
        self._q_taken: NDArray[np.float64] = Q[self._row_idx, data.A]
        self._q_mrdr_taken: NDArray[np.float64] = Q_MRDR[self._row_idx, data.A]
        self._q_camrdr_taken: NDArray[np.float64] = Q_CAMRDR[self._row_idx, data.A]

        # Policy-value weighted sums: V(x_i) = sum_a pi*(a|x_i) Q(x_i, a)
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
    # Estimators
    # ------------------------------------------------------------------

    def _truth(self) -> tuple[float, float]:
        # Oracle: variance is zero by definition (expected value, not a sample).
        mean, _ = self._summarise(self._Q0_star)
        return mean, 0.0

    def _dm(self) -> tuple[float, float]:
        return self._summarise(self._Q_star)

    def _ips(self) -> tuple[float, float]:
        phi = (self._g_star_a / self.data.P) * self.data.Y
        return self._summarise(phi)

    def _dr(self) -> tuple[float, float]:
        phi = self._Q_star + (self._g_star_a / self.data.P) * (self.data.Y - self._q_taken)
        return self._summarise(phi)

    def _adr(self) -> tuple[float, float]:
        phi = self._Q_star + (self._g_star_a / self.data.P) * (self.data.Y - self._q_taken)
        w = np.sqrt(self.data.P)
        return self._summarise(phi, w)

    def _cadr(self) -> tuple[float, float]:
        return self._adaptive_dr(self._Q_star, self._q_taken)

    def _mrdr(self) -> tuple[float, float]:
        phi = (
            self._Q_MRDR_star
            + (self._g_star_a / self.data.P) * (self.data.Y - self._q_mrdr_taken)
        )
        return self._summarise(phi)

    def _camrdr(self) -> tuple[float, float]:
        return self._adaptive_dr(self._Q_CAMRDR_star, self._q_camrdr_taken)

    # ------------------------------------------------------------------
    # Adaptive doubly-robust weighting (shared by CADR and CAMRDR)
    # ------------------------------------------------------------------

    def _adaptive_dr(
        self,
        Q_star: NDArray[np.float64],
        q_taken: NDArray[np.float64],
    ) -> tuple[float, float]:
        """Compute the CADR/CAMRDR estimator with observation-level adaptive weights.

        For observation i, the weight is ``1 / sqrt(sigma2_i)`` where
        ``sigma2_i`` is the empirical variance of the DR functional evaluated
        over all observations before i using the propensity history entry for
        i's batch.  Weight zero is assigned when the variance estimate is
        non-positive.

        Vectorised over observations within each batch using prefix cumsums,
        giving O(B) passes instead of the naive O(N^2).
        """
        data = self.data
        P, Y = data.P, data.Y
        N = len(P)
        bs = data.batch_size
        B = len(data.propensity_history)

        # Full doubly-robust functional (vectorised).
        phi = Q_star + (self._g_star_a / P) * (Y - q_taken)
        w = np.zeros(N, dtype=np.float64)  # w[0] stays zero

        for b in range(B):
            # Retroactive propensity under model b for all observations 0..n_b-1.
            pi_b = data.propensity_history[b]
            n_b = len(pi_b)

            # Importance-ratio weighted influence values for the prefix 0..n_b-1.
            rho = pi_b / P[:n_b]
            phi_b = Q_star[:n_b] + (self._g_star_a[:n_b] / pi_b) * (Y[:n_b] - q_taken[:n_b])

            # Prefix cumsums let us read off the mean and second moment at any
            # index i in O(1) rather than re-summing from scratch.
            cs1 = np.cumsum(rho * phi_b ** 2)
            cs2 = np.cumsum(rho * phi_b)

            i_start = max(1, b * bs)
            i_end = min((b + 1) * bs, N)
            if i_start >= i_end:
                continue

            idx = np.arange(i_start, i_end)
            variance_est = cs1[idx - 1] / idx - (cs2[idx - 1] / idx) ** 2
            # Guard against sqrt(0): use 1.0 in the inactive branch of np.where.
            safe_var = np.where(variance_est > 0.0, variance_est, 1.0)
            w[idx] = np.where(variance_est > 0.0, 1.0 / np.sqrt(safe_var), 0.0)

        return self._summarise(phi, w)

    # ------------------------------------------------------------------
    # Shared utility
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise(
        phi: NDArray[np.float64],
        w: NDArray[np.float64] | None = None,
    ) -> tuple[float, float]:
        """Return the (mean, variance-of-mean) estimate of a policy value.

        Uses the weighted mean ``(phi . w) / sum(w)`` and the sandwich
        variance ``sum(w_i^2 (phi_i - mu)^2) / (sum w)^2``.  Falls back to
        uniform weights when *w* is None or all-zero.
        """
        if w is None:
            w = np.ones_like(phi)
        w_sum = w.sum()
        if w_sum == 0.0:
            w = np.ones_like(phi)
            w_sum = float(len(phi))
        mean = float(phi.dot(w) / w_sum)
        variance = float((w ** 2).dot((phi - mean) ** 2) / w_sum ** 2)
        return mean, variance
