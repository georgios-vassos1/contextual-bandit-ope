from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class BanditData:
    """Data collected by a logging policy over multiple batches."""

    X: NDArray[np.float64]  # (N, d) contexts
    A: NDArray[np.intp]  # (N,)   chosen arms
    Y: NDArray[np.float64]  # (N,)   observed rewards
    P: NDArray[np.float64]  # (N,)   logging propensities P(A|X) under logging policy
    propensity_history: list[NDArray[np.float64]]  # per-batch re-evaluated propensities (for CADR)
    epsilon: NDArray[np.float64]  # (N,)   epsilon values at each step
    regret: NDArray[np.intp]  # (N,)   0/1 regret
    batch_size: int

    def __post_init__(self) -> None:
        N = self.X.shape[0]
        for name, arr in (("A", self.A), ("Y", self.Y), ("P", self.P),
                          ("epsilon", self.epsilon), ("regret", self.regret)):
            if arr.shape != (N,):
                raise ValueError(
                    f"BanditData.{name} must have shape ({N},), got {arr.shape}"
                )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")


@dataclass
class OPEResult:
    """Results from all OPE estimators for a single simulation run.

    Each field is a (mean, variance) tuple or None if not computed.
    """

    truth: tuple[float, float] | None = None
    dm: tuple[float, float] | None = None
    ips: tuple[float, float] | None = None
    dr: tuple[float, float] | None = None
    adr: tuple[float, float] | None = None
    cadr: tuple[float, float] | None = None
    mrdr: tuple[float, float] | None = None
    camrdr: tuple[float, float] | None = None
