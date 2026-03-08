from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pcmabinf.data import OPEResult

ESTIMATOR_FIELDS: list[str] = ["truth", "dm", "ips", "dr", "adr", "cadr", "mrdr", "camrdr"]


def compute_coverage_metrics(
    ope_results: list[OPEResult],
    estimators: list[str] = ESTIMATOR_FIELDS,
    ci_level: float = 0.95,
) -> dict[str, NDArray[np.float64]]:
    """Compute 95 % CI coverage for each estimator relative to truth.

    For each simulation replicate *s* and estimator *e*, a CI is formed as::

        [mean_e - z * sqrt(var_e), mean_e + z * sqrt(var_e)]

    where ``z = scipy.stats.norm.ppf(0.5 + ci_level / 2)``.  Coverage is the
    fraction of replicates in which the CI contains the truth estimate.

    Parameters
    ----------
    ope_results:
        One :class:`OPEResult` per simulation replicate.
    estimators:
        Subset of :data:`ESTIMATOR_FIELDS` to include.
    ci_level:
        Nominal coverage level (default 0.95).

    Returns
    -------
    dict
        Keys ``'{estimator}_mean'`` and ``'{estimator}_stderr'`` for each
        estimator in *estimators*.  Values are scalars wrapped in 0-d arrays.
    """
    from scipy.stats import norm  # local import to avoid top-level dependency at import time

    z = float(norm.ppf(0.5 + ci_level / 2.0))
    S = len(ope_results)

    # Collect (mean, variance) for each estimator across simulations
    all_results: dict[str, NDArray[np.float64]] = {}
    for name in estimators:
        vals = np.array(
            [getattr(r, name) for r in ope_results if getattr(r, name) is not None],
            dtype=np.float64,
        )
        all_results[name] = vals  # shape (S, 2)

    truth_means = all_results["truth"][:, 0]  # shape (S,)

    out: dict[str, NDArray[np.float64]] = {}
    for name in estimators:
        vals = all_results[name]
        means = vals[:, 0]
        stds = np.sqrt(np.maximum(vals[:, 1], 0.0))
        lo = means - z * stds
        hi = means + z * stds
        covered = ((lo <= truth_means) & (truth_means <= hi)).astype(np.float64)
        out[f"{name}_mean"] = np.array(np.mean(covered))
        out[f"{name}_stderr"] = np.array(np.sqrt(np.var(covered) / S))
    return out
