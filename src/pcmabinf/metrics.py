from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from pcmabinf.data import OPEResult

ESTIMATOR_FIELDS: list[str] = ["truth", "dm", "ips", "dr", "adr", "cadr", "mrdr", "camrdr"]


def compute_coverage_metrics(
    ope_results: list[OPEResult],
    estimators: list[str] = ESTIMATOR_FIELDS,
    ci_level: float = 0.95,
) -> dict[str, NDArray[np.float64]]:
    """Compute CI coverage for each estimator relative to truth.

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

    Raises
    ------
    ValueError
        If ``ci_level`` is not in (0, 1).
    ValueError
        If all results for ``'truth'`` are ``None``.
    """
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")

    from scipy.stats import norm  # local import to avoid top-level dependency at import time

    z = float(norm.ppf(0.5 + ci_level / 2.0))
    n_reps = len(ope_results)

    # Collect raw (mean, variance) tuples and record which indices are valid.
    estimator_vals: dict[str, list[tuple[float, float] | None]] = {
        name: [getattr(r, name) for r in ope_results] for name in estimators
    }

    # Warn about None entries per estimator.
    for name in estimators:
        n_none = sum(v is None for v in estimator_vals[name])
        if n_none > 0:
            warnings.warn(
                f"{n_none}/{n_reps} OPEResult entries have {name}=None and will be excluded "
                f"from coverage computation.",
                stacklevel=2,
            )
        if all(v is None for v in estimator_vals[name]):
            raise ValueError(f"No valid results for estimator '{name}'")

    out: dict[str, NDArray[np.float64]] = {}
    truth_vals = estimator_vals["truth"]

    for name in estimators:
        # Use only replicates where BOTH truth and the current estimator are non-None.
        aligned = [
            (t, e)
            for t, e in zip(truth_vals, estimator_vals[name])
            if t is not None and e is not None
        ]
        if len(aligned) == 0:
            raise ValueError(f"No replicates with valid results for both 'truth' and '{name}'")

        t_arr = np.array([p[0] for p in aligned], dtype=np.float64)
        e_arr = np.array([p[1] for p in aligned], dtype=np.float64)

        truth_means = t_arr[:, 0]
        est_means = e_arr[:, 0]
        stds = np.sqrt(np.maximum(e_arr[:, 1], 0.0))
        lo = est_means - z * stds
        hi = est_means + z * stds
        hits = ((lo <= truth_means) & (truth_means <= hi)).astype(np.float64)
        n_valid = len(hits)
        out[f"{name}_mean"] = np.array(np.mean(hits))
        out[f"{name}_stderr"] = np.array(np.sqrt(np.var(hits) / n_valid))
    return out
