"""Tests for compute_coverage_metrics."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from pcmabinf.data import OPEResult
from pcmabinf.metrics import compute_coverage_metrics


def _make_result(mean: float, var: float, truth: float = 0.5) -> OPEResult:
    return OPEResult(
        truth=(truth, 0.0),
        dm=(mean, var),
        ips=(mean, var),
        dr=(mean, var),
        adr=(mean, var),
        cadr=(mean, var),
        mrdr=(mean, var),
        camrdr=(mean, var),
    )


def test_perfect_coverage() -> None:
    """Estimator CI always contains truth → coverage = 1.0."""
    truth = 0.5
    results = [_make_result(mean=truth, var=1e-6, truth=truth) for _ in range(20)]
    metrics = compute_coverage_metrics(results)
    assert float(metrics["dm_mean"]) == pytest.approx(1.0)


def test_zero_coverage() -> None:
    """Estimator is very far from truth → coverage = 0."""
    truth = 0.5
    results = [_make_result(mean=100.0, var=1e-10, truth=truth) for _ in range(20)]
    metrics = compute_coverage_metrics(results)
    assert float(metrics["dm_mean"]) == pytest.approx(0.0)


def test_output_keys() -> None:
    results = [_make_result(0.5, 0.1) for _ in range(10)]
    metrics = compute_coverage_metrics(results)
    expected_keys = {
        f"{e}_{s}"
        for e in ["truth", "dm", "ips", "dr", "adr", "cadr", "mrdr", "camrdr"]
        for s in ["mean", "stderr"]
    }
    assert set(metrics.keys()) == expected_keys


def test_stderr_nonnegative() -> None:
    results = [_make_result(0.5 + i * 0.01, 0.1) for i in range(20)]
    metrics = compute_coverage_metrics(results)
    for key, val in metrics.items():
        if "stderr" in key:
            assert float(val) >= 0.0


def test_invalid_ci_level_raises() -> None:
    results = [_make_result(0.5, 0.1) for _ in range(5)]
    with pytest.raises(ValueError, match="ci_level"):
        compute_coverage_metrics(results, ci_level=95.0)
    with pytest.raises(ValueError, match="ci_level"):
        compute_coverage_metrics(results, ci_level=0.0)


def test_none_results_warn_and_are_excluded() -> None:
    results = [_make_result(0.5, 0.1) for _ in range(10)]
    # Inject a None for dm in two replicates
    results[0] = OPEResult(
        truth=(0.5, 0.0), dm=None, ips=(0.5, 0.1), dr=(0.5, 0.1),
        adr=(0.5, 0.1), cadr=(0.5, 0.1), mrdr=(0.5, 0.1), camrdr=(0.5, 0.1),
    )
    results[1] = OPEResult(
        truth=(0.5, 0.0), dm=None, ips=(0.5, 0.1), dr=(0.5, 0.1),
        adr=(0.5, 0.1), cadr=(0.5, 0.1), mrdr=(0.5, 0.1), camrdr=(0.5, 0.1),
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metrics = compute_coverage_metrics(results)
    assert any("dm" in str(warning.message) for warning in w)
    # Coverage should still be computable from the 8 valid replicates
    assert "dm_mean" in metrics


def test_all_none_raises() -> None:
    results = [
        OPEResult(truth=(0.5, 0.0), dm=None, ips=None, dr=None,
                  adr=None, cadr=None, mrdr=None, camrdr=None)
        for _ in range(5)
    ]
    with pytest.raises(ValueError, match="No valid results"):
        compute_coverage_metrics(results)
