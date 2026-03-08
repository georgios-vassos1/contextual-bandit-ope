"""Tests for compute_coverage_metrics."""
from __future__ import annotations

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
    """Estimator always equals truth exactly → coverage = 1.0."""
    truth = 0.5
    results = [_make_result(mean=truth, var=0.0, truth=truth) for _ in range(20)]
    metrics = compute_coverage_metrics(results)
    # dm mean should be 1.0 (always covered since mean == truth and var=0 → CI is a point at truth)
    # Actually with var=0 the CI collapses to a point so coverage depends on ties; let's use tiny var
    results2 = [_make_result(mean=truth, var=1e-6, truth=truth) for _ in range(20)]
    metrics2 = compute_coverage_metrics(results2)
    assert float(metrics2["dm_mean"]) == pytest.approx(1.0)


def test_zero_coverage() -> None:
    """Estimator is very far from truth → coverage ≈ 0."""
    truth = 0.5
    far = 100.0
    results = [_make_result(mean=far, var=1e-10, truth=truth) for _ in range(20)]
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
