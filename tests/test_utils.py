"""Tests for _utils.score()."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from pcmabinf._utils import score


def _fit_classifier(X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)
    return clf


def test_regressor_score() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    y = rng.standard_normal(50)
    model = DecisionTreeRegressor(random_state=0)
    model.fit(X, y)
    out = score(model, X)
    assert out.shape == (50,)
    np.testing.assert_allclose(out, model.predict(X))


def test_classifier_multiclass_predict_proba() -> None:
    """Binary classifier → predict_proba[:, 1]."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 3))
    y = (X[:, 0] > 0).astype(int)  # two classes guaranteed
    clf = _fit_classifier(X, y)
    out = score(clf, X)
    assert out.shape == (60,)
    np.testing.assert_allclose(out, clf.predict_proba(X)[:, 1])


def test_classifier_single_class_returns_constant() -> None:
    """When training data has only one class, return that constant."""
    rng = np.random.default_rng(2)
    X_train = rng.standard_normal((20, 3))
    y_train = np.zeros(20, dtype=int)  # only class 0

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    assert len(clf.classes_) == 1

    X_test = rng.standard_normal((10, 3))
    out = score(clf, X_test)
    assert out.shape == (10,)
    np.testing.assert_array_equal(out, np.zeros(10))
