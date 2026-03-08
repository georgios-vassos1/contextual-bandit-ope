from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, is_classifier


def score(model: BaseEstimator, X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return scalar outcome predictions for *X* from any sklearn-compatible estimator.

    Classifiers are handled by returning ``predict_proba(X)[:, 1]`` (the
    positive-class probability).  A degenerate classifier trained on a single
    class returns a constant array equal to that class value.  Regressors are
    handled by returning ``predict(X)`` directly.
    """
    if is_classifier(model):
        classes = model.classes_  # type: ignore[attr-defined]
        if len(classes) == 1:
            return np.full(len(X), float(classes[0]))
        return model.predict_proba(X)[:, 1]  # type: ignore[return-value]
    return model.predict(X)  # type: ignore[return-value]
