from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, is_classifier


def predict(model: BaseEstimator, X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unified predict for classifiers (predict_proba[:,1]) and regressors (.predict).

    For a classifier with only one class in its training data, returns a constant
    array equal to that class label.
    """
    if is_classifier(model):
        classes = model.classes_  # type: ignore[attr-defined]
        if len(classes) == 1:
            return np.full(len(X), float(classes[0]))
        return model.predict_proba(X)[:, 1]  # type: ignore[return-value]
    return model.predict(X)  # type: ignore[return-value]
