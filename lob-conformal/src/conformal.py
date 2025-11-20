import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SplitConformalRegressor:
    """
    Simple split conformal for regression.
    Assumes model has fit(X, y) and predict(X) methods.
    """
    alpha: float = 0.1
    qhat_: float | None = None

    def fit(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
    ):
        # Fit base model
        model.fit(X_train, y_train)
        self.model_ = model

        # Compute residuals on calibration set
        y_calib_pred = self.model_.predict(X_calib)
        scores = np.abs(y_calib - y_calib_pred)

        n = len(scores)
        k = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        k = min(max(k, 0), n - 1)
        self.qhat_ = np.sort(scores)[k]

        return self

    def predict_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.qhat_ is None:
            raise ValueError("Call fit() before predict_interval().")

        y_pred = self.model_.predict(X)
        lower = y_pred - self.qhat_
        upper = y_pred + self.qhat_
        return lower, upper
