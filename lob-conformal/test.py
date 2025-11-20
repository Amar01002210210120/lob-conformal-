import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.conformal import SplitConformalRegressor

# Synthetic data
rng = np.random.RandomState(0)
n = 2000
X = rng.uniform(-1, 1, size=(n, 1))
y = 2 * X[:, 0] + rng.normal(0, 0.5, size=n)

# Split into train / calib / test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.5, random_state=0
)
X_calib, X_test, y_calib, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=0
)

# Run CP
cp = SplitConformalRegressor(alpha=0.1)
cp.fit(LinearRegression(), X_train, y_train, X_calib, y_calib)
lower, upper = cp.predict_interval(X_test)

coverage = ((y_test >= lower) & (y_test <= upper)).mean()
print("Empirical coverage:", coverage)
