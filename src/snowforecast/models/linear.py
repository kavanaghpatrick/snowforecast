"""Linear regression model for snowfall prediction baseline.

This module provides a simple linear regression model that serves as a baseline
for comparing more complex models. It uses scikit-learn's Ridge regression
with optional feature normalization.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from snowforecast.models.base import BaseModel


class LinearRegressionModel(BaseModel):
    """Linear regression baseline model for snowfall prediction.

    Uses Ridge regression (L2 regularization) with optional feature normalization
    via StandardScaler. This provides a simple baseline to compare against more
    complex models like gradient boosting or neural networks.

    Attributes:
        normalize: Whether to normalize features before fitting.
        scaler: StandardScaler instance (if normalize=True).
        model: The underlying sklearn Ridge regressor.
        feature_names: List of feature names from training.
        is_fitted: Whether the model has been fitted.

    Example:
        >>> model = LinearRegressionModel(normalize=True)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> importance = model.get_feature_importance()
    """

    def __init__(self, normalize: bool = True):
        """Initialize the linear regression model.

        Args:
            normalize: Whether to normalize features using StandardScaler.
                Default is True, which typically improves performance when
                features have different scales.
        """
        super().__init__()
        self.normalize = normalize
        self.scaler: StandardScaler | None = None

    def __repr__(self) -> str:
        """Return string representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        n_features = len(self.feature_names) if self.feature_names else 0
        return f"LinearRegressionModel({status}, features={n_features}, normalize={self.normalize})"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> "LinearRegressionModel":
        """Fit the linear regression model to training data.

        Args:
            X: Training features as a pandas DataFrame.
            y: Training targets as a pandas Series.
            eval_set: Optional validation set (X_val, y_val).
                Ignored for linear regression but accepted for API compatibility.

        Returns:
            self: The fitted model instance for method chaining.

        Raises:
            ValueError: If X or y have invalid format, contain NaN values,
                or have mismatched lengths.
        """
        # Validate inputs
        self._validate_X(X)
        self._validate_y(y)

        # Check length match
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length, got {len(X)} and {len(y)}"
            )

        # Check for NaN values
        if X.isna().any().any():
            raise ValueError("X contains NaN values")
        if y.isna().any():
            raise ValueError("y contains NaN values")

        # Store feature names
        self.feature_names = list(X.columns)

        # Prepare features
        X_values = X.values

        # Normalize if requested
        if self.normalize:
            self.scaler = StandardScaler()
            X_values = self.scaler.fit_transform(X_values)
        else:
            self.scaler = None

        # Fit Ridge regression
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_values, y.values)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Features as a pandas DataFrame with same columns as training data.

        Returns:
            Predicted values as a numpy array.

        Raises:
            ValueError: If model is not fitted, X has wrong features,
                or X contains NaN values.
        """
        self._validate_fitted()
        self._validate_X(X)

        # Check feature match
        if list(X.columns) != self.feature_names:
            raise ValueError(
                f"Feature mismatch. Expected {self.feature_names}, got {list(X.columns)}"
            )

        # Check for NaN values
        if X.isna().any().any():
            raise ValueError("X contains NaN values")

        # Prepare features
        X_values = X.values

        # Apply same normalization as training
        if self.normalize and self.scaler is not None:
            X_values = self.scaler.transform(X_values)

        return self.model.predict(X_values)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance based on absolute coefficient values.

        For linear models, feature importance is derived from the absolute
        value of the coefficients. Higher absolute coefficient means more
        influence on predictions.

        Returns:
            DataFrame with columns:
                - feature: Feature name
                - importance: Absolute coefficient value
                - coefficient: Raw coefficient value (can be negative)
            Sorted by importance descending.

        Raises:
            ValueError: If model is not fitted.
        """
        self._validate_fitted()

        coefficients = self.model.coef_
        abs_coefficients = np.abs(coefficients)

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": abs_coefficients,
            "coefficient": coefficients,
        })

        # Sort by importance descending
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)

        return df

    def get_coefficients(self) -> pd.DataFrame:
        """Get the model coefficients.

        Returns:
            DataFrame with columns:
                - feature: Feature name
                - coefficient: Coefficient value

        Raises:
            ValueError: If model is not fitted.
        """
        self._validate_fitted()

        return pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": self.model.coef_,
        })

    def get_intercept(self) -> float:
        """Get the model intercept.

        Returns:
            The intercept (bias) term.

        Raises:
            ValueError: If model is not fitted.
        """
        self._validate_fitted()
        return float(self.model.intercept_)

    def get_params(self) -> dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary with:
                - normalize: Whether normalization is enabled
                - n_features: Number of features (if fitted)
                - intercept: Model intercept (if fitted)
        """
        params = {"normalize": self.normalize}

        if self.is_fitted:
            params["n_features"] = len(self.feature_names)
            params["intercept"] = float(self.model.intercept_)

        return params

    def set_params(self, **params: Any) -> "LinearRegressionModel":
        """Set model parameters.

        Args:
            **params: Parameters to set. Currently supports:
                - normalize: bool

        Returns:
            self: The model instance for method chaining.

        Raises:
            ValueError: If trying to change parameters on a fitted model.
        """
        if self.is_fitted:
            raise ValueError(
                "Cannot change parameters on a fitted model. "
                "Create a new model instance instead."
            )

        if "normalize" in params:
            self.normalize = params["normalize"]

        return self

    def save(self, path: str | Path) -> None:
        """Save the model to disk using pickle.

        Args:
            path: Path to save the model file. Parent directories will
                be created if they don't exist.

        Raises:
            ValueError: If model is not fitted.
        """
        self._validate_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save all necessary state
        state = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "normalize": self.normalize,
            "is_fitted": self.is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "LinearRegressionModel":
        """Load a model from disk.

        Args:
            path: Path to the saved model file.

        Returns:
            Loaded model instance.

        Raises:
            FileNotFoundError: If model file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        # Create new instance and restore state
        instance = cls(normalize=state["normalize"])
        instance.model = state["model"]
        instance.scaler = state["scaler"]
        instance.feature_names = state["feature_names"]
        instance.is_fitted = state["is_fitted"]

        return instance
