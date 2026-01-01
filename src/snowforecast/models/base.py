"""Base model class for snowfall prediction.

This module provides the abstract base class that all prediction models must inherit from.
It defines the standard interface for training, prediction, and model persistence.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all snowfall prediction models.

    This class defines the standard interface that all models must implement.
    It provides validation methods and common functionality for model management.

    Attributes:
        model: The underlying model object (set after fitting).
        feature_names: List of feature names used during training.
        is_fitted: Whether the model has been fitted.

    Example:
        >>> class MyModel(BaseModel):
        ...     def fit(self, X, y, eval_set=None):
        ...         # Implement fitting logic
        ...         pass
        ...     def predict(self, X):
        ...         # Implement prediction logic
        ...         pass
        ...     # ... implement other abstract methods
    """

    def __init__(self):
        """Initialize the base model with default state."""
        self.model: Any | None = None
        self.feature_names: list[str] | None = None
        self.is_fitted: bool = False

    def __repr__(self) -> str:
        """Return string representation of the model."""
        class_name = self.__class__.__name__
        status = "fitted" if self.is_fitted else "not fitted"
        n_features = len(self.feature_names) if self.feature_names else 0
        return f"{class_name}({status}, features={n_features})"

    # =========================================================================
    # Validation methods
    # =========================================================================

    def _validate_X(self, X: Any) -> None:
        """Validate input features.

        Args:
            X: Input features to validate.

        Raises:
            ValueError: If X is not a pandas DataFrame or is empty.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                f"X must be a pandas DataFrame, got {type(X).__name__}"
            )
        if X.empty:
            raise ValueError("X cannot be empty")

    def _validate_y(self, y: Any) -> None:
        """Validate target values.

        Args:
            y: Target values to validate.

        Raises:
            ValueError: If y is not a pandas Series or is empty.
        """
        if not isinstance(y, pd.Series):
            raise ValueError(
                f"y must be a pandas Series, got {type(y).__name__}"
            )
        if len(y) == 0:
            raise ValueError("y cannot be empty")

    def _validate_fitted(self) -> None:
        """Validate that the model has been fitted.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

    # =========================================================================
    # Abstract methods to be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None
    ) -> "BaseModel":
        """Fit the model to training data.

        Args:
            X: Training features as a pandas DataFrame.
            y: Training targets as a pandas Series.
            eval_set: Optional validation set as (X_val, y_val) tuple.
                Used for early stopping in models that support it.

        Returns:
            self: The fitted model instance for method chaining.

        Raises:
            ValueError: If X or y have invalid format.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Features as a pandas DataFrame with same columns as training data.

        Returns:
            Predicted values as a numpy array.

        Raises:
            ValueError: If model is not fitted or X has wrong features.
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores.

        Returns:
            DataFrame with columns:
                - feature: Feature name
                - importance: Importance score (higher = more important)

        Raises:
            ValueError: If model is not fitted.
        """
        pass

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of model parameters.
        """
        pass

    @abstractmethod
    def set_params(self, **params: Any) -> "BaseModel":
        """Set model parameters.

        Args:
            **params: Parameters to set.

        Returns:
            self: The model instance for method chaining.

        Raises:
            ValueError: If trying to set params on a fitted model.
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save the model to disk.

        Args:
            path: Path to save the model file.

        Raises:
            ValueError: If model is not fitted.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseModel":
        """Load a model from disk.

        Args:
            path: Path to the model file.

        Returns:
            Loaded model instance.

        Raises:
            FileNotFoundError: If model file does not exist.
        """
        pass
