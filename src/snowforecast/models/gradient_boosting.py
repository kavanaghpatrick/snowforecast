"""Gradient boosting model for snowfall prediction.

Uses LightGBM as the primary implementation with XGBoost fallback.
Supports early stopping, categorical features, and feature importance.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from snowforecast.models.base import BaseModel


class GradientBoostingModel(BaseModel):
    """Gradient boosting model using LightGBM or XGBoost backend.

    Supports hyperparameter tuning, early stopping, categorical features,
    and feature importance extraction.

    Args:
        n_estimators: Number of boosting iterations. Default 1000.
        learning_rate: Boosting learning rate. Default 0.05.
        max_depth: Maximum tree depth. Default 6.
        num_leaves: Maximum number of leaves (LightGBM only). Default 31.
        min_child_samples: Minimum samples in leaf. Default 20.
        subsample: Row subsampling ratio. Default 0.8.
        colsample_bytree: Column subsampling ratio. Default 0.8.
        reg_alpha: L1 regularization. Default 0.0.
        reg_lambda: L2 regularization. Default 1.0.
        early_stopping_rounds: Rounds without improvement to stop. Default None.
        backend: Model backend ("lightgbm" or "xgboost"). Default "lightgbm".
        categorical_features: List of categorical feature names. Default None.
        random_state: Random seed. Default 42.
        verbose: Verbosity level (-1 for silent). Default -1.

    Attributes:
        is_fitted: Whether model has been trained.
        feature_names: List of training feature names.
        n_features: Number of features.
        best_iteration: Best iteration from early stopping, or None.

    Example:
        >>> model = GradientBoostingModel(n_estimators=100, learning_rate=0.1)
        >>> model.fit(X_train, y_train, eval_set=(X_val, y_val))
        >>> predictions = model.predict(X_test)
        >>> importance = model.get_feature_importance()
    """

    VALID_BACKENDS = ("lightgbm", "xgboost")

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int | None = None,
        backend: str = "lightgbm",
        categorical_features: list[str] | None = None,
        random_state: int = 42,
        verbose: int = -1,
    ):
        """Initialize the gradient boosting model."""
        super().__init__()

        if backend not in self.VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {self.VALID_BACKENDS}, got '{backend}'"
            )

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.backend = backend
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.verbose = verbose

        self._model = None
        self._best_iteration: int | None = None
        self._categorical_indices: list[int] | None = None
        self._n_features: int = 0

    @property
    def n_features(self) -> int:
        """Number of features in training data."""
        return self._n_features

    @property
    def best_iteration(self) -> int | None:
        """Best iteration number from early stopping, or None."""
        return self._best_iteration

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> "GradientBoostingModel":
        """Fit the gradient boosting model.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            eval_set: Optional (X_val, y_val) tuple for early stopping.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If input data is empty.
        """
        if X.empty:
            raise ValueError("Input data cannot be empty")

        # Store feature information
        self.feature_names = list(X.columns)
        self._n_features = len(X.columns)

        # Handle categorical features
        X_processed = self._preprocess_features(X)
        X_val_processed = None

        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_processed = self._preprocess_features(X_val)

        if self.backend == "lightgbm":
            self._fit_lightgbm(X_processed, y, X_val_processed, y_val if eval_set else None)
        else:
            self._fit_xgboost(X_processed, y, X_val_processed, y_val if eval_set else None)

        self.is_fitted = True
        return self

    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features including categorical encoding.

        Args:
            X: Input DataFrame.

        Returns:
            Processed DataFrame.
        """
        X_processed = X.copy()

        if self.categorical_features:
            # Store categorical indices for LightGBM
            self._categorical_indices = [
                X.columns.get_loc(col)
                for col in self.categorical_features
                if col in X.columns
            ]

            # Convert categorical columns to category dtype
            for col in self.categorical_features:
                if col in X_processed.columns:
                    X_processed[col] = X_processed[col].astype("category")

        return X_processed

    def _fit_lightgbm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
    ) -> None:
        """Fit using LightGBM backend."""
        import lightgbm as lgb

        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "force_col_wise": True,  # Avoid warning
        }

        callbacks = []
        if self.early_stopping_rounds and X_val is not None:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds)
            )
        if self.verbose < 0:
            callbacks.append(lgb.log_evaluation(period=0))  # Suppress logging

        self._model = lgb.LGBMRegressor(**params)

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["callbacks"] = callbacks

        self._model.fit(X, y, **fit_params)

        # Store best iteration
        if hasattr(self._model, "best_iteration_") and self._model.best_iteration_ > 0:
            self._best_iteration = self._model.best_iteration_

    def _fit_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
    ) -> None:
        """Fit using XGBoost backend."""
        import xgboost as xgb

        params = {
            "objective": "reg:squarederror",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "verbosity": 0 if self.verbose < 0 else 1,
            "enable_categorical": True if self.categorical_features else False,
        }

        # Add early stopping if specified
        if self.early_stopping_rounds and X_val is not None:
            params["early_stopping_rounds"] = self.early_stopping_rounds

        self._model = xgb.XGBRegressor(**params)

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        self._model.fit(X, y, **fit_params)

        # Store best iteration
        if hasattr(self._model, "best_iteration") and self._model.best_iteration > 0:
            self._best_iteration = self._model.best_iteration

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of predictions.

        Raises:
            ValueError: If model not fitted or features don't match.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        # Validate features match training
        if list(X.columns) != self.feature_names:
            raise ValueError(
                f"Feature mismatch. Expected {self.feature_names}, "
                f"got {list(X.columns)}"
            )

        X_processed = self._preprocess_features(X)
        return self._model.predict(X_processed)

    def get_feature_importance(
        self, importance_type: str = "gain"
    ) -> pd.DataFrame:
        """Get feature importance scores.

        Args:
            importance_type: Type of importance:
                - "gain": Total gain from splits using each feature.
                - "split": Number of times each feature is used in splits.

        Returns:
            DataFrame with columns ["feature", "importance"], sorted descending.

        Raises:
            ValueError: If model not fitted or invalid importance_type.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        valid_types = ("gain", "split")
        if importance_type not in valid_types:
            raise ValueError(
                f"importance_type must be one of {valid_types}, got '{importance_type}'"
            )

        if self.backend == "lightgbm":
            importance = self._model.booster_.feature_importance(
                importance_type=importance_type
            )
        else:
            # XGBoost uses different naming
            xgb_type = "gain" if importance_type == "gain" else "weight"
            importance = self._model.get_booster().get_score(
                importance_type=xgb_type
            )
            # XGBoost returns dict, convert to array in feature order
            importance = np.array([
                importance.get(f"f{i}", importance.get(name, 0))
                for i, name in enumerate(self.feature_names)
            ])

        # Create DataFrame
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        })

        # Sort descending by importance
        importance_df = importance_df.sort_values(
            "importance", ascending=False
        ).reset_index(drop=True)

        return importance_df

    def get_params(self) -> dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of parameter names and values.
        """
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "early_stopping_rounds": self.early_stopping_rounds,
            "backend": self.backend,
            "categorical_features": self.categorical_features,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> "GradientBoostingModel":
        """Set model parameters.

        Args:
            **params: Parameter names and values.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If invalid parameter name.
        """
        valid_params = set(self.get_params().keys())

        for param_name, value in params.items():
            if param_name not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{param_name}'. "
                    f"Valid parameters: {sorted(valid_params)}"
                )
            setattr(self, param_name, value)

        return self

    def save(self, path: Path | str) -> None:
        """Save model to disk.

        Creates parent directories if needed. Adds .pkl extension if missing.

        Args:
            path: Path to save model file.
        """
        path = Path(path)

        # Add .pkl extension if missing
        if path.suffix != ".pkl":
            path = path.with_suffix(".pkl")

        # Create parent directories
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        state = {
            "params": self.get_params(),
            "model": self._model,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "n_features": self._n_features,
            "best_iteration": self._best_iteration,
            "categorical_indices": self._categorical_indices,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path | str) -> "GradientBoostingModel":
        """Load model from disk.

        Args:
            path: Path to saved model file.

        Returns:
            Loaded model instance.

        Raises:
            FileNotFoundError: If path doesn't exist.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        # Create new instance with saved params
        model = cls(**state["params"])

        # Restore state
        model._model = state["model"]
        model.is_fitted = state["is_fitted"]
        model.feature_names = state["feature_names"]
        model._n_features = state["n_features"]
        model._best_iteration = state["best_iteration"]
        model._categorical_indices = state.get("categorical_indices")

        return model

    def __repr__(self) -> str:
        """String representation of the model."""
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return (
            f"GradientBoostingModel("
            f"backend={self.backend}, "
            f"n_estimators={self.n_estimators}, "
            f"learning_rate={self.learning_rate}, "
            f"{fitted_str}, "
            f"features={self._n_features})"
        )
