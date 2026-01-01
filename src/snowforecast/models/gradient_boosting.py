"""Gradient Boosting models for snowfall prediction.

This module provides gradient boosting implementations using LightGBM and XGBoost
backends for snowfall prediction.

Classes:
- GradientBoostingModel: Unified interface for LightGBM/XGBoost models
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from snowforecast.models.base import BaseModel

logger = logging.getLogger(__name__)


class GradientBoostingModel(BaseModel):
    """Gradient Boosting model for snowfall prediction.

    Supports both LightGBM and XGBoost backends with a unified interface.

    Attributes:
        n_estimators: Number of boosting rounds.
        learning_rate: Learning rate for boosting.
        max_depth: Maximum tree depth.
        num_leaves: Number of leaves (LightGBM only).
        min_child_samples: Minimum samples in leaf.
        subsample: Row subsampling ratio.
        colsample_bytree: Column subsampling ratio.
        reg_alpha: L1 regularization.
        reg_lambda: L2 regularization.
        backend: "lightgbm" or "xgboost".
        categorical_features: List of categorical feature names.
        early_stopping_rounds: Rounds for early stopping (with eval_set).
        verbose: Verbosity level (-1 for silent).
        best_iteration: Best iteration from training (if early stopping used).

    Example:
        >>> model = GradientBoostingModel(
        ...     n_estimators=1000,
        ...     learning_rate=0.05,
        ...     max_depth=6,
        ...     backend="lightgbm",
        ... )
        >>> model.fit(X_train, y_train, eval_set=(X_val, y_val))
        >>> predictions = model.predict(X_test)
    """

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
        backend: Literal["lightgbm", "xgboost"] = "lightgbm",
        categorical_features: list[str] | None = None,
        early_stopping_rounds: int | None = 50,
        verbose: int = -1,
        random_state: int = 42,
    ):
        """Initialize GradientBoostingModel.

        Args:
            n_estimators: Number of boosting iterations.
            learning_rate: Step size shrinkage.
            max_depth: Maximum depth of a tree.
            num_leaves: Number of leaves in a tree (LightGBM only).
            min_child_samples: Minimum number of data needed in a leaf.
            subsample: Subsample ratio of the training instance.
            colsample_bytree: Subsample ratio of columns when constructing each tree.
            reg_alpha: L1 regularization term on weights.
            reg_lambda: L2 regularization term on weights.
            backend: Which library to use ("lightgbm" or "xgboost").
            categorical_features: List of categorical column names.
            early_stopping_rounds: Activates early stopping if eval_set provided.
            verbose: Verbosity level (-1 for silent).
            random_state: Random seed for reproducibility.

        Raises:
            ValueError: If backend is not "lightgbm" or "xgboost".
            ImportError: If the selected backend is not installed.
        """
        super().__init__()

        if backend not in ("lightgbm", "xgboost"):
            raise ValueError(f"backend must be 'lightgbm' or 'xgboost', got {backend}")

        if backend == "lightgbm" and not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")

        if backend == "xgboost" and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.backend = backend
        self.categorical_features = categorical_features or []
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.random_state = random_state
        self.best_iteration: int | None = None
        self._n_features: int = 0

    @property
    def n_features(self) -> int:
        """Number of features used in training."""
        return self._n_features

    def _get_lgb_params(self) -> dict[str, Any]:
        """Get LightGBM parameters."""
        return {
            "objective": "regression",
            "metric": "rmse",
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
            "verbosity": self.verbose,
        }

    def _get_xgb_params(self) -> dict[str, Any]:
        """Get XGBoost parameters."""
        return {
            "objective": "reg:squarederror",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "verbosity": 0 if self.verbose < 0 else self.verbose,
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> "GradientBoostingModel":
        """Fit the model to training data.

        Args:
            X: Training features as DataFrame.
            y: Training targets as Series.
            eval_set: Optional validation set (X_val, y_val) for early stopping.

        Returns:
            self for method chaining.
        """
        self._validate_X(X)
        self._validate_y(y)

        self.feature_names = list(X.columns)
        self._n_features = len(self.feature_names)

        # Prepare categorical features
        X_train = X.copy()
        if self.categorical_features:
            for col in self.categorical_features:
                if col in X_train.columns:
                    X_train[col] = X_train[col].astype("category")

        if self.backend == "lightgbm":
            self._fit_lgb(X_train, y, eval_set)
        else:
            self._fit_xgb(X_train, y, eval_set)

        self.is_fitted = True
        return self

    def _fit_lgb(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None,
    ) -> None:
        """Fit using LightGBM."""
        params = self._get_lgb_params()

        callbacks = []
        if self.early_stopping_rounds and eval_set:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        fit_kwargs = {
            "X": X,
            "y": y,
            "categorical_feature": self.categorical_features if self.categorical_features else "auto",
        }

        if eval_set:
            X_val, y_val = eval_set
            X_val = X_val.copy()
            if self.categorical_features:
                for col in self.categorical_features:
                    if col in X_val.columns:
                        X_val[col] = X_val[col].astype("category")
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = callbacks

        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(**fit_kwargs)

        if hasattr(self.model, "best_iteration_"):
            self.best_iteration = self.model.best_iteration_

    def _fit_xgb(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None,
    ) -> None:
        """Fit using XGBoost."""
        params = self._get_xgb_params()

        if self.categorical_features:
            params["enable_categorical"] = True

        fit_kwargs: dict[str, Any] = {}
        if eval_set:
            X_val, y_val = eval_set
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            if self.early_stopping_rounds:
                params["early_stopping_rounds"] = self.early_stopping_rounds

        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X, y, **fit_kwargs, verbose=False)

        if hasattr(self.model, "best_iteration"):
            self.best_iteration = self.model.best_iteration

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Features as DataFrame.

        Returns:
            Predicted values as numpy array.
        """
        self._validate_fitted()
        self._validate_X(X)

        if set(X.columns) != set(self.feature_names):
            raise ValueError(
                f"Feature mismatch. Expected {set(self.feature_names)}, "
                f"got {set(X.columns)}"
            )

        # Prepare categorical features
        X_pred = X.copy()
        if self.categorical_features:
            for col in self.categorical_features:
                if col in X_pred.columns:
                    X_pred[col] = X_pred[col].astype("category")

        # Reorder columns to match training order
        X_pred = X_pred[self.feature_names]

        return self.model.predict(X_pred)

    def get_feature_importance(
        self,
        importance_type: Literal["gain", "split"] = "gain",
    ) -> pd.DataFrame:
        """Get feature importance scores.

        Args:
            importance_type: Type of importance ("gain" or "split").

        Returns:
            DataFrame with feature and importance columns.
        """
        self._validate_fitted()

        if importance_type not in ("gain", "split"):
            raise ValueError(f"importance_type must be 'gain' or 'split', got {importance_type}")

        if self.backend == "lightgbm":
            imp_type = importance_type
            importance = self.model.booster_.feature_importance(importance_type=imp_type)
        else:
            # XGBoost uses different names
            imp_type = "weight" if importance_type == "split" else "gain"
            importance_dict = self.model.get_booster().get_score(importance_type=imp_type)
            importance = [importance_dict.get(f, 0) for f in self.feature_names]

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
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
            "backend": self.backend,
            "categorical_features": self.categorical_features,
            "early_stopping_rounds": self.early_stopping_rounds,
            "verbose": self.verbose,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "GradientBoostingModel":
        """Set model parameters.

        Args:
            **params: Parameters to set.

        Returns:
            self for method chaining.
        """
        valid_params = set(self.get_params().keys())

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter: {key}. "
                    f"Valid parameters: {valid_params}"
                )
            setattr(self, key, value)

        return self

    def save(self, path: str | Path) -> None:
        """Save the model to disk.

        Args:
            path: File path to save to.
        """
        self._validate_fitted()

        path = Path(path)
        if path.suffix != ".pkl":
            path = path.with_suffix(".pkl")

        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "model": self.model,
            "params": self.get_params(),
            "feature_names": self.feature_names,
            "n_features": self._n_features,
            "best_iteration": self.best_iteration,
        }

        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path: str | Path) -> "GradientBoostingModel":
        """Load a model from disk.

        Args:
            path: File path to load from.

        Returns:
            Loaded model instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        # Create new instance with saved params
        params = save_dict["params"]
        model = cls(**params)

        # Restore state
        model.model = save_dict["model"]
        model.feature_names = save_dict["feature_names"]
        model._n_features = save_dict["n_features"]
        model.best_iteration = save_dict["best_iteration"]
        model.is_fitted = True

        return model

    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"GradientBoostingModel("
            f"{self.backend}, "
            f"n_estimators={self.n_estimators}, "
            f"{status}, "
            f"features={self.n_features})"
        )
