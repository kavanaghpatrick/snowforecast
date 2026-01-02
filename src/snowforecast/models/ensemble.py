"""Ensemble methods for snowfall prediction.

This module provides ensemble methods that combine multiple models
to improve prediction accuracy and robustness.

Classes:
- SimpleEnsemble: Combines predictions via averaging or weighted averaging
- StackingEnsemble: Two-level stacking with a meta-learner

Functions:
- create_ensemble: Factory function to create ensembles
- get_model_weights: Extract learned weights from an ensemble

Example:
    >>> from snowforecast.models import LinearRegressionModel, GradientBoostingModel
    >>> from snowforecast.models.ensemble import SimpleEnsemble, create_ensemble
    >>>
    >>> # Create and fit base models
    >>> model1 = LinearRegressionModel().fit(X_train, y_train)
    >>> model2 = GradientBoostingModel(n_estimators=100).fit(X_train, y_train)
    >>>
    >>> # Create ensemble
    >>> ensemble = SimpleEnsemble(models=[model1, model2])
    >>> predictions = ensemble.predict(X_test)
"""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from snowforecast.models.base import BaseModel

logger = logging.getLogger(__name__)


class SimpleEnsemble(BaseModel):
    """Simple ensemble that combines predictions via averaging.

    Takes a list of fitted models and combines their predictions using
    simple averaging or weighted averaging. This is the simplest form
    of ensemble and often provides good results with minimal overhead.

    Attributes:
        models: List of fitted BaseModel instances.
        weights: Optional weights for each model (must sum to 1).
        aggregation: Method used for combining predictions ("mean" or "median").
        is_fitted: Whether the ensemble has been initialized with models.

    Example:
        >>> from snowforecast.models import LinearRegressionModel
        >>> from snowforecast.models.ensemble import SimpleEnsemble
        >>>
        >>> # Fit base models
        >>> model1 = LinearRegressionModel().fit(X_train, y_train)
        >>> model2 = LinearRegressionModel(normalize=False).fit(X_train, y_train)
        >>>
        >>> # Simple averaging
        >>> ensemble = SimpleEnsemble(models=[model1, model2])
        >>> predictions = ensemble.predict(X_test)
        >>>
        >>> # Weighted averaging
        >>> ensemble_weighted = SimpleEnsemble(
        ...     models=[model1, model2],
        ...     weights=[0.7, 0.3]
        ... )
    """

    def __init__(
        self,
        models: list[BaseModel] | None = None,
        weights: list[float] | np.ndarray | None = None,
        aggregation: Literal["mean", "median"] = "mean",
    ):
        """Initialize the SimpleEnsemble.

        Args:
            models: List of fitted BaseModel instances. If provided, the
                ensemble is immediately ready to predict. If None, models
                must be provided via fit().
            weights: Optional weights for each model. If None, equal weights
                are used. Must sum to 1.0 (will be normalized if not).
            aggregation: Method for combining predictions. "mean" uses
                weighted average, "median" uses median (ignores weights).

        Raises:
            ValueError: If weights length doesn't match models length.
        """
        super().__init__()

        self.aggregation = aggregation
        self._models: list[BaseModel] = []
        self._weights: np.ndarray | None = None

        if models is not None:
            self._set_models(models, weights)

    def _set_models(
        self,
        models: list[BaseModel],
        weights: list[float] | np.ndarray | None = None,
    ) -> None:
        """Set the models and weights for the ensemble.

        Args:
            models: List of fitted BaseModel instances.
            weights: Optional weights for each model.

        Raises:
            ValueError: If weights length doesn't match models, or if
                any model is not fitted.
        """
        if not models:
            raise ValueError("At least one model is required")

        # Validate all models are fitted
        for i, model in enumerate(models):
            if not model.is_fitted:
                raise ValueError(
                    f"Model at index {i} ({model.__class__.__name__}) is not fitted. "
                    "All models must be fitted before creating an ensemble."
                )

        self._models = list(models)

        # Set up weights
        if weights is not None:
            weights = np.array(weights, dtype=float)
            if len(weights) != len(models):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of models ({len(models)})"
                )
            # Normalize weights to sum to 1
            weights = weights / weights.sum()
            self._weights = weights
        else:
            # Equal weights
            self._weights = np.ones(len(models)) / len(models)

        # Use feature names from first model
        self.feature_names = list(models[0].feature_names)
        self.is_fitted = True

    @property
    def models(self) -> list[BaseModel]:
        """List of models in the ensemble."""
        return self._models

    @property
    def weights(self) -> np.ndarray | None:
        """Weights for each model."""
        return self._weights.copy() if self._weights is not None else None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> "SimpleEnsemble":
        """Fit the ensemble (no-op if models already provided).

        For SimpleEnsemble, fit() is primarily used when you want to
        set models via keyword arguments or if models weren't provided
        at initialization. This method does NOT train the individual
        models - they must already be fitted.

        Args:
            X: Training features (used to validate feature names match).
            y: Training targets (not used, for API compatibility).
            eval_set: Validation set (not used, for API compatibility).

        Returns:
            self for method chaining.

        Raises:
            ValueError: If no models have been set.
        """
        self._validate_X(X)
        self._validate_y(y)

        if not self._models:
            raise ValueError(
                "No models provided. Either pass models to __init__ or "
                "use set_models() before calling fit()."
            )

        # Validate feature names match
        expected_features = set(self.feature_names)
        actual_features = set(X.columns)
        if expected_features != actual_features:
            raise ValueError(
                f"Feature mismatch. Ensemble expects {sorted(expected_features)}, "
                f"got {sorted(actual_features)}"
            )

        return self

    def set_models(
        self,
        models: list[BaseModel],
        weights: list[float] | np.ndarray | None = None,
    ) -> "SimpleEnsemble":
        """Set or replace the models in the ensemble.

        Args:
            models: List of fitted BaseModel instances.
            weights: Optional weights for each model.

        Returns:
            self for method chaining.
        """
        self._set_models(models, weights)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions by combining all model predictions.

        Args:
            X: Features as DataFrame with same columns as training data.

        Returns:
            Combined predictions as numpy array.

        Raises:
            ValueError: If ensemble is not fitted or features don't match.
        """
        self._validate_fitted()
        self._validate_X(X)

        # Collect predictions from all models, handling failures gracefully
        all_predictions = []
        valid_weights = []

        for i, model in enumerate(self._models):
            try:
                pred = model.predict(X)
                # Check for NaN/inf in predictions
                if np.isfinite(pred).all():
                    all_predictions.append(pred)
                    valid_weights.append(self._weights[i])
                else:
                    logger.warning(
                        f"Model {i} ({model.__class__.__name__}) returned "
                        "non-finite predictions, excluding from ensemble"
                    )
            except Exception as e:
                logger.warning(
                    f"Model {i} ({model.__class__.__name__}) failed to predict: {e}. "
                    "Excluding from ensemble."
                )

        if not all_predictions:
            raise ValueError(
                "All models failed to make valid predictions. "
                "Cannot compute ensemble prediction."
            )

        # Stack predictions: shape (n_models, n_samples)
        predictions_array = np.vstack(all_predictions)

        # Combine predictions based on aggregation method
        if self.aggregation == "median":
            # Median ignores weights
            combined = np.median(predictions_array, axis=0)
        else:
            # Weighted mean
            valid_weights = np.array(valid_weights)
            valid_weights = valid_weights / valid_weights.sum()  # Renormalize
            combined = np.average(predictions_array, axis=0, weights=valid_weights)

        return combined

    def get_feature_importance(self) -> pd.DataFrame:
        """Get averaged feature importance across all models.

        Combines feature importance from all models using the ensemble
        weights, then normalizes to sum to 1.

        Returns:
            DataFrame with 'feature' and 'importance' columns.

        Raises:
            ValueError: If ensemble is not fitted.
        """
        self._validate_fitted()

        # Collect importance from all models
        all_importance = {}

        for i, model in enumerate(self._models):
            try:
                importance_df = model.get_feature_importance()
                weight = self._weights[i]

                for _, row in importance_df.iterrows():
                    feature = row["feature"]
                    imp = row["importance"]

                    if feature not in all_importance:
                        all_importance[feature] = 0.0
                    all_importance[feature] += weight * imp

            except Exception as e:
                logger.warning(
                    f"Model {i} ({model.__class__.__name__}) failed to get "
                    f"feature importance: {e}"
                )

        if not all_importance:
            # Return equal importance if all failed
            n_features = len(self.feature_names)
            return pd.DataFrame({
                "feature": self.feature_names,
                "importance": [1.0 / n_features] * n_features,
            })

        # Create DataFrame and normalize
        result = pd.DataFrame([
            {"feature": f, "importance": imp}
            for f, imp in all_importance.items()
        ])

        # Normalize to sum to 1
        result["importance"] = result["importance"] / result["importance"].sum()

        return result.sort_values("importance", ascending=False).reset_index(drop=True)

    def get_params(self) -> dict[str, Any]:
        """Get ensemble parameters.

        Returns:
            Dictionary with ensemble configuration.
        """
        return {
            "n_models": len(self._models),
            "aggregation": self.aggregation,
            "weights": self._weights.tolist() if self._weights is not None else None,
            "model_types": [m.__class__.__name__ for m in self._models],
        }

    def set_params(self, **params: Any) -> "SimpleEnsemble":
        """Set ensemble parameters.

        Args:
            **params: Parameters to set. Supports:
                - aggregation: str ("mean" or "median")
                - weights: list[float] or np.ndarray

        Returns:
            self for method chaining.
        """
        if "aggregation" in params:
            if params["aggregation"] not in ("mean", "median"):
                raise ValueError(
                    f"aggregation must be 'mean' or 'median', "
                    f"got '{params['aggregation']}'"
                )
            self.aggregation = params["aggregation"]

        if "weights" in params:
            weights = np.array(params["weights"], dtype=float)
            if len(self._models) > 0 and len(weights) != len(self._models):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of models ({len(self._models)})"
                )
            self._weights = weights / weights.sum()

        return self

    def save(self, path: str | Path) -> None:
        """Save the ensemble to disk.

        Saves ensemble configuration and all individual models.

        Args:
            path: Path to save the ensemble file.

        Raises:
            ValueError: If ensemble is not fitted.
        """
        self._validate_fitted()

        path = Path(path)
        if path.suffix != ".pkl":
            path = path.with_suffix(".pkl")

        path.parent.mkdir(parents=True, exist_ok=True)

        # Save state including serialized models
        state = {
            "models": self._models,
            "weights": self._weights,
            "aggregation": self.aggregation,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "SimpleEnsemble":
        """Load an ensemble from disk.

        Args:
            path: Path to the saved ensemble file.

        Returns:
            Loaded SimpleEnsemble instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Ensemble file not found: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        # Create instance and restore state
        instance = cls(aggregation=state["aggregation"])
        instance._models = state["models"]
        instance._weights = state["weights"]
        instance.feature_names = state["feature_names"]
        instance.is_fitted = state["is_fitted"]

        return instance

    def __repr__(self) -> str:
        """String representation of the ensemble."""
        status = "fitted" if self.is_fitted else "not fitted"
        n_models = len(self._models)
        return f"SimpleEnsemble(n_models={n_models}, aggregation={self.aggregation}, {status})"


class StackingEnsemble(BaseModel):
    """Two-level stacking ensemble with a meta-learner.

    Level 0: Base models generate predictions
    Level 1: Meta-learner combines base predictions into final output

    The meta-learner is trained on the predictions of the base models,
    learning optimal weights for combining them.

    Attributes:
        base_models: List of fitted base models (level 0).
        meta_learner: Model that combines base predictions (level 1).
        use_features: Whether to include original features in meta-learner input.
        cv_folds: Number of cross-validation folds for generating meta-features.

    Example:
        >>> from snowforecast.models import LinearRegressionModel, GradientBoostingModel
        >>> from snowforecast.models.ensemble import StackingEnsemble
        >>>
        >>> # Create base models (will be fitted during stacking)
        >>> base_models = [
        ...     LinearRegressionModel(),
        ...     GradientBoostingModel(n_estimators=100),
        ... ]
        >>>
        >>> # Create stacking ensemble with linear meta-learner
        >>> ensemble = StackingEnsemble(
        ...     base_models=base_models,
        ...     meta_learner=LinearRegressionModel(),
        ... )
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
    """

    def __init__(
        self,
        base_models: list[BaseModel] | None = None,
        meta_learner: BaseModel | None = None,
        use_features: bool = False,
        cv_folds: int = 5,
    ):
        """Initialize the StackingEnsemble.

        Args:
            base_models: List of base models to use. They can be unfitted
                (will be fitted during fit()) or pre-fitted.
            meta_learner: Model to use as meta-learner. If None, uses
                LinearRegressionModel. Will be fitted during fit().
            use_features: If True, original features are passed to the
                meta-learner along with base model predictions.
            cv_folds: Number of cross-validation folds for generating
                out-of-fold predictions for meta-learner training.

        Raises:
            ValueError: If cv_folds < 2.
        """
        super().__init__()

        if cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {cv_folds}")

        self._base_models: list[BaseModel] = base_models or []
        self._fitted_base_models: list[BaseModel] = []
        self.use_features = use_features
        self.cv_folds = cv_folds

        # Default meta-learner is linear regression
        if meta_learner is None:
            from snowforecast.models.linear import LinearRegressionModel
            self._meta_learner = LinearRegressionModel(normalize=True)
        else:
            self._meta_learner = meta_learner

    @property
    def base_models(self) -> list[BaseModel]:
        """List of fitted base models."""
        return self._fitted_base_models

    @property
    def meta_learner(self) -> BaseModel:
        """The meta-learner model."""
        return self._meta_learner

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> "StackingEnsemble":
        """Fit the stacking ensemble.

        1. Generates out-of-fold predictions from base models using CV
        2. Trains final base models on full training data
        3. Trains meta-learner on out-of-fold predictions

        Args:
            X: Training features as DataFrame.
            y: Training targets as Series.
            eval_set: Optional validation set (passed to base models).

        Returns:
            self for method chaining.

        Raises:
            ValueError: If no base models provided or data is invalid.
        """
        self._validate_X(X)
        self._validate_y(y)

        if not self._base_models:
            raise ValueError(
                "No base models provided. Set base_models in __init__ or "
                "use set_base_models() before calling fit()."
            )

        self.feature_names = list(X.columns)
        n_samples = len(X)

        # Generate out-of-fold predictions for meta-learner training
        oof_predictions = np.zeros((n_samples, len(self._base_models)))

        # Create fold indices
        fold_size = n_samples // self.cv_folds
        indices = np.arange(n_samples)

        for fold_idx in range(self.cv_folds):
            # Define fold boundaries
            val_start = fold_idx * fold_size
            val_end = val_start + fold_size if fold_idx < self.cv_folds - 1 else n_samples

            # Create masks
            val_mask = np.zeros(n_samples, dtype=bool)
            val_mask[val_start:val_end] = True
            train_mask = ~val_mask

            X_fold_train = X.iloc[train_mask]
            y_fold_train = y.iloc[train_mask]
            X_fold_val = X.iloc[val_mask]

            # Get predictions from each base model
            for model_idx, model in enumerate(self._base_models):
                try:
                    # Clone model for this fold (create new instance)
                    fold_model = model.__class__(**model.get_params())
                    fold_model.fit(X_fold_train, y_fold_train)
                    pred = fold_model.predict(X_fold_val)
                    oof_predictions[val_mask, model_idx] = pred
                except Exception as e:
                    logger.warning(
                        f"Base model {model_idx} ({model.__class__.__name__}) "
                        f"failed in fold {fold_idx}: {e}. Using NaN."
                    )
                    oof_predictions[val_mask, model_idx] = np.nan

        # Train final base models on full training data
        self._fitted_base_models = []
        for model_idx, model in enumerate(self._base_models):
            try:
                # Create fresh model instance
                fitted_model = model.__class__(**model.get_params())
                fitted_model.fit(X, y, eval_set=eval_set)
                self._fitted_base_models.append(fitted_model)
            except Exception as e:
                logger.warning(
                    f"Base model {model_idx} ({model.__class__.__name__}) "
                    f"failed to fit: {e}. Excluding from ensemble."
                )

        if not self._fitted_base_models:
            raise ValueError("All base models failed to fit. Cannot create ensemble.")

        # Prepare meta-features
        # Only keep columns for models that successfully fitted
        valid_model_indices = [
            i for i, model in enumerate(self._base_models)
            if any(m.__class__ == model.__class__ for m in self._fitted_base_models)
        ]
        # Simple approach: keep first N columns where N = number of fitted models
        oof_predictions = oof_predictions[:, :len(self._fitted_base_models)]

        # Handle any NaN values in out-of-fold predictions
        # Replace NaN with column mean
        for col in range(oof_predictions.shape[1]):
            col_mean = np.nanmean(oof_predictions[:, col])
            nan_mask = np.isnan(oof_predictions[:, col])
            if nan_mask.any():
                oof_predictions[nan_mask, col] = col_mean
                logger.warning(
                    f"Replaced {nan_mask.sum()} NaN values in model {col} "
                    "predictions with column mean"
                )

        # Create meta-feature DataFrame
        meta_feature_names = [f"model_{i}_pred" for i in range(len(self._fitted_base_models))]
        meta_features = pd.DataFrame(oof_predictions, columns=meta_feature_names)

        if self.use_features:
            # Include original features (reset index to align)
            X_reset = X.reset_index(drop=True)
            meta_features = pd.concat([meta_features, X_reset], axis=1)

        # Train meta-learner
        y_reset = y.reset_index(drop=True)
        self._meta_learner.fit(meta_features, y_reset)

        self.is_fitted = True
        return self

    def set_base_models(self, models: list[BaseModel]) -> "StackingEnsemble":
        """Set or replace the base models.

        Args:
            models: List of BaseModel instances (fitted or unfitted).

        Returns:
            self for method chaining.
        """
        self._base_models = list(models)
        self._fitted_base_models = []
        self.is_fitted = False
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the stacking ensemble.

        1. Get predictions from all base models
        2. Pass to meta-learner for final prediction

        Args:
            X: Features as DataFrame.

        Returns:
            Predicted values as numpy array.

        Raises:
            ValueError: If ensemble is not fitted or features don't match.
        """
        self._validate_fitted()
        self._validate_X(X)

        # Get predictions from all fitted base models
        base_predictions = []
        for i, model in enumerate(self._fitted_base_models):
            try:
                pred = model.predict(X)
                if np.isfinite(pred).all():
                    base_predictions.append(pred)
                else:
                    # Use mean of other predictions for this model
                    logger.warning(
                        f"Model {i} ({model.__class__.__name__}) returned "
                        "non-finite predictions"
                    )
                    base_predictions.append(np.full(len(X), np.nan))
            except Exception as e:
                logger.warning(
                    f"Model {i} ({model.__class__.__name__}) failed to predict: {e}"
                )
                base_predictions.append(np.full(len(X), np.nan))

        # Stack predictions
        predictions_array = np.column_stack(base_predictions)

        # Handle NaN values - replace with column mean
        for col in range(predictions_array.shape[1]):
            col_mean = np.nanmean(predictions_array[:, col])
            nan_mask = np.isnan(predictions_array[:, col])
            if nan_mask.any():
                predictions_array[nan_mask, col] = col_mean

        # Create meta-feature DataFrame
        meta_feature_names = [f"model_{i}_pred" for i in range(len(self._fitted_base_models))]
        meta_features = pd.DataFrame(predictions_array, columns=meta_feature_names)

        if self.use_features:
            X_reset = X.reset_index(drop=True)
            meta_features = pd.concat([meta_features, X_reset], axis=1)

        return self._meta_learner.predict(meta_features)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the meta-learner.

        For stacking, importance reflects how much each base model
        contributes to the final prediction.

        Returns:
            DataFrame with 'feature' and 'importance' columns.

        Raises:
            ValueError: If ensemble is not fitted.
        """
        self._validate_fitted()
        return self._meta_learner.get_feature_importance()

    def get_base_model_weights(self) -> pd.DataFrame:
        """Get the learned weights for each base model.

        Extracts coefficients from the meta-learner that correspond
        to base model predictions.

        Returns:
            DataFrame with 'model' and 'weight' columns.

        Raises:
            ValueError: If ensemble is not fitted or meta-learner
                doesn't support coefficient extraction.
        """
        self._validate_fitted()

        try:
            importance = self._meta_learner.get_feature_importance()

            # Filter to only model prediction features
            model_features = [f"model_{i}_pred" for i in range(len(self._fitted_base_models))]
            model_importance = importance[importance["feature"].isin(model_features)].copy()

            # Create result DataFrame
            result = pd.DataFrame({
                "model": [f"Model {i}: {m.__class__.__name__}"
                         for i, m in enumerate(self._fitted_base_models)],
                "weight": model_importance["importance"].values,
            })

            return result.sort_values("weight", ascending=False).reset_index(drop=True)

        except Exception as e:
            logger.warning(f"Could not extract base model weights: {e}")
            # Return equal weights
            n_models = len(self._fitted_base_models)
            return pd.DataFrame({
                "model": [f"Model {i}: {m.__class__.__name__}"
                         for i, m in enumerate(self._fitted_base_models)],
                "weight": [1.0 / n_models] * n_models,
            })

    def get_params(self) -> dict[str, Any]:
        """Get ensemble parameters.

        Returns:
            Dictionary with ensemble configuration.
        """
        return {
            "n_base_models": len(self._fitted_base_models),
            "base_model_types": [m.__class__.__name__ for m in self._fitted_base_models],
            "meta_learner_type": self._meta_learner.__class__.__name__,
            "use_features": self.use_features,
            "cv_folds": self.cv_folds,
        }

    def set_params(self, **params: Any) -> "StackingEnsemble":
        """Set ensemble parameters.

        Args:
            **params: Parameters to set. Supports:
                - use_features: bool
                - cv_folds: int

        Returns:
            self for method chaining.

        Raises:
            ValueError: If trying to set params on fitted ensemble.
        """
        if self.is_fitted:
            raise ValueError(
                "Cannot change parameters on a fitted ensemble. "
                "Create a new instance instead."
            )

        if "use_features" in params:
            self.use_features = params["use_features"]

        if "cv_folds" in params:
            if params["cv_folds"] < 2:
                raise ValueError(f"cv_folds must be >= 2, got {params['cv_folds']}")
            self.cv_folds = params["cv_folds"]

        return self

    def save(self, path: str | Path) -> None:
        """Save the stacking ensemble to disk.

        Args:
            path: Path to save the ensemble file.

        Raises:
            ValueError: If ensemble is not fitted.
        """
        self._validate_fitted()

        path = Path(path)
        if path.suffix != ".pkl":
            path = path.with_suffix(".pkl")

        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "fitted_base_models": self._fitted_base_models,
            "meta_learner": self._meta_learner,
            "use_features": self.use_features,
            "cv_folds": self.cv_folds,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "StackingEnsemble":
        """Load a stacking ensemble from disk.

        Args:
            path: Path to the saved ensemble file.

        Returns:
            Loaded StackingEnsemble instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Ensemble file not found: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        instance = cls(
            use_features=state["use_features"],
            cv_folds=state["cv_folds"],
        )
        instance._fitted_base_models = state["fitted_base_models"]
        instance._meta_learner = state["meta_learner"]
        instance.feature_names = state["feature_names"]
        instance.is_fitted = state["is_fitted"]

        return instance

    def __repr__(self) -> str:
        """String representation of the ensemble."""
        status = "fitted" if self.is_fitted else "not fitted"
        n_models = len(self._fitted_base_models) if self._fitted_base_models else len(self._base_models)
        meta_type = self._meta_learner.__class__.__name__
        return f"StackingEnsemble(n_base_models={n_models}, meta={meta_type}, {status})"


# =============================================================================
# Helper functions
# =============================================================================


def create_ensemble(
    models: list[BaseModel],
    method: Literal["average", "weighted", "stacking"] = "average",
    weights: list[float] | np.ndarray | None = None,
    meta_learner: BaseModel | None = None,
    **kwargs,
) -> SimpleEnsemble | StackingEnsemble:
    """Factory function to create an ensemble.

    Args:
        models: List of fitted (for average/weighted) or unfitted (for stacking)
            BaseModel instances.
        method: Ensemble method to use:
            - "average": SimpleEnsemble with equal weights
            - "weighted": SimpleEnsemble with custom weights
            - "stacking": StackingEnsemble with meta-learner
        weights: Weights for each model (only for "weighted" method).
        meta_learner: Meta-learner model (only for "stacking" method).
            If None, uses LinearRegressionModel.
        **kwargs: Additional arguments passed to ensemble constructor.

    Returns:
        Configured ensemble instance.

    Raises:
        ValueError: If method is invalid or arguments don't match method.

    Example:
        >>> # Simple averaging
        >>> ensemble = create_ensemble([model1, model2], method="average")
        >>>
        >>> # Weighted averaging
        >>> ensemble = create_ensemble(
        ...     [model1, model2],
        ...     method="weighted",
        ...     weights=[0.7, 0.3]
        ... )
        >>>
        >>> # Stacking
        >>> ensemble = create_ensemble(
        ...     [model1, model2],
        ...     method="stacking",
        ...     meta_learner=GradientBoostingModel()
        ... )
    """
    valid_methods = ("average", "weighted", "stacking")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    if method == "average":
        return SimpleEnsemble(models=models, weights=None, **kwargs)

    elif method == "weighted":
        if weights is None:
            raise ValueError("weights must be provided for method='weighted'")
        return SimpleEnsemble(models=models, weights=weights, **kwargs)

    else:  # stacking
        return StackingEnsemble(
            base_models=models,
            meta_learner=meta_learner,
            **kwargs,
        )


def get_model_weights(ensemble: SimpleEnsemble | StackingEnsemble) -> pd.DataFrame:
    """Extract learned weights from an ensemble.

    For SimpleEnsemble: Returns the configured weights for each model.
    For StackingEnsemble: Returns the meta-learner's learned weights.

    Args:
        ensemble: A fitted SimpleEnsemble or StackingEnsemble instance.

    Returns:
        DataFrame with 'model' and 'weight' columns.

    Raises:
        ValueError: If ensemble is not fitted or is not a valid ensemble type.

    Example:
        >>> ensemble = SimpleEnsemble(models=[model1, model2], weights=[0.6, 0.4])
        >>> weights = get_model_weights(ensemble)
        >>> print(weights)
           model                           weight
        0  Model 0: LinearRegressionModel   0.6
        1  Model 1: GradientBoostingModel   0.4
    """
    if not isinstance(ensemble, (SimpleEnsemble, StackingEnsemble)):
        raise ValueError(
            f"ensemble must be SimpleEnsemble or StackingEnsemble, "
            f"got {type(ensemble).__name__}"
        )

    if not ensemble.is_fitted:
        raise ValueError("Ensemble is not fitted. Call fit() first.")

    if isinstance(ensemble, StackingEnsemble):
        return ensemble.get_base_model_weights()

    else:  # SimpleEnsemble
        models = ensemble.models
        weights = ensemble.weights

        return pd.DataFrame({
            "model": [f"Model {i}: {m.__class__.__name__}"
                     for i, m in enumerate(models)],
            "weight": weights,
        })
