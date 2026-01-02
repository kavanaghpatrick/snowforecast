"""Cross-validation utilities for snowfall prediction.

This module provides cross-validation strategies that respect the spatial
and temporal structure of snow data:

- StationKFold: K-fold CV that splits by station_id, ensuring no data leakage
  between training and validation sets from the same location.
- TemporalSplit: Time-based train/test split for final holdout evaluation.
- CVResults: Container for cross-validation results with summary statistics.

IMPORTANT: Station-based splitting is critical because:
1. Nearby stations have correlated weather patterns
2. Same-station samples across time are highly correlated
3. We need to evaluate generalization to unseen locations
"""

import logging
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StationKFold:
    """K-fold cross-validation that splits by station_id.

    Unlike standard K-fold which splits individual samples, this splitter
    groups all samples from each station together. This ensures:
    - No data leakage from train to validation for the same location
    - Model is evaluated on ability to generalize to unseen stations
    - Temporal correlation within stations doesn't inflate validation scores

    This follows the sklearn splitter interface, making it compatible with
    sklearn's cross_val_score and similar utilities.

    Example:
        >>> cv = StationKFold(n_splits=5, shuffle=True, random_state=42)
        >>> for train_idx, val_idx in cv.split(X, groups=X["station_id"]):
        ...     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        ...     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        ...     model.fit(X_train, y_train)
        ...     scores.append(model.score(X_val, y_val))

    Attributes:
        n_splits: Number of folds (default 5)
        shuffle: Whether to shuffle stations before splitting (default True)
        random_state: Random seed for reproducibility (default 42)
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        """Initialize StationKFold splitter.

        Args:
            n_splits: Number of folds. Must be at least 2.
            shuffle: Whether to shuffle station order before splitting.
                Recommended True for better randomization across folds.
            random_state: Random seed for reproducibility when shuffle=True.

        Raises:
            ValueError: If n_splits < 2
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        groups: pd.Series | np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation indices for each fold.

        Splits stations into n_splits folds, then returns indices of all
        samples belonging to each station group.

        Args:
            X: Feature DataFrame. Must contain station_id column or groups
                must be provided.
            y: Target values (unused, included for sklearn compatibility).
            groups: Array-like of station IDs for each sample. If None,
                X must have a "station_id" column.

        Yields:
            Tuple of (train_indices, val_indices) as numpy arrays.

        Raises:
            ValueError: If groups is None and X has no "station_id" column.
            ValueError: If number of unique stations < n_splits.

        Example:
            >>> cv = StationKFold(n_splits=5)
            >>> for fold, (train_idx, val_idx) in enumerate(cv.split(df)):
            ...     print(f"Fold {fold}: {len(train_idx)} train, {len(val_idx)} val")
        """
        # Get station IDs
        if groups is not None:
            station_ids = np.asarray(groups)
        elif "station_id" in X.columns:
            station_ids = X["station_id"].values
        else:
            raise ValueError(
                "groups must be provided or X must have 'station_id' column"
            )

        # Get unique stations
        unique_stations = np.unique(station_ids)
        n_stations = len(unique_stations)

        if n_stations < self.n_splits:
            raise ValueError(
                f"Cannot split {n_stations} stations into {self.n_splits} folds. "
                f"Need at least {self.n_splits} unique stations."
            )

        # Shuffle stations if requested
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(unique_stations)

        # Split stations into folds
        fold_sizes = np.full(self.n_splits, n_stations // self.n_splits)
        fold_sizes[: n_stations % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            # Stations in validation set for this fold
            val_stations = set(unique_stations[current : current + fold_size])

            # Get indices for train and validation
            val_mask = np.isin(station_ids, list(val_stations))
            train_idx = np.where(~val_mask)[0]
            val_idx = np.where(val_mask)[0]

            logger.debug(
                f"Fold: {len(val_stations)} validation stations, "
                f"{len(train_idx)} train samples, {len(val_idx)} val samples"
            )

            yield train_idx, val_idx
            current += fold_size

    def get_n_splits(
        self,
        X: pd.DataFrame | None = None,
        y: pd.Series | np.ndarray | None = None,
        groups: pd.Series | np.ndarray | None = None,
    ) -> int:
        """Return the number of splits.

        Args:
            X: Ignored, exists for sklearn compatibility.
            y: Ignored, exists for sklearn compatibility.
            groups: Ignored, exists for sklearn compatibility.

        Returns:
            Number of folds (n_splits).
        """
        return self.n_splits


class TemporalSplit:
    """Time-based train/test split for final holdout evaluation.

    Splits data by time, reserving the most recent years for testing.
    This ensures models are evaluated on their ability to predict future
    conditions, not just interpolate within known time periods.

    For snowfall prediction, we typically:
    - Train on data before 2024
    - Hold out 2024-2025 for final evaluation

    Example:
        >>> splitter = TemporalSplit(test_years=2)
        >>> train_df, test_df = splitter.split(df)
        >>> print(f"Train: {train_df['date'].min()} to {train_df['date'].max()}")
        >>> print(f"Test: {test_df['date'].min()} to {test_df['date'].max()}")

    Attributes:
        test_years: Number of most recent years to hold out (default 2)
        cutoff_date: Explicit cutoff date (optional, overrides test_years)
    """

    def __init__(
        self,
        test_years: int = 2,
        cutoff_date: str | pd.Timestamp | None = None,
    ):
        """Initialize TemporalSplit.

        Args:
            test_years: Number of years to hold out for testing.
                Data from the most recent `test_years` years will be in
                the test set.
            cutoff_date: Explicit cutoff date (YYYY-MM-DD format or Timestamp).
                If provided, overrides test_years. Data before this date
                goes to training, on or after goes to test.

        Raises:
            ValueError: If test_years < 1 and cutoff_date is None.
        """
        if cutoff_date is None and test_years < 1:
            raise ValueError(f"test_years must be at least 1, got {test_years}")

        self.test_years = test_years
        self.cutoff_date = pd.Timestamp(cutoff_date) if cutoff_date else None

    def split(
        self,
        df: pd.DataFrame,
        datetime_col: str = "datetime",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split DataFrame into train and test sets by time.

        Args:
            df: DataFrame with datetime column.
            datetime_col: Name of the datetime column. Also accepts "date".

        Returns:
            Tuple of (train_df, test_df).

        Raises:
            ValueError: If datetime_col not found in DataFrame.
            ValueError: If DataFrame is empty.

        Example:
            >>> splitter = TemporalSplit(test_years=2)
            >>> train, test = splitter.split(df, datetime_col="date")
        """
        if df.empty:
            raise ValueError("Cannot split empty DataFrame")

        # Find datetime column
        if datetime_col not in df.columns:
            # Try common alternatives
            for alt_col in ["date", "datetime", "timestamp", "time"]:
                if alt_col in df.columns:
                    datetime_col = alt_col
                    break
            else:
                raise ValueError(
                    f"datetime_col '{datetime_col}' not found in DataFrame columns: "
                    f"{list(df.columns)}"
                )

        df = df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])

        # Determine cutoff date
        if self.cutoff_date is not None:
            cutoff = self.cutoff_date
        else:
            # Use test_years from the most recent date
            max_date = df[datetime_col].max()
            cutoff = max_date - pd.DateOffset(years=self.test_years)

        # Split
        train_mask = df[datetime_col] < cutoff
        test_mask = df[datetime_col] >= cutoff

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        logger.info(
            f"Temporal split at {cutoff.date()}: "
            f"{len(train_df)} train samples, {len(test_df)} test samples"
        )

        return train_df, test_df

    def get_cutoff_date(
        self,
        df: pd.DataFrame,
        datetime_col: str = "datetime",
    ) -> pd.Timestamp:
        """Calculate the cutoff date for a given DataFrame.

        Args:
            df: DataFrame with datetime column.
            datetime_col: Name of the datetime column.

        Returns:
            Cutoff date as Timestamp.
        """
        if self.cutoff_date is not None:
            return self.cutoff_date

        # Find datetime column
        if datetime_col not in df.columns:
            for alt_col in ["date", "datetime", "timestamp", "time"]:
                if alt_col in df.columns:
                    datetime_col = alt_col
                    break

        max_date = pd.to_datetime(df[datetime_col]).max()
        return max_date - pd.DateOffset(years=self.test_years)


@dataclass
class CVResults:
    """Container for cross-validation results with summary statistics.

    Stores per-fold metrics and predictions, providing methods to compute
    aggregate statistics across folds.

    Example:
        >>> results = CVResults()
        >>> for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        ...     # Train model, make predictions
        ...     metrics = {"rmse": 12.5, "mae": 8.3}
        ...     predictions = pd.DataFrame({"y_true": y_val, "y_pred": preds})
        ...     results.add_fold(fold, metrics, predictions)
        >>> summary = results.summary()
        >>> print(f"RMSE: {summary['rmse']['mean']:.2f} +/- {summary['rmse']['std']:.2f}")

    Attributes:
        fold_metrics: List of metric dictionaries, one per fold.
        fold_predictions: List of prediction DataFrames, one per fold.
        predictions: Combined predictions from all folds (after aggregation).
    """

    fold_metrics: list[dict] = field(default_factory=list)
    fold_predictions: list[pd.DataFrame] = field(default_factory=list)
    predictions: pd.DataFrame | None = None

    def add_fold(
        self,
        fold: int,
        metrics: dict[str, float],
        predictions: pd.DataFrame | None = None,
    ) -> None:
        """Add results from a single fold.

        Args:
            fold: Fold index (0-based).
            metrics: Dictionary of metric names to values.
            predictions: Optional DataFrame with predictions for this fold.
                Should contain at least "y_true" and "y_pred" columns.

        Example:
            >>> results.add_fold(
            ...     fold=0,
            ...     metrics={"rmse": 12.5, "mae": 8.3, "f1": 0.82},
            ...     predictions=pd.DataFrame({
            ...         "y_true": y_val,
            ...         "y_pred": model.predict(X_val),
            ...         "station_id": X_val["station_id"],
            ...     })
            ... )
        """
        # Add fold index to metrics
        metrics_with_fold = {"fold": fold, **metrics}
        self.fold_metrics.append(metrics_with_fold)

        if predictions is not None:
            pred_with_fold = predictions.copy()
            pred_with_fold["fold"] = fold
            self.fold_predictions.append(pred_with_fold)

        logger.debug(f"Added fold {fold} results: {metrics}")

    def summary(self) -> dict[str, dict[str, float]]:
        """Compute summary statistics across all folds.

        Returns:
            Dictionary mapping metric names to statistics:
                - mean: Mean value across folds
                - std: Standard deviation across folds
                - min: Minimum value across folds
                - max: Maximum value across folds

        Example:
            >>> summary = results.summary()
            >>> print(f"RMSE: {summary['rmse']['mean']:.2f} +/- {summary['rmse']['std']:.2f}")
            >>> print(f"Best fold RMSE: {summary['rmse']['min']:.2f}")
        """
        if not self.fold_metrics:
            return {}

        # Get all metric names (excluding 'fold')
        metric_names = set()
        for fold_result in self.fold_metrics:
            metric_names.update(k for k in fold_result.keys() if k != "fold")

        summary = {}
        for metric in metric_names:
            values = [
                fm[metric] for fm in self.fold_metrics if metric in fm
            ]
            if values:
                summary[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

        return summary

    def get_fold_metrics_df(self) -> pd.DataFrame:
        """Get fold metrics as a DataFrame.

        Returns:
            DataFrame with one row per fold, columns are metrics.

        Example:
            >>> df = results.get_fold_metrics_df()
            >>> print(df.to_string())
        """
        if not self.fold_metrics:
            return pd.DataFrame()
        return pd.DataFrame(self.fold_metrics)

    def get_all_predictions(self) -> pd.DataFrame:
        """Get combined predictions from all folds.

        Returns:
            DataFrame with predictions from all folds concatenated.
            Includes a 'fold' column to identify which fold each
            prediction came from.

        Example:
            >>> all_preds = results.get_all_predictions()
            >>> rmse_by_station = all_preds.groupby("station_id").apply(
            ...     lambda g: np.sqrt(np.mean((g["y_true"] - g["y_pred"])**2))
            ... )
        """
        if not self.fold_predictions:
            return pd.DataFrame()

        combined = pd.concat(self.fold_predictions, ignore_index=True)
        self.predictions = combined
        return combined

    def n_folds(self) -> int:
        """Return number of folds with results."""
        return len(self.fold_metrics)

    def __repr__(self) -> str:
        """String representation of CVResults."""
        n_folds = self.n_folds()
        if n_folds == 0:
            return "CVResults(empty)"

        summary = self.summary()
        metric_strs = []
        for metric, stats in summary.items():
            metric_strs.append(f"{metric}={stats['mean']:.3f}+/-{stats['std']:.3f}")

        return f"CVResults(n_folds={n_folds}, {', '.join(metric_strs)})"
