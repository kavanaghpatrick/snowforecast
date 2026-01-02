"""Model comparison and selection utilities for snowfall prediction.

This module provides tools for comparing multiple models and selecting
the best performing one:

- ModelComparison: Class for evaluating and comparing multiple models
- ComparisonResults: Container for comparison results with export methods
- Helper functions for convenience wrappers and report generation

Statistical significance testing uses paired t-tests to determine if
performance differences between models are significant.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy import stats

from snowforecast.models.metrics import (
    METRICS,
    compute_all_metrics,
    compute_metrics_by_group,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class PredictorProtocol(Protocol):
    """Protocol for models that can make predictions.

    Any object with a predict method that takes features and returns
    predictions will satisfy this protocol.
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on input features."""
        ...


@dataclass
class ComparisonResults:
    """Container for model comparison results with export methods.

    Stores per-model metrics, stratified metrics by group, and provides
    methods for summarizing and exporting results.

    Attributes:
        model_metrics: Dictionary mapping model names to their metric dictionaries.
        stratified_metrics: Dictionary mapping model names to DataFrames of
            metrics by group (e.g., by elevation band or region).
        predictions: Dictionary mapping model names to their predictions array.
        y_true: The true target values used in the comparison.

    Example:
        >>> results = ComparisonResults()
        >>> results.add_model_metrics("model_a", {"rmse": 12.5, "mae": 8.3})
        >>> results.add_model_metrics("model_b", {"rmse": 10.2, "mae": 7.1})
        >>> print(results.summary())
    """

    model_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    stratified_metrics: dict[str, pd.DataFrame] = field(default_factory=dict)
    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    y_true: np.ndarray | None = None

    def add_model_metrics(
        self,
        model_name: str,
        metrics: dict[str, float],
        predictions: np.ndarray | None = None,
    ) -> None:
        """Add metrics for a model.

        Args:
            model_name: Name of the model.
            metrics: Dictionary of metric name to value.
            predictions: Optional array of model predictions.
        """
        self.model_metrics[model_name] = metrics
        if predictions is not None:
            self.predictions[model_name] = predictions

    def add_stratified_metrics(
        self,
        model_name: str,
        metrics_df: pd.DataFrame,
    ) -> None:
        """Add stratified metrics for a model.

        Args:
            model_name: Name of the model.
            metrics_df: DataFrame with metrics by group.
        """
        self.stratified_metrics[model_name] = metrics_df

    def summary(self) -> str:
        """Generate a summary string of comparison results.

        Returns:
            Formatted string summarizing model comparison results,
            including best model for each metric.
        """
        if not self.model_metrics:
            return "No comparison results available."

        lines = ["=" * 60]
        lines.append("MODEL COMPARISON SUMMARY")
        lines.append("=" * 60)
        lines.append("")

        # Get all metrics from first model
        first_model = next(iter(self.model_metrics.values()))
        metric_names = list(first_model.keys())

        # Per-model results
        lines.append("Per-Model Metrics:")
        lines.append("-" * 40)

        for model_name, metrics in self.model_metrics.items():
            lines.append(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                if np.isnan(value):
                    lines.append(f"  {metric_name}: N/A")
                else:
                    lines.append(f"  {metric_name}: {value:.4f}")

        # Best model for each metric
        lines.append("")
        lines.append("-" * 40)
        lines.append("Best Model by Metric:")
        lines.append("-" * 40)

        for metric_name in metric_names:
            values = {}
            for model_name, metrics in self.model_metrics.items():
                if metric_name in metrics and not np.isnan(metrics[metric_name]):
                    values[model_name] = metrics[metric_name]

            if values:
                # Lower is better for rmse, mae; higher is better for f1, precision, recall
                lower_is_better = metric_name in ["rmse", "mae"]

                if lower_is_better:
                    best_model = min(values, key=values.get)
                else:
                    best_model = max(values, key=values.get)

                lines.append(f"  {metric_name}: {best_model} ({values[best_model]:.4f})")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert comparison results to a DataFrame.

        Returns:
            DataFrame with models as rows and metrics as columns.
        """
        if not self.model_metrics:
            return pd.DataFrame()

        rows = []
        for model_name, metrics in self.model_metrics.items():
            row = {"model": model_name, **metrics}
            rows.append(row)

        df = pd.DataFrame(rows)

        # Reorder columns to put model first
        cols = ["model"] + [c for c in df.columns if c != "model"]
        return df[cols]

    def get_best_model_name(self, metric: str = "rmse") -> str | None:
        """Get the name of the best performing model.

        Args:
            metric: Metric to use for comparison. Default "rmse".

        Returns:
            Name of best model, or None if no valid comparisons.
        """
        if not self.model_metrics:
            return None

        values = {}
        for model_name, metrics in self.model_metrics.items():
            if metric in metrics and not np.isnan(metrics[metric]):
                values[model_name] = metrics[metric]

        if not values:
            return None

        # Lower is better for rmse, mae, bias(abs); higher is better for f1, precision, recall
        lower_is_better = metric in ["rmse", "mae"]

        if lower_is_better:
            return min(values, key=values.get)
        else:
            return max(values, key=values.get)

    def __repr__(self) -> str:
        """String representation of results."""
        n_models = len(self.model_metrics)
        if n_models == 0:
            return "ComparisonResults(empty)"

        model_names = list(self.model_metrics.keys())
        return f"ComparisonResults(models={model_names})"


class ModelComparison:
    """Class for comparing multiple models on the same dataset.

    Provides methods to evaluate all models, compare performance by group,
    identify the best model, and test statistical significance of differences.

    Attributes:
        models: Dictionary mapping model names to model instances.
        results: ComparisonResults storing evaluation results.

    Example:
        >>> models = {
        ...     "linear": LinearRegressionModel(),
        ...     "gbm": GradientBoostingModel(),
        ... }
        >>> # Fit models first...
        >>> comparison = ModelComparison(models)
        >>> results_df = comparison.evaluate_all(X_test, y_test)
        >>> best_name, best_model = comparison.get_best_model(metric="rmse")
        >>> print(f"Best model: {best_name}")
    """

    def __init__(self, models: dict[str, Any]):
        """Initialize model comparison.

        Args:
            models: Dictionary mapping model names to model instances.
                Models should have a predict(X) method.

        Raises:
            ValueError: If models dict is empty.
        """
        if not models:
            raise ValueError("models dict cannot be empty")

        self.models = models
        self.results = ComparisonResults()
        self._evaluated = False

    def evaluate_all(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series | np.ndarray,
        snowfall_threshold_cm: float = 2.5,
    ) -> pd.DataFrame:
        """Evaluate all models on test data.

        Args:
            X_test: Test features as DataFrame.
            y_test: Test targets as Series or array.
            snowfall_threshold_cm: Threshold for binary event classification.

        Returns:
            DataFrame with one row per model and columns for each metric.

        Raises:
            ValueError: If X_test is empty.
        """
        if X_test.empty:
            raise ValueError("X_test cannot be empty")

        y_test = np.asarray(y_test)
        self.results = ComparisonResults()
        self.results.y_true = y_test

        for model_name, model in self.models.items():
            try:
                # Get predictions
                predictions = model.predict(X_test)
                predictions = np.asarray(predictions).flatten()

                # Compute all metrics
                metrics = compute_all_metrics(
                    y_test, predictions, snowfall_threshold_cm
                )

                # Store results
                self.results.add_model_metrics(model_name, metrics, predictions)

                logger.info(
                    f"Evaluated {model_name}: RMSE={metrics['rmse']:.3f}, "
                    f"MAE={metrics['mae']:.3f}, F1={metrics['f1']:.3f}"
                )

            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                # Store NaN metrics for failed models
                nan_metrics = {k: np.nan for k in METRICS.keys()}
                self.results.add_model_metrics(model_name, nan_metrics)

        self._evaluated = True
        return self.results.to_dataframe()

    def compare_by_group(
        self,
        df: pd.DataFrame,
        group_col: str,
        y_true_col: str = "y_true",
        snowfall_threshold_cm: float = 2.5,
    ) -> dict[str, pd.DataFrame]:
        """Compute stratified metrics for each model by group.

        This method requires predictions to be added to the DataFrame
        as columns named "{model_name}_pred" for each model.

        Args:
            df: DataFrame with true values and predictions for each model.
                Must contain y_true_col and columns "{model_name}_pred" for
                each model in self.models.
            group_col: Column to group by (e.g., "elevation_band", "region").
            y_true_col: Column name for true values.
            snowfall_threshold_cm: Threshold for binary event classification.

        Returns:
            Dictionary mapping model names to DataFrames of metrics by group.

        Raises:
            ValueError: If required columns are missing.
        """
        if group_col not in df.columns:
            raise ValueError(f"group_col '{group_col}' not found in DataFrame")

        if y_true_col not in df.columns:
            raise ValueError(f"y_true_col '{y_true_col}' not found in DataFrame")

        stratified_results = {}

        for model_name in self.models.keys():
            pred_col = f"{model_name}_pred"

            if pred_col not in df.columns:
                logger.warning(
                    f"Prediction column '{pred_col}' not found, skipping {model_name}"
                )
                continue

            # Create temporary df with y_true, y_pred columns
            temp_df = df[[group_col, y_true_col, pred_col]].copy()
            temp_df = temp_df.rename(columns={pred_col: "y_pred", y_true_col: "y_true"})

            # Compute metrics by group
            metrics_df = compute_metrics_by_group(
                temp_df,
                group_col,
                y_true_col="y_true",
                y_pred_col="y_pred",
                snowfall_threshold_cm=snowfall_threshold_cm,
            )

            stratified_results[model_name] = metrics_df
            self.results.add_stratified_metrics(model_name, metrics_df)

            logger.info(f"Computed stratified metrics for {model_name}")

        return stratified_results

    def get_best_model(
        self,
        metric: str = "rmse",
    ) -> tuple[str, Any] | tuple[None, None]:
        """Get the best performing model based on specified metric.

        Args:
            metric: Metric to use for comparison. Options:
                - "rmse": Root Mean Square Error (lower is better)
                - "mae": Mean Absolute Error (lower is better)
                - "bias": Mean Error (closest to 0 is best)
                - "f1": F1 score (higher is better)
                - "precision": Precision (higher is better)
                - "recall": Recall (higher is better)

        Returns:
            Tuple of (model_name, model_instance), or (None, None) if
            no valid models.

        Raises:
            ValueError: If evaluate_all has not been called.
            ValueError: If metric is not recognized.
        """
        if not self._evaluated:
            raise ValueError(
                "Must call evaluate_all() before get_best_model()"
            )

        valid_metrics = ["rmse", "mae", "bias", "f1", "precision", "recall"]
        if metric not in valid_metrics:
            raise ValueError(
                f"Unknown metric '{metric}'. Valid options: {valid_metrics}"
            )

        best_name = self.results.get_best_model_name(metric)

        if best_name is None:
            return None, None

        return best_name, self.models[best_name]

    def statistical_significance_test(
        self,
        model1: str,
        model2: str,
        y_true: np.ndarray | pd.Series | None = None,
        y_pred1: np.ndarray | None = None,
        y_pred2: np.ndarray | None = None,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Perform paired t-test to compare two models.

        Tests whether the difference in squared errors between two models
        is statistically significant using a paired t-test.

        Args:
            model1: Name of first model.
            model2: Name of second model.
            y_true: True values. If None, uses stored values from evaluate_all.
            y_pred1: Predictions from model1. If None, uses stored predictions.
            y_pred2: Predictions from model2. If None, uses stored predictions.
            alpha: Significance level (default 0.05).

        Returns:
            Dictionary with:
                - t_statistic: The t-test statistic
                - p_value: Two-tailed p-value
                - significant: Whether difference is significant at alpha level
                - model1_mean_se: Mean squared error for model1
                - model2_mean_se: Mean squared error for model2
                - better_model: Name of statistically better model (if significant)

        Raises:
            ValueError: If predictions not available for either model.
        """
        # Get true values
        if y_true is None:
            if self.results.y_true is None:
                raise ValueError(
                    "y_true not provided and evaluate_all() not called"
                )
            y_true = self.results.y_true

        y_true = np.asarray(y_true)

        # Get predictions for model1
        if y_pred1 is None:
            if model1 not in self.results.predictions:
                raise ValueError(f"No predictions available for '{model1}'")
            y_pred1 = self.results.predictions[model1]

        y_pred1 = np.asarray(y_pred1).flatten()

        # Get predictions for model2
        if y_pred2 is None:
            if model2 not in self.results.predictions:
                raise ValueError(f"No predictions available for '{model2}'")
            y_pred2 = self.results.predictions[model2]

        y_pred2 = np.asarray(y_pred2).flatten()

        # Validate shapes
        if len(y_true) != len(y_pred1) or len(y_true) != len(y_pred2):
            raise ValueError(
                f"Shape mismatch: y_true={len(y_true)}, "
                f"y_pred1={len(y_pred1)}, y_pred2={len(y_pred2)}"
            )

        # Handle NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred1) | np.isnan(y_pred2))
        if not mask.any():
            return {
                "t_statistic": np.nan,
                "p_value": np.nan,
                "significant": False,
                "model1_mean_se": np.nan,
                "model2_mean_se": np.nan,
                "better_model": None,
            }

        y_true_valid = y_true[mask]
        y_pred1_valid = y_pred1[mask]
        y_pred2_valid = y_pred2[mask]

        # Calculate squared errors
        se1 = (y_true_valid - y_pred1_valid) ** 2
        se2 = (y_true_valid - y_pred2_valid) ** 2

        # Paired t-test on squared errors
        # Null hypothesis: mean(se1) = mean(se2)
        t_stat, p_value = stats.ttest_rel(se1, se2)

        significant = p_value < alpha

        mean_se1 = float(np.mean(se1))
        mean_se2 = float(np.mean(se2))

        # Determine better model
        better_model = None
        if significant:
            better_model = model1 if mean_se1 < mean_se2 else model2

        result = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": significant,
            "model1_mean_se": mean_se1,
            "model2_mean_se": mean_se2,
            "better_model": better_model,
        }

        logger.info(
            f"Significance test {model1} vs {model2}: "
            f"t={t_stat:.3f}, p={p_value:.4f}, significant={significant}"
        )

        return result

    def get_results(self) -> ComparisonResults:
        """Get the comparison results object.

        Returns:
            ComparisonResults with all stored metrics and predictions.
        """
        return self.results


def compare_models(
    models_dict: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    snowfall_threshold_cm: float = 2.5,
) -> ComparisonResults:
    """Convenience function to compare multiple models.

    Args:
        models_dict: Dictionary mapping model names to fitted model instances.
        X_test: Test features.
        y_test: Test targets.
        snowfall_threshold_cm: Threshold for binary event classification.

    Returns:
        ComparisonResults with evaluation results for all models.

    Example:
        >>> results = compare_models(
        ...     {"linear": linear_model, "gbm": gbm_model},
        ...     X_test, y_test
        ... )
        >>> print(results.summary())
    """
    comparison = ModelComparison(models_dict)
    comparison.evaluate_all(X_test, y_test, snowfall_threshold_cm)
    return comparison.get_results()


def rank_models(
    comparison_results: ComparisonResults,
    metric: str = "rmse",
) -> list[tuple[str, float]]:
    """Rank models by specified metric.

    Args:
        comparison_results: Results from model comparison.
        metric: Metric to rank by. Default "rmse".

    Returns:
        List of (model_name, metric_value) tuples, sorted from best to worst.

    Example:
        >>> rankings = rank_models(results, metric="f1")
        >>> for rank, (name, score) in enumerate(rankings, 1):
        ...     print(f"{rank}. {name}: {score:.4f}")
    """
    if not comparison_results.model_metrics:
        return []

    # Collect valid values
    values = []
    for model_name, metrics in comparison_results.model_metrics.items():
        if metric in metrics and not np.isnan(metrics[metric]):
            values.append((model_name, metrics[metric]))

    if not values:
        return []

    # Lower is better for rmse, mae; higher is better for f1, precision, recall
    lower_is_better = metric in ["rmse", "mae"]

    sorted_values = sorted(values, key=lambda x: x[1], reverse=not lower_is_better)

    return sorted_values


def create_comparison_report(
    comparison_results: ComparisonResults,
    include_stratified: bool = True,
) -> str:
    """Create a formatted comparison report.

    Args:
        comparison_results: Results from model comparison.
        include_stratified: Whether to include stratified metrics if available.

    Returns:
        Formatted string report of model comparison.

    Example:
        >>> report = create_comparison_report(results)
        >>> print(report)
    """
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append("SNOWFALL PREDICTION MODEL COMPARISON REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Overall metrics table
    df = comparison_results.to_dataframe()
    if df.empty:
        lines.append("No model comparison results available.")
        return "\n".join(lines)

    lines.append("OVERALL METRICS")
    lines.append("-" * 70)

    # Format metrics table
    lines.append(f"{'Model':<20} {'RMSE':>10} {'MAE':>10} {'Bias':>10} {'F1':>10}")
    lines.append("-" * 70)

    for _, row in df.iterrows():
        model = row["model"]
        rmse_val = f"{row.get('rmse', np.nan):.3f}" if not np.isnan(row.get('rmse', np.nan)) else "N/A"
        mae_val = f"{row.get('mae', np.nan):.3f}" if not np.isnan(row.get('mae', np.nan)) else "N/A"
        bias_val = f"{row.get('bias', np.nan):.3f}" if not np.isnan(row.get('bias', np.nan)) else "N/A"
        f1_val = f"{row.get('f1', np.nan):.3f}" if not np.isnan(row.get('f1', np.nan)) else "N/A"

        lines.append(f"{model:<20} {rmse_val:>10} {mae_val:>10} {bias_val:>10} {f1_val:>10}")

    lines.append("")

    # Rankings
    lines.append("MODEL RANKINGS")
    lines.append("-" * 70)

    for metric in ["rmse", "f1"]:
        rankings = rank_models(comparison_results, metric)
        if rankings:
            metric_label = "RMSE (lower is better)" if metric == "rmse" else "F1 (higher is better)"
            lines.append(f"\nBy {metric_label}:")
            for rank, (name, score) in enumerate(rankings, 1):
                lines.append(f"  {rank}. {name}: {score:.4f}")

    lines.append("")

    # Best models
    lines.append("BEST MODELS")
    lines.append("-" * 70)

    best_rmse = comparison_results.get_best_model_name("rmse")
    best_f1 = comparison_results.get_best_model_name("f1")

    if best_rmse:
        rmse_val = comparison_results.model_metrics[best_rmse]["rmse"]
        lines.append(f"Best by RMSE: {best_rmse} ({rmse_val:.4f})")

    if best_f1:
        f1_val = comparison_results.model_metrics[best_f1]["f1"]
        lines.append(f"Best by F1:   {best_f1} ({f1_val:.4f})")

    lines.append("")

    # Stratified metrics if available
    if include_stratified and comparison_results.stratified_metrics:
        lines.append("STRATIFIED METRICS")
        lines.append("-" * 70)

        for model_name, strat_df in comparison_results.stratified_metrics.items():
            lines.append(f"\n{model_name}:")
            lines.append(strat_df.to_string(index=False))
            lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)
