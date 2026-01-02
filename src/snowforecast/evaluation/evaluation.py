"""Evaluation module for final model assessment on temporal holdout set.

Provides comprehensive evaluation tools including:
- PRD metrics computation
- Performance breakdown by groupings
- Residual analysis
- Bootstrap confidence intervals
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from snowforecast.models.base import BaseModel
from snowforecast.models.metrics import (
    SNOWFALL_EVENT_THRESHOLD_CM,
    bias,
    f1_score_snowfall,
    mae,
    precision_snowfall,
    recall_snowfall,
    rmse,
)

logger = logging.getLogger(__name__)


# PRD-defined targets
PRD_TARGETS = {
    "rmse": 15.0,  # Snow depth RMSE < 15 cm
    "f1": 0.85,  # Snowfall F1-score > 85%
    "baseline_improvement": 0.20,  # > 20% improvement over baseline
    "bias": 5.0,  # Bias < 5%
}


@dataclass
class PRDMetrics:
    """Container for PRD-defined success metrics.

    Attributes:
        rmse: Root mean square error for snow depth (cm)
        mae: Mean absolute error for snow depth (cm)
        bias: Mean prediction bias (cm)
        f1: F1-score for snowfall event detection
        precision: Precision for snowfall events
        recall: Recall for snowfall events
        baseline_rmse: Baseline model RMSE for comparison
        improvement_pct: Percentage improvement over baseline
        targets_met: Dict of which PRD targets were met
    """

    rmse: float
    mae: float
    bias: float
    f1: float
    precision: float
    recall: float
    baseline_rmse: float | None = None
    improvement_pct: float | None = None
    targets_met: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        """Compute improvement and check targets."""
        if self.baseline_rmse is not None and self.baseline_rmse > 0:
            self.improvement_pct = (self.baseline_rmse - self.rmse) / self.baseline_rmse

        self.targets_met = {
            "rmse": self.rmse < PRD_TARGETS["rmse"],
            "f1": self.f1 > PRD_TARGETS["f1"],
            "bias": abs(self.bias) < PRD_TARGETS["bias"],
        }
        if self.improvement_pct is not None:
            self.targets_met["baseline_improvement"] = (
                self.improvement_pct > PRD_TARGETS["baseline_improvement"]
            )

    def all_targets_met(self) -> bool:
        """Check if all PRD targets are met."""
        return all(self.targets_met.values())

    def summary(self) -> str:
        """Generate human-readable summary."""
        rmse_status = "[PASS]" if self.targets_met.get("rmse") else "[FAIL]"
        bias_status = "[PASS]" if self.targets_met.get("bias") else "[FAIL]"
        f1_status = "[PASS]" if self.targets_met.get("f1") else "[FAIL]"
        lines = [
            "PRD Metrics Summary",
            "=" * 40,
            f"RMSE: {self.rmse:.2f} cm (target: <{PRD_TARGETS['rmse']} cm) {rmse_status}",
            f"MAE: {self.mae:.2f} cm",
            f"Bias: {self.bias:+.2f} cm (target: <{PRD_TARGETS['bias']} cm) {bias_status}",
            f"F1-score: {self.f1:.1%} (target: >{PRD_TARGETS['f1']:.0%}) {f1_status}",
            f"Precision: {self.precision:.1%}",
            f"Recall: {self.recall:.1%}",
        ]

        if self.baseline_rmse is not None:
            lines.append(f"Baseline RMSE: {self.baseline_rmse:.2f} cm")
        if self.improvement_pct is not None:
            lines.append(
                f"Improvement: {self.improvement_pct:.1%} (target: >{PRD_TARGETS['baseline_improvement']:.0%}) "
                f"{'[PASS]' if self.targets_met.get('baseline_improvement') else '[FAIL]'}"
            )

        lines.append("=" * 40)
        lines.append(
            f"Overall: {'ALL TARGETS MET' if self.all_targets_met() else 'SOME TARGETS NOT MET'}"
        )

        return "\n".join(lines)


@dataclass
class ConfidenceInterval:
    """Bootstrap confidence interval for a metric.

    Attributes:
        metric_name: Name of the metric
        point_estimate: Point estimate of the metric
        lower: Lower bound of confidence interval
        upper: Upper bound of confidence interval
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples used
    """

    metric_name: str
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float = 0.95
    n_bootstrap: int = 1000

    def __str__(self) -> str:
        return (
            f"{self.metric_name}: {self.point_estimate:.3f} "
            f"({self.confidence_level:.0%} CI: [{self.lower:.3f}, {self.upper:.3f}])"
        )


class HoldoutEvaluator:
    """Evaluator for temporal holdout set.

    Runs inference on holdout data and computes comprehensive metrics.

    Example:
        >>> evaluator = HoldoutEvaluator(model, baseline_model)
        >>> results = evaluator.evaluate(X_holdout, y_holdout)
        >>> print(results.summary())
    """

    def __init__(
        self,
        model: BaseModel,
        baseline_model: BaseModel | None = None,
        threshold_cm: float = SNOWFALL_EVENT_THRESHOLD_CM,
    ):
        """Initialize the evaluator.

        Args:
            model: Trained model to evaluate
            baseline_model: Optional baseline for comparison
            threshold_cm: Threshold for snowfall event classification
        """
        self.model = model
        self.baseline_model = baseline_model
        self.threshold_cm = threshold_cm

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> PRDMetrics:
        """Evaluate model on holdout set.

        Args:
            X: Features for holdout set
            y: True values for holdout set

        Returns:
            PRDMetrics with all computed metrics
        """
        # Get predictions
        y_pred = self.model.predict(X)
        y_true = np.asarray(y)
        y_pred = np.asarray(y_pred)

        # Compute core metrics
        metrics = PRDMetrics(
            rmse=rmse(y_true, y_pred),
            mae=mae(y_true, y_pred),
            bias=bias(y_true, y_pred),
            f1=f1_score_snowfall(y_true, y_pred, threshold_cm=self.threshold_cm),
            precision=precision_snowfall(y_true, y_pred, threshold_cm=self.threshold_cm),
            recall=recall_snowfall(y_true, y_pred, threshold_cm=self.threshold_cm),
        )

        # Compute baseline if available
        if self.baseline_model is not None:
            y_baseline = self.baseline_model.predict(X)
            baseline_rmse = rmse(y_true, np.asarray(y_baseline))
            # Recreate with baseline info
            metrics = PRDMetrics(
                rmse=metrics.rmse,
                mae=metrics.mae,
                bias=metrics.bias,
                f1=metrics.f1,
                precision=metrics.precision,
                recall=metrics.recall,
                baseline_rmse=baseline_rmse,
            )

        logger.info(f"Holdout evaluation complete: RMSE={metrics.rmse:.2f}")
        return metrics

    def compute_confidence_intervals(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: int | None = None,
    ) -> dict[str, ConfidenceInterval]:
        """Compute bootstrap confidence intervals for all metrics.

        Args:
            X: Features
            y: True values
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (default 0.95 for 95% CI)
            random_state: Random seed for reproducibility

        Returns:
            Dict mapping metric name to ConfidenceInterval
        """
        rng = np.random.default_rng(random_state)
        y_true = np.asarray(y)
        y_pred = np.asarray(self.model.predict(X))
        n = len(y_true)

        # Define metrics to compute
        metric_funcs = {
            "rmse": lambda yt, yp: rmse(yt, yp),
            "mae": lambda yt, yp: mae(yt, yp),
            "bias": lambda yt, yp: bias(yt, yp),
            "f1": lambda yt, yp: f1_score_snowfall(yt, yp, threshold_cm=self.threshold_cm),
        }

        # Bootstrap
        bootstrap_results: dict[str, list[float]] = {name: [] for name in metric_funcs}

        for _ in range(n_bootstrap):
            # Sample with replacement
            idx = rng.choice(n, size=n, replace=True)
            yt_boot = y_true[idx]
            yp_boot = y_pred[idx]

            for name, func in metric_funcs.items():
                try:
                    val = func(yt_boot, yp_boot)
                    if not np.isnan(val):
                        bootstrap_results[name].append(val)
                except Exception:
                    pass

        # Compute confidence intervals
        alpha = 1 - confidence_level
        intervals = {}

        for name, values in bootstrap_results.items():
            if len(values) > 0:
                point = metric_funcs[name](y_true, y_pred)
                lower = np.percentile(values, 100 * alpha / 2)
                upper = np.percentile(values, 100 * (1 - alpha / 2))
                intervals[name] = ConfidenceInterval(
                    metric_name=name,
                    point_estimate=point,
                    lower=lower,
                    upper=upper,
                    confidence_level=confidence_level,
                    n_bootstrap=n_bootstrap,
                )

        return intervals


class MetricsBreakdown:
    """Compute metrics broken down by various groupings.

    Supports breakdown by:
    - Month
    - Station/location
    - Elevation band
    - Storm intensity

    Example:
        >>> breakdown = MetricsBreakdown(y_true, y_pred, metadata_df)
        >>> by_month = breakdown.by_month()
        >>> by_elevation = breakdown.by_elevation_band()
    """

    def __init__(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        metadata: pd.DataFrame | None = None,
        threshold_cm: float = SNOWFALL_EVENT_THRESHOLD_CM,
    ):
        """Initialize breakdown analyzer.

        Args:
            y_true: True values
            y_pred: Predicted values
            metadata: DataFrame with columns like 'date', 'station_id', 'elevation'
            threshold_cm: Threshold for snowfall event classification
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.metadata = metadata
        self.threshold_cm = threshold_cm

    def _compute_group_metrics(self, mask: np.ndarray) -> dict[str, float]:
        """Compute metrics for a group defined by mask."""
        yt = self.y_true[mask]
        yp = self.y_pred[mask]

        if len(yt) == 0:
            return {
                "n_samples": 0,
                "rmse": np.nan,
                "mae": np.nan,
                "bias": np.nan,
                "f1": np.nan,
            }

        return {
            "n_samples": len(yt),
            "rmse": rmse(yt, yp),
            "mae": mae(yt, yp),
            "bias": bias(yt, yp),
            "f1": f1_score_snowfall(yt, yp, threshold_cm=self.threshold_cm),
        }

    def by_month(self, date_column: str = "date") -> pd.DataFrame:
        """Breakdown metrics by month.

        Args:
            date_column: Name of date column in metadata

        Returns:
            DataFrame with metrics for each month (1-12)
        """
        if self.metadata is None or date_column not in self.metadata.columns:
            raise ValueError(f"metadata must contain '{date_column}' column")

        dates = pd.to_datetime(self.metadata[date_column])
        months = dates.dt.month

        results = []
        for month in range(1, 13):
            mask = (months == month).values
            metrics = self._compute_group_metrics(mask)
            metrics["month"] = month
            results.append(metrics)

        return pd.DataFrame(results).set_index("month")

    def by_station(self, station_column: str = "station_id") -> pd.DataFrame:
        """Breakdown metrics by station.

        Args:
            station_column: Name of station ID column in metadata

        Returns:
            DataFrame with metrics for each station
        """
        if self.metadata is None or station_column not in self.metadata.columns:
            raise ValueError(f"metadata must contain '{station_column}' column")

        stations = self.metadata[station_column]
        unique_stations = stations.unique()

        results = []
        for station in unique_stations:
            mask = (stations == station).values
            metrics = self._compute_group_metrics(mask)
            metrics["station_id"] = station
            results.append(metrics)

        return pd.DataFrame(results).set_index("station_id")

    def by_elevation_band(
        self,
        elevation_column: str = "elevation",
        bands: list[tuple[float, float]] | None = None,
    ) -> pd.DataFrame:
        """Breakdown metrics by elevation bands.

        Args:
            elevation_column: Name of elevation column in metadata
            bands: List of (min, max) tuples for elevation bands.
                   Default: 0-2000, 2000-2500, 2500-3000, 3000-3500, 3500+ meters

        Returns:
            DataFrame with metrics for each elevation band
        """
        if self.metadata is None or elevation_column not in self.metadata.columns:
            raise ValueError(f"metadata must contain '{elevation_column}' column")

        if bands is None:
            bands = [
                (0, 2000),
                (2000, 2500),
                (2500, 3000),
                (3000, 3500),
                (3500, float("inf")),
            ]

        elevations = self.metadata[elevation_column].values

        results = []
        for low, high in bands:
            mask = (elevations >= low) & (elevations < high)
            metrics = self._compute_group_metrics(mask)
            metrics["elevation_band"] = f"{int(low)}-{int(high) if high != float('inf') else '+'}"
            results.append(metrics)

        return pd.DataFrame(results).set_index("elevation_band")

    def by_storm_intensity(
        self,
        intensity_bins: list[float] | None = None,
    ) -> pd.DataFrame:
        """Breakdown metrics by storm intensity (actual snowfall amount).

        Args:
            intensity_bins: Bin edges for intensity classification.
                           Default: 0, 2.5, 10, 25, 50, 100+ cm

        Returns:
            DataFrame with metrics for each intensity category
        """
        if intensity_bins is None:
            intensity_bins = [0, 2.5, 10, 25, 50, 100, float("inf")]

        results = []
        for i in range(len(intensity_bins) - 1):
            low, high = intensity_bins[i], intensity_bins[i + 1]
            mask = (self.y_true >= low) & (self.y_true < high)
            metrics = self._compute_group_metrics(mask)

            if high == float("inf"):
                label = f"{int(low)}+ cm"
            else:
                label = f"{int(low)}-{int(high)} cm"

            metrics["intensity"] = label
            results.append(metrics)

        return pd.DataFrame(results).set_index("intensity")


class ResidualAnalyzer:
    """Analyze prediction residuals for patterns.

    Identifies:
    - Temporal patterns in errors
    - Spatial patterns in errors
    - Correlations with atmospheric conditions

    Example:
        >>> analyzer = ResidualAnalyzer(y_true, y_pred, metadata_df)
        >>> temporal = analyzer.temporal_patterns()
        >>> correlations = analyzer.condition_correlations(['temp', 'humidity'])
    """

    def __init__(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        metadata: pd.DataFrame | None = None,
    ):
        """Initialize residual analyzer.

        Args:
            y_true: True values
            y_pred: Predicted values
            metadata: DataFrame with date, location, and atmospheric columns
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.residuals = self.y_pred - self.y_true  # Positive = over-prediction
        self.metadata = metadata

    def summary_stats(self) -> dict[str, float]:
        """Compute summary statistics for residuals."""
        return {
            "mean": float(np.mean(self.residuals)),
            "std": float(np.std(self.residuals)),
            "median": float(np.median(self.residuals)),
            "min": float(np.min(self.residuals)),
            "max": float(np.max(self.residuals)),
            "skewness": float(self._skewness(self.residuals)),
            "kurtosis": float(self._kurtosis(self.residuals)),
        }

    def _skewness(self, x: np.ndarray) -> float:
        """Compute skewness."""
        n = len(x)
        if n < 3:
            return np.nan
        m3 = np.mean((x - np.mean(x)) ** 3)
        m2 = np.var(x)
        if m2 == 0:
            return np.nan
        return m3 / (m2 ** 1.5)

    def _kurtosis(self, x: np.ndarray) -> float:
        """Compute excess kurtosis."""
        n = len(x)
        if n < 4:
            return np.nan
        m4 = np.mean((x - np.mean(x)) ** 4)
        m2 = np.var(x)
        if m2 == 0:
            return np.nan
        return m4 / (m2 ** 2) - 3

    def temporal_patterns(self, date_column: str = "date") -> pd.DataFrame:
        """Analyze temporal patterns in residuals.

        Returns:
            DataFrame with mean residual by month, showing systematic bias patterns
        """
        if self.metadata is None or date_column not in self.metadata.columns:
            raise ValueError(f"metadata must contain '{date_column}' column")

        dates = pd.to_datetime(self.metadata[date_column])

        df = pd.DataFrame({
            "date": dates,
            "residual": self.residuals,
            "abs_residual": np.abs(self.residuals),
        })

        # Aggregate by month
        monthly = df.groupby(df["date"].dt.month).agg({
            "residual": ["mean", "std", "count"],
            "abs_residual": "mean",
        })

        monthly.columns = ["mean_bias", "std_residual", "n_samples", "mean_abs_error"]
        monthly.index.name = "month"

        return monthly

    def spatial_patterns(self, station_column: str = "station_id") -> pd.DataFrame:
        """Analyze spatial patterns in residuals by station.

        Returns:
            DataFrame with mean residual by station
        """
        if self.metadata is None or station_column not in self.metadata.columns:
            raise ValueError(f"metadata must contain '{station_column}' column")

        df = pd.DataFrame({
            "station_id": self.metadata[station_column],
            "residual": self.residuals,
            "abs_residual": np.abs(self.residuals),
        })

        spatial = df.groupby("station_id").agg({
            "residual": ["mean", "std", "count"],
            "abs_residual": "mean",
        })

        spatial.columns = ["mean_bias", "std_residual", "n_samples", "mean_abs_error"]

        return spatial

    def condition_correlations(
        self,
        condition_columns: list[str],
    ) -> pd.DataFrame:
        """Compute correlations between residuals and atmospheric conditions.

        Args:
            condition_columns: List of column names for atmospheric variables

        Returns:
            DataFrame with correlation coefficients and p-values
        """
        if self.metadata is None:
            raise ValueError("metadata required for condition correlations")

        results = []
        for col in condition_columns:
            if col not in self.metadata.columns:
                logger.warning(f"Column {col} not in metadata, skipping")
                continue

            values = self.metadata[col].values
            mask = ~np.isnan(values) & ~np.isnan(self.residuals)

            if mask.sum() < 3:
                results.append({
                    "condition": col,
                    "correlation": np.nan,
                    "n_samples": mask.sum(),
                })
                continue

            corr = np.corrcoef(values[mask], self.residuals[mask])[0, 1]
            results.append({
                "condition": col,
                "correlation": corr,
                "n_samples": mask.sum(),
            })

        return pd.DataFrame(results).set_index("condition")

    def trend_over_time(self, date_column: str = "date") -> dict[str, float]:
        """Check for performance degradation over time.

        Fits a linear trend to absolute residuals over time to detect
        if model performance is degrading.

        Returns:
            Dict with slope, intercept, and significance indicator
        """
        if self.metadata is None or date_column not in self.metadata.columns:
            raise ValueError(f"metadata must contain '{date_column}' column")

        dates = pd.to_datetime(self.metadata[date_column])
        # Convert to days since first date
        days = (dates - dates.min()).dt.days.values

        abs_residuals = np.abs(self.residuals)

        # Simple linear regression
        n = len(days)
        if n < 2:
            return {"slope": np.nan, "intercept": np.nan, "degrading": False}

        x_mean = np.mean(days)
        y_mean = np.mean(abs_residuals)

        numerator = np.sum((days - x_mean) * (abs_residuals - y_mean))
        denominator = np.sum((days - x_mean) ** 2)

        if denominator == 0:
            return {"slope": 0.0, "intercept": y_mean, "degrading": False}

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # Slope > 0 means errors increasing over time (degradation)
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "slope_per_year": float(slope * 365),  # Slope in cm/year
            "degrading": bool(slope > 0.01),  # Threshold for meaningful degradation
        }


def evaluate_holdout(
    model: BaseModel,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    baseline_model: BaseModel | None = None,
    metadata: pd.DataFrame | None = None,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Convenience function for full holdout evaluation.

    Args:
        model: Trained model to evaluate
        X: Features
        y: True values
        baseline_model: Optional baseline for comparison
        metadata: Optional metadata for breakdown analysis
        n_bootstrap: Number of bootstrap samples for CI

    Returns:
        Dict with 'prd_metrics', 'confidence_intervals', 'breakdowns', 'residual_analysis'
    """
    evaluator = HoldoutEvaluator(model, baseline_model)

    results: dict[str, Any] = {
        "prd_metrics": evaluator.evaluate(X, y),
        "confidence_intervals": evaluator.compute_confidence_intervals(
            X, y, n_bootstrap=n_bootstrap
        ),
    }

    if metadata is not None:
        y_pred = model.predict(X)
        breakdown = MetricsBreakdown(y, y_pred, metadata)
        residuals = ResidualAnalyzer(y, y_pred, metadata)

        results["breakdowns"] = {}
        results["residual_analysis"] = {}

        # Try each breakdown if column exists
        if "date" in metadata.columns:
            try:
                results["breakdowns"]["by_month"] = breakdown.by_month()
                results["residual_analysis"]["temporal"] = residuals.temporal_patterns()
                results["residual_analysis"]["trend"] = residuals.trend_over_time()
            except Exception as e:
                logger.warning(f"Could not compute date-based analysis: {e}")

        if "station_id" in metadata.columns:
            try:
                results["breakdowns"]["by_station"] = breakdown.by_station()
                results["residual_analysis"]["spatial"] = residuals.spatial_patterns()
            except Exception as e:
                logger.warning(f"Could not compute station-based analysis: {e}")

        if "elevation" in metadata.columns:
            try:
                results["breakdowns"]["by_elevation"] = breakdown.by_elevation_band()
            except Exception as e:
                logger.warning(f"Could not compute elevation-based analysis: {e}")

        results["breakdowns"]["by_intensity"] = breakdown.by_storm_intensity()
        results["residual_analysis"]["summary"] = residuals.summary_stats()

    return results


def compute_prd_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_baseline: np.ndarray | pd.Series | None = None,
    threshold_cm: float = SNOWFALL_EVENT_THRESHOLD_CM,
) -> PRDMetrics:
    """Compute PRD metrics from arrays directly.

    Args:
        y_true: True values
        y_pred: Model predictions
        y_baseline: Optional baseline predictions for comparison
        threshold_cm: Snowfall event threshold

    Returns:
        PRDMetrics with all computed values
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    baseline_rmse = None
    if y_baseline is not None:
        baseline_rmse = rmse(yt, np.asarray(y_baseline))

    return PRDMetrics(
        rmse=rmse(yt, yp),
        mae=mae(yt, yp),
        bias=bias(yt, yp),
        f1=f1_score_snowfall(yt, yp, threshold_cm=threshold_cm),
        precision=precision_snowfall(yt, yp, threshold_cm=threshold_cm),
        recall=recall_snowfall(yt, yp, threshold_cm=threshold_cm),
        baseline_rmse=baseline_rmse,
    )


def check_prd_targets(metrics: PRDMetrics) -> dict[str, bool]:
    """Check which PRD targets are met.

    Args:
        metrics: PRDMetrics to check

    Returns:
        Dict mapping target name to whether it was met
    """
    return metrics.targets_met.copy()
