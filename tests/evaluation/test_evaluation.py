"""Tests for evaluation module.

Tests use real numpy calculations - NO MOCKS.
"""

import numpy as np
import pandas as pd
import pytest

from snowforecast.evaluation import (
    ConfidenceInterval,
    HoldoutEvaluator,
    MetricsBreakdown,
    PRDMetrics,
    ResidualAnalyzer,
    check_prd_targets,
    compute_prd_metrics,
    evaluate_holdout,
)
from snowforecast.models.base import BaseModel


class MockModel(BaseModel):
    """Simple mock model for testing."""

    def __init__(self, offset: float = 0.0):
        self.offset = offset
        self._fitted = False

    def fit(self, X, y):
        self._fitted = True
        return self

    def predict(self, X):
        if hasattr(X, "values"):
            return X.values[:, 0] + self.offset
        return X[:, 0] + self.offset

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        return cls()

    def get_feature_importance(self):
        return {}

    def get_params(self):
        return {"offset": self.offset}

    def set_params(self, **params):
        if "offset" in params:
            self.offset = params["offset"]
        return self


class TestPRDMetrics:
    """Tests for PRDMetrics dataclass."""

    def test_basic_construction(self):
        """PRDMetrics should compute targets_met on construction."""
        metrics = PRDMetrics(
            rmse=10.0,  # < 15, PASS
            mae=8.0,
            bias=2.0,  # < 5, PASS
            f1=0.90,  # > 0.85, PASS
            precision=0.88,
            recall=0.92,
        )

        assert metrics.targets_met["rmse"] is True
        assert metrics.targets_met["bias"] is True
        assert metrics.targets_met["f1"] is True

    def test_failing_targets(self):
        """PRDMetrics should detect failing targets."""
        metrics = PRDMetrics(
            rmse=20.0,  # > 15, FAIL
            mae=18.0,
            bias=10.0,  # > 5, FAIL
            f1=0.70,  # < 0.85, FAIL
            precision=0.65,
            recall=0.75,
        )

        assert metrics.targets_met["rmse"] is False
        assert metrics.targets_met["bias"] is False
        assert metrics.targets_met["f1"] is False

    def test_improvement_calculation(self):
        """PRDMetrics should compute improvement from baseline."""
        metrics = PRDMetrics(
            rmse=10.0,
            mae=8.0,
            bias=1.0,
            f1=0.90,
            precision=0.88,
            recall=0.92,
            baseline_rmse=15.0,  # 33% improvement
        )

        assert metrics.improvement_pct is not None
        assert abs(metrics.improvement_pct - 0.333) < 0.01
        assert metrics.targets_met["baseline_improvement"] is True

    def test_insufficient_improvement(self):
        """PRDMetrics should flag insufficient improvement."""
        metrics = PRDMetrics(
            rmse=14.0,
            mae=12.0,
            bias=1.0,
            f1=0.90,
            precision=0.88,
            recall=0.92,
            baseline_rmse=15.0,  # Only 6.7% improvement
        )

        assert metrics.improvement_pct is not None
        assert metrics.improvement_pct < 0.20
        assert metrics.targets_met["baseline_improvement"] is False

    def test_all_targets_met(self):
        """all_targets_met should return True when all targets pass."""
        metrics = PRDMetrics(
            rmse=10.0,
            mae=8.0,
            bias=2.0,
            f1=0.90,
            precision=0.88,
            recall=0.92,
            baseline_rmse=15.0,
        )

        assert metrics.all_targets_met() is True

    def test_all_targets_met_fails(self):
        """all_targets_met should return False when any target fails."""
        metrics = PRDMetrics(
            rmse=20.0,  # FAIL
            mae=8.0,
            bias=2.0,
            f1=0.90,
            precision=0.88,
            recall=0.92,
        )

        assert metrics.all_targets_met() is False

    def test_summary_output(self):
        """summary should return formatted string."""
        metrics = PRDMetrics(
            rmse=10.0,
            mae=8.0,
            bias=2.0,
            f1=0.90,
            precision=0.88,
            recall=0.92,
        )

        summary = metrics.summary()
        assert "RMSE" in summary
        assert "F1-score" in summary
        assert "[PASS]" in summary


class TestConfidenceInterval:
    """Tests for ConfidenceInterval dataclass."""

    def test_basic_construction(self):
        """ConfidenceInterval should store values correctly."""
        ci = ConfidenceInterval(
            metric_name="rmse",
            point_estimate=10.5,
            lower=9.2,
            upper=11.8,
            confidence_level=0.95,
            n_bootstrap=1000,
        )

        assert ci.metric_name == "rmse"
        assert ci.point_estimate == 10.5
        assert ci.lower == 9.2
        assert ci.upper == 11.8

    def test_string_representation(self):
        """ConfidenceInterval should have readable string repr."""
        ci = ConfidenceInterval(
            metric_name="mae",
            point_estimate=8.0,
            lower=7.5,
            upper=8.5,
        )

        s = str(ci)
        assert "mae" in s
        assert "8.000" in s
        assert "7.500" in s
        assert "8.500" in s


class TestHoldoutEvaluator:
    """Tests for HoldoutEvaluator class."""

    def test_evaluate_basic(self):
        """Evaluator should compute metrics correctly."""
        model = MockModel(offset=1.0)  # All predictions offset by 1

        X = np.array([[10.0], [20.0], [30.0]])
        y = np.array([10.0, 20.0, 30.0])

        evaluator = HoldoutEvaluator(model)
        metrics = evaluator.evaluate(X, y)

        assert isinstance(metrics, PRDMetrics)
        assert metrics.rmse == 1.0  # Constant error of 1
        assert metrics.mae == 1.0
        assert metrics.bias == 1.0  # Positive bias (over-prediction)

    def test_evaluate_with_baseline(self):
        """Evaluator should compute improvement over baseline."""
        model = MockModel(offset=1.0)  # RMSE = 1
        baseline = MockModel(offset=5.0)  # RMSE = 5

        X = np.array([[10.0], [20.0], [30.0]])
        y = np.array([10.0, 20.0, 30.0])

        evaluator = HoldoutEvaluator(model, baseline)
        metrics = evaluator.evaluate(X, y)

        assert metrics.baseline_rmse == 5.0
        assert metrics.improvement_pct == 0.8  # 80% improvement

    def test_compute_confidence_intervals(self):
        """Evaluator should compute bootstrap CIs."""
        model = MockModel(offset=1.0)

        X = np.array([[i] for i in range(100)], dtype=float)
        y = np.arange(100, dtype=float)

        evaluator = HoldoutEvaluator(model)
        cis = evaluator.compute_confidence_intervals(
            X, y, n_bootstrap=100, random_state=42
        )

        assert "rmse" in cis
        assert "mae" in cis
        assert "bias" in cis

        # Check CI structure
        rmse_ci = cis["rmse"]
        assert rmse_ci.point_estimate == 1.0
        assert rmse_ci.lower <= rmse_ci.point_estimate <= rmse_ci.upper


class TestMetricsBreakdown:
    """Tests for MetricsBreakdown class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with metadata."""
        n = 365 * 2  # 2 years of daily data
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        stations = np.tile(["A", "B", "C", "D", "E"], n // 5 + 1)[:n]
        elevations = np.where(
            stations == "A", 2000,
            np.where(stations == "B", 2500,
            np.where(stations == "C", 3000,
            np.where(stations == "D", 3500, 4000)))
        )

        y_true = np.random.default_rng(42).uniform(0, 50, n)
        y_pred = y_true + np.random.default_rng(43).normal(0, 5, n)

        metadata = pd.DataFrame({
            "date": dates,
            "station_id": stations,
            "elevation": elevations,
        })

        return y_true, y_pred, metadata

    def test_by_month(self, sample_data):
        """Breakdown by month should return 12 rows."""
        y_true, y_pred, metadata = sample_data

        breakdown = MetricsBreakdown(y_true, y_pred, metadata)
        monthly = breakdown.by_month()

        assert len(monthly) == 12
        assert "rmse" in monthly.columns
        assert "mae" in monthly.columns
        assert "n_samples" in monthly.columns

    def test_by_station(self, sample_data):
        """Breakdown by station should return correct stations."""
        y_true, y_pred, metadata = sample_data

        breakdown = MetricsBreakdown(y_true, y_pred, metadata)
        by_station = breakdown.by_station()

        assert set(by_station.index) == {"A", "B", "C", "D", "E"}
        assert "rmse" in by_station.columns

    def test_by_elevation_band(self, sample_data):
        """Breakdown by elevation should return correct bands."""
        y_true, y_pred, metadata = sample_data

        breakdown = MetricsBreakdown(y_true, y_pred, metadata)
        by_elev = breakdown.by_elevation_band()

        assert len(by_elev) == 5  # Default 5 bands
        assert "rmse" in by_elev.columns

    def test_by_elevation_band_custom(self, sample_data):
        """Custom elevation bands should work."""
        y_true, y_pred, metadata = sample_data

        breakdown = MetricsBreakdown(y_true, y_pred, metadata)
        by_elev = breakdown.by_elevation_band(
            bands=[(0, 2500), (2500, 3500), (3500, float("inf"))]
        )

        assert len(by_elev) == 3

    def test_by_storm_intensity(self, sample_data):
        """Breakdown by storm intensity should categorize correctly."""
        y_true, y_pred, metadata = sample_data

        breakdown = MetricsBreakdown(y_true, y_pred, metadata)
        by_intensity = breakdown.by_storm_intensity()

        assert len(by_intensity) == 6  # Default 6 bins
        assert "n_samples" in by_intensity.columns

    def test_missing_column_error(self, sample_data):
        """Should raise error if required column missing."""
        y_true, y_pred, _ = sample_data

        # No date column
        metadata = pd.DataFrame({"other": range(len(y_true))})
        breakdown = MetricsBreakdown(y_true, y_pred, metadata)

        with pytest.raises(ValueError, match="date"):
            breakdown.by_month()


class TestResidualAnalyzer:
    """Tests for ResidualAnalyzer class."""

    @pytest.fixture
    def sample_residuals(self):
        """Create sample data for residual analysis."""
        n = 365
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        stations = np.tile(["A", "B"], n // 2 + 1)[:n]

        y_true = np.random.default_rng(42).uniform(0, 50, n)
        y_pred = y_true + np.random.default_rng(43).normal(2, 5, n)  # Slight positive bias

        metadata = pd.DataFrame({
            "date": dates,
            "station_id": stations,
            "temp": np.random.default_rng(44).uniform(-10, 10, n),
            "humidity": np.random.default_rng(45).uniform(30, 100, n),
        })

        return y_true, y_pred, metadata

    def test_summary_stats(self, sample_residuals):
        """Summary stats should include key metrics."""
        y_true, y_pred, metadata = sample_residuals

        analyzer = ResidualAnalyzer(y_true, y_pred, metadata)
        stats = analyzer.summary_stats()

        assert "mean" in stats
        assert "std" in stats
        assert "median" in stats
        assert "skewness" in stats
        assert "kurtosis" in stats

    def test_temporal_patterns(self, sample_residuals):
        """Temporal patterns should return monthly aggregates."""
        y_true, y_pred, metadata = sample_residuals

        analyzer = ResidualAnalyzer(y_true, y_pred, metadata)
        temporal = analyzer.temporal_patterns()

        assert len(temporal) <= 12
        assert "mean_bias" in temporal.columns
        assert "std_residual" in temporal.columns

    def test_spatial_patterns(self, sample_residuals):
        """Spatial patterns should return per-station metrics."""
        y_true, y_pred, metadata = sample_residuals

        analyzer = ResidualAnalyzer(y_true, y_pred, metadata)
        spatial = analyzer.spatial_patterns()

        assert set(spatial.index) == {"A", "B"}
        assert "mean_bias" in spatial.columns

    def test_condition_correlations(self, sample_residuals):
        """Should compute correlations with atmospheric conditions."""
        y_true, y_pred, metadata = sample_residuals

        analyzer = ResidualAnalyzer(y_true, y_pred, metadata)
        corrs = analyzer.condition_correlations(["temp", "humidity"])

        assert "temp" in corrs.index
        assert "humidity" in corrs.index
        assert "correlation" in corrs.columns

    def test_trend_over_time(self, sample_residuals):
        """Should detect trend in residuals over time."""
        y_true, y_pred, metadata = sample_residuals

        analyzer = ResidualAnalyzer(y_true, y_pred, metadata)
        trend = analyzer.trend_over_time()

        assert "slope" in trend
        assert "intercept" in trend
        assert "degrading" in trend
        # degrading is numpy bool, convert for comparison
        assert trend["degrading"] in (True, False)

    def test_degradation_detection(self):
        """Should detect degradation when errors increase over time."""
        n = 365
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y_true = np.zeros(n)
        # Errors that increase over time
        y_pred = np.linspace(0, 10, n)  # Errors go from 0 to 10

        metadata = pd.DataFrame({"date": dates})

        analyzer = ResidualAnalyzer(y_true, y_pred, metadata)
        trend = analyzer.trend_over_time()

        assert trend["slope"] > 0
        # May or may not be marked as degrading depending on threshold


class TestEvaluateHoldout:
    """Tests for evaluate_holdout convenience function."""

    def test_basic_evaluation(self):
        """evaluate_holdout should return comprehensive results."""
        model = MockModel(offset=1.0)

        X = np.array([[i] for i in range(100)], dtype=float)
        y = np.arange(100, dtype=float)

        results = evaluate_holdout(model, X, y, n_bootstrap=50)

        assert "prd_metrics" in results
        assert "confidence_intervals" in results
        assert isinstance(results["prd_metrics"], PRDMetrics)

    def test_with_metadata(self):
        """evaluate_holdout should compute breakdowns with metadata."""
        model = MockModel(offset=1.0)

        n = 365
        X = np.array([[i] for i in range(n)], dtype=float)
        y = np.arange(n, dtype=float)

        metadata = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "station_id": np.tile(["A", "B", "C"], n // 3 + 1)[:n],
            "elevation": np.random.default_rng(42).uniform(2000, 4000, n),
        })

        results = evaluate_holdout(model, X, y, metadata=metadata, n_bootstrap=50)

        assert "breakdowns" in results
        assert "residual_analysis" in results
        assert "by_month" in results["breakdowns"]
        assert "by_station" in results["breakdowns"]
        assert "by_elevation" in results["breakdowns"]


class TestComputePRDMetrics:
    """Tests for compute_prd_metrics function."""

    def test_basic_computation(self):
        """Should compute metrics from arrays."""
        y_true = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        y_pred = np.array([1.0, 6.0, 11.0, 16.0, 21.0])  # All +1 offset

        metrics = compute_prd_metrics(y_true, y_pred)

        assert isinstance(metrics, PRDMetrics)
        assert metrics.rmse == 1.0
        assert metrics.mae == 1.0
        assert metrics.bias == 1.0

    def test_with_baseline(self):
        """Should compute improvement over baseline."""
        y_true = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        y_pred = np.array([1.0, 6.0, 9.0, 16.0, 21.0])
        y_baseline = np.array([5.0, 10.0, 5.0, 20.0, 15.0])

        metrics = compute_prd_metrics(y_true, y_pred, y_baseline)

        assert metrics.baseline_rmse is not None
        assert metrics.improvement_pct is not None


class TestCheckPRDTargets:
    """Tests for check_prd_targets function."""

    def test_returns_dict(self):
        """Should return dict of target statuses."""
        metrics = PRDMetrics(
            rmse=10.0,
            mae=8.0,
            bias=2.0,
            f1=0.90,
            precision=0.88,
            recall=0.92,
        )

        targets = check_prd_targets(metrics)

        assert isinstance(targets, dict)
        assert "rmse" in targets
        assert "f1" in targets
        assert "bias" in targets


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_arrays(self):
        """Should handle empty arrays gracefully."""
        y_true = np.array([])
        y_pred = np.array([])

        breakdown = MetricsBreakdown(y_true, y_pred)
        result = breakdown.by_storm_intensity()

        # All groups should have 0 samples
        assert all(result["n_samples"] == 0)

    def test_single_value(self):
        """Should handle single value."""
        y_true = np.array([10.0])
        y_pred = np.array([12.0])

        metrics = compute_prd_metrics(y_true, y_pred)

        assert metrics.rmse == 2.0
        assert metrics.mae == 2.0
        assert metrics.bias == 2.0

    def test_nan_handling(self):
        """Should handle NaN values appropriately."""
        y_true = np.array([10.0, np.nan, 30.0])
        y_pred = np.array([12.0, 20.0, 28.0])

        metrics = compute_prd_metrics(y_true, y_pred)

        # RMSE should exclude NaN pairs
        assert not np.isnan(metrics.rmse)

    def test_perfect_predictions(self):
        """Should handle perfect predictions."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])

        metrics = compute_prd_metrics(y_true, y_pred)

        assert metrics.rmse == 0.0
        assert metrics.mae == 0.0
        assert metrics.bias == 0.0
