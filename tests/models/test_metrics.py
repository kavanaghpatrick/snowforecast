"""Tests for evaluation metrics.

Tests use real numpy calculations - NO MOCKS.
"""

import numpy as np
import pandas as pd
import pytest

from snowforecast.models.metrics import (
    rmse,
    mae,
    bias,
    f1_score_snowfall,
    precision_snowfall,
    recall_snowfall,
    compute_all_metrics,
    compute_metrics_by_group,
    SNOWFALL_EVENT_THRESHOLD_CM,
    METRICS,
)


class TestRMSE:
    """Tests for Root Mean Square Error."""

    def test_perfect_predictions(self):
        """RMSE should be 0 for perfect predictions."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        assert rmse(y_true, y_pred) == 0.0

    def test_known_values(self):
        """Test RMSE with known values."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 32.0])
        # Errors: 2, -2, 2 -> squared: 4, 4, 4 -> mean: 4 -> sqrt: 2
        assert rmse(y_true, y_pred) == 2.0

    def test_single_value(self):
        """RMSE should work with single values."""
        y_true = np.array([10.0])
        y_pred = np.array([15.0])
        assert rmse(y_true, y_pred) == 5.0

    def test_with_pandas_series(self):
        """RMSE should accept pandas Series."""
        y_true = pd.Series([10.0, 20.0, 30.0])
        y_pred = pd.Series([12.0, 18.0, 32.0])
        assert rmse(y_true, y_pred) == 2.0

    def test_handles_nan_values(self):
        """RMSE should exclude NaN pairs."""
        y_true = np.array([10.0, np.nan, 30.0])
        y_pred = np.array([12.0, 20.0, 32.0])
        # Only uses indices 0 and 2: errors 2, 2
        assert rmse(y_true, y_pred) == 2.0

    def test_all_nan_returns_nan(self):
        """RMSE should return NaN if all values are NaN."""
        y_true = np.array([np.nan, np.nan])
        y_pred = np.array([1.0, 2.0])
        assert np.isnan(rmse(y_true, y_pred))


class TestMAE:
    """Tests for Mean Absolute Error."""

    def test_perfect_predictions(self):
        """MAE should be 0 for perfect predictions."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        assert mae(y_true, y_pred) == 0.0

    def test_known_values(self):
        """Test MAE with known values."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 32.0])
        # Errors: |2|, |-2|, |2| -> mean: 2
        assert mae(y_true, y_pred) == 2.0

    def test_with_negative_errors(self):
        """MAE should use absolute values."""
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([15.0, 10.0])
        # Errors: 5, -10 -> abs: 5, 10 -> mean: 7.5
        assert mae(y_true, y_pred) == 7.5

    def test_handles_nan(self):
        """MAE should exclude NaN pairs."""
        y_true = np.array([10.0, np.nan, 30.0])
        y_pred = np.array([12.0, 20.0, 28.0])
        # Only indices 0, 2: |2|, |2| -> mean: 2
        assert mae(y_true, y_pred) == 2.0


class TestBias:
    """Tests for Mean Error (Bias)."""

    def test_no_bias(self):
        """Bias should be 0 for perfect predictions."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        assert bias(y_true, y_pred) == 0.0

    def test_positive_bias(self):
        """Positive bias indicates over-prediction."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 22.0, 32.0])
        # All over-predicting by 2
        assert bias(y_true, y_pred) == 2.0

    def test_negative_bias(self):
        """Negative bias indicates under-prediction."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([8.0, 18.0, 28.0])
        # All under-predicting by 2
        assert bias(y_true, y_pred) == -2.0

    def test_mixed_errors_cancel(self):
        """Positive and negative errors should cancel."""
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([15.0, 15.0])
        # Errors: +5, -5 -> mean: 0
        assert bias(y_true, y_pred) == 0.0


class TestF1ScoreSnowfall:
    """Tests for F1-score of snowfall events."""

    def test_perfect_classification(self):
        """F1 should be 1.0 for perfect classification."""
        y_true = np.array([0.0, 5.0, 10.0, 0.0])  # Events at index 1, 2
        y_pred = np.array([0.0, 5.0, 10.0, 0.0])  # Same events
        assert f1_score_snowfall(y_true, y_pred) == 1.0

    def test_no_true_positives(self):
        """F1 should be 0 when no true positives."""
        y_true = np.array([0.0, 5.0, 10.0])  # Events at 1, 2
        y_pred = np.array([0.0, 0.0, 0.0])   # No predictions
        assert f1_score_snowfall(y_true, y_pred) == 0.0

    def test_custom_threshold(self):
        """F1 should respect custom threshold."""
        y_true = np.array([0.0, 3.0, 5.0])
        y_pred = np.array([0.0, 3.0, 5.0])

        # With default 2.5cm threshold: 2 events
        f1_default = f1_score_snowfall(y_true, y_pred, threshold_cm=2.5)

        # With 4cm threshold: only 1 event
        f1_higher = f1_score_snowfall(y_true, y_pred, threshold_cm=4.0)

        assert f1_default == 1.0
        assert f1_higher == 1.0  # Still perfect classification

    def test_no_positive_cases_returns_nan(self):
        """F1 should return NaN if no actual positive cases."""
        y_true = np.array([0.0, 1.0, 2.0])  # All below threshold
        y_pred = np.array([3.0, 3.0, 3.0])  # All above threshold
        assert np.isnan(f1_score_snowfall(y_true, y_pred))

    def test_known_f1_calculation(self):
        """Test F1 with known precision/recall."""
        # True events: indices 1, 2 (5.0 and 10.0 > 2.5 threshold)
        # Predicted events: indices 0, 1, 2, 3 (all > 2.5)
        # TP = 2 (indices 1, 2), FP = 2 (indices 0, 3), FN = 0
        y_true = np.array([0.0, 5.0, 10.0, 1.0])
        y_pred = np.array([3.0, 6.0, 8.0, 3.0])
        # Precision = TP/(TP+FP) = 2/4 = 0.5
        # Recall = TP/(TP+FN) = 2/2 = 1.0
        # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 1.0 / 1.5 = 0.6667
        assert abs(f1_score_snowfall(y_true, y_pred) - 0.6667) < 0.01


class TestPrecisionRecall:
    """Tests for precision and recall."""

    def test_perfect_precision(self):
        """Precision should be 1.0 when all predictions correct."""
        y_true = np.array([5.0, 10.0, 0.0])  # 2 events
        y_pred = np.array([5.0, 10.0, 0.0])  # 2 correct predictions
        assert precision_snowfall(y_true, y_pred) == 1.0

    def test_perfect_recall(self):
        """Recall should be 1.0 when all events caught."""
        y_true = np.array([5.0, 10.0, 0.0])
        y_pred = np.array([5.0, 10.0, 0.0])
        assert recall_snowfall(y_true, y_pred) == 1.0

    def test_low_precision_high_recall(self):
        """Test case with many false positives."""
        y_true = np.array([5.0, 0.0, 0.0, 0.0])  # 1 event
        y_pred = np.array([5.0, 5.0, 5.0, 5.0])  # 4 predictions
        # Precision = 1/4, Recall = 1/1
        assert precision_snowfall(y_true, y_pred) == 0.25
        assert recall_snowfall(y_true, y_pred) == 1.0

    def test_high_precision_low_recall(self):
        """Test case with many false negatives."""
        y_true = np.array([5.0, 10.0, 15.0, 20.0])  # 4 events
        y_pred = np.array([5.0, 0.0, 0.0, 0.0])     # 1 prediction
        # Precision = 1/1, Recall = 1/4
        assert precision_snowfall(y_true, y_pred) == 1.0
        assert recall_snowfall(y_true, y_pred) == 0.25


class TestComputeAllMetrics:
    """Tests for compute_all_metrics convenience function."""

    def test_returns_all_metrics(self):
        """Should return all standard metrics."""
        y_true = np.array([0.0, 5.0, 10.0, 15.0])
        y_pred = np.array([1.0, 6.0, 9.0, 14.0])

        metrics = compute_all_metrics(y_true, y_pred)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "bias" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_metrics_are_consistent(self):
        """Individual metrics should match."""
        y_true = np.array([0.0, 5.0, 10.0, 15.0])
        y_pred = np.array([1.0, 6.0, 9.0, 14.0])

        metrics = compute_all_metrics(y_true, y_pred)

        assert metrics["rmse"] == rmse(y_true, y_pred)
        assert metrics["mae"] == mae(y_true, y_pred)
        assert metrics["bias"] == bias(y_true, y_pred)
        assert metrics["f1"] == f1_score_snowfall(y_true, y_pred)


class TestComputeMetricsByGroup:
    """Tests for grouped metric computation."""

    def test_basic_grouping(self):
        """Should compute metrics for each group."""
        df = pd.DataFrame({
            "station_id": ["A", "A", "B", "B"],
            "y_true": [10.0, 20.0, 15.0, 25.0],
            "y_pred": [12.0, 18.0, 14.0, 26.0],
        })

        result = compute_metrics_by_group(df, "station_id")

        assert len(result) == 2
        assert set(result["station_id"]) == {"A", "B"}
        assert "rmse" in result.columns
        assert "n_samples" in result.columns

    def test_sample_counts(self):
        """Should include correct sample counts."""
        df = pd.DataFrame({
            "station_id": ["A", "A", "A", "B", "B"],
            "y_true": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y_pred": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        result = compute_metrics_by_group(df, "station_id")

        a_row = result[result["station_id"] == "A"]
        b_row = result[result["station_id"] == "B"]
        assert a_row["n_samples"].values[0] == 3
        assert b_row["n_samples"].values[0] == 2


class TestMetricsRegistry:
    """Tests for the METRICS registry."""

    def test_all_metrics_registered(self):
        """All main metrics should be in registry."""
        assert "rmse" in METRICS
        assert "mae" in METRICS
        assert "bias" in METRICS
        assert "f1" in METRICS
        assert "precision" in METRICS
        assert "recall" in METRICS

    def test_registry_functions_callable(self):
        """Registry functions should be callable."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        for name, func in METRICS.items():
            result = func(y_true, y_pred)
            assert isinstance(result, (float, np.floating))


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_empty_arrays(self):
        """Metrics should handle empty arrays gracefully."""
        y_true = np.array([])
        y_pred = np.array([])

        # These may raise or return NaN depending on implementation
        # Just ensure no crash
        try:
            result = rmse(y_true, y_pred)
            assert np.isnan(result) or result >= 0
        except (ValueError, ZeroDivisionError):
            pass  # Also acceptable

    def test_large_values(self):
        """Metrics should handle large values."""
        y_true = np.array([1e10, 2e10, 3e10])
        y_pred = np.array([1e10 + 100, 2e10 + 100, 3e10 + 100])

        result = rmse(y_true, y_pred)
        assert abs(result - 100.0) < 1e-6

    def test_snowfall_threshold_constant(self):
        """Default threshold should be 2.5cm."""
        assert SNOWFALL_EVENT_THRESHOLD_CM == 2.5
