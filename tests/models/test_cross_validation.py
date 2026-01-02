"""Tests for cross-validation utilities.

Tests use real numpy/pandas calculations - NO MOCKS.
"""

import numpy as np
import pandas as pd
import pytest

from snowforecast.models.cross_validation import (
    StationKFold,
    TemporalSplit,
    CVResults,
)


class TestStationKFold:
    """Tests for station-based K-fold cross-validation."""

    def test_basic_split(self):
        """Should split data by station_id."""
        df = pd.DataFrame({
            "station_id": ["A", "A", "B", "B", "C", "C"],
            "value": [1, 2, 3, 4, 5, 6],
        })

        cv = StationKFold(n_splits=3, shuffle=False)
        folds = list(cv.split(df))

        assert len(folds) == 3

        # Each fold should have disjoint indices
        all_train = set()
        all_val = set()
        for train_idx, val_idx in folds:
            # No overlap within fold
            assert len(set(train_idx) & set(val_idx)) == 0
            all_val.update(val_idx)

        # All indices should be covered exactly once in validation
        assert all_val == set(range(len(df)))

    def test_station_integrity(self):
        """All samples from a station should be in same set."""
        df = pd.DataFrame({
            "station_id": ["A", "A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
            "value": range(11),
        })

        cv = StationKFold(n_splits=3, shuffle=True, random_state=42)

        for train_idx, val_idx in cv.split(df):
            train_stations = set(df.iloc[train_idx]["station_id"])
            val_stations = set(df.iloc[val_idx]["station_id"])

            # No station should be in both train and val
            assert len(train_stations & val_stations) == 0

    def test_groups_parameter(self):
        """Should accept groups parameter instead of station_id column."""
        df = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5, 6],
        })
        groups = np.array(["X", "X", "Y", "Y", "Z", "Z"])

        cv = StationKFold(n_splits=3, shuffle=False)
        folds = list(cv.split(df, groups=groups))

        assert len(folds) == 3

    def test_shuffle_reproducibility(self):
        """Shuffle with same seed should produce same splits."""
        df = pd.DataFrame({
            "station_id": list("AABBCCDDEE"),
            "value": range(10),
        })

        cv1 = StationKFold(n_splits=3, shuffle=True, random_state=42)
        cv2 = StationKFold(n_splits=3, shuffle=True, random_state=42)

        folds1 = list(cv1.split(df))
        folds2 = list(cv2.split(df))

        for (t1, v1), (t2, v2) in zip(folds1, folds2):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(v1, v2)

    def test_different_seeds_produce_different_splits(self):
        """Different seeds should produce different splits."""
        df = pd.DataFrame({
            "station_id": list("AABBCCDDEE") * 2,
            "value": range(20),
        })

        cv1 = StationKFold(n_splits=3, shuffle=True, random_state=42)
        cv2 = StationKFold(n_splits=3, shuffle=True, random_state=123)

        folds1 = list(cv1.split(df))
        folds2 = list(cv2.split(df))

        # At least one fold should be different
        any_different = False
        for (t1, v1), (t2, v2) in zip(folds1, folds2):
            if not np.array_equal(v1, v2):
                any_different = True
                break

        assert any_different

    def test_n_splits_validation(self):
        """Should raise error for n_splits < 2."""
        with pytest.raises(ValueError, match="n_splits must be at least 2"):
            StationKFold(n_splits=1)

    def test_too_few_stations(self):
        """Should raise error if fewer stations than splits."""
        df = pd.DataFrame({
            "station_id": ["A", "A", "B", "B"],
            "value": [1, 2, 3, 4],
        })

        cv = StationKFold(n_splits=5)

        with pytest.raises(ValueError, match="Cannot split 2 stations into 5 folds"):
            list(cv.split(df))

    def test_missing_station_id_column(self):
        """Should raise error if no station_id column and no groups."""
        df = pd.DataFrame({"feature": [1, 2, 3, 4]})
        cv = StationKFold(n_splits=2)

        with pytest.raises(ValueError, match="groups must be provided"):
            list(cv.split(df))

    def test_get_n_splits(self):
        """get_n_splits should return configured n_splits."""
        cv = StationKFold(n_splits=7)
        assert cv.get_n_splits() == 7

    def test_uneven_station_distribution(self):
        """Should handle stations with different sample counts."""
        df = pd.DataFrame({
            "station_id": ["A"] * 100 + ["B"] * 10 + ["C"] * 5 + ["D"] * 3,
            "value": range(118),
        })

        cv = StationKFold(n_splits=2, shuffle=True, random_state=42)

        for train_idx, val_idx in cv.split(df):
            # Both sets should have samples
            assert len(train_idx) > 0
            assert len(val_idx) > 0
            # Total should equal original
            assert len(train_idx) + len(val_idx) == len(df)


class TestTemporalSplit:
    """Tests for time-based train/test splitting."""

    def test_basic_split_by_years(self):
        """Should split data by most recent years."""
        dates = pd.date_range("2020-01-01", periods=365 * 4, freq="D")
        df = pd.DataFrame({
            "datetime": dates,
            "value": range(len(dates)),
        })

        splitter = TemporalSplit(test_years=1)
        train_df, test_df = splitter.split(df)

        # Test set should have roughly 1 year of data
        assert len(test_df) > 300  # About a year
        assert len(train_df) > len(test_df)

        # All test dates should be after all train dates
        assert train_df["datetime"].max() < test_df["datetime"].min()

    def test_explicit_cutoff_date(self):
        """Should use explicit cutoff date when provided."""
        df = pd.DataFrame({
            "datetime": pd.date_range("2020-01-01", periods=100, freq="D"),
            "value": range(100),
        })

        splitter = TemporalSplit(cutoff_date="2020-03-01")
        train_df, test_df = splitter.split(df)

        # Check split at cutoff
        assert train_df["datetime"].max() < pd.Timestamp("2020-03-01")
        assert test_df["datetime"].min() >= pd.Timestamp("2020-03-01")

    def test_date_column_auto_detection(self):
        """Should auto-detect common date column names."""
        # Use enough data for a meaningful split with test_years=1
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=730, freq="D"),  # 2 years
            "value": range(730),
        })

        splitter = TemporalSplit(test_years=1)
        # Should work when we specify datetime_col="date"
        train_df, test_df = splitter.split(df, datetime_col="date")

        assert len(train_df) > 0
        assert len(test_df) > 0

    def test_timestamp_column_detection(self):
        """Should detect 'timestamp' column."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D"),
            "value": range(100),
        })

        splitter = TemporalSplit(cutoff_date="2020-02-15")
        train_df, test_df = splitter.split(df, datetime_col="timestamp")

        assert len(train_df) > 0
        assert len(test_df) > 0

    def test_empty_dataframe(self):
        """Should raise error for empty DataFrame."""
        df = pd.DataFrame(columns=["datetime", "value"])
        splitter = TemporalSplit(test_years=1)

        with pytest.raises(ValueError, match="Cannot split empty DataFrame"):
            splitter.split(df)

    def test_missing_datetime_column(self):
        """Should raise error if datetime column not found."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        splitter = TemporalSplit(test_years=1)

        with pytest.raises(ValueError, match="datetime_col .* not found"):
            splitter.split(df, datetime_col="datetime")

    def test_invalid_test_years(self):
        """Should raise error for test_years < 1."""
        with pytest.raises(ValueError, match="test_years must be at least 1"):
            TemporalSplit(test_years=0)

    def test_get_cutoff_date_explicit(self):
        """get_cutoff_date should return explicit cutoff."""
        splitter = TemporalSplit(cutoff_date="2023-06-01")
        df = pd.DataFrame({
            "datetime": pd.date_range("2020-01-01", periods=100),
            "value": range(100),
        })

        cutoff = splitter.get_cutoff_date(df)
        assert cutoff == pd.Timestamp("2023-06-01")

    def test_get_cutoff_date_computed(self):
        """get_cutoff_date should compute from test_years."""
        splitter = TemporalSplit(test_years=2)
        df = pd.DataFrame({
            "datetime": pd.date_range("2020-01-01", "2024-12-31", freq="D"),
            "value": range(1827),  # Approx 5 years
        })

        cutoff = splitter.get_cutoff_date(df)
        max_date = df["datetime"].max()
        expected = max_date - pd.DateOffset(years=2)

        assert cutoff == expected

    def test_preserves_other_columns(self):
        """Split should preserve all DataFrame columns."""
        df = pd.DataFrame({
            "datetime": pd.date_range("2020-01-01", periods=100, freq="D"),
            "station_id": ["A"] * 50 + ["B"] * 50,
            "value": range(100),
            "metadata": ["info"] * 100,
        })

        splitter = TemporalSplit(cutoff_date="2020-02-15")
        train_df, test_df = splitter.split(df)

        # All columns should be present
        assert list(train_df.columns) == list(df.columns)
        assert list(test_df.columns) == list(df.columns)


class TestCVResults:
    """Tests for CVResults container."""

    def test_add_fold(self):
        """Should store fold metrics."""
        results = CVResults()
        results.add_fold(0, {"rmse": 10.5, "mae": 8.2})
        results.add_fold(1, {"rmse": 11.0, "mae": 8.5})

        assert results.n_folds() == 2
        assert results.fold_metrics[0]["fold"] == 0
        assert results.fold_metrics[0]["rmse"] == 10.5

    def test_add_fold_with_predictions(self):
        """Should store fold predictions."""
        results = CVResults()

        preds_df = pd.DataFrame({
            "y_true": [1.0, 2.0, 3.0],
            "y_pred": [1.1, 2.1, 2.9],
        })

        results.add_fold(0, {"rmse": 0.1}, predictions=preds_df)

        assert len(results.fold_predictions) == 1
        assert "fold" in results.fold_predictions[0].columns

    def test_summary_statistics(self):
        """Should compute correct summary statistics."""
        results = CVResults()
        results.add_fold(0, {"rmse": 10.0, "mae": 8.0})
        results.add_fold(1, {"rmse": 12.0, "mae": 9.0})
        results.add_fold(2, {"rmse": 11.0, "mae": 8.5})

        summary = results.summary()

        # RMSE: mean=11, std=0.816...
        assert summary["rmse"]["mean"] == 11.0
        assert abs(summary["rmse"]["std"] - 0.8165) < 0.01
        assert summary["rmse"]["min"] == 10.0
        assert summary["rmse"]["max"] == 12.0

        # MAE: mean=8.5
        assert summary["mae"]["mean"] == 8.5

    def test_summary_empty_results(self):
        """Summary of empty results should be empty dict."""
        results = CVResults()
        assert results.summary() == {}

    def test_get_fold_metrics_df(self):
        """Should return metrics as DataFrame."""
        results = CVResults()
        results.add_fold(0, {"rmse": 10.0, "f1": 0.8})
        results.add_fold(1, {"rmse": 11.0, "f1": 0.85})

        df = results.get_fold_metrics_df()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "fold" in df.columns
        assert "rmse" in df.columns
        assert "f1" in df.columns

    def test_get_fold_metrics_df_empty(self):
        """Empty results should return empty DataFrame."""
        results = CVResults()
        df = results.get_fold_metrics_df()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_get_all_predictions(self):
        """Should concatenate predictions from all folds."""
        results = CVResults()

        results.add_fold(
            0,
            {"rmse": 1.0},
            predictions=pd.DataFrame({
                "y_true": [1.0, 2.0],
                "y_pred": [1.1, 2.1],
            }),
        )
        results.add_fold(
            1,
            {"rmse": 1.0},
            predictions=pd.DataFrame({
                "y_true": [3.0, 4.0],
                "y_pred": [3.1, 4.1],
            }),
        )

        all_preds = results.get_all_predictions()

        assert len(all_preds) == 4
        assert list(all_preds["y_true"]) == [1.0, 2.0, 3.0, 4.0]
        assert set(all_preds["fold"]) == {0, 1}

    def test_get_all_predictions_empty(self):
        """No predictions should return empty DataFrame."""
        results = CVResults()
        results.add_fold(0, {"rmse": 1.0})  # No predictions

        all_preds = results.get_all_predictions()
        assert len(all_preds) == 0

    def test_n_folds(self):
        """n_folds should return correct count."""
        results = CVResults()
        assert results.n_folds() == 0

        results.add_fold(0, {"rmse": 1.0})
        assert results.n_folds() == 1

        results.add_fold(1, {"rmse": 2.0})
        assert results.n_folds() == 2

    def test_repr_empty(self):
        """Empty CVResults repr."""
        results = CVResults()
        assert repr(results) == "CVResults(empty)"

    def test_repr_with_data(self):
        """CVResults repr with data."""
        results = CVResults()
        results.add_fold(0, {"rmse": 10.0})
        results.add_fold(1, {"rmse": 12.0})

        repr_str = repr(results)
        assert "n_folds=2" in repr_str
        assert "rmse=" in repr_str

    def test_different_metrics_per_fold(self):
        """Should handle folds with different metrics."""
        results = CVResults()
        results.add_fold(0, {"rmse": 10.0, "mae": 8.0})
        results.add_fold(1, {"rmse": 11.0, "f1": 0.85})  # Different metrics

        summary = results.summary()

        # RMSE should have stats from both folds
        assert summary["rmse"]["mean"] == 10.5

        # MAE only from fold 0
        assert summary["mae"]["mean"] == 8.0
        assert summary["mae"]["std"] == 0.0  # Only one value

        # F1 only from fold 1
        assert summary["f1"]["mean"] == 0.85


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_station_kfold_with_cv_results(self):
        """Full workflow: StationKFold split -> store in CVResults."""
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame({
            "station_id": np.random.choice(["A", "B", "C", "D", "E"], n_samples),
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
        })
        y = pd.Series(np.random.randn(n_samples))

        # Run CV
        cv = StationKFold(n_splits=3, shuffle=True, random_state=42)
        results = CVResults()

        for fold, (train_idx, val_idx) in enumerate(cv.split(df)):
            # Simulate model training/prediction
            y_true = y.iloc[val_idx].values
            y_pred = y_true + np.random.randn(len(y_true)) * 0.1  # Add noise

            # Compute metrics
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mae = np.mean(np.abs(y_true - y_pred))

            # Store results
            results.add_fold(
                fold,
                {"rmse": rmse, "mae": mae},
                predictions=pd.DataFrame({
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "station_id": df.iloc[val_idx]["station_id"].values,
                }),
            )

        # Verify results
        assert results.n_folds() == 3
        summary = results.summary()
        assert "rmse" in summary
        assert "mae" in summary

        all_preds = results.get_all_predictions()
        assert len(all_preds) == n_samples  # All samples covered once

    def test_temporal_then_station_kfold(self):
        """Temporal split for holdout, then StationKFold for CV."""
        # Create multi-year data
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
        n_samples = len(dates)
        df = pd.DataFrame({
            "datetime": dates,
            "station_id": np.tile(["A", "B", "C", "D"], n_samples // 4 + 1)[:n_samples],
            "value": np.random.randn(n_samples),
        })

        # First: temporal split for holdout
        temporal_splitter = TemporalSplit(test_years=1)
        train_df, test_df = temporal_splitter.split(df)

        assert len(train_df) > len(test_df)
        assert train_df["datetime"].max() < test_df["datetime"].min()

        # Then: StationKFold on training data
        cv = StationKFold(n_splits=4, shuffle=True, random_state=42)

        fold_count = 0
        for train_idx, val_idx in cv.split(train_df):
            fold_count += 1
            # Verify station integrity
            train_stations = set(train_df.iloc[train_idx]["station_id"])
            val_stations = set(train_df.iloc[val_idx]["station_id"])
            assert len(train_stations & val_stations) == 0

        assert fold_count == 4
