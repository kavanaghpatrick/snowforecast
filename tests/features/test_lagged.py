"""Tests for lagged and rolling window feature engineering."""

import numpy as np
import pandas as pd
import pytest

from snowforecast.features.lagged import LaggedFeatures


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame with two stations and 10 days of data."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")

    # Station 1: temperatures from 0 to 9, precip from 1 to 10
    station1 = pd.DataFrame({
        "date": dates,
        "station_id": "station_1",
        "temperature": list(range(10)),  # 0, 1, 2, ..., 9
        "precip": list(range(1, 11)),    # 1, 2, 3, ..., 10
    })

    # Station 2: different values
    station2 = pd.DataFrame({
        "date": dates,
        "station_id": "station_2",
        "temperature": list(range(10, 20)),  # 10, 11, ..., 19
        "precip": list(range(5, 15)),        # 5, 6, 7, ..., 14
    })

    return pd.concat([station1, station2], ignore_index=True)


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Create a simple single-station DataFrame for basic tests."""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=7, freq="D"),
        "station_id": "station_1",
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    })


@pytest.fixture
def lagged_features() -> LaggedFeatures:
    """Create a LaggedFeatures instance with default settings."""
    return LaggedFeatures()


class TestLaggedFeaturesInit:
    """Tests for LaggedFeatures initialization."""

    def test_default_initialization(self):
        """Test default lag and window values."""
        lf = LaggedFeatures()
        assert lf.default_lags == [1, 2, 3, 7]
        assert lf.default_windows == [3, 7, 14]

    def test_custom_lags(self):
        """Test custom lag values."""
        lf = LaggedFeatures(default_lags=[1, 3, 5])
        assert lf.default_lags == [1, 3, 5]
        assert lf.default_windows == [3, 7, 14]  # Default windows unchanged

    def test_custom_windows(self):
        """Test custom window values."""
        lf = LaggedFeatures(default_windows=[7, 14, 30])
        assert lf.default_lags == [1, 2, 3, 7]  # Default lags unchanged
        assert lf.default_windows == [7, 14, 30]

    def test_custom_both(self):
        """Test custom lags and windows."""
        lf = LaggedFeatures(default_lags=[1, 7], default_windows=[7, 14])
        assert lf.default_lags == [1, 7]
        assert lf.default_windows == [7, 14]


class TestCreateLags:
    """Tests for create_lags method."""

    def test_basic_lag_creation(self, simple_df, lagged_features):
        """Test creating basic lag features."""
        result = lagged_features.create_lags(simple_df, ["value"], lags=[1, 2])

        assert "value_lag_1d" in result.columns
        assert "value_lag_2d" in result.columns

        # First value should be NaN for lag 1
        assert pd.isna(result["value_lag_1d"].iloc[0])
        # Second value should be the first value
        assert result["value_lag_1d"].iloc[1] == 1.0
        # First two values should be NaN for lag 2
        assert pd.isna(result["value_lag_2d"].iloc[0])
        assert pd.isna(result["value_lag_2d"].iloc[1])
        assert result["value_lag_2d"].iloc[2] == 1.0

    def test_lag_respects_groups(self, sample_df, lagged_features):
        """Test that lags don't leak between stations."""
        result = lagged_features.create_lags(sample_df, ["temperature"], lags=[1])

        # Check station 1 - first row should be NaN
        station1 = result[result["station_id"] == "station_1"]
        assert pd.isna(station1["temperature_lag_1d"].iloc[0])
        assert station1["temperature_lag_1d"].iloc[1] == 0.0  # Lagged from first value

        # Check station 2 - first row should also be NaN (not leaking from station 1)
        station2 = result[result["station_id"] == "station_2"]
        assert pd.isna(station2["temperature_lag_1d"].iloc[0])
        assert station2["temperature_lag_1d"].iloc[1] == 10.0  # Lagged from station 2's first

    def test_lag_multiple_variables(self, sample_df, lagged_features):
        """Test creating lags for multiple variables."""
        result = lagged_features.create_lags(
            sample_df, ["temperature", "precip"], lags=[1]
        )

        assert "temperature_lag_1d" in result.columns
        assert "precip_lag_1d" in result.columns

    def test_lag_missing_variable_raises(self, sample_df, lagged_features):
        """Test that missing variable raises ValueError."""
        with pytest.raises(ValueError, match="Variables not found"):
            lagged_features.create_lags(sample_df, ["nonexistent"])

    def test_lag_uses_default_lags(self, simple_df, lagged_features):
        """Test that default lags are used when not specified."""
        result = lagged_features.create_lags(simple_df, ["value"])

        for lag in lagged_features.default_lags:
            assert f"value_lag_{lag}d" in result.columns


class TestCreateRollingMean:
    """Tests for create_rolling_mean method."""

    def test_basic_rolling_mean(self, simple_df, lagged_features):
        """Test creating basic rolling mean features."""
        result = lagged_features.create_rolling_mean(simple_df, ["value"], windows=[3])

        assert "value_roll_mean_3d" in result.columns

        # First value should equal itself (min_periods=1)
        assert result["value_roll_mean_3d"].iloc[0] == 1.0
        # Third value should be mean of [1, 2, 3]
        assert result["value_roll_mean_3d"].iloc[2] == 2.0

    def test_rolling_mean_respects_groups(self, sample_df, lagged_features):
        """Test that rolling mean doesn't leak between stations."""
        result = lagged_features.create_rolling_mean(
            sample_df, ["temperature"], windows=[3]
        )

        # Station 1's rolling mean shouldn't include station 2's values
        station1 = result[result["station_id"] == "station_1"]
        # Third value: mean of [0, 1, 2] = 1.0
        assert station1["temperature_roll_mean_3d"].iloc[2] == 1.0

        # Station 2's first value should just be itself
        station2 = result[result["station_id"] == "station_2"]
        assert station2["temperature_roll_mean_3d"].iloc[0] == 10.0


class TestCreateRollingStd:
    """Tests for create_rolling_std method."""

    def test_basic_rolling_std(self, simple_df, lagged_features):
        """Test creating basic rolling std features."""
        result = lagged_features.create_rolling_std(simple_df, ["value"], windows=[3])

        assert "value_roll_std_3d" in result.columns

        # First value should be NaN (need min_periods=2)
        assert pd.isna(result["value_roll_std_3d"].iloc[0])
        # Third value should be std of [1, 2, 3]
        expected_std = pd.Series([1.0, 2.0, 3.0]).std()
        assert abs(result["value_roll_std_3d"].iloc[2] - expected_std) < 0.001

    def test_rolling_std_respects_groups(self, sample_df, lagged_features):
        """Test that rolling std doesn't leak between stations."""
        result = lagged_features.create_rolling_std(
            sample_df, ["temperature"], windows=[3]
        )

        station2 = result[result["station_id"] == "station_2"]
        # Station 2's first value should be NaN
        assert pd.isna(station2["temperature_roll_std_3d"].iloc[0])


class TestCreateRollingMinMax:
    """Tests for create_rolling_min_max method."""

    def test_basic_rolling_min_max(self, simple_df, lagged_features):
        """Test creating basic rolling min/max features."""
        result = lagged_features.create_rolling_min_max(
            simple_df, ["value"], windows=[3]
        )

        assert "value_roll_min_3d" in result.columns
        assert "value_roll_max_3d" in result.columns

        # At index 2, window is [1, 2, 3]
        assert result["value_roll_min_3d"].iloc[2] == 1.0
        assert result["value_roll_max_3d"].iloc[2] == 3.0

    def test_rolling_min_max_respects_groups(self, sample_df, lagged_features):
        """Test that rolling min/max doesn't leak between stations."""
        result = lagged_features.create_rolling_min_max(
            sample_df, ["temperature"], windows=[3]
        )

        station2 = result[result["station_id"] == "station_2"]
        # Station 2's third value: min/max of [10, 11, 12]
        assert station2["temperature_roll_min_3d"].iloc[2] == 10.0
        assert station2["temperature_roll_max_3d"].iloc[2] == 12.0


class TestCreateTrendFeatures:
    """Tests for create_trend_features method."""

    def test_basic_trend_features(self, simple_df, lagged_features):
        """Test creating basic trend features."""
        result = lagged_features.create_trend_features(
            simple_df, ["value"], periods=[1, 2]
        )

        assert "value_change_1d" in result.columns
        assert "value_pct_change_1d" in result.columns
        assert "value_change_2d" in result.columns

        # Change from index 0 to 1: 2 - 1 = 1
        assert result["value_change_1d"].iloc[1] == 1.0
        # Percent change: (2-1)/1 * 100 = 100%
        assert result["value_pct_change_1d"].iloc[1] == 100.0

    def test_trend_respects_groups(self, sample_df, lagged_features):
        """Test that trends don't leak between stations."""
        result = lagged_features.create_trend_features(
            sample_df, ["temperature"], periods=[1]
        )

        # Station 2's first value should have NaN change (not change from station 1)
        station2 = result[result["station_id"] == "station_2"]
        assert pd.isna(station2["temperature_change_1d"].iloc[0])
        # Station 2's second value: 11 - 10 = 1
        assert station2["temperature_change_1d"].iloc[1] == 1.0

    def test_trend_handles_zero_division(self, lagged_features):
        """Test that percent change handles division by zero."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "station_id": "station_1",
            "value": [0.0, 1.0, 2.0],
        })

        result = lagged_features.create_trend_features(df, ["value"], periods=[1])

        # Percent change from 0 should be NaN
        assert pd.isna(result["value_pct_change_1d"].iloc[1])


class TestCreateCumulativeFeatures:
    """Tests for create_cumulative_features method."""

    def test_basic_cumulative_sum(self, simple_df, lagged_features):
        """Test creating basic cumulative sum features."""
        result = lagged_features.create_cumulative_features(
            simple_df, ["value"], windows=[3]
        )

        assert "value_cumsum_3d" in result.columns

        # At index 2, cumsum should be 1 + 2 + 3 = 6
        assert result["value_cumsum_3d"].iloc[2] == 6.0

    def test_cumulative_respects_groups(self, sample_df, lagged_features):
        """Test that cumulative sums don't leak between stations."""
        result = lagged_features.create_cumulative_features(
            sample_df, ["precip"], windows=[3]
        )

        # Station 1's third value: sum of [1, 2, 3] = 6
        station1 = result[result["station_id"] == "station_1"]
        assert station1["precip_cumsum_3d"].iloc[2] == 6.0

        # Station 2's third value: sum of [5, 6, 7] = 18
        station2 = result[result["station_id"] == "station_2"]
        assert station2["precip_cumsum_3d"].iloc[2] == 18.0


class TestComputeAll:
    """Tests for compute_all method."""

    def test_compute_all_creates_all_features(self, simple_df, lagged_features):
        """Test that compute_all creates all feature types."""
        result = lagged_features.compute_all(
            simple_df,
            variables=["value"],
            lags=[1],
            windows=[3],
        )

        # Check all feature types are created
        assert "value_lag_1d" in result.columns
        assert "value_roll_mean_3d" in result.columns
        assert "value_roll_std_3d" in result.columns
        assert "value_roll_min_3d" in result.columns
        assert "value_roll_max_3d" in result.columns
        assert "value_change_1d" in result.columns
        assert "value_pct_change_1d" in result.columns
        assert "value_cumsum_3d" in result.columns

    def test_compute_all_uses_defaults(self, simple_df, lagged_features):
        """Test that compute_all uses default lags and windows."""
        result = lagged_features.compute_all(simple_df, variables=["value"])

        # Check features for all default lags
        for lag in lagged_features.default_lags:
            assert f"value_lag_{lag}d" in result.columns

        # Check features for all default windows
        for window in lagged_features.default_windows:
            assert f"value_roll_mean_{window}d" in result.columns

    def test_compute_all_respects_groups(self, sample_df, lagged_features):
        """Test that compute_all respects station grouping."""
        result = lagged_features.compute_all(
            sample_df,
            variables=["temperature"],
            lags=[1],
            windows=[3],
        )

        # Verify no leakage between stations
        station2 = result[result["station_id"] == "station_2"]
        assert pd.isna(station2["temperature_lag_1d"].iloc[0])


class TestGetFeatureNames:
    """Tests for get_feature_names method."""

    def test_get_feature_names_basic(self, lagged_features):
        """Test getting feature names for one variable."""
        names = lagged_features.get_feature_names(["temperature"], lags=[1], windows=[3])

        assert "temperature_lag_1d" in names["lags"]
        assert "temperature_roll_mean_3d" in names["rolling_mean"]
        assert "temperature_roll_std_3d" in names["rolling_std"]
        assert "temperature_roll_min_3d" in names["rolling_min"]
        assert "temperature_roll_max_3d" in names["rolling_max"]
        assert "temperature_change_1d" in names["change"]
        assert "temperature_pct_change_1d" in names["pct_change"]
        assert "temperature_cumsum_3d" in names["cumsum"]

    def test_get_feature_names_multiple_variables(self, lagged_features):
        """Test getting feature names for multiple variables."""
        names = lagged_features.get_feature_names(
            ["temp", "precip"], lags=[1, 7], windows=[3, 14]
        )

        # Check both variables are represented
        assert "temp_lag_1d" in names["lags"]
        assert "temp_lag_7d" in names["lags"]
        assert "precip_lag_1d" in names["lags"]
        assert "precip_lag_7d" in names["lags"]

        assert "temp_roll_mean_3d" in names["rolling_mean"]
        assert "temp_roll_mean_14d" in names["rolling_mean"]
        assert "precip_roll_mean_3d" in names["rolling_mean"]
        assert "precip_roll_mean_14d" in names["rolling_mean"]


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_single_row_per_group(self, lagged_features):
        """Test handling of single row per group."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01"],
            "station_id": ["station_1", "station_2"],
            "value": [1.0, 2.0],
        })

        result = lagged_features.create_lags(df, ["value"], lags=[1])

        # All lag values should be NaN
        assert result["value_lag_1d"].isna().all()

    def test_empty_dataframe(self, lagged_features):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=["date", "station_id", "value"])

        result = lagged_features.create_lags(df, ["value"], lags=[1])

        assert "value_lag_1d" in result.columns
        assert len(result) == 0

    def test_with_nan_values(self, lagged_features):
        """Test handling of NaN values in input."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "station_id": "station_1",
            "value": [1.0, np.nan, 3.0, 4.0, 5.0],
        })

        result = lagged_features.create_lags(df, ["value"], lags=[1])

        # NaN should propagate
        assert pd.isna(result["value_lag_1d"].iloc[2])  # Was NaN in original

    def test_without_group_column(self, lagged_features):
        """Test handling when group column doesn't exist."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        result = lagged_features.create_lags(
            df, ["value"], lags=[1], group_col="station_id"
        )

        # Should still work, treating all data as one group
        assert "value_lag_1d" in result.columns
        assert pd.isna(result["value_lag_1d"].iloc[0])
        assert result["value_lag_1d"].iloc[1] == 1.0

    def test_datetime_index(self, lagged_features):
        """Test handling of DataFrame with datetime index."""
        df = pd.DataFrame({
            "station_id": "station_1",
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }, index=pd.date_range("2024-01-01", periods=5, freq="D"))

        result = lagged_features.create_lags(df, ["value"], lags=[1])

        assert "value_lag_1d" in result.columns
        assert result["value_lag_1d"].iloc[1] == 1.0

    def test_preserves_original_columns(self, sample_df, lagged_features):
        """Test that original columns are preserved."""
        original_cols = set(sample_df.columns)

        result = lagged_features.compute_all(
            sample_df, ["temperature"], lags=[1], windows=[3]
        )

        # All original columns should still be present
        for col in original_cols:
            assert col in result.columns

    def test_original_data_unchanged(self, sample_df, lagged_features):
        """Test that original DataFrame is not modified."""
        original_cols = list(sample_df.columns)

        lagged_features.compute_all(sample_df, ["temperature"], lags=[1], windows=[3])

        # Original DataFrame should be unchanged
        assert list(sample_df.columns) == original_cols
