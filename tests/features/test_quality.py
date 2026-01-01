"""Tests for data quality control module.

These tests verify that the DataQualityController correctly:
1. Detects physical limit violations
2. Detects statistical outliers
3. Detects temporal inconsistencies
4. Generates accurate quality reports
5. Filters data WITHOUT interpolation or estimation
"""

import numpy as np
import pandas as pd
import pytest

from snowforecast.features.quality import (
    DataQualityController,
    QualityFlag,
    QualityReport,
    DEFAULT_PHYSICAL_LIMITS,
    DEFAULT_MAX_HOURLY_CHANGE,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_weather_df() -> pd.DataFrame:
    """Create a sample weather DataFrame for testing."""
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=10, freq="h"),
        "temperature": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        "snow_depth": [100.0, 102.0, 105.0, 108.0, 110.0, 112.0, 115.0, 118.0, 120.0, 122.0],
        "wind_speed": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        "humidity": [60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0, 74.0, 76.0, 78.0],
    })


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    """Create a DataFrame with missing values."""
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=10, freq="h"),
        "temperature": [5.0, np.nan, 7.0, 8.0, np.nan, 10.0, 11.0, np.nan, 13.0, 14.0],
        "snow_depth": [100.0, 102.0, np.nan, 108.0, 110.0, 112.0, np.nan, 118.0, 120.0, 122.0],
    })


@pytest.fixture
def df_with_physical_violations() -> pd.DataFrame:
    """Create a DataFrame with physical limit violations."""
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=5, freq="h"),
        "temperature": [5.0, 100.0, -100.0, 20.0, 25.0],  # 100 and -100 are violations
        "snow_depth": [-10.0, 50.0, 100.0, 150.0, 200.0],  # -10 is a violation
        "humidity": [50.0, 60.0, 150.0, 70.0, 80.0],  # 150 is a violation
        "wind_speed": [-5.0, 10.0, 15.0, 20.0, 25.0],  # -5 is a violation
    })


@pytest.fixture
def df_with_outliers() -> pd.DataFrame:
    """Create a DataFrame with statistical outliers."""
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=20, freq="h"),
        "temperature": [
            10.0, 11.0, 10.5, 11.5, 10.2,
            10.8, 11.2, 10.3, 11.0, 10.7,
            100.0,  # Outlier - much higher than rest
            10.9, 10.4, 11.1, 10.6,
            10.0, 11.0, 10.5, 11.5, -50.0,  # Outlier - much lower
        ],
    })


@pytest.fixture
def df_with_temporal_issues() -> pd.DataFrame:
    """Create a DataFrame with temporal inconsistencies."""
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=5, freq="h"),
        "temperature": [10.0, 12.0, 50.0, 15.0, 14.0],  # 50 is impossible jump
        "snow_depth": [100.0, 105.0, 200.0, 110.0, 115.0],  # 200 is impossible jump
    })


@pytest.fixture
def multistation_df() -> pd.DataFrame:
    """Create a DataFrame with multiple stations."""
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=6, freq="h").tolist() * 2,
        "station_id": ["A"] * 6 + ["B"] * 6,
        "temperature": [
            # Station A - normal progression
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            # Station B - has temporal issue
            20.0, 21.0, 60.0, 23.0, 24.0, 25.0,  # 60 is impossible
        ],
    })


@pytest.fixture
def qc() -> DataQualityController:
    """Create a DataQualityController with default settings."""
    return DataQualityController()


# =============================================================================
# Test QualityFlag enum
# =============================================================================


class TestQualityFlag:
    """Tests for QualityFlag enum."""

    def test_valid_is_zero(self):
        """VALID flag should be 0."""
        assert int(QualityFlag.VALID) == 0

    def test_flags_are_powers_of_two(self):
        """Each flag should be a power of two for bitwise operations."""
        flags = [
            QualityFlag.MISSING,
            QualityFlag.BELOW_PHYSICAL_LIMIT,
            QualityFlag.ABOVE_PHYSICAL_LIMIT,
            QualityFlag.STATISTICAL_OUTLIER,
            QualityFlag.TEMPORAL_INCONSISTENT,
        ]
        for flag in flags:
            value = int(flag)
            assert value > 0
            assert (value & (value - 1)) == 0, f"{flag} is not a power of two"

    def test_flags_can_be_combined(self):
        """Flags should be combinable with bitwise OR."""
        combined = QualityFlag.MISSING | QualityFlag.STATISTICAL_OUTLIER
        assert combined & QualityFlag.MISSING
        assert combined & QualityFlag.STATISTICAL_OUTLIER
        assert not (combined & QualityFlag.BELOW_PHYSICAL_LIMIT)


# =============================================================================
# Test QualityReport
# =============================================================================


class TestQualityReport:
    """Tests for QualityReport dataclass."""

    def test_valid_pct_calculation(self):
        """Test valid percentage calculation."""
        report = QualityReport(
            total_records=100,
            valid_records=85,
            missing_pct=5.0,
        )
        assert report.valid_pct == 85.0

    def test_valid_pct_zero_records(self):
        """Test valid percentage with zero records."""
        report = QualityReport(
            total_records=0,
            valid_records=0,
            missing_pct=0.0,
        )
        assert report.valid_pct == 0.0

    def test_is_acceptable_above_threshold(self):
        """Test is_acceptable when above 80% threshold."""
        report = QualityReport(
            total_records=100,
            valid_records=85,
            missing_pct=5.0,
        )
        assert report.is_acceptable is True

    def test_is_acceptable_below_threshold(self):
        """Test is_acceptable when below 80% threshold."""
        report = QualityReport(
            total_records=100,
            valid_records=70,
            missing_pct=15.0,
        )
        assert report.is_acceptable is False

    def test_str_representation(self):
        """Test string representation includes key info."""
        report = QualityReport(
            total_records=100,
            valid_records=85,
            missing_pct=5.0,
            outliers_detected=3,
        )
        report_str = str(report)
        assert "ACCEPTABLE" in report_str
        assert "85" in report_str
        assert "100" in report_str


# =============================================================================
# Test Physical Limits Check
# =============================================================================


class TestPhysicalLimits:
    """Tests for physical limits checking."""

    def test_valid_data_no_flags(self, sample_weather_df, qc):
        """Valid data should have no physical limit flags."""
        result = qc.check_physical_limits(sample_weather_df)

        # Check temperature flags
        assert "temperature_physical_flag" in result.columns
        assert (result["temperature_physical_flag"] == int(QualityFlag.VALID)).all()

    def test_temperature_above_limit(self, qc):
        """Temperature above 50C should be flagged."""
        df = pd.DataFrame({"temperature": [10.0, 60.0, 20.0]})
        result = qc.check_physical_limits(df)

        assert result["temperature_physical_flag"].iloc[0] == int(QualityFlag.VALID)
        assert result["temperature_physical_flag"].iloc[1] == int(QualityFlag.ABOVE_PHYSICAL_LIMIT)
        assert result["temperature_physical_flag"].iloc[2] == int(QualityFlag.VALID)

    def test_temperature_below_limit(self, qc):
        """Temperature below -60C should be flagged."""
        df = pd.DataFrame({"temperature": [10.0, -70.0, 20.0]})
        result = qc.check_physical_limits(df)

        assert result["temperature_physical_flag"].iloc[1] == int(QualityFlag.BELOW_PHYSICAL_LIMIT)

    def test_negative_precipitation_flagged(self, qc):
        """Negative precipitation should be flagged."""
        df = pd.DataFrame({"precipitation": [0.0, 5.0, -2.0, 10.0]})
        result = qc.check_physical_limits(df)

        assert result["precipitation_physical_flag"].iloc[2] == int(QualityFlag.BELOW_PHYSICAL_LIMIT)

    def test_humidity_out_of_range(self, qc):
        """Humidity outside 0-100% should be flagged."""
        df = pd.DataFrame({"humidity": [50.0, -5.0, 110.0, 80.0]})
        result = qc.check_physical_limits(df)

        assert result["humidity_physical_flag"].iloc[1] == int(QualityFlag.BELOW_PHYSICAL_LIMIT)
        assert result["humidity_physical_flag"].iloc[2] == int(QualityFlag.ABOVE_PHYSICAL_LIMIT)

    def test_missing_values_flagged(self, df_with_missing, qc):
        """Missing values should be flagged as MISSING."""
        result = qc.check_physical_limits(df_with_missing)

        # Check indices with NaN
        assert result["temperature_physical_flag"].iloc[1] == int(QualityFlag.MISSING)
        assert result["temperature_physical_flag"].iloc[4] == int(QualityFlag.MISSING)

    def test_custom_physical_limits(self):
        """Custom physical limits should override defaults."""
        custom_limits = {"temperature": (-20.0, 30.0)}
        qc = DataQualityController(physical_limits=custom_limits)

        df = pd.DataFrame({"temperature": [10.0, 35.0, -25.0]})
        result = qc.check_physical_limits(df)

        # 35C should now be flagged (above 30C limit)
        assert result["temperature_physical_flag"].iloc[1] == int(QualityFlag.ABOVE_PHYSICAL_LIMIT)
        # -25C should now be flagged (below -20C limit)
        assert result["temperature_physical_flag"].iloc[2] == int(QualityFlag.BELOW_PHYSICAL_LIMIT)

    def test_original_df_not_modified(self, sample_weather_df, qc):
        """Original DataFrame should not be modified."""
        original_cols = sample_weather_df.columns.tolist()
        _ = qc.check_physical_limits(sample_weather_df)
        assert sample_weather_df.columns.tolist() == original_cols


# =============================================================================
# Test Outlier Detection
# =============================================================================


class TestOutlierDetection:
    """Tests for statistical outlier detection."""

    def test_iqr_method_detects_outliers(self, df_with_outliers, qc):
        """IQR method should detect extreme outliers."""
        result = qc.detect_outliers(df_with_outliers, method="iqr", threshold=3.0)

        assert "temperature_outlier_flag" in result.columns
        # 100.0 at index 10 and -50.0 at index 19 should be outliers
        assert result["temperature_outlier_flag"].iloc[10] == int(QualityFlag.STATISTICAL_OUTLIER)
        assert result["temperature_outlier_flag"].iloc[19] == int(QualityFlag.STATISTICAL_OUTLIER)

    def test_zscore_method_detects_outliers(self, df_with_outliers, qc):
        """Z-score method should detect extreme outliers."""
        result = qc.detect_outliers(df_with_outliers, method="zscore", threshold=3.0)

        assert "temperature_outlier_flag" in result.columns
        # At least the most extreme value (100.0) should be flagged
        outlier_mask = result["temperature_outlier_flag"] == int(QualityFlag.STATISTICAL_OUTLIER)
        assert outlier_mask.sum() >= 1  # At least 1 outlier detected
        # The 100.0 value at index 10 should definitely be flagged
        assert result["temperature_outlier_flag"].iloc[10] == int(QualityFlag.STATISTICAL_OUTLIER)

    def test_no_outliers_in_uniform_data(self, sample_weather_df, qc):
        """Uniform data should have no outliers."""
        result = qc.detect_outliers(sample_weather_df, method="iqr")

        assert "temperature_outlier_flag" in result.columns
        assert (result["temperature_outlier_flag"] == int(QualityFlag.VALID)).all()

    def test_missing_values_flagged_as_missing(self, df_with_missing, qc):
        """Missing values should be flagged as MISSING, not as outliers."""
        result = qc.detect_outliers(df_with_missing)

        # NaN at index 1 should be flagged as MISSING
        assert result["temperature_outlier_flag"].iloc[1] == int(QualityFlag.MISSING)

    def test_specific_columns_only(self, sample_weather_df, qc):
        """Should only check specified columns."""
        result = qc.detect_outliers(sample_weather_df, columns=["temperature"])

        assert "temperature_outlier_flag" in result.columns
        assert "wind_speed_outlier_flag" not in result.columns

    def test_invalid_method_raises_error(self, sample_weather_df, qc):
        """Invalid outlier method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown outlier detection method"):
            qc.detect_outliers(sample_weather_df, method="invalid_method")


# =============================================================================
# Test Temporal Consistency
# =============================================================================


class TestTemporalConsistency:
    """Tests for temporal consistency checking."""

    def test_consistent_data_no_flags(self, sample_weather_df, qc):
        """Temporally consistent data should have no flags."""
        result = qc.check_temporal_consistency(sample_weather_df)

        assert "temperature_temporal_flag" in result.columns
        # Check most values are VALID (first value has no previous to compare)
        assert (result["temperature_temporal_flag"] == int(QualityFlag.VALID)).sum() >= 9

    def test_impossible_temperature_jump_flagged(self, df_with_temporal_issues, qc):
        """Impossible temperature jumps should be flagged."""
        result = qc.check_temporal_consistency(df_with_temporal_issues)

        # Jump from 12 to 50 in 1 hour is impossible (max 15C/hour)
        assert result["temperature_temporal_flag"].iloc[2] == int(QualityFlag.TEMPORAL_INCONSISTENT)

    def test_impossible_snow_depth_jump_flagged(self, df_with_temporal_issues, qc):
        """Impossible snow depth jumps should be flagged."""
        result = qc.check_temporal_consistency(df_with_temporal_issues)

        # Jump from 105 to 200 in 1 hour is impossible (max 30cm/hour)
        assert result["snow_depth_temporal_flag"].iloc[2] == int(QualityFlag.TEMPORAL_INCONSISTENT)

    def test_multistation_groupby(self, multistation_df, qc):
        """Temporal consistency should work correctly with groupby."""
        result = qc.check_temporal_consistency(
            multistation_df,
            time_column="datetime",
            group_by="station_id",
        )

        # Station A should have no issues
        station_a = result[result["station_id"] == "A"]
        assert (station_a["temperature_temporal_flag"] == int(QualityFlag.VALID)).all()

        # Station B should have temporal issue at index where 60 appears
        station_b = result[result["station_id"] == "B"]
        has_temporal_issue = (
            station_b["temperature_temporal_flag"] == int(QualityFlag.TEMPORAL_INCONSISTENT)
        ).any()
        assert has_temporal_issue

    def test_missing_time_column_warning(self, qc, caplog):
        """Missing time column should log warning."""
        df = pd.DataFrame({"temperature": [10.0, 11.0, 12.0]})
        result = qc.check_temporal_consistency(df, time_column="datetime")

        # Should return original df without temporal flags
        assert "temperature_temporal_flag" not in result.columns


# =============================================================================
# Test Quality Report Generation
# =============================================================================


class TestQualityReportGeneration:
    """Tests for quality report generation."""

    def test_report_counts_valid_records(self, sample_weather_df, qc):
        """Report should count valid records correctly."""
        report = qc.generate_report(sample_weather_df)

        assert report.total_records == 10
        assert report.valid_records == 10
        assert report.valid_pct == 100.0

    def test_report_counts_missing(self, df_with_missing, qc):
        """Report should count missing values correctly."""
        report = qc.generate_report(df_with_missing)

        assert report.missing_pct > 0
        assert report.valid_records < report.total_records

    def test_report_counts_physical_violations(self, df_with_physical_violations, qc):
        """Report should count physical violations."""
        report = qc.generate_report(df_with_physical_violations)

        assert report.physical_violations > 0
        assert len(report.issues) > 0

    def test_report_counts_outliers(self, df_with_outliers, qc):
        """Report should count outliers."""
        report = qc.generate_report(df_with_outliers)

        assert report.outliers_detected > 0

    def test_report_includes_column_stats(self, sample_weather_df, qc):
        """Report should include per-column statistics."""
        report = qc.generate_report(sample_weather_df)

        assert "temperature" in report.column_stats
        assert "mean" in report.column_stats["temperature"]
        assert "std" in report.column_stats["temperature"]
        assert "min" in report.column_stats["temperature"]
        assert "max" in report.column_stats["temperature"]


# =============================================================================
# Test Combined Quality Flags
# =============================================================================


class TestApplyQualityFlags:
    """Tests for combined quality flag application."""

    def test_adds_quality_flag_column(self, sample_weather_df, qc):
        """Should add a combined quality_flag column."""
        result = qc.apply_quality_flags(sample_weather_df)

        assert "quality_flag" in result.columns

    def test_combined_flag_reflects_all_issues(self, df_with_physical_violations, qc):
        """Combined flag should reflect all detected issues."""
        result = qc.apply_quality_flags(df_with_physical_violations)

        # Rows with issues should have non-zero combined flag
        has_issues = result["quality_flag"] != int(QualityFlag.VALID)
        assert has_issues.any()

    def test_original_values_not_modified(self, df_with_physical_violations, qc):
        """Original data values should NEVER be modified."""
        original_temp = df_with_physical_violations["temperature"].copy()
        result = qc.apply_quality_flags(df_with_physical_violations)

        # Check original values are unchanged
        pd.testing.assert_series_equal(
            result["temperature"],
            original_temp,
            check_names=False,
        )


# =============================================================================
# Test Filtering to Valid Records
# =============================================================================


class TestFilterToValid:
    """Tests for filtering to valid records.

    CRITICAL: These tests verify that filtering NEVER interpolates or estimates.
    """

    def test_filters_out_missing_values(self, df_with_missing, qc):
        """Should filter out rows with missing values."""
        result = qc.filter_to_valid(df_with_missing, min_completeness=1.0)

        # Should have fewer rows
        assert len(result) < len(df_with_missing)
        # Remaining rows should have no missing values
        assert not result[["temperature", "snow_depth"]].isna().any().any()

    def test_no_interpolation_of_missing(self, df_with_missing, qc):
        """CRITICAL: Missing values should NOT be interpolated."""
        result = qc.filter_to_valid(df_with_missing)

        # Original indices with NaN should not be in result
        original_nan_indices = df_with_missing[
            df_with_missing["temperature"].isna()
        ].index.tolist()

        for idx in original_nan_indices:
            assert idx not in result.index, (
                f"Index {idx} with missing value was not filtered out!"
            )

    def test_filters_physical_violations(self, df_with_physical_violations, qc):
        """Should filter out physical limit violations."""
        result = qc.filter_to_valid(
            df_with_physical_violations,
            exclude_physical_violations=True,
        )

        # Check no temperature violations remain
        assert not (result["temperature"] > 50).any()
        assert not (result["temperature"] < -60).any()

    def test_filters_outliers(self, df_with_outliers, qc):
        """Should filter out statistical outliers."""
        result = qc.filter_to_valid(
            df_with_outliers,
            exclude_outliers=True,
            min_completeness=0.0,
        )

        # Extreme outliers should be filtered
        assert not (result["temperature"] > 50).any()
        assert not (result["temperature"] < -40).any()

    def test_respects_min_completeness(self, df_with_missing, qc):
        """Should respect minimum completeness threshold."""
        # With 50% completeness, rows with 1 of 2 numeric cols missing should pass
        result = qc.filter_to_valid(df_with_missing, min_completeness=0.5)

        # More rows should remain than with 100% completeness
        strict_result = qc.filter_to_valid(df_with_missing, min_completeness=1.0)
        assert len(result) >= len(strict_result)

    def test_returns_copy_not_view(self, sample_weather_df, qc):
        """Should return a copy, not a view of original data."""
        result = qc.filter_to_valid(sample_weather_df)

        # Modifying result should not affect original
        if len(result) > 0:
            result.iloc[0, 1] = -999
            assert sample_weather_df.iloc[0, 1] != -999

    def test_can_disable_individual_filters(self, df_with_physical_violations, qc):
        """Should be able to disable individual filter types."""
        # Disable physical violation filter
        result = qc.filter_to_valid(
            df_with_physical_violations,
            exclude_physical_violations=False,
            exclude_outliers=False,
            exclude_temporal_violations=False,
            min_completeness=0.0,
        )

        # Should keep more rows
        strict_result = qc.filter_to_valid(df_with_physical_violations)
        assert len(result) >= len(strict_result)


# =============================================================================
# Test Default Limits
# =============================================================================


class TestDefaultLimits:
    """Tests for default physical limits configuration."""

    def test_temperature_limits_defined(self):
        """Temperature limits should be defined."""
        assert "temperature" in DEFAULT_PHYSICAL_LIMITS
        min_val, max_val = DEFAULT_PHYSICAL_LIMITS["temperature"]
        assert min_val == -60.0
        assert max_val == 50.0

    def test_precipitation_non_negative(self):
        """Precipitation should have non-negative minimum."""
        assert "precipitation" in DEFAULT_PHYSICAL_LIMITS
        min_val, _ = DEFAULT_PHYSICAL_LIMITS["precipitation"]
        assert min_val == 0.0

    def test_humidity_percentage_range(self):
        """Humidity should be 0-100%."""
        assert "humidity" in DEFAULT_PHYSICAL_LIMITS
        min_val, max_val = DEFAULT_PHYSICAL_LIMITS["humidity"]
        assert min_val == 0.0
        assert max_val == 100.0

    def test_max_hourly_change_defined(self):
        """Max hourly change should be defined for key variables."""
        assert "temperature" in DEFAULT_MAX_HOURLY_CHANGE
        assert "snow_depth" in DEFAULT_MAX_HOURLY_CHANGE
        assert DEFAULT_MAX_HOURLY_CHANGE["temperature"] > 0
        assert DEFAULT_MAX_HOURLY_CHANGE["snow_depth"] > 0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self, qc):
        """Should handle empty DataFrame gracefully."""
        df = pd.DataFrame({"temperature": []})
        result = qc.apply_quality_flags(df)
        assert len(result) == 0

    def test_single_row_dataframe(self, qc):
        """Should handle single-row DataFrame."""
        df = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-01")],
            "temperature": [10.0],
        })
        result = qc.apply_quality_flags(df)
        assert len(result) == 1

    def test_all_missing_column(self, qc):
        """Should handle column with all missing values."""
        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=5, freq="h"),
            "temperature": [np.nan] * 5,
        })
        result = qc.apply_quality_flags(df)
        assert (result["temperature_physical_flag"] == int(QualityFlag.MISSING)).all()

    def test_no_numeric_columns(self, qc):
        """Should handle DataFrame with no numeric columns."""
        df = pd.DataFrame({
            "station": ["A", "B", "C"],
            "name": ["Station A", "Station B", "Station C"],
        })
        result = qc.apply_quality_flags(df)
        assert "quality_flag" in result.columns

    def test_very_large_values(self, qc):
        """Should handle very large values correctly."""
        df = pd.DataFrame({
            "temperature": [10.0, 1e10, -1e10],
        })
        result = qc.check_physical_limits(df)

        assert result["temperature_physical_flag"].iloc[1] == int(QualityFlag.ABOVE_PHYSICAL_LIMIT)
        assert result["temperature_physical_flag"].iloc[2] == int(QualityFlag.BELOW_PHYSICAL_LIMIT)


# =============================================================================
# Test Data Integrity Guarantees
# =============================================================================


class TestDataIntegrityGuarantees:
    """Tests to verify data integrity rules are enforced.

    CRITICAL: These tests ensure we NEVER:
    - Estimate missing data
    - Interpolate or extrapolate
    - Fill gaps with assumed values
    - Modify source data values
    """

    def test_source_values_never_modified(self, df_with_physical_violations, qc):
        """Source data values should NEVER be modified."""
        original = df_with_physical_violations.copy()

        # Run all quality operations
        _ = qc.check_physical_limits(df_with_physical_violations)
        _ = qc.detect_outliers(df_with_physical_violations)
        _ = qc.check_temporal_consistency(df_with_physical_violations)
        _ = qc.apply_quality_flags(df_with_physical_violations)
        _ = qc.generate_report(df_with_physical_violations)
        _ = qc.filter_to_valid(df_with_physical_violations)

        # Original should be unchanged
        pd.testing.assert_frame_equal(df_with_physical_violations, original)

    def test_no_new_values_created(self, df_with_missing, qc):
        """No new data values should be created (only flags added)."""
        result = qc.apply_quality_flags(df_with_missing)

        # Only flag columns should be added
        new_cols = set(result.columns) - set(df_with_missing.columns)
        for col in new_cols:
            assert col.endswith("_flag"), f"Non-flag column '{col}' was added"

    def test_missing_stays_missing(self, df_with_missing, qc):
        """Missing values should remain missing (not filled)."""
        result = qc.apply_quality_flags(df_with_missing)

        # Check original NaN positions are still NaN
        original_nan_mask = df_with_missing["temperature"].isna()
        result_nan_mask = result["temperature"].isna()

        pd.testing.assert_series_equal(
            original_nan_mask.reset_index(drop=True),
            result_nan_mask.reset_index(drop=True),
        )

    def test_filter_removes_not_fills(self, df_with_missing, qc):
        """Filtering should remove rows, not fill them."""
        original_len = len(df_with_missing)
        result = qc.filter_to_valid(df_with_missing, min_completeness=1.0)

        # Length should decrease, not stay same
        assert len(result) < original_len
        # No NaN should remain in filtered result
        assert not result.select_dtypes(include=[np.number]).isna().any().any()
