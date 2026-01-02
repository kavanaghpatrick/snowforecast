"""Tests for temporal alignment and resampling.

Tests the TemporalAligner class which handles:
- Resampling hourly data to daily
- Timezone conversions to UTC
- Aligning data to common date ranges
- NEVER interpolating missing data
"""


import numpy as np
import pandas as pd
import pytest

from snowforecast.features.temporal import DEFAULT_AGGREGATIONS, TemporalAligner


class TestTemporalAlignerInit:
    """Tests for TemporalAligner initialization."""

    def test_default_initialization(self):
        """Test default initialization with daily frequency and UTC timezone."""
        aligner = TemporalAligner()
        assert aligner.target_freq == "1D"
        assert aligner.timezone == "UTC"

    def test_custom_initialization(self):
        """Test initialization with custom frequency and timezone."""
        aligner = TemporalAligner(target_freq="1h", timezone="America/Denver")
        assert aligner.target_freq == "1h"
        assert aligner.timezone == "America/Denver"


class TestResampleHourlyToDaily:
    """Tests for resample_hourly_to_daily method."""

    @pytest.fixture
    def hourly_data(self):
        """Create sample hourly data for testing."""
        dates = pd.date_range(
            start="2023-01-01 00:00:00",
            end="2023-01-03 23:00:00",
            freq="h",
            tz="UTC",
        )
        return pd.DataFrame({
            "datetime": dates,
            "temp_avg_c": np.random.uniform(-10, 10, len(dates)),
            "precipitation": np.random.uniform(0, 5, len(dates)),
            "snow_depth_cm": np.random.uniform(0, 100, len(dates)),
        })

    def test_resample_basic(self, hourly_data):
        """Test basic resampling of hourly to daily."""
        aligner = TemporalAligner()
        result = aligner.resample_hourly_to_daily(hourly_data)

        # Should have 3 days
        assert len(result) == 3
        assert "date" in result.columns

    def test_resample_temperature_uses_mean(self, hourly_data):
        """Test that temperature is aggregated using mean."""
        aligner = TemporalAligner()
        result = aligner.resample_hourly_to_daily(hourly_data)

        # Calculate expected mean for first day manually
        first_day_mask = hourly_data["datetime"].dt.date == pd.Timestamp("2023-01-01").date()
        expected_mean = hourly_data.loc[first_day_mask, "temp_avg_c"].mean()

        # Check result (allowing for floating point differences)
        assert np.isclose(result.loc[0, "temp_avg_c"], expected_mean, rtol=1e-5)

    def test_resample_precipitation_uses_sum(self, hourly_data):
        """Test that precipitation is aggregated using sum."""
        aligner = TemporalAligner()
        result = aligner.resample_hourly_to_daily(hourly_data)

        # Calculate expected sum for first day manually
        first_day_mask = hourly_data["datetime"].dt.date == pd.Timestamp("2023-01-01").date()
        expected_sum = hourly_data.loc[first_day_mask, "precipitation"].sum()

        # Check result
        assert np.isclose(result.loc[0, "precipitation"], expected_sum, rtol=1e-5)

    def test_resample_with_custom_agg_methods(self, hourly_data):
        """Test resampling with custom aggregation methods."""
        aligner = TemporalAligner()
        result = aligner.resample_hourly_to_daily(
            hourly_data,
            agg_methods={"temp_avg_c": "max", "snow_depth_cm": "min"},
        )

        # Calculate expected values manually
        first_day_mask = hourly_data["datetime"].dt.date == pd.Timestamp("2023-01-01").date()
        expected_max_temp = hourly_data.loc[first_day_mask, "temp_avg_c"].max()
        expected_min_snow = hourly_data.loc[first_day_mask, "snow_depth_cm"].min()

        assert np.isclose(result.loc[0, "temp_avg_c"], expected_max_temp, rtol=1e-5)
        assert np.isclose(result.loc[0, "snow_depth_cm"], expected_min_snow, rtol=1e-5)

    def test_resample_with_group_cols(self):
        """Test resampling with grouping by station_id."""
        dates = pd.date_range(
            start="2023-01-01 00:00:00",
            end="2023-01-01 23:00:00",
            freq="h",
            tz="UTC",
        )
        df = pd.DataFrame({
            "datetime": list(dates) * 2,
            "station_id": ["A"] * len(dates) + ["B"] * len(dates),
            "temp_avg_c": np.random.uniform(-10, 10, len(dates) * 2),
        })

        aligner = TemporalAligner()
        result = aligner.resample_hourly_to_daily(df, group_cols=["station_id"])

        # Should have 2 rows (one per station)
        assert len(result) == 2
        assert set(result["station_id"]) == {"A", "B"}

    def test_resample_empty_dataframe(self):
        """Test resampling an empty DataFrame returns empty DataFrame."""
        df = pd.DataFrame(columns=["datetime", "temp_avg_c"])
        aligner = TemporalAligner()
        result = aligner.resample_hourly_to_daily(df)

        assert result.empty

    def test_resample_missing_datetime_col_raises(self):
        """Test that missing datetime column raises ValueError."""
        df = pd.DataFrame({"temp": [1, 2, 3]})
        aligner = TemporalAligner()

        with pytest.raises(ValueError, match="datetime_col"):
            aligner.resample_hourly_to_daily(df)

    def test_resample_preserves_nan_values(self):
        """Test that NaN values are preserved (not interpolated)."""
        dates = pd.date_range(
            start="2023-01-01 00:00:00",
            end="2023-01-01 23:00:00",
            freq="h",
            tz="UTC",
        )
        df = pd.DataFrame({
            "datetime": dates,
            "temp_avg_c": [np.nan] * len(dates),  # All NaN
        })

        aligner = TemporalAligner()
        result = aligner.resample_hourly_to_daily(df)

        # Result should have NaN
        assert pd.isna(result.loc[0, "temp_avg_c"])

    def test_resample_timezone_naive_assumes_utc(self):
        """Test that timezone-naive data is treated as UTC."""
        dates = pd.date_range(
            start="2023-01-01 00:00:00",
            end="2023-01-01 23:00:00",
            freq="h",
        )  # No timezone
        df = pd.DataFrame({
            "datetime": dates,
            "temp_avg_c": np.random.uniform(-10, 10, len(dates)),
        })

        aligner = TemporalAligner()
        # Should not raise, but log a warning
        result = aligner.resample_hourly_to_daily(df)

        assert len(result) == 1


class TestAlignToDateRange:
    """Tests for align_to_date_range method."""

    @pytest.fixture
    def sparse_daily_data(self):
        """Create sparse daily data with gaps."""
        return pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-05"]),
            "temp_avg_c": [5.0, 3.0, -2.0],
        })

    def test_align_fills_missing_dates_with_nan(self, sparse_daily_data):
        """Test that missing dates are filled with NaN (not interpolated)."""
        aligner = TemporalAligner()
        result = aligner.align_to_date_range(
            sparse_daily_data,
            start_date="2023-01-01",
            end_date="2023-01-05",
        )

        # Should have 5 rows
        assert len(result) == 5

        # Missing dates should have NaN values
        jan_2 = result[result["date"] == pd.Timestamp("2023-01-02")]
        assert len(jan_2) == 1
        assert pd.isna(jan_2.iloc[0]["temp_avg_c"])

        jan_4 = result[result["date"] == pd.Timestamp("2023-01-04")]
        assert len(jan_4) == 1
        assert pd.isna(jan_4.iloc[0]["temp_avg_c"])

    def test_align_extends_date_range(self, sparse_daily_data):
        """Test alignment extends beyond original data range."""
        aligner = TemporalAligner()
        result = aligner.align_to_date_range(
            sparse_daily_data,
            start_date="2022-12-30",
            end_date="2023-01-07",
        )

        # Should have 9 days (Dec 30 to Jan 7)
        assert len(result) == 9

        # Extended dates should be NaN
        dec_30 = result[result["date"] == pd.Timestamp("2022-12-30")]
        assert pd.isna(dec_30.iloc[0]["temp_avg_c"])

    def test_align_with_group_cols(self):
        """Test alignment with station grouping."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-01", "2023-01-02"]),
            "station_id": ["A", "A", "B", "B"],
            "temp_avg_c": [5.0, 3.0, 6.0, 4.0],
        })

        aligner = TemporalAligner()
        result = aligner.align_to_date_range(
            df,
            start_date="2023-01-01",
            end_date="2023-01-03",
            group_cols=["station_id"],
        )

        # Should have 6 rows (3 days x 2 stations)
        assert len(result) == 6

        # Check station A has gap on Jan 2
        station_a = result[result["station_id"] == "A"]
        jan_2_a = station_a[station_a["date"] == pd.Timestamp("2023-01-02")]
        assert pd.isna(jan_2_a.iloc[0]["temp_avg_c"])

    def test_align_empty_dataframe(self):
        """Test alignment of empty DataFrame."""
        df = pd.DataFrame(columns=["date", "temp_avg_c"])
        aligner = TemporalAligner()
        result = aligner.align_to_date_range(
            df,
            start_date="2023-01-01",
            end_date="2023-01-05",
        )

        # Should have date range even with empty input
        assert len(result) == 5
        assert "date" in result.columns

    def test_align_missing_date_col_raises(self):
        """Test that missing date column raises ValueError."""
        df = pd.DataFrame({"temp": [1, 2, 3]})
        aligner = TemporalAligner()

        with pytest.raises(ValueError, match="date_col"):
            aligner.align_to_date_range(df, "2023-01-01", "2023-01-05")

    def test_align_handles_duplicate_dates(self):
        """Test that duplicate dates are handled (keep first)."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"]),
            "temp_avg_c": [5.0, 10.0, 3.0],  # Two values for Jan 1
        })

        aligner = TemporalAligner()
        result = aligner.align_to_date_range(
            df,
            start_date="2023-01-01",
            end_date="2023-01-02",
        )

        # Should keep first value for Jan 1
        assert len(result) == 2
        jan_1 = result[result["date"] == pd.Timestamp("2023-01-01")]
        assert jan_1.iloc[0]["temp_avg_c"] == 5.0


class TestLocalizeToUtc:
    """Tests for localize_to_utc method."""

    def test_localize_timezone_naive_with_source_tz(self):
        """Test localization of timezone-naive data with known source timezone."""
        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2023-01-01 12:00:00", "2023-01-01 13:00:00"]),
            "temp": [5.0, 6.0],
        })

        aligner = TemporalAligner()
        result = aligner.localize_to_utc(df, source_tz="America/Denver")

        # Denver is UTC-7 in winter
        # 12:00 Denver = 19:00 UTC
        assert result["datetime"].iloc[0].hour == 19
        assert str(result["datetime"].dt.tz) == "UTC"

    def test_localize_timezone_naive_without_source_tz(self):
        """Test localization of timezone-naive data assumes UTC."""
        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2023-01-01 12:00:00"]),
            "temp": [5.0],
        })

        aligner = TemporalAligner()
        result = aligner.localize_to_utc(df)

        # Should remain 12:00 UTC
        assert result["datetime"].iloc[0].hour == 12
        assert str(result["datetime"].dt.tz) == "UTC"

    def test_localize_already_utc(self):
        """Test that UTC data remains unchanged."""
        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2023-01-01 12:00:00"]).tz_localize("UTC"),
            "temp": [5.0],
        })

        aligner = TemporalAligner()
        result = aligner.localize_to_utc(df)

        assert result["datetime"].iloc[0].hour == 12
        assert str(result["datetime"].dt.tz) == "UTC"

    def test_localize_converts_other_tz_to_utc(self):
        """Test conversion from other timezone to UTC."""
        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2023-07-01 12:00:00"]).tz_localize("America/Los_Angeles"),
            "temp": [25.0],
        })

        aligner = TemporalAligner()
        result = aligner.localize_to_utc(df)

        # LA is UTC-7 in summer (PDT)
        # 12:00 LA = 19:00 UTC
        assert result["datetime"].iloc[0].hour == 19
        assert str(result["datetime"].dt.tz) == "UTC"

    def test_localize_empty_dataframe(self):
        """Test localization of empty DataFrame."""
        df = pd.DataFrame(columns=["datetime", "temp"])
        aligner = TemporalAligner()
        result = aligner.localize_to_utc(df)

        assert result.empty

    def test_localize_missing_datetime_col_raises(self):
        """Test that missing datetime column raises ValueError."""
        df = pd.DataFrame({"temp": [1, 2, 3]})
        aligner = TemporalAligner()

        with pytest.raises(ValueError, match="datetime_col"):
            aligner.localize_to_utc(df)


class TestCreateDateIndex:
    """Tests for create_date_index method."""

    def test_create_daily_index(self):
        """Test creation of daily date index."""
        aligner = TemporalAligner(target_freq="1D")
        dates = aligner.create_date_index("2023-01-01", "2023-01-10")

        assert len(dates) == 10
        assert dates[0] == pd.Timestamp("2023-01-01")
        assert dates[-1] == pd.Timestamp("2023-01-10")

    def test_create_index_single_day(self):
        """Test creation of index for single day."""
        aligner = TemporalAligner()
        dates = aligner.create_date_index("2023-01-01", "2023-01-01")

        assert len(dates) == 1
        assert dates[0] == pd.Timestamp("2023-01-01")

    def test_create_index_leap_year(self):
        """Test date index handles leap year correctly."""
        aligner = TemporalAligner()
        dates = aligner.create_date_index("2024-02-28", "2024-03-01")

        assert len(dates) == 3  # Feb 28, 29, Mar 1
        assert pd.Timestamp("2024-02-29") in dates


class TestMergeTemporalSources:
    """Tests for merge_temporal_sources method."""

    def test_merge_two_sources(self):
        """Test merging two temporal sources."""
        df1 = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "temp_snotel": [5.0, 6.0, 4.0],
        })
        df2 = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "temp_ghcn": [5.5, 5.8, 4.2],
        })

        aligner = TemporalAligner()
        result = aligner.merge_temporal_sources([df1, df2])

        assert len(result) == 3
        assert "temp_snotel" in result.columns
        assert "temp_ghcn" in result.columns

    def test_merge_with_missing_dates(self):
        """Test merging sources with different date ranges."""
        df1 = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "temp_a": [5.0, 6.0],
        })
        df2 = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-02", "2023-01-03"]),
            "temp_b": [5.5, 4.2],
        })

        aligner = TemporalAligner()
        result = aligner.merge_temporal_sources([df1, df2])

        # Should have 3 dates (outer join)
        assert len(result) == 3

        # Jan 1 should have NaN for temp_b
        jan_1 = result[result["date"] == pd.Timestamp("2023-01-01")]
        assert pd.isna(jan_1.iloc[0]["temp_b"])

    def test_merge_empty_list(self):
        """Test merging empty list returns empty DataFrame."""
        aligner = TemporalAligner()
        result = aligner.merge_temporal_sources([])

        assert result.empty

    def test_merge_single_source(self):
        """Test merging single source returns copy."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01"]),
            "temp": [5.0],
        })

        aligner = TemporalAligner()
        result = aligner.merge_temporal_sources([df])

        assert len(result) == 1
        assert result is not df  # Should be a copy


class TestGetTemporalCoverage:
    """Tests for get_temporal_coverage method."""

    def test_coverage_complete(self):
        """Test coverage calculation for complete data."""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", "2023-01-10", freq="D"),
            "temp": range(10),
        })

        aligner = TemporalAligner()
        coverage = aligner.get_temporal_coverage(df)

        assert coverage["total_days"] == 10
        assert coverage["present_days"] == 10
        assert coverage["missing_days"] == 0
        assert coverage["coverage_pct"] == 100.0
        assert len(coverage["gaps"]) == 0

    def test_coverage_with_gaps(self):
        """Test coverage calculation for data with gaps."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-05", "2023-01-10"]),
            "temp": [5.0, 4.0, 3.0],
        })

        aligner = TemporalAligner()
        coverage = aligner.get_temporal_coverage(
            df,
            start_date="2023-01-01",
            end_date="2023-01-10",
        )

        assert coverage["total_days"] == 10
        assert coverage["present_days"] == 3
        assert coverage["missing_days"] == 7
        assert len(coverage["gaps"]) == 2  # Jan 2-4 and Jan 6-9

    def test_coverage_empty_dataframe(self):
        """Test coverage calculation for empty DataFrame."""
        df = pd.DataFrame(columns=["date", "temp"])

        aligner = TemporalAligner()
        coverage = aligner.get_temporal_coverage(df)

        assert coverage["total_days"] == 0
        assert coverage["coverage_pct"] == 0.0


class TestDefaultAggregations:
    """Tests for DEFAULT_AGGREGATIONS constant."""

    def test_temperature_uses_mean(self):
        """Test that temperature variables default to mean."""
        assert DEFAULT_AGGREGATIONS["temp"] == "mean"
        assert DEFAULT_AGGREGATIONS["temperature"] == "mean"
        assert DEFAULT_AGGREGATIONS["temp_avg_c"] == "mean"

    def test_precipitation_uses_sum(self):
        """Test that precipitation variables default to sum."""
        assert DEFAULT_AGGREGATIONS["precip"] == "sum"
        assert DEFAULT_AGGREGATIONS["precipitation"] == "sum"
        assert DEFAULT_AGGREGATIONS["prcp"] == "sum"

    def test_snowfall_uses_sum(self):
        """Test that snowfall variables default to sum."""
        assert DEFAULT_AGGREGATIONS["snowfall"] == "sum"
        assert DEFAULT_AGGREGATIONS["snow"] == "sum"

    def test_snow_depth_uses_mean(self):
        """Test that snow depth variables default to mean."""
        assert DEFAULT_AGGREGATIONS["snow_depth"] == "mean"
        assert DEFAULT_AGGREGATIONS["snow_depth_cm"] == "mean"


class TestNeverInterpolate:
    """Tests to verify that missing data is NEVER interpolated."""

    def test_resample_does_not_interpolate(self):
        """Test that resampling does not interpolate missing hours."""
        # Create data with a missing hour
        dates = pd.to_datetime([
            "2023-01-01 00:00:00",
            "2023-01-01 01:00:00",
            # Missing 02:00
            "2023-01-01 03:00:00",
        ])
        df = pd.DataFrame({
            "datetime": dates.tz_localize("UTC"),
            "temp_avg_c": [5.0, 6.0, 8.0],
        })

        aligner = TemporalAligner()
        result = aligner.resample_hourly_to_daily(df)

        # Mean should be (5+6+8)/3 = 6.33..., not interpolated to include 7.0 for 02:00
        expected_mean = (5.0 + 6.0 + 8.0) / 3
        assert np.isclose(result.loc[0, "temp_avg_c"], expected_mean, rtol=1e-5)

    def test_align_creates_nan_not_interpolated(self):
        """Test that alignment creates NaN for missing dates, not interpolated values."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-05"]),
            "temp_avg_c": [0.0, 100.0],
        })

        aligner = TemporalAligner()
        result = aligner.align_to_date_range(
            df,
            start_date="2023-01-01",
            end_date="2023-01-05",
        )

        # Middle dates should be NaN, not interpolated (e.g., 25, 50, 75)
        for date_str in ["2023-01-02", "2023-01-03", "2023-01-04"]:
            row = result[result["date"] == pd.Timestamp(date_str)]
            assert pd.isna(row.iloc[0]["temp_avg_c"]), \
                f"Expected NaN for {date_str}, got {row.iloc[0]['temp_avg_c']}"
