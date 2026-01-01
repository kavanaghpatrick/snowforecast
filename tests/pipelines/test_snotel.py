"""Tests for the SNOTEL data ingestion pipeline."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from snowforecast.pipelines.snotel import SnotelPipeline, StationMetadata
from snowforecast.utils import ValidationResult


class TestStationMetadata:
    """Tests for StationMetadata dataclass."""

    def test_station_metadata_creation(self):
        """Should create StationMetadata with all fields."""
        station = StationMetadata(
            station_id="1050:CO:SNTL",
            name="Berthoud Summit",
            lat=39.80,
            lon=-105.78,
            elevation=3450.0,
            state="CO",
        )
        assert station.station_id == "1050:CO:SNTL"
        assert station.name == "Berthoud Summit"
        assert station.lat == 39.80
        assert station.lon == -105.78
        assert station.elevation == 3450.0
        assert station.state == "CO"


class TestSnotelPipelineInit:
    """Tests for SnotelPipeline initialization."""

    def test_default_directories(self, tmp_path):
        """Should use default data directories."""
        with patch("snowforecast.pipelines.snotel.get_data_path") as mock_get_path:
            mock_get_path.side_effect = lambda p, s: tmp_path / s / p
            pipeline = SnotelPipeline()
            assert mock_get_path.call_count == 2

    def test_custom_directories(self, tmp_path):
        """Should accept custom directories."""
        raw_dir = tmp_path / "custom_raw"
        processed_dir = tmp_path / "custom_processed"
        raw_dir.mkdir()
        processed_dir.mkdir()

        pipeline = SnotelPipeline(raw_dir=raw_dir, processed_dir=processed_dir)
        assert pipeline.raw_dir == raw_dir
        assert pipeline.processed_dir == processed_dir


class TestGetStationMetadata:
    """Tests for get_station_metadata method."""

    @patch("snowforecast.pipelines.snotel.SnotelPipeline._get_metloom_imports")
    def test_get_station_metadata(self, mock_imports, tmp_path):
        """Should return list of StationMetadata objects."""
        # Create mock metloom objects
        mock_point1 = MagicMock()
        mock_point1.id = "1050:CO:SNTL"
        mock_point1.name = "Berthoud Summit"
        mock_point1.x = -105.78
        mock_point1.y = 39.80
        mock_point1.elevation = 3450.0

        mock_point2 = MagicMock()
        mock_point2.id = "978:UT:SNTL"
        mock_point2.name = "Brighton"
        mock_point2.x = -111.58
        mock_point2.y = 40.60
        mock_point2.elevation = 2670.0

        mock_snotel_point_data = MagicMock()
        mock_snotel_point_data.points_from_geometry.return_value = [mock_point1, mock_point2]

        mock_imports.return_value = (mock_snotel_point_data, MagicMock())

        pipeline = SnotelPipeline(raw_dir=tmp_path, processed_dir=tmp_path)
        stations = pipeline.get_station_metadata()

        assert len(stations) == 2
        assert isinstance(stations[0], StationMetadata)
        assert stations[0].station_id == "1050:CO:SNTL"
        assert stations[0].state == "CO"
        assert stations[1].station_id == "978:UT:SNTL"
        assert stations[1].state == "UT"

    @patch("snowforecast.pipelines.snotel.SnotelPipeline._get_metloom_imports")
    def test_get_station_metadata_by_state(self, mock_imports, tmp_path):
        """Should filter stations by state code."""
        mock_point1 = MagicMock()
        mock_point1.id = "1050:CO:SNTL"
        mock_point1.name = "Berthoud Summit"
        mock_point1.x = -105.78
        mock_point1.y = 39.80
        mock_point1.elevation = 3450.0

        mock_point2 = MagicMock()
        mock_point2.id = "978:UT:SNTL"
        mock_point2.name = "Brighton"
        mock_point2.x = -111.58
        mock_point2.y = 40.60
        mock_point2.elevation = 2670.0

        mock_snotel_point_data = MagicMock()
        mock_snotel_point_data.points_from_geometry.return_value = [mock_point1, mock_point2]

        mock_imports.return_value = (mock_snotel_point_data, MagicMock())

        pipeline = SnotelPipeline(raw_dir=tmp_path, processed_dir=tmp_path)
        stations = pipeline.get_station_metadata(state="CO")

        assert len(stations) == 1
        assert stations[0].state == "CO"
        assert stations[0].station_id == "1050:CO:SNTL"


class TestDownloadStation:
    """Tests for download_station method."""

    @patch("snowforecast.pipelines.snotel.SnotelPipeline._get_metloom_imports")
    def test_download_single_station(self, mock_imports, tmp_path):
        """Should download data for one station to parquet."""
        # Create mock data
        mock_df = pd.DataFrame({
            "SNOWDEPTH": [50.0, 52.0, 48.0],
            "SWE": [15.0, 16.0, 14.0],
            "TEMPAVG": [-5.0, -3.0, -7.0],
        }, index=pd.date_range("2023-01-01", periods=3, name="datetime"))

        mock_station = MagicMock()
        mock_station.get_daily_data.return_value = mock_df

        mock_snotel_point_data = MagicMock(return_value=mock_station)
        mock_variables = MagicMock()
        mock_variables.SNOWDEPTH = "SNOWDEPTH"
        mock_variables.SWE = "SWE"
        mock_variables.TEMPAVG = "TEMPAVG"

        mock_imports.return_value = (mock_snotel_point_data, mock_variables)

        pipeline = SnotelPipeline(raw_dir=tmp_path, processed_dir=tmp_path)
        result_path = pipeline.download_station(
            station_id="1050:CO:SNTL",
            start_date="2023-01-01",
            end_date="2023-01-03",
        )

        assert result_path.exists()
        assert result_path.suffix == ".parquet"

        # Verify data was saved correctly
        saved_df = pd.read_parquet(result_path)
        assert len(saved_df) == 3


class TestProcess:
    """Tests for process method."""

    def test_process_raw_data(self, tmp_path):
        """Should convert raw data to standardized format."""
        # Create raw data file
        raw_df = pd.DataFrame({
            "SNOWDEPTH": [50.0, 52.0, None],
            "SWE": [15.0, 16.0, 14.0],
            "TEMPAVG": [-5.0, -3.0, -7.0],
        }, index=pd.date_range("2023-01-01", periods=3, tz="UTC", name="datetime"))

        raw_file = tmp_path / "1050_CO_SNTL_2023-01-01_2023-01-03.parquet"
        raw_df.to_parquet(raw_file)

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        pipeline = SnotelPipeline(raw_dir=tmp_path, processed_dir=processed_dir)
        result = pipeline.process(raw_file)

        # Check standardized columns
        assert list(result.columns) == ["station_id", "datetime", "snow_depth_cm", "swe_mm", "temp_avg_c", "quality_flag"]
        assert len(result) == 3
        assert result["station_id"].iloc[0] == "1050:CO:SNTL"

    def test_process_empty_file(self, tmp_path):
        """Should handle empty files gracefully."""
        raw_df = pd.DataFrame()
        raw_file = tmp_path / "empty_2023-01-01_2023-01-03.parquet"
        raw_df.to_parquet(raw_file)

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        pipeline = SnotelPipeline(raw_dir=tmp_path, processed_dir=processed_dir)
        result = pipeline.process(raw_file)

        assert result.empty
        assert list(result.columns) == ["station_id", "datetime", "snow_depth_cm", "swe_mm", "temp_avg_c", "quality_flag"]

    def test_process_multiple_files(self, tmp_path):
        """Should process and combine multiple raw files."""
        # Create two raw data files
        for i, station_id in enumerate(["1050_CO_SNTL", "978_UT_SNTL"]):
            raw_df = pd.DataFrame({
                "SNOWDEPTH": [50.0 + i * 10],
                "SWE": [15.0 + i * 5],
                "TEMPAVG": [-5.0 - i],
            }, index=pd.date_range("2023-01-01", periods=1, tz="UTC", name="datetime"))
            raw_file = tmp_path / f"{station_id}_2023-01-01_2023-01-01.parquet"
            raw_df.to_parquet(raw_file)

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        pipeline = SnotelPipeline(raw_dir=tmp_path, processed_dir=processed_dir)
        result = pipeline.process([
            tmp_path / "1050_CO_SNTL_2023-01-01_2023-01-01.parquet",
            tmp_path / "978_UT_SNTL_2023-01-01_2023-01-01.parquet",
        ])

        assert len(result) == 2
        assert set(result["station_id"].unique()) == {"1050:CO:SNTL", "978:UT:SNTL"}


class TestValidation:
    """Tests for validate method."""

    def test_validate_good_data(self, tmp_path):
        """Should pass validation for complete data."""
        df = pd.DataFrame({
            "station_id": ["1050:CO:SNTL"] * 3,
            "datetime": pd.date_range("2023-01-01", periods=3, tz="UTC"),
            "snow_depth_cm": [127.0, 132.08, 121.92],  # Valid cm values
            "swe_mm": [381.0, 406.4, 355.6],  # Valid mm values
            "temp_avg_c": [-5.0, -3.0, -7.0],  # Valid Celsius
            "quality_flag": ["good", "good", "good"],
        })

        pipeline = SnotelPipeline(raw_dir=tmp_path, processed_dir=tmp_path)
        result = pipeline.validate(df)

        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert result.total_rows == 3
        assert result.missing_pct == 0.0
        assert result.outliers_count == 0
        assert len(result.issues) == 0

    def test_validate_missing_data(self, tmp_path):
        """Should report missing data percentage."""
        df = pd.DataFrame({
            "station_id": ["1050:CO:SNTL"] * 3,
            "datetime": pd.date_range("2023-01-01", periods=3, tz="UTC"),
            "snow_depth_cm": [127.0, None, None],  # 66% missing
            "swe_mm": [381.0, None, None],
            "temp_avg_c": [-5.0, None, None],
            "quality_flag": ["good", "missing", "missing"],
        })

        pipeline = SnotelPipeline(raw_dir=tmp_path, processed_dir=tmp_path)
        result = pipeline.validate(df)

        assert isinstance(result, ValidationResult)
        assert result.valid is False  # Too much missing data
        assert result.missing_pct > 60.0
        assert "missing_by_column" in result.stats

    def test_validate_empty_data(self, tmp_path):
        """Should fail validation for empty data."""
        df = pd.DataFrame()

        pipeline = SnotelPipeline(raw_dir=tmp_path, processed_dir=tmp_path)
        result = pipeline.validate(df)

        assert result.valid is False
        assert result.total_rows == 0
        assert "No data available" in result.issues

    def test_validate_outliers(self, tmp_path):
        """Should detect outlier values."""
        df = pd.DataFrame({
            "station_id": ["1050:CO:SNTL"] * 3,
            "datetime": pd.date_range("2023-01-01", periods=3, tz="UTC"),
            "snow_depth_cm": [127.0, -50.0, 3000.0],  # Negative and too high
            "swe_mm": [381.0, -100.0, 6000.0],  # Negative and too high
            "temp_avg_c": [-5.0, -70.0, 60.0],  # Too cold and too hot
            "quality_flag": ["good", "good", "good"],
        })

        pipeline = SnotelPipeline(raw_dir=tmp_path, processed_dir=tmp_path)
        result = pipeline.validate(df)

        assert result.valid is False
        assert result.outliers_count > 0


class TestUnitsConversion:
    """Tests for unit conversion."""

    def test_units_conversion_inches_to_metric(self, tmp_path):
        """Should convert inches to metric units correctly."""
        # Create raw data in inches (typical SNOTEL format)
        raw_df = pd.DataFrame({
            "SNOWDEPTH": [50.0, 52.0, 48.0],  # inches (should convert to cm)
            "SWE": [15.0, 16.0, 14.0],  # inches (should convert to mm)
            "TEMPAVG": [-5.0, -3.0, -7.0],  # Already in Celsius
        }, index=pd.date_range("2023-01-01", periods=3, tz="UTC", name="datetime"))

        raw_file = tmp_path / "1050_CO_SNTL_2023-01-01_2023-01-03.parquet"
        raw_df.to_parquet(raw_file)

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        pipeline = SnotelPipeline(raw_dir=tmp_path, processed_dir=processed_dir)
        result = pipeline.process(raw_file)

        # Check that inches were converted to cm
        # 50 inches * 2.54 = 127 cm
        assert abs(result["snow_depth_cm"].iloc[0] - 127.0) < 1.0

        # Check that inches were converted to mm
        # 15 inches * 25.4 = 381 mm
        assert abs(result["swe_mm"].iloc[0] - 381.0) < 1.0

    def test_fahrenheit_to_celsius(self, tmp_path):
        """Should convert Fahrenheit to Celsius."""
        # Create raw data with temperature in Fahrenheit
        raw_df = pd.DataFrame({
            "SNOWDEPTH": [127.0, 132.0, 122.0],  # Already in cm (>200 threshold)
            "SWE": [381.0, 406.0, 356.0],  # Already in mm (>80 threshold)
            "TEMPAVG": [77.0, 68.0, 86.0],  # Fahrenheit (mean > 50)
        }, index=pd.date_range("2023-07-01", periods=3, tz="UTC", name="datetime"))

        raw_file = tmp_path / "1050_CO_SNTL_2023-07-01_2023-07-03.parquet"
        raw_df.to_parquet(raw_file)

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        pipeline = SnotelPipeline(raw_dir=tmp_path, processed_dir=processed_dir)
        result = pipeline.process(raw_file)

        # Check that Fahrenheit was converted to Celsius
        # 77F = 25C
        assert abs(result["temp_avg_c"].iloc[0] - 25.0) < 1.0


class TestRetryLogic:
    """Tests for network retry logic."""

    def test_retry_with_backoff_success(self, tmp_path):
        """Should succeed after retry."""
        pipeline = SnotelPipeline(raw_dir=tmp_path, processed_dir=tmp_path)

        call_count = 0

        def failing_then_succeeding():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Network error")
            return "success"

        # Reduce retry delay for testing
        pipeline.RETRY_DELAY = 0.01
        result = pipeline._retry_with_backoff(failing_then_succeeding)

        assert result == "success"
        assert call_count == 2

    def test_retry_with_backoff_all_fail(self, tmp_path):
        """Should raise after max retries."""
        pipeline = SnotelPipeline(raw_dir=tmp_path, processed_dir=tmp_path)
        pipeline.RETRY_DELAY = 0.01  # Fast for tests

        def always_fails():
            raise ConnectionError("Network error")

        with pytest.raises(ConnectionError):
            pipeline._retry_with_backoff(always_fails)


class TestQualityFlag:
    """Tests for quality flag computation."""

    def test_quality_flag_good(self, tmp_path):
        """Should mark complete rows as good."""
        pipeline = SnotelPipeline(raw_dir=tmp_path, processed_dir=tmp_path)

        row = pd.Series({
            "snow_depth_cm": 127.0,
            "swe_mm": 381.0,
            "temp_avg_c": -5.0,
        })

        assert pipeline._compute_quality_flag(row) == "good"

    def test_quality_flag_partial(self, tmp_path):
        """Should mark partial rows as partial."""
        pipeline = SnotelPipeline(raw_dir=tmp_path, processed_dir=tmp_path)

        row = pd.Series({
            "snow_depth_cm": 127.0,
            "swe_mm": None,
            "temp_avg_c": -5.0,
        })

        assert pipeline._compute_quality_flag(row) == "partial"

    def test_quality_flag_missing(self, tmp_path):
        """Should mark empty rows as missing."""
        pipeline = SnotelPipeline(raw_dir=tmp_path, processed_dir=tmp_path)

        row = pd.Series({
            "snow_depth_cm": None,
            "swe_mm": None,
            "temp_avg_c": None,
        })

        assert pipeline._compute_quality_flag(row) == "missing"


class TestMetloomImportError:
    """Tests for graceful handling of missing metloom."""

    def test_import_error_message(self, tmp_path):
        """Should provide helpful error message when metloom is missing."""
        pipeline = SnotelPipeline(raw_dir=tmp_path, processed_dir=tmp_path)

        with patch.dict("sys.modules", {"metloom": None, "metloom.pointdata": None}):
            # Force reimport to trigger error
            with patch("builtins.__import__", side_effect=ImportError("No module named 'metloom'")):
                with pytest.raises(ImportError) as exc_info:
                    pipeline._get_metloom_imports()

                assert "metloom is required" in str(exc_info.value)
                assert "pip install" in str(exc_info.value)
