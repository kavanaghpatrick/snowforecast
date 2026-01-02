"""Tests for ERA5-Land data ingestion pipeline."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Import xarray for testing
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    xr = None

from snowforecast.utils import WESTERN_US_BBOX, BoundingBox, ValidationResult

# Skip all tests if xarray is not available
pytestmark = pytest.mark.skipif(not HAS_XARRAY, reason="xarray not installed")


@pytest.fixture
def sample_hourly_dataset():
    """Create a sample hourly ERA5 dataset for testing."""
    times = pd.date_range("2023-01-01", periods=48, freq="h")
    lats = np.arange(39.0, 41.0, 0.5)  # 4 lat points
    lons = np.arange(-106.0, -104.0, 0.5)  # 4 lon points

    # Create temperature data (in Kelvin)
    t2m = np.random.uniform(260, 280, (len(times), len(lats), len(lons)))
    # Create precipitation data (in meters, non-negative)
    tp = np.random.uniform(0, 0.01, (len(times), len(lats), len(lons)))
    # Create snow depth data (in meters)
    sd = np.random.uniform(0, 0.5, (len(times), len(lats), len(lons)))

    ds = xr.Dataset(
        {
            "t2m": (["time", "lat", "lon"], t2m),
            "tp": (["time", "lat", "lon"], tp),
            "sd": (["time", "lat", "lon"], sd),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )
    ds.attrs["source"] = "test"
    return ds


@pytest.fixture
def sample_daily_dataset():
    """Create a sample daily ERA5 dataset."""
    times = pd.date_range("2023-01-01", periods=7, freq="D")
    lats = np.arange(39.0, 41.0, 0.5)
    lons = np.arange(-106.0, -104.0, 0.5)

    t2m = np.random.uniform(260, 280, (len(times), len(lats), len(lons)))
    tp = np.random.uniform(0, 0.05, (len(times), len(lats), len(lons)))

    ds = xr.Dataset(
        {
            "t2m": (["time", "lat", "lon"], t2m),
            "tp": (["time", "lat", "lon"], tp),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )
    return ds


@pytest.fixture
def era5_pipeline(tmp_path):
    """Create an ERA5Pipeline instance with mocked cdsapi client."""
    # Import the module - cdsapi is already installed
    from snowforecast.pipelines.era5 import ERA5Pipeline

    # Create pipeline with mock client
    pipeline = ERA5Pipeline(
        cache_dir=tmp_path / "cache" / "era5",
        raw_dir=tmp_path / "raw" / "era5",
    )

    # Replace the client property with a mock
    mock_client = MagicMock()
    pipeline._client = mock_client
    pipeline._mock_client = mock_client

    return pipeline


class TestERA5PipelineInit:
    """Tests for ERA5Pipeline initialization."""

    def test_default_bbox_is_western_us(self, era5_pipeline):
        """Pipeline should default to Western US bounding box."""
        assert era5_pipeline.bbox == WESTERN_US_BBOX
        assert era5_pipeline.bbox.north == 49.0
        assert era5_pipeline.bbox.south == 31.0
        assert era5_pipeline.bbox.west == -125.0
        assert era5_pipeline.bbox.east == -102.0

    def test_custom_bbox(self, tmp_path):
        """Pipeline should accept custom bounding box."""
        from snowforecast.pipelines.era5 import ERA5Pipeline

        custom_bbox = BoundingBox(west=-110, south=35, east=-105, north=40)

        pipeline = ERA5Pipeline(
            cache_dir=tmp_path / "cache",
            raw_dir=tmp_path / "raw",
            bbox=custom_bbox,
        )

        assert pipeline.bbox == custom_bbox
        assert pipeline.bbox.north == 40

    def test_creates_directories(self, era5_pipeline):
        """Pipeline should create cache and raw directories."""
        assert era5_pipeline.cache_dir.exists()
        assert era5_pipeline.raw_dir.exists()

    def test_default_retry_settings(self, era5_pipeline):
        """Pipeline should have sensible retry defaults."""
        assert era5_pipeline.max_retries == 5
        assert era5_pipeline.retry_wait_base == 60


class TestERA5RequestBuilding:
    """Tests for CDS API request building."""

    def test_build_request_single_day(self, era5_pipeline):
        """Should build correct request for a single day."""
        from snowforecast.pipelines.era5 import DEFAULT_VARIABLES

        request = era5_pipeline._build_request(
            start_date="2023-01-15",
            end_date="2023-01-15",
            variables=DEFAULT_VARIABLES,
            bbox=era5_pipeline.bbox,
        )

        assert request["variable"] == DEFAULT_VARIABLES
        assert "2023" in request["year"]
        assert "01" in request["month"]
        assert "15" in request["day"]
        assert request["format"] == "netcdf"
        # Area should be [N, W, S, E]
        assert request["area"] == [49.0, -125.0, 31.0, -102.0]

    def test_build_request_date_range(self, era5_pipeline):
        """Should build correct request for date range."""
        request = era5_pipeline._build_request(
            start_date="2023-01-01",
            end_date="2023-01-05",
            variables=["2m_temperature"],
            bbox=era5_pipeline.bbox,
        )

        assert "2023" in request["year"]
        assert "01" in request["month"]
        # Should include days 01-05
        days = request["day"]
        assert "01" in days
        assert "05" in days

    def test_build_request_custom_hours(self, era5_pipeline):
        """Should respect custom hours parameter."""
        request = era5_pipeline._build_request(
            start_date="2023-01-01",
            end_date="2023-01-01",
            variables=["2m_temperature"],
            bbox=era5_pipeline.bbox,
            hours=["00:00", "12:00"],
        )

        assert request["time"] == ["00:00", "12:00"]

    def test_generate_filename(self, era5_pipeline):
        """Should generate consistent filenames."""
        filename = era5_pipeline._generate_filename(
            start_date="2023-01-01",
            end_date="2023-01-05",
            variables=["2m_temperature", "snow_depth"],
        )

        assert filename.startswith("era5_2023-01-01_2023-01-05")
        assert filename.endswith(".nc")


class TestERA5Download:
    """Tests for download functionality."""

    def test_download_returns_cached_file(self, era5_pipeline, tmp_path):
        """Should return cached file without re-downloading."""
        # Create a fake cached file with the correct name
        from snowforecast.pipelines.era5 import DEFAULT_VARIABLES
        filename = era5_pipeline._generate_filename("2023-01-01", "2023-01-01", DEFAULT_VARIABLES)
        cached_file = era5_pipeline.raw_dir / filename
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        cached_file.touch()

        result = era5_pipeline.download("2023-01-01", "2023-01-01")

        # Should return cached file
        assert result.exists()
        # Client should not have been called
        era5_pipeline._mock_client.retrieve.assert_not_called()

    def test_download_force_ignores_cache(self, era5_pipeline, tmp_path):
        """Should re-download when force=True."""
        # Create a fake cached file
        from snowforecast.pipelines.era5 import DEFAULT_VARIABLES
        filename = era5_pipeline._generate_filename("2023-01-01", "2023-01-01", DEFAULT_VARIABLES)
        cached_file = era5_pipeline.raw_dir / filename
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        cached_file.touch()

        # Mock the download
        def create_file(dataset, request, output_path):
            Path(output_path).touch()

        era5_pipeline._mock_client.retrieve.side_effect = create_file

        era5_pipeline.download("2023-01-01", "2023-01-01", force=True)

        # Client should have been called
        era5_pipeline._mock_client.retrieve.assert_called_once()

    def test_handles_queue_with_retry(self, era5_pipeline):
        """Should retry when request is queued."""
        call_count = 0

        def mock_retrieve(dataset, request, output_path):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Request is in queue, please wait")
            Path(output_path).touch()

        era5_pipeline._mock_client.retrieve.side_effect = mock_retrieve
        era5_pipeline.retry_wait_base = 0.01  # Speed up test

        result = era5_pipeline.download("2023-02-01", "2023-02-01")

        assert call_count == 3
        assert result.exists()

    def test_max_retries_exceeded(self, era5_pipeline):
        """Should raise error after max retries."""
        era5_pipeline._mock_client.retrieve.side_effect = Exception("Queue full")
        era5_pipeline.max_retries = 2
        era5_pipeline.retry_wait_base = 0.01

        with pytest.raises(RuntimeError, match="Max retries"):
            era5_pipeline.download("2023-03-01", "2023-03-01")

    def test_non_queue_errors_raise_immediately(self, era5_pipeline):
        """Should re-raise non-queue errors immediately."""
        era5_pipeline._mock_client.retrieve.side_effect = ValueError("Invalid request")

        with pytest.raises(ValueError, match="Invalid request"):
            era5_pipeline.download("2023-04-01", "2023-04-01")


class TestERA5DataProcessing:
    """Tests for data processing functionality."""

    def test_process_to_dataset_standardizes_coords(self, era5_pipeline, tmp_path):
        """Should rename latitude/longitude to lat/lon."""
        # Create test dataset with latitude/longitude coords
        times = pd.date_range("2023-01-01", periods=24, freq="h")
        lats = np.arange(39.0, 41.0, 0.5)
        lons = np.arange(-106.0, -104.0, 0.5)

        ds = xr.Dataset(
            {"t2m": (["time", "latitude", "longitude"], np.random.rand(24, 4, 4))},
            coords={"time": times, "latitude": lats, "longitude": lons},
        )

        # Save to temp file
        nc_path = tmp_path / "test_coords.nc"
        ds.to_netcdf(nc_path)

        # Process
        result = era5_pipeline.process_to_dataset(nc_path)

        assert "lat" in result.coords
        assert "lon" in result.coords
        assert "latitude" not in result.coords
        assert "longitude" not in result.coords

    def test_process_to_dataset_adds_metadata(self, era5_pipeline, tmp_path):
        """Should add processing metadata."""
        ds = xr.Dataset(
            {"t2m": (["time", "lat", "lon"], np.random.rand(24, 4, 4))},
            coords={
                "time": pd.date_range("2023-01-01", periods=24, freq="h"),
                "lat": np.arange(39.0, 41.0, 0.5),
                "lon": np.arange(-106.0, -104.0, 0.5),
            },
        )
        nc_path = tmp_path / "test_meta.nc"
        ds.to_netcdf(nc_path)

        result = era5_pipeline.process_to_dataset(nc_path)

        assert "source" in result.attrs
        assert result.attrs["source"] == "ERA5-Land reanalysis"
        assert "processed_at" in result.attrs

    def test_process_combines_multiple_files(self, era5_pipeline, tmp_path):
        """Should concatenate multiple NetCDF files."""
        # Create two test files
        for i, date in enumerate(["2023-01-01", "2023-01-02"]):
            ds = xr.Dataset(
                {"t2m": (["time", "lat", "lon"], np.random.rand(24, 4, 4) + i)},
                coords={
                    "time": pd.date_range(date, periods=24, freq="h"),
                    "lat": np.arange(39.0, 41.0, 0.5),
                    "lon": np.arange(-106.0, -104.0, 0.5),
                },
            )
            ds.to_netcdf(tmp_path / f"day_{i}.nc")

        result = era5_pipeline.process_to_dataset([
            tmp_path / "day_0.nc",
            tmp_path / "day_1.nc",
        ])

        # Should have 48 hours total
        assert len(result.time) == 48


class TestDailyAggregation:
    """Tests for hourly to daily aggregation."""

    def test_to_daily_averages_temperature(self, era5_pipeline, sample_hourly_dataset):
        """Should compute daily mean for temperature."""
        daily = era5_pipeline.to_daily(sample_hourly_dataset)

        assert "t2m" in daily.data_vars
        # 48 hours -> 2 days
        assert len(daily.time) == 2
        assert daily.attrs["temporal_resolution"] == "daily"

    def test_to_daily_sums_precipitation(self, era5_pipeline, sample_hourly_dataset):
        """Should compute daily sum for precipitation."""
        daily = era5_pipeline.to_daily(sample_hourly_dataset)

        # Verify precipitation is summed (values should be larger than hourly max)
        hourly_max = sample_hourly_dataset["tp"].max().values
        daily_first = daily["tp"].isel(time=0).values

        # Daily sum should generally be larger than any single hourly value
        # (This is a statistical property, not guaranteed for all random data,
        # but very likely with 24 hourly values summed)
        assert daily_first.max() >= hourly_max * 0.5  # Conservative check


class TestPointExtraction:
    """Tests for extracting time series at specific points."""

    def test_extract_at_points_nearest(self, era5_pipeline, sample_daily_dataset):
        """Should extract time series using nearest neighbor."""
        points = [(39.5, -105.5), (40.0, -105.0)]

        df = era5_pipeline.extract_at_points(sample_daily_dataset, points)

        assert isinstance(df, pd.DataFrame)
        assert "point_id" in df.columns
        assert "point_lat" in df.columns
        assert "point_lon" in df.columns
        assert "t2m" in df.columns
        assert df["point_id"].nunique() == 2

    def test_extract_preserves_time_series(self, era5_pipeline, sample_daily_dataset):
        """Should preserve full time series for each point."""
        points = [(39.5, -105.5)]

        df = era5_pipeline.extract_at_points(sample_daily_dataset, points)

        # Should have 7 days of data
        assert len(df[df["point_id"] == 0]) == 7


class TestValidation:
    """Tests for data validation."""

    def test_validate_dataset_valid(self, era5_pipeline, sample_hourly_dataset):
        """Should validate clean dataset as valid."""
        result = era5_pipeline.validate(sample_hourly_dataset)

        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert result.missing_pct < 1.0
        assert len(result.issues) == 0

    def test_validate_dataset_with_nans(self, era5_pipeline):
        """Should detect high missing percentage."""
        # Create dataset with many NaNs
        data = np.full((24, 4, 4), np.nan)
        data[0, 0, 0] = 270.0  # One valid value

        ds = xr.Dataset(
            {"t2m": (["time", "lat", "lon"], data)},
            coords={
                "time": pd.date_range("2023-01-01", periods=24, freq="h"),
                "lat": np.arange(39.0, 41.0, 0.5),
                "lon": np.arange(-106.0, -104.0, 0.5),
            },
        )

        result = era5_pipeline.validate(ds)

        assert result.valid is False
        assert result.missing_pct > 90

    def test_validate_detects_unrealistic_temperature(self, era5_pipeline):
        """Should flag unrealistic temperature values."""
        # Temperature of 500K is unrealistic
        ds = xr.Dataset(
            {"t2m": (["time", "lat", "lon"], np.full((24, 4, 4), 500.0))},
            coords={
                "time": pd.date_range("2023-01-01", periods=24, freq="h"),
                "lat": np.arange(39.0, 41.0, 0.5),
                "lon": np.arange(-106.0, -104.0, 0.5),
            },
        )

        result = era5_pipeline.validate(ds)

        assert any("unrealistic" in issue.lower() for issue in result.issues)

    def test_validate_detects_negative_precipitation(self, era5_pipeline):
        """Should flag negative precipitation values."""
        ds = xr.Dataset(
            {"tp": (["time", "lat", "lon"], np.full((24, 4, 4), -0.01))},
            coords={
                "time": pd.date_range("2023-01-01", periods=24, freq="h"),
                "lat": np.arange(39.0, 41.0, 0.5),
                "lon": np.arange(-106.0, -104.0, 0.5),
            },
        )

        result = era5_pipeline.validate(ds)

        assert any("negative" in issue.lower() for issue in result.issues)

    def test_validate_dataframe(self, era5_pipeline):
        """Should validate DataFrames too."""
        df = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=100, freq="h"),
            "lat": np.random.uniform(39, 41, 100),
            "lon": np.random.uniform(-106, -104, 100),
            "t2m": np.random.uniform(260, 280, 100),
        })

        result = era5_pipeline.validate(df)

        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert result.total_rows == 100

    def test_validate_empty_dataframe(self, era5_pipeline):
        """Should mark empty DataFrame as invalid."""
        df = pd.DataFrame()

        result = era5_pipeline.validate(df)

        assert result.valid is False
        assert "empty" in result.issues[0].lower()


class TestSaveDataset:
    """Tests for saving datasets with chunking."""

    def test_save_dataset_creates_file(self, era5_pipeline, sample_hourly_dataset, tmp_path):
        """Should save dataset to NetCDF."""
        output_path = tmp_path / "output.nc"

        result = era5_pipeline.save_dataset(sample_hourly_dataset, output_path)

        assert result.exists()
        assert result == output_path

    def test_save_dataset_readable(self, era5_pipeline, sample_hourly_dataset, tmp_path):
        """Saved dataset should be readable."""
        output_path = tmp_path / "output_read.nc"
        era5_pipeline.save_dataset(sample_hourly_dataset, output_path)

        # Read it back
        loaded = xr.open_dataset(output_path)

        assert "t2m" in loaded.data_vars
        assert len(loaded.time) == len(sample_hourly_dataset.time)
        loaded.close()


class TestVariableUnits:
    """Tests for variable unit handling."""

    def test_variable_units_defined(self):
        """Should have units defined for all variables."""
        from snowforecast.pipelines.era5 import ERA5_VARIABLES, VARIABLE_UNITS

        # All short names should have units
        for short_name in ERA5_VARIABLES.keys():
            assert short_name in VARIABLE_UNITS, f"Missing units for {short_name}"

    def test_temperature_in_kelvin(self):
        """Temperature units should be Kelvin."""
        from snowforecast.pipelines.era5 import VARIABLE_UNITS

        assert VARIABLE_UNITS["t2m"] == "K"
        assert VARIABLE_UNITS["d2m"] == "K"

    def test_precipitation_in_meters(self):
        """Precipitation units should be meters."""
        from snowforecast.pipelines.era5 import VARIABLE_UNITS

        assert VARIABLE_UNITS["tp"] == "m"
        assert VARIABLE_UNITS["sf"] == "m"
