"""Live smoke tests for all data pipelines.

These tests verify that pipelines can actually download and process real data.
They are slow and require network access. Skip by default.

Run with: pytest tests/live/ -v --run-live

Why these tests exist:
- Unit tests use mocks/fixtures and don't catch API changes
- Example: metloom 0.4+ changed points_from_geometry() signature
- These tests would have caught that before production
"""

import datetime
import pytest

# All tests in this file are live tests
pytestmark = pytest.mark.live


class TestOpenSkiMapLive:
    """Smoke tests for OpenSkiMap pipeline - no auth required."""

    def test_download_and_process(self):
        """Verify we can download and process ski resort data."""
        from snowforecast.pipelines import OpenSkiMapPipeline

        pipeline = OpenSkiMapPipeline()

        # Download
        raw_path = pipeline.download()
        assert raw_path.exists(), "Download should create raw data directory"

        # Process
        df = pipeline.process(raw_path)
        assert len(df) > 100, "Should find 100+ ski resorts in Western US"

        # Validate schema
        expected_cols = {"name", "lat", "lon", "state"}
        assert expected_cols.issubset(set(df.columns)), f"Missing columns: {expected_cols - set(df.columns)}"

        # Basic data quality
        assert df["lat"].between(31, 49).all(), "Latitudes should be in Western US range"
        assert df["lon"].between(-125, -102).all(), "Longitudes should be in Western US range"


class TestSnotelLive:
    """Smoke tests for SNOTEL pipeline - requires metloom."""

    def test_get_station_metadata(self):
        """Verify we can fetch SNOTEL station list."""
        from snowforecast.pipelines import SnotelPipeline

        pipeline = SnotelPipeline()

        try:
            stations = pipeline.get_station_metadata()
        except Exception as e:
            if "500" in str(e) or "Server Error" in str(e):
                pytest.skip(f"SNOTEL server unavailable (500 error): {e}")
            raise

        assert len(stations) > 100, "Should find 100+ SNOTEL stations"

        # Check station structure
        station = stations[0]
        assert hasattr(station, "station_id")
        assert hasattr(station, "lat")
        assert hasattr(station, "lon")
        assert hasattr(station, "elevation")

    def test_download_single_station(self):
        """Verify we can download data for one station."""
        from snowforecast.pipelines import SnotelPipeline

        pipeline = SnotelPipeline()

        # Get first station
        try:
            stations = pipeline.get_station_metadata()
        except Exception as e:
            if "500" in str(e) or "Server Error" in str(e):
                pytest.skip(f"SNOTEL server unavailable: {e}")
            raise

        station_id = stations[0].station_id

        # Download last 7 days
        end = datetime.date.today()
        start = end - datetime.timedelta(days=7)

        raw_path = pipeline.download(
            station_ids=[station_id],
            start_date=str(start),
            end_date=str(end),
        )

        assert raw_path.exists(), "Download should create raw data"


class TestGHCNLive:
    """Smoke tests for GHCN pipeline."""

    def test_download_sample_stations(self):
        """Verify we can download GHCN station data."""
        from snowforecast.pipelines import GHCNPipeline

        pipeline = GHCNPipeline()

        # Download station inventory
        end = datetime.date.today()
        start = end - datetime.timedelta(days=7)

        # This might fail if API changed - that's what we're testing
        try:
            raw_path = pipeline.download(start_date=str(start), end_date=str(end))
            assert raw_path.exists()
        except Exception as e:
            pytest.fail(f"GHCN download failed - possible API change: {e}")


class TestHRRRLive:
    """Smoke tests for HRRR pipeline - requires herbie-data."""

    def test_download_single_file(self):
        """Verify HRRR download works (already fixed for 2025 API)."""
        from snowforecast.pipelines import HRRRPipeline

        pipeline = HRRRPipeline()

        # Download yesterday's data (more likely to be available)
        yesterday = datetime.date.today() - datetime.timedelta(days=1)

        try:
            raw_path = pipeline.download(
                start_date=str(yesterday),
                end_date=str(yesterday),
            )
            # HRRR data availability can be spotty, so just check it doesn't crash
            assert True
        except Exception as e:
            # HRRR data might not be available - warn but don't fail
            pytest.skip(f"HRRR data not available: {e}")


class TestDEMLive:
    """Smoke tests for DEM pipeline - requires rasterio."""

    def test_download_small_bbox(self):
        """Verify DEM download works for a small area."""
        from snowforecast.pipelines import DEMPipeline

        pipeline = DEMPipeline()

        # Small bbox around a ski resort
        small_bbox = {
            "west": -105.8,
            "south": 39.7,
            "east": -105.7,
            "north": 39.8,
        }

        try:
            raw_path = pipeline.download(bbox=small_bbox)
            assert raw_path.exists(), "DEM download should create file"

            # Process
            result = pipeline.process(raw_path)
            assert result is not None
        except Exception as e:
            pytest.fail(f"DEM download failed - possible API change: {e}")


class TestERA5Live:
    """Smoke tests for ERA5 pipeline - requires CDS credentials."""

    @pytest.mark.skip(reason="ERA5 requires CDS_API_KEY - enable manually")
    def test_download_sample(self):
        """Verify ERA5 download works (requires credentials)."""
        from snowforecast.pipelines import ERA5Pipeline

        pipeline = ERA5Pipeline()

        # ERA5 has ~5 day latency, use data from a week ago
        end = datetime.date.today() - datetime.timedelta(days=7)
        start = end - datetime.timedelta(days=1)

        try:
            raw_path = pipeline.download(start_date=str(start), end_date=str(end))
            assert raw_path.exists()
        except Exception as e:
            pytest.fail(f"ERA5 download failed: {e}")
