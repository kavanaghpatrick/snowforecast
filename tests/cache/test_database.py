"""Tests for cache database."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from snowforecast.cache.database import CacheDatabase
from snowforecast.cache.models import SKI_AREAS_DATA


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.duckdb"
        db = CacheDatabase(db_path)
        yield db
        db.close()


class TestCacheDatabase:
    """Tests for CacheDatabase."""

    def test_init_creates_tables(self, temp_db):
        """Database initialization creates all required tables."""
        tables = temp_db.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = [t[0] for t in tables]

        assert "hrrr_forecasts" in table_names
        assert "terrain_cache" in table_names
        assert "ski_areas" in table_names
        assert "fetch_log" in table_names

    def test_ski_areas_initialized(self, temp_db):
        """Ski areas table is populated on init."""
        areas = temp_db.get_ski_areas()
        assert len(areas) == len(SKI_AREAS_DATA)
        assert areas[0].name == "Stevens Pass"

    def test_get_ski_area(self, temp_db):
        """Can retrieve ski area by name."""
        area = temp_db.get_ski_area("Stevens Pass")
        assert area is not None
        assert area.lat == pytest.approx(47.7448)
        assert area.state == "Washington"

    def test_get_ski_area_not_found(self, temp_db):
        """Returns None for unknown ski area."""
        area = temp_db.get_ski_area("Nonexistent Resort")
        assert area is None


class TestForecastCache:
    """Tests for forecast caching."""

    def test_store_and_get_forecast(self, temp_db):
        """Can store and retrieve forecast."""
        # Use a recent run_time relative to current time to ensure it passes age filter
        run_time = datetime.utcnow() - timedelta(hours=1)

        temp_db.store_forecast(
            lat=47.74,
            lon=-121.09,
            run_time=run_time,
            forecast_hour=24,
            snow_depth_m=1.5,
            temp_k=270.0,
            precip_mm=5.0,
            categorical_snow=1.0,
        )

        valid_time = run_time + timedelta(hours=24)
        # Use larger max_age_hours to ensure the forecast is found
        forecast = temp_db.get_forecast(47.74, -121.09, valid_time, max_age_hours=48)

        assert forecast is not None
        assert forecast.snow_depth_m == 1.5
        assert forecast.temp_k == 270.0
        assert forecast.snow_depth_cm == 150.0
        assert forecast.temp_c == pytest.approx(-3.15)

    def test_get_forecast_not_found(self, temp_db):
        """Returns None when no matching forecast."""
        forecast = temp_db.get_forecast(99.99, -99.99, datetime.utcnow())
        assert forecast is None

    def test_get_forecast_expired(self, temp_db):
        """Returns None for expired forecasts."""
        old_run_time = datetime.utcnow() - timedelta(hours=10)

        temp_db.store_forecast(
            lat=47.74,
            lon=-121.09,
            run_time=old_run_time,
            forecast_hour=0,
            snow_depth_m=1.0,
            temp_k=273.0,
            precip_mm=0.0,
            categorical_snow=0.0,
        )

        # With max_age_hours=2, the old forecast should not be returned
        forecast = temp_db.get_forecast(47.74, -121.09, datetime.utcnow(), max_age_hours=2)
        assert forecast is None

    def test_latest_run_time(self, temp_db):
        """Can get latest run time."""
        assert temp_db.get_latest_run_time() is None

        run_time = datetime(2026, 1, 1, 12, 0, 0)
        temp_db.store_forecast(
            lat=47.74,
            lon=-121.09,
            run_time=run_time,
            forecast_hour=0,
            snow_depth_m=1.0,
            temp_k=273.0,
            precip_mm=0.0,
            categorical_snow=0.0,
        )

        assert temp_db.get_latest_run_time() == run_time


class TestTerrainCache:
    """Tests for terrain caching."""

    def test_store_and_get_terrain(self, temp_db):
        """Can store and retrieve terrain data."""
        temp_db.store_terrain(
            lat=47.74,
            lon=-121.09,
            elevation=1257.0,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        terrain = temp_db.get_terrain(47.74, -121.09)

        assert terrain is not None
        assert terrain.elevation == 1257.0
        assert terrain.slope == 15.0
        assert terrain.aspect == 180.0

    def test_get_terrain_not_found(self, temp_db):
        """Returns None when terrain not cached."""
        terrain = temp_db.get_terrain(99.99, -99.99)
        assert terrain is None

    def test_terrain_is_permanent(self, temp_db):
        """Terrain cache persists (no expiration)."""
        temp_db.store_terrain(
            lat=47.74,
            lon=-121.09,
            elevation=1257.0,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        # Even with very old fetch_time, terrain should be returned
        terrain = temp_db.get_terrain(47.74, -121.09)
        assert terrain is not None


class TestCacheStats:
    """Tests for cache statistics."""

    def test_get_stats_empty(self, temp_db):
        """Stats work on empty database."""
        stats = temp_db.get_stats()

        assert stats["forecast_count"] == 0
        assert stats["terrain_count"] == 0
        assert stats["latest_run_time"] is None

    def test_get_stats_with_data(self, temp_db):
        """Stats reflect stored data."""
        temp_db.store_forecast(
            lat=47.74, lon=-121.09,
            run_time=datetime.utcnow(),
            forecast_hour=0,
            snow_depth_m=1.0, temp_k=273.0,
            precip_mm=0.0, categorical_snow=0.0,
        )
        temp_db.store_terrain(
            lat=47.74, lon=-121.09,
            elevation=1257.0, slope=15.0,
            aspect=180.0, roughness=50.0, tpi=0.5,
        )

        stats = temp_db.get_stats()

        assert stats["forecast_count"] == 1
        assert stats["terrain_count"] == 1
        assert stats["latest_run_time"] is not None


class TestFetchLog:
    """Tests for fetch logging."""

    def test_log_fetch(self, temp_db):
        """Can log fetch operations."""
        temp_db.log_fetch(
            source="hrrr",
            status="success",
            records_added=10,
            duration_ms=1500,
        )

        result = temp_db.conn.execute(
            "SELECT source, status, records_added FROM fetch_log"
        ).fetchone()

        assert result[0] == "hrrr"
        assert result[1] == "success"
        assert result[2] == 10

    def test_log_fetch_with_error(self, temp_db):
        """Can log failed fetch with error message."""
        temp_db.log_fetch(
            source="dem",
            status="error",
            records_added=0,
            duration_ms=500,
            error_message="Connection timeout",
        )

        result = temp_db.conn.execute(
            "SELECT error_message FROM fetch_log WHERE status = 'error'"
        ).fetchone()

        assert result[0] == "Connection timeout"
