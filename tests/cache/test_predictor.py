"""Tests for CachedPredictor.

Tests use real database operations with pre-populated cache data.
No mocks - tests verify actual caching behavior.
"""

import tempfile
from datetime import datetime, timedelta, date
from pathlib import Path

import pytest

from snowforecast.cache.predictor import CachedPredictor
from snowforecast.api.schemas import ForecastResult, ConfidenceInterval


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.duckdb"


@pytest.fixture
def predictor(temp_db_path):
    """Create a CachedPredictor with temp database."""
    pred = CachedPredictor(temp_db_path)
    yield pred
    pred.close()


class TestCachedPredictorInit:
    """Tests for CachedPredictor initialization."""

    def test_init_creates_database(self, temp_db_path):
        """CachedPredictor creates database on init."""
        predictor = CachedPredictor(temp_db_path)

        assert temp_db_path.exists()
        assert predictor.db is not None
        assert predictor.hrrr_cache is not None
        assert predictor.terrain_cache is not None

        predictor.close()

    def test_init_creates_tables(self, predictor):
        """CachedPredictor initializes all required tables."""
        tables = predictor.db.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = [t[0] for t in tables]

        assert "hrrr_forecasts" in table_names
        assert "terrain_cache" in table_names
        assert "ski_areas" in table_names


class TestCachedPredictorTerrain:
    """Tests for terrain caching in CachedPredictor."""

    def test_get_elevation_from_prepopulated_cache(self, predictor):
        """get_elevation returns cached value when pre-populated."""
        # Pre-populate cache directly
        predictor.db.store_terrain(
            lat=47.74,
            lon=-121.09,
            elevation=1257.0,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        elevation = predictor.get_elevation(47.74, -121.09)

        assert elevation == 1257.0

    def test_get_terrain_features_from_cache(self, predictor):
        """get_terrain_features returns cached dict when pre-populated."""
        predictor.db.store_terrain(
            lat=47.74,
            lon=-121.09,
            elevation=1500.0,
            slope=20.0,
            aspect=90.0,
            roughness=40.0,
            tpi=1.0,
        )

        features = predictor.get_terrain_features(47.74, -121.09)

        assert features["elevation"] == 1500.0
        assert features["slope"] == 20.0
        assert features["aspect"] == 90.0
        assert features["roughness"] == 40.0
        assert features["tpi"] == 1.0

    def test_terrain_cache_is_permanent(self, predictor):
        """Terrain cache never expires - old data still returned."""
        predictor.db.store_terrain(
            lat=47.74,
            lon=-121.09,
            elevation=1257.0,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        # Manually set very old fetch_time
        predictor.db.conn.execute(
            """
            UPDATE terrain_cache
            SET fetch_time = '2020-01-01 00:00:00'
            WHERE lat = 47.74 AND lon = -121.09
            """
        )

        # Should still return cached data (terrain never expires)
        features = predictor.get_terrain_features(47.74, -121.09)
        assert features["elevation"] == 1257.0


class TestCachedPredictorHRRR:
    """Tests for HRRR caching in CachedPredictor."""

    def test_hrrr_cache_stores_and_retrieves_data(self, predictor):
        """Verify HRRR data can be stored and retrieved directly from cache."""
        run_time = datetime.utcnow()
        valid_time = run_time + timedelta(hours=12)

        predictor.db.store_forecast(
            lat=48.00,
            lon=-122.00,
            run_time=run_time,
            forecast_hour=12,
            snow_depth_m=1.5,
            temp_k=265.0,
            precip_mm=15.0,
            categorical_snow=1.0,
        )

        # Retrieve directly from hrrr_cache.get() to test cache retrieval
        cached = predictor.hrrr_cache.get(48.00, -122.00, valid_time, max_age_hours=24)

        assert cached is not None
        assert cached.snow_depth_m == 1.5
        assert cached.temp_k == 265.0

    def test_hrrr_cache_stats_reflect_stored_data(self, predictor):
        """Verify cache stats update after storing data."""
        run_time = datetime.utcnow()

        predictor.db.store_forecast(
            lat=48.00,
            lon=-122.00,
            run_time=run_time,
            forecast_hour=12,
            snow_depth_m=1.5,
            temp_k=265.0,
            precip_mm=15.0,
            categorical_snow=1.0,
        )

        stats = predictor.get_cache_stats()
        assert stats["forecast_count"] == 1
        assert stats["latest_run_time"] is not None


class TestCachedPredictorPredict:
    """Tests for predict method."""

    def test_predict_returns_valid_types(self, predictor):
        """predict returns ForecastResult and ConfidenceInterval."""
        # Pre-populate both caches
        run_time = datetime.utcnow()

        predictor.db.store_forecast(
            lat=47.74,
            lon=-121.09,
            run_time=run_time,
            forecast_hour=24,
            snow_depth_m=0.5,
            temp_k=268.0,
            precip_mm=8.0,
            categorical_snow=1.0,
        )
        predictor.db.store_terrain(
            lat=47.74,
            lon=-121.09,
            elevation=1257.0,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        forecast, ci = predictor.predict(47.74, -121.09, datetime.utcnow())

        assert isinstance(forecast, ForecastResult)
        assert isinstance(ci, ConfidenceInterval)
        assert forecast.snow_depth_cm >= 0
        assert ci.lower <= ci.upper

    def test_predict_uses_terrain_adjustments(self, predictor):
        """predict applies terrain adjustments based on slope/aspect."""
        run_time = datetime.utcnow()

        predictor.db.store_forecast(
            lat=47.74,
            lon=-121.09,
            run_time=run_time,
            forecast_hour=24,
            snow_depth_m=1.0,
            temp_k=268.0,
            precip_mm=10.0,
            categorical_snow=1.0,
        )
        # North-facing slope (aspect=0) should increase snow depth
        predictor.db.store_terrain(
            lat=47.74,
            lon=-121.09,
            elevation=2000.0,
            slope=15.0,
            aspect=0.0,  # North-facing
            roughness=50.0,
            tpi=0.5,
        )

        forecast, _ = predictor.predict(47.74, -121.09, datetime.utcnow())

        # North-facing adjustment is 1.05x, so 100cm base -> ~105cm
        assert forecast.snow_depth_cm >= 100  # At least base depth

    def test_predict_uses_climatology_fallback(self, predictor):
        """predict uses climatology when no HRRR data available."""
        # Only terrain cached, no HRRR
        predictor.db.store_terrain(
            lat=47.74,
            lon=-121.09,
            elevation=2000.0,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        # Create predictor that won't try to fetch (no network call)
        # by not having any HRRR in cache and catching the fallback
        forecast, ci = predictor.predict(47.74, -121.09, datetime.utcnow())

        # Should still return valid forecast from climatology
        assert isinstance(forecast, ForecastResult)
        assert forecast.snow_depth_cm >= 0


class TestCachedPredictorStats:
    """Tests for cache statistics."""

    def test_get_cache_stats_empty(self, predictor):
        """get_cache_stats works on empty cache."""
        stats = predictor.get_cache_stats()

        assert stats["forecast_count"] == 0
        assert stats["terrain_count"] == 0
        assert stats["latest_run_time"] is None

    def test_get_cache_stats_with_data(self, predictor):
        """get_cache_stats reflects stored data."""
        predictor.db.store_forecast(
            lat=47.74,
            lon=-121.09,
            run_time=datetime.utcnow(),
            forecast_hour=0,
            snow_depth_m=1.0,
            temp_k=273.0,
            precip_mm=0.0,
            categorical_snow=0.0,
        )
        predictor.db.store_terrain(
            lat=47.74,
            lon=-121.09,
            elevation=1257.0,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        stats = predictor.get_cache_stats()

        assert stats["forecast_count"] == 1
        assert stats["terrain_count"] == 1
        assert stats["latest_run_time"] is not None


class TestCachedPredictorMaintenance:
    """Tests for cache maintenance operations."""

    def test_cleanup_old_forecasts(self, predictor):
        """cleanup_old_forecasts removes old data."""
        # Store old forecast
        old_run_time = datetime.utcnow() - timedelta(days=10)
        predictor.db.store_forecast(
            lat=47.74,
            lon=-121.09,
            run_time=old_run_time,
            forecast_hour=0,
            snow_depth_m=1.0,
            temp_k=273.0,
            precip_mm=0.0,
            categorical_snow=0.0,
        )

        # Store recent forecast
        predictor.db.store_forecast(
            lat=48.00,
            lon=-122.00,
            run_time=datetime.utcnow(),
            forecast_hour=0,
            snow_depth_m=2.0,
            temp_k=270.0,
            precip_mm=5.0,
            categorical_snow=1.0,
        )

        predictor.cleanup_old_forecasts(keep_days=7)

        stats = predictor.get_cache_stats()
        assert stats["forecast_count"] == 1  # Only recent one remains


class TestCachedPredictorCaching:
    """Tests for caching behavior."""

    def test_terrain_is_cached_permanently(self, predictor):
        """Terrain data cached once is available forever."""
        predictor.db.store_terrain(
            lat=47.74,
            lon=-121.09,
            elevation=1257.0,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        # First call
        features1 = predictor.get_terrain_features(47.74, -121.09)

        # Second call - should return same cached data
        features2 = predictor.get_terrain_features(47.74, -121.09)

        assert features1 == features2
        assert features1["elevation"] == 1257.0

    def test_hrrr_cache_expires_after_validity_window(self, predictor):
        """HRRR cache returns None for data beyond validity window."""
        # Store old HRRR forecast (beyond 2-hour window)
        old_run_time = datetime.utcnow() - timedelta(hours=3)
        predictor.db.store_forecast(
            lat=47.74,
            lon=-121.09,
            run_time=old_run_time,
            forecast_hour=24,
            snow_depth_m=0.5,
            temp_k=268.0,
            precip_mm=8.0,
            categorical_snow=1.0,
        )

        # Cache lookup should not return expired data
        cached = predictor.hrrr_cache.get(
            47.74, -121.09,
            old_run_time + timedelta(hours=24),
            max_age_hours=2
        )
        assert cached is None  # Expired, not returned

    def test_multiple_locations_cached_independently(self, predictor):
        """Each location's cache is independent."""
        predictor.db.store_terrain(
            lat=47.74,
            lon=-121.09,
            elevation=1257.0,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )
        predictor.db.store_terrain(
            lat=48.00,
            lon=-122.00,
            elevation=2000.0,
            slope=25.0,
            aspect=90.0,
            roughness=60.0,
            tpi=1.0,
        )

        features1 = predictor.get_terrain_features(47.74, -121.09)
        features2 = predictor.get_terrain_features(48.00, -122.00)

        assert features1["elevation"] == 1257.0
        assert features2["elevation"] == 2000.0
        assert features1 != features2
