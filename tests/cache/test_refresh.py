"""Tests for background refresh.

Tests use real database operations with pre-populated cache data.
No mocks - tests verify actual refresh behavior.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from snowforecast.cache.database import CacheDatabase
from snowforecast.cache.models import SKI_AREAS_DATA, SkiArea
from snowforecast.cache.refresh import (
    RefreshResult,
    get_cache_status,
    refresh_hrrr_for_ski_areas,
    refresh_terrain_for_ski_areas,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.duckdb"


@pytest.fixture
def db(temp_db_path):
    """Create a CacheDatabase with temp database."""
    database = CacheDatabase(temp_db_path)
    yield database
    database.close()


@pytest.fixture
def test_ski_areas():
    """Subset of ski areas for faster tests."""
    return SKI_AREAS_DATA[:3]  # First 3: Stevens Pass, Crystal Mountain, Mt. Baker


class TestRefreshResult:
    """Tests for RefreshResult dataclass."""

    def test_success_rate_all_success(self):
        """success_rate returns 100 when all succeed."""
        result = RefreshResult(
            total=10, success=10, failed=0, skipped=0, duration_ms=1000
        )
        assert result.success_rate == 100.0

    def test_success_rate_partial(self):
        """success_rate calculates correctly for partial success."""
        result = RefreshResult(
            total=10, success=5, failed=3, skipped=2, duration_ms=1000
        )
        assert result.success_rate == 50.0

    def test_success_rate_all_failed(self):
        """success_rate returns 0 when all fail."""
        result = RefreshResult(
            total=10, success=0, failed=10, skipped=0, duration_ms=1000
        )
        assert result.success_rate == 0.0

    def test_success_rate_empty(self):
        """success_rate returns 0 for empty run."""
        result = RefreshResult(
            total=0, success=0, failed=0, skipped=0, duration_ms=0
        )
        assert result.success_rate == 0.0

    def test_str_representation(self):
        """__str__ returns readable summary."""
        result = RefreshResult(
            total=22, success=20, failed=1, skipped=1, duration_ms=5000
        )
        string = str(result)

        assert "20/22" in string
        assert "1 failed" in string
        assert "1 skipped" in string
        assert "5000ms" in string


class TestRefreshHRRR:
    """Tests for HRRR refresh functionality."""

    def test_refresh_skips_fresh_cache(self, db, test_ski_areas):
        """refresh_hrrr_for_ski_areas skips locations with fresh cache."""
        # Pre-populate cache with fresh data for first ski area
        area = test_ski_areas[0]
        run_time = datetime.utcnow()

        # Use forecast_hour=0 so valid_time = run_time (matches current time check)
        db.store_forecast(
            lat=area.lat,
            lon=area.lon,
            run_time=run_time,
            forecast_hour=0,  # Valid at current time
            snow_depth_m=0.5,
            temp_k=268.0,
            precip_mm=8.0,
            categorical_snow=1.0,
        )

        # Only refresh the first ski area
        single_area = test_ski_areas[:1]
        result = refresh_hrrr_for_ski_areas(db, ski_areas=single_area, force=False)

        # Should skip because cache is fresh
        assert result.skipped == 1
        assert result.success == 0
        assert result.failed == 0

    def test_refresh_force_ignores_cache(self, db, test_ski_areas):
        """refresh_hrrr_for_ski_areas with force=True ignores fresh cache."""
        # Pre-populate cache with fresh data
        area = test_ski_areas[0]
        run_time = datetime.utcnow()

        db.store_forecast(
            lat=area.lat,
            lon=area.lon,
            run_time=run_time,
            forecast_hour=0,  # Valid at current time
            snow_depth_m=0.5,
            temp_k=268.0,
            precip_mm=8.0,
            categorical_snow=1.0,
        )

        # Only test with first area to avoid network calls
        single_area = test_ski_areas[:1]

        # With force=True, it should attempt to fetch even if cached
        # This will fail without network but demonstrates force logic
        result = refresh_hrrr_for_ski_areas(db, ski_areas=single_area, force=True)

        # Should not be skipped when force=True
        assert result.skipped == 0
        # Will either succeed (if network available) or fail
        assert result.success + result.failed == 1

    def test_refresh_returns_valid_result_type(self, db, test_ski_areas):
        """refresh_hrrr_for_ski_areas returns RefreshResult."""
        # Pre-populate cache to avoid network calls
        for area in test_ski_areas:
            db.store_forecast(
                lat=area.lat,
                lon=area.lon,
                run_time=datetime.utcnow(),
                forecast_hour=0,  # Valid at current time
                snow_depth_m=0.5,
                temp_k=268.0,
                precip_mm=8.0,
                categorical_snow=1.0,
            )

        result = refresh_hrrr_for_ski_areas(db, ski_areas=test_ski_areas)

        assert isinstance(result, RefreshResult)
        assert result.total == len(test_ski_areas)
        assert result.duration_ms >= 0

    def test_refresh_handles_stale_cache(self, db, test_ski_areas):
        """refresh_hrrr_for_ski_areas fetches when cache is stale."""
        # Store old data (beyond 2-hour validity window)
        area = test_ski_areas[0]
        old_run_time = datetime.utcnow() - timedelta(hours=3)

        db.store_forecast(
            lat=area.lat,
            lon=area.lon,
            run_time=old_run_time,
            forecast_hour=0,  # Valid at run_time (which is 3 hours ago)
            snow_depth_m=0.5,
            temp_k=268.0,
            precip_mm=8.0,
            categorical_snow=1.0,
        )

        # Should attempt to fetch because cache is stale
        single_area = test_ski_areas[:1]
        result = refresh_hrrr_for_ski_areas(db, ski_areas=single_area)

        # Should not be skipped (cache is stale)
        assert result.skipped == 0
        # Will either succeed or fail based on network
        assert result.total == 1


class TestRefreshTerrain:
    """Tests for terrain refresh functionality."""

    def test_refresh_skips_cached_terrain(self, db, test_ski_areas):
        """refresh_terrain_for_ski_areas skips already-cached locations."""
        # Pre-populate terrain cache
        for area in test_ski_areas:
            db.store_terrain(
                lat=area.lat,
                lon=area.lon,
                elevation=area.base_elevation,
                slope=15.0,
                aspect=180.0,
                roughness=50.0,
                tpi=0.5,
            )

        result = refresh_terrain_for_ski_areas(db, ski_areas=test_ski_areas)

        # All should be skipped (already cached)
        assert result.skipped == len(test_ski_areas)
        assert result.success == 0
        assert result.failed == 0

    def test_refresh_terrain_returns_valid_result(self, db, test_ski_areas):
        """refresh_terrain_for_ski_areas returns RefreshResult."""
        # Pre-populate to avoid network calls
        for area in test_ski_areas:
            db.store_terrain(
                lat=area.lat,
                lon=area.lon,
                elevation=area.base_elevation,
                slope=15.0,
                aspect=180.0,
                roughness=50.0,
                tpi=0.5,
            )

        result = refresh_terrain_for_ski_areas(db, ski_areas=test_ski_areas)

        assert isinstance(result, RefreshResult)
        assert result.total == len(test_ski_areas)
        assert result.duration_ms >= 0

    @pytest.mark.integration
    def test_refresh_terrain_counts_correctly(self, db, test_ski_areas):
        """refresh_terrain_for_ski_areas counts operations correctly."""
        # Cache only first area
        first = test_ski_areas[0]
        db.store_terrain(
            lat=first.lat,
            lon=first.lon,
            elevation=first.base_elevation,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        result = refresh_terrain_for_ski_areas(db, ski_areas=test_ski_areas)

        # First should be skipped, others attempted
        assert result.skipped >= 1
        assert result.total == len(test_ski_areas)


class TestCacheStatus:
    """Tests for cache status functionality."""

    def test_get_cache_status_empty_db(self, db):
        """get_cache_status works on empty database."""
        status = get_cache_status(db.db_path)

        assert status["total_ski_areas"] == len(SKI_AREAS_DATA)
        assert status["hrrr_cached"] == 0
        assert status["terrain_cached"] == 0
        assert status["forecast_count"] == 0
        assert status["terrain_count"] == 0

    def test_get_cache_status_with_data(self, db, test_ski_areas):
        """get_cache_status reflects stored data."""
        # Store HRRR for first area - use forecast_hour=0 for current time validity
        first = test_ski_areas[0]
        db.store_forecast(
            lat=first.lat,
            lon=first.lon,
            run_time=datetime.utcnow(),
            forecast_hour=0,  # Valid at current time
            snow_depth_m=0.5,
            temp_k=268.0,
            precip_mm=8.0,
            categorical_snow=1.0,
        )

        # Store terrain for first two areas
        for area in test_ski_areas[:2]:
            db.store_terrain(
                lat=area.lat,
                lon=area.lon,
                elevation=area.base_elevation,
                slope=15.0,
                aspect=180.0,
                roughness=50.0,
                tpi=0.5,
            )

        status = get_cache_status(db.db_path)

        assert status["hrrr_cached"] == 1
        assert status["terrain_cached"] == 2
        assert status["forecast_count"] == 1
        assert status["terrain_count"] == 2

    def test_get_cache_status_includes_ski_area_details(self, db, test_ski_areas):
        """get_cache_status includes per-ski-area status."""
        status = get_cache_status(db.db_path)

        assert "ski_areas" in status
        assert len(status["ski_areas"]) == len(SKI_AREAS_DATA)

        for area_status in status["ski_areas"]:
            assert "name" in area_status
            assert "state" in area_status
            assert "hrrr_cached" in area_status
            assert "terrain_cached" in area_status

    def test_get_cache_status_returns_db_path(self, db):
        """get_cache_status includes database path."""
        status = get_cache_status(db.db_path)

        assert "db_path" in status
        assert str(db.db_path) in status["db_path"]


class TestRefreshWithAllSkiAreas:
    """Tests using the full ski areas list."""

    def test_all_ski_areas_are_known(self, db):
        """Verify all 22 ski areas are in the list."""
        assert len(SKI_AREAS_DATA) == 22

    def test_ski_areas_have_valid_coordinates(self, db):
        """Verify all ski areas have valid lat/lon."""
        for area in SKI_AREAS_DATA:
            assert isinstance(area, SkiArea)
            assert -90 <= area.lat <= 90
            assert -180 <= area.lon <= 180
            assert area.base_elevation > 0

    @pytest.mark.integration
    def test_refresh_counts_match_total(self, db):
        """Verify refresh counts add up to total."""
        # Pre-populate some data
        for i, area in enumerate(SKI_AREAS_DATA):
            if i % 3 == 0:  # Every 3rd area has HRRR
                db.store_forecast(
                    lat=area.lat,
                    lon=area.lon,
                    run_time=datetime.utcnow(),
                    forecast_hour=0,  # Valid at current time
                    snow_depth_m=0.5,
                    temp_k=268.0,
                    precip_mm=8.0,
                    categorical_snow=1.0,
                )
            if i % 2 == 0:  # Every 2nd area has terrain
                db.store_terrain(
                    lat=area.lat,
                    lon=area.lon,
                    elevation=area.base_elevation,
                    slope=15.0,
                    aspect=180.0,
                    roughness=50.0,
                    tpi=0.5,
                )

        # Check terrain refresh counts
        terrain_result = refresh_terrain_for_ski_areas(db)
        assert terrain_result.success + terrain_result.failed + terrain_result.skipped == terrain_result.total
        assert terrain_result.total == len(SKI_AREAS_DATA)

        # Check HRRR refresh counts
        hrrr_result = refresh_hrrr_for_ski_areas(db)
        assert hrrr_result.success + hrrr_result.failed + hrrr_result.skipped == hrrr_result.total
        assert hrrr_result.total == len(SKI_AREAS_DATA)


class TestRefreshErrorHandling:
    """Tests for error handling in refresh operations."""

    def test_refresh_continues_after_error(self, db):
        """Refresh continues processing after one area fails."""
        # Create a custom ski area list with an invalid location
        # that might cause issues but won't crash
        areas = [
            SKI_AREAS_DATA[0],  # Valid
            SkiArea("Invalid", 0.0, 0.0, "Test", 100),  # Edge case
            SKI_AREAS_DATA[1],  # Valid
        ]

        # Store data for valid areas to ensure they're skipped
        for area in [areas[0], areas[2]]:
            db.store_terrain(
                lat=area.lat,
                lon=area.lon,
                elevation=area.base_elevation,
                slope=15.0,
                aspect=180.0,
                roughness=50.0,
                tpi=0.5,
            )

        result = refresh_terrain_for_ski_areas(db, ski_areas=areas)

        # Should process all 3, regardless of any issues
        assert result.total == 3

    def test_refresh_handles_empty_list(self, db):
        """Refresh handles empty ski area list gracefully."""
        result = refresh_terrain_for_ski_areas(db, ski_areas=[])

        assert result.total == 0
        assert result.success == 0
        assert result.failed == 0
        assert result.skipped == 0


class TestRefreshDatabaseIntegrity:
    """Tests for database integrity during refresh."""

    def test_refresh_preserves_existing_data(self, db, test_ski_areas):
        """Refresh doesn't corrupt existing cache data."""
        # Store initial data
        area = test_ski_areas[0]
        initial_elevation = 1500.0

        db.store_terrain(
            lat=area.lat,
            lon=area.lon,
            elevation=initial_elevation,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        # Run terrain refresh (should skip this area)
        refresh_terrain_for_ski_areas(db, ski_areas=test_ski_areas)

        # Verify original data is preserved
        terrain = db.get_terrain(area.lat, area.lon)
        assert terrain.elevation == initial_elevation

    def test_refresh_updates_stats_correctly(self, db, test_ski_areas):
        """Refresh updates database stats correctly."""
        initial_stats = db.get_stats()
        assert initial_stats["terrain_count"] == 0

        # Pre-populate terrain
        for area in test_ski_areas:
            db.store_terrain(
                lat=area.lat,
                lon=area.lon,
                elevation=area.base_elevation,
                slope=15.0,
                aspect=180.0,
                roughness=50.0,
                tpi=0.5,
            )

        refresh_terrain_for_ski_areas(db, ski_areas=test_ski_areas)

        final_stats = db.get_stats()
        assert final_stats["terrain_count"] == len(test_ski_areas)
