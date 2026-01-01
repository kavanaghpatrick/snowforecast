"""Tests for terrain cache layer."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from snowforecast.cache.database import CacheDatabase
from snowforecast.cache.terrain import TerrainCache
from snowforecast.cache.models import CachedTerrain, SKI_AREAS_DATA


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.duckdb"
        db = CacheDatabase(db_path)
        yield db
        db.close()


@pytest.fixture
def terrain_cache(temp_db):
    """Create a TerrainCache with temp database."""
    return TerrainCache(temp_db)


@pytest.fixture
def mock_predictor():
    """Mock predictor that returns terrain features."""
    predictor = MagicMock()
    predictor.get_terrain_features.return_value = {
        "elevation": 1257.0,
        "slope": 15.0,
        "aspect": 180.0,
        "roughness": 50.0,
        "tpi": 0.5,
    }
    return predictor


class TestTerrainCacheGet:
    """Tests for TerrainCache.get()."""

    def test_get_returns_none_when_not_cached(self, terrain_cache):
        """get() returns None when location not in cache."""
        result = terrain_cache.get(47.74, -121.09)
        assert result is None

    def test_get_returns_cached_terrain(self, terrain_cache, temp_db):
        """get() returns CachedTerrain when location is cached."""
        # Pre-populate cache via db directly
        temp_db.store_terrain(
            lat=47.74,
            lon=-121.09,
            elevation=1257.0,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        result = terrain_cache.get(47.74, -121.09)

        assert result is not None
        assert isinstance(result, CachedTerrain)
        assert result.elevation == 1257.0
        assert result.slope == 15.0
        assert result.aspect == 180.0
        assert result.roughness == 50.0
        assert result.tpi == 0.5

    def test_get_exact_coordinates_required(self, terrain_cache, temp_db):
        """get() requires exact lat/lon match."""
        temp_db.store_terrain(
            lat=47.74,
            lon=-121.09,
            elevation=1257.0,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        # Slightly different coordinates should not match
        result = terrain_cache.get(47.75, -121.09)
        assert result is None


class TestTerrainCacheStore:
    """Tests for TerrainCache.store()."""

    def test_store_persists_terrain(self, terrain_cache, temp_db):
        """store() persists terrain data to database."""
        terrain_cache.store(
            lat=47.74,
            lon=-121.09,
            elevation=1257.0,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        # Verify via direct db access
        result = temp_db.get_terrain(47.74, -121.09)
        assert result is not None
        assert result.elevation == 1257.0

    def test_store_updates_existing(self, terrain_cache, temp_db):
        """store() updates existing terrain data (upsert)."""
        terrain_cache.store(
            lat=47.74,
            lon=-121.09,
            elevation=1257.0,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        # Update with new values
        terrain_cache.store(
            lat=47.74,
            lon=-121.09,
            elevation=1300.0,
            slope=20.0,
            aspect=90.0,
            roughness=60.0,
            tpi=1.0,
        )

        result = temp_db.get_terrain(47.74, -121.09)
        assert result.elevation == 1300.0
        assert result.slope == 20.0


class TestTerrainCacheFetchAndCache:
    """Tests for TerrainCache.fetch_and_cache()."""

    def test_fetch_and_cache_returns_from_cache(self, terrain_cache, temp_db):
        """fetch_and_cache() returns cached data without calling predictor."""
        # Pre-populate cache
        temp_db.store_terrain(
            lat=47.74,
            lon=-121.09,
            elevation=1257.0,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        # Should not need predictor
        result = terrain_cache.fetch_and_cache(47.74, -121.09)

        assert result is not None
        assert result.elevation == 1257.0
        # Predictor should not have been initialized
        assert terrain_cache._predictor is None

    def test_fetch_and_cache_fetches_on_miss(self, terrain_cache, mock_predictor):
        """fetch_and_cache() fetches from DEM on cache miss."""
        # Inject mock predictor
        terrain_cache._predictor = mock_predictor

        result = terrain_cache.fetch_and_cache(47.74, -121.09)

        assert result is not None
        assert result.elevation == 1257.0
        mock_predictor.get_terrain_features.assert_called_once_with(47.74, -121.09)

    def test_fetch_and_cache_stores_result(self, terrain_cache, mock_predictor):
        """fetch_and_cache() stores fetched data in cache."""
        terrain_cache._predictor = mock_predictor

        terrain_cache.fetch_and_cache(47.74, -121.09)

        # Second call should hit cache
        mock_predictor.get_terrain_features.reset_mock()
        result = terrain_cache.fetch_and_cache(47.74, -121.09)

        assert result is not None
        mock_predictor.get_terrain_features.assert_not_called()

    def test_fetch_and_cache_logs_fetch(self, terrain_cache, mock_predictor, temp_db):
        """fetch_and_cache() logs the fetch operation."""
        terrain_cache._predictor = mock_predictor

        terrain_cache.fetch_and_cache(47.74, -121.09)

        # Check fetch log
        log = temp_db.conn.execute(
            "SELECT source, status, records_added FROM fetch_log WHERE source = 'dem'"
        ).fetchone()

        assert log is not None
        assert log[0] == "dem"
        assert log[1] == "success"
        assert log[2] == 1


class TestTerrainCachePrefetch:
    """Tests for TerrainCache.prefetch_all_ski_areas()."""

    def test_prefetch_all_ski_areas_count(self, terrain_cache, mock_predictor):
        """prefetch_all_ski_areas() returns count of cached areas."""
        terrain_cache._predictor = mock_predictor

        count = terrain_cache.prefetch_all_ski_areas()

        assert count == len(SKI_AREAS_DATA)
        assert count == 22  # Known number of ski areas

    def test_prefetch_skips_already_cached(self, terrain_cache, mock_predictor, temp_db):
        """prefetch_all_ski_areas() skips already cached locations."""
        # Pre-cache Stevens Pass
        stevens = SKI_AREAS_DATA[0]
        temp_db.store_terrain(
            lat=stevens.lat,
            lon=stevens.lon,
            elevation=1241.0,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        terrain_cache._predictor = mock_predictor

        count = terrain_cache.prefetch_all_ski_areas()

        # Should still report full count (cache hit counts too)
        assert count == 22
        # But predictor should only be called 21 times (skipping Stevens Pass)
        assert mock_predictor.get_terrain_features.call_count == 21

    def test_prefetch_continues_on_error(self, terrain_cache):
        """prefetch_all_ski_areas() continues if one fetch fails."""
        mock_predictor = MagicMock()
        call_count = [0]

        def side_effect(lat, lon):
            call_count[0] += 1
            if call_count[0] == 5:  # Fail on 5th call
                raise Exception("DEM fetch failed")
            return {
                "elevation": 2000.0,
                "slope": 15.0,
                "aspect": 180.0,
                "roughness": 50.0,
                "tpi": 0.5,
            }

        mock_predictor.get_terrain_features.side_effect = side_effect
        terrain_cache._predictor = mock_predictor

        count = terrain_cache.prefetch_all_ski_areas()

        # Should complete all 22 attempts, but one failed
        assert count == 21  # 22 - 1 failed
        assert mock_predictor.get_terrain_features.call_count == 22


class TestTerrainCacheHelpers:
    """Tests for helper methods."""

    def test_get_cached_count_empty(self, terrain_cache):
        """get_cached_count() returns 0 when empty."""
        assert terrain_cache.get_cached_count() == 0

    def test_get_cached_count_with_data(self, terrain_cache, temp_db):
        """get_cached_count() returns correct count."""
        for i in range(5):
            temp_db.store_terrain(
                lat=47.0 + i * 0.1,
                lon=-121.0,
                elevation=1000.0 + i * 100,
                slope=15.0,
                aspect=180.0,
                roughness=50.0,
                tpi=0.5,
            )

        assert terrain_cache.get_cached_count() == 5

    def test_is_ski_area_cached_false(self, terrain_cache):
        """is_ski_area_cached() returns False when not cached."""
        assert terrain_cache.is_ski_area_cached("Stevens Pass") is False

    def test_is_ski_area_cached_true(self, terrain_cache, temp_db):
        """is_ski_area_cached() returns True when cached."""
        stevens = SKI_AREAS_DATA[0]
        temp_db.store_terrain(
            lat=stevens.lat,
            lon=stevens.lon,
            elevation=1241.0,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        assert terrain_cache.is_ski_area_cached("Stevens Pass") is True

    def test_is_ski_area_cached_unknown_area(self, terrain_cache):
        """is_ski_area_cached() returns False for unknown area."""
        assert terrain_cache.is_ski_area_cached("Nonexistent Resort") is False


class TestTerrainCachePerformance:
    """Tests for performance characteristics."""

    def test_cache_hit_is_fast(self, terrain_cache, temp_db):
        """Cache hit should be much faster than typical DEM fetch (~2s)."""
        import time

        # Pre-populate cache
        temp_db.store_terrain(
            lat=47.74,
            lon=-121.09,
            elevation=1257.0,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        # Time 100 cache hits
        start = time.time()
        for _ in range(100):
            terrain_cache.get(47.74, -121.09)
        elapsed = time.time() - start

        # Average should be <10ms per lookup (vs 2s for DEM)
        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 10, f"Cache lookup took {avg_ms:.2f}ms on average"


class TestTerrainCachePermanence:
    """Tests for permanent caching (no expiration)."""

    def test_terrain_never_expires(self, terrain_cache, temp_db):
        """Terrain cache entries never expire."""
        # Store with old timestamp (simulated by direct db access)
        temp_db.store_terrain(
            lat=47.74,
            lon=-121.09,
            elevation=1257.0,
            slope=15.0,
            aspect=180.0,
            roughness=50.0,
            tpi=0.5,
        )

        # Manually set old fetch_time
        temp_db.conn.execute(
            """
            UPDATE terrain_cache
            SET fetch_time = '2020-01-01 00:00:00'
            WHERE lat = 47.74 AND lon = -121.09
            """
        )

        # Should still return cached data
        result = terrain_cache.get(47.74, -121.09)
        assert result is not None
        assert result.elevation == 1257.0
