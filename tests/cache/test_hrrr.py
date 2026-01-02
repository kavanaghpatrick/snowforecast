"""Tests for HRRR caching layer."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from snowforecast.cache.database import CacheDatabase
from snowforecast.cache.hrrr import HRRRCache, CACHE_VALIDITY_HOURS
from snowforecast.cache.models import CachedForecast


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.duckdb"
        db = CacheDatabase(db_path)
        yield db
        db.close()


@pytest.fixture
def hrrr_cache(temp_db):
    """Create HRRRCache with temp database."""
    return HRRRCache(temp_db)


class TestHRRRCacheGet:
    """Tests for HRRRCache.get() method."""

    def test_get_returns_none_when_empty(self, hrrr_cache):
        """Returns None when cache is empty."""
        result = hrrr_cache.get(47.74, -121.09, datetime.utcnow())
        assert result is None

    def test_get_returns_cached_forecast(self, hrrr_cache, temp_db):
        """Returns cached forecast when available."""
        run_time = datetime.utcnow()
        valid_time = run_time + timedelta(hours=24)

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

        result = hrrr_cache.get(47.74, -121.09, valid_time)

        assert result is not None
        assert result.snow_depth_m == 1.5
        assert result.temp_k == 270.0
        assert result.forecast_hour == 24

    def test_get_respects_max_age(self, hrrr_cache, temp_db):
        """Expired forecasts are not returned."""
        old_run_time = datetime.utcnow() - timedelta(hours=10)
        valid_time = old_run_time + timedelta(hours=24)

        temp_db.store_forecast(
            lat=47.74,
            lon=-121.09,
            run_time=old_run_time,
            forecast_hour=24,
            snow_depth_m=1.5,
            temp_k=270.0,
            precip_mm=5.0,
            categorical_snow=1.0,
        )

        # Default max_age is 2 hours, so old forecast should not be returned
        result = hrrr_cache.get(47.74, -121.09, valid_time)
        assert result is None

    def test_get_with_custom_max_age(self, hrrr_cache, temp_db):
        """Custom max_age allows older forecasts."""
        old_run_time = datetime.utcnow() - timedelta(hours=10)
        valid_time = old_run_time + timedelta(hours=24)

        temp_db.store_forecast(
            lat=47.74,
            lon=-121.09,
            run_time=old_run_time,
            forecast_hour=24,
            snow_depth_m=1.5,
            temp_k=270.0,
            precip_mm=5.0,
            categorical_snow=1.0,
        )

        # With max_age_hours=24, old forecast should be returned
        result = hrrr_cache.get(47.74, -121.09, valid_time, max_age_hours=24)
        assert result is not None
        assert result.snow_depth_m == 1.5


class TestHRRRCacheStore:
    """Tests for HRRRCache.store() method."""

    def test_store_returns_cached_forecast(self, hrrr_cache):
        """Store returns a CachedForecast object."""
        run_time = datetime.utcnow()

        result = hrrr_cache.store(
            lat=47.74,
            lon=-121.09,
            run_time=run_time,
            forecast_hour=12,
            snow_depth_m=2.0,
            temp_k=268.0,
            precip_mm=10.0,
            categorical_snow=1.0,
        )

        assert isinstance(result, CachedForecast)
        assert result.lat == 47.74
        assert result.lon == -121.09
        assert result.snow_depth_m == 2.0
        assert result.forecast_hour == 12

    def test_store_persists_data(self, hrrr_cache, temp_db):
        """Stored data can be retrieved."""
        run_time = datetime.utcnow()
        valid_time = run_time + timedelta(hours=12)

        hrrr_cache.store(
            lat=47.74,
            lon=-121.09,
            run_time=run_time,
            forecast_hour=12,
            snow_depth_m=2.0,
            temp_k=268.0,
            precip_mm=10.0,
            categorical_snow=1.0,
        )

        result = temp_db.get_forecast(47.74, -121.09, valid_time, max_age_hours=24)
        assert result is not None
        assert result.snow_depth_m == 2.0

    def test_store_calculates_valid_time(self, hrrr_cache):
        """Valid time is calculated from run_time + forecast_hour."""
        run_time = datetime(2026, 1, 1, 0, 0, 0)

        result = hrrr_cache.store(
            lat=47.74,
            lon=-121.09,
            run_time=run_time,
            forecast_hour=24,
            snow_depth_m=1.0,
            temp_k=273.0,
            precip_mm=0.0,
            categorical_snow=0.0,
        )

        expected_valid_time = datetime(2026, 1, 2, 0, 0, 0)
        assert result.valid_time == expected_valid_time


class TestHRRRCacheFetchAndCache:
    """Tests for HRRRCache.fetch_and_cache() method."""

    def test_returns_cached_if_available(self, hrrr_cache, temp_db):
        """Returns cached data without fetching if fresh."""
        run_time = datetime.utcnow()
        valid_time = run_time + timedelta(hours=12)

        # Pre-populate cache
        temp_db.store_forecast(
            lat=47.74,
            lon=-121.09,
            run_time=run_time,
            forecast_hour=12,
            snow_depth_m=1.5,
            temp_k=270.0,
            precip_mm=5.0,
            categorical_snow=1.0,
        )

        # Mock predictor to ensure it's not called
        with patch.object(hrrr_cache, '_get_predictor') as mock_pred:
            result = hrrr_cache.fetch_and_cache(47.74, -121.09, valid_time)

            # Should return cached data without calling predictor
            mock_pred.assert_not_called()
            assert result is not None
            assert result.snow_depth_m == 1.5

    def test_fetches_when_cache_miss(self, hrrr_cache, temp_db):
        """Fetches from HRRR when cache is empty."""
        valid_time = datetime.utcnow() + timedelta(hours=24)

        mock_hrrr_data = {
            "snow_depth_m": 0.5,
            "temp_k": 268.0,
            "precip_mm": 8.0,
            "categorical_snow": 1.0,
        }

        mock_predictor = MagicMock()
        mock_predictor.fetch_hrrr_forecast.return_value = mock_hrrr_data

        with patch.object(hrrr_cache, '_get_predictor', return_value=mock_predictor):
            result = hrrr_cache.fetch_and_cache(47.74, -121.09, valid_time)

            mock_predictor.fetch_hrrr_forecast.assert_called_once()
            assert result is not None
            assert result.snow_depth_m == 0.5
            assert result.temp_k == 268.0

    def test_caches_fetched_data(self, hrrr_cache, temp_db):
        """Fetched data is stored in cache."""
        valid_time = datetime.utcnow() + timedelta(hours=24)

        mock_hrrr_data = {
            "snow_depth_m": 0.75,
            "temp_k": 265.0,
            "precip_mm": 12.0,
            "categorical_snow": 1.0,
        }

        mock_predictor = MagicMock()
        mock_predictor.fetch_hrrr_forecast.return_value = mock_hrrr_data

        with patch.object(hrrr_cache, '_get_predictor', return_value=mock_predictor):
            hrrr_cache.fetch_and_cache(47.74, -121.09, valid_time)

        # Verify data is now cached
        stats = temp_db.get_stats()
        assert stats["forecast_count"] == 1

    def test_returns_none_on_fetch_failure(self, hrrr_cache, temp_db):
        """Returns None when HRRR fetch fails."""
        valid_time = datetime.utcnow() + timedelta(hours=24)

        mock_predictor = MagicMock()
        mock_predictor.fetch_hrrr_forecast.return_value = None

        with patch.object(hrrr_cache, '_get_predictor', return_value=mock_predictor):
            result = hrrr_cache.fetch_and_cache(47.74, -121.09, valid_time)

            assert result is None

    def test_logs_fetch_success(self, hrrr_cache, temp_db):
        """Successful fetch is logged."""
        valid_time = datetime.utcnow() + timedelta(hours=24)

        mock_hrrr_data = {
            "snow_depth_m": 0.5,
            "temp_k": 268.0,
            "precip_mm": 8.0,
            "categorical_snow": 1.0,
        }

        mock_predictor = MagicMock()
        mock_predictor.fetch_hrrr_forecast.return_value = mock_hrrr_data

        with patch.object(hrrr_cache, '_get_predictor', return_value=mock_predictor):
            hrrr_cache.fetch_and_cache(47.74, -121.09, valid_time)

        # Check fetch log
        result = temp_db.conn.execute(
            "SELECT source, status, records_added FROM fetch_log"
        ).fetchone()

        assert result[0] == "hrrr"
        assert result[1] == "success"
        assert result[2] == 1

    def test_logs_fetch_failure(self, hrrr_cache, temp_db):
        """Failed fetch is logged with error message."""
        valid_time = datetime.utcnow() + timedelta(hours=24)

        mock_predictor = MagicMock()
        mock_predictor.fetch_hrrr_forecast.side_effect = Exception("Network timeout")

        with patch.object(hrrr_cache, '_get_predictor', return_value=mock_predictor):
            result = hrrr_cache.fetch_and_cache(47.74, -121.09, valid_time)

        assert result is None

        # Check fetch log
        log_entry = temp_db.conn.execute(
            "SELECT status, error_message FROM fetch_log WHERE source = 'hrrr'"
        ).fetchone()

        assert log_entry[0] == "error"
        assert "Network timeout" in log_entry[1]

    def test_logs_null_result_as_error(self, hrrr_cache, temp_db):
        """When HRRR returns None, it's logged as an error."""
        valid_time = datetime.utcnow() + timedelta(hours=24)

        mock_predictor = MagicMock()
        mock_predictor.fetch_hrrr_forecast.return_value = None

        with patch.object(hrrr_cache, '_get_predictor', return_value=mock_predictor):
            hrrr_cache.fetch_and_cache(47.74, -121.09, valid_time)

        log_entry = temp_db.conn.execute(
            "SELECT status, error_message FROM fetch_log WHERE source = 'hrrr'"
        ).fetchone()

        assert log_entry[0] == "error"
        assert "None" in log_entry[1]


class TestHRRRCacheGetOrFetch:
    """Tests for convenience method get_or_fetch()."""

    def test_get_or_fetch_is_alias(self, hrrr_cache, temp_db):
        """get_or_fetch is an alias for fetch_and_cache."""
        run_time = datetime.utcnow()
        valid_time = run_time + timedelta(hours=12)

        # Pre-populate cache
        temp_db.store_forecast(
            lat=47.74,
            lon=-121.09,
            run_time=run_time,
            forecast_hour=12,
            snow_depth_m=1.5,
            temp_k=270.0,
            precip_mm=5.0,
            categorical_snow=1.0,
        )

        result = hrrr_cache.get_or_fetch(47.74, -121.09, valid_time)

        assert result is not None
        assert result.snow_depth_m == 1.5


class TestHRRRCacheCleanup:
    """Tests for cache cleanup functionality."""

    def test_cleanup_old_data(self, hrrr_cache, temp_db):
        """Old data is removed during cleanup."""
        # Store old forecast
        old_run_time = datetime.utcnow() - timedelta(days=10)
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

        # Store recent forecast
        recent_run_time = datetime.utcnow()
        temp_db.store_forecast(
            lat=48.00,
            lon=-122.00,
            run_time=recent_run_time,
            forecast_hour=0,
            snow_depth_m=2.0,
            temp_k=270.0,
            precip_mm=5.0,
            categorical_snow=1.0,
        )

        # Clean up with 7-day retention
        deleted = hrrr_cache.cleanup_old_data(keep_days=7)

        # Old forecast should be deleted
        stats = temp_db.get_stats()
        assert stats["forecast_count"] == 1

    def test_cleanup_with_custom_retention(self, hrrr_cache, temp_db):
        """Custom retention period is respected."""
        # Store forecast from 3 days ago
        three_days_ago = datetime.utcnow() - timedelta(days=3)
        temp_db.store_forecast(
            lat=47.74,
            lon=-121.09,
            run_time=three_days_ago,
            forecast_hour=0,
            snow_depth_m=1.0,
            temp_k=273.0,
            precip_mm=0.0,
            categorical_snow=0.0,
        )

        # Clean up with 2-day retention - should delete the 3-day-old data
        hrrr_cache.cleanup_old_data(keep_days=2)
        stats = temp_db.get_stats()
        assert stats["forecast_count"] == 0


class TestHRRRCacheStats:
    """Tests for cache statistics."""

    def test_get_stats_empty(self, hrrr_cache):
        """Stats work on empty cache."""
        stats = hrrr_cache.get_stats()

        assert stats["forecast_count"] == 0
        assert stats["latest_run_time"] is None

    def test_get_stats_with_data(self, hrrr_cache, temp_db):
        """Stats reflect stored data."""
        run_time = datetime.utcnow()
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

        stats = hrrr_cache.get_stats()

        assert stats["forecast_count"] == 1
        assert stats["latest_run_time"] is not None


class TestCacheValidityHours:
    """Tests for cache validity configuration."""

    def test_default_cache_validity(self):
        """Default cache validity is 2 hours."""
        assert CACHE_VALIDITY_HOURS == 2

    def test_cache_respects_validity_window(self, hrrr_cache, temp_db):
        """Cache returns data within validity window."""
        # Store forecast from 1 hour ago (within 2-hour window)
        recent_run_time = datetime.utcnow() - timedelta(hours=1)
        valid_time = recent_run_time + timedelta(hours=12)

        temp_db.store_forecast(
            lat=47.74,
            lon=-121.09,
            run_time=recent_run_time,
            forecast_hour=12,
            snow_depth_m=1.5,
            temp_k=270.0,
            precip_mm=5.0,
            categorical_snow=1.0,
        )

        result = hrrr_cache.get(47.74, -121.09, valid_time)
        assert result is not None

    def test_cache_expires_outside_validity_window(self, hrrr_cache, temp_db):
        """Cache returns None for data outside validity window."""
        # Store forecast from 3 hours ago (outside 2-hour window)
        old_run_time = datetime.utcnow() - timedelta(hours=3)
        valid_time = old_run_time + timedelta(hours=12)

        temp_db.store_forecast(
            lat=47.74,
            lon=-121.09,
            run_time=old_run_time,
            forecast_hour=12,
            snow_depth_m=1.5,
            temp_k=270.0,
            precip_mm=5.0,
            categorical_snow=1.0,
        )

        result = hrrr_cache.get(47.74, -121.09, valid_time)
        assert result is None
