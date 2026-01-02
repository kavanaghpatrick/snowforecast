"""DuckDB cache database for snowforecast."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import duckdb

from snowforecast.cache.models import (
    CachedForecast,
    CachedTerrain,
    SkiArea,
    SKI_AREAS_DATA,
)

logger = logging.getLogger(__name__)

# Default database path - use project root to ensure consistent path
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "cache" / "snowforecast.duckdb"

# SQL schema - DuckDB uses sequences for auto-increment
SCHEMA_SQL = """
-- Create sequences for auto-increment IDs
CREATE SEQUENCE IF NOT EXISTS seq_hrrr_id START 1;
CREATE SEQUENCE IF NOT EXISTS seq_terrain_id START 1;
CREATE SEQUENCE IF NOT EXISTS seq_ski_areas_id START 1;
CREATE SEQUENCE IF NOT EXISTS seq_fetch_log_id START 1;

-- HRRR forecast cache
CREATE TABLE IF NOT EXISTS hrrr_forecasts (
    id INTEGER DEFAULT nextval('seq_hrrr_id') PRIMARY KEY,
    fetch_time TIMESTAMP NOT NULL,
    run_time TIMESTAMP NOT NULL,
    forecast_hour INTEGER NOT NULL,
    valid_time TIMESTAMP NOT NULL,
    lat DOUBLE NOT NULL,
    lon DOUBLE NOT NULL,
    snow_depth_m DOUBLE,
    temp_k DOUBLE,
    precip_mm DOUBLE,
    categorical_snow DOUBLE,
    UNIQUE(run_time, forecast_hour, lat, lon)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_hrrr_valid ON hrrr_forecasts(valid_time, lat, lon);
CREATE INDEX IF NOT EXISTS idx_hrrr_run ON hrrr_forecasts(run_time);
CREATE INDEX IF NOT EXISTS idx_hrrr_location ON hrrr_forecasts(lat, lon);

-- Terrain cache (permanent - terrain doesn't change)
CREATE TABLE IF NOT EXISTS terrain_cache (
    id INTEGER DEFAULT nextval('seq_terrain_id') PRIMARY KEY,
    lat DOUBLE NOT NULL,
    lon DOUBLE NOT NULL,
    elevation DOUBLE,
    slope DOUBLE,
    aspect DOUBLE,
    roughness DOUBLE,
    tpi DOUBLE,
    fetch_time TIMESTAMP NOT NULL,
    UNIQUE(lat, lon)
);

CREATE INDEX IF NOT EXISTS idx_terrain_location ON terrain_cache(lat, lon);

-- Ski areas (reference data)
CREATE TABLE IF NOT EXISTS ski_areas (
    id INTEGER DEFAULT nextval('seq_ski_areas_id') PRIMARY KEY,
    name VARCHAR NOT NULL UNIQUE,
    lat DOUBLE NOT NULL,
    lon DOUBLE NOT NULL,
    state VARCHAR NOT NULL,
    base_elevation DOUBLE
);

-- Fetch log for debugging/monitoring
CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER DEFAULT nextval('seq_fetch_log_id') PRIMARY KEY,
    source VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    status VARCHAR NOT NULL,
    records_added INTEGER,
    duration_ms INTEGER,
    error_message VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_fetch_log_time ON fetch_log(timestamp);
"""


class CacheDatabase:
    """DuckDB cache database manager.

    Provides persistent caching for HRRR forecasts and terrain data.
    Uses DuckDB for fast analytical queries on time-series data.

    Example:
        >>> db = CacheDatabase()
        >>> db.get_forecast(47.74, -121.09, datetime.now())
        CachedForecast(...)
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database connection.

        Args:
            db_path: Path to DuckDB file. Creates if doesn't exist.
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = None
        self._init_schema()

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get database connection (lazy initialization with retry)."""
        if self._conn is None:
            self._conn = self._connect_with_retry()
        return self._conn

    def _connect_with_retry(self, max_retries: int = 3) -> duckdb.DuckDBPyConnection:
        """Connect to database with retry logic for lock handling."""
        import time
        last_error = None
        for attempt in range(max_retries):
            try:
                return duckdb.connect(str(self.db_path))
            except duckdb.IOException as e:
                last_error = e
                if "lock" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = 0.5 * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Database locked, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
        raise last_error

    def _init_schema(self) -> None:
        """Initialize database schema."""
        # DuckDB executes multiple statements with execute()
        for statement in SCHEMA_SQL.split(";"):
            statement = statement.strip()
            if statement:
                self.conn.execute(statement)
        self._init_ski_areas()
        logger.info(f"Cache database initialized at {self.db_path}")

    def _init_ski_areas(self) -> None:
        """Populate ski_areas table with reference data."""
        existing = self.conn.execute("SELECT COUNT(*) FROM ski_areas").fetchone()[0]
        if existing > 0:
            return

        for area in SKI_AREAS_DATA:
            self.conn.execute(
                """
                INSERT INTO ski_areas (name, lat, lon, state, base_elevation)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (name) DO NOTHING
                """,
                [area.name, area.lat, area.lon, area.state, area.base_elevation],
            )
        logger.info(f"Initialized {len(SKI_AREAS_DATA)} ski areas")

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -------------------------------------------------------------------------
    # Forecast Cache Operations
    # -------------------------------------------------------------------------

    def get_forecast(
        self,
        lat: float,
        lon: float,
        valid_time: datetime,
        max_age_hours: int = 2,
    ) -> Optional[CachedForecast]:
        """Get cached forecast for location and time.

        Args:
            lat: Latitude
            lon: Longitude
            valid_time: Target forecast valid time
            max_age_hours: Maximum age of cached data in hours

        Returns:
            CachedForecast if found and fresh, None otherwise
        """
        min_run_time = datetime.utcnow() - timedelta(hours=max_age_hours)

        result = self.conn.execute(
            """
            SELECT
                lat, lon, run_time, forecast_hour, valid_time,
                snow_depth_m, temp_k, precip_mm, categorical_snow, fetch_time
            FROM hrrr_forecasts
            WHERE lat = ? AND lon = ?
              AND valid_time BETWEEN ? AND ?
              AND run_time >= ?
            ORDER BY run_time DESC, ABS(EXTRACT(EPOCH FROM (valid_time - ?)))
            LIMIT 1
            """,
            [
                lat,
                lon,
                valid_time - timedelta(hours=1),
                valid_time + timedelta(hours=1),
                min_run_time,
                valid_time,
            ],
        ).fetchone()

        if result is None:
            return None

        return CachedForecast(
            lat=result[0],
            lon=result[1],
            run_time=result[2],
            forecast_hour=result[3],
            valid_time=result[4],
            snow_depth_m=result[5] or 0.0,
            temp_k=result[6] or 273.0,
            precip_mm=result[7] or 0.0,
            categorical_snow=result[8] or 0.0,
            fetch_time=result[9],
        )

    def store_forecast(
        self,
        lat: float,
        lon: float,
        run_time: datetime,
        forecast_hour: int,
        snow_depth_m: float,
        temp_k: float,
        precip_mm: float,
        categorical_snow: float,
    ) -> None:
        """Store forecast in cache.

        Args:
            lat: Latitude
            lon: Longitude
            run_time: HRRR model run time
            forecast_hour: Forecast hour (0-48)
            snow_depth_m: Snow depth in meters
            temp_k: Temperature in Kelvin
            precip_mm: Precipitation in mm
            categorical_snow: Categorical snow flag (0 or 1)
        """
        valid_time = run_time + timedelta(hours=forecast_hour)
        fetch_time = datetime.utcnow()

        self.conn.execute(
            """
            INSERT INTO hrrr_forecasts
            (fetch_time, run_time, forecast_hour, valid_time, lat, lon,
             snow_depth_m, temp_k, precip_mm, categorical_snow)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_time, forecast_hour, lat, lon)
            DO UPDATE SET
                fetch_time = EXCLUDED.fetch_time,
                snow_depth_m = EXCLUDED.snow_depth_m,
                temp_k = EXCLUDED.temp_k,
                precip_mm = EXCLUDED.precip_mm,
                categorical_snow = EXCLUDED.categorical_snow
            """,
            [
                fetch_time,
                run_time,
                forecast_hour,
                valid_time,
                lat,
                lon,
                snow_depth_m,
                temp_k,
                precip_mm,
                categorical_snow,
            ],
        )

    def get_latest_run_time(self) -> Optional[datetime]:
        """Get the most recent HRRR run time in cache."""
        result = self.conn.execute(
            "SELECT MAX(run_time) FROM hrrr_forecasts"
        ).fetchone()
        return result[0] if result and result[0] else None

    def cleanup_old_forecasts(self, keep_days: int = 7) -> int:
        """Remove forecasts older than keep_days.

        Returns:
            Number of rows deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=keep_days)
        result = self.conn.execute(
            "DELETE FROM hrrr_forecasts WHERE run_time < ?",
            [cutoff],
        )
        deleted = result.fetchone()[0] if result else 0
        logger.info(f"Cleaned up {deleted} old forecast records")
        return deleted

    # -------------------------------------------------------------------------
    # Terrain Cache Operations
    # -------------------------------------------------------------------------

    def get_terrain(self, lat: float, lon: float) -> Optional[CachedTerrain]:
        """Get cached terrain data for location.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            CachedTerrain if found, None otherwise
        """
        result = self.conn.execute(
            """
            SELECT lat, lon, elevation, slope, aspect, roughness, tpi, fetch_time
            FROM terrain_cache
            WHERE lat = ? AND lon = ?
            """,
            [lat, lon],
        ).fetchone()

        if result is None:
            return None

        return CachedTerrain(
            lat=result[0],
            lon=result[1],
            elevation=result[2] or 0.0,
            slope=result[3] or 0.0,
            aspect=result[4] or 0.0,
            roughness=result[5] or 0.0,
            tpi=result[6] or 0.0,
            fetch_time=result[7],
        )

    def store_terrain(
        self,
        lat: float,
        lon: float,
        elevation: float,
        slope: float,
        aspect: float,
        roughness: float,
        tpi: float,
    ) -> None:
        """Store terrain data in cache (permanent).

        Args:
            lat: Latitude
            lon: Longitude
            elevation: Elevation in meters
            slope: Slope in degrees
            aspect: Aspect in degrees (0-360)
            roughness: Terrain roughness
            tpi: Topographic Position Index
        """
        fetch_time = datetime.utcnow()

        self.conn.execute(
            """
            INSERT INTO terrain_cache
            (lat, lon, elevation, slope, aspect, roughness, tpi, fetch_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (lat, lon)
            DO UPDATE SET
                elevation = EXCLUDED.elevation,
                slope = EXCLUDED.slope,
                aspect = EXCLUDED.aspect,
                roughness = EXCLUDED.roughness,
                tpi = EXCLUDED.tpi,
                fetch_time = EXCLUDED.fetch_time
            """,
            [lat, lon, elevation, slope, aspect, roughness, tpi, fetch_time],
        )

    # -------------------------------------------------------------------------
    # Ski Area Operations
    # -------------------------------------------------------------------------

    def get_ski_areas(self) -> list[SkiArea]:
        """Get all ski areas."""
        results = self.conn.execute(
            "SELECT name, lat, lon, state, base_elevation FROM ski_areas"
        ).fetchall()

        return [
            SkiArea(
                name=row[0],
                lat=row[1],
                lon=row[2],
                state=row[3],
                base_elevation=row[4],
            )
            for row in results
        ]

    def get_ski_area(self, name: str) -> Optional[SkiArea]:
        """Get ski area by name."""
        result = self.conn.execute(
            "SELECT name, lat, lon, state, base_elevation FROM ski_areas WHERE name = ?",
            [name],
        ).fetchone()

        if result is None:
            return None

        return SkiArea(
            name=result[0],
            lat=result[1],
            lon=result[2],
            state=result[3],
            base_elevation=result[4],
        )

    # -------------------------------------------------------------------------
    # Logging Operations
    # -------------------------------------------------------------------------

    def log_fetch(
        self,
        source: str,
        status: str,
        records_added: int,
        duration_ms: int,
        error_message: Optional[str] = None,
    ) -> None:
        """Log a data fetch operation."""
        self.conn.execute(
            """
            INSERT INTO fetch_log (source, timestamp, status, records_added, duration_ms, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [source, datetime.utcnow(), status, records_added, duration_ms, error_message],
        )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get cache statistics."""
        forecast_count = self.conn.execute(
            "SELECT COUNT(*) FROM hrrr_forecasts"
        ).fetchone()[0]

        terrain_count = self.conn.execute(
            "SELECT COUNT(*) FROM terrain_cache"
        ).fetchone()[0]

        latest_run = self.get_latest_run_time()

        return {
            "forecast_count": forecast_count,
            "terrain_count": terrain_count,
            "latest_run_time": latest_run,
            "db_path": str(self.db_path),
        }
