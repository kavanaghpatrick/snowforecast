# PRD: Data Caching Layer

## Problem Statement

Currently, every dashboard load fetches fresh data from NOAA HRRR (~6MB per resort).
- 22 resorts × 6MB = ~130MB downloaded per full page load
- Same HRRR run (updated hourly) fetched repeatedly by different users
- DEM terrain data (static) re-fetched on every request
- No historical data storage for model training/validation
- Slow user experience (2+ minutes for full load)

## Goals

1. **Reduce latency**: Dashboard loads in <2 seconds for cached data
2. **Reduce bandwidth**: Fetch each HRRR run only once
3. **Enable offline**: Serve cached data when NOAA is unavailable
4. **Store history**: Keep historical forecasts for model training
5. **Simple deployment**: No external database servers required

## Non-Goals

- Multi-region deployment (single machine for now)
- Real-time streaming updates
- User-specific caching

## Technical Design

### Database: DuckDB

**Why DuckDB:**
- Single-file database (no server)
- Excellent for analytical/time-series queries
- 10-100x faster than SQLite for aggregations
- Native Python/Pandas integration
- Can query Parquet files directly
- Easy backup (copy one file)

**Location:** `data/cache/snowforecast.duckdb`

### Schema

```sql
-- HRRR forecast cache
CREATE TABLE hrrr_forecasts (
    id INTEGER PRIMARY KEY,
    fetch_time TIMESTAMP NOT NULL,      -- When we fetched this
    run_time TIMESTAMP NOT NULL,        -- HRRR model run time (00Z, 06Z, etc)
    forecast_hour INTEGER NOT NULL,     -- fxx (0-48)
    valid_time TIMESTAMP NOT NULL,      -- run_time + forecast_hour
    lat DOUBLE NOT NULL,
    lon DOUBLE NOT NULL,
    snow_depth_m DOUBLE,
    temp_k DOUBLE,
    precip_mm DOUBLE,
    categorical_snow DOUBLE,
    UNIQUE(run_time, forecast_hour, lat, lon)
);

-- Index for fast lookups
CREATE INDEX idx_hrrr_valid ON hrrr_forecasts(valid_time, lat, lon);
CREATE INDEX idx_hrrr_run ON hrrr_forecasts(run_time);

-- Terrain cache (permanent - terrain doesn't change)
CREATE TABLE terrain_cache (
    id INTEGER PRIMARY KEY,
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

-- Ski areas (reference data)
CREATE TABLE ski_areas (
    id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL UNIQUE,
    lat DOUBLE NOT NULL,
    lon DOUBLE NOT NULL,
    state VARCHAR NOT NULL,
    base_elevation DOUBLE
);

-- Fetch log for debugging/monitoring
CREATE TABLE fetch_log (
    id INTEGER PRIMARY KEY,
    source VARCHAR NOT NULL,  -- 'hrrr', 'dem'
    timestamp TIMESTAMP NOT NULL,
    status VARCHAR NOT NULL,  -- 'success', 'error'
    records_added INTEGER,
    duration_ms INTEGER,
    error_message VARCHAR
);
```

### Caching Strategy

#### HRRR Forecasts

```
On data request:
1. Check DuckDB for valid cached data
   - Look for run_time within last 2 hours
   - Match lat/lon within 0.01 degrees
2. If cache hit: return cached data (~5ms)
3. If cache miss:
   - Fetch from NOAA HRRR via Herbie
   - Store in DuckDB
   - Return data (~5-10 seconds)
```

**Cache Invalidation:**
- HRRR runs every hour (00Z, 01Z, 02Z, ...)
- New run typically available ~45 mins after run time
- Cache expires when newer run is available
- Keep old runs for 7 days (historical analysis)

#### Terrain Data

```
On terrain request:
1. Check terrain_cache for lat/lon
2. If exists: return cached (~1ms)
3. If not: fetch from DEM, store permanently
```

Terrain is static - cache forever.

#### Background Refresh

Optional scheduled job (cron):
```bash
# Every hour at :50 (before new HRRR run available)
50 * * * * python -m snowforecast.cache.refresh
```

Pre-warms cache for all 22 ski areas before users request.

### API Changes

```python
# New caching layer
class ForecastCache:
    def __init__(self, db_path: Path = "data/cache/snowforecast.duckdb"):
        self.db = duckdb.connect(str(db_path))
        self._init_schema()

    def get_forecast(self, lat: float, lon: float,
                     valid_time: datetime) -> Optional[dict]:
        """Get cached forecast or None if not available."""

    def store_forecast(self, lat: float, lon: float,
                       run_time: datetime, forecast_hour: int,
                       data: dict) -> None:
        """Store forecast in cache."""

    def get_terrain(self, lat: float, lon: float) -> Optional[dict]:
        """Get cached terrain or None."""

    def store_terrain(self, lat: float, lon: float,
                      terrain: dict) -> None:
        """Store terrain permanently."""

# Updated predictor
class CachedPredictor:
    def __init__(self):
        self.cache = ForecastCache()
        self.hrrr = RealPredictor()  # Fallback to NOAA

    def predict(self, lat, lon, target_date, forecast_hours=24):
        # Try cache first
        cached = self.cache.get_forecast(lat, lon, target_date)
        if cached:
            return cached  # ~5ms

        # Fall back to NOAA
        result = self.hrrr.predict(lat, lon, target_date, forecast_hours)

        # Store in cache for next time
        self.cache.store_forecast(lat, lon, ...)

        return result
```

### Dashboard Integration

```python
# In dashboard/app.py
@st.cache_resource
def get_predictor():
    from snowforecast.cache import CachedPredictor
    return CachedPredictor()
```

No other dashboard changes needed - caching is transparent.

## Implementation Plan

### Issue #29: Cache Infrastructure (Foundation)
**Branch:** `phase5/29-cache-infrastructure`
**Parallel:** No (must complete first)

Tasks:
- [ ] Add DuckDB dependency to pyproject.toml
- [ ] Create `src/snowforecast/cache/__init__.py`
- [ ] Create `src/snowforecast/cache/database.py` with schema
- [ ] Create `src/snowforecast/cache/models.py` with dataclasses
- [ ] Write tests for database operations
- [ ] Initialize ski_areas table with 22 resorts

### Issue #30: HRRR Caching
**Branch:** `phase5/30-hrrr-cache`
**Parallel:** After #29

Tasks:
- [ ] Create `src/snowforecast/cache/hrrr.py`
- [ ] Implement `HRRRCache.get()` - check for valid cached data
- [ ] Implement `HRRRCache.store()` - save forecast to DB
- [ ] Implement cache invalidation logic
- [ ] Add freshness checking (is newer HRRR run available?)
- [ ] Write tests with mock data

### Issue #31: Terrain Caching
**Branch:** `phase5/31-terrain-cache`
**Parallel:** After #29, can run parallel with #30

Tasks:
- [ ] Create `src/snowforecast/cache/terrain.py`
- [ ] Implement `TerrainCache.get()` - permanent cache
- [ ] Implement `TerrainCache.store()` - one-time store
- [ ] Pre-populate cache for all 22 ski areas
- [ ] Write tests

### Issue #32: Cached Predictor
**Branch:** `phase5/32-cached-predictor`
**Parallel:** After #30 and #31

Tasks:
- [ ] Create `src/snowforecast/cache/predictor.py`
- [ ] Implement `CachedPredictor` class
- [ ] Cache-first logic with NOAA fallback
- [ ] Integrate with existing RealPredictor
- [ ] Write integration tests

### Issue #33: Dashboard Integration
**Branch:** `phase5/33-dashboard-cache`
**Parallel:** After #32

Tasks:
- [ ] Update dashboard to use CachedPredictor
- [ ] Add cache status indicator (fresh/stale)
- [ ] Add "last updated" timestamp
- [ ] Test full flow

### Issue #34: Background Refresh (Optional)
**Branch:** `phase5/34-background-refresh`
**Parallel:** After #32

Tasks:
- [ ] Create `src/snowforecast/cache/refresh.py`
- [ ] Implement hourly refresh logic
- [ ] Add CLI command: `python -m snowforecast.cache.refresh`
- [ ] Document cron setup

## Dependency Graph

```
#29 Cache Infrastructure
    ├── #30 HRRR Caching
    │       └── #32 Cached Predictor ──→ #33 Dashboard
    └── #31 Terrain Caching ─────────┘        │
                                              └── #34 Background Refresh
```

**Parallelization:**
- #30 and #31 can run in parallel (both depend only on #29)
- #32 waits for both #30 and #31
- #33 and #34 can run in parallel (both depend on #32)

## Success Metrics

| Metric | Before | After |
|--------|--------|-------|
| First load (cold cache) | 2+ min | 30 sec (stores in cache) |
| Subsequent loads | 2+ min | <2 sec |
| Data freshness | Always live | Max 1 hour old |
| NOAA bandwidth | 130MB/load | 130MB/hour total |
| Historical data | None | 7 days retained |

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| DuckDB file corruption | Regular backups, WAL mode |
| Cache grows too large | Auto-cleanup of data >7 days |
| NOAA unavailable | Serve stale cache with warning |
| Schema changes | Migration scripts |

## Timeline

| Phase | Issues | Can Parallelize |
|-------|--------|-----------------|
| 1 | #29 | No (foundation) |
| 2 | #30, #31 | Yes (2 agents) |
| 3 | #32 | No (integration) |
| 4 | #33, #34 | Yes (2 agents) |

Total: 6 issues, 4 phases, max 2 parallel agents
