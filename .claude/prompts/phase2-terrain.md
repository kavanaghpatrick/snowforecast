# Phase 2 Agent Prompt: Terrain Feature Engineering

## Issue #13: Engineer terrain feature set

**Branch**: phase2/13-terrain
**Worktree**: ~/snowforecast-worktrees/phase2-terrain

## Objective

Create terrain-based features from DEM data that influence snow accumulation.

## Context

The DEM pipeline (`src/snowforecast/pipelines/dem.py`) already provides:
- TerrainFeatures dataclass: elevation, slope, aspect, aspect_sin, aspect_cos, roughness, tpi

This agent extends those with additional derived features.

## Tasks

1. Create `src/snowforecast/features/terrain.py`:
   ```python
   class TerrainFeatures:
       """Engineer terrain features for snow prediction."""

       def __init__(self, dem_pipeline=None):
           """Initialize with optional DEM pipeline instance."""

       def compute_all(self, locations: pd.DataFrame) -> pd.DataFrame:
           """Compute all terrain features for locations.

           Args:
               locations: DataFrame with lat, lon columns

           Returns:
               DataFrame with terrain features added
           """

       def get_base_terrain(self, lat: float, lon: float) -> dict:
           """Get base terrain from DEM pipeline."""

       def compute_elevation_features(self, df: pd.DataFrame) -> pd.DataFrame:
           """
           Features:
           - elevation_m: Elevation in meters
           - elevation_km: Elevation in kilometers
           - elevation_band: Categorical (low/mid/high/alpine)
           """

       def compute_slope_features(self, df: pd.DataFrame) -> pd.DataFrame:
           """
           Features:
           - slope_deg: Slope in degrees
           - slope_category: flat/gentle/moderate/steep/extreme
           """

       def compute_aspect_features(self, df: pd.DataFrame) -> pd.DataFrame:
           """
           Features:
           - aspect_deg: Aspect in degrees (0=N, 90=E, 180=S, 270=W)
           - aspect_sin: sin(aspect) for cyclical encoding
           - aspect_cos: cos(aspect) for cyclical encoding
           - aspect_cardinal: N/NE/E/SE/S/SW/W/NW
           - north_facing: 1 if aspect 315-45, else 0
           """

       def compute_exposure_features(self, df: pd.DataFrame) -> pd.DataFrame:
           """
           Features:
           - wind_exposure: Based on TPI (positive = exposed ridge)
           - solar_exposure: Based on aspect (south-facing = more sun)
           - terrain_roughness: Standard deviation of nearby elevations
           """

       def compute_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
           """
           Features:
           - distance_to_coast_km: Distance to Pacific coast (maritime influence)
           - latitude_normalized: Latitude scaled 0-1 for Western US
           """
   ```

2. Create `tests/features/test_terrain.py` with 15+ tests

## Implementation Notes

1. Reuse DEMPipeline's calculations where possible
2. Elevation bands:
   - low: < 2000m
   - mid: 2000-2500m
   - high: 2500-3000m
   - alpine: > 3000m
3. For distance to coast, use a simplified Pacific coastline approximation

## File Ownership

You own:
- `src/snowforecast/features/terrain.py`
- `tests/features/test_terrain.py`

## When Complete

1. Update `.claude/handoff.md`
2. Commit referencing Issue #13
3. Push to origin/phase2/13-terrain
