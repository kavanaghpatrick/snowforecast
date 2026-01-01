# Agent Handoff Document

## Current Status
- [x] Complete
- [ ] In Progress
- [ ] Blocked

## Task Definition
Issue #7: Extract ski resort locations from OpenSkiMap
Branch: phase1/7-openskimap-pipeline
Description: Implement ski resort location extraction pipeline from OpenSkiMap using geojson and shapely

## Files Created/Modified
- `src/snowforecast/pipelines/openskimap.py` - OpenSkiMapPipeline class with:
  - `SkiResort` dataclass (name, lat, lon, base/summit elevation, vertical_drop, country, state, nearest_snotel)
  - `download_ski_areas()` - Download ski areas GeoJSON from OpenSkiMap
  - `download_lifts()` - Download lifts GeoJSON from OpenSkiMap
  - `parse_ski_areas()` - Parse ski area features from GeoJSON
  - `extract_elevations_from_lifts()` - Get base/summit elevation from lift endpoints
  - `filter_region()` - Filter resorts by bounding box and/or country
  - `find_nearest_snotel()` - Find nearest SNOTEL station to a resort
  - `get_western_us_resorts()` - Get all ski resorts in Western US
  - `export_to_dataframe()` - Convert to DataFrame for parquet export
  - `save_to_parquet()` - Save to processed/openskimap/resorts.parquet
  - `validate()` - Validate resort data quality
- `tests/pipelines/__init__.py` - Tests module init
- `tests/pipelines/test_openskimap.py` - Comprehensive unit tests (27 tests)

## Dependencies Used
From `[project.optional-dependencies.openskimap]`:
- geojson>=3.0.0
- shapely>=2.0.0
- requests>=2.28.0

## Tests Status
- [x] Unit tests pass
- [x] All existing tests still pass

```
67 passed in 0.33s
- tests/pipelines/test_openskimap.py (27 tests)
- tests/utils/test_base.py (14 tests)
- tests/utils/test_geo.py (16 tests)
- tests/utils/test_io.py (10 tests)
```

## Test Coverage
- TestSkiResortDataclass: SkiResort dataclass creation and optional fields
- TestDownloadSkiAreas: Download ski areas and lifts GeoJSON with mocked requests
- TestParseSkiAreas: Parse ski area features, filter nameless areas
- TestExtractElevations: Extract base/summit elevation from lift coordinates
- TestFilterRegion: Filter by Western US bounding box and country
- TestFindNearestSnotel: Find nearest SNOTEL station within max distance
- TestHaversineDistance: Haversine distance calculation (uses util from geo.py)
- TestExportDataframe: Convert to DataFrame, handle empty list
- TestValidation: Validate data quality, detect invalid coordinates
- TestGetWesternUSResorts: Full integration test of resort extraction
- TestSaveToParquet: Save DataFrame to parquet format

## Data Sources
- Ski areas: https://tiles.openskimap.org/geojson/ski_areas.geojson
- Lifts: https://tiles.openskimap.org/geojson/lifts.geojson

## Output Format
```
data/processed/openskimap/resorts.parquet
Columns: name, lat, lon, base_elevation, summit_elevation,
         vertical_drop, country, state, nearest_snotel
```

## Outstanding Work
- None

## Blocking Items
- None

## Notes for Next Agent
1. Pipeline inherits from `StaticPipeline` and follows the established pattern
2. Uses `get_data_path("openskimap", "raw")` and `get_data_path("openskimap", "processed")`
3. Uses `haversine()` from `snowforecast.utils.geo` (not reimplementing)
4. Uses `WESTERN_US_BBOX` from `snowforecast.utils` for filtering
5. Elevation data comes from lift 3D coordinates [lon, lat, elevation]
6. Returns `ValidationResult` from `validate()` method as required

## Usage Example
```python
from snowforecast.pipelines.openskimap import OpenSkiMapPipeline

# Run full pipeline
pipeline = OpenSkiMapPipeline()
resorts_df, validation = pipeline.run()
print(f"Found {len(resorts_df)} Western US ski resorts")

# Or step-by-step
resorts = pipeline.get_western_us_resorts()
df = pipeline.export_to_dataframe(resorts)
pipeline.save_to_parquet(df)
```
