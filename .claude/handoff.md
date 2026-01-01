# Agent Handoff Document

## Current Status
- [x] Complete
- [ ] In Progress
- [ ] Blocked

## Task Definition
Issue #3: Implement GHCN-Daily data ingestion pipeline
Branch: phase1/3-ghcn-pipeline
Description: GHCN-Daily data ingestion pipeline for supplemental ground truth on snowfall and temperature

## Files Created/Modified
- `src/snowforecast/pipelines/ghcn.py` - Main GHCN pipeline class with:
  - `GHCNStation` dataclass for station metadata
  - `GHCNPipeline` class inheriting from `TemporalPipeline`
  - Methods: `get_station_inventory()`, `download_station()`, `download_stations()`, `parse_dly_file()`, `filter_mountain_stations()`, `convert_units()`, `filter_quality()`
  - Full implementation of `download()`, `process()`, and `validate()` from TemporalPipeline interface
- `tests/pipelines/__init__.py` - Pipelines test module init
- `tests/pipelines/test_ghcn.py` - Unit tests (22 tests)

## Dependencies Added
Uses existing dependencies from `[project.optional-dependencies.ghcn]`:
- requests>=2.28.0

## Tests Status
- [x] Unit tests pass
- [x] Full test suite passes (62 tests)

```
22 passed in tests/pipelines/test_ghcn.py:
- TestGHCNStation (1 test)
- TestGetStationInventory (2 tests)
- TestFilterByBbox (3 tests)
- TestFilterMountainStations (2 tests)
- TestParseDlyFile (3 tests)
- TestQualityFlags (3 tests)
- TestUnitConversion (1 test)
- TestValidation (4 tests)
- TestPipelineIntegration (3 tests)
```

## GHCN Pipeline Features

### Data Sources
- Station inventory: https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt
- Station data: https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly

### Variables Supported
| Element | Description | Raw Units | Converted Units |
|---------|-------------|-----------|-----------------|
| TMAX | Maximum temperature | tenths of C | C |
| TMIN | Minimum temperature | tenths of C | C |
| PRCP | Precipitation | tenths of mm | mm |
| SNOW | Snowfall | mm | cm |
| SNWD | Snow depth | mm | cm |

### Quality Flags Handled
D, G, I, K, L, M, N, O, R, S, T, W, X, Z (see pipeline code for descriptions)

### Key Methods
```python
from snowforecast.pipelines.ghcn import GHCNPipeline, GHCNStation

pipeline = GHCNPipeline()

# Get station inventory
stations = pipeline.get_station_inventory(bbox={"west": -109, "south": 37, "east": -102, "north": 41})

# Filter mountain stations
mountain_stations = pipeline.filter_mountain_stations(min_elevation=2000)

# Download station data
path = pipeline.download_station("USC00010008")

# Parse .dly file
df = pipeline.parse_dly_file(path)

# Convert units and filter quality
df = pipeline.convert_units(df)
df = pipeline.filter_quality(df)

# Full pipeline run
df, validation = pipeline.run("2023-01-01", "2023-12-31", station_ids=["USC00010008"])
```

## Outstanding Work
- None

## Blocking Items
- None

## Notes for Next Agent
1. The pipeline inherits from `TemporalPipeline` and implements `download()`, `process()`, and `validate()`
2. Station inventory is cached in memory after first download
3. Station data files are cached in `data/raw/ghcn/`
4. The pipeline uses fixed-width parsing for .dly files (268 chars per line)
5. Quality flags are preserved in the output DataFrame and can be filtered with `filter_quality()`
6. Unit conversion is applied during `process()` - check `value_converted` column for converted values
