# Agent Handoff Document

## Current Status
- [x] Complete
- [ ] In Progress
- [ ] Blocked

## Task Definition
Issue #4: Implement ERA5-Land data ingestion pipeline
Branch: phase1/4-era5-pipeline
Description: ERA5-Land reanalysis data ingestion pipeline using cdsapi library

## Files Created/Modified
- `src/snowforecast/pipelines/era5.py` - Main ERA5Pipeline class inheriting from GriddedPipeline
- `tests/pipelines/__init__.py` - Tests module init
- `tests/pipelines/test_era5.py` - Comprehensive unit tests (31 tests)

## Dependencies
The ERA5 optional dependencies in pyproject.toml were already configured:
- cdsapi>=0.6.0
- netcdf4>=1.6.0

## Tests Status
- [x] Unit tests pass (31 tests)
- [x] All project tests pass (71 tests total)

```
71 passed in 0.60s
- tests/pipelines/test_era5.py (31 tests)
- tests/utils/test_base.py (14 tests)
- tests/utils/test_geo.py (16 tests)
- tests/utils/test_io.py (10 tests)
```

## ERA5Pipeline Features

### Core Methods
- `download(start_date, end_date, variables, bbox, hours, force)` - Download ERA5-Land data
- `process_to_dataset(raw_path)` - Process raw NetCDF to xarray Dataset
- `process(raw_path)` - Process to pandas DataFrame
- `to_daily(hourly_ds)` - Aggregate hourly to daily (mean for temp, sum for precip)
- `extract_at_points(data, points, method)` - Extract time series at lat/lon points
- `validate(data)` - Validate Dataset or DataFrame
- `save_dataset(ds, output_path, chunksizes)` - Save with optimized chunking

### Configuration
- Default bbox: Western US (49N, -125W, 31S, -102E)
- Default variables: 2m_temperature, 2m_dewpoint_temperature, snow_depth, total_precipitation, snowfall
- CDS queue retry logic with exponential backoff (max 5 retries)

### Data Handling
- Standardizes coordinate names (latitude/longitude -> lat/lon)
- Adds processing metadata
- Handles multiple NetCDF file concatenation
- Validation checks for:
  - Missing data percentage
  - Unrealistic temperature values (Kelvin)
  - Negative precipitation values
  - Temporal gaps

## CDS API Setup Required
Users need `~/.cdsapirc` with:
```
url: https://cds.climate.copernicus.eu/api/v2
key: {UID}:{API_KEY}
```

## Usage Example
```python
from snowforecast.pipelines.era5 import ERA5Pipeline

pipeline = ERA5Pipeline()
nc_path = pipeline.download("2023-01-01", "2023-01-03")
ds = pipeline.process_to_dataset(nc_path)
daily_ds = pipeline.to_daily(ds)

# Extract at specific points
points = [(39.5, -105.5), (40.0, -106.0)]
df = pipeline.extract_at_points(ds, points)

# Validate
result = pipeline.validate(ds)
print(result)  # ValidationResult(VALID, rows=..., missing=0.0%)
```

## Outstanding Work
- None

## Blocking Items
- None

## Notes for Coordinator
1. ERA5 pipeline inherits from GriddedPipeline as specified
2. Uses get_data_path("era5", "raw") and get_data_path("era5", "cache") for paths
3. All 8 variables from the spec are mapped (t2m, d2m, u10, v10, sp, tp, sd, sf)
4. Default download uses 5 core snow-related variables
5. Tests use mock CDS client to avoid real API calls
