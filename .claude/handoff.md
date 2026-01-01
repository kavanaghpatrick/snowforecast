# Agent Handoff Document

## Current Status
- [x] Complete
- [ ] In Progress
- [ ] Blocked

## Task Definition
Issue #5: Implement HRRR archive data ingestion pipeline
Branch: phase1/5-hrrr-pipeline
Description: Create HRRR data ingestion pipeline using the herbie-data library

## Files Created/Modified
- `src/snowforecast/pipelines/hrrr.py` - Main HRRRPipeline class inheriting from GriddedPipeline
- `tests/pipelines/__init__.py` - Tests module init
- `tests/pipelines/test_hrrr.py` - Unit tests (20 tests)

## Dependencies
Already in pyproject.toml:
- herbie-data>=2024.1.0
- cfgrib>=0.9.10

## HRRRPipeline API

### Constructor
```python
HRRRPipeline(
    product: str = "sfc",           # HRRR product type
    bbox: BoundingBox | None = None, # Bounding box (default: Western US)
    max_workers: int = 4,           # Parallel download workers
    save_format: str = "netcdf",    # Output format (netcdf or zarr)
)
```

### Key Methods

```python
# Download analysis (f00) for a single date
ds = pipeline.download_analysis("2023-01-15", variables=["TMP:2 m above ground"])

# Download forecasts for multiple lead times
results = pipeline.download_forecast("2023-01-15", forecast_hours=[0, 6, 12, 18, 24])

# Download date range (parallel by default)
paths = pipeline.download_date_range("2023-01-01", "2023-01-31")

# Extract values at specific points
df = pipeline.extract_at_points(ds, [(40.0, -105.0), (39.5, -106.0)])

# Full pipeline run (inherited from GriddedPipeline)
df, validation = pipeline.run("2023-01-01", "2023-01-31")
```

### Default Variables
- TMP:2 m above ground (2m temperature)
- SNOD:surface (Snow depth)
- WEASD:surface (Water equivalent of accumulated snow depth)
- PRATE:surface (Precipitation rate)
- CSNOW:surface (Categorical snow)

Extended variables also include:
- UGRD:10 m above ground (U wind)
- VGRD:10 m above ground (V wind)

### Data Output
Files saved to: `data/raw/hrrr/{year}/{month}/hrrr_YYYYMMDD_fXX.nc`

## Tests Status
- [x] Unit tests pass
- [x] Coverage verified

```
59 passed, 1 skipped in 0.41s
- tests/pipelines/test_hrrr.py (20 tests, 19 passed, 1 skipped)
  - TestHRRRPipelineInit (3 tests)
  - TestHRRRPipelineValidation (5 tests)
  - TestHRRRPipelineExtractAtPoints (2 tests)
  - TestHRRRPipelineDefaultVariables (2 tests)
  - TestHRRRPipelineProcessing (3 tests)
  - TestHRRRPipelineSaveDataset (3 tests, 1 skipped - zarr not installed)
  - TestHRRRPipelineInheritance (2 tests)
- tests/utils/* (40 tests)
```

## Integration Tests
Integration tests are available but marked with @pytest.mark.integration and @pytest.mark.slow.
These require herbie library and network access.

Run with: `pytest -m integration tests/pipelines/test_hrrr.py`

## Outstanding Work
- None

## Blocking Items
- None

## Notes for Next Agent
1. The herbie library lazily imports - tests can run without it installed
2. Use `from snowforecast.pipelines.hrrr import HRRRPipeline` after merging
3. Pipeline inherits from GriddedPipeline which provides run() method
4. For production use, install with: `pip install .[hrrr]`
5. Zarr support requires additional `zarr` package installation
