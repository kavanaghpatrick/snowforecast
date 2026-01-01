# Agent Handoff Document

## Current Status
- [x] Complete
- [ ] In Progress
- [ ] Blocked

## Task Definition
Issue #2: Implement SNOTEL data ingestion pipeline
Branch: phase1/2-snotel-pipeline
Description: SNOTEL data ingestion pipeline using metloom library

## Files Created/Modified
- `src/snowforecast/pipelines/snotel.py` - Main SNOTEL pipeline class (SnotelPipeline, StationMetadata)
- `src/snowforecast/pipelines/__init__.py` - Updated to export SnotelPipeline and StationMetadata
- `tests/pipelines/__init__.py` - Test module init
- `tests/pipelines/test_snotel.py` - Unit tests for SNOTEL pipeline (21 tests)

## Dependencies Added
snotel:
- metloom>=0.3.0 (already in pyproject.toml from project setup)

## Tests Status
- [x] Unit tests pass
- [x] All project tests pass (61 tests)

```
pytest tests/pipelines/test_snotel.py -v
21 passed in 0.34s

pytest tests/ -v
61 passed in 0.37s
```

## Implementation Details

### SnotelPipeline class
Inherits from `TemporalPipeline` and implements:
- `get_station_metadata(state=None)` - Get SNOTEL station metadata, optionally filtered by state
- `download_station(station_id, start_date, end_date, variables=None)` - Download single station data
- `download_all_stations(start_date, end_date, states=None)` - Download all stations data
- `download(start_date, end_date, **kwargs)` - TemporalPipeline interface method
- `process(raw_path)` - Convert raw data to standardized format
- `validate(df)` - Validate processed data quality

### Output Schema (Parquet)
| Column | Type | Description |
|--------|------|-------------|
| station_id | str | SNOTEL station ID (e.g., "1050:CO:SNTL") |
| datetime | datetime64[UTC] | UTC timestamp |
| snow_depth_cm | float | Snow depth in centimeters |
| swe_mm | float | Snow Water Equivalent in millimeters |
| temp_avg_c | float | Average temperature in Celsius |
| quality_flag | str | 'good', 'partial', or 'missing' |

### Features
- Network timeout retry with exponential backoff (3 retries)
- Automatic unit conversion (inches to cm/mm, Fahrenheit to Celsius)
- Quality flag computation based on data completeness
- Data validation with outlier detection
- Lazy metloom import (helpful error message if not installed)

## Outstanding Work
- None

## Blocking Items
- None

## Notes for Next Agent
1. The SnotelPipeline is now available via `from snowforecast.pipelines import SnotelPipeline, StationMetadata`
2. metloom library is required - install with `pip install 'snowforecast[snotel]'`
3. Default data paths are `data/raw/snotel/` and `data/processed/snotel/`
4. Use `pipeline.run(start_date, end_date, states=["CO"])` for full pipeline execution
