# Snowforecast

ML model for predicting mountain snowfall and snow depth at Western US ski resorts.

## Installation

```bash
# Install with all dependencies
pip install -e ".[all]"

# Or install specific pipelines
pip install -e ".[snotel,dev]"
```

## Project Structure

```
snowforecast/
├── src/snowforecast/
│   ├── pipelines/     # Data ingestion pipelines
│   ├── features/      # Feature engineering
│   ├── models/        # ML models
│   └── utils/         # Shared utilities
├── data/
│   ├── raw/           # Raw data from sources
│   ├── processed/     # Processed data
│   └── cache/         # Cached data
└── tests/             # Test suite
```

## Pipelines

| Pipeline | Source | Type |
|----------|--------|------|
| SNOTEL | Snow Telemetry Network | Temporal |
| GHCN | Global Historical Climatology Network | Temporal |
| ERA5 | ERA5-Land Reanalysis | Gridded |
| HRRR | High-Resolution Rapid Refresh | Gridded |
| DEM | Copernicus Digital Elevation Model | Static |
| OpenSkiMap | Ski resort locations | Static |

## Development

```bash
# Run tests
pytest tests/ -v

# Run specific pipeline tests
pytest tests/pipelines/test_snotel.py -v
```
