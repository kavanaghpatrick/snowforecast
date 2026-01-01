# Agent Prompt: ERA5-Land Pipeline

## Your Assignment

You are implementing the ERA5-Land reanalysis data ingestion pipeline.

**Issue**: #4 - Implement ERA5-Land data ingestion pipeline
**Branch**: `phase1/4-era5-pipeline`
**Worktree**: `~/snowforecast-worktrees/pipeline-era5`

## Context

ERA5-Land is a reanalysis dataset from ECMWF/Copernicus providing hourly data at ~9km resolution from 1950 to present. This is our primary source for historical atmospheric variables.

## Your Deliverables

1. `src/snowforecast/pipelines/era5.py` - Main pipeline class
2. `tests/pipelines/test_era5.py` - Unit tests
3. Update `pyproject.toml` - Add deps to `[project.optional-dependencies.era5]`
4. Update `.claude/handoff.md`

## Technical Requirements

### Library: cdsapi

```python
import cdsapi

client = cdsapi.Client()

client.retrieve(
    'reanalysis-era5-land',
    {
        'variable': [
            '2m_temperature',
            '2m_dewpoint_temperature',
            'snow_depth',
            'total_precipitation',
        ],
        'year': '2023',
        'month': '01',
        'day': ['01', '02', '03'],
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'area': [49, -125, 31, -102],  # North, West, South, East (Western US)
        'format': 'netcdf',
    },
    'download.nc'
)
```

### CDS API Setup

Requires `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api/v2
key: {UID}:{API_KEY}
```

### Variables to Download

| Variable | ERA5 Name | Description |
|----------|-----------|-------------|
| 2m Temperature | `2m_temperature` | Air temp at 2m |
| 2m Dewpoint | `2m_dewpoint_temperature` | Dewpoint at 2m |
| 10m U Wind | `10m_u_component_of_wind` | East-west wind |
| 10m V Wind | `10m_v_component_of_wind` | North-south wind |
| Surface Pressure | `surface_pressure` | Atmospheric pressure |
| Total Precipitation | `total_precipitation` | Accumulated precip |
| Snow Depth | `snow_depth` | Snow depth (m water equiv) |
| Snowfall | `snowfall` | Snowfall rate |

### Bounding Box

Western US: `[49, -125, 31, -102]` (N, W, S, E)

### Pipeline Interface

```python
from pathlib import Path
import xarray as xr

class ERA5Pipeline:
    """ERA5-Land data ingestion pipeline."""

    def __init__(self, cache_dir: Path = None):
        self.client = cdsapi.Client()
        self.cache_dir = cache_dir or get_data_path("era5", "cache")

    def download(
        self,
        start_date: str,
        end_date: str,
        variables: list[str] = None,
        bbox: tuple = None
    ) -> Path:
        """Download ERA5-Land data for date range.

        Handles CDS queue system with retries.
        Returns path to downloaded NetCDF file.
        """
        ...

    def extract_at_points(
        self,
        nc_path: Path,
        points: list[tuple[float, float]]  # (lat, lon) pairs
    ) -> pd.DataFrame:
        """Extract time series at specific lat/lon points."""
        ...

    def to_daily(self, hourly_ds: xr.Dataset) -> xr.Dataset:
        """Aggregate hourly data to daily."""
        ...
```

### Data Output

Store as NetCDF with chunking:
```python
ds.to_netcdf(
    path,
    encoding={var: {'chunksizes': (24, 100, 100)} for var in ds.data_vars}
)
```

### Handle CDS Queue

The CDS API queues requests. Implement retry logic:

```python
import time

def download_with_retry(self, request, output_path, max_retries=5):
    for attempt in range(max_retries):
        try:
            self.client.retrieve('reanalysis-era5-land', request, output_path)
            return output_path
        except Exception as e:
            if "queue" in str(e).lower():
                wait_time = 60 * (attempt + 1)
                logger.info(f"Request queued, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError("Max retries exceeded")
```

## Tests to Implement

```python
# tests/pipelines/test_era5.py

def test_download_single_day(tmp_path):
    """Should download one day of data."""

def test_extract_at_points():
    """Should extract time series at station locations."""

def test_to_daily_aggregation():
    """Should correctly aggregate hourly to daily."""

def test_handles_queue(mocker):
    """Should retry when request is queued."""

def test_variable_units():
    """Should have correct units in output."""
```

## When Complete

Update `.claude/handoff.md` and push:
```bash
git add .
git commit -m "Implement ERA5-Land data ingestion pipeline (#4)"
git push origin phase1/4-era5-pipeline
```

## Resources

- [ERA5-Land documentation](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land)
- [CDS API documentation](https://cds.climate.copernicus.eu/api-how-to)
- [xarray documentation](https://docs.xarray.dev/)
