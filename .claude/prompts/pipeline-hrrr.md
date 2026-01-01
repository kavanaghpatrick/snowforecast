# Agent Prompt: HRRR Pipeline

## Your Assignment

You are implementing the HRRR (High-Resolution Rapid Refresh) archive data ingestion pipeline.

**Issue**: #5 - Implement HRRR archive data ingestion pipeline
**Branch**: `phase1/5-hrrr-pipeline`
**Worktree**: `~/snowforecast-worktrees/pipeline-hrrr`

## Context

HRRR is NOAA's high-resolution (3km) weather model, updated hourly. The archive on AWS provides data from 2014 to present. This is our high-resolution data source for model fine-tuning.

## Your Deliverables

1. `src/snowforecast/pipelines/hrrr.py` - Main pipeline class
2. `tests/pipelines/test_hrrr.py` - Unit tests
3. Update `pyproject.toml` - Add `herbie-data` to `[project.optional-dependencies.hrrr]`
4. Update `.claude/handoff.md`

## Technical Requirements

### Library: herbie

```python
from herbie import Herbie

# Get HRRR analysis (f00)
H = Herbie(
    "2023-01-15",
    model="hrrr",
    product="sfc",  # surface fields
    fxx=0  # Analysis (forecast hour 0)
)

# Download specific variables
ds = H.xarray("TMP:2 m")  # 2m temperature
ds = H.xarray("SNOD")     # Snow depth
ds = H.xarray("WEASD")    # Water equiv of accum snow depth

# Get subset for region
ds = H.xarray("TMP:2 m", subset=dict(lat=slice(31, 49), lon=slice(-125, -102)))
```

### Variables to Download

| Variable | GRIB Name | Description |
|----------|-----------|-------------|
| 2m Temperature | `TMP:2 m above ground` | Air temp |
| Snow Depth | `SNOD:surface` | Snow depth (m) |
| Snow Water Equiv | `WEASD:surface` | SWE (kg/m²) |
| Precipitation Rate | `PRATE:surface` | Precip rate |
| Categorical Snow | `CSNOW:surface` | Is it snowing? |
| 10m U Wind | `UGRD:10 m above ground` | U wind |
| 10m V Wind | `VGRD:10 m above ground` | V wind |

### Pipeline Interface

```python
from pathlib import Path
from datetime import datetime
import xarray as xr

class HRRRPipeline:
    """HRRR archive data ingestion pipeline."""

    def download_analysis(
        self,
        date: str,
        variables: list[str] = None,
        bbox: dict = None
    ) -> xr.Dataset:
        """Download HRRR analysis (f00) for a date.

        Args:
            date: Date string "YYYY-MM-DD"
            variables: List of GRIB variable patterns
            bbox: {"west": -125, "east": -102, "south": 31, "north": 49}

        Returns:
            xarray Dataset with requested variables
        """
        ...

    def download_forecast(
        self,
        date: str,
        forecast_hours: list[int] = None,
        variables: list[str] = None
    ) -> dict[int, xr.Dataset]:
        """Download HRRR forecast for multiple lead times.

        Returns dict mapping forecast hour to Dataset.
        """
        ...

    def extract_at_points(
        self,
        ds: xr.Dataset,
        points: list[tuple[float, float]]
    ) -> pd.DataFrame:
        """Extract values at specific lat/lon points."""
        ...

    def download_date_range(
        self,
        start_date: str,
        end_date: str,
        variables: list[str] = None,
        parallel: bool = True
    ) -> list[Path]:
        """Download multiple days, optionally in parallel."""
        ...
```

### Parallel Downloads

HRRR data is large. Use parallel downloads:

```python
from concurrent.futures import ThreadPoolExecutor

def download_date_range(self, start, end, variables, parallel=True):
    dates = pd.date_range(start, end)

    if parallel:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(
                lambda d: self.download_analysis(d, variables),
                dates
            ))
    else:
        results = [self.download_analysis(d, variables) for d in dates]

    return results
```

### Data Output

Save as Zarr for efficient chunked access:
```python
ds.to_zarr(path, mode='w')
```

Or as individual NetCDF files per date:
```
data/raw/hrrr/2023/01/hrrr_20230115_f00.nc
```

## Tests to Implement

```python
# tests/pipelines/test_hrrr.py

def test_download_single_analysis():
    """Should download f00 analysis for one date."""

def test_download_with_bbox():
    """Should subset to bounding box."""

def test_extract_at_points():
    """Should extract values at station locations."""

def test_forecast_hours():
    """Should download multiple forecast lead times."""

def test_parallel_download():
    """Should download multiple dates in parallel."""

def test_handles_missing_date():
    """Should handle dates with no data gracefully."""
```

## When Complete

Update `.claude/handoff.md` and push:
```bash
git add .
git commit -m "Implement HRRR archive data ingestion pipeline (#5)"
git push origin phase1/5-hrrr-pipeline
```

## Resources

- [Herbie documentation](https://herbie.readthedocs.io/)
- [HRRR on AWS](https://registry.opendata.aws/noaa-hrrr-pds/)
- [HRRR variable list](https://www.nco.ncep.noaa.gov/pmb/products/hrrr/)
