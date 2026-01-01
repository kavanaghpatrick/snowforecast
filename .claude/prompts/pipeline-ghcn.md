# Agent Prompt: GHCN-Daily Pipeline

## Your Assignment

You are implementing the GHCN-Daily (Global Historical Climatology Network) data ingestion pipeline.

**Issue**: #3 - Implement GHCN-Daily data ingestion pipeline
**Branch**: `phase1/3-ghcn-pipeline`
**Worktree**: `~/snowforecast-worktrees/pipeline-ghcn`

## Context

GHCN-Daily provides daily climate summaries from land surface stations worldwide. We use it as supplemental ground truth for snowfall and temperature, with 100+ years of history.

## Your Deliverables

1. `src/snowforecast/pipelines/ghcn.py` - Main pipeline class
2. `tests/pipelines/test_ghcn.py` - Unit tests
3. Update `pyproject.toml` - Add deps to `[project.optional-dependencies.ghcn]`
4. Update `.claude/handoff.md`

## Technical Requirements

### Data Access

GHCN-Daily data is available via:
1. NOAA FTP: `ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/`
2. AWS Open Data: `s3://noaa-ghcn-pds/`

### Data Format

Files are CSV with fixed-width fields:
```
USC00010008,20230101,TMAX,156,,,7,
USC00010008,20230101,TMIN,28,,,7,
USC00010008,20230101,PRCP,0,,,7,
USC00010008,20230101,SNOW,0,,,7,
```

### Variables

| Element | Description | Units |
|---------|-------------|-------|
| TMAX | Maximum temperature | tenths of °C |
| TMIN | Minimum temperature | tenths of °C |
| PRCP | Precipitation | tenths of mm |
| SNOW | Snowfall | mm |
| SNWD | Snow depth | mm |

### Quality Flags

| Flag | Meaning |
|------|---------|
| (blank) | No quality flag |
| D | Failed duplicate check |
| G | Failed gap check |
| I | Failed internal consistency |
| K | Failed streak check |
| L | Failed temperature limits |
| M | Failed megaconsistency |
| N | Failed naught check |
| O | Failed climatological outlier |
| R | Failed lagged range check |
| S | Failed spatial consistency |
| T | Failed temporal consistency |
| W | Failed bounds check |
| X | Failed bounds check |
| Z | Flagged as a result of multiday accumulation |

### Pipeline Interface

```python
from pathlib import Path
import pandas as pd

@dataclass
class GHCNStation:
    station_id: str
    name: str
    lat: float
    lon: float
    elevation: float
    state: str

class GHCNPipeline:
    """GHCN-Daily data ingestion pipeline."""

    def get_station_inventory(
        self,
        bbox: dict = None,
        min_years: int = 10
    ) -> list[GHCNStation]:
        """Get station metadata filtered by region and data availability."""
        ...

    def download_station(
        self,
        station_id: str,
        start_year: int = None,
        end_year: int = None
    ) -> Path:
        """Download data for a single station."""
        ...

    def download_stations(
        self,
        station_ids: list[str],
        parallel: bool = True
    ) -> list[Path]:
        """Download data for multiple stations."""
        ...

    def parse_dly_file(self, path: Path) -> pd.DataFrame:
        """Parse GHCN .dly file format."""
        ...

    def filter_mountain_stations(
        self,
        min_elevation: float = 1500
    ) -> list[GHCNStation]:
        """Get only stations above elevation threshold."""
        ...
```

### Unit Conversion

Convert from GHCN units to standard:
```python
def convert_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert GHCN units to metric."""
    df = df.copy()
    df['tmax_c'] = df['TMAX'] / 10.0  # tenths of C → C
    df['tmin_c'] = df['TMIN'] / 10.0
    df['prcp_mm'] = df['PRCP'] / 10.0  # tenths of mm → mm
    df['snow_cm'] = df['SNOW'] / 10.0  # mm → cm
    df['snwd_cm'] = df['SNWD'] / 10.0
    return df
```

### Station Inventory URL

```
https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt
```

Format (fixed-width):
```
ID            LATITUDE   LONGITUDE  ELEVATION STATE NAME
USC00010008   32.9500   -85.9500    201.0     AL    ABBEVILLE
```

## Tests to Implement

```python
# tests/pipelines/test_ghcn.py

def test_get_station_inventory():
    """Should parse station inventory file."""

def test_filter_by_bbox():
    """Should filter stations by bounding box."""

def test_filter_mountain_stations():
    """Should filter by elevation."""

def test_parse_dly_file():
    """Should parse .dly fixed-width format."""

def test_quality_flags():
    """Should preserve quality flags."""

def test_unit_conversion():
    """Should convert to metric units."""

def test_handle_missing_data():
    """Should handle missing values correctly."""
```

## When Complete

Update `.claude/handoff.md` and push:
```bash
git add .
git commit -m "Implement GHCN-Daily data ingestion pipeline (#3)"
git push origin phase1/3-ghcn-pipeline
```

## Resources

- [GHCN-Daily documentation](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)
- [GHCN-Daily README](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt)
- [AWS GHCN-Daily](https://registry.opendata.aws/noaa-ghcn-pds/)
