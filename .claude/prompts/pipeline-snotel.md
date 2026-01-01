# Agent Prompt: SNOTEL Pipeline

## Your Assignment

You are implementing the SNOTEL data ingestion pipeline for the snowforecast project.

**Issue**: #2 - Implement SNOTEL data ingestion pipeline
**Branch**: `phase1/2-snotel-pipeline`
**Worktree**: `~/snowforecast-worktrees/pipeline-snotel`

## Context

SNOTEL (Snow Telemetry) is the primary ground truth data source. It's a network of 800+ automated stations in the Western US mountains that measure snow depth, Snow Water Equivalent (SWE), and temperature.

## Your Deliverables

1. `src/snowforecast/pipelines/snotel.py` - Main pipeline class
2. `tests/pipelines/test_snotel.py` - Unit tests
3. Update `pyproject.toml` - Add `metloom` to `[project.optional-dependencies.snotel]`
4. Update `.claude/handoff.md` - Document completion status

## Technical Requirements

### Library: metloom

```python
from metloom.pointdata import SnotelPointData
from metloom.variables import SnotelVariables

# Get station metadata
station = SnotelPointData("1050:CO:SNTL", "Test Station")

# Get available variables
# SnotelVariables.SWE - Snow Water Equivalent
# SnotelVariables.SNOWDEPTH - Snow Depth
# SnotelVariables.TEMPAVG - Average Temperature

# Download data
df = station.get_daily_data(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2020, 12, 31),
    variables=[SnotelVariables.SNOWDEPTH, SnotelVariables.SWE]
)
```

### Pipeline Interface

Implement this interface (defined in `src/snowforecast/utils/base.py`):

```python
from pathlib import Path
from dataclasses import dataclass
import pandas as pd

@dataclass
class StationMetadata:
    station_id: str
    name: str
    lat: float
    lon: float
    elevation: float
    state: str

class SnotelPipeline:
    """SNOTEL data ingestion pipeline."""

    def get_station_metadata(self, state: str = None) -> list[StationMetadata]:
        """Get metadata for all SNOTEL stations, optionally filtered by state."""
        ...

    def download_station(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        variables: list[str] = None
    ) -> Path:
        """Download data for a single station to data/raw/snotel/"""
        ...

    def download_all_stations(
        self,
        start_date: str,
        end_date: str,
        states: list[str] = None
    ) -> list[Path]:
        """Download data for all stations (or filtered by state)"""
        ...

    def process(self, raw_path: Path) -> pd.DataFrame:
        """Process raw data into standardized format"""
        ...

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate processed data, return ValidationResult"""
        ...
```

### Data Output Format

Save as Parquet with this schema:

| Column | Type | Description |
|--------|------|-------------|
| `station_id` | str | SNOTEL station ID |
| `datetime` | datetime64 | UTC timestamp |
| `snow_depth_cm` | float | Snow depth in cm |
| `swe_mm` | float | Snow Water Equivalent in mm |
| `temp_avg_c` | float | Average temp in Celsius |
| `quality_flag` | str | Data quality flag |

### File Paths

```python
from snowforecast.utils.io import get_data_path

raw_dir = get_data_path("snotel", "raw")       # data/raw/snotel/
processed_dir = get_data_path("snotel", "processed")  # data/processed/snotel/
```

### Error Handling

- Handle network timeouts gracefully (retry 3 times)
- Log stations with no data (don't fail)
- Preserve quality flags from source

## Tests to Implement

```python
# tests/pipelines/test_snotel.py

def test_get_station_metadata():
    """Should return list of StationMetadata objects."""

def test_get_station_metadata_by_state():
    """Should filter stations by state code."""

def test_download_single_station(tmp_path):
    """Should download data for one station to parquet."""

def test_process_raw_data():
    """Should convert raw data to standardized format."""

def test_validate_good_data():
    """Should pass validation for complete data."""

def test_validate_missing_data():
    """Should report missing data percentage."""

def test_units_conversion():
    """Should convert to metric units correctly."""
```

## Before You Start

```bash
cd ~/snowforecast-worktrees/pipeline-snotel
git fetch origin
git rebase origin/develop  # Get latest shared code
```

## When Complete

1. Run tests: `pytest tests/pipelines/test_snotel.py -v`
2. Update `.claude/handoff.md`:
   ```markdown
   ## Status: Complete
   ## Files: src/snowforecast/pipelines/snotel.py, tests/pipelines/test_snotel.py
   ## Tests: pytest tests/pipelines/test_snotel.py ✅ (X passed)
   ## Deps: metloom>=0.3.0
   ## Blocking: None
   ```
3. Commit and push:
   ```bash
   git add .
   git commit -m "Implement SNOTEL data ingestion pipeline (#2)"
   git push origin phase1/2-snotel-pipeline
   ```

## Resources

- [metloom documentation](https://github.com/M3Works/metloom)
- [SNOTEL data info](https://www.nrcs.usda.gov/wps/portal/wcc/home/snowClimateMonitoring/snowpack/)
- [SNOTEL station map](https://www.nrcs.usda.gov/Internet/WCIS/AWS_PLOTS/siteCharts/POR/WTEQ/CO/)
