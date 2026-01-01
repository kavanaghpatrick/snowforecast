# Agent Prompt: DEM Pipeline

## Your Assignment

You are implementing the Copernicus DEM terrain data pipeline.

**Issue**: #6 - Implement Copernicus DEM terrain data pipeline
**Branch**: `phase1/6-dem-pipeline`
**Worktree**: `~/snowforecast-worktrees/pipeline-dem`

## Context

The Copernicus GLO-30 DEM provides 30-meter resolution elevation data globally. We use it to derive terrain features (elevation, slope, aspect) that are critical for understanding snow accumulation patterns.

## Your Deliverables

1. `src/snowforecast/pipelines/dem.py` - Main pipeline class
2. `tests/pipelines/test_dem.py` - Unit tests
3. Update `pyproject.toml` - Add deps to `[project.optional-dependencies.dem]`
4. Update `.claude/handoff.md`

## Technical Requirements

### Data Access

Copernicus DEM is on AWS Open Data:
```
s3://copernicus-dem-30m/
```

Tiles are named by lat/lon:
```
Copernicus_DSM_COG_10_N40_00_W112_00_DEM/Copernicus_DSM_COG_10_N40_00_W112_00_DEM.tif
```

### Libraries

```python
import rasterio
import rioxarray
import numpy as np
from scipy import ndimage
```

### Pipeline Interface

```python
from pathlib import Path
from dataclasses import dataclass
import numpy as np

@dataclass
class TerrainFeatures:
    elevation: float        # meters
    slope: float           # degrees
    aspect: float          # degrees from north (0-360)
    aspect_sin: float      # sin(aspect) for cyclical encoding
    aspect_cos: float      # cos(aspect)
    roughness: float       # terrain roughness index
    tpi: float             # topographic position index
    regional_elev_1km: float   # mean elevation in 1km radius
    regional_elev_5km: float   # mean elevation in 5km radius

class DEMPipeline:
    """Copernicus DEM terrain data pipeline."""

    def download_tile(self, lat: int, lon: int) -> Path:
        """Download a single DEM tile."""
        ...

    def download_region(
        self,
        bbox: dict  # west, east, south, north
    ) -> Path:
        """Download and merge all tiles for a region."""
        ...

    def get_elevation(self, lat: float, lon: float) -> float:
        """Get elevation at a point (bilinear interpolation)."""
        ...

    def calculate_slope(self, dem: np.ndarray, resolution: float) -> np.ndarray:
        """Calculate slope in degrees from DEM."""
        ...

    def calculate_aspect(self, dem: np.ndarray, resolution: float) -> np.ndarray:
        """Calculate aspect (direction of slope) in degrees."""
        ...

    def calculate_roughness(self, dem: np.ndarray, window: int = 3) -> np.ndarray:
        """Calculate terrain roughness index."""
        ...

    def calculate_tpi(self, dem: np.ndarray, radius: int = 10) -> np.ndarray:
        """Calculate topographic position index."""
        ...

    def get_terrain_features(self, lat: float, lon: float) -> TerrainFeatures:
        """Get all terrain features for a point."""
        ...

    def get_features_for_stations(
        self,
        stations: list[dict]  # [{lat, lon, station_id}, ...]
    ) -> pd.DataFrame:
        """Get terrain features for all station locations."""
        ...
```

### Terrain Calculations

**Slope** (degrees):
```python
def calculate_slope(dem, cell_size=30):
    """Calculate slope using numpy gradient."""
    dy, dx = np.gradient(dem, cell_size)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope_rad)
```

**Aspect** (degrees from north):
```python
def calculate_aspect(dem, cell_size=30):
    """Calculate aspect (0=N, 90=E, 180=S, 270=W)."""
    dy, dx = np.gradient(dem, cell_size)
    aspect_rad = np.arctan2(-dx, dy)  # Note: -dx for correct orientation
    aspect_deg = np.degrees(aspect_rad)
    # Convert to 0-360 range
    aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360, aspect_deg)
    return aspect_deg
```

**Terrain Roughness Index**:
```python
def calculate_roughness(dem, window=3):
    """TRI = mean of absolute differences from neighbors."""
    from scipy.ndimage import uniform_filter
    mean = uniform_filter(dem, size=window)
    roughness = uniform_filter(np.abs(dem - mean), size=window)
    return roughness
```

**Topographic Position Index**:
```python
def calculate_tpi(dem, radius=10):
    """TPI = elevation - mean elevation in neighborhood.
    Positive = ridge, Negative = valley
    """
    from scipy.ndimage import uniform_filter
    mean = uniform_filter(dem, size=2*radius+1)
    return dem - mean
```

### Output Format

Create a lookup table for stations:
```python
# data/processed/dem/terrain_features.parquet
# Columns: station_id, lat, lon, elevation, slope, aspect, ...
```

## Tests to Implement

```python
# tests/pipelines/test_dem.py

def test_download_tile():
    """Should download a single DEM tile."""

def test_get_elevation():
    """Should return correct elevation for known point."""

def test_calculate_slope():
    """Should calculate slope correctly for synthetic terrain."""

def test_calculate_aspect_north():
    """Aspect of north-facing slope should be ~0 or ~360."""

def test_aspect_cyclical_encoding():
    """sin/cos encoding should be continuous."""

def test_roughness_flat():
    """Flat terrain should have low roughness."""

def test_tpi_ridge_vs_valley():
    """Ridge should be positive, valley negative."""

def test_regional_elevation():
    """Should average elevation in radius correctly."""
```

## When Complete

Update `.claude/handoff.md` and push:
```bash
git add .
git commit -m "Implement Copernicus DEM terrain data pipeline (#6)"
git push origin phase1/6-dem-pipeline
```

## Resources

- [Copernicus DEM on AWS](https://registry.opendata.aws/copernicus-dem/)
- [rasterio documentation](https://rasterio.readthedocs.io/)
- [rioxarray documentation](https://corteva.github.io/rioxarray/)
