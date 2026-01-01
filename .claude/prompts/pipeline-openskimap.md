# Agent Prompt: OpenSkiMap Pipeline

## Your Assignment

You are implementing the ski resort location extraction pipeline from OpenSkiMap.

**Issue**: #7 - Extract ski resort locations from OpenSkiMap
**Branch**: `phase1/7-openskimap-pipeline`
**Worktree**: `~/snowforecast-worktrees/pipeline-openskimap`

## Context

OpenSkiMap is an open-source ski map built on OpenStreetMap data. We extract ski resort locations (lat/lon, base/summit elevations) to define target locations for our snow forecasts.

## Your Deliverables

1. `src/snowforecast/pipelines/openskimap.py` - Main pipeline class
2. `tests/pipelines/test_openskimap.py` - Unit tests
3. Update `pyproject.toml` - Add deps to `[project.optional-dependencies.openskimap]`
4. Update `.claude/handoff.md`

## Technical Requirements

### Data Source

OpenSkiMap provides GeoJSON exports:
- Ski areas: `https://tiles.openskimap.org/geojson/ski_areas.geojson`
- Lifts: `https://tiles.openskimap.org/geojson/lifts.geojson`
- Runs: `https://tiles.openskimap.org/geojson/runs.geojson`

Or use Overpass API for OSM data directly.

### Libraries

```python
import geojson
import requests
from shapely.geometry import shape, Point
```

### Pipeline Interface

```python
from pathlib import Path
from dataclasses import dataclass
import pandas as pd

@dataclass
class SkiResort:
    name: str
    lat: float              # Representative latitude
    lon: float              # Representative longitude
    base_elevation: float   # Base area elevation (m)
    summit_elevation: float # Summit elevation (m)
    vertical_drop: float    # Summit - base (m)
    country: str
    state: str              # For US/Canada
    nearest_snotel: str     # Nearest SNOTEL station ID

class OpenSkiMapPipeline:
    """Ski resort location extraction pipeline."""

    def download_ski_areas(self) -> Path:
        """Download ski areas GeoJSON."""
        ...

    def download_lifts(self) -> Path:
        """Download lifts GeoJSON."""
        ...

    def parse_ski_areas(self, geojson_path: Path) -> list[dict]:
        """Parse ski area features from GeoJSON."""
        ...

    def extract_elevations_from_lifts(
        self,
        ski_area_name: str,
        lifts_geojson: dict
    ) -> tuple[float, float]:
        """Get base and summit elevation from lift endpoints."""
        ...

    def filter_region(
        self,
        resorts: list[SkiResort],
        bbox: dict = None,
        countries: list[str] = None
    ) -> list[SkiResort]:
        """Filter resorts by region."""
        ...

    def find_nearest_snotel(
        self,
        resort: SkiResort,
        snotel_stations: pd.DataFrame,
        max_distance_km: float = 50
    ) -> str:
        """Find nearest SNOTEL station to resort."""
        ...

    def get_western_us_resorts(self) -> list[SkiResort]:
        """Get all ski resorts in Western US."""
        ...

    def export_to_dataframe(self, resorts: list[SkiResort]) -> pd.DataFrame:
        """Convert to DataFrame for easy use."""
        ...
```

### Western US Bounding Box

```python
WESTERN_US = {
    "west": -125.0,
    "east": -102.0,
    "south": 31.0,
    "north": 49.0
}
```

### Elevation Extraction from Lifts

Lifts have elevation data at their endpoints:

```python
def extract_elevations_from_lifts(lifts, ski_area_name):
    """Extract min/max elevation from lift endpoints."""
    elevations = []

    for lift in lifts:
        if lift['properties'].get('ski_area') == ski_area_name:
            coords = lift['geometry']['coordinates']
            # Coordinates may have [lon, lat, elevation] format
            if len(coords[0]) == 3:
                elevations.extend([c[2] for c in coords])

    if elevations:
        return min(elevations), max(elevations)
    return None, None
```

### Find Representative Point

For ski area polygons, find centroid:

```python
from shapely.geometry import shape

def get_representative_point(geometry):
    """Get representative lat/lon for a ski area geometry."""
    shp = shape(geometry)
    centroid = shp.centroid
    return centroid.y, centroid.x  # lat, lon
```

### Haversine Distance

```python
from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points."""
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    return R * c
```

### Output Format

```python
# data/processed/openskimap/resorts.parquet
# Columns: name, lat, lon, base_elevation, summit_elevation,
#          vertical_drop, country, state, nearest_snotel
```

## Tests to Implement

```python
# tests/pipelines/test_openskimap.py

def test_download_ski_areas():
    """Should download ski areas GeoJSON."""

def test_parse_ski_areas():
    """Should parse ski area features correctly."""

def test_filter_western_us():
    """Should filter to Western US only."""

def test_extract_elevations():
    """Should get base/summit from lifts."""

def test_find_nearest_snotel():
    """Should find closest SNOTEL station."""

def test_haversine_distance():
    """Should calculate distance correctly."""

def test_export_dataframe():
    """Should create valid DataFrame."""
```

## When Complete

Update `.claude/handoff.md` and push:
```bash
git add .
git commit -m "Implement OpenSkiMap ski resort location pipeline (#7)"
git push origin phase1/7-openskimap-pipeline
```

## Resources

- [OpenSkiMap](https://openskimap.org/)
- [OpenSkiMap GitHub](https://github.com/russellporter/openskimap.org)
- [Overpass API](https://overpass-turbo.eu/)
- [Shapely documentation](https://shapely.readthedocs.io/)
