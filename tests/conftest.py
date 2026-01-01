"""Shared pytest fixtures for snowforecast tests."""

import pytest
from pathlib import Path
import tempfile


@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root) -> Path:
    """Get the data directory."""
    return project_root / "data"


@pytest.fixture
def tmp_data_dir(tmp_path) -> Path:
    """Create a temporary data directory for testing."""
    data_dir = tmp_path / "data"
    for stage in ["raw", "processed", "cache"]:
        for pipeline in ["snotel", "ghcn", "era5", "hrrr", "dem", "openskimap"]:
            (data_dir / stage / pipeline).mkdir(parents=True)
    return data_dir


@pytest.fixture
def sample_stations() -> list[dict]:
    """Sample SNOTEL/GHCN station data for testing."""
    return [
        {
            "station_id": "1050:CO:SNTL",
            "name": "Berthoud Summit",
            "lat": 39.80,
            "lon": -105.78,
            "elevation": 3450,
            "state": "CO",
        },
        {
            "station_id": "1051:CO:SNTL",
            "name": "Loveland Basin",
            "lat": 39.68,
            "lon": -105.90,
            "elevation": 3520,
            "state": "CO",
        },
        {
            "station_id": "978:UT:SNTL",
            "name": "Brighton",
            "lat": 40.60,
            "lon": -111.58,
            "elevation": 2670,
            "state": "UT",
        },
    ]


@pytest.fixture
def sample_bbox() -> dict:
    """Sample bounding box for testing (small area in Colorado)."""
    return {
        "west": -106.0,
        "south": 39.5,
        "east": -105.5,
        "north": 40.0,
    }


@pytest.fixture
def western_us_bbox() -> dict:
    """Full Western US bounding box."""
    return {
        "west": -125.0,
        "south": 31.0,
        "east": -102.0,
        "north": 49.0,
    }
