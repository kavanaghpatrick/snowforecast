"""Shared pytest fixtures for snowforecast tests.

Test Tiers:
- unit: Fast tests with fixtures, no network (default)
- integration: Tests with VCR cassettes (recorded API responses)
- live: Real API tests, slow, requires network and may need credentials

Run live tests with: pytest -m live --run-live
"""

from pathlib import Path

import pytest


def pytest_addoption(parser):
    """Add command line options for test configuration."""
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run live API tests (slow, requires network)",
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: fast unit tests using fixtures")
    config.addinivalue_line("markers", "integration: tests with recorded API responses")
    config.addinivalue_line("markers", "live: real API tests (slow, requires network)")


def pytest_collection_modifyitems(config, items):
    """Skip live tests unless --run-live is specified."""
    if config.getoption("--run-live"):
        # --run-live given: don't skip live tests
        return

    skip_live = pytest.mark.skip(reason="need --run-live option to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


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
