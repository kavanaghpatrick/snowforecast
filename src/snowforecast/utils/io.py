"""I/O utilities for data paths and file operations."""

from pathlib import Path

# Project root is 4 levels up from this file
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def get_data_path(pipeline: str, stage: str = "raw") -> Path:
    """Get standardized data path for a pipeline.

    Args:
        pipeline: One of 'snotel', 'ghcn', 'era5', 'hrrr', 'dem', 'openskimap'
        stage: One of 'raw', 'processed', 'cache'

    Returns:
        Path to the data directory (creates if doesn't exist)

    Example:
        >>> path = get_data_path("snotel", "raw")
        >>> path
        PosixPath('.../snowforecast/data/raw/snotel')
    """
    valid_pipelines = {"snotel", "ghcn", "era5", "hrrr", "dem", "openskimap"}
    valid_stages = {"raw", "processed", "cache"}

    if pipeline not in valid_pipelines:
        raise ValueError(f"Invalid pipeline: {pipeline}. Must be one of {valid_pipelines}")
    if stage not in valid_stages:
        raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")

    path = _PROJECT_ROOT / "data" / stage / pipeline
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Get the project root directory."""
    return _PROJECT_ROOT
