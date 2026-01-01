"""Tests for I/O utilities."""

import pytest
from pathlib import Path
from snowforecast.utils.io import get_data_path, get_project_root


class TestGetDataPath:
    """Tests for get_data_path function."""

    def test_valid_pipeline_and_stage(self):
        """Should return path for valid pipeline and stage."""
        path = get_data_path("snotel", "raw")
        assert path.name == "snotel"
        assert path.parent.name == "raw"
        assert "data" in str(path)

    def test_all_pipelines(self):
        """Should work for all valid pipelines."""
        pipelines = ["snotel", "ghcn", "era5", "hrrr", "dem", "openskimap"]
        for pipeline in pipelines:
            path = get_data_path(pipeline, "raw")
            assert path.name == pipeline

    def test_all_stages(self):
        """Should work for all valid stages."""
        stages = ["raw", "processed", "cache"]
        for stage in stages:
            path = get_data_path("snotel", stage)
            assert path.parent.name == stage

    def test_creates_directory(self, tmp_path, monkeypatch):
        """Should create directory if it doesn't exist."""
        # This test uses the actual implementation which creates real dirs
        path = get_data_path("snotel", "raw")
        assert path.exists()

    def test_invalid_pipeline(self):
        """Should raise ValueError for invalid pipeline."""
        with pytest.raises(ValueError, match="Invalid pipeline"):
            get_data_path("invalid", "raw")

    def test_invalid_stage(self):
        """Should raise ValueError for invalid stage."""
        with pytest.raises(ValueError, match="Invalid stage"):
            get_data_path("snotel", "invalid")

    def test_default_stage(self):
        """Should use 'raw' as default stage."""
        path = get_data_path("snotel")
        assert path.parent.name == "raw"


class TestGetProjectRoot:
    """Tests for get_project_root function."""

    def test_returns_path(self):
        """Should return a Path object."""
        root = get_project_root()
        assert isinstance(root, Path)

    def test_contains_src(self):
        """Should point to directory containing src."""
        root = get_project_root()
        assert (root / "src").exists()

    def test_contains_pyproject(self):
        """Should point to directory containing pyproject.toml."""
        root = get_project_root()
        assert (root / "pyproject.toml").exists()
