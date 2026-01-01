"""Tests for HRRR pipeline.

These tests verify the HRRR pipeline functionality. Tests marked with
@pytest.mark.integration require the herbie library and network access.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import xarray as xr

from snowforecast.pipelines.hrrr import (
    HRRRPipeline,
    DEFAULT_VARIABLES,
    EXTENDED_VARIABLES,
)
from snowforecast.utils import ValidationResult, BoundingBox, WESTERN_US_BBOX


class TestHRRRPipelineInit:
    """Tests for HRRRPipeline initialization."""

    def test_default_init(self):
        """Should initialize with default parameters."""
        pipeline = HRRRPipeline()
        assert pipeline.product == "sfc"
        assert pipeline.bbox == WESTERN_US_BBOX
        assert pipeline.max_workers == 4
        assert pipeline.save_format == "netcdf"

    def test_custom_init(self):
        """Should accept custom initialization parameters."""
        custom_bbox = BoundingBox(west=-120, south=35, east=-110, north=45)
        pipeline = HRRRPipeline(
            product="prs",
            bbox=custom_bbox,
            max_workers=8,
            save_format="zarr",
        )
        assert pipeline.product == "prs"
        assert pipeline.bbox == custom_bbox
        assert pipeline.max_workers == 8
        assert pipeline.save_format == "zarr"

    def test_data_path_created(self):
        """Should create data path directory."""
        pipeline = HRRRPipeline()
        assert pipeline._data_path.exists()
        assert "hrrr" in str(pipeline._data_path)


class TestHRRRPipelineValidation:
    """Tests for HRRR data validation."""

    def test_validate_valid_dataframe(self):
        """Should validate a valid DataFrame."""
        pipeline = HRRRPipeline()
        df = pd.DataFrame({
            "latitude": [40.0, 40.5, 41.0],
            "longitude": [-105.0, -105.5, -106.0],
            "t2m": [280.0, 275.0, 270.0],  # Temperature in Kelvin
            "sde": [0.5, 1.0, 1.5],  # Snow depth in meters
        })

        result = pipeline.validate(df)
        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert result.total_rows == 3
        assert result.missing_pct == 0.0

    def test_validate_empty_dataframe(self):
        """Should mark empty DataFrame as invalid."""
        pipeline = HRRRPipeline()
        df = pd.DataFrame()

        result = pipeline.validate(df)
        assert result.valid is False
        assert result.total_rows == 0
        assert "No data" in result.issues

    def test_validate_high_missing(self):
        """Should flag high missing percentage."""
        pipeline = HRRRPipeline()
        df = pd.DataFrame({
            "latitude": [40.0, 40.5, 41.0, None, None],
            "longitude": [-105.0, None, -106.0, None, None],
            "t2m": [280.0, None, None, None, None],
        })

        result = pipeline.validate(df)
        assert result.valid is False
        assert result.missing_pct > 50
        assert any("missing" in issue.lower() for issue in result.issues)

    def test_validate_xarray_dataset(self):
        """Should validate an xarray Dataset."""
        pipeline = HRRRPipeline()
        ds = xr.Dataset({
            "t2m": (["latitude", "longitude"], [[280.0, 275.0], [270.0, 265.0]]),
        }, coords={
            "latitude": [40.0, 41.0],
            "longitude": [-105.0, -106.0],
        })

        result = pipeline.validate(ds)
        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert result.total_rows == 4  # Flattened to DataFrame

    def test_validate_temperature_out_of_range(self):
        """Should flag temperature values outside reasonable range."""
        pipeline = HRRRPipeline()
        df = pd.DataFrame({
            "latitude": [40.0, 40.5],
            "longitude": [-105.0, -105.5],
            "t2m": [100.0, 500.0],  # Unreasonable Kelvin values
        })

        result = pipeline.validate(df)
        assert result.valid is False
        assert any("temperature" in issue.lower() for issue in result.issues)


class TestHRRRPipelineExtractAtPoints:
    """Tests for point extraction from gridded data."""

    def test_extract_at_points_basic(self):
        """Should extract values at specified points."""
        pipeline = HRRRPipeline()

        # Create a test dataset
        ds = xr.Dataset({
            "t2m": (["latitude", "longitude"], [[280.0, 275.0, 270.0], [285.0, 280.0, 275.0]]),
            "sde": (["latitude", "longitude"], [[0.5, 1.0, 1.5], [0.0, 0.5, 1.0]]),
        }, coords={
            "latitude": [40.0, 41.0],
            "longitude": [-107.0, -106.0, -105.0],
        })
        ds.attrs["date"] = "2023-01-15"
        ds.attrs["forecast_hour"] = 0

        points = [(40.0, -106.0), (41.0, -105.0)]
        df = pipeline.extract_at_points(ds, points)

        assert len(df) == 2
        assert "t2m" in df.columns
        assert "sde" in df.columns
        assert "lat" in df.columns
        assert "lon" in df.columns
        assert "date" in df.columns
        assert "forecast_hour" in df.columns

    def test_extract_at_points_nearest_neighbor(self):
        """Should use nearest neighbor interpolation."""
        pipeline = HRRRPipeline()

        ds = xr.Dataset({
            "t2m": (["latitude", "longitude"], [[280.0]]),
        }, coords={
            "latitude": [40.0],
            "longitude": [-105.0],
        })
        ds.attrs["date"] = "2023-01-15"
        ds.attrs["forecast_hour"] = 0

        # Request a point not exactly on the grid
        points = [(40.1, -105.1)]
        df = pipeline.extract_at_points(ds, points)

        assert len(df) == 1
        assert df.iloc[0]["actual_lat"] == 40.0
        assert df.iloc[0]["actual_lon"] == -105.0


class TestHRRRPipelineDefaultVariables:
    """Tests for variable constants."""

    def test_default_variables_defined(self):
        """Should have default variables defined."""
        assert len(DEFAULT_VARIABLES) >= 5
        assert "TMP:2 m above ground" in DEFAULT_VARIABLES
        assert "SNOD:surface" in DEFAULT_VARIABLES
        assert "WEASD:surface" in DEFAULT_VARIABLES

    def test_extended_variables_include_wind(self):
        """Extended variables should include wind components."""
        assert "UGRD:10 m above ground" in EXTENDED_VARIABLES
        assert "VGRD:10 m above ground" in EXTENDED_VARIABLES
        # Extended should include all default
        for var in DEFAULT_VARIABLES:
            assert var in EXTENDED_VARIABLES


class TestHRRRPipelineProcessing:
    """Tests for data processing methods."""

    def test_process_to_dataset_single_file(self, tmp_path):
        """Should load a single NetCDF file as Dataset."""
        pipeline = HRRRPipeline()

        # Create a test NetCDF file
        ds = xr.Dataset({
            "t2m": (["latitude", "longitude"], [[280.0, 275.0], [270.0, 265.0]]),
        }, coords={
            "latitude": [40.0, 41.0],
            "longitude": [-105.0, -106.0],
        })
        test_file = tmp_path / "test.nc"
        ds.to_netcdf(test_file)

        # Load it back
        result = pipeline.process_to_dataset(test_file)
        assert isinstance(result, xr.Dataset)
        assert "t2m" in result.data_vars

    def test_process_to_dataset_multiple_files(self, tmp_path):
        """Should concatenate multiple files."""
        pipeline = HRRRPipeline()

        files = []
        for i, time in enumerate(["2023-01-15", "2023-01-16"]):
            ds = xr.Dataset({
                "t2m": (["time", "latitude", "longitude"],
                        [[[280.0 + i, 275.0 + i], [270.0 + i, 265.0 + i]]]),
            }, coords={
                "time": [pd.Timestamp(time)],
                "latitude": [40.0, 41.0],
                "longitude": [-105.0, -106.0],
            })
            path = tmp_path / f"test_{i}.nc"
            ds.to_netcdf(path)
            files.append(path)

        result = pipeline.process_to_dataset(files)
        assert isinstance(result, xr.Dataset)
        assert len(result.time) == 2

    def test_process_returns_dataframe(self, tmp_path):
        """Should convert dataset to DataFrame."""
        pipeline = HRRRPipeline()

        ds = xr.Dataset({
            "t2m": (["latitude", "longitude"], [[280.0, 275.0], [270.0, 265.0]]),
        }, coords={
            "latitude": [40.0, 41.0],
            "longitude": [-105.0, -106.0],
        })
        test_file = tmp_path / "test.nc"
        ds.to_netcdf(test_file)

        df = pipeline.process(test_file)
        assert isinstance(df, pd.DataFrame)
        assert "t2m" in df.columns
        assert "latitude" in df.columns
        assert "longitude" in df.columns


class TestHRRRPipelineSaveDataset:
    """Tests for saving datasets to disk."""

    def test_save_dataset_netcdf(self, tmp_path):
        """Should save dataset as NetCDF."""
        pipeline = HRRRPipeline(save_format="netcdf")
        pipeline._data_path = tmp_path

        ds = xr.Dataset({
            "t2m": (["latitude", "longitude"], [[280.0]]),
        }, coords={
            "latitude": [40.0],
            "longitude": [-105.0],
        })

        path = pipeline._save_dataset(ds, "2023-01-15", fxx=0)

        assert path.exists()
        assert path.suffix == ".nc"
        assert "hrrr_20230115_f00" in path.name

    def test_save_dataset_zarr(self, tmp_path):
        """Should save dataset as Zarr."""
        pytest.importorskip("zarr")
        pipeline = HRRRPipeline(save_format="zarr")
        pipeline._data_path = tmp_path

        ds = xr.Dataset({
            "t2m": (["latitude", "longitude"], [[280.0]]),
        }, coords={
            "latitude": [40.0],
            "longitude": [-105.0],
        })

        path = pipeline._save_dataset(ds, "2023-01-15", fxx=0)

        assert path.exists()
        assert path.suffix == ".zarr"

    def test_save_creates_year_month_dirs(self, tmp_path):
        """Should create year/month directory structure."""
        pipeline = HRRRPipeline()
        pipeline._data_path = tmp_path

        ds = xr.Dataset({
            "t2m": (["latitude", "longitude"], [[280.0]]),
        }, coords={
            "latitude": [40.0],
            "longitude": [-105.0],
        })

        path = pipeline._save_dataset(ds, "2023-03-15", fxx=6)

        assert "2023" in str(path)
        assert "03" in str(path)
        assert "_f06" in path.name


class TestHRRRPipelineInheritance:
    """Tests for class inheritance."""

    def test_inherits_from_gridded_pipeline(self):
        """Should inherit from GriddedPipeline."""
        from snowforecast.utils.base import GriddedPipeline
        assert issubclass(HRRRPipeline, GriddedPipeline)

    def test_has_required_methods(self):
        """Should implement all required abstract methods."""
        pipeline = HRRRPipeline()

        # From GriddedPipeline
        assert hasattr(pipeline, "extract_at_points")
        assert hasattr(pipeline, "process_to_dataset")

        # From TemporalPipeline
        assert hasattr(pipeline, "download")
        assert hasattr(pipeline, "process")
        assert hasattr(pipeline, "validate")

        # From BasePipeline
        assert hasattr(pipeline, "run")


# Integration tests - require herbie library and network access
@pytest.mark.integration
class TestHRRRPipelineIntegration:
    """Integration tests that require herbie and network access.

    Run with: pytest -m integration tests/pipelines/test_hrrr.py
    """

    @pytest.mark.slow
    def test_download_single_analysis(self):
        """Should download f00 analysis for one date."""
        pipeline = HRRRPipeline()
        # Use a known available date
        ds = pipeline.download_analysis("2023-01-15", variables=["TMP:2 m above ground"])
        assert isinstance(ds, xr.Dataset)
        assert ds.attrs["date"] == "2023-01-15"
        assert ds.attrs["forecast_hour"] == 0

    @pytest.mark.slow
    def test_download_with_bbox(self):
        """Should subset to bounding box."""
        small_bbox = BoundingBox(west=-110, south=39, east=-105, north=41)
        pipeline = HRRRPipeline(bbox=small_bbox)
        ds = pipeline.download_analysis("2023-01-15", variables=["TMP:2 m above ground"])

        # Verify data is subsetted
        lats = ds.latitude.values
        lons = ds.longitude.values
        assert lats.min() >= small_bbox.south - 0.5  # Allow for grid cell edges
        assert lats.max() <= small_bbox.north + 0.5
        assert lons.min() >= small_bbox.west - 0.5
        assert lons.max() <= small_bbox.east + 0.5

    @pytest.mark.slow
    def test_forecast_hours(self):
        """Should download multiple forecast lead times."""
        pipeline = HRRRPipeline()
        results = pipeline.download_forecast(
            "2023-01-15",
            forecast_hours=[0, 6],
            variables=["TMP:2 m above ground"],
        )
        assert isinstance(results, dict)
        assert 0 in results or 6 in results

    @pytest.mark.slow
    def test_handles_missing_date(self):
        """Should handle dates with no data gracefully."""
        pipeline = HRRRPipeline()
        # Use a date before HRRR archive started
        with pytest.raises(Exception):
            pipeline.download_analysis("2010-01-01")
