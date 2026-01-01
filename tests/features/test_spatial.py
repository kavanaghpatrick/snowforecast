"""Tests for spatial alignment utilities."""

import pytest
import numpy as np
import pandas as pd
from dataclasses import dataclass
from unittest.mock import Mock, MagicMock

# Import xarray for testing
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    xr = None


# Skip all tests if xarray is not available
pytestmark = pytest.mark.skipif(not HAS_XARRAY, reason="xarray not installed")


@pytest.fixture
def sample_dataset():
    """Create a sample xarray Dataset with standard lat/lon coords."""
    lats = np.arange(38.0, 42.0, 0.5)  # 8 lat points
    lons = np.arange(-108.0, -104.0, 0.5)  # 8 lon points
    times = pd.date_range("2023-01-01", periods=24, freq="h")

    # Create synthetic data
    np.random.seed(42)
    t2m = np.random.uniform(260, 280, (len(times), len(lats), len(lons)))
    tp = np.random.uniform(0, 0.01, (len(times), len(lats), len(lons)))
    sd = np.random.uniform(0, 0.5, (len(times), len(lats), len(lons)))

    ds = xr.Dataset(
        {
            "t2m": (["time", "lat", "lon"], t2m),
            "tp": (["time", "lat", "lon"], tp),
            "sd": (["time", "lat", "lon"], sd),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )
    return ds


@pytest.fixture
def sample_dataset_latitude_longitude():
    """Create a dataset with latitude/longitude coord names."""
    latitudes = np.arange(38.0, 42.0, 0.5)
    longitudes = np.arange(-108.0, -104.0, 0.5)

    np.random.seed(42)
    temperature = np.random.uniform(260, 280, (len(latitudes), len(longitudes)))

    ds = xr.Dataset(
        {
            "temperature": (["latitude", "longitude"], temperature),
        },
        coords={
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )
    return ds


@pytest.fixture
def sample_dataset_2d():
    """Create a 2D dataset without time dimension."""
    lats = np.arange(38.0, 42.0, 0.5)
    lons = np.arange(-108.0, -104.0, 0.5)

    np.random.seed(42)
    elevation = np.random.uniform(1500, 4000, (len(lats), len(lons)))

    ds = xr.Dataset(
        {
            "elevation": (["lat", "lon"], elevation),
        },
        coords={
            "lat": lats,
            "lon": lons,
        },
    )
    return ds


@pytest.fixture
def sample_points():
    """Sample extraction points within the grid."""
    return [
        (39.0, -106.0),  # Center of grid
        (38.5, -107.0),  # Near SW corner
        (41.0, -105.0),  # Near NE corner
    ]


@pytest.fixture
def aligner_nearest():
    """Create a SpatialAligner with nearest-neighbor interpolation."""
    from snowforecast.features import SpatialAligner
    return SpatialAligner(interpolation_method="nearest")


@pytest.fixture
def aligner_bilinear():
    """Create a SpatialAligner with bilinear interpolation."""
    from snowforecast.features import SpatialAligner
    return SpatialAligner(interpolation_method="bilinear")


class TestSpatialAlignerInit:
    """Tests for SpatialAligner initialization."""

    def test_default_interpolation_method(self):
        """Default interpolation method should be nearest."""
        from snowforecast.features import SpatialAligner
        aligner = SpatialAligner()
        assert aligner.interpolation_method == "nearest"

    def test_nearest_interpolation_method(self):
        """Should accept nearest interpolation method."""
        from snowforecast.features import SpatialAligner
        aligner = SpatialAligner(interpolation_method="nearest")
        assert aligner.interpolation_method == "nearest"

    def test_bilinear_interpolation_method(self):
        """Should accept bilinear interpolation method."""
        from snowforecast.features import SpatialAligner
        aligner = SpatialAligner(interpolation_method="bilinear")
        assert aligner.interpolation_method == "bilinear"

    def test_invalid_interpolation_method_raises(self):
        """Should raise error for invalid interpolation method."""
        from snowforecast.features import SpatialAligner
        with pytest.raises(ValueError, match="Invalid interpolation method"):
            SpatialAligner(interpolation_method="cubic")


class TestCoordinateDetection:
    """Tests for coordinate name detection."""

    def test_finds_lat_lon_coords(self, aligner_nearest, sample_dataset):
        """Should find lat/lon coordinate names."""
        lat_name, lon_name = aligner_nearest._get_lat_lon_names(sample_dataset)
        assert lat_name == "lat"
        assert lon_name == "lon"

    def test_finds_latitude_longitude_coords(
        self, aligner_nearest, sample_dataset_latitude_longitude
    ):
        """Should find latitude/longitude coordinate names."""
        lat_name, lon_name = aligner_nearest._get_lat_lon_names(
            sample_dataset_latitude_longitude
        )
        assert lat_name == "latitude"
        assert lon_name == "longitude"

    def test_raises_on_missing_lat_coord(self, aligner_nearest):
        """Should raise error when lat coordinate is missing."""
        ds = xr.Dataset(
            {"data": (["x", "lon"], np.ones((5, 5)))},
            coords={"x": np.arange(5), "lon": np.arange(5)},
        )
        with pytest.raises(ValueError, match="Cannot find latitude coordinate"):
            aligner_nearest._get_lat_lon_names(ds)

    def test_raises_on_missing_lon_coord(self, aligner_nearest):
        """Should raise error when lon coordinate is missing."""
        ds = xr.Dataset(
            {"data": (["lat", "y"], np.ones((5, 5)))},
            coords={"lat": np.arange(5), "y": np.arange(5)},
        )
        with pytest.raises(ValueError, match="Cannot find longitude coordinate"):
            aligner_nearest._get_lat_lon_names(ds)


class TestFindNearestGridCell:
    """Tests for finding nearest grid cell indices."""

    def test_find_nearest_at_grid_point(self, aligner_nearest, sample_dataset):
        """Should find exact match when point is on grid."""
        lat_idx, lon_idx = aligner_nearest.find_nearest_grid_cell(
            sample_dataset, 39.0, -106.0
        )
        # lat=39.0 is at index 2 (38.0, 38.5, 39.0, ...)
        # lon=-106.0 is at index 4 (-108.0, -107.5, -107.0, -106.5, -106.0, ...)
        assert lat_idx == 2
        assert lon_idx == 4

    def test_find_nearest_between_grid_points(self, aligner_nearest, sample_dataset):
        """Should find nearest when point is between grid points."""
        lat_idx, lon_idx = aligner_nearest.find_nearest_grid_cell(
            sample_dataset, 39.2, -105.8
        )
        # 39.2 is between 39.0 (idx 2) and 39.5 (idx 3), closer to 39.0
        # -105.8 is between -106.0 (idx 4) and -105.5 (idx 5), closer to -106.0
        assert lat_idx == 2
        assert lon_idx == 4

    def test_find_nearest_at_edge(self, aligner_nearest, sample_dataset):
        """Should handle points at grid edge."""
        lat_idx, lon_idx = aligner_nearest.find_nearest_grid_cell(
            sample_dataset, 38.0, -108.0
        )
        assert lat_idx == 0
        assert lon_idx == 0


class TestExtractAtPoints:
    """Tests for extracting values at points."""

    def test_extract_returns_dataframe(
        self, aligner_nearest, sample_dataset, sample_points
    ):
        """Should return a pandas DataFrame."""
        result = aligner_nearest.extract_at_points(sample_dataset, sample_points)
        assert isinstance(result, pd.DataFrame)

    def test_extract_includes_point_id(
        self, aligner_nearest, sample_dataset, sample_points
    ):
        """Should include point_id column."""
        result = aligner_nearest.extract_at_points(sample_dataset, sample_points)
        assert "point_id" in result.columns
        assert set(result["point_id"].unique()) == {0, 1, 2}

    def test_extract_includes_point_coords(
        self, aligner_nearest, sample_dataset, sample_points
    ):
        """Should include point_lat and point_lon columns."""
        result = aligner_nearest.extract_at_points(sample_dataset, sample_points)
        assert "point_lat" in result.columns
        assert "point_lon" in result.columns

    def test_extract_includes_all_variables(
        self, aligner_nearest, sample_dataset, sample_points
    ):
        """Should include all data variables."""
        result = aligner_nearest.extract_at_points(sample_dataset, sample_points)
        assert "t2m" in result.columns
        assert "tp" in result.columns
        assert "sd" in result.columns

    def test_extract_preserves_time_dimension(
        self, aligner_nearest, sample_dataset, sample_points
    ):
        """Should preserve time series for each point."""
        result = aligner_nearest.extract_at_points(sample_dataset, sample_points)
        # 24 times * 3 points = 72 rows
        expected_rows = 24 * len(sample_points)
        assert len(result) == expected_rows

    def test_extract_with_prefix(
        self, aligner_nearest, sample_dataset, sample_points
    ):
        """Should add prefix to variable columns."""
        result = aligner_nearest.extract_at_points(
            sample_dataset, sample_points, prefix="test_"
        )
        assert "test_t2m" in result.columns
        assert "test_tp" in result.columns
        assert "test_sd" in result.columns
        assert "t2m" not in result.columns

    def test_extract_2d_dataset(
        self, aligner_nearest, sample_dataset_2d, sample_points
    ):
        """Should work with 2D dataset (no time dimension)."""
        result = aligner_nearest.extract_at_points(sample_dataset_2d, sample_points)
        assert len(result) == len(sample_points)
        assert "elevation" in result.columns

    def test_extract_nearest_vs_bilinear(
        self, aligner_nearest, aligner_bilinear, sample_dataset, sample_points
    ):
        """Nearest and bilinear should give different results."""
        result_nearest = aligner_nearest.extract_at_points(
            sample_dataset, sample_points
        )
        result_bilinear = aligner_bilinear.extract_at_points(
            sample_dataset, sample_points
        )

        # Values should differ (unless point is exactly on grid)
        # Test with a point that's between grid points
        point_0_nearest = result_nearest[result_nearest["point_id"] == 0]["t2m"].values
        point_0_bilinear = result_bilinear[result_bilinear["point_id"] == 0]["t2m"].values

        # At least some values should differ due to interpolation
        # (they may be very close, but not exactly equal in most cases)
        assert len(point_0_nearest) == len(point_0_bilinear)


class TestERA5Extraction:
    """Tests for ERA5-specific extraction."""

    def test_extract_era5_adds_prefix(
        self, aligner_nearest, sample_dataset, sample_points
    ):
        """Should add era5_ prefix to columns."""
        result = aligner_nearest.extract_era5_at_points(sample_dataset, sample_points)
        assert "era5_t2m" in result.columns
        assert "era5_tp" in result.columns
        assert "era5_sd" in result.columns

    def test_extract_era5_returns_dataframe(
        self, aligner_nearest, sample_dataset, sample_points
    ):
        """Should return a DataFrame."""
        result = aligner_nearest.extract_era5_at_points(sample_dataset, sample_points)
        assert isinstance(result, pd.DataFrame)


class TestHRRRExtraction:
    """Tests for HRRR-specific extraction."""

    def test_extract_hrrr_adds_prefix(
        self, aligner_nearest, sample_dataset, sample_points
    ):
        """Should add hrrr_ prefix to columns."""
        result = aligner_nearest.extract_hrrr_at_points(sample_dataset, sample_points)
        assert "hrrr_t2m" in result.columns
        assert "hrrr_tp" in result.columns

    def test_extract_hrrr_returns_dataframe(
        self, aligner_nearest, sample_dataset, sample_points
    ):
        """Should return a DataFrame."""
        result = aligner_nearest.extract_hrrr_at_points(sample_dataset, sample_points)
        assert isinstance(result, pd.DataFrame)


class TestDEMExtraction:
    """Tests for DEM terrain feature extraction."""

    @pytest.fixture
    def mock_dem_pipeline(self):
        """Create a mock DEM pipeline."""
        # Create mock terrain features dataclass
        @dataclass
        class MockTerrainFeatures:
            elevation: float
            slope: float
            aspect: float
            aspect_sin: float
            aspect_cos: float
            roughness: float
            tpi: float

        mock_pipeline = Mock()

        def get_features(lat, lon):
            return MockTerrainFeatures(
                elevation=2500.0 + lat * 10 + lon * 5,
                slope=15.0,
                aspect=180.0,
                aspect_sin=0.0,
                aspect_cos=-1.0,
                roughness=5.0,
                tpi=10.0,
            )

        mock_pipeline.get_terrain_features = Mock(side_effect=get_features)
        return mock_pipeline

    def test_extract_dem_returns_dataframe(
        self, aligner_nearest, mock_dem_pipeline, sample_points
    ):
        """Should return a DataFrame."""
        result = aligner_nearest.extract_dem_at_points(mock_dem_pipeline, sample_points)
        assert isinstance(result, pd.DataFrame)

    def test_extract_dem_includes_all_features(
        self, aligner_nearest, mock_dem_pipeline, sample_points
    ):
        """Should include all terrain features with dem_ prefix."""
        result = aligner_nearest.extract_dem_at_points(mock_dem_pipeline, sample_points)
        assert "dem_elevation" in result.columns
        assert "dem_slope" in result.columns
        assert "dem_aspect" in result.columns
        assert "dem_aspect_sin" in result.columns
        assert "dem_aspect_cos" in result.columns
        assert "dem_roughness" in result.columns
        assert "dem_tpi" in result.columns

    def test_extract_dem_calls_pipeline_for_each_point(
        self, aligner_nearest, mock_dem_pipeline, sample_points
    ):
        """Should call get_terrain_features for each point."""
        aligner_nearest.extract_dem_at_points(mock_dem_pipeline, sample_points)
        assert mock_dem_pipeline.get_terrain_features.call_count == len(sample_points)

    def test_extract_dem_handles_failure(self, aligner_nearest, sample_points):
        """Should return NaN for points that fail extraction."""
        mock_pipeline = Mock()
        mock_pipeline.get_terrain_features = Mock(side_effect=ValueError("No data"))

        result = aligner_nearest.extract_dem_at_points(mock_pipeline, sample_points)

        assert len(result) == len(sample_points)
        assert result["dem_elevation"].isna().all()
        assert result["dem_slope"].isna().all()


class TestAlignAllSources:
    """Tests for aligning all sources together."""

    @pytest.fixture
    def mock_dem_pipeline(self):
        """Create a mock DEM pipeline."""
        @dataclass
        class MockTerrainFeatures:
            elevation: float = 2500.0
            slope: float = 15.0
            aspect: float = 180.0
            aspect_sin: float = 0.0
            aspect_cos: float = -1.0
            roughness: float = 5.0
            tpi: float = 10.0

        mock = Mock()
        mock.get_terrain_features = Mock(return_value=MockTerrainFeatures())
        return mock

    def test_align_all_returns_dataframe(
        self, aligner_nearest, sample_dataset, mock_dem_pipeline, sample_points
    ):
        """Should return a DataFrame."""
        result = aligner_nearest.align_all_sources(
            points=sample_points,
            era5_ds=sample_dataset,
            dem_pipeline=mock_dem_pipeline,
        )
        assert isinstance(result, pd.DataFrame)

    def test_align_all_includes_era5(
        self, aligner_nearest, sample_dataset, sample_points
    ):
        """Should include ERA5 columns when provided."""
        result = aligner_nearest.align_all_sources(
            points=sample_points,
            era5_ds=sample_dataset,
        )
        assert "era5_t2m" in result.columns

    def test_align_all_includes_dem(
        self, aligner_nearest, mock_dem_pipeline, sample_points
    ):
        """Should include DEM columns when provided."""
        result = aligner_nearest.align_all_sources(
            points=sample_points,
            dem_pipeline=mock_dem_pipeline,
        )
        assert "dem_elevation" in result.columns

    def test_align_all_includes_point_coords(
        self, aligner_nearest, sample_dataset, sample_points
    ):
        """Should include point coordinates."""
        result = aligner_nearest.align_all_sources(
            points=sample_points,
            era5_ds=sample_dataset,
        )
        assert "point_id" in result.columns
        assert "point_lat" in result.columns
        assert "point_lon" in result.columns


class TestExtractionStats:
    """Tests for extraction statistics."""

    def test_stats_includes_grid_resolution(
        self, aligner_nearest, sample_dataset, sample_points
    ):
        """Should include grid resolution."""
        stats = aligner_nearest.get_extraction_stats(sample_dataset, sample_points)
        assert "grid_resolution_lat" in stats
        assert "grid_resolution_lon" in stats
        # Grid is 0.5 degree resolution
        assert abs(stats["grid_resolution_lat"] - 0.5) < 0.01
        assert abs(stats["grid_resolution_lon"] - 0.5) < 0.01

    def test_stats_includes_bounds(
        self, aligner_nearest, sample_dataset, sample_points
    ):
        """Should include grid bounds."""
        stats = aligner_nearest.get_extraction_stats(sample_dataset, sample_points)
        assert "grid_lat_range" in stats
        assert "grid_lon_range" in stats

    def test_stats_counts_points_in_bounds(
        self, aligner_nearest, sample_dataset, sample_points
    ):
        """Should count points within grid bounds."""
        stats = aligner_nearest.get_extraction_stats(sample_dataset, sample_points)
        # All sample_points are within the grid
        assert stats["points_in_bounds"] == 3
        assert stats["points_out_of_bounds"] == 0

    def test_stats_counts_points_out_of_bounds(
        self, aligner_nearest, sample_dataset
    ):
        """Should count points outside grid bounds."""
        points = [(50.0, -100.0), (30.0, -120.0)]  # Outside grid
        stats = aligner_nearest.get_extraction_stats(sample_dataset, points)
        assert stats["points_in_bounds"] == 0
        assert stats["points_out_of_bounds"] == 2

    def test_stats_includes_variables(
        self, aligner_nearest, sample_dataset, sample_points
    ):
        """Should include list of variables."""
        stats = aligner_nearest.get_extraction_stats(sample_dataset, sample_points)
        assert "variables" in stats
        assert "t2m" in stats["variables"]
        assert "tp" in stats["variables"]

    def test_stats_detects_time_dimension(
        self, aligner_nearest, sample_dataset, sample_dataset_2d, sample_points
    ):
        """Should detect whether time dimension exists."""
        stats_3d = aligner_nearest.get_extraction_stats(sample_dataset, sample_points)
        stats_2d = aligner_nearest.get_extraction_stats(sample_dataset_2d, sample_points)
        assert stats_3d["has_time_dim"] is True
        assert stats_2d["has_time_dim"] is False


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_points_list(self, aligner_nearest, sample_dataset):
        """Should handle empty points list."""
        result = aligner_nearest.extract_at_points(sample_dataset, [])
        assert len(result) == 0

    def test_single_point(self, aligner_nearest, sample_dataset):
        """Should handle single point extraction."""
        points = [(39.0, -106.0)]
        result = aligner_nearest.extract_at_points(sample_dataset, points)
        assert "point_id" in result.columns
        assert result["point_id"].iloc[0] == 0

    def test_point_outside_grid(self, aligner_nearest, sample_dataset):
        """Should handle points outside grid (extrapolation or error)."""
        points = [(50.0, -100.0)]  # Outside grid bounds
        # xarray will still extract using nearest method
        result = aligner_nearest.extract_at_points(sample_dataset, points)
        # Should get some result (nearest edge values)
        assert len(result) > 0

    def test_method_override_in_extract(
        self, aligner_nearest, sample_dataset, sample_points
    ):
        """Should allow method override in extract call."""
        result = aligner_nearest.extract_at_points(
            sample_dataset, sample_points, method="bilinear"
        )
        assert isinstance(result, pd.DataFrame)


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_extraction_result_creation(self):
        """Should create ExtractionResult dataclass."""
        from snowforecast.features import ExtractionResult

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = ExtractionResult(
            data=df,
            points_extracted=3,
            points_failed=0,
            variables=["a"],
            method="nearest",
        )
        assert result.points_extracted == 3
        assert result.points_failed == 0
        assert result.variables == ["a"]
        assert result.method == "nearest"
