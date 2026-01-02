"""Tests for DEM pipeline.

These tests use synthetic DEM data to verify terrain calculations without
requiring actual downloads from AWS.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd

from snowforecast.pipelines.dem import (
    COPERNICUS_DEM_URL,
    DEMPipeline,
    TerrainFeatures,
    _tile_name,
    _tile_url,
)


class TestTileNaming:
    """Tests for tile naming functions."""

    def test_tile_name_north_west(self):
        """Test tile name for N/W location."""
        name = _tile_name(40, -112)
        assert name == "Copernicus_DSM_COG_10_N40_00_W112_00_DEM"

    def test_tile_name_south_east(self):
        """Test tile name for S/E location."""
        name = _tile_name(-10, 20)
        assert name == "Copernicus_DSM_COG_10_S10_00_E020_00_DEM"

    def test_tile_name_zero(self):
        """Test tile name for 0,0 location."""
        name = _tile_name(0, 0)
        assert name == "Copernicus_DSM_COG_10_N00_00_E000_00_DEM"

    def test_tile_url(self):
        """Test tile URL construction."""
        url = _tile_url(40, -112)
        assert url.startswith(COPERNICUS_DEM_URL)
        assert "N40_00_W112_00" in url
        assert url.endswith(".tif")


class TestTerrainFeatures:
    """Tests for TerrainFeatures dataclass."""

    def test_terrain_features_creation(self):
        """Test creating TerrainFeatures."""
        features = TerrainFeatures(
            elevation=3000.0,
            slope=25.0,
            aspect=180.0,
            aspect_sin=0.0,
            aspect_cos=-1.0,
            roughness=5.0,
            tpi=10.0,
        )
        assert features.elevation == 3000.0
        assert features.slope == 25.0
        assert features.aspect == 180.0
        assert features.aspect_sin == 0.0
        assert features.aspect_cos == -1.0


class TestSlopeCalculation:
    """Tests for slope calculation."""

    def test_calculate_slope_flat(self):
        """Flat terrain should have zero slope."""
        pipeline = DEMPipeline()
        dem = np.ones((10, 10)) * 1000  # Flat at 1000m
        slope = pipeline.calculate_slope(dem)

        assert slope.shape == (10, 10)
        assert np.allclose(slope, 0, atol=0.01)

    def test_calculate_slope_incline(self):
        """Test slope on a linear incline."""
        pipeline = DEMPipeline()

        # Create a ramp: 30m rise per cell (30m horizontal) = 45 degrees
        dem = np.zeros((10, 10))
        for i in range(10):
            dem[i, :] = i * 30  # 30m rise per row

        slope = pipeline.calculate_slope(dem, resolution=30)

        # Should be ~45 degrees at interior points
        center_slopes = slope[2:8, 2:8]
        assert np.mean(center_slopes) > 40
        assert np.mean(center_slopes) < 50

    def test_calculate_slope_steep(self):
        """Steep terrain should have high slope values."""
        pipeline = DEMPipeline()

        # Create very steep terrain: 60m rise per 30m = ~63 degrees
        dem = np.zeros((10, 10))
        for i in range(10):
            dem[i, :] = i * 60

        slope = pipeline.calculate_slope(dem, resolution=30)

        # Should have slopes > 50 degrees
        center_slopes = slope[2:8, 2:8]
        assert np.mean(center_slopes) > 55


class TestAspectCalculation:
    """Tests for aspect calculation."""

    def test_calculate_aspect_north_facing(self):
        """North-facing slope should have aspect near 0 or 360."""
        pipeline = DEMPipeline()

        # Create slope facing north (elevation increases to south)
        dem = np.zeros((10, 10))
        for i in range(10):
            dem[i, :] = i * 30  # Higher in south (larger row index)

        aspect = pipeline.calculate_aspect(dem, resolution=30)

        # North-facing = aspect near 0 or 360
        center_aspect = aspect[5, 5]
        # Should be close to 0 or 360
        assert center_aspect < 30 or center_aspect > 330

    def test_calculate_aspect_south_facing(self):
        """South-facing slope should have aspect near 180."""
        pipeline = DEMPipeline()

        # Create slope facing south (elevation increases to north)
        dem = np.zeros((10, 10))
        for i in range(10):
            dem[i, :] = (9 - i) * 30  # Higher in north (smaller row index)

        aspect = pipeline.calculate_aspect(dem, resolution=30)

        # South-facing = aspect near 180
        center_aspect = aspect[5, 5]
        assert 150 < center_aspect < 210

    def test_calculate_aspect_east_facing(self):
        """East-facing slope should have aspect near 90."""
        pipeline = DEMPipeline()

        # Create slope facing east (elevation increases to west)
        dem = np.zeros((10, 10))
        for j in range(10):
            dem[:, j] = (9 - j) * 30  # Higher in west (smaller col index)

        aspect = pipeline.calculate_aspect(dem, resolution=30)

        # East-facing = aspect near 90
        center_aspect = aspect[5, 5]
        assert 60 < center_aspect < 120

    def test_calculate_aspect_west_facing(self):
        """West-facing slope should have aspect near 270."""
        pipeline = DEMPipeline()

        # Create slope facing west (elevation increases to east)
        dem = np.zeros((10, 10))
        for j in range(10):
            dem[:, j] = j * 30  # Higher in east (larger col index)

        aspect = pipeline.calculate_aspect(dem, resolution=30)

        # West-facing = aspect near 270
        center_aspect = aspect[5, 5]
        assert 240 < center_aspect < 300

    def test_aspect_range(self):
        """Aspect should be in 0-360 range."""
        pipeline = DEMPipeline()

        # Random terrain
        np.random.seed(42)
        dem = np.cumsum(np.cumsum(np.random.randn(20, 20), axis=0), axis=1) * 10 + 2000

        aspect = pipeline.calculate_aspect(dem)

        assert aspect.min() >= 0
        assert aspect.max() <= 360


class TestAspectCyclicalEncoding:
    """Tests for aspect cyclical encoding."""

    def test_aspect_sin_cos_continuous(self):
        """Sin/cos encoding should be continuous across 0/360 boundary."""
        DEMPipeline()

        # Test values near 0 and 360 should have similar sin/cos
        aspect_0 = 1.0
        aspect_360 = 359.0

        sin_0 = np.sin(np.radians(aspect_0))
        cos_0 = np.cos(np.radians(aspect_0))
        sin_360 = np.sin(np.radians(aspect_360))
        cos_360 = np.cos(np.radians(aspect_360))

        # Should be very similar
        assert abs(sin_0 - sin_360) < 0.04
        assert abs(cos_0 - cos_360) < 0.001

    def test_aspect_encoding_north(self):
        """North aspect (0) should have sin=0, cos=1."""
        sin_val = np.sin(np.radians(0))
        cos_val = np.cos(np.radians(0))
        assert abs(sin_val) < 0.001
        assert abs(cos_val - 1.0) < 0.001

    def test_aspect_encoding_east(self):
        """East aspect (90) should have sin=1, cos=0."""
        sin_val = np.sin(np.radians(90))
        cos_val = np.cos(np.radians(90))
        assert abs(sin_val - 1.0) < 0.001
        assert abs(cos_val) < 0.001

    def test_aspect_encoding_south(self):
        """South aspect (180) should have sin=0, cos=-1."""
        sin_val = np.sin(np.radians(180))
        cos_val = np.cos(np.radians(180))
        assert abs(sin_val) < 0.001
        assert abs(cos_val + 1.0) < 0.001


class TestRoughnessCalculation:
    """Tests for terrain roughness calculation."""

    def test_roughness_flat(self):
        """Flat terrain should have very low roughness."""
        pipeline = DEMPipeline()
        dem = np.ones((20, 20)) * 2000  # Flat at 2000m
        roughness = pipeline.calculate_roughness(dem)

        assert roughness.shape == (20, 20)
        # Roughness should be near zero for flat terrain
        assert np.mean(roughness) < 0.01

    def test_roughness_varied(self):
        """Varied terrain should have higher roughness."""
        pipeline = DEMPipeline()

        # Create checkerboard pattern (high variability)
        dem = np.zeros((20, 20))
        dem[::2, ::2] = 100
        dem[1::2, 1::2] = 100

        roughness = pipeline.calculate_roughness(dem)

        # Should have non-zero roughness
        assert np.mean(roughness) > 10

    def test_roughness_gradient(self):
        """Smooth gradient should have lower roughness than noisy terrain."""
        pipeline = DEMPipeline()

        # Smooth gradient
        smooth = np.zeros((20, 20))
        for i in range(20):
            smooth[i, :] = i * 10

        # Noisy terrain
        np.random.seed(42)
        noisy = smooth + np.random.randn(20, 20) * 20

        roughness_smooth = pipeline.calculate_roughness(smooth)
        roughness_noisy = pipeline.calculate_roughness(noisy)

        # Noisy terrain should be rougher
        assert np.mean(roughness_noisy) > np.mean(roughness_smooth)


class TestTPICalculation:
    """Tests for Topographic Position Index calculation."""

    def test_tpi_ridge_positive(self):
        """Ridge (higher than surroundings) should have positive TPI."""
        pipeline = DEMPipeline()

        # Create a ridge in the center
        dem = np.ones((21, 21)) * 2000
        dem[10, 10] = 2100  # Ridge at center

        tpi = pipeline.calculate_tpi(dem, radius=5)

        # Center should have positive TPI
        assert tpi[10, 10] > 0

    def test_tpi_valley_negative(self):
        """Valley (lower than surroundings) should have negative TPI."""
        pipeline = DEMPipeline()

        # Create a valley in the center
        dem = np.ones((21, 21)) * 2000
        dem[10, 10] = 1900  # Valley at center

        tpi = pipeline.calculate_tpi(dem, radius=5)

        # Center should have negative TPI
        assert tpi[10, 10] < 0

    def test_tpi_flat(self):
        """Flat terrain should have TPI near zero."""
        pipeline = DEMPipeline()
        dem = np.ones((21, 21)) * 2000

        tpi = pipeline.calculate_tpi(dem, radius=5)

        # All TPI values should be near zero
        assert np.allclose(tpi[5:-5, 5:-5], 0, atol=0.01)

    def test_tpi_ridge_vs_valley(self):
        """Ridge TPI should be greater than valley TPI."""
        pipeline = DEMPipeline()

        # Ridge
        ridge_dem = np.ones((21, 21)) * 2000
        ridge_dem[10, 10] = 2100

        # Valley
        valley_dem = np.ones((21, 21)) * 2000
        valley_dem[10, 10] = 1900

        ridge_tpi = pipeline.calculate_tpi(ridge_dem, radius=5)
        valley_tpi = pipeline.calculate_tpi(valley_dem, radius=5)

        assert ridge_tpi[10, 10] > valley_tpi[10, 10]


class TestValidation:
    """Tests for data validation."""

    def test_validate_empty_dataframe(self):
        """Empty DataFrame should be invalid."""
        pipeline = DEMPipeline()
        df = pd.DataFrame()

        result = pipeline.validate(df)

        assert not result.valid
        assert "empty" in result.issues[0].lower()

    def test_validate_non_dataframe(self):
        """Non-DataFrame input should be invalid."""
        pipeline = DEMPipeline()

        result = pipeline.validate("not a dataframe")

        assert not result.valid
        assert "not a DataFrame" in result.issues[0]

    def test_validate_terrain_features(self):
        """Valid terrain features should pass validation."""
        pipeline = DEMPipeline()
        df = pd.DataFrame({
            "station_id": ["A", "B", "C"],
            "lat": [40.0, 41.0, 42.0],
            "lon": [-105.0, -106.0, -107.0],
            "elevation": [3000.0, 3100.0, 3200.0],
            "slope": [20.0, 25.0, 30.0],
            "aspect": [180.0, 90.0, 270.0],
            "roughness": [5.0, 6.0, 7.0],
            "tpi": [10.0, -5.0, 0.0],
        })

        result = pipeline.validate(df)

        assert result.valid
        assert result.total_rows == 3
        assert result.missing_pct == 0.0

    def test_validate_with_outliers(self):
        """Should detect elevation outliers."""
        pipeline = DEMPipeline()
        df = pd.DataFrame({
            "elevation": [3000.0, 10000.0, -1000.0],  # Two outliers
            "slope": [20.0, 25.0, 30.0],
            "aspect": [180.0, 90.0, 270.0],
            "roughness": [5.0, 6.0, 7.0],
            "tpi": [10.0, -5.0, 0.0],
        })

        result = pipeline.validate(df)

        assert result.outliers_count >= 2
        assert "outlier" in result.issues[0].lower()

    def test_validate_tile_metadata(self):
        """Tile metadata should pass validation."""
        pipeline = DEMPipeline()
        df = pd.DataFrame({
            "tile_name": ["Tile1", "Tile2"],
            "path": ["/data/tile1.tif", "/data/tile2.tif"],
            "west": [-106.0, -107.0],
            "south": [39.0, 40.0],
            "east": [-105.0, -106.0],
            "north": [40.0, 41.0],
        })

        result = pipeline.validate(df)

        assert result.valid
        assert result.stats.get("type") == "tile_metadata"

    def test_validate_with_missing_values(self):
        """Should calculate missing percentage correctly."""
        pipeline = DEMPipeline()
        df = pd.DataFrame({
            "elevation": [3000.0, np.nan, 3200.0],
            "slope": [20.0, 25.0, np.nan],
            "aspect": [180.0, 90.0, 270.0],
            "roughness": [5.0, 6.0, 7.0],
            "tpi": [10.0, -5.0, 0.0],
        })

        result = pipeline.validate(df)

        # 2 missing out of 15 values = 13.33%
        assert 10 < result.missing_pct < 20


class TestGetFeaturesForStations:
    """Tests for get_features_for_stations method."""

    def test_get_features_returns_dataframe(self):
        """Should return DataFrame with all required columns."""
        pipeline = DEMPipeline()

        # Mock the get_terrain_features method
        with patch.object(pipeline, "get_terrain_features") as mock_features:
            mock_features.return_value = TerrainFeatures(
                elevation=3000.0,
                slope=25.0,
                aspect=180.0,
                aspect_sin=0.0,
                aspect_cos=-1.0,
                roughness=5.0,
                tpi=10.0,
            )

            stations = [
                {"station_id": "A", "lat": 40.0, "lon": -105.0},
                {"station_id": "B", "lat": 41.0, "lon": -106.0},
            ]

            df = pipeline.get_features_for_stations(stations)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "station_id" in df.columns
            assert "elevation" in df.columns
            assert "slope" in df.columns
            assert "aspect" in df.columns
            assert "aspect_sin" in df.columns
            assert "aspect_cos" in df.columns
            assert "roughness" in df.columns
            assert "tpi" in df.columns

    def test_get_features_handles_missing_data(self):
        """Should handle locations with no DEM data gracefully."""
        pipeline = DEMPipeline()

        # Mock to raise ValueError for missing data
        with patch.object(pipeline, "get_terrain_features") as mock_features:
            mock_features.side_effect = ValueError("No DEM data")

            stations = [{"station_id": "A", "lat": 40.0, "lon": -105.0}]

            df = pipeline.get_features_for_stations(stations)

            assert len(df) == 1
            assert df.iloc[0]["station_id"] == "A"
            assert pd.isna(df.iloc[0]["elevation"])


class TestPipelineIntegration:
    """Integration tests for DEMPipeline."""

    def test_pipeline_inherits_from_static_pipeline(self):
        """DEMPipeline should inherit from StaticPipeline."""
        from snowforecast.utils import StaticPipeline

        pipeline = DEMPipeline()
        assert isinstance(pipeline, StaticPipeline)

    def test_pipeline_has_required_methods(self):
        """Pipeline should have all required methods."""
        pipeline = DEMPipeline()

        assert hasattr(pipeline, "download")
        assert hasattr(pipeline, "download_tile")
        assert hasattr(pipeline, "download_region")
        assert hasattr(pipeline, "process")
        assert hasattr(pipeline, "validate")
        assert hasattr(pipeline, "get_elevation")
        assert hasattr(pipeline, "calculate_slope")
        assert hasattr(pipeline, "calculate_aspect")
        assert hasattr(pipeline, "calculate_roughness")
        assert hasattr(pipeline, "calculate_tpi")
        assert hasattr(pipeline, "get_terrain_features")
        assert hasattr(pipeline, "get_features_for_stations")

    def test_pipeline_default_cell_size(self):
        """Pipeline should have default 30m cell size."""
        pipeline = DEMPipeline()
        assert pipeline.cell_size == 30.0

    def test_pipeline_custom_cell_size(self):
        """Pipeline should accept custom cell size."""
        pipeline = DEMPipeline(cell_size=90.0)
        assert pipeline.cell_size == 90.0

    def test_pipeline_paths_exist(self):
        """Pipeline should create data paths."""
        pipeline = DEMPipeline()

        assert pipeline.raw_path.exists()
        assert pipeline.processed_path.exists()
        assert "dem" in str(pipeline.raw_path)
        assert "dem" in str(pipeline.processed_path)


class TestDownloadRegion:
    """Tests for download_region method."""

    def test_download_region_calculates_correct_tiles(self):
        """Should identify correct tiles for a region."""
        pipeline = DEMPipeline()

        # Track which tiles would be downloaded
        downloaded_tiles = []

        def mock_download(lat, lon, force=False):
            downloaded_tiles.append((lat, lon))
            raise RuntimeError("Mock - no actual download")

        with patch.object(pipeline, "download_tile", side_effect=mock_download):
            bbox = {
                "west": -106.5,
                "south": 39.5,
                "east": -105.5,
                "north": 40.5,
            }
            pipeline.download_region(bbox)

        # Should try to download tiles for lat 39, 40 and lon -107, -106
        expected_tiles = {(39, -107), (39, -106), (40, -107), (40, -106)}
        assert set(downloaded_tiles) == expected_tiles
