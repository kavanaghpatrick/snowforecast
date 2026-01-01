"""Tests for terrain feature engineering.

This module tests the TerrainFeatureEngineer class and all helper functions
for computing derived terrain features.
"""

import math
import numpy as np
import pandas as pd
import pytest

from snowforecast.features.terrain import (
    TerrainFeatureEngineer,
    _distance_to_coast,
    _get_elevation_band,
    _get_slope_category,
    _get_aspect_cardinal,
    _is_north_facing,
    _compute_solar_exposure,
    _compute_wind_exposure,
    ELEVATION_BANDS,
    SLOPE_CATEGORIES,
    WESTERN_US_LAT_MIN,
    WESTERN_US_LAT_MAX,
)


# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture
def sample_locations():
    """Sample locations with basic lat/lon data."""
    return pd.DataFrame({
        "lat": [40.0, 45.0, 35.0],
        "lon": [-111.0, -121.0, -118.0],
    })


@pytest.fixture
def sample_terrain_data():
    """Sample data with base terrain features."""
    return pd.DataFrame({
        "lat": [40.0, 45.0, 35.0, 38.0],
        "lon": [-111.0, -121.0, -118.0, -119.5],
        "elevation": [2500.0, 1500.0, 3200.0, 1800.0],
        "slope": [15.0, 5.0, 35.0, 50.0],
        "aspect": [0.0, 90.0, 180.0, 270.0],
        "tpi": [10.0, -20.0, 0.0, 50.0],
        "roughness": [5.0, 2.0, 15.0, 8.0],
    })


@pytest.fixture
def terrain_engineer():
    """TerrainFeatureEngineer instance without DEM pipeline."""
    return TerrainFeatureEngineer()


# =============================================================================
# Test elevation band helper
# =============================================================================


class TestGetElevationBand:
    """Tests for _get_elevation_band function."""

    def test_low_elevation(self):
        """Elevation below 2000m should be 'low'."""
        assert _get_elevation_band(500) == "low"
        assert _get_elevation_band(1500) == "low"
        assert _get_elevation_band(1999) == "low"

    def test_mid_elevation(self):
        """Elevation 2000-2500m should be 'mid'."""
        assert _get_elevation_band(2000) == "mid"
        assert _get_elevation_band(2250) == "mid"
        assert _get_elevation_band(2499) == "mid"

    def test_high_elevation(self):
        """Elevation 2500-3000m should be 'high'."""
        assert _get_elevation_band(2500) == "high"
        assert _get_elevation_band(2750) == "high"
        assert _get_elevation_band(2999) == "high"

    def test_alpine_elevation(self):
        """Elevation above 3000m should be 'alpine'."""
        assert _get_elevation_band(3000) == "alpine"
        assert _get_elevation_band(3500) == "alpine"
        assert _get_elevation_band(4500) == "alpine"

    def test_nan_elevation(self):
        """NaN elevation should return 'unknown'."""
        assert _get_elevation_band(np.nan) == "unknown"


# =============================================================================
# Test slope category helper
# =============================================================================


class TestGetSlopeCategory:
    """Tests for _get_slope_category function."""

    def test_flat_slope(self):
        """Slope 0-5 degrees should be 'flat'."""
        assert _get_slope_category(0) == "flat"
        assert _get_slope_category(3) == "flat"
        assert _get_slope_category(4.9) == "flat"

    def test_gentle_slope(self):
        """Slope 5-15 degrees should be 'gentle'."""
        assert _get_slope_category(5) == "gentle"
        assert _get_slope_category(10) == "gentle"
        assert _get_slope_category(14.9) == "gentle"

    def test_moderate_slope(self):
        """Slope 15-30 degrees should be 'moderate'."""
        assert _get_slope_category(15) == "moderate"
        assert _get_slope_category(22) == "moderate"
        assert _get_slope_category(29.9) == "moderate"

    def test_steep_slope(self):
        """Slope 30-45 degrees should be 'steep'."""
        assert _get_slope_category(30) == "steep"
        assert _get_slope_category(37) == "steep"
        assert _get_slope_category(44.9) == "steep"

    def test_extreme_slope(self):
        """Slope above 45 degrees should be 'extreme'."""
        assert _get_slope_category(45) == "extreme"
        assert _get_slope_category(60) == "extreme"
        assert _get_slope_category(90) == "extreme"

    def test_nan_slope(self):
        """NaN slope should return 'unknown'."""
        assert _get_slope_category(np.nan) == "unknown"


# =============================================================================
# Test aspect cardinal helper
# =============================================================================


class TestGetAspectCardinal:
    """Tests for _get_aspect_cardinal function."""

    def test_north(self):
        """Aspect 337.5-22.5 should be 'N'."""
        assert _get_aspect_cardinal(0) == "N"
        assert _get_aspect_cardinal(10) == "N"
        assert _get_aspect_cardinal(350) == "N"
        assert _get_aspect_cardinal(360) == "N"

    def test_northeast(self):
        """Aspect 22.5-67.5 should be 'NE'."""
        assert _get_aspect_cardinal(45) == "NE"
        assert _get_aspect_cardinal(30) == "NE"
        assert _get_aspect_cardinal(60) == "NE"

    def test_east(self):
        """Aspect 67.5-112.5 should be 'E'."""
        assert _get_aspect_cardinal(90) == "E"
        assert _get_aspect_cardinal(70) == "E"
        assert _get_aspect_cardinal(110) == "E"

    def test_southeast(self):
        """Aspect 112.5-157.5 should be 'SE'."""
        assert _get_aspect_cardinal(135) == "SE"

    def test_south(self):
        """Aspect 157.5-202.5 should be 'S'."""
        assert _get_aspect_cardinal(180) == "S"

    def test_southwest(self):
        """Aspect 202.5-247.5 should be 'SW'."""
        assert _get_aspect_cardinal(225) == "SW"

    def test_west(self):
        """Aspect 247.5-292.5 should be 'W'."""
        assert _get_aspect_cardinal(270) == "W"

    def test_northwest(self):
        """Aspect 292.5-337.5 should be 'NW'."""
        assert _get_aspect_cardinal(315) == "NW"

    def test_nan_aspect(self):
        """NaN aspect should return 'unknown'."""
        assert _get_aspect_cardinal(np.nan) == "unknown"


# =============================================================================
# Test north-facing helper
# =============================================================================


class TestIsNorthFacing:
    """Tests for _is_north_facing function."""

    def test_north_facing_true(self):
        """Aspect 315-45 should be north-facing."""
        assert _is_north_facing(0) == 1
        assert _is_north_facing(10) == 1
        assert _is_north_facing(45) == 1
        assert _is_north_facing(315) == 1
        assert _is_north_facing(350) == 1

    def test_north_facing_false(self):
        """Aspect 46-314 should not be north-facing."""
        assert _is_north_facing(46) == 0
        assert _is_north_facing(90) == 0
        assert _is_north_facing(180) == 0
        assert _is_north_facing(270) == 0
        assert _is_north_facing(314) == 0

    def test_nan_aspect(self):
        """NaN aspect should return 0."""
        assert _is_north_facing(np.nan) == 0


# =============================================================================
# Test solar exposure helper
# =============================================================================


class TestComputeSolarExposure:
    """Tests for _compute_solar_exposure function."""

    def test_south_facing_steep(self):
        """South-facing steep slope should have high solar exposure."""
        exposure = _compute_solar_exposure(180, 45)
        assert exposure > 0.8

    def test_north_facing_steep(self):
        """North-facing steep slope should have low solar exposure."""
        exposure = _compute_solar_exposure(0, 45)
        assert exposure < 0.2

    def test_flat_terrain(self):
        """Flat terrain should have neutral solar exposure regardless of aspect."""
        exposure_north = _compute_solar_exposure(0, 0)
        exposure_south = _compute_solar_exposure(180, 0)
        assert abs(exposure_north - 0.5) < 0.1
        assert abs(exposure_south - 0.5) < 0.1

    def test_east_facing(self):
        """East-facing slope should have moderate solar exposure."""
        exposure = _compute_solar_exposure(90, 30)
        assert 0.3 < exposure < 0.7

    def test_nan_values(self):
        """NaN input should return neutral exposure."""
        assert _compute_solar_exposure(np.nan, 30) == 0.5
        assert _compute_solar_exposure(180, np.nan) == 0.5


# =============================================================================
# Test wind exposure helper
# =============================================================================


class TestComputeWindExposure:
    """Tests for _compute_wind_exposure function."""

    def test_ridge_high_exposure(self):
        """Positive TPI (ridge) should have high wind exposure."""
        exposure = _compute_wind_exposure(50)
        assert exposure > 0.7

    def test_valley_low_exposure(self):
        """Negative TPI (valley) should have low wind exposure."""
        exposure = _compute_wind_exposure(-50)
        assert exposure < 0.3

    def test_neutral_terrain(self):
        """Zero TPI should have neutral wind exposure."""
        exposure = _compute_wind_exposure(0)
        assert abs(exposure - 0.5) < 0.01

    def test_clamping(self):
        """Extreme TPI values should be clamped to 0-1 range."""
        assert _compute_wind_exposure(100) == 1.0
        assert _compute_wind_exposure(-100) == 0.0

    def test_nan_value(self):
        """NaN TPI should return neutral exposure."""
        assert _compute_wind_exposure(np.nan) == 0.5


# =============================================================================
# Test distance to coast
# =============================================================================


class TestDistanceToCoast:
    """Tests for _distance_to_coast function."""

    def test_coastal_location(self):
        """Location near coast should have small distance."""
        # San Francisco area
        dist = _distance_to_coast(37.7, -122.4)
        assert dist < 100  # Less than 100 km from coast

    def test_inland_location(self):
        """Inland location should have larger distance."""
        # Salt Lake City area
        dist = _distance_to_coast(40.7, -111.9)
        assert dist > 500  # More than 500 km from coast

    def test_distance_increases_inland(self):
        """Distance should increase as we move inland."""
        dist_coast = _distance_to_coast(37.7, -122.4)  # Near coast
        dist_central = _distance_to_coast(37.7, -120.0)  # Central Valley
        dist_mountain = _distance_to_coast(37.7, -118.0)  # Sierra Nevada
        assert dist_coast < dist_central < dist_mountain

    def test_positive_distance(self):
        """Distance should always be positive."""
        dist = _distance_to_coast(40.0, -111.0)
        assert dist > 0


# =============================================================================
# Test TerrainFeatureEngineer methods
# =============================================================================


class TestTerrainFeatureEngineerInit:
    """Tests for TerrainFeatureEngineer initialization."""

    def test_init_without_pipeline(self):
        """Should initialize without DEM pipeline."""
        engineer = TerrainFeatureEngineer()
        assert engineer.dem_pipeline is None

    def test_init_with_pipeline_object(self):
        """Should accept a DEM pipeline instance (or any object)."""
        # Create a simple mock object without pytest-mock
        class MockPipeline:
            pass
        mock_pipeline = MockPipeline()
        engineer = TerrainFeatureEngineer(dem_pipeline=mock_pipeline)
        assert engineer.dem_pipeline is mock_pipeline

    def test_get_base_terrain_without_pipeline(self):
        """Should raise ValueError when no pipeline configured."""
        engineer = TerrainFeatureEngineer()
        with pytest.raises(ValueError, match="No DEM pipeline configured"):
            engineer.get_base_terrain(40.0, -111.0)


class TestComputeElevationFeatures:
    """Tests for compute_elevation_features method."""

    def test_elevation_m_column(self, terrain_engineer, sample_terrain_data):
        """Should preserve elevation_m if present."""
        df = sample_terrain_data.copy()
        df["elevation_m"] = df["elevation"]
        result = terrain_engineer.compute_elevation_features(df)
        assert "elevation_m" in result.columns

    def test_elevation_km_calculation(self, terrain_engineer, sample_terrain_data):
        """Should correctly compute elevation in kilometers."""
        result = terrain_engineer.compute_elevation_features(sample_terrain_data)
        expected_km = sample_terrain_data["elevation"] / 1000.0
        np.testing.assert_array_almost_equal(result["elevation_km"], expected_km)

    def test_elevation_band_assignment(self, terrain_engineer, sample_terrain_data):
        """Should correctly assign elevation bands."""
        result = terrain_engineer.compute_elevation_features(sample_terrain_data)
        assert result["elevation_band"].iloc[0] == "high"  # 2500m
        assert result["elevation_band"].iloc[1] == "low"   # 1500m
        assert result["elevation_band"].iloc[2] == "alpine"  # 3200m
        assert result["elevation_band"].iloc[3] == "low"   # 1800m

    def test_missing_elevation_column(self, terrain_engineer):
        """Should raise ValueError if no elevation column."""
        df = pd.DataFrame({"lat": [40.0], "lon": [-111.0]})
        with pytest.raises(ValueError, match="must contain 'elevation'"):
            terrain_engineer.compute_elevation_features(df)


class TestComputeSlopeFeatures:
    """Tests for compute_slope_features method."""

    def test_slope_deg_column(self, terrain_engineer, sample_terrain_data):
        """Should create slope_deg from slope column."""
        result = terrain_engineer.compute_slope_features(sample_terrain_data)
        np.testing.assert_array_equal(result["slope_deg"], sample_terrain_data["slope"])

    def test_slope_category_assignment(self, terrain_engineer, sample_terrain_data):
        """Should correctly assign slope categories."""
        result = terrain_engineer.compute_slope_features(sample_terrain_data)
        assert result["slope_category"].iloc[0] == "moderate"  # 15 degrees
        assert result["slope_category"].iloc[1] == "gentle"    # 5 degrees
        assert result["slope_category"].iloc[2] == "steep"     # 35 degrees
        assert result["slope_category"].iloc[3] == "extreme"   # 50 degrees

    def test_missing_slope_column(self, terrain_engineer):
        """Should raise ValueError if no slope column."""
        df = pd.DataFrame({"lat": [40.0], "lon": [-111.0]})
        with pytest.raises(ValueError, match="must contain 'slope'"):
            terrain_engineer.compute_slope_features(df)


class TestComputeAspectFeatures:
    """Tests for compute_aspect_features method."""

    def test_aspect_deg_column(self, terrain_engineer, sample_terrain_data):
        """Should create aspect_deg from aspect column."""
        result = terrain_engineer.compute_aspect_features(sample_terrain_data)
        np.testing.assert_array_equal(result["aspect_deg"], sample_terrain_data["aspect"])

    def test_aspect_sin_cos_encoding(self, terrain_engineer, sample_terrain_data):
        """Should correctly compute sin/cos encoding of aspect."""
        result = terrain_engineer.compute_aspect_features(sample_terrain_data)

        # North (0 degrees): sin=0, cos=1
        assert abs(result["aspect_sin"].iloc[0]) < 0.01
        assert abs(result["aspect_cos"].iloc[0] - 1.0) < 0.01

        # East (90 degrees): sin=1, cos=0
        assert abs(result["aspect_sin"].iloc[1] - 1.0) < 0.01
        assert abs(result["aspect_cos"].iloc[1]) < 0.01

        # South (180 degrees): sin=0, cos=-1
        assert abs(result["aspect_sin"].iloc[2]) < 0.01
        assert abs(result["aspect_cos"].iloc[2] + 1.0) < 0.01

    def test_aspect_cardinal_assignment(self, terrain_engineer, sample_terrain_data):
        """Should correctly assign cardinal directions."""
        result = terrain_engineer.compute_aspect_features(sample_terrain_data)
        assert result["aspect_cardinal"].iloc[0] == "N"   # 0 degrees
        assert result["aspect_cardinal"].iloc[1] == "E"   # 90 degrees
        assert result["aspect_cardinal"].iloc[2] == "S"   # 180 degrees
        assert result["aspect_cardinal"].iloc[3] == "W"   # 270 degrees

    def test_north_facing_assignment(self, terrain_engineer, sample_terrain_data):
        """Should correctly identify north-facing slopes."""
        result = terrain_engineer.compute_aspect_features(sample_terrain_data)
        assert result["north_facing"].iloc[0] == 1  # 0 degrees (N)
        assert result["north_facing"].iloc[1] == 0  # 90 degrees (E)
        assert result["north_facing"].iloc[2] == 0  # 180 degrees (S)
        assert result["north_facing"].iloc[3] == 0  # 270 degrees (W)


class TestComputeExposureFeatures:
    """Tests for compute_exposure_features method."""

    def test_wind_exposure_from_tpi(self, terrain_engineer, sample_terrain_data):
        """Should compute wind exposure from TPI."""
        result = terrain_engineer.compute_exposure_features(sample_terrain_data)
        # TPI 10 -> exposure > 0.5
        assert result["wind_exposure"].iloc[0] > 0.5
        # TPI -20 -> exposure < 0.5
        assert result["wind_exposure"].iloc[1] < 0.5
        # TPI 0 -> exposure = 0.5
        assert abs(result["wind_exposure"].iloc[2] - 0.5) < 0.01

    def test_solar_exposure_calculation(self, terrain_engineer, sample_terrain_data):
        """Should compute solar exposure from aspect and slope."""
        result = terrain_engineer.compute_exposure_features(sample_terrain_data)
        # South-facing (180 deg) steep slope -> high exposure
        assert result["solar_exposure"].iloc[2] > 0.5

    def test_terrain_roughness_passthrough(self, terrain_engineer, sample_terrain_data):
        """Should pass through roughness as terrain_roughness."""
        result = terrain_engineer.compute_exposure_features(sample_terrain_data)
        np.testing.assert_array_equal(
            result["terrain_roughness"], sample_terrain_data["roughness"]
        )


class TestComputeGeographicFeatures:
    """Tests for compute_geographic_features method."""

    def test_distance_to_coast(self, terrain_engineer, sample_terrain_data):
        """Should compute distance to Pacific coast."""
        result = terrain_engineer.compute_geographic_features(sample_terrain_data)
        assert "distance_to_coast_km" in result.columns
        # All locations should have positive distance
        assert (result["distance_to_coast_km"] > 0).all()

    def test_latitude_normalized(self, terrain_engineer, sample_terrain_data):
        """Should normalize latitude to 0-1 for Western US range."""
        result = terrain_engineer.compute_geographic_features(sample_terrain_data)
        # Latitude 35 (near southern boundary) should be low
        assert result["latitude_normalized"].iloc[2] < 0.3
        # Latitude 45 (near northern boundary) should be high
        assert result["latitude_normalized"].iloc[1] > 0.7
        # All values should be in 0-1 range
        assert (result["latitude_normalized"] >= 0).all()
        assert (result["latitude_normalized"] <= 1).all()

    def test_missing_lat_lon(self, terrain_engineer):
        """Should raise ValueError if lat/lon missing."""
        df = pd.DataFrame({"elevation": [2500.0]})
        with pytest.raises(ValueError, match="must contain 'lat' and 'lon'"):
            terrain_engineer.compute_geographic_features(df)


class TestComputeAll:
    """Tests for compute_all method."""

    def test_computes_all_features(self, terrain_engineer, sample_terrain_data):
        """Should compute all feature categories."""
        result = terrain_engineer.compute_all(sample_terrain_data)

        # Check elevation features
        assert "elevation_m" in result.columns
        assert "elevation_km" in result.columns
        assert "elevation_band" in result.columns

        # Check slope features
        assert "slope_deg" in result.columns
        assert "slope_category" in result.columns

        # Check aspect features
        assert "aspect_deg" in result.columns
        assert "aspect_sin" in result.columns
        assert "aspect_cos" in result.columns
        assert "aspect_cardinal" in result.columns
        assert "north_facing" in result.columns

        # Check exposure features
        assert "wind_exposure" in result.columns
        assert "solar_exposure" in result.columns
        assert "terrain_roughness" in result.columns

        # Check geographic features
        assert "distance_to_coast_km" in result.columns
        assert "latitude_normalized" in result.columns

    def test_preserves_original_columns(self, terrain_engineer, sample_terrain_data):
        """Should preserve original DataFrame columns."""
        result = terrain_engineer.compute_all(sample_terrain_data)
        assert "lat" in result.columns
        assert "lon" in result.columns

    def test_same_row_count(self, terrain_engineer, sample_terrain_data):
        """Should preserve row count."""
        result = terrain_engineer.compute_all(sample_terrain_data)
        assert len(result) == len(sample_terrain_data)


class TestGetFeatureNames:
    """Tests for get_feature_names method."""

    def test_returns_list(self, terrain_engineer):
        """Should return a list of strings."""
        names = terrain_engineer.get_feature_names()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_contains_all_features(self, terrain_engineer):
        """Should contain all expected feature names."""
        names = terrain_engineer.get_feature_names()
        expected = [
            "elevation_m", "elevation_km", "elevation_band",
            "slope_deg", "slope_category",
            "aspect_deg", "aspect_sin", "aspect_cos", "aspect_cardinal", "north_facing",
            "wind_exposure", "solar_exposure", "terrain_roughness",
            "distance_to_coast_km", "latitude_normalized",
        ]
        for feature in expected:
            assert feature in names, f"Missing feature: {feature}"
