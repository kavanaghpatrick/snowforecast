"""Tests for forecast overlay component.

Tests the HexagonLayer overlay functions from
snowforecast.dashboard.components.forecast_overlay.
"""

import pytest
import pandas as pd
import numpy as np
import pydeck as pdk

from snowforecast.dashboard.components.forecast_overlay import (
    create_forecast_overlay,
    generate_grid_points,
    render_overlay_toggle,
    OVERLAY_COLOR_RANGE,
)
from snowforecast.visualization import SNOW_DEPTH_SCALE


@pytest.fixture
def sample_forecast_points() -> pd.DataFrame:
    """Create sample forecast points for testing."""
    return pd.DataFrame({
        'lat': [39.64, 40.65, 39.48, 40.58, 37.63],
        'lon': [-106.37, -111.51, -106.04, -111.65, -119.03],
        'new_snow_cm': [15, 8, 25, 40, 12],
    })


@pytest.fixture
def large_forecast_grid() -> pd.DataFrame:
    """Create a larger grid of forecast points."""
    grid = generate_grid_points(
        lat_min=37.0,
        lat_max=40.0,
        lon_min=-111.0,
        lon_max=-106.0,
        resolution=0.5,
    )
    # Add random snowfall predictions
    np.random.seed(42)
    grid['new_snow_cm'] = np.random.uniform(0, 50, len(grid))
    return grid


class TestCreateForecastOverlay:
    """Tests for create_forecast_overlay function."""

    def test_returns_pydeck_layer(self, sample_forecast_points):
        """Should return a PyDeck Layer object."""
        layer = create_forecast_overlay(sample_forecast_points)
        assert isinstance(layer, pdk.Layer)

    def test_layer_type_is_hexagon(self, sample_forecast_points):
        """Should create a HexagonLayer."""
        layer = create_forecast_overlay(sample_forecast_points)
        assert layer.type == "HexagonLayer"

    def test_layer_is_pickable(self, sample_forecast_points):
        """Layer should be pickable for tooltip interaction."""
        layer = create_forecast_overlay(sample_forecast_points)
        assert layer.pickable is True

    def test_default_radius_is_5000(self, sample_forecast_points):
        """Default radius should be 5000 meters (5km)."""
        layer = create_forecast_overlay(sample_forecast_points)
        assert layer.radius == 5000

    def test_custom_radius(self, sample_forecast_points):
        """Should accept custom radius."""
        layer = create_forecast_overlay(sample_forecast_points, radius=10000)
        assert layer.radius == 10000

    def test_default_opacity_is_half(self, sample_forecast_points):
        """Default opacity should be 0.5."""
        layer = create_forecast_overlay(sample_forecast_points)
        assert layer.opacity == 0.5

    def test_custom_opacity(self, sample_forecast_points):
        """Should accept custom opacity."""
        layer = create_forecast_overlay(sample_forecast_points, opacity=0.7)
        assert layer.opacity == 0.7

    def test_default_not_extruded(self, sample_forecast_points):
        """Default should not be extruded (2D)."""
        layer = create_forecast_overlay(sample_forecast_points)
        assert layer.extruded is False

    def test_can_enable_extrusion(self, sample_forecast_points):
        """Should be able to enable 3D extrusion."""
        layer = create_forecast_overlay(sample_forecast_points, extruded=True)
        assert layer.extruded is True

    def test_uses_correct_color_range(self, sample_forecast_points):
        """Should use OVERLAY_COLOR_RANGE for colors."""
        layer = create_forecast_overlay(sample_forecast_points)
        assert layer.color_range == OVERLAY_COLOR_RANGE

    def test_handles_empty_dataframe(self):
        """Should handle empty DataFrame without error."""
        empty_data = pd.DataFrame({
            'lat': [],
            'lon': [],
            'new_snow_cm': [],
        })
        layer = create_forecast_overlay(empty_data)
        assert layer is not None

    def test_handles_large_grid(self, large_forecast_grid):
        """Should handle larger forecast grids efficiently."""
        layer = create_forecast_overlay(large_forecast_grid)
        assert layer is not None
        assert isinstance(layer, pdk.Layer)


class TestGenerateGridPoints:
    """Tests for generate_grid_points function."""

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        grid = generate_grid_points(
            lat_min=39.0,
            lat_max=40.0,
            lon_min=-106.0,
            lon_max=-105.0,
        )
        assert isinstance(grid, pd.DataFrame)

    def test_has_lat_lon_columns(self):
        """Should have lat and lon columns."""
        grid = generate_grid_points(
            lat_min=39.0,
            lat_max=40.0,
            lon_min=-106.0,
            lon_max=-105.0,
        )
        assert 'lat' in grid.columns
        assert 'lon' in grid.columns

    def test_correct_number_of_points(self):
        """Should generate correct number of grid points."""
        grid = generate_grid_points(
            lat_min=39.0,
            lat_max=40.0,
            lon_min=-106.0,
            lon_max=-105.0,
            resolution=0.5,
        )
        # 3 lat points (39.0, 39.5, 40.0) x 3 lon points
        assert len(grid) == 9

    def test_default_resolution(self):
        """Default resolution should be 0.1 degrees."""
        grid = generate_grid_points(
            lat_min=39.0,
            lat_max=40.0,
            lon_min=-106.0,
            lon_max=-105.0,
        )
        # 11 lat points x 11 lon points with 0.1 resolution
        assert len(grid) == 121

    def test_lat_values_in_range(self):
        """Latitude values should be within specified range."""
        grid = generate_grid_points(
            lat_min=39.0,
            lat_max=40.0,
            lon_min=-106.0,
            lon_max=-105.0,
        )
        assert grid['lat'].min() >= 39.0
        assert grid['lat'].max() <= 40.0

    def test_lon_values_in_range(self):
        """Longitude values should be within specified range."""
        grid = generate_grid_points(
            lat_min=39.0,
            lat_max=40.0,
            lon_min=-106.0,
            lon_max=-105.0,
        )
        assert grid['lon'].min() >= -106.0
        assert grid['lon'].max() <= -105.0

    def test_covers_western_us_bounding_box(self):
        """Should work with Western US bounding box."""
        grid = generate_grid_points(
            lat_min=31.0,
            lat_max=49.0,
            lon_min=-125.0,
            lon_max=-102.0,
            resolution=0.5,
        )
        # Verify grid covers the expected area
        assert grid['lat'].min() == 31.0
        assert grid['lon'].min() == -125.0
        assert len(grid) > 1000  # Large grid

    def test_single_point_grid(self):
        """Should handle case where min equals max."""
        grid = generate_grid_points(
            lat_min=39.0,
            lat_max=39.0,
            lon_min=-106.0,
            lon_max=-106.0,
            resolution=0.1,
        )
        assert len(grid) == 1
        assert grid['lat'].iloc[0] == 39.0
        assert grid['lon'].iloc[0] == -106.0


class TestOverlayColorRange:
    """Tests for OVERLAY_COLOR_RANGE constant."""

    def test_matches_snow_depth_scale_length(self):
        """OVERLAY_COLOR_RANGE should have same number of colors as SNOW_DEPTH_SCALE."""
        assert len(OVERLAY_COLOR_RANGE) == len(SNOW_DEPTH_SCALE)

    def test_colors_are_rgb_lists(self):
        """Each color should be a list of 3 RGB integers."""
        for color in OVERLAY_COLOR_RANGE:
            assert isinstance(color, list)
            assert len(color) == 3
            assert all(isinstance(c, int) for c in color)

    def test_colors_in_valid_range(self):
        """RGB values should be 0-255."""
        for color in OVERLAY_COLOR_RANGE:
            for value in color:
                assert 0 <= value <= 255

    def test_first_color_is_light_blue(self):
        """First color (Trace) should be light blue."""
        first_color = OVERLAY_COLOR_RANGE[0]
        # Light blue has high blue and moderate red/green
        assert first_color[2] >= 200  # Blue channel high
        assert first_color[0] >= 200  # Light overall

    def test_last_color_is_purple(self):
        """Last color (Extreme) should be purple."""
        last_color = OVERLAY_COLOR_RANGE[-1]
        # Purple has significant red and blue, low green
        assert last_color[0] > 100  # Red present
        assert last_color[2] > 200   # Blue present
        assert last_color[1] < 100   # Low green


class TestRenderOverlayToggle:
    """Tests for render_overlay_toggle function.

    Note: These tests verify the function signature and behavior
    without actually rendering in Streamlit.
    """

    def test_function_exists(self):
        """Function should exist and be callable."""
        assert callable(render_overlay_toggle)

    def test_accepts_container_argument(self):
        """Function should accept optional container argument."""
        import inspect
        sig = inspect.signature(render_overlay_toggle)
        params = list(sig.parameters.keys())
        assert 'container' in params

    def test_container_default_is_none(self):
        """Container parameter should default to None."""
        import inspect
        sig = inspect.signature(render_overlay_toggle)
        container_param = sig.parameters['container']
        assert container_param.default is None


class TestIntegration:
    """Integration tests for forecast overlay components."""

    def test_grid_to_overlay_pipeline(self):
        """Full pipeline: generate grid -> create overlay."""
        # Generate grid
        grid = generate_grid_points(
            lat_min=39.0,
            lat_max=40.0,
            lon_min=-106.0,
            lon_max=-105.0,
            resolution=0.5,
        )

        # Add forecast data
        np.random.seed(42)
        grid['new_snow_cm'] = np.random.uniform(0, 50, len(grid))

        # Create overlay
        layer = create_forecast_overlay(grid)

        # Verify structure
        assert isinstance(layer, pdk.Layer)
        assert layer.type == "HexagonLayer"
        assert len(layer.data) == len(grid)

    def test_overlay_with_various_parameters(self, sample_forecast_points):
        """Overlay should work with various parameter combinations."""
        # Test different configurations
        configs = [
            {'radius': 5000, 'opacity': 0.5, 'extruded': False},
            {'radius': 10000, 'opacity': 0.3, 'extruded': True},
            {'radius': 2500, 'opacity': 0.8, 'extruded': False},
        ]

        for config in configs:
            layer = create_forecast_overlay(sample_forecast_points, **config)
            assert layer.radius == config['radius']
            assert layer.opacity == config['opacity']
            assert layer.extruded == config['extruded']
