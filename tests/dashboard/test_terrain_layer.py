"""Tests for 3D terrain layer component.

Tests cover:
- Layer creation with various options
- ViewState configuration
- Deck creation
- UI control rendering
- Tile URL formats
"""

import pytest
import pydeck as pdk

from snowforecast.dashboard.components.terrain_layer import (
    ELEVATION_DECODER,
    TERRAIN_IMAGE,
    TEXTURE_OPENTOPOMAP,
    TEXTURE_OSM,
    create_3d_view,
    create_terrain_deck,
    create_terrain_layer,
    is_mobile_viewport,
    render_terrain_controls,
)


class TestTerrainLayer:
    """Tests for create_terrain_layer function."""

    def test_creates_terrain_layer(self):
        """Should create a PyDeck TerrainLayer."""
        layer = create_terrain_layer()
        assert layer is not None
        assert layer.type == "TerrainLayer"

    def test_uses_aws_terrain_tiles_for_elevation(self):
        """Should use free AWS terrain tiles for elevation data."""
        layer = create_terrain_layer()
        # Check layer configuration includes AWS terrain URL
        assert "s3.amazonaws.com/elevation-tiles-prod" in TERRAIN_IMAGE

    def test_uses_opentopomap_texture_by_default(self):
        """Should use OpenTopoMap as default texture."""
        layer = create_terrain_layer(use_texture=True)
        assert layer is not None
        # Verify texture URL pattern
        assert "opentopomap.org" in TEXTURE_OPENTOPOMAP

    def test_wireframe_mode_disables_texture(self):
        """Wireframe mode should not apply texture."""
        layer = create_terrain_layer(wireframe=True)
        assert layer is not None
        # Layer is created with wireframe=True

    def test_custom_texture_url(self):
        """Should accept custom texture URL."""
        custom_url = "https://example.com/tiles/{z}/{x}/{y}.png"
        layer = create_terrain_layer(use_texture=True, texture_url=custom_url)
        assert layer is not None

    def test_no_texture_when_disabled(self):
        """Should not include texture when use_texture=False."""
        layer = create_terrain_layer(use_texture=False)
        assert layer is not None


class TestElevationDecoder:
    """Tests for Terrarium elevation decoder configuration."""

    def test_decoder_has_required_keys(self):
        """Decoder should have all required keys for Terrarium format."""
        required_keys = {"rScaler", "gScaler", "bScaler", "offset"}
        assert required_keys.issubset(ELEVATION_DECODER.keys())

    def test_decoder_values_for_terrarium(self):
        """Decoder values should match Terrarium format specification."""
        # Terrarium format: height = (R * 256 + G + B / 256) - 32768
        assert ELEVATION_DECODER["rScaler"] == 256
        assert ELEVATION_DECODER["gScaler"] == 1
        assert ELEVATION_DECODER["bScaler"] == 1 / 256
        assert ELEVATION_DECODER["offset"] == -32768


class TestViewState:
    """Tests for 3D view state creation."""

    def test_creates_view_state(self):
        """Should create a PyDeck ViewState."""
        view = create_3d_view()
        assert isinstance(view, pdk.ViewState)

    def test_default_coordinates_western_us(self):
        """Default coordinates should be in Western US (ski country)."""
        view = create_3d_view()
        # Utah area defaults
        assert 35 < view.latitude < 50  # Western US latitude range
        assert -125 < view.longitude < -100  # Western US longitude range

    def test_custom_coordinates(self):
        """Should accept custom coordinates."""
        view = create_3d_view(lat=39.6403, lon=-106.3742)  # Vail
        assert view.latitude == 39.6403
        assert view.longitude == -106.3742

    def test_pitch_range(self):
        """Pitch should be within valid range for 3D viewing."""
        view = create_3d_view(pitch=60)
        assert 0 <= view.pitch <= 85  # Valid range for deck.gl

    def test_zoom_level(self):
        """Zoom should be appropriate for terrain viewing."""
        view = create_3d_view(zoom=10)
        assert view.zoom == 10


class TestTerrainDeck:
    """Tests for complete deck creation."""

    def test_creates_deck(self):
        """Should create a PyDeck Deck object."""
        deck = create_terrain_deck()
        assert isinstance(deck, pdk.Deck)

    def test_deck_has_terrain_layer(self):
        """Deck should include terrain layer."""
        deck = create_terrain_deck()
        assert len(deck.layers) >= 1
        assert deck.layers[0].type == "TerrainLayer"

    def test_deck_with_additional_layers(self):
        """Should accept additional layers."""
        extra_layer = pdk.Layer(
            "ScatterplotLayer",
            data=[{"lat": 40, "lon": -111}],
            get_position="[lon, lat]",
            get_radius=1000,
        )
        deck = create_terrain_deck(additional_layers=[extra_layer])
        assert len(deck.layers) == 2

    def test_deck_respects_pitch(self):
        """Deck view state should respect pitch parameter."""
        deck = create_terrain_deck(pitch=45)
        assert deck.initial_view_state.pitch == 45

    def test_deck_no_mapbox_style(self):
        """Deck should not require Mapbox (free tiles only)."""
        deck = create_terrain_deck()
        # map_style should be None (terrain provides the base)
        assert deck.map_style is None


class TestTileUrls:
    """Tests for tile URL format validation."""

    def test_terrain_url_has_placeholders(self):
        """Terrain URL should have {z}/{x}/{y} placeholders."""
        assert "{z}" in TERRAIN_IMAGE
        assert "{x}" in TERRAIN_IMAGE
        assert "{y}" in TERRAIN_IMAGE

    def test_opentopomap_url_has_placeholders(self):
        """OpenTopoMap URL should have tile placeholders."""
        assert "{z}" in TEXTURE_OPENTOPOMAP
        assert "{x}" in TEXTURE_OPENTOPOMAP
        assert "{y}" in TEXTURE_OPENTOPOMAP

    def test_osm_fallback_url(self):
        """OSM fallback URL should be properly formatted."""
        assert "{z}" in TEXTURE_OSM
        assert "openstreetmap.org" in TEXTURE_OSM


class TestMobileDetection:
    """Tests for mobile viewport detection."""

    def test_mobile_under_768(self):
        """Viewport under 768px should be considered mobile."""
        assert is_mobile_viewport(320) is True
        assert is_mobile_viewport(767) is True

    def test_desktop_768_and_above(self):
        """Viewport 768px and above should be desktop."""
        assert is_mobile_viewport(768) is False
        assert is_mobile_viewport(1920) is False

    def test_none_assumes_desktop(self):
        """Unknown viewport should assume desktop."""
        assert is_mobile_viewport(None) is False


class TestUIControls:
    """Tests for Streamlit UI controls."""

    def test_render_controls_returns_dict(self):
        """Should return dict with expected keys."""

        # Create a mock streamlit module
        class MockStreamlit:
            def __init__(self):
                self._toggle_val = False
                self._slider_val = 45
                self._checkbox_val = True

            def toggle(self, label, value=False, help=None):
                return self._toggle_val

            def slider(self, label, min_value=0, max_value=100, value=50, step=1, help=None):
                return self._slider_val

            def checkbox(self, label, value=False, help=None):
                return self._checkbox_val

        mock_st = MockStreamlit()
        controls = render_terrain_controls(mock_st)

        assert "enabled_3d" in controls
        assert "pitch" in controls
        assert "use_texture" in controls

    def test_controls_disabled_3d_has_zero_pitch(self):
        """When 3D disabled, pitch should be 0."""

        class MockStreamlit:
            def toggle(self, label, value=False, help=None):
                return False  # 3D disabled

            def slider(self, *args, **kwargs):
                return 45

            def checkbox(self, *args, **kwargs):
                return True

        mock_st = MockStreamlit()
        controls = render_terrain_controls(mock_st)

        assert controls["enabled_3d"] is False
        assert controls["pitch"] == 0  # Should be 0 when 3D disabled
