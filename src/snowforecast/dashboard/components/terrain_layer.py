"""3D Terrain Layer component for Snow Forecast dashboard.

Uses FREE tile sources only:
- Elevation: AWS Terrain Tiles (Terrarium format)
- Texture: OpenTopoMap (topographic map overlay)

No Mapbox API key required.
"""

from typing import Optional

import pydeck as pdk

# FREE AWS Terrain Tiles (Terrarium format)
# Documentation: https://registry.opendata.aws/terrain-tiles/
TERRAIN_IMAGE = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# Terrarium elevation decoder
# RGB values encode elevation as: height = (R * 256 + G + B / 256) - 32768
ELEVATION_DECODER = {
    "rScaler": 256,
    "gScaler": 1,
    "bScaler": 1 / 256,
    "offset": -32768,
}

# FREE texture options - OpenTopoMap for topographic detail
# OpenTopoMap uses OpenStreetMap data with topographic styling
TEXTURE_OPENTOPOMAP = "https://a.tile.opentopomap.org/{z}/{x}/{y}.png"

# Alternative free texture: OpenStreetMap standard
TEXTURE_OSM = "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"

# Stadia Stamen Terrain (requires free API key for production)
# TEXTURE_STAMEN_TERRAIN = "https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg"


def create_terrain_layer(
    use_texture: bool = True,
    texture_url: Optional[str] = None,
    wireframe: bool = False,
    elevation_scale: float = 1.0,
) -> pdk.Layer:
    """Create 3D terrain layer with free tiles.

    Args:
        use_texture: Whether to apply texture overlay. If False, shows wireframe.
        texture_url: Custom texture URL. Defaults to OpenTopoMap.
        wireframe: Show wireframe instead of solid surface.
        elevation_scale: Vertical exaggeration factor (1.0 = actual elevation).

    Returns:
        PyDeck TerrainLayer configured with free tile sources.

    Example:
        >>> layer = create_terrain_layer(use_texture=True)
        >>> deck = pdk.Deck(layers=[layer], initial_view_state=view)
    """
    texture = None
    if use_texture and not wireframe:
        texture = texture_url or TEXTURE_OPENTOPOMAP

    return pdk.Layer(
        "TerrainLayer",
        elevation_data=TERRAIN_IMAGE,
        elevation_decoder=ELEVATION_DECODER,
        texture=texture,
        wireframe=wireframe,
        bounds=None,  # Auto-determine from view
    )


def create_3d_view(
    lat: float = 40.0,
    lon: float = -111.0,
    zoom: float = 9,
    pitch: float = 60,
    bearing: float = 30,
) -> pdk.ViewState:
    """Create 3D view state centered on coordinates.

    Args:
        lat: Center latitude (default: Utah ski country)
        lon: Center longitude
        zoom: Zoom level (7-12 typical for terrain)
        pitch: Camera tilt angle in degrees (0=flat, 60=tilted)
        bearing: Camera rotation in degrees (0=north)

    Returns:
        PyDeck ViewState for 3D terrain viewing.
    """
    return pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=zoom,
        pitch=pitch,
        bearing=bearing,
    )


def create_terrain_deck(
    lat: float = 40.0,
    lon: float = -111.0,
    zoom: float = 9,
    pitch: float = 60,
    bearing: float = 30,
    use_texture: bool = True,
    wireframe: bool = False,
    additional_layers: Optional[list] = None,
) -> pdk.Deck:
    """Create complete PyDeck visualization with 3D terrain.

    Args:
        lat: Center latitude
        lon: Center longitude
        zoom: Zoom level
        pitch: Camera pitch angle (0-60)
        bearing: Camera bearing
        use_texture: Whether to show texture overlay
        wireframe: Show wireframe instead of solid
        additional_layers: Extra layers to render on top of terrain

    Returns:
        PyDeck Deck ready for rendering.

    Example:
        >>> deck = create_terrain_deck(lat=39.6403, lon=-106.3742)  # Vail
        >>> st.pydeck_chart(deck)
    """
    view = create_3d_view(lat=lat, lon=lon, zoom=zoom, pitch=pitch, bearing=bearing)
    terrain = create_terrain_layer(use_texture=use_texture, wireframe=wireframe)

    layers = [terrain]
    if additional_layers:
        layers.extend(additional_layers)

    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        # No map_style needed - terrain layer provides the base
        map_style=None,
    )


def render_terrain_controls(streamlit_module) -> dict:
    """Render Streamlit UI controls for terrain visualization.

    Args:
        streamlit_module: The streamlit module (passed to avoid import issues)

    Returns:
        Dict with user selections: {'enabled_3d', 'pitch', 'use_texture'}

    Example:
        >>> import streamlit as st
        >>> controls = render_terrain_controls(st)
        >>> if controls['enabled_3d']:
        ...     deck = create_terrain_deck(pitch=controls['pitch'])
    """
    st = streamlit_module

    # 2D/3D toggle
    enabled_3d = st.toggle("3D Terrain View", value=False, help="Enable 3D terrain visualization")

    controls = {"enabled_3d": enabled_3d, "pitch": 0, "use_texture": True}

    if enabled_3d:
        # Pitch slider (only shown in 3D mode)
        controls["pitch"] = st.slider(
            "Camera Pitch",
            min_value=0,
            max_value=60,
            value=45,
            step=5,
            help="Camera tilt angle (0 = top-down, 60 = angled)",
        )

        # Texture toggle
        controls["use_texture"] = st.checkbox(
            "Show Topographic Overlay",
            value=True,
            help="Overlay OpenTopoMap texture on terrain",
        )

    return controls


def is_mobile_viewport(viewport_width: Optional[int] = None) -> bool:
    """Check if viewport suggests mobile device.

    3D terrain can be heavy on mobile. This helps decide whether
    to default to 2D or show a warning.

    Args:
        viewport_width: Viewport width in pixels (if available)

    Returns:
        True if likely mobile device
    """
    if viewport_width is None:
        # Cannot determine - assume desktop
        return False
    return viewport_width < 768
