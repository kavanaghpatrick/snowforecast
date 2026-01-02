"""Color scale definitions for snow and elevation visualization.

This module provides consistent color scales used across the dashboard:
- Snow depth: Blue-purple gradient (light blue → purple for deep snow)
- Elevation: Green-brown-white gradient (valleys → peaks)

All colors are provided in multiple formats:
- Hex strings for CSS/HTML
- RGB lists for PyDeck layers
- Category names for legends
"""



# =============================================================================
# SNOW DEPTH COLOR SCALE
# =============================================================================

# (threshold_cm, hex_color, category_name)
SNOW_DEPTH_SCALE = [
    (10, "#E6F3FF", "Trace"),
    (30, "#ADD8E6", "Light"),
    (60, "#6495ED", "Moderate"),
    (100, "#4169E1", "Heavy"),
    (150, "#0000CD", "Very Heavy"),
    (9999, "#8A2BE2", "Extreme"),
]


def snow_depth_to_hex(depth_cm: float) -> str:
    """Convert snow depth to hex color string.

    Args:
        depth_cm: Snow depth in centimeters

    Returns:
        Hex color string (e.g., "#ADD8E6")

    Examples:
        >>> snow_depth_to_hex(5)
        '#E6F3FF'
        >>> snow_depth_to_hex(50)
        '#6495ED'
        >>> snow_depth_to_hex(200)
        '#8A2BE2'
    """
    if depth_cm < 0:
        depth_cm = 0

    for threshold, color, _ in SNOW_DEPTH_SCALE:
        if depth_cm < threshold:
            return color

    return SNOW_DEPTH_SCALE[-1][1]


def snow_depth_to_rgb(depth_cm: float, alpha: int = 200) -> list[int]:
    """Convert snow depth to RGBA list for PyDeck.

    Args:
        depth_cm: Snow depth in centimeters
        alpha: Alpha transparency (0-255), default 200

    Returns:
        List of [R, G, B, A] values (0-255)

    Examples:
        >>> snow_depth_to_rgb(50)
        [100, 149, 237, 200]
        >>> snow_depth_to_rgb(50, alpha=128)
        [100, 149, 237, 128]
    """
    hex_color = snow_depth_to_hex(depth_cm)

    # Convert hex to RGB
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    return [r, g, b, alpha]


def snow_depth_category(depth_cm: float) -> str:
    """Get category name for snow depth.

    Args:
        depth_cm: Snow depth in centimeters

    Returns:
        Category name (e.g., "Light", "Heavy")

    Examples:
        >>> snow_depth_category(5)
        'Trace'
        >>> snow_depth_category(120)
        'Very Heavy'
    """
    if depth_cm < 0:
        depth_cm = 0

    for threshold, _, category in SNOW_DEPTH_SCALE:
        if depth_cm < threshold:
            return category

    return SNOW_DEPTH_SCALE[-1][2]


# =============================================================================
# ELEVATION COLOR SCALE
# =============================================================================

# (threshold_m, [R, G, B], category_name)
ELEVATION_SCALE = [
    (1500, [34, 139, 34], "Valley"),  # Forest green
    (2000, [143, 188, 143], "Foothills"),  # Sage green
    (2500, [210, 180, 140], "Mid-Mountain"),  # Tan
    (3000, [139, 119, 101], "High Mountain"),  # Brown
    (3500, [169, 169, 169], "Alpine"),  # Gray
    (9999, [255, 255, 255], "Peak"),  # White
]


def elevation_to_rgb(elevation_m: float, alpha: int = 200) -> list[int]:
    """Convert elevation to RGBA list for PyDeck.

    Uses traditional hypsometric tints:
    - Green for valleys/forests
    - Brown for mid-elevations
    - Gray/white for alpine/peaks

    Args:
        elevation_m: Elevation in meters
        alpha: Alpha transparency (0-255), default 200

    Returns:
        List of [R, G, B, A] values (0-255)

    Examples:
        >>> elevation_to_rgb(1000)
        [34, 139, 34, 200]
        >>> elevation_to_rgb(3200)
        [169, 169, 169, 200]
    """
    if elevation_m < 0:
        elevation_m = 0

    for threshold, rgb, _ in ELEVATION_SCALE:
        if elevation_m < threshold:
            return rgb + [alpha]

    return ELEVATION_SCALE[-1][1] + [alpha]


def elevation_category(elevation_m: float) -> str:
    """Get category name for elevation.

    Args:
        elevation_m: Elevation in meters

    Returns:
        Category name (e.g., "Alpine", "Valley")
    """
    if elevation_m < 0:
        elevation_m = 0

    for threshold, _, category in ELEVATION_SCALE:
        if elevation_m < threshold:
            return category

    return ELEVATION_SCALE[-1][2]


# =============================================================================
# LEGEND RENDERING
# =============================================================================

def render_snow_legend(container=None) -> None:
    """Render snow depth color legend in Streamlit.

    Args:
        container: Streamlit container (st, st.sidebar, st.columns()[0], etc.)
                   If None, uses st directly.

    Example:
        >>> render_snow_legend(st.sidebar)
    """
    import streamlit as st

    target = container if container is not None else st

    target.markdown("**Snow Depth**")

    for threshold, color, category in SNOW_DEPTH_SCALE[:-1]:
        target.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0;">'
            f'<span style="background:{color};width:20px;height:14px;display:inline-block;'
            f'border:1px solid #ccc;border-radius:2px;"></span>'
            f'<span style="font-size:12px;">{category} (&lt;{threshold}cm)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Last category (extreme)
    _, color, category = SNOW_DEPTH_SCALE[-1]
    prev_threshold = SNOW_DEPTH_SCALE[-2][0]
    target.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0;">'
        f'<span style="background:{color};width:20px;height:14px;display:inline-block;'
        f'border:1px solid #ccc;border-radius:2px;"></span>'
        f'<span style="font-size:12px;">{category} (&gt;{prev_threshold}cm)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_elevation_legend(container=None) -> None:
    """Render elevation color legend in Streamlit.

    Args:
        container: Streamlit container. If None, uses st directly.
    """
    import streamlit as st

    target = container if container is not None else st

    target.markdown("**Elevation**")

    for i, (threshold, rgb, category) in enumerate(ELEVATION_SCALE[:-1]):
        color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        target.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0;">'
            f'<span style="background:{color};width:20px;height:14px;display:inline-block;'
            f'border:1px solid #ccc;border-radius:2px;"></span>'
            f'<span style="font-size:12px;">{category} (&lt;{threshold}m)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Last category (peak)
    _, rgb, category = ELEVATION_SCALE[-1]
    prev_threshold = ELEVATION_SCALE[-2][0]
    color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    target.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0;">'
        f'<span style="background:{color};width:20px;height:14px;display:inline-block;'
        f'border:1px solid #ccc;border-radius:2px;"></span>'
        f'<span style="font-size:12px;">{category} (&gt;{prev_threshold}m)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple.

    Args:
        hex_color: Hex color string (e.g., "#ADD8E6")

    Returns:
        Tuple of (R, G, B) values (0-255)
    """
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB values to hex color string.

    Args:
        r: Red value (0-255)
        g: Green value (0-255)
        b: Blue value (0-255)

    Returns:
        Hex color string (e.g., "#ADD8E6")
    """
    return f"#{r:02x}{g:02x}{b:02x}"
