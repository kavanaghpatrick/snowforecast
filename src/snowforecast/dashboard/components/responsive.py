"""Responsive layout utilities for the Snowforecast dashboard.

Provides breakpoint detection and responsive layout helpers for mobile,
tablet, and desktop viewports.
"""

import streamlit as st
from typing import Literal

Breakpoint = Literal["mobile", "tablet", "desktop"]

# Breakpoint thresholds (in pixels)
MOBILE_MAX = 768
TABLET_MAX = 1024


def get_viewport_width() -> int:
    """Get estimated viewport width from session state or default.

    Returns:
        Viewport width in pixels. Defaults to 1200 if not set.
    """
    return st.session_state.get('viewport_width', 1200)


def get_breakpoint() -> Breakpoint:
    """Determine current breakpoint based on viewport.

    Returns:
        "mobile" for width < 768px
        "tablet" for width 768-1024px
        "desktop" for width > 1024px
    """
    width = get_viewport_width()
    if width < MOBILE_MAX:
        return "mobile"
    elif width < TABLET_MAX:
        return "tablet"
    return "desktop"


def is_mobile() -> bool:
    """Check if current viewport is mobile.

    Returns:
        True if viewport width < 768px.
    """
    return get_breakpoint() == "mobile"


def is_tablet() -> bool:
    """Check if current viewport is tablet.

    Returns:
        True if viewport width is 768-1024px.
    """
    return get_breakpoint() == "tablet"


def is_desktop() -> bool:
    """Check if current viewport is desktop.

    Returns:
        True if viewport width > 1024px.
    """
    return get_breakpoint() == "desktop"


def get_column_ratio() -> tuple[int, int]:
    """Get map:detail column ratio for current breakpoint.

    Returns:
        Tuple of (map_weight, detail_weight):
        - Desktop: (2, 1) - 2:1 ratio
        - Tablet: (3, 2) - 3:2 ratio
        - Mobile: (1, 1) - equal (stacked layout)
    """
    bp = get_breakpoint()
    if bp == "desktop":
        return (2, 1)  # 2:1 ratio
    elif bp == "tablet":
        return (3, 2)  # 3:2 ratio
    return (1, 1)  # Full width each on mobile


def should_show_3d() -> bool:
    """Determine if 3D terrain should be enabled.

    Returns:
        True on tablet/desktop, False on mobile for performance.
    """
    # Disable on mobile for performance
    return not is_mobile()


def get_touch_target_size() -> int:
    """Get minimum touch target size in pixels.

    Returns:
        44 pixels on mobile (Apple HIG guideline), 32 otherwise.
    """
    return 44 if is_mobile() else 32


def render_responsive_columns():
    """Return appropriate column layout for current breakpoint.

    Returns:
        st.columns object on tablet/desktop, None on mobile
        to signal stacked layout should be used.
    """
    bp = get_breakpoint()
    if bp == "mobile":
        # Single column - return None to signal stacked layout
        return None
    else:
        ratio = get_column_ratio()
        return st.columns(ratio)


def inject_responsive_css() -> None:
    """Inject CSS for responsive behavior.

    Adds mobile-first responsive styles including:
    - Larger touch targets on mobile
    - Reduced padding on mobile
    - Responsive container sizing
    """
    st.markdown('''
    <style>
    /* Mobile-first responsive styles */
    @media (max-width: 768px) {
        /* Ensure buttons meet 44px minimum touch target */
        .stButton button {
            min-height: 44px;
            min-width: 44px;
        }

        /* Reduce container padding on mobile */
        .main .block-container {
            padding: 1rem;
        }

        /* Make select boxes more touch-friendly */
        .stSelectbox > div > div {
            min-height: 44px;
        }

        /* Stack columns on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
    }

    @media (max-width: 1024px) and (min-width: 769px) {
        /* Tablet-specific styles */
        .main .block-container {
            padding: 1.5rem;
        }
    }

    /* Desktop styles */
    @media (min-width: 1025px) {
        .main .block-container {
            padding: 2rem;
        }
    }
    </style>
    ''', unsafe_allow_html=True)


def set_viewport_width(width: int) -> None:
    """Set viewport width in session state.

    This is typically called from JavaScript that detects the actual
    browser viewport width.

    Args:
        width: Viewport width in pixels.
    """
    st.session_state['viewport_width'] = width


def inject_viewport_detector() -> None:
    """Inject JavaScript to detect viewport width.

    This script detects the browser viewport width and stores it
    in a hidden div, which can be read on subsequent reruns.
    """
    st.markdown('''
    <script>
    // Report viewport width to Streamlit
    const reportViewportWidth = () => {
        const width = window.innerWidth;
        // Use Streamlit's setComponentValue if available
        if (window.parent && window.parent.postMessage) {
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: width
            }, '*');
        }
    };

    // Report on load and resize
    reportViewportWidth();
    window.addEventListener('resize', reportViewportWidth);
    </script>
    ''', unsafe_allow_html=True)
