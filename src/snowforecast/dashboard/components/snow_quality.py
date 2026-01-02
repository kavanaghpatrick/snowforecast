"""Snow quality indicators for skiing conditions.

This module provides snow quality metrics including:
- SLR (Snow-to-Liquid Ratio) calculation
- Temperature trend analysis
- Quality classification (Powder, Good, Wet/Heavy, Icy)
- Streamlit display components with badges and explanations
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SnowQuality(Enum):
    """Snow quality classification for skiing conditions."""

    POWDER = "powder"      # SLR > 12, temp < -5C
    GOOD = "good"          # SLR > 8, temp < 0C
    WET_HEAVY = "wet"      # SLR < 8 or temp > 0C
    ICY = "icy"            # Temp oscillating around 0C (freeze-thaw cycle)


@dataclass
class QualityMetrics:
    """Container for snow quality measurements.

    Attributes:
        slr: Snow-to-Liquid Ratio (higher = lighter snow)
        temp_c: Current temperature in Celsius
        temp_trend: Temperature trend ("warming", "cooling", "stable")
        quality: Classified snow quality
    """

    slr: float
    temp_c: float
    temp_trend: str
    quality: SnowQuality


def calculate_slr(snow_depth_change: float, precip_mm: float) -> float:
    """Calculate Snow-to-Liquid Ratio.

    SLR indicates snow density. Higher ratios mean lighter, fluffier snow.
    Typical ranges:
    - Powder: 15-20:1
    - Average: 10-12:1
    - Wet snow: 5-8:1

    Args:
        snow_depth_change: Change in snow depth in cm (positive = accumulation)
        precip_mm: Precipitation amount in mm (liquid equivalent)

    Returns:
        Snow-to-liquid ratio. Returns 10.0 (average) when precip is zero
        or negative to avoid division errors.

    Examples:
        >>> calculate_slr(15.0, 10.0)  # 15cm snow from 10mm water = 15:1
        15.0
        >>> calculate_slr(8.0, 10.0)   # 8cm snow from 10mm water = 8:1
        8.0
        >>> calculate_slr(10.0, 0.0)   # No precip, return default
        10.0
    """
    if precip_mm <= 0:
        return 10.0  # Default average SLR when no precipitation

    # Convert snow depth from cm to mm for ratio calculation
    # snow_depth_change (cm) * 10 = mm
    snow_mm = snow_depth_change * 10
    return snow_mm / precip_mm


def get_temp_trend(temps: list[float]) -> str:
    """Determine temperature trend from hourly temperatures.

    Analyzes temperature sequence to determine if conditions are
    warming, cooling, or stable.

    Args:
        temps: List of temperatures in Celsius (chronological order).
               First element is oldest, last element is most recent.

    Returns:
        - "warming" if temperature increased by more than 2C
        - "cooling" if temperature decreased by more than 2C
        - "stable" if change is within 2C or insufficient data

    Examples:
        >>> get_temp_trend([-10, -8, -5, -2])  # Rising 8 degrees
        'warming'
        >>> get_temp_trend([0, -2, -5, -8])    # Dropping 8 degrees
        'cooling'
        >>> get_temp_trend([-5, -4, -5, -6])   # Within 2 degrees
        'stable'
        >>> get_temp_trend([-5])               # Single value
        'stable'
    """
    if len(temps) < 2:
        return "stable"

    delta = temps[-1] - temps[0]

    if delta > 2:
        return "warming"
    elif delta < -2:
        return "cooling"
    return "stable"


def classify_snow_quality(slr: float, temp_c: float, temp_trend: str) -> SnowQuality:
    """Classify snow quality based on metrics.

    Classification logic:
    - Powder: SLR > 12 AND temp < -5C (cold, light snow)
    - Icy: temp_trend is "warming" AND temp was recently below 0C
           AND now above -2C (freeze-thaw cycle indicator)
    - Wet/Heavy: SLR < 8 OR temp > 0C (dense/melting snow)
    - Good: Everything else with reasonable metrics

    Args:
        slr: Snow-to-liquid ratio
        temp_c: Current temperature in Celsius
        temp_trend: Temperature trend ("warming", "cooling", "stable")

    Returns:
        SnowQuality enum value

    Examples:
        >>> classify_snow_quality(15.0, -8.0, "stable")
        <SnowQuality.POWDER: 'powder'>
        >>> classify_snow_quality(10.0, -3.0, "stable")
        <SnowQuality.GOOD: 'good'>
        >>> classify_snow_quality(6.0, 2.0, "warming")
        <SnowQuality.WET_HEAVY: 'wet'>
    """
    # Check for icy conditions first (freeze-thaw cycle)
    # This happens when warming after being frozen, temps around 0C
    if temp_trend == "warming" and -2 <= temp_c <= 2:
        return SnowQuality.ICY

    # Fresh Powder: High SLR and cold temperatures
    if slr > 12 and temp_c < -5:
        return SnowQuality.POWDER

    # Wet/Heavy: Low SLR or above freezing
    if slr < 8 or temp_c > 0:
        return SnowQuality.WET_HEAVY

    # Good: Moderate conditions (SLR >= 8, temp <= 0C)
    if slr >= 8 and temp_c <= 0:
        return SnowQuality.GOOD

    # Default fallback
    return SnowQuality.GOOD


def get_quality_badge(quality: SnowQuality) -> tuple[str, str, str]:
    """Return display properties for quality badge.

    Args:
        quality: SnowQuality enum value

    Returns:
        Tuple of (emoji, label, hex_color) for badge display

    Examples:
        >>> get_quality_badge(SnowQuality.POWDER)
        ('*', 'Fresh Powder', '#60a5fa')
        >>> get_quality_badge(SnowQuality.GOOD)
        ('+', 'Good Snow', '#22c55e')
    """
    badges = {
        SnowQuality.POWDER: ("*", "Fresh Powder", "#60a5fa"),    # Blue
        SnowQuality.GOOD: ("+", "Good Snow", "#22c55e"),         # Green
        SnowQuality.WET_HEAVY: ("~", "Wet/Heavy", "#f59e0b"),    # Yellow/Orange
        SnowQuality.ICY: ("#", "Icy", "#94a3b8"),                # Gray
    }
    return badges.get(quality, ("?", "Unknown", "#6b7280"))


def get_slr_description(slr: float) -> str:
    """Get human-readable description of SLR value.

    Args:
        slr: Snow-to-liquid ratio

    Returns:
        Description string
    """
    if slr >= 15:
        return "Very light powder"
    elif slr >= 12:
        return "Light powder"
    elif slr >= 10:
        return "Average density"
    elif slr >= 8:
        return "Slightly dense"
    else:
        return "Heavy/wet snow"


def get_quality_explanation(quality: SnowQuality) -> str:
    """Get explanation text for snow quality classification.

    Args:
        quality: SnowQuality enum value

    Returns:
        Explanation string suitable for tooltips
    """
    explanations = {
        SnowQuality.POWDER: (
            "Fresh powder conditions - Light, fluffy snow with high "
            "snow-to-liquid ratio. Ideal for skiing and riding."
        ),
        SnowQuality.GOOD: (
            "Good skiing conditions - Moderate density snow that holds "
            "edges well. Pleasant for all skill levels."
        ),
        SnowQuality.WET_HEAVY: (
            "Wet or heavy snow - Higher water content makes snow sticky "
            "and harder to ski. Best in early morning before warming."
        ),
        SnowQuality.ICY: (
            "Icy conditions - Freeze-thaw cycle has created hard, "
            "slick surfaces. Requires sharp edges and caution."
        ),
    }
    return explanations.get(quality, "Unknown conditions")


def render_snow_quality_badge(metrics: QualityMetrics, container=None) -> None:
    """Render snow quality badge in Streamlit.

    Displays a colored badge with emoji and quality label.

    Args:
        metrics: QualityMetrics dataclass with quality information
        container: Streamlit container (st, column, etc.). If None, uses st.

    Side Effects:
        Renders HTML badge to Streamlit container
    """
    import streamlit as st

    target = container if container is not None else st

    emoji, label, color = get_quality_badge(metrics.quality)

    badge_html = f"""
    <div style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: {color}22;
        border: 1px solid {color};
        border-radius: 16px;
        padding: 4px 12px;
        font-size: 14px;
    ">
        <span style="font-size: 16px;">{emoji}</span>
        <span style="color: {color}; font-weight: 600;">{label}</span>
    </div>
    """

    target.markdown(badge_html, unsafe_allow_html=True)


def render_snow_quality_details(metrics: QualityMetrics, container=None) -> None:
    """Render detailed snow quality section with SLR and explanation.

    Displays:
    - Quality badge
    - SLR ratio display (e.g., "SLR: 15:1 (Powder)")
    - Temperature and trend
    - Explanation tooltip

    Args:
        metrics: QualityMetrics dataclass with quality information
        container: Streamlit container (st, column, etc.). If None, uses st.

    Side Effects:
        Renders multiple Streamlit elements to container
    """
    import streamlit as st

    target = container if container is not None else st

    # Quality badge at top
    render_snow_quality_badge(metrics, target)

    target.markdown("")

    # SLR display
    slr_desc = get_slr_description(metrics.slr)
    slr_text = f"**SLR:** {metrics.slr:.1f}:1 ({slr_desc})"
    target.markdown(slr_text)

    # Temperature with trend indicator
    trend_arrows = {
        "warming": " ^",
        "cooling": " v",
        "stable": " -",
    }
    trend_arrow = trend_arrows.get(metrics.temp_trend, "")
    temp_text = f"**Temperature:** {metrics.temp_c:.1f}C{trend_arrow} ({metrics.temp_trend})"
    target.markdown(temp_text)

    # Explanation in expander
    explanation = get_quality_explanation(metrics.quality)
    with target.expander("What does this mean?"):
        st.write(explanation)


def render_snow_quality_compact(metrics: QualityMetrics, container=None) -> None:
    """Render compact snow quality display.

    Single-line format suitable for tables or limited space:
    "SLR: 15:1 | Fresh Powder"

    Args:
        metrics: QualityMetrics dataclass with quality information
        container: Streamlit container (st, column, etc.). If None, uses st.

    Side Effects:
        Renders compact text to Streamlit container
    """
    import streamlit as st

    target = container if container is not None else st

    emoji, label, color = get_quality_badge(metrics.quality)

    compact_html = f"""
    <span style="font-size: 13px;">
        SLR: {metrics.slr:.0f}:1 |
        <span style="color: {color}; font-weight: 500;">{emoji} {label}</span>
    </span>
    """

    target.markdown(compact_html, unsafe_allow_html=True)


def create_quality_metrics(
    snow_depth_change: float,
    precip_mm: float,
    temp_c: float,
    hourly_temps: Optional[list[float]] = None,
) -> QualityMetrics:
    """Create QualityMetrics from raw weather data.

    Convenience function that calculates SLR, determines trend,
    and classifies quality in one call.

    Args:
        snow_depth_change: Change in snow depth in cm
        precip_mm: Precipitation in mm (liquid equivalent)
        temp_c: Current temperature in Celsius
        hourly_temps: Optional list of hourly temperatures for trend.
                     If None, trend defaults to "stable".

    Returns:
        QualityMetrics dataclass with all fields populated

    Examples:
        >>> metrics = create_quality_metrics(15.0, 10.0, -8.0, [-10, -9, -8])
        >>> metrics.quality
        <SnowQuality.POWDER: 'powder'>
    """
    slr = calculate_slr(snow_depth_change, precip_mm)

    if hourly_temps:
        temp_trend = get_temp_trend(hourly_temps)
    else:
        temp_trend = "stable"

    quality = classify_snow_quality(slr, temp_c, temp_trend)

    return QualityMetrics(
        slr=slr,
        temp_c=temp_c,
        temp_trend=temp_trend,
        quality=quality,
    )
