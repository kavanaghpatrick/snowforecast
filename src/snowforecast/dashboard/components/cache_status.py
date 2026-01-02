"""Cache status components for the dashboard.

Provides UI components for displaying cache freshness and status
information to users, including warnings for stale data.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

# Freshness thresholds
FRESHNESS_THRESHOLDS = {
    "fresh": timedelta(minutes=30),
    "recent": timedelta(hours=2),
    "stale": timedelta(hours=6),
    # Anything older is "old"
}

# Colors for freshness indicators
FRESHNESS_COLORS = {
    "fresh": "#22c55e",   # Green
    "recent": "#3b82f6",  # Blue
    "stale": "#f59e0b",   # Amber/Orange
    "old": "#ef4444",     # Red
    "unknown": "#9ca3af", # Gray
}


def get_cache_freshness(last_updated: Optional[datetime]) -> Tuple[str, str]:
    """Return (status_text, color) based on cache age.

    Args:
        last_updated: Datetime when cache was last updated, or None

    Returns:
        Tuple of (status_text, hex_color)

    Example:
        >>> from datetime import datetime, timedelta
        >>> recent_time = datetime.now() - timedelta(minutes=15)
        >>> status, color = get_cache_freshness(recent_time)
        >>> status
        'Fresh'
        >>> color
        '#22c55e'
    """
    if last_updated is None:
        return ("Unknown", FRESHNESS_COLORS["unknown"])

    # Calculate age
    now = datetime.now()
    # Handle timezone-aware datetimes by making both naive for comparison
    if last_updated.tzinfo is not None:
        last_updated = last_updated.replace(tzinfo=None)

    age = now - last_updated

    # Determine freshness level
    if age < FRESHNESS_THRESHOLDS["fresh"]:
        return ("Fresh", FRESHNESS_COLORS["fresh"])
    elif age < FRESHNESS_THRESHOLDS["recent"]:
        return ("Recent", FRESHNESS_COLORS["recent"])
    elif age < FRESHNESS_THRESHOLDS["stale"]:
        return ("Stale", FRESHNESS_COLORS["stale"])
    else:
        return ("Old", FRESHNESS_COLORS["old"])


def get_cache_freshness_emoji(last_updated: Optional[datetime]) -> str:
    """Return emoji indicator based on cache freshness.

    Args:
        last_updated: Datetime when cache was last updated

    Returns:
        Streamlit-compatible emoji string

    Example:
        >>> from datetime import datetime
        >>> get_cache_freshness_emoji(datetime.now())
        ':green_circle:'
    """
    status, _ = get_cache_freshness(last_updated)
    emoji_map = {
        "Fresh": ":green_circle:",
        "Recent": ":blue_circle:",
        "Stale": ":orange_circle:",
        "Old": ":red_circle:",
        "Unknown": ":white_circle:",
    }
    return emoji_map.get(status, ":white_circle:")


def format_cache_age(last_updated: Optional[datetime]) -> str:
    """Format cache age as human-readable string.

    Args:
        last_updated: Datetime when cache was last updated

    Returns:
        Human-readable age string like "5 min ago" or "2.3 hrs ago"

    Example:
        >>> from datetime import datetime, timedelta
        >>> recent = datetime.now() - timedelta(minutes=5)
        >>> format_cache_age(recent)
        '5 min ago'
    """
    if last_updated is None:
        return "Never"

    now = datetime.now()
    if last_updated.tzinfo is not None:
        last_updated = last_updated.replace(tzinfo=None)

    age = now - last_updated
    total_seconds = age.total_seconds()

    if total_seconds < 60:
        return "Just now"
    elif total_seconds < 3600:
        minutes = int(total_seconds / 60)
        return f"{minutes} min ago"
    elif total_seconds < 86400:
        hours = total_seconds / 3600
        return f"{hours:.1f} hrs ago"
    else:
        days = total_seconds / 86400
        return f"{days:.1f} days ago"


def render_cache_status(
    predictor: Any,
    container=None,
    show_details: bool = True,
) -> None:
    """Render cache status indicator.

    Args:
        predictor: CachedPredictor instance with get_cache_stats() method
        container: Streamlit container to render in (default: sidebar)
        show_details: Whether to show detailed stats

    Example:
        >>> predictor = CachedPredictor()
        >>> render_cache_status(predictor)
    """
    target = container if container is not None else st.sidebar

    # Get cache stats
    if not hasattr(predictor, 'get_cache_stats'):
        target.warning("Cache status unavailable")
        return

    try:
        stats = predictor.get_cache_stats()
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        target.warning("Could not retrieve cache status")
        return

    # Extract data
    forecast_count = stats.get("forecast_count", 0)
    terrain_count = stats.get("terrain_count", 0)
    latest_run = stats.get("latest_run_time")

    # Get freshness status
    status_text, color = get_cache_freshness(latest_run)
    emoji = get_cache_freshness_emoji(latest_run)

    # Render status header
    target.markdown("---")
    target.markdown("**Cache Status**")

    # Freshness indicator
    target.markdown(f"{emoji} **{status_text}**")

    # Age display
    age_str = format_cache_age(latest_run)
    target.caption(f"Last updated: {age_str}")

    # Detailed timestamp if recent enough
    if latest_run is not None and show_details:
        if latest_run.tzinfo is not None:
            latest_run = latest_run.replace(tzinfo=None)
        timestamp_str = latest_run.strftime("%Y-%m-%d %H:%M")
        target.caption(f"({timestamp_str})")

    # Cache stats
    if show_details:
        target.caption(f"Forecasts: {forecast_count} | Terrain: {terrain_count}")


def render_cache_status_badge(
    predictor: Any,
    container=None,
) -> None:
    """Render compact cache status badge.

    Args:
        predictor: CachedPredictor instance
        container: Streamlit container to render in

    Example:
        >>> col1, col2 = st.columns(2)
        >>> with col1:
        ...     render_cache_status_badge(predictor)
    """
    target = container if container is not None else st

    if not hasattr(predictor, 'get_cache_stats'):
        return

    try:
        stats = predictor.get_cache_stats()
        latest_run = stats.get("latest_run_time")
    except Exception:
        return

    status_text, color = get_cache_freshness(latest_run)
    age_str = format_cache_age(latest_run)

    # Render as colored badge
    target.markdown(
        f"""
        <span style="
            background-color: {color}20;
            color: {color};
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        ">
            {status_text} - {age_str}
        </span>
        """,
        unsafe_allow_html=True,
    )


def render_data_warning(
    last_updated: Optional[datetime],
    threshold_hours: float = 2.0,
    container=None,
) -> None:
    """Show warning if data is stale.

    Args:
        last_updated: Datetime when data was last updated
        threshold_hours: Hours after which to show warning
        container: Streamlit container to render in

    Example:
        >>> from datetime import datetime, timedelta
        >>> old_time = datetime.now() - timedelta(hours=3)
        >>> render_data_warning(old_time)  # Shows warning
    """
    target = container if container is not None else st

    if last_updated is None:
        target.warning("Data freshness unknown - using cached data")
        return

    now = datetime.now()
    if last_updated.tzinfo is not None:
        last_updated = last_updated.replace(tzinfo=None)

    age = now - last_updated
    age_hours = age.total_seconds() / 3600

    if age_hours > threshold_hours:
        target.warning(
            f"Data from {age_hours:.1f} hours ago may be outdated. "
            "Click Refresh to update."
        )


def render_fallback_notice(
    reason: str,
    container=None,
) -> None:
    """Show notice when using fallback data.

    Args:
        reason: Reason for using fallback (e.g., "API unavailable")
        container: Streamlit container to render in

    Example:
        >>> render_fallback_notice("HRRR API timeout")
    """
    target = container if container is not None else st

    target.info(f"Using cached data: {reason}")


def render_data_source_indicator(
    source: str,
    is_cached: bool = True,
    container=None,
) -> None:
    """Show indicator of where data came from.

    Args:
        source: Data source name (e.g., "HRRR", "NBM", "Cache")
        is_cached: Whether data came from cache
        container: Streamlit container to render in

    Example:
        >>> render_data_source_indicator("HRRR", is_cached=True)
    """
    target = container if container is not None else st

    color = "#3b82f6" if is_cached else "#22c55e"

    target.markdown(
        f"""
        <span style="
            color: {color};
            font-size: 11px;
        ">
            {'From cache' if is_cached else 'Live'} ({source})
        </span>
        """,
        unsafe_allow_html=True,
    )


def should_show_stale_warning(last_updated: Optional[datetime]) -> bool:
    """Check if stale data warning should be shown.

    Args:
        last_updated: Datetime when data was last updated

    Returns:
        True if data is old enough to warrant a warning

    Example:
        >>> from datetime import datetime, timedelta
        >>> old = datetime.now() - timedelta(hours=3)
        >>> should_show_stale_warning(old)
        True
    """
    if last_updated is None:
        return True

    now = datetime.now()
    if last_updated.tzinfo is not None:
        last_updated = last_updated.replace(tzinfo=None)

    age = now - last_updated
    return age > FRESHNESS_THRESHOLDS["recent"]
