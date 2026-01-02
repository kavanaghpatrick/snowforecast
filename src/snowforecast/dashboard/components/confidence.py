"""Confidence visualization components for the Snowforecast dashboard.

Provides functions to display forecast uncertainty and model confidence
using badges, color coding, and formatted text.
"""

from typing import Optional

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from snowforecast.api.schemas import ConfidenceInterval


def get_confidence_level(ci: ConfidenceInterval, probability: float) -> str:
    """Classify confidence as 'high', 'medium', or 'low'.

    Uses CI width and probability thresholds:
    - High: CI width < 5cm AND probability > 0.7
    - Medium: CI width < 15cm AND probability > 0.4
    - Low: Everything else

    Args:
        ci: Confidence interval with lower and upper bounds
        probability: Snowfall probability (0-1)

    Returns:
        Confidence level string: 'high', 'medium', or 'low'
    """
    ci_width = ci.upper - ci.lower
    if ci_width < 5 and probability > 0.7:
        return "high"
    elif ci_width < 15 and probability > 0.4:
        return "medium"
    return "low"


def get_confidence_badge(ci: ConfidenceInterval, probability: float) -> str:
    """Return badge string with emoji and text.

    Args:
        ci: Confidence interval with lower and upper bounds
        probability: Snowfall probability (0-1)

    Returns:
        Badge string like "Green Circle High Confidence"
    """
    level = get_confidence_level(ci, probability)
    badges = {
        "high": "🟢 High Confidence",
        "medium": "🟡 Medium Confidence",
        "low": "🔴 Low Confidence",
    }
    return badges[level]


def get_confidence_color(level: str) -> str:
    """Get CSS color for confidence level.

    Args:
        level: Confidence level ('high', 'medium', 'low')

    Returns:
        CSS color hex code
    """
    colors = {
        "high": "#22c55e",   # Green
        "medium": "#eab308",  # Yellow
        "low": "#ef4444",     # Red
    }
    return colors.get(level, "#6b7280")  # Gray as fallback


def format_forecast_with_ci(value: float, ci: ConfidenceInterval) -> str:
    """Format forecast value with confidence interval as margin of error.

    Args:
        value: Forecast value in cm
        ci: Confidence interval with lower and upper bounds

    Returns:
        Formatted string like '15 +/- 5 cm'
    """
    margin = (ci.upper - ci.lower) / 2
    return f"{value:.0f} +/- {margin:.0f} cm"


def render_confidence_badge(
    ci: ConfidenceInterval,
    probability: float,
    container: Optional["st.container"] = None
) -> None:
    """Render confidence badge in Streamlit.

    Displays a colored badge indicating the confidence level
    with appropriate styling.

    Args:
        ci: Confidence interval with lower and upper bounds
        probability: Snowfall probability (0-1)
        container: Optional Streamlit container to render in
    """
    if not HAS_STREAMLIT:
        return

    level = get_confidence_level(ci, probability)
    badge_text = get_confidence_badge(ci, probability)
    color = get_confidence_color(level)

    badge_html = f"""
    <div style="
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        background-color: {color}20;
        border: 1px solid {color};
        color: {color};
        font-weight: 500;
        font-size: 0.875rem;
    ">
        {badge_text}
    </div>
    """

    target = container if container else st
    target.markdown(badge_html, unsafe_allow_html=True)


def render_confidence_explanation(container: Optional["st.container"] = None) -> None:
    """Render help text explaining confidence levels.

    Displays an expandable section with explanations of what
    each confidence level means and how it's calculated.

    Args:
        container: Optional Streamlit container to render in
    """
    if not HAS_STREAMLIT:
        return

    target = container if container else st

    with target.expander("What does confidence level mean?"):
        st.markdown("""
**Confidence levels indicate forecast reliability:**

- **🟢 High Confidence**: Narrow uncertainty range (<5cm) with high probability (>70%).
  The forecast is well-constrained by model data.

- **🟡 Medium Confidence**: Moderate uncertainty range (<15cm) with reasonable probability (>40%).
  Consider the full range when planning.

- **🔴 Low Confidence**: Wide uncertainty or low probability.
  Conditions are hard to predict; check back for updates.

**How it's calculated:**
- Uses the 95% confidence interval width from the ensemble model
- Combined with the snowfall probability estimate
- Both factors must meet thresholds for higher confidence
        """)


def render_forecast_with_confidence(
    value: float,
    ci: ConfidenceInterval,
    probability: float,
    label: str = "Forecast",
    container: Optional["st.container"] = None
) -> None:
    """Render a complete forecast display with confidence information.

    Shows the forecast value, confidence interval, badge, and
    optional explanation.

    Args:
        value: Forecast value in cm
        ci: Confidence interval with lower and upper bounds
        probability: Snowfall probability (0-1)
        label: Label for the forecast (e.g., "New Snow", "Snow Depth")
        container: Optional Streamlit container to render in
    """
    if not HAS_STREAMLIT:
        return

    target = container if container else st

    level = get_confidence_level(ci, probability)
    color = get_confidence_color(level)
    formatted = format_forecast_with_ci(value, ci)

    col1, col2 = target.columns([2, 1])

    with col1:
        st.markdown(f"**{label}**")
        st.markdown(
            f"<span style='font-size: 1.5rem; color: {color};'>{formatted}</span>",
            unsafe_allow_html=True
        )
        st.caption(f"Range: {ci.lower:.0f} - {ci.upper:.0f} cm")

    with col2:
        render_confidence_badge(ci, probability, container=st)
