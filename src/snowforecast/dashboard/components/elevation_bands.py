"""Elevation band forecast display component.

Renders a table showing snow forecasts at Base/Mid/Summit elevations,
similar to Mountain-Forecast.com format.
"""

import streamlit as st

from snowforecast.cache.elevation_bands import (
    ElevationBandResult,
    PrecipType,
)


def get_precip_emoji(precip_type: PrecipType) -> str:
    """Get display emoji for precipitation type.

    Args:
        precip_type: PrecipType enum value

    Returns:
        Emoji string for display
    """
    emoji_map = {
        PrecipType.SNOW: "Snow",
        PrecipType.MIXED: "Mixed",
        PrecipType.RAIN: "Rain",
        PrecipType.NONE: "-",
    }
    return emoji_map.get(precip_type, "-")


def format_snow_amount(cm: float) -> str:
    """Format snow amount for display.

    Args:
        cm: Snow amount in centimeters

    Returns:
        Formatted string
    """
    if cm <= 0:
        return "-"
    return f"{cm:.0f}cm"


def format_elevation(m: float) -> str:
    """Format elevation for display.

    Args:
        m: Elevation in meters

    Returns:
        Formatted string with both metric and imperial
    """
    feet = m * 3.28084
    return f"{m:.0f}m ({feet:.0f}ft)"


def render_elevation_bands(result: ElevationBandResult) -> None:
    """Render elevation band forecast table.

    Displays a table with Summit, Mid, Base rows showing:
    - Elevation
    - Temperature
    - New Snow
    - Precipitation Type

    Args:
        result: ElevationBandResult from compute_elevation_bands()
    """
    st.markdown("### Forecast by Elevation")
    st.caption(f"Snow line: {result.snow_line_m:.0f}m ({result.snow_line_m * 3.28084:.0f}ft)")

    # Build table data
    table_data = []
    for band in result.bands:  # Summit, Mid, Base order
        table_data.append({
            "Elevation": band.name,
            "Altitude": format_elevation(band.elevation_m),
            "Temp": f"{band.temp_c:.0f}C",
            "New Snow": format_snow_amount(band.new_snow_cm),
            "Type": get_precip_emoji(band.precip_type),
        })

    # Display as dataframe
    st.dataframe(
        table_data,
        width='stretch',
        hide_index=True,
        column_config={
            "Elevation": st.column_config.TextColumn("Band", width="small"),
            "Altitude": st.column_config.TextColumn("Altitude", width="medium"),
            "Temp": st.column_config.TextColumn("Temp", width="small"),
            "New Snow": st.column_config.TextColumn("New Snow", width="small"),
            "Type": st.column_config.TextColumn("Type", width="small"),
        },
    )


def render_elevation_bands_detailed(result: ElevationBandResult) -> None:
    """Render detailed elevation band forecast with metrics.

    Alternative display with Streamlit metrics for more visual impact.

    Args:
        result: ElevationBandResult from compute_elevation_bands()
    """
    st.markdown("### Forecast by Elevation")

    # Snow line indicator
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.metric("Snow Line", f"{result.snow_line_m:.0f}m")
    with col_info2:
        st.metric("Snow Line", f"{result.snow_line_m * 3.28084:.0f}ft")

    st.markdown("---")

    # Display each band
    for band in result.bands:
        with st.container():
            st.markdown(f"**{band.name}** - {format_elevation(band.elevation_m)}")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                # Temperature with color based on value
                st.metric("Temp", f"{band.temp_c:.0f}C")

            with col2:
                st.metric("New Snow", format_snow_amount(band.new_snow_cm))

            with col3:
                st.metric("Base Depth", format_snow_amount(band.snow_depth_cm))

            with col4:
                st.metric("Type", get_precip_emoji(band.precip_type))

            st.markdown("---")


def render_elevation_bands_compact(result: ElevationBandResult) -> None:
    """Render compact elevation band display.

    Single-line per band, suitable for sidebar or narrow layouts.

    Args:
        result: ElevationBandResult from compute_elevation_bands()
    """
    st.markdown("**Elevation Forecast**")
    st.caption(f"Snow line: {result.snow_line_m:.0f}m")

    for band in result.bands:
        snow_str = format_snow_amount(band.new_snow_cm)
        type_str = get_precip_emoji(band.precip_type)
        st.markdown(
            f"**{band.name}** ({band.elevation_m:.0f}m): "
            f"{band.temp_c:.0f}C | {snow_str} | {type_str}"
        )
