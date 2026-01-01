"""Streamlit dashboard for Snow Forecast visualization.

Run with: streamlit run src/snowforecast/dashboard/app.py
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Try to import streamlit, provide helpful error if missing
try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Install with: pip install streamlit")
    sys.exit(1)

# Page configuration
st.set_page_config(
    page_title="Snow Forecast Dashboard",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def generate_sample_data(n_days: int = 30, n_stations: int = 10) -> pd.DataFrame:
    """Generate sample data for dashboard demo.

    Args:
        n_days: Number of days of data
        n_stations: Number of stations

    Returns:
        DataFrame with sample predictions and observations
    """
    np.random.seed(42)

    dates = pd.date_range(end=date.today(), periods=n_days, freq="D")
    stations = [f"Station_{i:02d}" for i in range(1, n_stations + 1)]

    # Station locations (Western US)
    station_locs = {
        "Station_01": (39.5, -106.0, "Colorado", 3200),
        "Station_02": (40.5, -111.5, "Utah", 2800),
        "Station_03": (46.8, -121.7, "Washington", 1800),
        "Station_04": (44.0, -121.5, "Oregon", 2100),
        "Station_05": (38.5, -119.5, "California", 2500),
        "Station_06": (45.5, -110.5, "Montana", 2400),
        "Station_07": (43.5, -110.8, "Wyoming", 2900),
        "Station_08": (36.5, -105.5, "New Mexico", 3100),
        "Station_09": (35.5, -111.5, "Arizona", 2800),
        "Station_10": (39.0, -114.5, "Nevada", 2600),
    }

    records = []
    for dt in dates:
        for station in stations:
            lat, lon, state, elev = station_locs[station]

            # Generate realistic snow data
            month = dt.month
            if month in (12, 1, 2):
                base_snow = 80 + np.random.normal(0, 20)
                new_snow = max(0, np.random.exponential(5))
            elif month in (3, 11):
                base_snow = 40 + np.random.normal(0, 15)
                new_snow = max(0, np.random.exponential(3))
            else:
                base_snow = max(0, 10 + np.random.normal(0, 10))
                new_snow = max(0, np.random.exponential(1))

            # Add prediction with some error
            pred_snow_depth = base_snow + np.random.normal(0, 5)
            pred_new_snow = new_snow + np.random.normal(0, 2)

            records.append({
                "date": dt,
                "station_id": station,
                "latitude": lat,
                "longitude": lon,
                "state": state,
                "elevation": elev,
                "observed_snow_depth": max(0, base_snow),
                "predicted_snow_depth": max(0, pred_snow_depth),
                "observed_new_snow": max(0, new_snow),
                "predicted_new_snow": max(0, pred_new_snow),
                "error_snow_depth": pred_snow_depth - base_snow,
                "error_new_snow": pred_new_snow - new_snow,
            })

    return pd.DataFrame(records)


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute performance metrics from data.

    Args:
        df: DataFrame with observed and predicted columns

    Returns:
        Dictionary of metrics
    """
    obs = df["observed_snow_depth"].values
    pred = df["predicted_snow_depth"].values
    errors = pred - obs

    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    bias = np.mean(errors)

    # F1 for snowfall events (>2.5cm)
    obs_event = df["observed_new_snow"] > 2.5
    pred_event = df["predicted_new_snow"] > 2.5
    tp = ((obs_event) & (pred_event)).sum()
    fp = ((~obs_event) & (pred_event)).sum()
    fn = ((obs_event) & (~pred_event)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "n_samples": len(df),
    }


def create_metric_cards(metrics: dict):
    """Create metric display cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="RMSE (cm)",
            value=f"{metrics['rmse']:.1f}",
            delta=f"Target: <15",
            delta_color="inverse" if metrics['rmse'] > 15 else "normal",
        )

    with col2:
        st.metric(
            label="MAE (cm)",
            value=f"{metrics['mae']:.1f}",
        )

    with col3:
        st.metric(
            label="Bias (cm)",
            value=f"{metrics['bias']:+.1f}",
            delta=f"Target: <5",
            delta_color="inverse" if abs(metrics['bias']) > 5 else "normal",
        )

    with col4:
        st.metric(
            label="F1 Score",
            value=f"{metrics['f1']:.1%}",
            delta=f"Target: >85%",
            delta_color="inverse" if metrics['f1'] < 0.85 else "normal",
        )


def create_map_view(df: pd.DataFrame):
    """Create map visualization of stations and predictions."""
    st.subheader("Station Map")

    # Get latest data per station
    latest = df.groupby("station_id").last().reset_index()

    # Create map data
    map_data = latest[["latitude", "longitude", "predicted_snow_depth"]].copy()
    map_data.columns = ["lat", "lon", "snow_depth"]

    st.map(map_data, size="snow_depth", color="#0000FF")

    st.caption("Circle size indicates predicted snow depth")


def create_time_series(df: pd.DataFrame, station_id: str):
    """Create time series plot for a station."""
    st.subheader(f"Time Series: {station_id}")

    station_df = df[df["station_id"] == station_id].sort_values("date")

    # Create plot data
    plot_data = station_df[["date", "observed_snow_depth", "predicted_snow_depth"]].copy()
    plot_data = plot_data.set_index("date")
    plot_data.columns = ["Observed", "Predicted"]

    st.line_chart(plot_data)


def create_error_distribution(df: pd.DataFrame):
    """Create error distribution histogram."""
    st.subheader("Error Distribution")

    errors = df["error_snow_depth"].values

    # Create histogram data
    hist_data = pd.DataFrame({"Error (cm)": errors})

    st.bar_chart(
        pd.cut(errors, bins=20).value_counts().sort_index(),
    )


def create_performance_by_month(df: pd.DataFrame):
    """Create performance breakdown by month."""
    st.subheader("Performance by Month")

    df["month"] = pd.to_datetime(df["date"]).dt.month

    monthly_metrics = []
    for month in range(1, 13):
        month_df = df[df["month"] == month]
        if len(month_df) > 0:
            metrics = compute_metrics(month_df)
            monthly_metrics.append({
                "Month": month,
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"],
                "Bias": metrics["bias"],
            })

    if monthly_metrics:
        monthly_df = pd.DataFrame(monthly_metrics)
        monthly_df = monthly_df.set_index("Month")
        st.bar_chart(monthly_df["RMSE"])


def create_sidebar(df: pd.DataFrame) -> tuple:
    """Create sidebar filters.

    Returns:
        Tuple of (filtered_df, selected_station)
    """
    st.sidebar.header("Filters")

    # Date range
    min_date = df["date"].min()
    max_date = df["date"].max()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # State filter
    states = ["All"] + sorted(df["state"].unique().tolist())
    selected_state = st.sidebar.selectbox("State", states)

    # Station filter
    if selected_state != "All":
        available_stations = df[df["state"] == selected_state]["station_id"].unique()
    else:
        available_stations = df["station_id"].unique()

    stations = ["All"] + sorted(available_stations.tolist())
    selected_station = st.sidebar.selectbox("Station", stations)

    # Filter data
    filtered_df = df.copy()

    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["date"] >= pd.Timestamp(date_range[0])) &
            (filtered_df["date"] <= pd.Timestamp(date_range[1]))
        ]

    if selected_state != "All":
        filtered_df = filtered_df[filtered_df["state"] == selected_state]

    if selected_station != "All":
        filtered_df = filtered_df[filtered_df["station_id"] == selected_station]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Showing {len(filtered_df):,} records**")

    return filtered_df, selected_station if selected_station != "All" else None


def main():
    """Main dashboard function."""
    st.title("❄️ Snow Forecast Dashboard")
    st.markdown("Interactive visualization of snow predictions and model performance")

    # Load or generate data
    df = generate_sample_data(n_days=90, n_stations=10)

    # Create sidebar and get filters
    filtered_df, selected_station = create_sidebar(df)

    # Compute metrics
    metrics = compute_metrics(filtered_df)

    # Display metric cards
    st.header("Performance Metrics")
    create_metric_cards(metrics)

    st.markdown("---")

    # Create layout
    col1, col2 = st.columns(2)

    with col1:
        create_map_view(filtered_df)

    with col2:
        if selected_station:
            create_time_series(df, selected_station)
        else:
            st.subheader("Station Time Series")
            st.info("Select a station from the sidebar to view time series")

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        create_error_distribution(filtered_df)

    with col4:
        create_performance_by_month(df)

    # Footer
    st.markdown("---")
    st.markdown(
        "*Dashboard powered by Streamlit | "
        "[Documentation](/docs) | "
        "[API](/docs)*"
    )


def run_dashboard():
    """Entry point for running dashboard."""
    main()


if __name__ == "__main__":
    main()
