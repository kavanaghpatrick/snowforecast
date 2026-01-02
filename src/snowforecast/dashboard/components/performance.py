"""Performance optimization utilities for Snowforecast dashboard.

Provides timing, caching, lazy loading, and performance monitoring tools
to optimize dashboard performance.

Performance Targets:
- Page load: <3 seconds
- Time step switch: <1 second
- Resort selection: <500ms
"""

import logging
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

import streamlit as st

T = TypeVar("T")
logger = logging.getLogger(__name__)

# Performance targets (in seconds)
TARGET_PAGE_LOAD = 3.0
TARGET_TIME_SWITCH = 1.0
TARGET_RESORT_SELECT = 0.5


class PerformanceTimer:
    """Context manager for timing operations.

    Records elapsed time and stores metrics in session state for profiling.

    Example:
        with PerformanceTimer("load_forecasts") as timer:
            data = load_all_forecasts()
        print(f"Took {timer.elapsed:.3f}s")
    """

    def __init__(self, operation: str, log: bool = True):
        """Initialize timer.

        Args:
            operation: Name of the operation being timed
            log: Whether to log the elapsed time
        """
        self.operation = operation
        self.log = log
        self.start_time: float = 0
        self.elapsed: float = 0

    def __enter__(self) -> "PerformanceTimer":
        self.start_time = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.time() - self.start_time
        if self.log:
            logger.info(f"{self.operation}: {self.elapsed:.3f}s")
        # Store in session state for profiling
        _record_metric(self.operation, self.elapsed)


def _record_metric(operation: str, elapsed: float) -> None:
    """Record a performance metric in session state.

    Args:
        operation: Name of the operation
        elapsed: Time elapsed in seconds
    """
    try:
        if "performance_metrics" not in st.session_state:
            st.session_state.performance_metrics = []
        st.session_state.performance_metrics.append(
            {
                "operation": operation,
                "elapsed": elapsed,
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception:
        # Session state may not be available in all contexts
        pass


def timed(operation: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to time function execution.

    Args:
        operation: Name of the operation for logging

    Returns:
        Decorated function that records timing

    Example:
        @timed("fetch_weather")
        def fetch_weather_data():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with PerformanceTimer(operation):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def get_cached_predictor() -> Any:
    """Get or create cached predictor (singleton).

    Returns the CachedPredictor instance, creating it if needed.
    Uses session state for singleton pattern.

    Returns:
        CachedPredictor instance
    """
    if "predictor" not in st.session_state:
        from snowforecast.cache import CachedPredictor

        st.session_state["predictor"] = CachedPredictor()
    return st.session_state["predictor"]


def prefetch_forecasts(resorts: list[dict]) -> None:
    """Pre-fetch forecasts for all resorts in background.

    Loads forecasts for all resorts upfront to avoid delays
    when user selects different resorts.

    Args:
        resorts: List of resort dicts with 'name', 'lat', 'lon' keys
    """
    if "all_forecasts" in st.session_state:
        return  # Already fetched

    predictor = get_cached_predictor()
    forecasts = {}
    for resort in resorts:
        try:
            # Fetch 24-hour forecast
            forecasts[resort["name"]] = predictor.predict(
                resort["lat"], resort["lon"], datetime.now(), forecast_hours=24
            )
        except Exception as e:
            logger.warning(f"Failed to prefetch forecast for {resort['name']}: {e}")
            forecasts[resort["name"]] = None
    st.session_state["all_forecasts"] = forecasts


def lazy_load(component_key: str) -> Callable[[Callable[..., T]], Callable[..., Optional[T]]]:
    """Decorator for lazy loading - only renders when visible.

    Components decorated with this will only render if their
    visibility flag in session state is True.

    Args:
        component_key: Unique key for the component

    Returns:
        Decorated function that checks visibility before rendering

    Example:
        @lazy_load("forecast_chart")
        def render_forecast_chart():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
            # Check if component should be rendered
            visibility_key = f"visible_{component_key}"
            if st.session_state.get(visibility_key, True):
                return func(*args, **kwargs)
            return None

        return wrapper

    return decorator


def render_performance_metrics(container: Any = None) -> None:
    """Render performance debugging panel (dev only).

    Shows the last 10 performance metrics with color-coded timing.

    Args:
        container: Optional Streamlit container to render in
    """
    target = container or st

    if "performance_metrics" not in st.session_state:
        target.info("No performance metrics collected yet")
        return

    metrics = st.session_state["performance_metrics"][-10:]  # Last 10

    target.markdown("**Performance Metrics**")
    for m in metrics:
        color = "green" if m["elapsed"] < 1 else "red" if m["elapsed"] > 3 else "orange"
        target.markdown(f"- {m['operation']}: :{color}[{m['elapsed']:.3f}s]")


def clear_performance_metrics() -> None:
    """Clear collected performance metrics."""
    st.session_state["performance_metrics"] = []


def get_performance_metrics() -> list[dict]:
    """Get all collected performance metrics.

    Returns:
        List of metric dicts with 'operation', 'elapsed', 'timestamp' keys
    """
    return st.session_state.get("performance_metrics", [])


def check_performance_targets() -> dict[str, bool]:
    """Check if recent operations met performance targets.

    Returns:
        Dict mapping target names to whether they were met
    """
    metrics = get_performance_metrics()
    if not metrics:
        return {}

    results = {}
    for m in metrics:
        op = m["operation"].lower()
        elapsed = m["elapsed"]

        if "page" in op or "load" in op:
            results["page_load"] = elapsed < TARGET_PAGE_LOAD
        elif "switch" in op or "time" in op:
            results["time_switch"] = elapsed < TARGET_TIME_SWITCH
        elif "resort" in op or "select" in op:
            results["resort_select"] = elapsed < TARGET_RESORT_SELECT

    return results
