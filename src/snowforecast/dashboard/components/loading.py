"""Loading states and error handling components for the dashboard.

Provides decorators and UI components for managing loading states,
error handling, and retry functionality in the Streamlit dashboard.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Callable, TypeVar, Any
from functools import wraps
import logging

T = TypeVar('T')

logger = logging.getLogger(__name__)


def with_loading(message: str = "Loading..."):
    """Decorator to show spinner during function execution.

    Args:
        message: Text to display in the spinner

    Returns:
        Decorator function

    Example:
        >>> @with_loading("Fetching forecast data...")
        ... def fetch_forecast(lat, lon):
        ...     return api.get_forecast(lat, lon)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with st.spinner(message):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def with_error_handling(fallback_message: str = "An error occurred"):
    """Decorator to catch errors and show user-friendly message.

    Args:
        fallback_message: Message to show when an error occurs

    Returns:
        Decorator function that catches exceptions and displays error

    Example:
        >>> @with_error_handling("Failed to load forecast")
        ... def risky_operation():
        ...     return api.call_that_might_fail()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{fallback_message}: {e}")
                st.error(f"{fallback_message}: {e}")
                return None
        return wrapper
    return decorator


def with_loading_and_error_handling(
    loading_message: str = "Loading...",
    error_message: str = "An error occurred",
):
    """Combined decorator for loading spinner and error handling.

    Args:
        loading_message: Text to display in spinner
        error_message: Message to show on error

    Returns:
        Decorator function combining both behaviors

    Example:
        >>> @with_loading_and_error_handling(
        ...     loading_message="Fetching HRRR data...",
        ...     error_message="Failed to fetch forecast"
        ... )
        ... def fetch_hrrr(lat, lon):
        ...     return hrrr_api.fetch(lat, lon)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            with st.spinner(loading_message):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"{error_message}: {e}")
                    st.error(f"{error_message}: {e}")
                    return None
        return wrapper
    return decorator


def render_loading_skeleton(
    height: int = 200,
    container=None,
    skeleton_type: str = "chart",
) -> None:
    """Render placeholder skeleton while content loads.

    Args:
        height: Height of the skeleton placeholder in pixels
        container: Streamlit container to render in (default: main area)
        skeleton_type: Type of skeleton ("chart", "table", "card", "text")

    Example:
        >>> placeholder = st.empty()
        >>> render_loading_skeleton(height=300, container=placeholder, skeleton_type="chart")
    """
    target = container if container is not None else st

    # Create visual skeleton based on type
    if skeleton_type == "chart":
        target.markdown(
            f"""
            <div style="
                background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
                background-size: 200% 100%;
                animation: shimmer 1.5s infinite;
                height: {height}px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #888;
            ">
                <span style="font-size: 14px;">Loading chart...</span>
            </div>
            <style>
                @keyframes shimmer {{
                    0% {{ background-position: 200% 0; }}
                    100% {{ background-position: -200% 0; }}
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    elif skeleton_type == "table":
        # Render table-like skeleton rows
        rows_html = "".join([
            f'<div style="height: 40px; background: #f0f0f0; margin: 4px 0; border-radius: 4px;"></div>'
            for _ in range(min(height // 40, 10))
        ])
        target.markdown(
            f"""
            <div style="padding: 10px;">
                <div style="height: 30px; background: #e0e0e0; border-radius: 4px; margin-bottom: 10px;"></div>
                {rows_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif skeleton_type == "card":
        target.markdown(
            f"""
            <div style="
                background: #f5f5f5;
                height: {height}px;
                border-radius: 12px;
                padding: 20px;
                display: flex;
                flex-direction: column;
                gap: 10px;
            ">
                <div style="height: 24px; width: 60%; background: #e0e0e0; border-radius: 4px;"></div>
                <div style="height: 48px; width: 40%; background: #e8e8e8; border-radius: 4px;"></div>
                <div style="height: 16px; width: 80%; background: #f0f0f0; border-radius: 4px;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:  # text
        lines_html = "".join([
            f'<div style="height: 16px; width: {80 - i*10}%; background: #e8e8e8; margin: 8px 0; border-radius: 4px;"></div>'
            for i in range(min(height // 24, 8))
        ])
        target.markdown(
            f"""
            <div style="padding: 10px;">
                {lines_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_retry_button(
    key: str,
    on_click: Callable = None,
    label: str = "Retry",
    container=None,
) -> bool:
    """Render retry button. Returns True if clicked.

    Args:
        key: Unique key for the button
        on_click: Optional callback function when clicked
        label: Button label text
        container: Streamlit container to render in

    Returns:
        True if button was clicked, False otherwise

    Example:
        >>> if render_retry_button("retry_forecast", on_click=clear_cache):
        ...     st.rerun()
    """
    target = container if container is not None else st

    clicked = target.button(
        f"🔄 {label}",
        key=f"retry_btn_{key}",
        use_container_width=False,
    )

    if clicked and on_click is not None:
        on_click()

    return clicked


def render_error_with_retry(
    error_message: str,
    key: str,
    on_retry: Callable = None,
    container=None,
) -> bool:
    """Render error message with retry button.

    Args:
        error_message: Error message to display
        key: Unique key for the retry button
        on_retry: Callback function when retry is clicked
        container: Streamlit container to render in

    Returns:
        True if retry button was clicked

    Example:
        >>> try:
        ...     data = fetch_data()
        ... except Exception as e:
        ...     if render_error_with_retry(str(e), "fetch_data"):
        ...         st.rerun()
    """
    target = container if container is not None else st

    target.error(f"Error: {error_message}")
    return render_retry_button(key, on_click=on_retry, container=target)


def render_empty_state(
    message: str = "No data available",
    icon: str = "info",
    suggestion: str = None,
    container=None,
) -> None:
    """Render empty state placeholder.

    Args:
        message: Main message to display
        icon: Icon type ("info", "warning", "error")
        suggestion: Optional suggestion text
        container: Streamlit container to render in

    Example:
        >>> if data.empty:
        ...     render_empty_state(
        ...         message="No forecasts found",
        ...         suggestion="Try selecting a different date range"
        ...     )
    """
    target = container if container is not None else st

    if icon == "warning":
        target.warning(message)
    elif icon == "error":
        target.error(message)
    else:
        target.info(message)

    if suggestion:
        target.caption(suggestion)
