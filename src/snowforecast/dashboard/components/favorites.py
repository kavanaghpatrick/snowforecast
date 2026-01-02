"""Favorites management with browser local storage.

This module provides functions for managing user favorites without requiring
an account. Favorites are persisted in browser localStorage.

Usage:
    from snowforecast.dashboard.components.favorites import (
        get_favorites,
        add_favorite,
        remove_favorite,
        is_favorite,
        render_favorite_toggle,
        render_favorites_filter,
    )

    # In Streamlit app
    render_favorite_toggle("Alta")
    if render_favorites_filter():
        resorts = [r for r in resorts if is_favorite(r)]
"""

import json
from typing import Optional

import streamlit as st
from streamlit_local_storage import LocalStorage

# Storage key for favorites list
FAVORITES_KEY = "snowforecast_favorites"


def get_storage() -> LocalStorage:
    """Get LocalStorage instance (cached in session state).

    Returns:
        LocalStorage instance for browser storage operations.
    """
    if "local_storage" not in st.session_state:
        st.session_state.local_storage = LocalStorage()
    return st.session_state.local_storage


def get_favorites() -> list[str]:
    """Get list of favorite resort names from local storage.

    Returns:
        List of resort names marked as favorites. Empty list if none.
    """
    storage = get_storage()
    raw = storage.getItem(FAVORITES_KEY)
    if raw:
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return result
            return []
        except json.JSONDecodeError:
            return []
    return []


def save_favorites(favs: list[str]) -> None:
    """Save favorites list to local storage.

    Args:
        favs: List of resort names to save as favorites.
    """
    storage = get_storage()
    storage.setItem(
        itemKey=FAVORITES_KEY,
        itemValue=json.dumps(favs)
    )


def add_favorite(resort_name: str) -> None:
    """Add resort to favorites list.

    Args:
        resort_name: Name of resort to add.
    """
    favs = get_favorites()
    if resort_name not in favs:
        favs.append(resort_name)
        save_favorites(favs)


def remove_favorite(resort_name: str) -> None:
    """Remove resort from favorites list.

    Args:
        resort_name: Name of resort to remove.
    """
    favs = get_favorites()
    if resort_name in favs:
        favs.remove(resort_name)
        save_favorites(favs)


def is_favorite(resort_name: str) -> bool:
    """Check if resort is in favorites.

    Args:
        resort_name: Name of resort to check.

    Returns:
        True if resort is a favorite, False otherwise.
    """
    return resort_name in get_favorites()


def render_favorite_toggle(resort_name: str, key_suffix: str = "") -> bool:
    """Render star toggle button for favorite status.

    Displays a star icon that toggles the favorite status when clicked.
    Uses filled star for favorited, empty star for not favorited.

    Args:
        resort_name: Name of resort to toggle.
        key_suffix: Optional suffix for unique Streamlit widget key.

    Returns:
        Current favorite status (before any toggle action).
    """
    is_fav = is_favorite(resort_name)
    icon = "\u2B50" if is_fav else "\u2606"  # Filled star vs empty star

    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button(
            icon,
            key=f"fav_{resort_name}_{key_suffix}",
            help="Toggle favorite"
        ):
            if is_fav:
                remove_favorite(resort_name)
            else:
                add_favorite(resort_name)
            st.rerun()

    return is_fav


def render_favorites_filter() -> bool:
    """Render 'Show Favorites Only' checkbox filter.

    Returns:
        True if filter is active (show only favorites), False otherwise.
    """
    return st.checkbox("\u2B50 Show Favorites Only", key="filter_favorites")


# Pure functions for testing (no Streamlit dependency)
def parse_favorites_json(raw: Optional[str]) -> list[str]:
    """Parse favorites from JSON string.

    Args:
        raw: JSON string from localStorage, or None.

    Returns:
        List of resort names. Empty list on invalid input.
    """
    if not raw:
        return []
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
        return []
    except json.JSONDecodeError:
        return []


def serialize_favorites(favs: list[str]) -> str:
    """Serialize favorites list to JSON string.

    Args:
        favs: List of resort names.

    Returns:
        JSON string for localStorage.
    """
    return json.dumps(favs)


def add_to_favorites_list(favs: list[str], resort_name: str) -> list[str]:
    """Add resort to favorites list (pure function).

    Args:
        favs: Current favorites list.
        resort_name: Resort to add.

    Returns:
        New list with resort added (if not already present).
    """
    if resort_name not in favs:
        return favs + [resort_name]
    return favs


def remove_from_favorites_list(favs: list[str], resort_name: str) -> list[str]:
    """Remove resort from favorites list (pure function).

    Args:
        favs: Current favorites list.
        resort_name: Resort to remove.

    Returns:
        New list with resort removed.
    """
    return [r for r in favs if r != resort_name]


def check_is_favorite(favs: list[str], resort_name: str) -> bool:
    """Check if resort is in favorites list (pure function).

    Args:
        favs: Current favorites list.
        resort_name: Resort to check.

    Returns:
        True if resort is in favorites.
    """
    return resort_name in favs
