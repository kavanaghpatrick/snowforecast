"""Tests for favorites component with local storage.

Tests the JSON serialization logic and favorites management functions.
Since localStorage requires a browser context, we test the pure functions
that handle serialization/deserialization separately.
"""

import json
from unittest.mock import MagicMock, patch

from snowforecast.dashboard.components.favorites import (
    FAVORITES_KEY,
    add_to_favorites_list,
    check_is_favorite,
    parse_favorites_json,
    remove_from_favorites_list,
    serialize_favorites,
)


class TestParsesFavoritesJson:
    """Tests for JSON parsing logic."""

    def test_parse_empty_string(self):
        """Empty string returns empty list."""
        assert parse_favorites_json("") == []

    def test_parse_none(self):
        """None returns empty list."""
        assert parse_favorites_json(None) == []

    def test_parse_valid_json_list(self):
        """Valid JSON list is parsed correctly."""
        raw = '["Alta", "Snowbird", "Park City"]'
        result = parse_favorites_json(raw)
        assert result == ["Alta", "Snowbird", "Park City"]

    def test_parse_empty_json_list(self):
        """Empty JSON list returns empty list."""
        assert parse_favorites_json("[]") == []

    def test_parse_invalid_json(self):
        """Invalid JSON returns empty list."""
        assert parse_favorites_json("not valid json") == []
        assert parse_favorites_json("{broken") == []

    def test_parse_json_object_returns_empty(self):
        """JSON object (not list) returns empty list."""
        assert parse_favorites_json('{"name": "Alta"}') == []

    def test_parse_json_string_returns_empty(self):
        """JSON string (not list) returns empty list."""
        assert parse_favorites_json('"Alta"') == []

    def test_parse_json_number_returns_empty(self):
        """JSON number returns empty list."""
        assert parse_favorites_json("42") == []

    def test_parse_preserves_unicode(self):
        """Unicode characters are preserved."""
        raw = '["Chamonix-Mont-Blanc", "Zermatt"]'
        result = parse_favorites_json(raw)
        assert "Chamonix-Mont-Blanc" in result


class TestSerializeFavorites:
    """Tests for JSON serialization logic."""

    def test_serialize_empty_list(self):
        """Empty list serializes to '[]'."""
        assert serialize_favorites([]) == "[]"

    def test_serialize_single_item(self):
        """Single item list serializes correctly."""
        result = serialize_favorites(["Alta"])
        assert json.loads(result) == ["Alta"]

    def test_serialize_multiple_items(self):
        """Multiple items serialize correctly."""
        favs = ["Alta", "Snowbird", "Park City"]
        result = serialize_favorites(favs)
        assert json.loads(result) == favs

    def test_serialize_preserves_order(self):
        """Order is preserved in serialization."""
        favs = ["Zermatt", "Alta", "Mammoth"]
        result = serialize_favorites(favs)
        assert json.loads(result) == favs

    def test_roundtrip(self):
        """Serialize then parse returns original."""
        original = ["Alta", "Snowbird", "Park City"]
        serialized = serialize_favorites(original)
        parsed = parse_favorites_json(serialized)
        assert parsed == original


class TestAddToFavoritesList:
    """Tests for adding items to favorites."""

    def test_add_to_empty_list(self):
        """Adding to empty list creates single-item list."""
        result = add_to_favorites_list([], "Alta")
        assert result == ["Alta"]

    def test_add_new_item(self):
        """Adding new item appends to list."""
        result = add_to_favorites_list(["Snowbird"], "Alta")
        assert result == ["Snowbird", "Alta"]

    def test_add_duplicate_item(self):
        """Adding duplicate item doesn't create duplicate."""
        result = add_to_favorites_list(["Alta", "Snowbird"], "Alta")
        assert result == ["Alta", "Snowbird"]

    def test_add_does_not_mutate_original(self):
        """Original list is not mutated."""
        original = ["Alta"]
        result = add_to_favorites_list(original, "Snowbird")
        assert original == ["Alta"]  # unchanged
        assert result == ["Alta", "Snowbird"]


class TestRemoveFromFavoritesList:
    """Tests for removing items from favorites."""

    def test_remove_from_single_item_list(self):
        """Removing only item returns empty list."""
        result = remove_from_favorites_list(["Alta"], "Alta")
        assert result == []

    def test_remove_from_multiple_items(self):
        """Removing item from list preserves others."""
        result = remove_from_favorites_list(["Alta", "Snowbird", "Park City"], "Snowbird")
        assert result == ["Alta", "Park City"]

    def test_remove_nonexistent_item(self):
        """Removing nonexistent item returns unchanged list."""
        result = remove_from_favorites_list(["Alta"], "Mammoth")
        assert result == ["Alta"]

    def test_remove_from_empty_list(self):
        """Removing from empty list returns empty list."""
        result = remove_from_favorites_list([], "Alta")
        assert result == []

    def test_remove_does_not_mutate_original(self):
        """Original list is not mutated."""
        original = ["Alta", "Snowbird"]
        result = remove_from_favorites_list(original, "Alta")
        assert original == ["Alta", "Snowbird"]  # unchanged
        assert result == ["Snowbird"]


class TestCheckIsFavorite:
    """Tests for checking favorite status."""

    def test_is_favorite_true(self):
        """Returns True when resort is in list."""
        assert check_is_favorite(["Alta", "Snowbird"], "Alta") is True

    def test_is_favorite_false(self):
        """Returns False when resort not in list."""
        assert check_is_favorite(["Alta", "Snowbird"], "Mammoth") is False

    def test_is_favorite_empty_list(self):
        """Returns False for empty list."""
        assert check_is_favorite([], "Alta") is False

    def test_is_favorite_case_sensitive(self):
        """Resort name matching is case-sensitive."""
        assert check_is_favorite(["Alta"], "alta") is False
        assert check_is_favorite(["Alta"], "ALTA") is False


class TestFavoritesKey:
    """Tests for the storage key constant."""

    def test_key_is_string(self):
        """Storage key is a string."""
        assert isinstance(FAVORITES_KEY, str)

    def test_key_is_not_empty(self):
        """Storage key is not empty."""
        assert len(FAVORITES_KEY) > 0

    def test_key_has_namespace(self):
        """Storage key has snowforecast namespace."""
        assert "snowforecast" in FAVORITES_KEY


class TestIntegrationScenarios:
    """Integration tests for typical usage patterns."""

    def test_add_remove_roundtrip(self):
        """Add then remove returns to original state."""
        favs = ["Alta"]
        favs = add_to_favorites_list(favs, "Snowbird")
        favs = remove_from_favorites_list(favs, "Snowbird")
        assert favs == ["Alta"]

    def test_typical_workflow(self):
        """Simulate typical user workflow."""
        # User starts with no favorites
        favs = parse_favorites_json(None)
        assert favs == []

        # User favorites Alta
        favs = add_to_favorites_list(favs, "Alta")
        assert check_is_favorite(favs, "Alta") is True

        # User favorites Snowbird
        favs = add_to_favorites_list(favs, "Snowbird")
        assert len(favs) == 2

        # Save and reload (simulating browser storage)
        stored = serialize_favorites(favs)
        reloaded = parse_favorites_json(stored)
        assert reloaded == ["Alta", "Snowbird"]

        # User unfavorites Alta
        favs = remove_from_favorites_list(reloaded, "Alta")
        assert check_is_favorite(favs, "Alta") is False
        assert check_is_favorite(favs, "Snowbird") is True

    def test_filter_resorts_by_favorites(self):
        """Filtering resorts by favorites."""
        all_resorts = ["Alta", "Snowbird", "Park City", "Mammoth", "Vail"]
        favorites = ["Alta", "Mammoth"]

        filtered = [r for r in all_resorts if check_is_favorite(favorites, r)]
        assert filtered == ["Alta", "Mammoth"]


class TestMockedStreamlitFunctions:
    """Tests for Streamlit-dependent functions using mocks."""

    @patch("snowforecast.dashboard.components.favorites.LocalStorage")
    @patch("snowforecast.dashboard.components.favorites.st")
    def test_get_storage_creates_once(self, mock_st, mock_local_storage):
        """get_storage creates LocalStorage only once."""
        # Import inside to use fresh mocks
        from snowforecast.dashboard.components.favorites import get_storage

        # Setup: session_state as a MagicMock that behaves like dict
        mock_session_state = MagicMock()
        mock_session_state.__contains__ = MagicMock(return_value=False)
        mock_st.session_state = mock_session_state

        # First call creates instance
        mock_instance = MagicMock()
        mock_local_storage.return_value = mock_instance
        result1 = get_storage()

        assert mock_local_storage.called
        assert result1 == mock_instance

    @patch("snowforecast.dashboard.components.favorites.get_storage")
    def test_get_favorites_parses_storage(self, mock_get_storage):
        """get_favorites retrieves and parses from storage."""
        from snowforecast.dashboard.components.favorites import get_favorites

        mock_storage = MagicMock()
        mock_storage.getItem.return_value = '["Alta", "Snowbird"]'
        mock_get_storage.return_value = mock_storage

        result = get_favorites()

        mock_storage.getItem.assert_called_once_with(FAVORITES_KEY)
        assert result == ["Alta", "Snowbird"]

    @patch("snowforecast.dashboard.components.favorites.get_storage")
    def test_get_favorites_handles_none(self, mock_get_storage):
        """get_favorites handles None from storage."""
        from snowforecast.dashboard.components.favorites import get_favorites

        mock_storage = MagicMock()
        mock_storage.getItem.return_value = None
        mock_get_storage.return_value = mock_storage

        result = get_favorites()
        assert result == []

    @patch("snowforecast.dashboard.components.favorites.get_storage")
    def test_save_favorites_serializes(self, mock_get_storage):
        """save_favorites serializes and stores."""
        from snowforecast.dashboard.components.favorites import save_favorites

        mock_storage = MagicMock()
        mock_get_storage.return_value = mock_storage

        save_favorites(["Alta", "Snowbird"])

        mock_storage.setItem.assert_called_once_with(
            itemKey=FAVORITES_KEY,
            itemValue='["Alta", "Snowbird"]'
        )

    @patch("snowforecast.dashboard.components.favorites.get_favorites")
    @patch("snowforecast.dashboard.components.favorites.save_favorites")
    def test_add_favorite_saves(self, mock_save, mock_get):
        """add_favorite retrieves, adds, and saves."""
        from snowforecast.dashboard.components.favorites import add_favorite

        mock_get.return_value = ["Alta"]

        add_favorite("Snowbird")

        mock_save.assert_called_once_with(["Alta", "Snowbird"])

    @patch("snowforecast.dashboard.components.favorites.get_favorites")
    @patch("snowforecast.dashboard.components.favorites.save_favorites")
    def test_add_favorite_skips_duplicate(self, mock_save, mock_get):
        """add_favorite doesn't add duplicate."""
        from snowforecast.dashboard.components.favorites import add_favorite

        mock_get.return_value = ["Alta", "Snowbird"]

        add_favorite("Alta")

        mock_save.assert_not_called()

    @patch("snowforecast.dashboard.components.favorites.get_favorites")
    @patch("snowforecast.dashboard.components.favorites.save_favorites")
    def test_remove_favorite_saves(self, mock_save, mock_get):
        """remove_favorite retrieves, removes, and saves."""
        from snowforecast.dashboard.components.favorites import remove_favorite

        mock_get.return_value = ["Alta", "Snowbird"]

        remove_favorite("Alta")

        mock_save.assert_called_once_with(["Snowbird"])

    @patch("snowforecast.dashboard.components.favorites.get_favorites")
    @patch("snowforecast.dashboard.components.favorites.save_favorites")
    def test_remove_favorite_skips_nonexistent(self, mock_save, mock_get):
        """remove_favorite doesn't save if item not present."""
        from snowforecast.dashboard.components.favorites import remove_favorite

        mock_get.return_value = ["Alta"]

        remove_favorite("Mammoth")

        mock_save.assert_not_called()

    @patch("snowforecast.dashboard.components.favorites.get_favorites")
    def test_is_favorite_checks_list(self, mock_get):
        """is_favorite checks against current favorites."""
        from snowforecast.dashboard.components.favorites import is_favorite

        mock_get.return_value = ["Alta", "Snowbird"]

        assert is_favorite("Alta") is True
        assert is_favorite("Mammoth") is False
