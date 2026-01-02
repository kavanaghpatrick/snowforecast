"""Tests for responsive layout utilities."""

import pytest
from unittest.mock import MagicMock, patch


class TestGetViewportWidth:
    """Tests for get_viewport_width function."""

    def test_returns_default_when_not_set(self):
        """Should return 1200 when viewport_width not in session state."""
        with patch('streamlit.session_state', {}):
            from snowforecast.dashboard.components.responsive import get_viewport_width
            # Need to reload to pick up mocked session_state
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.get_viewport_width() == 1200

    def test_returns_session_state_value(self):
        """Should return value from session state when set."""
        mock_session = {'viewport_width': 500}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.get_viewport_width() == 500


class TestGetBreakpoint:
    """Tests for get_breakpoint function."""

    @pytest.mark.parametrize("width,expected", [
        (320, "mobile"),   # Small mobile
        (375, "mobile"),   # iPhone
        (414, "mobile"),   # iPhone Plus
        (767, "mobile"),   # Just under mobile max
        (768, "tablet"),   # Tablet min
        (800, "tablet"),   # Mid tablet
        (1023, "tablet"),  # Just under tablet max
        (1024, "desktop"), # Desktop min
        (1200, "desktop"), # Common desktop
        (1920, "desktop"), # Full HD
    ])
    def test_breakpoint_ranges(self, width, expected):
        """Should return correct breakpoint for width ranges."""
        mock_session = {'viewport_width': width}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.get_breakpoint() == expected


class TestIsMobile:
    """Tests for is_mobile function."""

    def test_returns_true_for_mobile_width(self):
        """Should return True when viewport is mobile size."""
        mock_session = {'viewport_width': 375}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.is_mobile() is True

    def test_returns_false_for_tablet_width(self):
        """Should return False when viewport is tablet size."""
        mock_session = {'viewport_width': 800}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.is_mobile() is False

    def test_returns_false_for_desktop_width(self):
        """Should return False when viewport is desktop size."""
        mock_session = {'viewport_width': 1200}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.is_mobile() is False


class TestIsTablet:
    """Tests for is_tablet function."""

    def test_returns_false_for_mobile_width(self):
        """Should return False when viewport is mobile size."""
        mock_session = {'viewport_width': 375}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.is_tablet() is False

    def test_returns_true_for_tablet_width(self):
        """Should return True when viewport is tablet size."""
        mock_session = {'viewport_width': 800}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.is_tablet() is True

    def test_returns_false_for_desktop_width(self):
        """Should return False when viewport is desktop size."""
        mock_session = {'viewport_width': 1200}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.is_tablet() is False


class TestIsDesktop:
    """Tests for is_desktop function."""

    def test_returns_false_for_mobile_width(self):
        """Should return False when viewport is mobile size."""
        mock_session = {'viewport_width': 375}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.is_desktop() is False

    def test_returns_false_for_tablet_width(self):
        """Should return False when viewport is tablet size."""
        mock_session = {'viewport_width': 800}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.is_desktop() is False

    def test_returns_true_for_desktop_width(self):
        """Should return True when viewport is desktop size."""
        mock_session = {'viewport_width': 1200}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.is_desktop() is True


class TestGetColumnRatio:
    """Tests for get_column_ratio function."""

    def test_desktop_ratio(self):
        """Should return 2:1 ratio for desktop."""
        mock_session = {'viewport_width': 1200}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.get_column_ratio() == (2, 1)

    def test_tablet_ratio(self):
        """Should return 3:2 ratio for tablet."""
        mock_session = {'viewport_width': 800}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.get_column_ratio() == (3, 2)

    def test_mobile_ratio(self):
        """Should return 1:1 ratio for mobile (stacked)."""
        mock_session = {'viewport_width': 375}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.get_column_ratio() == (1, 1)


class TestShouldShow3d:
    """Tests for should_show_3d function."""

    def test_disabled_on_mobile(self):
        """Should return False on mobile for performance."""
        mock_session = {'viewport_width': 375}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.should_show_3d() is False

    def test_enabled_on_tablet(self):
        """Should return True on tablet."""
        mock_session = {'viewport_width': 800}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.should_show_3d() is True

    def test_enabled_on_desktop(self):
        """Should return True on desktop."""
        mock_session = {'viewport_width': 1200}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.should_show_3d() is True


class TestGetTouchTargetSize:
    """Tests for get_touch_target_size function."""

    def test_mobile_touch_target(self):
        """Should return 44px on mobile (Apple HIG guideline)."""
        mock_session = {'viewport_width': 375}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.get_touch_target_size() == 44

    def test_tablet_touch_target(self):
        """Should return 32px on tablet."""
        mock_session = {'viewport_width': 800}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.get_touch_target_size() == 32

    def test_desktop_touch_target(self):
        """Should return 32px on desktop."""
        mock_session = {'viewport_width': 1200}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            assert responsive.get_touch_target_size() == 32


class TestRenderResponsiveColumns:
    """Tests for render_responsive_columns function."""

    def test_returns_none_on_mobile(self):
        """Should return None on mobile to signal stacked layout."""
        mock_session = {'viewport_width': 375}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            with patch.object(responsive.st, 'columns') as mock_columns:
                result = responsive.render_responsive_columns()
                assert result is None
                mock_columns.assert_not_called()

    def test_returns_columns_on_tablet(self):
        """Should return st.columns with 3:2 ratio on tablet."""
        mock_session = {'viewport_width': 800}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            mock_cols = MagicMock()
            with patch.object(responsive.st, 'columns', return_value=mock_cols) as mock_columns:
                result = responsive.render_responsive_columns()
                mock_columns.assert_called_once_with((3, 2))
                assert result == mock_cols

    def test_returns_columns_on_desktop(self):
        """Should return st.columns with 2:1 ratio on desktop."""
        mock_session = {'viewport_width': 1200}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            mock_cols = MagicMock()
            with patch.object(responsive.st, 'columns', return_value=mock_cols) as mock_columns:
                result = responsive.render_responsive_columns()
                mock_columns.assert_called_once_with((2, 1))
                assert result == mock_cols


class TestInjectResponsiveCss:
    """Tests for inject_responsive_css function."""

    def test_injects_css(self):
        """Should call st.markdown with CSS styles."""
        import importlib
        import snowforecast.dashboard.components.responsive as responsive
        importlib.reload(responsive)

        with patch.object(responsive.st, 'markdown') as mock_markdown:
            responsive.inject_responsive_css()
            mock_markdown.assert_called_once()
            call_args = mock_markdown.call_args
            assert 'unsafe_allow_html' in call_args.kwargs
            assert call_args.kwargs['unsafe_allow_html'] is True
            # Check CSS content includes key media queries
            css_content = call_args.args[0]
            assert '@media (max-width: 768px)' in css_content
            assert 'min-height: 44px' in css_content


class TestSetViewportWidth:
    """Tests for set_viewport_width function."""

    def test_sets_session_state(self):
        """Should set viewport_width in session state."""
        mock_session = {}
        with patch('streamlit.session_state', mock_session):
            import importlib
            import snowforecast.dashboard.components.responsive as responsive
            importlib.reload(responsive)

            responsive.set_viewport_width(1024)
            assert mock_session['viewport_width'] == 1024


class TestConstants:
    """Tests for module constants."""

    def test_mobile_max(self):
        """MOBILE_MAX should be 768."""
        from snowforecast.dashboard.components.responsive import MOBILE_MAX
        assert MOBILE_MAX == 768

    def test_tablet_max(self):
        """TABLET_MAX should be 1024."""
        from snowforecast.dashboard.components.responsive import TABLET_MAX
        assert TABLET_MAX == 1024
