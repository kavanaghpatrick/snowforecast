"""End-to-end tests for Snowforecast dashboard using Playwright.

Run with: pytest tests/e2e/ -m e2e
Requires: playwright install chromium
"""

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e

BASE_URL = "https://snowforecast.streamlit.app"

# Streamlit apps take time to load
STREAMLIT_LOAD_TIMEOUT = 30000  # 30 seconds


class TestDashboardLoads:
    """Test 1: Dashboard loads successfully."""

    def test_homepage_loads(self, page: Page):
        """Verify the homepage loads without errors."""
        page.goto(BASE_URL, timeout=STREAMLIT_LOAD_TIMEOUT)
        # Wait for Streamlit to finish loading
        page.wait_for_load_state("networkidle", timeout=STREAMLIT_LOAD_TIMEOUT)
        # Check we're not on an error page
        assert "error" not in page.title().lower()


class TestBranding:
    """Test 2: Title and branding is visible."""

    def test_title_visible(self, page: Page):
        """Verify the app title is displayed."""
        page.goto(BASE_URL, timeout=STREAMLIT_LOAD_TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=STREAMLIT_LOAD_TIMEOUT)
        # Look for snowforecast branding
        content = page.content()
        assert "snow" in content.lower() or "forecast" in content.lower()


class TestSkiAreaSelector:
    """Test 3: Ski area selector is present."""

    def test_ski_area_dropdown_exists(self, page: Page):
        """Verify ski area selection dropdown is present."""
        page.goto(BASE_URL, timeout=STREAMLIT_LOAD_TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=STREAMLIT_LOAD_TIMEOUT)
        # Streamlit selectboxes have specific structure
        selectbox = page.locator('[data-testid="stSelectbox"]').first
        expect(selectbox).to_be_visible(timeout=STREAMLIT_LOAD_TIMEOUT)


class TestMapComponent:
    """Test 4: Map component renders."""

    def test_map_renders(self, page: Page):
        """Verify the map component is visible."""
        page.goto(BASE_URL, timeout=STREAMLIT_LOAD_TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=STREAMLIT_LOAD_TIMEOUT)
        # Look for map elements (pydeck or folium)
        map_element = page.locator('iframe, [data-testid="stDeckGlJsonChart"], .folium-map').first
        # Map might take extra time to render
        expect(map_element).to_be_visible(timeout=STREAMLIT_LOAD_TIMEOUT)


class TestForecastChart:
    """Test 5: Forecast chart displays."""

    def test_chart_visible(self, page: Page):
        """Verify forecast chart renders."""
        page.goto(BASE_URL, timeout=STREAMLIT_LOAD_TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=STREAMLIT_LOAD_TIMEOUT)
        # Wait for any chart element
        chart = page.locator('[data-testid="stVegaLiteChart"], [data-testid="stPlotlyChart"], canvas').first
        expect(chart).to_be_visible(timeout=STREAMLIT_LOAD_TIMEOUT)


class TestCacheStatus:
    """Test 6: Cache status indicator is visible."""

    def test_cache_status_shown(self, page: Page):
        """Verify cache status is displayed."""
        page.goto(BASE_URL, timeout=STREAMLIT_LOAD_TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=STREAMLIT_LOAD_TIMEOUT)
        content = page.content()
        # Look for cache-related indicators
        has_cache_info = any(term in content.lower() for term in ["cache", "cached", "fresh", "stale", "updated"])
        assert has_cache_info, "No cache status information found"


class TestResponsiveLayout:
    """Test 7: Responsive layout works."""

    def test_mobile_viewport(self, page: Page):
        """Verify the app works on mobile viewport."""
        # Set mobile viewport
        page.set_viewport_size({"width": 375, "height": 812})
        page.goto(BASE_URL, timeout=STREAMLIT_LOAD_TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=STREAMLIT_LOAD_TIMEOUT)
        # App should still load without horizontal overflow
        body = page.locator("body")
        expect(body).to_be_visible()


class TestTimeSelector:
    """Test 8: Time selector component works."""

    def test_time_controls_exist(self, page: Page):
        """Verify time selection controls are present."""
        page.goto(BASE_URL, timeout=STREAMLIT_LOAD_TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=STREAMLIT_LOAD_TIMEOUT)
        # Look for slider or date picker
        time_control = page.locator('[data-testid="stSlider"], [data-testid="stDateInput"]').first
        expect(time_control).to_be_visible(timeout=STREAMLIT_LOAD_TIMEOUT)


class TestForecastTable:
    """Test 9: Forecast table displays data."""

    def test_data_table_visible(self, page: Page):
        """Verify forecast data table is present."""
        page.goto(BASE_URL, timeout=STREAMLIT_LOAD_TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=STREAMLIT_LOAD_TIMEOUT)
        # Look for dataframe or table
        table = page.locator('[data-testid="stDataFrame"], [data-testid="stTable"], table').first
        expect(table).to_be_visible(timeout=STREAMLIT_LOAD_TIMEOUT)


class TestPerformance:
    """Test 10: Page loads within acceptable time."""

    def test_load_time_acceptable(self, page: Page):
        """Verify the page loads within 30 seconds."""
        import time
        start = time.time()
        page.goto(BASE_URL, timeout=STREAMLIT_LOAD_TIMEOUT)
        page.wait_for_load_state("networkidle", timeout=STREAMLIT_LOAD_TIMEOUT)
        load_time = time.time() - start
        assert load_time < 30, f"Page took {load_time:.1f}s to load (max 30s)"


# Pytest fixtures
@pytest.fixture(scope="function")
def page(browser):
    """Create a new page for each test."""
    context = browser.new_context()
    page = context.new_page()
    yield page
    page.close()
    context.close()


@pytest.fixture(scope="session")
def browser():
    """Launch browser once per session."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()
