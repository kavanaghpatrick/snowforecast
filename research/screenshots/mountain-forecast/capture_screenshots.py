#!/usr/bin/env python3
"""
Capture screenshots of Mountain-Forecast.com for research purposes.
"""

from playwright.sync_api import sync_playwright
import time
from pathlib import Path

SCREENSHOT_DIR = Path("/Users/patrickkavanagh/snowforecast/research/screenshots/mountain-forecast")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

def capture_mountain_forecast():
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)

        # Desktop viewport
        desktop_context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )
        desktop_page = desktop_context.new_page()

        # Mobile viewport
        mobile_context = browser.new_context(
            viewport={"width": 390, "height": 844},
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15",
            is_mobile=True
        )
        mobile_page = mobile_context.new_page()

        print("Navigating to Mount Rainier forecast page...")

        # Desktop: Main forecast page
        desktop_page.goto("https://www.mountain-forecast.com/peaks/Mount-Rainier/forecasts/4392", timeout=60000)
        desktop_page.wait_for_load_state("networkidle")
        time.sleep(2)

        # Screenshot 1: Full page desktop view
        desktop_page.screenshot(
            path=str(SCREENSHOT_DIR / "01_mount_rainier_full_desktop.png"),
            full_page=True
        )
        print("Captured: Full desktop page")

        # Screenshot 2: Above-the-fold view
        desktop_page.screenshot(
            path=str(SCREENSHOT_DIR / "02_mount_rainier_above_fold.png"),
            full_page=False
        )
        print("Captured: Above-the-fold view")

        # Screenshot 3: Focus on forecast table
        try:
            forecast_table = desktop_page.locator(".forecast-table, .forecast__table, table.forecast").first
            if forecast_table.is_visible():
                forecast_table.screenshot(path=str(SCREENSHOT_DIR / "03_forecast_table_element.png"))
                print("Captured: Forecast table element")
        except Exception as e:
            print(f"Could not capture forecast table element: {e}")

        # Screenshot 4: Elevation selector/tabs if present
        try:
            elevation_selector = desktop_page.locator(".elevation-picker, .elevation-selector, .altitude-tabs").first
            if elevation_selector.is_visible():
                elevation_selector.screenshot(path=str(SCREENSHOT_DIR / "04_elevation_selector.png"))
                print("Captured: Elevation selector")
        except Exception as e:
            print(f"Could not capture elevation selector: {e}")

        # Mobile: Same page
        print("\nNavigating to mobile view...")
        mobile_page.goto("https://www.mountain-forecast.com/peaks/Mount-Rainier/forecasts/4392", timeout=60000)
        mobile_page.wait_for_load_state("networkidle")
        time.sleep(2)

        # Screenshot 5: Mobile full page
        mobile_page.screenshot(
            path=str(SCREENSHOT_DIR / "05_mount_rainier_mobile_full.png"),
            full_page=True
        )
        print("Captured: Mobile full page")

        # Screenshot 6: Mobile above-the-fold
        mobile_page.screenshot(
            path=str(SCREENSHOT_DIR / "06_mount_rainier_mobile_viewport.png"),
            full_page=False
        )
        print("Captured: Mobile viewport")

        # Navigate to another peak for comparison
        print("\nNavigating to Whistler Blackcomb...")
        desktop_page.goto("https://www.mountain-forecast.com/peaks/Whistler-Mountain/forecasts/2182", timeout=60000)
        desktop_page.wait_for_load_state("networkidle")
        time.sleep(2)

        # Screenshot 7: Whistler full page
        desktop_page.screenshot(
            path=str(SCREENSHOT_DIR / "07_whistler_full_desktop.png"),
            full_page=True
        )
        print("Captured: Whistler full page")

        # Navigate to home page to see resort listings
        print("\nNavigating to homepage/search...")
        desktop_page.goto("https://www.mountain-forecast.com/", timeout=60000)
        desktop_page.wait_for_load_state("networkidle")
        time.sleep(2)

        # Screenshot 8: Homepage
        desktop_page.screenshot(
            path=str(SCREENSHOT_DIR / "08_homepage.png"),
            full_page=False
        )
        print("Captured: Homepage")

        # Try different elevation on Mount Rainier
        print("\nChecking different elevations for Mount Rainier...")

        # Try summit elevation
        desktop_page.goto("https://www.mountain-forecast.com/peaks/Mount-Rainier/forecasts/4392", timeout=60000)
        desktop_page.wait_for_load_state("networkidle")
        time.sleep(2)

        # Screenshot 9: Summit elevation
        desktop_page.screenshot(
            path=str(SCREENSHOT_DIR / "09_rainier_summit_4392m.png"),
            full_page=True
        )
        print("Captured: Rainier summit elevation")

        # Try base elevation
        desktop_page.goto("https://www.mountain-forecast.com/peaks/Mount-Rainier/forecasts/2000", timeout=60000)
        desktop_page.wait_for_load_state("networkidle")
        time.sleep(2)

        # Screenshot 10: Base elevation
        desktop_page.screenshot(
            path=str(SCREENSHOT_DIR / "10_rainier_base_2000m.png"),
            full_page=True
        )
        print("Captured: Rainier base elevation")

        # Try mid elevation
        desktop_page.goto("https://www.mountain-forecast.com/peaks/Mount-Rainier/forecasts/3000", timeout=60000)
        desktop_page.wait_for_load_state("networkidle")
        time.sleep(2)

        # Screenshot 11: Mid elevation
        desktop_page.screenshot(
            path=str(SCREENSHOT_DIR / "11_rainier_mid_3000m.png"),
            full_page=True
        )
        print("Captured: Rainier mid elevation")

        # Extract page content for analysis
        print("\nExtracting page structure...")

        # Get the forecast table HTML structure
        desktop_page.goto("https://www.mountain-forecast.com/peaks/Mount-Rainier/forecasts/4392", timeout=60000)
        desktop_page.wait_for_load_state("networkidle")
        time.sleep(2)

        # Get page content for analysis
        page_content = desktop_page.content()
        with open(SCREENSHOT_DIR / "page_structure.html", "w") as f:
            f.write(page_content)
        print("Saved: Page HTML structure")

        # Get all visible text for reference
        visible_text = desktop_page.inner_text("body")
        with open(SCREENSHOT_DIR / "visible_text.txt", "w") as f:
            f.write(visible_text)
        print("Saved: Visible page text")

        # Close browsers
        browser.close()
        print("\nAll screenshots captured successfully!")

if __name__ == "__main__":
    capture_mountain_forecast()
