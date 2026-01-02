#!/usr/bin/env python3
"""
Enhanced Windy.com research - capture specific visualization elements.
"""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

SCREENSHOT_DIR = Path("/Users/patrickkavanagh/snowforecast/research/screenshots/windy")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

def capture_windy_details():
    """Capture detailed Windy.com UI elements."""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            device_scale_factor=2
        )
        page = context.new_page()

        # Navigate to Colorado mountains with snow cover layer
        print("Loading Windy with snow cover layer...")
        page.goto("https://www.windy.com/-Snow-cover-snowcover?snowcover,39.739,-104.990,8",
                  wait_until="networkidle", timeout=60000)
        time.sleep(5)

        # Close popups
        try:
            page.keyboard.press("Escape")
            time.sleep(1)
        except:
            pass

        page.screenshot(path=str(SCREENSHOT_DIR / "11_snow_cover_layer.png"), full_page=False)

        # Navigate to new snow layer
        print("Loading new snow layer...")
        page.goto("https://www.windy.com/-New-snow-newSnow?newSnow,39.739,-104.990,8",
                  wait_until="networkidle", timeout=60000)
        time.sleep(5)
        page.screenshot(path=str(SCREENSHOT_DIR / "12_new_snow_layer.png"), full_page=False)

        # Navigate to precipitation layer
        print("Loading precipitation layer...")
        page.goto("https://www.windy.com/-Rain-thunder-rain?rain,39.739,-104.990,8",
                  wait_until="networkidle", timeout=60000)
        time.sleep(5)
        page.screenshot(path=str(SCREENSHOT_DIR / "13_rain_thunder_layer.png"), full_page=False)

        # Navigate to temperature layer
        print("Loading temperature layer...")
        page.goto("https://www.windy.com/-Temperature-temp?temp,39.739,-104.990,8",
                  wait_until="networkidle", timeout=60000)
        time.sleep(5)
        page.screenshot(path=str(SCREENSHOT_DIR / "14_temperature_layer.png"), full_page=False)

        # Zoom into detailed mountain terrain (Aspen area)
        print("Loading detailed mountain view...")
        page.goto("https://www.windy.com/-Snow-cover-snowcover?snowcover,39.191,-106.817,12",
                  wait_until="networkidle", timeout=60000)
        time.sleep(5)
        page.screenshot(path=str(SCREENSHOT_DIR / "15_detailed_mountain_snow.png"), full_page=False)

        # Capture the right sidebar menu
        print("Capturing layer sidebar...")
        try:
            # Look for the layer menu on the right
            sidebar = page.locator('#menu-container').first
            if sidebar.is_visible():
                sidebar.screenshot(path=str(SCREENSHOT_DIR / "16_sidebar_menu.png"))
        except:
            pass

        # Try to capture the legend/scale bar at bottom left
        print("Looking for scale/legend elements...")
        try:
            legend = page.locator('.legend').first
            if legend.is_visible():
                legend.screenshot(path=str(SCREENSHOT_DIR / "17_color_legend.png"))
        except:
            pass

        # Capture timeline controls at bottom
        print("Capturing timeline details...")
        try:
            timeline = page.locator('#bottom').first
            if timeline.is_visible():
                timeline.screenshot(path=str(SCREENSHOT_DIR / "18_timeline_controls.png"))
        except:
            pass

        # Navigate to webcams/stations view
        print("Loading weather stations...")
        page.goto("https://www.windy.com/-Webcams-webcams?webcams,39.739,-104.990,8",
                  wait_until="networkidle", timeout=60000)
        time.sleep(4)
        page.screenshot(path=str(SCREENSHOT_DIR / "19_webcams_stations.png"), full_page=False)

        # Capture meteogram view (forecast chart)
        print("Looking for meteogram/forecast chart...")
        try:
            page.click('[data-plugin="detail"]', timeout=3000)
            time.sleep(2)
            page.screenshot(path=str(SCREENSHOT_DIR / "20_meteogram_forecast.png"), full_page=False)
        except:
            pass

        browser.close()
        print(f"\nAdditional screenshots saved to: {SCREENSHOT_DIR}")

if __name__ == "__main__":
    capture_windy_details()
