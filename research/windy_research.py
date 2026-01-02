#!/usr/bin/env python3
"""
Research Windy.com weather visualization patterns using Playwright.
Captures screenshots of key UI elements for design analysis.
"""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

SCREENSHOT_DIR = Path("/Users/patrickkavanagh/snowforecast/research/screenshots/windy")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

def capture_windy_screenshots():
    """Capture screenshots of Windy.com visualization elements."""

    with sync_playwright() as p:
        # Launch browser with larger viewport for better captures
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            device_scale_factor=2  # Retina quality
        )
        page = context.new_page()

        print("Navigating to Windy.com...")
        # Start with a mountain location (Colorado Rockies)
        page.goto("https://www.windy.com/39.739/-104.990?39.739,-104.990,8", wait_until="networkidle", timeout=60000)

        # Wait for map to fully load
        time.sleep(5)

        # Close any popups or modals
        try:
            page.click('[data-testid="close-button"]', timeout=3000)
        except:
            pass
        try:
            page.click('.close-button', timeout=2000)
        except:
            pass
        try:
            page.click('[class*="close"]', timeout=2000)
        except:
            pass

        time.sleep(2)

        # Screenshot 1: Main map view (default layer)
        print("Capturing main map view...")
        page.screenshot(path=str(SCREENSHOT_DIR / "01_main_map_view.png"), full_page=False)

        # Screenshot 2: Open layer menu and capture
        print("Capturing layer controls...")
        try:
            # Click on the layer picker (usually shows current layer name)
            layer_picker = page.locator('[data-plugin="picker"]').first
            if layer_picker.is_visible():
                layer_picker.click()
                time.sleep(1)
            else:
                # Try alternative selectors
                page.click('.overlay-select', timeout=3000)
        except:
            try:
                page.click('[class*="picker"]', timeout=3000)
            except:
                pass

        time.sleep(2)
        page.screenshot(path=str(SCREENSHOT_DIR / "02_layer_controls.png"), full_page=False)

        # Screenshot 3: Switch to Snow layer
        print("Switching to snow layer...")
        try:
            # Try clicking on snow-related layer options
            snow_buttons = page.locator('text=Snow').all()
            for btn in snow_buttons:
                if btn.is_visible():
                    btn.click()
                    break
        except:
            try:
                page.click('[data-layer="snow"]', timeout=3000)
            except:
                pass

        time.sleep(3)
        page.screenshot(path=str(SCREENSHOT_DIR / "03_snow_layer.png"), full_page=False)

        # Screenshot 4: Try to get precipitation layer
        print("Switching to precipitation layer...")
        try:
            page.click('text=Rain', timeout=2000)
        except:
            try:
                page.click('text=Precip', timeout=2000)
            except:
                try:
                    page.click('[data-layer="rain"]', timeout=2000)
                except:
                    pass

        time.sleep(3)
        page.screenshot(path=str(SCREENSHOT_DIR / "04_precipitation_layer.png"), full_page=False)

        # Screenshot 5: Zoom into mountain region for terrain detail
        print("Zooming into mountain terrain...")
        # Navigate to a specific mountain area (Aspen, CO)
        page.goto("https://www.windy.com/39.191/-106.817?39.191,-106.817,11", wait_until="networkidle", timeout=60000)
        time.sleep(4)
        page.screenshot(path=str(SCREENSHOT_DIR / "05_mountain_terrain.png"), full_page=False)

        # Screenshot 6: Capture the timeline/forecast controls at bottom
        print("Capturing timeline controls...")
        # The timeline is usually at the bottom of the screen
        page.screenshot(path=str(SCREENSHOT_DIR / "06_full_interface.png"), full_page=False)

        # Screenshot 7: Try to capture legend/scale
        print("Looking for legend/scale...")
        try:
            legend = page.locator('[class*="legend"]').first
            if legend.is_visible():
                legend.screenshot(path=str(SCREENSHOT_DIR / "07_legend.png"))
        except:
            pass

        # Screenshot 8: Try opening settings or more options
        print("Capturing additional UI elements...")
        try:
            page.click('[class*="settings"]', timeout=2000)
            time.sleep(1)
            page.screenshot(path=str(SCREENSHOT_DIR / "08_settings_menu.png"), full_page=False)
        except:
            pass

        # Screenshot 9: Navigate to Alps for different mountain viz
        print("Capturing European Alps view...")
        page.goto("https://www.windy.com/46.500/8.000?46.500,8.000,9", wait_until="networkidle", timeout=60000)
        time.sleep(4)
        page.screenshot(path=str(SCREENSHOT_DIR / "09_alps_view.png"), full_page=False)

        # Screenshot 10: Try new snow accumulation layer
        print("Looking for snow accumulation layer...")
        try:
            # Open layer picker again
            page.click('[data-plugin="picker"]', timeout=3000)
            time.sleep(1)
            # Look for accumulated snow or similar
            page.click('text=New snow', timeout=2000)
            time.sleep(3)
            page.screenshot(path=str(SCREENSHOT_DIR / "10_new_snow_layer.png"), full_page=False)
        except:
            pass

        # Capture page source for analysis
        print("Capturing page structure...")
        html_content = page.content()
        with open(SCREENSHOT_DIR / "page_structure.html", "w") as f:
            f.write(html_content)

        browser.close()
        print(f"\nScreenshots saved to: {SCREENSHOT_DIR}")
        print(f"Files captured: {list(SCREENSHOT_DIR.glob('*.png'))}")

if __name__ == "__main__":
    capture_windy_screenshots()
