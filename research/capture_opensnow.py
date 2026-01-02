#!/usr/bin/env python3
"""
Capture screenshots and analyze OpenSnow website for UI/UX research.
"""

import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright

SCREENSHOT_DIR = Path("/Users/patrickkavanagh/snowforecast/research/screenshots/opensnow")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

async def capture_page(page, url: str, name: str, wait_time: int = 3000, full_page: bool = True):
    """Navigate to URL and capture screenshot."""
    print(f"Navigating to: {url}")
    try:
        await page.goto(url, wait_until="load", timeout=30000)
        await page.wait_for_timeout(wait_time)  # Extra wait for dynamic content

        screenshot_path = SCREENSHOT_DIR / f"{name}.png"
        await page.screenshot(path=str(screenshot_path), full_page=full_page)
        print(f"  Saved: {screenshot_path}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

async def analyze_page_structure(page):
    """Extract UI component structure from the page."""
    structure = await page.evaluate("""
        () => {
            const result = {
                headings: [],
                buttons: [],
                cards: [],
                maps: [],
                charts: []
            };

            // Get headings
            document.querySelectorAll('h1, h2, h3').forEach(h => {
                result.headings.push({tag: h.tagName, text: h.innerText?.slice(0, 100)});
            });

            // Get buttons
            document.querySelectorAll('button, [role="button"], .btn').forEach(b => {
                result.buttons.push(b.innerText?.slice(0, 50) || b.getAttribute('aria-label'));
            });

            // Look for card-like components
            document.querySelectorAll('[class*="card"], [class*="Card"]').forEach(c => {
                result.cards.push(c.className);
            });

            // Look for map containers
            document.querySelectorAll('[class*="map"], [class*="Map"], .mapboxgl-map, .leaflet-container').forEach(m => {
                result.maps.push(m.className);
            });

            // Look for chart elements
            document.querySelectorAll('canvas, svg[class*="chart"], [class*="chart"]').forEach(c => {
                result.charts.push(c.tagName + ':' + c.className);
            });

            return result;
        }
    """)
    return structure

async def extract_snow_legend_colors(page):
    """Try to extract snow amount color coding from legend."""
    colors = await page.evaluate("""
        () => {
            const colorMap = [];
            // Look for legend items with colors
            const legendItems = document.querySelectorAll('[class*="legend"] *, [class*="Legend"] *');
            legendItems.forEach(item => {
                const style = window.getComputedStyle(item);
                if (style.backgroundColor && style.backgroundColor !== 'rgba(0, 0, 0, 0)') {
                    colorMap.push({
                        text: item.innerText?.slice(0, 50),
                        bgColor: style.backgroundColor
                    });
                }
            });

            // Also check for colored spans/divs that might indicate snow amounts
            document.querySelectorAll('[style*="background"]').forEach(el => {
                const text = el.innerText?.trim();
                if (text && (text.includes('"') || text.includes('inch') || /\\d+/.test(text))) {
                    colorMap.push({
                        text: text.slice(0, 50),
                        bgColor: el.style.backgroundColor
                    });
                }
            });

            return colorMap.slice(0, 20);
        }
    """)
    return colors

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        findings = {
            "pages_captured": [],
            "page_structure": {},
            "color_schemes": {}
        }

        # 1. Homepage
        print("\n=== 1. Homepage ===")
        if await capture_page(page, "https://opensnow.com", "01_homepage", wait_time=5000, full_page=False):
            findings["pages_captured"].append("homepage")
            findings["page_structure"]["homepage"] = await analyze_page_structure(page)

        # Scroll and capture more
        await page.evaluate("window.scrollBy(0, 600)")
        await page.wait_for_timeout(2000)
        await page.screenshot(path=str(SCREENSHOT_DIR / "01b_homepage_scrolled.png"), full_page=False)

        # 2. Daily Snow Map
        print("\n=== 2. Daily Snow Map ===")
        if await capture_page(page, "https://opensnow.com/dailysnow", "02_dailysnow", wait_time=5000, full_page=False):
            findings["pages_captured"].append("dailysnow")
            findings["page_structure"]["dailysnow"] = await analyze_page_structure(page)

        # 3. Alta Resort
        print("\n=== 3. Alta Resort ===")
        if await capture_page(page, "https://opensnow.com/location/altaskiresort", "03_alta_resort", wait_time=5000):
            findings["pages_captured"].append("alta_resort")
            findings["page_structure"]["alta"] = await analyze_page_structure(page)
            findings["color_schemes"]["resort"] = await extract_snow_legend_colors(page)

        # Capture the forecast section specifically
        try:
            forecast_section = page.locator('[class*="forecast"], [class*="Forecast"]').first
            if await forecast_section.is_visible(timeout=3000):
                await forecast_section.screenshot(path=str(SCREENSHOT_DIR / "03b_alta_forecast_section.png"))
        except:
            pass

        # 4. Park City
        print("\n=== 4. Park City ===")
        await capture_page(page, "https://opensnow.com/location/parkcity", "04_parkcity", wait_time=5000)

        # 5. Utah Daily Snow (10-day view)
        print("\n=== 5. Utah 10-Day ===")
        if await capture_page(page, "https://opensnow.com/dailysnow/utah", "05_utah_10day", wait_time=5000):
            findings["pages_captured"].append("utah_10day")
            findings["page_structure"]["utah"] = await analyze_page_structure(page)

        # 6. Colorado State
        print("\n=== 6. Colorado State ===")
        if await capture_page(page, "https://opensnow.com/state/colorado", "06_colorado_state", wait_time=5000):
            findings["pages_captured"].append("colorado_state")

        # 7. Map Page
        print("\n=== 7. Map Page ===")
        if await capture_page(page, "https://opensnow.com/map", "07_map", wait_time=8000, full_page=False):
            findings["pages_captured"].append("map")
            findings["page_structure"]["map"] = await analyze_page_structure(page)
            findings["color_schemes"]["map_legend"] = await extract_snow_legend_colors(page)

        # Try to interact with map
        try:
            # Look for layer controls
            layer_btn = page.locator('button:has-text("Layers"), [aria-label*="layer"]').first
            if await layer_btn.is_visible(timeout=2000):
                await layer_btn.click()
                await page.wait_for_timeout(1500)
                await page.screenshot(path=str(SCREENSHOT_DIR / "07b_map_layers.png"), full_page=False)
        except:
            pass

        # 8. Powder Alerts
        print("\n=== 8. Powder Alerts ===")
        await capture_page(page, "https://opensnow.com/powder", "08_powder", wait_time=5000, full_page=False)

        # 9. All Locations/Browse
        print("\n=== 9. Browse Locations ===")
        await capture_page(page, "https://opensnow.com/browse", "09_browse", wait_time=5000, full_page=False)

        # 10. Try to capture a specific forecast detail
        print("\n=== 10. Mammoth Mountain ===")
        if await capture_page(page, "https://opensnow.com/location/mammothmountain", "10_mammoth", wait_time=5000):
            findings["pages_captured"].append("mammoth")

        # Scroll to see full 10-day
        await page.evaluate("window.scrollBy(0, 800)")
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(SCREENSHOT_DIR / "10b_mammoth_10day.png"), full_page=False)

        # 11. Wyoming overview
        print("\n=== 11. Wyoming ===")
        await capture_page(page, "https://opensnow.com/state/wyoming", "11_wyoming", wait_time=4000, full_page=False)

        await browser.close()

        # Save findings
        with open(SCREENSHOT_DIR / "findings.json", "w") as f:
            json.dump(findings, f, indent=2, default=str)

        print(f"\n\nScreenshots saved to: {SCREENSHOT_DIR}")
        print(f"Pages captured: {len(findings['pages_captured'])}")

if __name__ == "__main__":
    asyncio.run(main())
