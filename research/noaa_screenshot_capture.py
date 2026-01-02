#!/usr/bin/env python3
"""
NOAA Weather Visualization Research Script
Captures screenshots and analyzes visualization patterns from weather.gov and NOAA
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

SCREENSHOT_DIR = Path("/Users/patrickkavanagh/snowforecast/research/screenshots/noaa")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Target URLs for NOAA snow and weather visualizations
URLS = {
    # National Weather Service Snow Forecasts
    "nws_national_snow": "https://www.weather.gov/forecastmaps",
    "nws_snow_ice": "https://www.weather.gov/crp/forecast",

    # National Blend of Models (NBM) - Primary forecast product
    "nbm_snow": "https://www.weather.gov/mdl/nbm_text",

    # Weather Prediction Center (WPC) - Official snow forecasts
    "wpc_snow_forecast": "https://www.wpc.ncep.noaa.gov/wwd/winter_wx.shtml",
    "wpc_qpf": "https://www.wpc.ncep.noaa.gov/qpf/qpf2.shtml",
    "wpc_5day_snow": "https://www.wpc.ncep.noaa.gov/pwpf/wwd_accum_probs.php",
    "wpc_snow_prob": "https://www.wpc.ncep.noaa.gov/pwpf/pwpf.php",

    # National Operational Hydrologic Remote Sensing Center (NOHRSC)
    "nohrsc_snow_analysis": "https://www.nohrsc.noaa.gov/nsa/",
    "nohrsc_interactive": "https://www.nohrsc.noaa.gov/interactive/html/map.html",

    # NOAA GIS/Mapping Services
    "noaa_nowdata": "https://www.weather.gov/source/crh/lsr/lsrmap.html",

    # Climate Prediction Center (CPC) - Extended outlooks
    "cpc_outlook": "https://www.cpc.ncep.noaa.gov/",

    # NOAA National Centers for Environmental Information
    "ncei_climate": "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/",

    # GIS Weather Services
    "nws_gis": "https://www.weather.gov/gis/",

    # Aviation Weather (good snow visualization examples)
    "awc_surface": "https://www.aviationweather.gov/progchart",

    # NWS Enhanced Data Display
    "nws_graphical": "https://graphical.weather.gov/",
    "nws_ndfd_conus": "https://graphical.weather.gov/sectors/conus.php#tabs",
}


async def capture_screenshots():
    """Capture screenshots from NOAA visualization sites."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )

        results = {}

        for name, url in URLS.items():
            page = await context.new_page()
            try:
                print(f"Capturing: {name} -> {url}")
                await page.goto(url, timeout=30000, wait_until="networkidle")

                # Wait a bit for dynamic content
                await page.wait_for_timeout(2000)

                # Full page screenshot
                screenshot_path = SCREENSHOT_DIR / f"{name}.png"
                await page.screenshot(path=str(screenshot_path), full_page=True)

                # Get page title and any relevant metadata
                title = await page.title()

                # Try to find any legend/color scale elements
                legend_elements = await page.query_selector_all("img[src*='legend'], img[src*='scale'], .legend, #legend")
                legend_count = len(legend_elements)

                # Check for interactive map elements
                map_elements = await page.query_selector_all("canvas, svg, .leaflet-container, #map, .ol-viewport")
                has_interactive_map = len(map_elements) > 0

                results[name] = {
                    "url": url,
                    "title": title,
                    "screenshot": str(screenshot_path),
                    "has_legend": legend_count > 0,
                    "has_interactive_map": has_interactive_map,
                    "status": "success"
                }

                print(f"  Success: {title}")

            except Exception as e:
                print(f"  Failed: {str(e)[:100]}")
                results[name] = {
                    "url": url,
                    "status": "failed",
                    "error": str(e)[:200]
                }
            finally:
                await page.close()

        await browser.close()
        return results


async def capture_wpc_snow_products():
    """Capture detailed WPC snow forecast products with specific interactions."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080}
        )

        # WPC Snow Accumulation Products
        wpc_products = [
            # Day 1-3 Snow Accumulation
            ("wpc_day1_snow", "https://www.wpc.ncep.noaa.gov/wwd/day1_wwd_prob.gif"),
            ("wpc_day2_snow", "https://www.wpc.ncep.noaa.gov/wwd/day2_wwd_prob.gif"),
            ("wpc_day3_snow", "https://www.wpc.ncep.noaa.gov/wwd/day3_wwd_prob.gif"),

            # Snow Probability Products
            ("wpc_prob_4inch", "https://www.wpc.ncep.noaa.gov/pwpf/4inch_pwpf_prob.png"),
            ("wpc_prob_8inch", "https://www.wpc.ncep.noaa.gov/pwpf/8inch_pwpf_prob.png"),
            ("wpc_prob_12inch", "https://www.wpc.ncep.noaa.gov/pwpf/12inch_pwpf_prob.png"),
        ]

        for name, url in wpc_products:
            page = await context.new_page()
            try:
                print(f"Capturing WPC product: {name}")
                await page.goto(url, timeout=30000)
                await page.wait_for_timeout(1000)
                screenshot_path = SCREENSHOT_DIR / f"{name}.png"
                await page.screenshot(path=str(screenshot_path))
                print(f"  Saved: {screenshot_path}")
            except Exception as e:
                print(f"  Failed: {str(e)[:100]}")
            finally:
                await page.close()

        await browser.close()


async def capture_nohrsc_products():
    """Capture NOHRSC snow analysis products."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080}
        )

        # NOHRSC National Snow Analysis products
        nohrsc_products = [
            ("nohrsc_snow_depth", "https://www.nohrsc.noaa.gov/snow_model/images/full/National/nsm_depth/"),
            ("nohrsc_swe", "https://www.nohrsc.noaa.gov/snow_model/images/full/National/nsm_swe/"),
            ("nohrsc_snowfall_24hr", "https://www.nohrsc.noaa.gov/snow_model/images/full/National/nsm_snowfall/"),
        ]

        for name, base_url in nohrsc_products:
            page = await context.new_page()
            try:
                print(f"Capturing NOHRSC: {name}")
                # Try to get the index page first
                await page.goto(base_url, timeout=30000)
                await page.wait_for_timeout(1500)
                screenshot_path = SCREENSHOT_DIR / f"{name}_index.png"
                await page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"  Saved index: {screenshot_path}")
            except Exception as e:
                print(f"  Failed: {str(e)[:100]}")
            finally:
                await page.close()

        # Main NOHRSC interactive viewer
        page = await context.new_page()
        try:
            print("Capturing NOHRSC Interactive Map")
            await page.goto("https://www.nohrsc.noaa.gov/interactive/html/map.html", timeout=60000)
            await page.wait_for_timeout(5000)  # Wait for map to load
            screenshot_path = SCREENSHOT_DIR / "nohrsc_interactive_map.png"
            await page.screenshot(path=str(screenshot_path))
            print(f"  Saved: {screenshot_path}")
        except Exception as e:
            print(f"  Failed: {str(e)[:100]}")
        finally:
            await page.close()

        await browser.close()


async def capture_graphical_forecast():
    """Capture NWS Graphical Forecast pages."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080}
        )

        page = await context.new_page()
        try:
            # Graphical weather forecast main page
            print("Capturing NWS Graphical Forecast")
            await page.goto("https://graphical.weather.gov/sectors/conus.php", timeout=30000)
            await page.wait_for_timeout(3000)

            # Try to navigate to snow parameters if available
            # First capture the main page
            await page.screenshot(
                path=str(SCREENSHOT_DIR / "nws_graphical_main.png"),
                full_page=True
            )

            # Try to find and click snow-related options
            snow_links = await page.query_selector_all("a:has-text('Snow'), a:has-text('snow'), option:has-text('Snow')")
            if snow_links:
                print(f"  Found {len(snow_links)} snow-related elements")

        except Exception as e:
            print(f"  Failed: {str(e)[:100]}")
        finally:
            await page.close()

        await browser.close()


async def main():
    """Main entry point."""
    print("=" * 60)
    print("NOAA Weather Visualization Research")
    print("=" * 60)

    print("\n1. Capturing main NOAA pages...")
    results = await capture_screenshots()

    print("\n2. Capturing WPC snow products...")
    await capture_wpc_snow_products()

    print("\n3. Capturing NOHRSC products...")
    await capture_nohrsc_products()

    print("\n4. Capturing NWS Graphical Forecast...")
    await capture_graphical_forecast()

    print("\n" + "=" * 60)
    print("Screenshot capture complete!")
    print(f"Screenshots saved to: {SCREENSHOT_DIR}")
    print("=" * 60)

    # Summary
    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    print(f"\nMain pages: {success_count}/{len(results)} successful")

    return results


if __name__ == "__main__":
    asyncio.run(main())
