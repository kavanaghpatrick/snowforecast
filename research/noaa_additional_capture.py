#!/usr/bin/env python3
"""
Additional NOAA screenshot captures - focused on snow color scales and legends
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

SCREENSHOT_DIR = Path("/Users/patrickkavanagh/snowforecast/research/screenshots/noaa")

# Direct image URLs for snow products with color scales
DIRECT_IMAGES = [
    # WPC Snow Analysis Images
    ("wpc_snow_analysis_conus", "https://www.wpc.ncep.noaa.gov/wwd/day1_wwd_prod.gif"),
    ("wpc_snowfall_day1", "https://www.wpc.ncep.noaa.gov/nohrsc/snow_analy/images/nohrsc_snowfall_conus.png"),

    # NDFD Snow Amount Images
    ("ndfd_snow_6hr", "https://graphical.weather.gov/images/conus/SnowAmt6_conus.png"),
    ("ndfd_snow_total", "https://graphical.weather.gov/images/conus/TotalSnowamt_conus.png"),

    # NWS Graphical forecast images
    ("ndfd_pop12", "https://graphical.weather.gov/images/conus/PoP12_conus.png"),
    ("ndfd_wx_conus", "https://graphical.weather.gov/images/conus/Wx1_conus.png"),
]


async def capture_specific_snow_pages():
    """Capture specific snow forecast pages with detail."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1920, "height": 1200})

        # Capture direct image URLs
        for name, url in DIRECT_IMAGES:
            page = await context.new_page()
            try:
                print(f"Capturing: {name}")
                await page.goto(url, timeout=30000)
                await page.wait_for_timeout(1000)
                screenshot_path = SCREENSHOT_DIR / f"{name}.png"
                await page.screenshot(path=str(screenshot_path))
                print(f"  Saved: {screenshot_path}")
            except Exception as e:
                print(f"  Failed: {str(e)[:100]}")
            finally:
                await page.close()

        # Capture NOHRSC with snow depth layer selected
        page = await context.new_page()
        try:
            print("Capturing NOHRSC with snow depth layer...")
            await page.goto("https://www.nohrsc.noaa.gov/interactive/html/map.html", timeout=60000)
            await page.wait_for_timeout(5000)

            # Try to select snow depth layer
            try:
                await page.select_option('select[name="element"]', 'snow_depth', timeout=5000)
                await page.wait_for_timeout(2000)
            except:
                pass

            screenshot_path = SCREENSHOT_DIR / "nohrsc_snow_depth_layer.png"
            await page.screenshot(path=str(screenshot_path))
            print(f"  Saved: {screenshot_path}")
        except Exception as e:
            print(f"  Failed: {str(e)[:100]}")
        finally:
            await page.close()

        # Capture WPC winter weather page with scrolling to get legends
        page = await context.new_page()
        try:
            print("Capturing WPC Winter Weather with legends...")
            await page.goto("https://www.wpc.ncep.noaa.gov/wwd/winter_wx.shtml", timeout=30000)
            await page.wait_for_timeout(2000)

            # Scroll down to capture legend sections
            await page.evaluate("window.scrollBy(0, 400)")
            await page.wait_for_timeout(500)
            screenshot_path = SCREENSHOT_DIR / "wpc_winter_legends_scroll1.png"
            await page.screenshot(path=str(screenshot_path))

            await page.evaluate("window.scrollBy(0, 600)")
            await page.wait_for_timeout(500)
            screenshot_path = SCREENSHOT_DIR / "wpc_winter_legends_scroll2.png"
            await page.screenshot(path=str(screenshot_path))

            print("  Saved scrolled captures")
        except Exception as e:
            print(f"  Failed: {str(e)[:100]}")
        finally:
            await page.close()

        # Capture NDGD snow products page
        page = await context.new_page()
        try:
            print("Capturing NWS Digital Services snow page...")
            await page.goto("https://digital.weather.gov/", timeout=30000)
            await page.wait_for_timeout(3000)
            screenshot_path = SCREENSHOT_DIR / "nws_digital_services.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"  Saved: {screenshot_path}")
        except Exception as e:
            print(f"  Failed: {str(e)[:100]}")
        finally:
            await page.close()

        await browser.close()


async def main():
    print("Additional NOAA screenshot captures...")
    await capture_specific_snow_pages()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
