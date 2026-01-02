#!/usr/bin/env python3
"""
Round 2: More specific pages and mobile views.
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

SCREENSHOT_DIR = Path("/Users/patrickkavanagh/snowforecast/research/screenshots/ski-apps")


async def capture_page(page, url: str, name: str, scroll_captures: bool = False):
    """Capture a page with optional scroll captures."""
    try:
        print(f"\nCapturing: {name}")
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(3)

        # Full page
        await page.screenshot(path=str(SCREENSHOT_DIR / f"{name}_full.png"), full_page=True)
        print(f"  [OK] {name}_full.png")

        # Viewport
        await page.screenshot(path=str(SCREENSHOT_DIR / f"{name}_viewport.png"))
        print(f"  [OK] {name}_viewport.png")

        title = await page.title()
        return {"success": True, "title": title, "url": url}

    except Exception as e:
        print(f"  [ERROR] {e}")
        return {"success": False, "error": str(e)}


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        # Desktop context
        desktop_ctx = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )
        desktop = await desktop_ctx.new_page()

        # Mobile context
        mobile_ctx = await browser.new_context(
            viewport={"width": 390, "height": 844},  # iPhone 14 Pro
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15"
        )
        mobile = await mobile_ctx.new_page()

        print("="*60)
        print("ROUND 2: ADDITIONAL PAGES + MOBILE VIEWS")
        print("="*60)

        # OnTheSnow - correct URLs
        pages = [
            ("https://www.onthesnow.com/colorado/skireport", "onthesnow_colorado_v2"),
            ("https://www.onthesnow.com/utah/skireport", "onthesnow_utah_v2"),
            ("https://www.onthesnow.com/california/skireport", "onthesnow_california_v2"),
        ]

        for url, name in pages:
            await capture_page(desktop, url, name)

        # OpenSnow additional pages
        opensnow_pages = [
            ("https://opensnow.com/location/mammoth", "opensnow_mammoth"),
            ("https://opensnow.com/location/vail", "opensnow_vail"),
            ("https://opensnow.com/location/jackson-hole", "opensnow_jackson"),
            ("https://opensnow.com/state/CA", "opensnow_state_ca"),
            ("https://opensnow.com/state/CO", "opensnow_state_co"),
        ]

        for url, name in opensnow_pages:
            await capture_page(desktop, url, name)

        # Snow-forecast.com additional pages
        snowforecast_pages = [
            ("https://www.snow-forecast.com/resorts/Vail/6day/mid", "snowforecast_vail"),
            ("https://www.snow-forecast.com/resorts/Jackson-Hole/6day/mid", "snowforecast_jackson"),
            ("https://www.snow-forecast.com/maps/dynamic/usa", "snowforecast_map_usa"),
        ]

        for url, name in snowforecast_pages:
            await capture_page(desktop, url, name)

        # Mobile captures for key sites
        print("\n" + "="*60)
        print("MOBILE CAPTURES")
        print("="*60)

        mobile_pages = [
            ("https://opensnow.com", "opensnow_mobile_home"),
            ("https://opensnow.com/dailysnow/colorado", "opensnow_mobile_daily"),
            ("https://opensnow.com/location/mammoth", "opensnow_mobile_resort"),
            ("https://www.snow-forecast.com", "snowforecast_mobile_home"),
            ("https://www.snow-forecast.com/resorts/Mammoth-Mountain/6day/mid", "snowforecast_mobile_mammoth"),
            ("https://www.onthesnow.com", "onthesnow_mobile_home"),
            ("https://snowbrains.com", "snowbrains_mobile_home"),
        ]

        for url, name in mobile_pages:
            await capture_page(mobile, url, name)

        await browser.close()

        print("\n" + "="*60)
        print("ROUND 2 COMPLETE")
        print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
