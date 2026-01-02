#!/usr/bin/env python3
"""
Playwright script to research ski-focused weather sites and capture screenshots.
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

SCREENSHOT_DIR = Path("/Users/patrickkavanagh/snowforecast/research/screenshots/ski-apps")

async def capture_site(page, url: str, name: str, selectors: list[dict] = None):
    """Navigate to a site and capture screenshots."""
    screenshots = []

    try:
        print(f"\n{'='*60}")
        print(f"Researching: {name} ({url})")
        print('='*60)

        # Navigate with extended timeout
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(3)  # Let dynamic content load

        # Full page screenshot
        full_path = SCREENSHOT_DIR / f"{name}_full.png"
        await page.screenshot(path=str(full_path), full_page=True)
        screenshots.append(str(full_path))
        print(f"  [OK] Full page: {full_path.name}")

        # Above the fold screenshot
        viewport_path = SCREENSHOT_DIR / f"{name}_viewport.png"
        await page.screenshot(path=str(viewport_path), full_page=False)
        screenshots.append(str(viewport_path))
        print(f"  [OK] Viewport: {viewport_path.name}")

        # Capture specific selectors if provided
        if selectors:
            for i, sel_info in enumerate(selectors):
                selector = sel_info.get("selector")
                desc = sel_info.get("description", f"element_{i}")
                try:
                    element = await page.query_selector(selector)
                    if element:
                        elem_path = SCREENSHOT_DIR / f"{name}_{desc}.png"
                        await element.screenshot(path=str(elem_path))
                        screenshots.append(str(elem_path))
                        print(f"  [OK] {desc}: {elem_path.name}")
                except Exception as e:
                    print(f"  [SKIP] {desc}: {e}")

        # Get page content info
        title = await page.title()
        print(f"  Page title: {title}")

        return {
            "url": url,
            "name": name,
            "title": title,
            "screenshots": screenshots,
            "success": True,
            "error": None
        }

    except PlaywrightTimeout:
        print(f"  [ERROR] Timeout loading {url}")
        return {"url": url, "name": name, "success": False, "error": "Timeout", "screenshots": []}
    except Exception as e:
        print(f"  [ERROR] {e}")
        return {"url": url, "name": name, "success": False, "error": str(e), "screenshots": []}


async def research_snowbrains(page):
    """Research SnowBrains website."""
    results = []

    # Main page
    result = await capture_site(page, "https://snowbrains.com", "snowbrains_home")
    results.append(result)

    # Try to find snow reports
    try:
        await page.goto("https://snowbrains.com/category/snow-reports/", wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(2)
        result = await capture_site(page, page.url, "snowbrains_reports")
        results.append(result)
    except Exception as e:
        print(f"  Could not access snow reports: {e}")

    # Try daily snow totals
    try:
        await page.goto("https://snowbrains.com/daily-snow/", wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(2)
        result = await capture_site(page, page.url, "snowbrains_daily")
        results.append(result)
    except Exception as e:
        print(f"  Could not access daily snow: {e}")

    return results


async def research_onthesnow(page):
    """Research OnTheSnow website."""
    results = []

    # Main page
    result = await capture_site(page, "https://www.onthesnow.com", "onthesnow_home")
    results.append(result)

    # Snow reports - try multiple paths
    snow_pages = [
        ("https://www.onthesnow.com/united-states/snow-reports", "onthesnow_us_reports"),
        ("https://www.onthesnow.com/colorado/snow-reports", "onthesnow_colorado"),
        ("https://www.onthesnow.com/california/snow-reports", "onthesnow_california"),
        ("https://www.onthesnow.com/utah/snow-reports", "onthesnow_utah"),
    ]

    for url, name in snow_pages:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)
            result = await capture_site(page, page.url, name)
            results.append(result)
        except Exception as e:
            print(f"  Could not access {name}: {e}")

    return results


async def research_skicom(page):
    """Research Ski.com website."""
    results = []

    # Main page
    result = await capture_site(page, "https://www.ski.com", "skicom_home")
    results.append(result)

    # Snow reports
    snow_pages = [
        ("https://www.ski.com/snow-reports", "skicom_reports"),
        ("https://www.ski.com/conditions", "skicom_conditions"),
    ]

    for url, name in snow_pages:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)
            result = await capture_site(page, page.url, name)
            results.append(result)
        except Exception as e:
            print(f"  Could not access {name}: {e}")

    return results


async def research_powder_project(page):
    """Research Powder Project website."""
    results = []

    # Main page
    result = await capture_site(page, "https://powderproject.com", "powderproject_home")
    results.append(result)

    # Try common paths
    pages = [
        ("https://powderproject.com/forecasts", "powderproject_forecasts"),
        ("https://powderproject.com/snow", "powderproject_snow"),
        ("https://powderproject.com/resorts", "powderproject_resorts"),
    ]

    for url, name in pages:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)
            result = await capture_site(page, page.url, name)
            results.append(result)
        except Exception as e:
            print(f"  Could not access {name}: {e}")

    return results


async def research_additional_sites(page):
    """Research additional ski weather sites for comparison."""
    results = []

    additional_sites = [
        ("https://opensnow.com", "opensnow_home"),
        ("https://opensnow.com/dailysnow/tahoe", "opensnow_tahoe"),
        ("https://opensnow.com/dailysnow/colorado", "opensnow_colorado"),
        ("https://www.snow-forecast.com", "snowforecast_home"),
        ("https://www.snow-forecast.com/resorts/Mammoth-Mountain/6day/mid", "snowforecast_mammoth"),
    ]

    for url, name in additional_sites:
        try:
            result = await capture_site(page, url, name)
            results.append(result)
        except Exception as e:
            print(f"  Could not access {name}: {e}")

    return results


async def main():
    """Main research function."""
    all_results = []

    async with async_playwright() as p:
        # Launch browser in headed mode for better rendering
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        # Research each site
        print("\n" + "="*70)
        print("SKI WEATHER SITES RESEARCH")
        print("="*70)

        # SnowBrains
        results = await research_snowbrains(page)
        all_results.extend(results)

        # OnTheSnow
        results = await research_onthesnow(page)
        all_results.extend(results)

        # Ski.com
        results = await research_skicom(page)
        all_results.extend(results)

        # Powder Project
        results = await research_powder_project(page)
        all_results.extend(results)

        # Additional sites
        results = await research_additional_sites(page)
        all_results.extend(results)

        await browser.close()

    # Summary
    print("\n" + "="*70)
    print("RESEARCH SUMMARY")
    print("="*70)

    successful = [r for r in all_results if r.get("success")]
    failed = [r for r in all_results if not r.get("success")]

    print(f"\nSuccessfully captured: {len(successful)} pages")
    print(f"Failed: {len(failed)} pages")

    if successful:
        print("\nSuccessful captures:")
        for r in successful:
            print(f"  - {r['name']}: {len(r.get('screenshots', []))} screenshots")

    if failed:
        print("\nFailed captures:")
        for r in failed:
            print(f"  - {r['name']}: {r.get('error', 'Unknown error')}")

    total_screenshots = sum(len(r.get("screenshots", [])) for r in all_results)
    print(f"\nTotal screenshots saved: {total_screenshots}")
    print(f"Location: {SCREENSHOT_DIR}")

    return all_results


if __name__ == "__main__":
    asyncio.run(main())
