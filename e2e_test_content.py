#!/usr/bin/env python3
"""
E2E test to capture full page content from snowforecast.streamlit.app
"""

from playwright.sync_api import sync_playwright
import time
import json

def run_test():
    """Run comprehensive content analysis of snowforecast app."""

    results = {
        "page_title": None,
        "ski_areas": [],
        "error_messages": [],
        "cache_status": None,
        "forecast_data_status": None,
        "data_sources": [],
        "all_text_content": None,
        "page_load_time": None,
        "screenshots": []
    }

    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080}
        )
        page = context.new_page()

        print("=" * 60)
        print("SNOWFORECAST E2E TEST - COMPREHENSIVE CONTENT ANALYSIS")
        print("=" * 60)

        # Navigate to the page
        url = "https://snowforecast.streamlit.app/~/+/"
        print(f"\n[1] Navigating to: {url}")

        start_time = time.time()

        try:
            page.goto(url, timeout=60000)
            print("[2] Initial page load complete")
        except Exception as e:
            print(f"[ERROR] Navigation failed: {e}")
            browser.close()
            return results

        # Wait for Streamlit to fully load
        print("[3] Waiting 20 seconds for full Streamlit render...")
        time.sleep(20)

        load_time = time.time() - start_time
        results["page_load_time"] = f"{load_time:.2f} seconds"
        print(f"[4] Total load time: {load_time:.2f} seconds")

        # Get page title
        results["page_title"] = page.title()
        print(f"\n[5] PAGE TITLE: {results['page_title']}")

        # Take screenshot
        screenshot_path = "/Users/patrickkavanagh/snowforecast/e2e_screenshot.png"
        page.screenshot(path=screenshot_path, full_page=True)
        results["screenshots"].append(screenshot_path)
        print(f"[6] Screenshot saved to: {screenshot_path}")

        # Extract all visible text content
        print("\n[7] EXTRACTING ALL TEXT CONTENT...")

        # Get the main content
        try:
            # Wait a bit more for any dynamic content
            time.sleep(2)

            # Get all text from the page
            all_text = page.inner_text("body")
            results["all_text_content"] = all_text

            # Print formatted text content
            print("\n" + "=" * 60)
            print("FULL PAGE TEXT CONTENT:")
            print("=" * 60)
            print(all_text[:5000] if len(all_text) > 5000 else all_text)
            if len(all_text) > 5000:
                print(f"\n... [truncated, total {len(all_text)} chars] ...")
            print("=" * 60)

        except Exception as e:
            print(f"[ERROR] Could not extract text: {e}")

        # Look for ski area names (common patterns)
        print("\n[8] SEARCHING FOR SKI AREAS...")
        ski_area_keywords = [
            "Ski", "Resort", "Mountain", "Peak", "Snow", "Basin",
            "Park City", "Alta", "Snowbird", "Deer Valley", "Brighton",
            "Solitude", "Sundance", "Powder Mountain", "Snowbasin",
            "Jackson Hole", "Mammoth", "Vail", "Aspen", "Telluride",
            "Breckenridge", "Keystone", "Copper Mountain"
        ]

        text_lower = all_text.lower() if all_text else ""
        found_ski_areas = []
        for keyword in ski_area_keywords:
            if keyword.lower() in text_lower:
                found_ski_areas.append(keyword)

        results["ski_areas"] = found_ski_areas
        print(f"Ski area keywords found: {found_ski_areas}")

        # Check for error messages
        print("\n[9] CHECKING FOR ERROR MESSAGES...")
        error_keywords = ["error", "failed", "exception", "not found", "unavailable", "timeout"]
        found_errors = []
        for keyword in error_keywords:
            if keyword in text_lower:
                # Extract context around the error
                idx = text_lower.find(keyword)
                context = all_text[max(0, idx-50):min(len(all_text), idx+100)] if all_text else ""
                found_errors.append(context.strip())

        results["error_messages"] = found_errors[:5]  # Limit to first 5
        if found_errors:
            print(f"Error keywords found: {found_errors[:3]}...")
        else:
            print("No error messages detected")

        # Check cache status
        print("\n[10] CHECKING CACHE STATUS...")
        cache_keywords = ["cache", "cached", "fresh", "stale", "last updated", "refresh"]
        cache_info = []
        for keyword in cache_keywords:
            if keyword in text_lower:
                idx = text_lower.find(keyword)
                context = all_text[max(0, idx-30):min(len(all_text), idx+80)] if all_text else ""
                cache_info.append(context.strip())

        results["cache_status"] = cache_info if cache_info else "No cache status information found"
        print(f"Cache info: {cache_info if cache_info else 'None found'}")

        # Check for forecast data
        print("\n[11] CHECKING FORECAST DATA STATUS...")
        forecast_keywords = ["snowfall", "snow depth", "temperature", "precipitation",
                           "forecast", "prediction", "24h", "48h", "inches", "cm", "mm"]
        forecast_found = []
        for keyword in forecast_keywords:
            if keyword in text_lower:
                forecast_found.append(keyword)

        if forecast_found:
            results["forecast_data_status"] = f"SHOWING DATA - Keywords found: {forecast_found}"
        else:
            results["forecast_data_status"] = "POTENTIALLY EMPTY - No forecast keywords found"
        print(f"Forecast status: {results['forecast_data_status']}")

        # Check for data sources
        print("\n[12] CHECKING DATA SOURCES...")
        source_keywords = ["HRRR", "ERA5", "SNOTEL", "DEM", "elevation", "GHCN",
                         "OpenSkiMap", "NOAA", "weather model", "atmospheric"]
        sources_found = []
        for keyword in source_keywords:
            if keyword.lower() in text_lower:
                sources_found.append(keyword)

        results["data_sources"] = sources_found
        print(f"Data sources found: {sources_found if sources_found else 'None explicitly mentioned'}")

        # Try to find specific Streamlit elements
        print("\n[13] CHECKING STREAMLIT ELEMENTS...")

        # Check for selectbox/dropdown
        try:
            selectboxes = page.query_selector_all("[data-testid='stSelectbox']")
            print(f"Found {len(selectboxes)} selectbox elements")
        except:
            print("Could not query selectboxes")

        # Check for dataframes
        try:
            dataframes = page.query_selector_all("[data-testid='stDataFrame']")
            print(f"Found {len(dataframes)} dataframe elements")
        except:
            print("Could not query dataframes")

        # Check for metrics
        try:
            metrics = page.query_selector_all("[data-testid='stMetric']")
            print(f"Found {len(metrics)} metric elements")
        except:
            print("Could not query metrics")

        # Check for charts
        try:
            charts = page.query_selector_all("[data-testid='stPlotlyChart'], [data-testid='stAltairChart']")
            print(f"Found {len(charts)} chart elements")
        except:
            print("Could not query charts")

        # Final summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Page Title: {results['page_title']}")
        print(f"Load Time: {results['page_load_time']}")
        print(f"Ski Areas Found: {len(results['ski_areas'])} keywords")
        print(f"Errors Detected: {len(results['error_messages'])} messages")
        print(f"Cache Status: {'Found' if results['cache_status'] != 'No cache status information found' else 'Not found'}")
        print(f"Forecast Data: {results['forecast_data_status']}")
        print(f"Data Sources: {len(results['data_sources'])} mentioned")
        print(f"Screenshot: {screenshot_path}")
        print("=" * 60)

        browser.close()

    return results

if __name__ == "__main__":
    results = run_test()

    # Save results to JSON
    with open("/Users/patrickkavanagh/snowforecast/e2e_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: /Users/patrickkavanagh/snowforecast/e2e_results.json")
