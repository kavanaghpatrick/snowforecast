"""
E2E test for 7-day forecast on the snowforecast dashboard.
Tests forecast display, data validation, and resort switching.
"""
import asyncio
import re
from pathlib import Path
from playwright.async_api import async_playwright

SCREENSHOTS_DIR = Path(__file__).parent / "screenshots"


async def test_forecast_charts():
    """Test the forecast functionality."""
    SCREENSHOTS_DIR.mkdir(exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await context.new_page()

        print("=" * 60)
        print("SNOWFORECAST DASHBOARD E2E TEST")
        print("=" * 60)

        # Try loading the page with retries
        max_retries = 3
        app_loaded = False

        for attempt in range(max_retries):
            print(f"\nAttempt {attempt + 1}/{max_retries}: Navigating to https://snowforecast.streamlit.app...")
            try:
                await page.goto("https://snowforecast.streamlit.app", timeout=60000, wait_until="load")
            except Exception as e:
                print(f"Navigation warning: {e}")

            # Wait for Streamlit to initialize
            print("Waiting for Streamlit app to load...")
            await asyncio.sleep(15)

            # Check if we got an error page
            page_text = await page.evaluate("() => document.body ? document.body.innerText : ''")

            if "Oh no" in page_text or "Error running app" in page_text:
                print(f"App error detected on attempt {attempt + 1}. Retrying...")
                await page.screenshot(path=str(SCREENSHOTS_DIR / f"error_attempt_{attempt + 1}.png"), full_page=True)
                # Reload the page
                await page.reload()
                await asyncio.sleep(10)
            else:
                app_loaded = True
                print("App loaded successfully!")
                break

        # Take screenshot
        await page.screenshot(path=str(SCREENSHOTS_DIR / "01_initial_page.png"), full_page=True)
        print("Screenshot 1: Initial page captured")

        if not app_loaded:
            print("\n" + "=" * 60)
            print("ERROR: App failed to load after all retries")
            print("The Streamlit app appears to be down or experiencing errors.")
            print("=" * 60)

            # Check if it's a temporary error
            final_text = await page.evaluate("() => document.body ? document.body.innerText : ''")
            print(f"\nPage content: {final_text[:200] if final_text else 'Empty'}")

            await browser.close()
            return

        # Get page title
        title = await page.title()
        print(f"\nPage title: {title}")

        # Get page content
        print("\n--- Extracting Page Content ---")
        all_text = await page.evaluate("() => document.body ? document.body.innerText : ''")
        print(f"Page text length: {len(all_text)} characters")
        if all_text:
            print(f"Content preview:\n{all_text[:1200]}...")

        # Look for specific dashboard elements
        print("\n--- Looking for Dashboard Elements ---")

        # Check for Forecast Chart tab
        forecast_chart_tab = await page.locator("text='Forecast Chart'").count()
        print(f"'Forecast Chart' tab found: {forecast_chart_tab > 0}")

        # If found, click on it
        if forecast_chart_tab > 0:
            try:
                await page.locator("text='Forecast Chart'").first.click()
                await asyncio.sleep(2)
                await page.screenshot(path=str(SCREENSHOTS_DIR / "02_forecast_chart_tab.png"), full_page=True)
                print("Clicked on Forecast Chart tab")
            except Exception as e:
                print(f"Could not click Forecast Chart tab: {e}")

        # Check for tabs
        tabs = await page.locator("[role='tab'], [data-baseweb='tab']").all()
        print(f"Tab elements found: {len(tabs)}")
        for tab in tabs:
            try:
                tab_text = await tab.inner_text()
                print(f"  Tab: '{tab_text}'")
            except:
                pass

        # Look for the 7-Day Forecast section
        print("\n--- 7-Day Forecast Section ---")
        forecast_section = await page.locator("text='7-Day Forecast'").count()
        print(f"'7-Day Forecast' section found: {forecast_section > 0}")

        # Look for forecast metrics
        snow_base = await page.locator("text='Snow Base'").count()
        new_snow = await page.locator("text='New Snow'").count()
        probability = await page.locator("text='Probability'").count()
        print(f"Snow Base label found: {snow_base > 0}")
        print(f"New Snow label found: {new_snow > 0}")
        print(f"Probability label found: {probability > 0}")

        # Extract numeric values
        print("\n--- Extracting Forecast Values ---")

        # Look for cm values
        cm_elements = await page.locator("text=/\\d+\\.?\\d*\\s*cm/i").all()
        print(f"Elements with 'cm' values: {len(cm_elements)}")
        for elem in cm_elements[:10]:
            try:
                text = await elem.inner_text()
                print(f"  Value: '{text}'")
            except:
                pass

        # Look for percentage values
        pct_elements = await page.locator("text=/\\d+%/").all()
        print(f"Elements with '%' values: {len(pct_elements)}")
        for elem in pct_elements[:5]:
            try:
                text = await elem.inner_text()
                print(f"  Value: '{text}'")
            except:
                pass

        # Check for the forecast table
        print("\n--- Forecast Table Analysis ---")
        tables = await page.locator("table").all()
        print(f"Table elements found: {len(tables)}")

        headers = await page.locator("th").all()
        print(f"Table headers found: {len(headers)}")
        header_texts = []
        for h in headers[:10]:
            try:
                text = await h.inner_text()
                header_texts.append(text)
            except:
                pass
        if header_texts:
            print(f"  Headers: {header_texts}")

        rows = await page.locator("tr").all()
        print(f"Table rows found: {len(rows)}")

        # Try to switch resort
        print("\n--- Resort Selection Test ---")

        # Look for the ski area dropdown
        selectbox = page.locator("[data-testid='stSelectbox']").first
        try:
            await selectbox.wait_for(timeout=5000)
            current_value = await selectbox.inner_text()
            print(f"Current ski area: {current_value}")

            # Click to open dropdown
            await selectbox.click()
            await asyncio.sleep(1)
            await page.screenshot(path=str(SCREENSHOTS_DIR / "03_dropdown_open.png"), full_page=True)

            # Find options
            options = await page.locator("[role='option']").all()
            print(f"Resort options available: {len(options)}")

            if len(options) > 1:
                second_resort = await options[1].inner_text()
                print(f"Switching to: {second_resort}")
                await options[1].click()
                await asyncio.sleep(5)
                await page.screenshot(path=str(SCREENSHOTS_DIR / "04_after_resort_switch.png"), full_page=True)
                print("Screenshot: After resort switch captured")

        except Exception as e:
            print(f"Could not interact with ski area dropdown: {e}")

        # Look for chart elements
        print("\n--- Chart Elements ---")

        svgs = await page.locator("svg").all()
        print(f"SVG elements: {len(svgs)}")
        for i, svg in enumerate(svgs[:5]):
            try:
                bbox = await svg.bounding_box()
                if bbox:
                    size = f"{bbox['width']:.0f}x{bbox['height']:.0f}"
                    print(f"  SVG {i}: {size} pixels")
                    if bbox['width'] > 200 and bbox['height'] > 100:
                        await svg.screenshot(path=str(SCREENSHOTS_DIR / f"05_svg_chart_{i}.png"))
                        print(f"    -> Screenshot captured")
            except:
                pass

        canvases = await page.locator("canvas").all()
        print(f"Canvas elements: {len(canvases)}")

        vega_charts = await page.locator(".vega-embed").all()
        print(f"Vega chart elements: {len(vega_charts)}")

        # Check for forecast time radio buttons
        print("\n--- Forecast Time Selector ---")
        time_options = ["Now", "Tonight", "Tomorrow AM", "Tomorrow PM", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]
        for opt in time_options:
            elem = await page.locator(f"text='{opt}'").count()
            if elem > 0:
                print(f"  Found: '{opt}'")

        # Click on Day 7 to test interaction
        try:
            day7_radio = page.locator("text='Day 7'").first
            if await day7_radio.count() > 0:
                await day7_radio.click()
                await asyncio.sleep(2)
                await page.screenshot(path=str(SCREENSHOTS_DIR / "06_day7_selected.png"), full_page=True)
                print("  Clicked 'Day 7' - screenshot captured")
        except Exception as e:
            print(f"  Could not click Day 7: {e}")

        # Final full page screenshot
        await page.screenshot(path=str(SCREENSHOTS_DIR / "07_final.png"), full_page=True)

        # Final summary
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)

        # Re-extract text for final analysis
        final_text = await page.evaluate("() => document.body ? document.body.innerText : ''")

        # Check for key indicators
        print("\n--- Key Elements Check ---")
        checks = {
            "'Forecast Chart' tab": "Forecast Chart" in final_text,
            "'New Snow (cm)' chart": "New Snow (cm)" in final_text,
            "'Base Depth (cm)' chart": "Base Depth (cm)" in final_text,
            "7-Day Forecast section": "7-Day Forecast" in final_text,
            "Snow Base metric": "Snow Base" in final_text,
            "New Snow metric": "New Snow" in final_text,
            "Probability metric": "Probability" in final_text,
        }

        for check, found in checks.items():
            status = "FOUND" if found else "NOT FOUND"
            print(f"  {check}: {status}")

        # Check for data values
        print("\n--- Data Values Check ---")
        cm_values = re.findall(r'(\d+\.?\d*)\s*cm', final_text)
        if cm_values:
            float_vals = [float(v) for v in cm_values]
            non_zero = [v for v in float_vals if v > 0]
            print(f"  CM values found: {len(cm_values)}")
            print(f"  Non-zero values: {len(non_zero)}")
            print(f"  Sample values: {cm_values[:10]}")
            if non_zero:
                print(f"  Non-zero sample: {non_zero[:5]}")
            else:
                print("  NOTE: All snow values are ZERO (may be valid for dry forecast)")
        else:
            print("  WARNING: No CM values found in page")

        # Check for anomalies
        print("\n--- Anomaly Check ---")
        anomalies_found = []
        if "NaN" in final_text:
            anomalies_found.append("NaN values present")
        if re.search(r'-\d+\.?\d*\s*cm', final_text):
            anomalies_found.append("Negative cm values")
        if "DB exists: False" in final_text:
            anomalies_found.append("Database not connected (DB exists: False)")

        if anomalies_found:
            for a in anomalies_found:
                print(f"  WARNING: {a}")
        else:
            print("  No anomalies detected")

        # Check for 7 days of data
        print("\n--- 7-Day Data Check ---")
        days = ["Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]
        days_present = [d for d in days if d in final_text]
        print(f"  Day selectors found: {len(days_present)}/5")

        weekdays = re.findall(r'\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b', final_text)
        unique_weekdays = list(set(weekdays))
        print(f"  Unique weekdays in data: {unique_weekdays}")

        await browser.close()
        print(f"\nScreenshots saved to: {SCREENSHOTS_DIR}")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_forecast_charts())
