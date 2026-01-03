#!/usr/bin/env python3
"""
Playwright test for snowforecast.streamlit.app - Time Selector Testing

Tests the Forecast Time radio buttons and verifies forecast updates.
Handles Streamlit's iframe-based rendering by working within the correct frame.
"""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

# Create screenshots directory
SCREENSHOTS_DIR = Path("/Users/patrickkavanagh/snowforecast/screenshots_time_selector")
SCREENSHOTS_DIR.mkdir(exist_ok=True)

def get_app_frame(page):
    """Find the Streamlit app frame - the one with ~/+/ in URL."""
    for frame in page.frames:
        url = frame.url
        # The Streamlit app content frame has ~/+/ pattern
        if "~/+/" in url:
            return frame
    # Fallback: find frame with stApp content
    for frame in page.frames:
        try:
            content = frame.content()
            if "Forecast Time" in content or "stApp" in content:
                return frame
        except:
            pass
    return page

def test_time_selector():
    """Test the time selector radio buttons on the snowforecast app."""

    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 1000})
        page = context.new_page()

        print("=" * 60)
        print("SNOWFORECAST TIME SELECTOR TEST")
        print("=" * 60)

        # Navigate to the app
        print("\n[1] Loading https://snowforecast.streamlit.app...")
        page.goto("https://snowforecast.streamlit.app", timeout=90000)

        # Wait for the app to fully load
        print("[2] Waiting for app to initialize...")
        time.sleep(15)

        # Take initial screenshot to verify app loaded
        page.screenshot(path=str(SCREENSHOTS_DIR / "01_initial_load.png"), full_page=True)
        print(f"    Screenshot saved: 01_initial_load.png")

        # Find the Streamlit app frame
        print("\n[3] Finding Streamlit app frame...")
        print(f"    Available frames:")
        for i, frame in enumerate(page.frames):
            print(f"      Frame {i}: {frame.url[:70]}...")

        app_frame = get_app_frame(page)
        print(f"    Selected frame: {app_frame.url[:70]}...")

        # Verify we have the right frame
        frame_content = app_frame.content()
        has_forecast = "Forecast Time" in frame_content
        print(f"    Frame has 'Forecast Time': {has_forecast}")

        if not has_forecast:
            print("    WARNING: Content not found, waiting longer...")
            time.sleep(10)
            app_frame = get_app_frame(page)
            frame_content = app_frame.content()
            print(f"    Retry - Frame has 'Forecast Time': {'Forecast Time' in frame_content}")

        # Test time selector options
        time_options = ["Now", "Tonight", "Tomorrow AM", "Tomorrow PM", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]
        results = []
        screenshot_num = 2

        print("\n[4] Testing Forecast Time radio buttons...")

        for time_option in time_options:
            print(f"\n    Testing: '{time_option}'")

            try:
                # Re-get the app frame in case it was detached
                app_frame = get_app_frame(page)

                # Get metrics before clicking
                metrics_before = app_frame.evaluate('''() => {
                    const metrics = [];
                    document.querySelectorAll('[data-testid="stMetricValue"]').forEach(el => {
                        metrics.push(el.textContent);
                    });
                    return metrics;
                }''')
                print(f"        Metrics before: {metrics_before[:4] if metrics_before else 'None'}")

                # Get caption before
                caption_before = app_frame.evaluate('''() => {
                    const text = document.body.innerText;
                    const lines = text.split("\\n");
                    for (const line of lines) {
                        if (line.includes("Forecast for:")) {
                            return line.trim();
                        }
                    }
                    return null;
                }''')

                # Click the radio option
                clicked = app_frame.evaluate('''(timeOption) => {
                    const allElements = document.querySelectorAll('p, span, div, label');
                    for (const el of allElements) {
                        if (el.textContent.trim() === timeOption) {
                            el.click();
                            return { clicked: true, tag: el.tagName };
                        }
                    }
                    return { clicked: false };
                }''', time_option)

                if clicked and clicked.get("clicked"):
                    print(f"        Clicked: {clicked}")

                    # Wait for UI update
                    time.sleep(2)

                    # Take screenshot
                    screenshot_name = f"{screenshot_num:02d}_time_{time_option.replace(' ', '_').lower()}.png"
                    page.screenshot(path=str(SCREENSHOTS_DIR / screenshot_name), full_page=True)
                    print(f"        Screenshot: {screenshot_name}")
                    screenshot_num += 1

                    # Re-get frame and get metrics after
                    app_frame = get_app_frame(page)
                    metrics_after = app_frame.evaluate('''() => {
                        const metrics = [];
                        document.querySelectorAll('[data-testid="stMetricValue"]').forEach(el => {
                            metrics.push(el.textContent);
                        });
                        return metrics;
                    }''')
                    print(f"        Metrics after: {metrics_after[:4] if metrics_after else 'None'}")

                    # Get caption after
                    caption_after = app_frame.evaluate('''() => {
                        const text = document.body.innerText;
                        const lines = text.split("\\n");
                        for (const line of lines) {
                            if (line.includes("Forecast for:")) {
                                return line.trim();
                            }
                        }
                        return null;
                    }''')
                    print(f"        Caption: {caption_after}")

                    results.append({
                        "option": time_option,
                        "clicked": True,
                        "metrics_before": metrics_before[:4] if metrics_before else [],
                        "metrics_after": metrics_after[:4] if metrics_after else [],
                        "metrics_changed": metrics_before != metrics_after,
                        "caption_before": caption_before,
                        "caption_after": caption_after,
                        "caption_changed": caption_before != caption_after
                    })
                else:
                    print(f"        Could not click element")
                    results.append({
                        "option": time_option,
                        "clicked": False,
                        "error": "Element not found"
                    })

            except Exception as e:
                print(f"        ERROR: {str(e)[:100]}")
                results.append({
                    "option": time_option,
                    "clicked": False,
                    "error": str(e)[:100]
                })

        # Test resort switching
        print("\n[5] Testing resort switching...")

        try:
            # Re-get frame
            app_frame = get_app_frame(page)

            # Get metrics before
            metrics_before_resort = app_frame.evaluate('''() => {
                const metrics = [];
                document.querySelectorAll('[data-testid="stMetricValue"]').forEach(el => {
                    metrics.push(el.textContent);
                });
                return metrics;
            }''')

            # Get current resort name
            current_resort = app_frame.evaluate('''() => {
                const selectboxes = document.querySelectorAll('[data-baseweb="select"]');
                if (selectboxes.length >= 2) {
                    return selectboxes[1].textContent;
                }
                return null;
            }''')
            print(f"    Current resort: {current_resort}")

            # Click the second selectbox (Ski Area)
            clicked = app_frame.evaluate('''() => {
                const selectboxes = document.querySelectorAll('[data-baseweb="select"]');
                if (selectboxes.length >= 2) {
                    selectboxes[1].click();
                    return true;
                }
                return false;
            }''')

            if clicked:
                time.sleep(1)
                page.screenshot(path=str(SCREENSHOTS_DIR / f"{screenshot_num:02d}_resort_dropdown.png"), full_page=True)
                print(f"    Screenshot: {screenshot_num:02d}_resort_dropdown.png")
                screenshot_num += 1

                # Re-get frame
                app_frame = get_app_frame(page)

                # Find and click a different option
                option_text = app_frame.evaluate('''() => {
                    const options = document.querySelectorAll('[role="option"]');
                    if (options.length > 1) {
                        const text = options[1].textContent;
                        options[1].click();
                        return text;
                    }
                    return null;
                }''')

                if option_text:
                    print(f"    Selected resort: {option_text}")
                    time.sleep(3)

                    page.screenshot(path=str(SCREENSHOTS_DIR / f"{screenshot_num:02d}_after_resort_change.png"), full_page=True)
                    print(f"    Screenshot: {screenshot_num:02d}_after_resort_change.png")
                    screenshot_num += 1

                    # Re-get frame and get metrics after
                    app_frame = get_app_frame(page)
                    metrics_after_resort = app_frame.evaluate('''() => {
                        const metrics = [];
                        document.querySelectorAll('[data-testid="stMetricValue"]').forEach(el => {
                            metrics.push(el.textContent);
                        });
                        return metrics;
                    }''')

                    print(f"    Metrics before: {metrics_before_resort[:4] if metrics_before_resort else 'None'}")
                    print(f"    Metrics after: {metrics_after_resort[:4] if metrics_after_resort else 'None'}")
                    print(f"    Resort change updated metrics: {metrics_before_resort != metrics_after_resort}")
                else:
                    print("    No dropdown options found")
                    page.keyboard.press("Escape")

        except Exception as e:
            print(f"    Resort switch error: {str(e)}")

        # Final screenshot
        page.screenshot(path=str(SCREENSHOTS_DIR / f"{screenshot_num:02d}_final.png"), full_page=True)

        # Final summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        successful = sum(1 for r in results if r.get("clicked", False))
        print(f"\nTime options tested: {len(results)}")
        print(f"Successfully clicked: {successful}/{len(time_options)}")

        if successful > 0:
            metric_changes = sum(1 for r in results if r.get("metrics_changed", False))
            caption_changes = sum(1 for r in results if r.get("caption_changed", False))
            print(f"Metric changes observed: {metric_changes}/{successful}")
            print(f"Caption changes observed: {caption_changes}/{successful}")

        print("\nDetailed Results:")
        for r in results:
            status = "OK" if r.get("clicked") else "FAIL"
            print(f"\n  [{status}] {r['option']}")
            if r.get("clicked"):
                print(f"       Caption: {r.get('caption_after', 'N/A')}")
                if r.get('metrics_after'):
                    print(f"       Metrics: {r.get('metrics_after')}")
                print(f"       Changes: caption={r.get('caption_changed')}, metrics={r.get('metrics_changed')}")
            else:
                print(f"       Error: {r.get('error', 'Unknown error')}")

        print(f"\nScreenshots saved to: {SCREENSHOTS_DIR}")

        browser.close()

        return results

if __name__ == "__main__":
    test_time_selector()
