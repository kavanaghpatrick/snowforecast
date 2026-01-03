#!/usr/bin/env python3
"""
Playwright test for Snowforecast All Resorts data table.
Tests the "All Resorts" tab and verifies data integrity.
"""

from playwright.sync_api import sync_playwright
import time
import re


def test_all_resorts():
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(viewport={'width': 1400, 'height': 1200})

        print("Navigating to snowforecast.streamlit.app...")

        # Try loading the page with retries
        max_retries = 5
        loaded = False

        for attempt in range(max_retries):
            print(f"\nAttempt {attempt + 1}/{max_retries}")

            try:
                page.goto("https://snowforecast.streamlit.app", timeout=90000)

                # Wait longer for initial load
                print("Waiting for initial page load...")
                time.sleep(10)

                # Check if we got an error page
                content = page.evaluate("document.body.innerText || ''")
                if "Oh no" in content or "Error running app" in content:
                    print("Got Streamlit error page, waiting 30s before retry...")
                    time.sleep(30)
                    page.reload()
                    time.sleep(15)
                    # Check again after reload
                    content = page.evaluate("document.body.innerText || ''")
                    if "Oh no" in content or "Error running app" in content:
                        continue

                # Wait for dashboard to load
                print("Waiting for dashboard content...")
                for wait_time in range(90):
                    content = page.evaluate("document.body.innerText || ''")
                    if "Snow Forecast Dashboard" in content:
                        print(f"Dashboard loaded after {wait_time}s")
                        loaded = True
                        time.sleep(8)  # Extra time for full render
                        break
                    time.sleep(1)
                    if wait_time % 15 == 0:
                        print(f"  Waiting... {wait_time}s")

                if loaded:
                    break

            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                time.sleep(15)

        if not loaded:
            print("WARNING: Dashboard did not load after all retries")
            # Take screenshot of error state
            page.screenshot(path="/Users/patrickkavanagh/snowforecast/screenshot_error.png", full_page=True)
            print("Error screenshot saved")
            browser.close()
            return "Dashboard failed to load"

        # Take initial screenshot
        page.screenshot(path="/Users/patrickkavanagh/snowforecast/screenshot_initial.png", full_page=True)
        print("Initial screenshot saved")

        # Get page content
        body_text = page.evaluate("document.body.innerText || ''")
        print(f"\nPage text length: {len(body_text)} chars")
        print(f"First 400 chars:\n{body_text[:400]}")

        # Find and click the "All Resorts" tab
        print("\n=== SEARCHING FOR ALL RESORTS TAB ===")

        # Scroll down to see tabs
        page.evaluate("window.scrollTo(0, 600)")
        time.sleep(2)

        # Find all buttons
        buttons = page.evaluate("""
            () => {
                const buttons = document.querySelectorAll('button');
                return Array.from(buttons).map(b => ({
                    text: b.innerText || '',
                    visible: b.offsetParent !== null
                })).filter(b => b.text.trim());
            }
        """)
        print(f"Found {len(buttons)} buttons:")
        for btn in buttons:
            print(f"  - '{btn['text'][:40]}' (visible: {btn['visible']})")

        # Try to click the "All Resorts" tab
        print("\nLooking for 'All Resorts' tab...")
        tab_clicked = False

        try:
            # Method 1: Use Playwright's get_by_role for tabs
            tab = page.get_by_role("tab", name=re.compile("All Resorts", re.IGNORECASE))
            if tab.count() > 0:
                tab.first.click()
                print("Clicked tab via role selector")
                tab_clicked = True
        except Exception as e:
            print(f"Role selector failed: {e}")

        if not tab_clicked:
            try:
                # Method 2: Try get_by_text
                tab = page.get_by_text("All Resorts")
                if tab.count() > 0:
                    tab.first.click()
                    print("Clicked via text selector")
                    tab_clicked = True
            except Exception as e:
                print(f"Text selector failed: {e}")

        if not tab_clicked:
            try:
                # Method 3: JavaScript click
                clicked = page.evaluate("""
                    () => {
                        const elements = document.querySelectorAll('button, [role="tab"]');
                        for (const el of elements) {
                            if (el.innerText && el.innerText.includes('All Resorts')) {
                                el.click();
                                return 'clicked';
                            }
                        }
                        return 'not found';
                    }
                """)
                print(f"JavaScript click result: {clicked}")
                tab_clicked = (clicked == 'clicked')
            except Exception as e:
                print(f"JS click failed: {e}")

        time.sleep(5)

        # Take screenshot of All Resorts tab
        page.screenshot(path="/Users/patrickkavanagh/snowforecast/screenshot_all_resorts_tab.png", full_page=True)
        print("All Resorts tab screenshot saved")

        # Get updated content
        body_text = page.evaluate("document.body.innerText || ''")

        # Save to file
        with open("/Users/patrickkavanagh/snowforecast/page_text.txt", "w") as f:
            f.write(body_text)

        print("\n=== ANALYZING DATA ===")

        # Check for expected columns
        columns = ["Resort", "State", "Elevation", "Base", "New", "Prob"]
        found_columns = [col for col in columns if col in body_text]
        print(f"Found columns: {found_columns}")

        if "All Ski Resorts" in body_text:
            print("'All Ski Resorts' heading found!")

        # Look for resort names
        resort_names = [
            "Stevens Pass", "Crystal Mountain", "Mt. Baker", "Snoqualmie Pass",
            "Mt. Hood Meadows", "Mt. Bachelor", "Timberline",
            "Mammoth Mountain", "Squaw Valley", "Heavenly", "Kirkwood",
            "Vail", "Breckenridge", "Aspen Snowmass", "Telluride",
            "Park City", "Snowbird", "Alta",
            "Big Sky", "Whitefish",
            "Jackson Hole",
            "Sun Valley"
        ]

        found_resorts = [r for r in resort_names if r in body_text]
        print(f"\nFound {len(found_resorts)}/22 resort names:")
        for r in found_resorts:
            print(f"  - {r}")

        # Extract table data
        print("\n=== TABLE DATA ===")
        try:
            table_data = page.evaluate("""
                () => {
                    const tables = document.querySelectorAll('table');
                    const result = [];
                    for (let i = 0; i < tables.length; i++) {
                        const table = tables[i];
                        const rows = table.querySelectorAll('tr');
                        const tableRows = [];
                        for (const row of rows) {
                            const cells = row.querySelectorAll('td, th');
                            const rowData = [];
                            for (const cell of cells) {
                                rowData.push(cell.innerText ? cell.innerText.trim() : '');
                            }
                            if (rowData.some(x => x)) tableRows.push(rowData);
                        }
                        result.push({index: i, rows: tableRows});
                    }
                    return result;
                }
            """)

            print(f"Found {len(table_data)} tables")
            for table in table_data:
                print(f"\nTable {table['index']+1} has {len(table['rows'])} rows:")
                for i, row in enumerate(table['rows'][:25]):
                    print(f"  Row {i}: {row}")

        except Exception as e:
            print(f"Table extraction error: {e}")

        # Analyze snow depth values
        print("\n=== SNOW DEPTH ANALYSIS ===")
        numbers = re.findall(r'\b(\d+)\b', body_text)
        if numbers:
            nums = [int(n) for n in numbers if 0 <= int(n) <= 500]
            zero_count = len([n for n in nums if n == 0])
            non_zero = [n for n in nums if n > 0]

            print(f"Numeric values in snow depth range (0-500):")
            print(f"  Total count: {len(nums)}")
            print(f"  Zero values: {zero_count}")
            print(f"  Non-zero values: {len(non_zero)}")
            if non_zero:
                print(f"  Non-zero samples: {sorted(set(non_zero))[:20]}")
                print(f"  Range: {min(non_zero)} - {max(non_zero)}")
            else:
                print("  WARNING: ALL VALUES ARE ZEROS!")

        # Print sample data from 5 resorts
        print("\n=== SAMPLE DATA FROM 5 RESORTS ===")
        sample_resorts = ["Alta", "Vail", "Mammoth Mountain", "Park City", "Jackson Hole"]
        for resort in sample_resorts:
            if resort in body_text:
                lines = body_text.split('\n')
                for line in lines:
                    if resort in line and len(line) > len(resort) + 3:
                        print(f"  {resort}: {line.strip()}")
                        break

        # Take final screenshot
        page.screenshot(path="/Users/patrickkavanagh/snowforecast/screenshot_final.png", full_page=True)
        print("\nFinal screenshot saved")

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"1. Dashboard loaded: {loaded}")
        print(f"2. All Resorts tab clicked: {tab_clicked}")
        print(f"3. Columns found: {len(found_columns)}/6 - {found_columns}")
        print(f"4. Resorts found: {len(found_resorts)}/22")

        if numbers:
            nums = [int(n) for n in numbers if 0 <= int(n) <= 500]
            non_zero = [n for n in nums if n > 0]
            print(f"5. Snow depth values:")
            print(f"   - Total: {len(nums)}, Non-zero: {len(non_zero)}")
            if len(non_zero) == 0 and len(nums) > 10:
                print("   - ISSUE: All snow depths are ZERO!")
            elif len(non_zero) > 0:
                print(f"   - Range: {min(non_zero)} - {max(non_zero)} cm")

        browser.close()
        return body_text


if __name__ == "__main__":
    text = test_all_resorts()
    print("\n=== FULL PAGE TEXT ===")
    print(text[:5000] if len(text) > 5000 else text)
