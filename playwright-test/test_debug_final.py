import asyncio
from playwright.async_api import async_playwright

async def test_debug_section():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1400, 'height': 1200})
        page.set_default_timeout(120000)
        
        print("Loading https://snowforecast.streamlit.app...")
        await page.goto('https://snowforecast.streamlit.app', wait_until='load', timeout=90000)
        
        print("Waiting 40 seconds for full load...")
        await asyncio.sleep(40)
        
        # Access Streamlit frame
        frames = page.frames
        streamlit_frame = None
        for frame in frames:
            if '~/+/' in frame.url:
                streamlit_frame = frame
                break
        
        if streamlit_frame:
            # Get all text
            full_text = await streamlit_frame.inner_text('body')
            print("\n" + "="*60)
            print("FULL PAGE TEXT")
            print("="*60)
            print(full_text)
            print("="*60)
            
            # Specific search for debug section
            print("\n" + "="*60)
            print("SEARCHING FOR DEBUG SECTION COMPONENTS")
            print("="*60)
            
            debug_patterns = [
                ('🔍 Debug', 'Debug header with emoji'),
                ('Debug', 'Debug text'),
                ('DB:', 'Database path'),
                ('FC:', 'Forecast count'),
                ('T:', 'Terrain count'),
                ('Run:', 'Run time'),
            ]
            
            for pattern, description in debug_patterns:
                if pattern in full_text:
                    idx = full_text.find(pattern)
                    context = full_text[max(0,idx-20):idx+80]
                    print(f"FOUND '{pattern}' ({description}):")
                    print(f"  Context: ...{context}...")
                else:
                    print(f"NOT FOUND: '{pattern}' ({description})")
            
            # Look for error messages
            print("\n" + "="*60)
            print("CHECKING FOR ERROR MESSAGES")
            print("="*60)
            error_patterns = ['error', 'Error', 'could not', 'Could not', 'failed', 'Failed']
            for pattern in error_patterns:
                if pattern in full_text:
                    idx = full_text.find(pattern)
                    context = full_text[max(0,idx-30):idx+100]
                    print(f"Found '{pattern}': ...{context}...")
        
        # Final screenshot
        await page.screenshot(path='final_debug_test.png', full_page=True)
        print("\nScreenshot saved: final_debug_test.png")
        
        await browser.close()
        print("\nTest complete!")

if __name__ == '__main__':
    asyncio.run(test_debug_section())
