import asyncio
from playwright.async_api import async_playwright

async def test_debug_section():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Very tall viewport
        page = await browser.new_page(viewport={'width': 1400, 'height': 2000})
        page.set_default_timeout(120000)
        
        print("Loading https://snowforecast.streamlit.app...")
        await page.goto('https://snowforecast.streamlit.app', wait_until='load', timeout=90000)
        
        print("Waiting 35 seconds for full load...")
        await asyncio.sleep(35)
        
        # Take full page screenshot with very tall height
        await page.screenshot(path='01_tall_view.png', full_page=True)
        print("Screenshot 01 saved")
        
        # Access the Streamlit iframe directly
        print("\nAccessing frames...")
        frames = page.frames
        
        streamlit_frame = None
        for frame in frames:
            if '~/+/' in frame.url or frame.url == page.url:
                # Try to get content from this frame
                try:
                    content = await frame.evaluate('() => document.body.innerText')
                    if content and len(content) > 100:
                        streamlit_frame = frame
                        print(f"Found content in frame: {frame.url[:50]}")
                        print(f"Content preview: {content[:200]}...")
                        break
                except:
                    pass
        
        # Try to access via the third frame (the ~/+/ one which is the actual Streamlit app)
        for frame in frames:
            if '~/+/' in frame.url:
                print(f"\nAccessing Streamlit frame: {frame.url}")
                try:
                    # Get all text from this frame
                    text = await frame.inner_text('body')
                    print(f"Frame body text ({len(text)} chars):")
                    print(text[:2000] if text else "Empty")
                    
                    # Look for debug
                    if 'debug' in text.lower():
                        print("\n*** FOUND DEBUG IN FRAME ***")
                        idx = text.lower().find('debug')
                        print(text[max(0,idx-50):idx+200])
                except Exception as e:
                    print(f"Error: {e}")
        
        # Use keyboard to scroll sidebar
        print("\nTrying to scroll sidebar with keyboard...")
        
        # Click on sidebar area first
        await page.mouse.click(100, 400)
        await asyncio.sleep(0.5)
        
        # Press Page Down multiple times
        for i in range(5):
            await page.keyboard.press('PageDown')
            await asyncio.sleep(0.3)
        
        await page.screenshot(path='02_after_pagedown.png', full_page=True)
        print("Screenshot 02 saved (after PageDown)")
        
        # Now try scrolling the entire page
        await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
        await asyncio.sleep(1)
        await page.screenshot(path='03_page_bottom.png', full_page=True)
        print("Screenshot 03 saved (page bottom)")
        
        # Take a screenshot of just the sidebar area
        sidebar_box = await page.evaluate('''() => {
            const sidebar = document.querySelector('[data-testid="stSidebar"]') || 
                           document.querySelector('section[data-testid="stSidebar"]');
            if (sidebar) {
                const rect = sidebar.getBoundingClientRect();
                return {x: rect.x, y: rect.y, width: rect.width, height: rect.height};
            }
            return null;
        }''')
        
        if sidebar_box:
            print(f"\nSidebar bounds: {sidebar_box}")
            # Clip screenshot to sidebar
            await page.screenshot(
                path='04_sidebar_only.png',
                clip={'x': 0, 'y': 0, 'width': 300, 'height': 2000}
            )
            print("Screenshot 04 saved (sidebar area)")
        
        await browser.close()
        print("\nTest complete!")

if __name__ == '__main__':
    asyncio.run(test_debug_section())
