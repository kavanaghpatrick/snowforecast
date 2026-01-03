const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    javaScriptEnabled: true,
    viewport: { width: 1280, height: 900 }
  });
  const page = await context.newPage();

  console.log('Loading page with JavaScript enabled...');

  try {
    await page.goto('https://snowforecast.streamlit.app', {
      waitUntil: 'load',
      timeout: 120000
    });
    console.log('Page loaded, waiting for Streamlit to initialize...');

    // Wait for Streamlit app to be ready - wait for specific content
    console.log('Waiting for page content to load...');
    await page.waitForTimeout(10000);

    // Take an initial screenshot
    await page.screenshot({
      path: '/Users/patrickkavanagh/snowforecast/streamlit_initial.png',
      fullPage: true
    });

    // Wait for the main iframe to load (Streamlit renders in an iframe)
    let frame = page.mainFrame();
    const frames = page.frames();
    console.log(`Found ${frames.length} frames`);

    // Try to find iframe content
    for (const f of frames) {
      const url = f.url();
      console.log(`Frame URL: ${url}`);
    }

    // Wait more for full render
    console.log('Waiting 30 more seconds for full render...');
    await page.waitForTimeout(30000);

    // Take screenshots
    await page.screenshot({
      path: '/Users/patrickkavanagh/snowforecast/streamlit_top.png',
      clip: { x: 0, y: 0, width: 1280, height: 400 }
    });

    await page.screenshot({
      path: '/Users/patrickkavanagh/snowforecast/streamlit_full.png',
      fullPage: true
    });

    // Get all frames and their content
    console.log('\n=== Extracting text from all frames ===');
    for (const f of page.frames()) {
      try {
        const text = await f.evaluate(() => document.body?.innerText || '');
        if (text && text.trim().length > 0) {
          console.log(`\n--- Frame: ${f.url()} ---`);
          console.log(text.substring(0, 2000));
        }
      } catch (e) {
        console.log(`Could not get text from frame: ${e.message}`);
      }
    }

    // Try using locator to find all text elements
    console.log('\n=== All locator text ===');
    const allTextLocators = await page.locator('body *').all();
    const textSet = new Set();
    for (const loc of allTextLocators.slice(0, 200)) {
      try {
        const text = await loc.innerText({ timeout: 100 });
        if (text && text.trim().length > 0 && text.trim().length < 300) {
          textSet.add(text.trim());
        }
      } catch (e) {
        // ignore
      }
    }
    console.log([...textSet].join('\n'));

    // Look specifically for version patterns in page source
    console.log('\n=== Checking page source for version ===');
    const html = await page.content();
    const versionMatch = html.match(/v\d{4}\.\d{2}\.\d{2}[^"<]*/g);
    if (versionMatch) {
      console.log('Found version in HTML:', versionMatch);
    }

    const cloudMatch = html.match(/Cloud:\s*(True|False)/gi);
    if (cloudMatch) {
      console.log('Found Cloud in HTML:', cloudMatch);
    }

    const dbMatch = html.match(/DB exists:\s*(True|False)/gi);
    if (dbMatch) {
      console.log('Found DB exists in HTML:', dbMatch);
    }

    // Look for any "caption" class elements
    console.log('\n=== Caption class elements ===');
    const captions = await page.locator('[class*="caption"], [class*="Caption"], small, .st-emotion-cache-*').all();
    console.log(`Found ${captions.length} caption-like elements`);
    for (const cap of captions.slice(0, 20)) {
      try {
        const text = await cap.innerText({ timeout: 200 });
        const box = await cap.boundingBox();
        if (text && text.trim()) {
          console.log(`[y=${box?.y || '?'}] "${text.trim().substring(0, 100)}"`);
        }
      } catch (e) {
        // ignore
      }
    }

  } catch (error) {
    console.error('Error:', error.message);
    await page.screenshot({
      path: '/Users/patrickkavanagh/snowforecast/streamlit_error.png',
      fullPage: true
    });
  }

  await browser.close();
  console.log('\nDone! Screenshots saved.');
})();
