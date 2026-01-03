const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    javaScriptEnabled: true,
    userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
  });
  const page = await context.newPage();

  console.log('Testing path fix - checking https://snowforecast.streamlit.app...');

  try {
    await page.goto('https://snowforecast.streamlit.app', {
      waitUntil: 'domcontentloaded',
      timeout: 60000
    });

    console.log('Waiting 30 seconds for full app load...');
    await page.waitForTimeout(30000);

    // Get the page text
    const text = await page.evaluate(() => document.body.innerText);

    // Look for version and DB status
    const versionMatch = text.match(/v2026\.01\.03\.\d+[^\n]*/);
    console.log('\n=== VERSION & DB STATUS ===');
    console.log(versionMatch ? versionMatch[0] : 'Version not found');

    // Check for error messages
    const hasError = text.includes('Oh no') || text.includes('Error running app');
    console.log('App crashed:', hasError);

    // Check for DB exists: True
    const dbExists = text.includes('DB exists: True');
    console.log('DB exists: True found:', dbExists);

    // Take screenshot
    await page.screenshot({ path: '/tmp/path_fix_test.png', fullPage: true });
    console.log('\nScreenshot saved: /tmp/path_fix_test.png');

    // Summary
    console.log('\n=== SUMMARY ===');
    if (dbExists && !hasError) {
      console.log('✅ FIX SUCCESSFUL - Database found!');
    } else if (hasError) {
      console.log('❌ App still crashing');
    } else {
      console.log('⚠️ DB exists: False - path issue persists');
    }

  } catch (error) {
    console.error('Error:', error.message);
  }

  await browser.close();
})();
