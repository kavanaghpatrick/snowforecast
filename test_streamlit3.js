const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ 
    headless: true,
  });
  const context = await browser.newContext({
    javaScriptEnabled: true,
    userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
  });
  const page = await context.newPage();
  
  console.log('Navigating to https://snowforecast.streamlit.app...');
  
  try {
    // Just wait for DOM content loaded, not network idle
    await page.goto('https://snowforecast.streamlit.app', { 
      waitUntil: 'domcontentloaded', 
      timeout: 60000 
    });
    
    console.log('Page navigated, waiting for Streamlit to boot...');
    
    // Wait for the app to wake up (Streamlit cloud can be slow)
    console.log('Waiting 20 seconds for app to wake up and load...');
    await page.waitForTimeout(20000);
    await page.screenshot({ path: '/tmp/snowforecast_20s.png', fullPage: true });
    console.log('Screenshot saved: /tmp/snowforecast_20s.png');
    
    // Get page content
    let pageText = await page.evaluate(() => document.body.innerText);
    console.log('\n--- Page text at 20s (first 3000 chars) ---');
    console.log(pageText.substring(0, 3000));
    
    // Check if still loading
    if (pageText.includes('Please wait') || pageText.includes('waking up') || pageText.includes('Booting')) {
      console.log('\n=== APP STILL WAKING UP - WAITING 30 MORE SECONDS ===');
      await page.waitForTimeout(30000);
      await page.screenshot({ path: '/tmp/snowforecast_50s.png', fullPage: true });
      console.log('Screenshot saved: /tmp/snowforecast_50s.png');
      pageText = await page.evaluate(() => document.body.innerText);
    }
    
    console.log('\n=== CHECKING FOR VERSION STRING ===');
    const versionMatch = pageText.match(/v\d{4}\.\d{2}\.\d{2}\.\d+/);
    if (versionMatch) {
      console.log('VERSION FOUND:', versionMatch[0]);
    } else {
      console.log('VERSION NOT FOUND - checking for any version-like text...');
      const anyVersion = pageText.match(/v\d+[\.\d]*/g);
      if (anyVersion) {
        console.log('Found version-like text:', anyVersion);
      } else {
        console.log('No version text found');
      }
    }
    
    console.log('\n=== CHECKING FOR LOADING STATE ===');
    if (pageText.includes('Loading forecast data')) {
      console.log('WARNING: "Loading forecast data..." text still present - data not loaded yet');
    } else {
      console.log('OK: No "Loading forecast data" text');
    }
    
    console.log('\n=== CHECKING FOR DEBUG SECTION ===');
    const dbPathMatch = pageText.match(/DB[:\s]+([^\n|]+)/i);
    const fcMatch = pageText.match(/FC[:\s]+(\d+)/i);
    const tMatch = pageText.match(/\bT[:\s]+(\d+)/i);
    const runTimeMatch = pageText.match(/Run[:\s]+([^\n|]+)/i);
    
    if (dbPathMatch) console.log('DB Path:', dbPathMatch[1].trim());
    else console.log('DB Path: NOT FOUND');
    
    if (fcMatch) console.log('FC (Forecast Count):', fcMatch[1]);
    else console.log('FC: NOT FOUND');
    
    if (tMatch) console.log('T (Terrain Count):', tMatch[1]);
    else console.log('T: NOT FOUND');
    
    if (runTimeMatch) console.log('Run Time:', runTimeMatch[1].trim());
    else console.log('Run Time: NOT FOUND');
    
    console.log('\n=== CHECKING MAIN METRICS ===');
    const snowBaseMatch = pageText.match(/Snow Base[^\d]*(\d+)/i);
    const newSnowMatch = pageText.match(/New Snow[^\d]*(\d+)/i);
    const probabilityMatch = pageText.match(/Probability[^\d]*(\d+)/i);
    const elevationMatch = pageText.match(/Elevation[^\d]*(\d+)/i);
    
    if (snowBaseMatch) console.log('Snow Base (cm):', snowBaseMatch[1]);
    else console.log('Snow Base: NOT FOUND');
    
    if (newSnowMatch) console.log('New Snow (cm):', newSnowMatch[1]);
    else console.log('New Snow: NOT FOUND');
    
    if (probabilityMatch) console.log('Probability (%):', probabilityMatch[1]);
    else console.log('Probability: NOT FOUND');
    
    if (elevationMatch) console.log('Elevation (m):', elevationMatch[1]);
    else console.log('Elevation: NOT FOUND');
    
    console.log('\n=== FULL PAGE TEXT ===');
    console.log(pageText);
    
    console.log('\n=== ERROR CHECK ===');
    if (pageText.toLowerCase().includes('error')) {
      console.log('WARNING: Page contains "error" text');
    }
    if (pageText.toLowerCase().includes('exception')) {
      console.log('WARNING: Page contains "exception" text');
    }
    if (pageText.toLowerCase().includes('failed')) {
      console.log('WARNING: Page contains "failed" text');
    }
    
  } catch (error) {
    console.error('Error during test:', error.message);
    try {
      await page.screenshot({ path: '/tmp/snowforecast_error.png', fullPage: true });
      console.log('Error screenshot saved');
    } catch (e) {}
  }
  
  await browser.close();
  console.log('\n=== TEST COMPLETE ===');
})();
