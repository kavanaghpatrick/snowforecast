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
    // Wait for network idle to ensure JS has loaded
    await page.goto('https://snowforecast.streamlit.app', { 
      waitUntil: 'networkidle', 
      timeout: 60000 
    });
    
    console.log('\n=== INITIAL PAGE LOAD (waiting for app to render) ===');
    
    // Wait for streamlit to fully render - look for main content
    try {
      await page.waitForSelector('[data-testid="stMainBlockContainer"], [data-testid="stAppViewContainer"], .main', { timeout: 30000 });
      console.log('Streamlit container detected');
    } catch (e) {
      console.log('Streamlit container not detected, continuing anyway...');
    }
    
    // Wait additional time for data to load
    console.log('Waiting 5 seconds for initial render...');
    await page.waitForTimeout(5000);
    await page.screenshot({ path: '/tmp/snowforecast_5s.png', fullPage: true });
    console.log('Screenshot saved: /tmp/snowforecast_5s.png');
    
    // Get page content via evaluate for rendered content
    let pageText = await page.evaluate(() => document.body.innerText);
    console.log('\n--- Page text at 5s (first 2000 chars) ---');
    console.log(pageText.substring(0, 2000));
    
    console.log('\n=== WAITING 15 MORE SECONDS FOR FULL DATA LOAD ===');
    await page.waitForTimeout(15000);
    await page.screenshot({ path: '/tmp/snowforecast_20s.png', fullPage: true });
    console.log('Screenshot saved: /tmp/snowforecast_20s.png');
    
    // Get full text content
    pageText = await page.evaluate(() => document.body.innerText);
    
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
      console.log('OK: No "Loading forecast data" text - data appears loaded');
    }
    
    if (pageText.includes('Loading')) {
      console.log('Note: Page still contains "Loading" somewhere');
    }
    
    console.log('\n=== CHECKING FOR DEBUG SECTION ===');
    // More flexible patterns
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
    // Look for metric values
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
    
    console.log('\n=== FULL PAGE TEXT (for analysis) ===');
    console.log(pageText);
    
    console.log('\n=== CHECKING FOR ERROR MESSAGES ===');
    if (pageText.toLowerCase().includes('error')) {
      console.log('WARNING: Page contains "error" text');
      // Extract context around error
      const errorIdx = pageText.toLowerCase().indexOf('error');
      console.log('Context:', pageText.substring(Math.max(0, errorIdx - 50), errorIdx + 100));
    }
    if (pageText.toLowerCase().includes('exception')) {
      console.log('WARNING: Page contains "exception" text');
    }
    if (pageText.toLowerCase().includes('failed')) {
      console.log('WARNING: Page contains "failed" text');
    }
    
    // Get HTML for debugging
    console.log('\n=== HTML STRUCTURE ===');
    const htmlContent = await page.content();
    console.log('HTML length:', htmlContent.length);
    
  } catch (error) {
    console.error('Error during test:', error.message);
    await page.screenshot({ path: '/tmp/snowforecast_error.png', fullPage: true });
    console.log('Error screenshot saved: /tmp/snowforecast_error.png');
  }
  
  await browser.close();
  console.log('\n=== TEST COMPLETE ===');
})();
