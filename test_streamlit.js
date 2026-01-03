const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  
  console.log('Navigating to https://snowforecast.streamlit.app...');
  
  try {
    await page.goto('https://snowforecast.streamlit.app', { waitUntil: 'domcontentloaded', timeout: 60000 });
    
    console.log('\n=== INITIAL PAGE LOAD (5 seconds) ===');
    await page.waitForTimeout(5000);
    await page.screenshot({ path: '/tmp/snowforecast_5s.png', fullPage: true });
    console.log('Screenshot saved: /tmp/snowforecast_5s.png');
    
    // Get initial text content
    let pageText = await page.textContent('body');
    console.log('\n--- Page text at 5s (first 2000 chars) ---');
    console.log(pageText.substring(0, 2000));
    
    console.log('\n=== WAITING 15 MORE SECONDS FOR FULL LOAD ===');
    await page.waitForTimeout(15000);
    await page.screenshot({ path: '/tmp/snowforecast_20s.png', fullPage: true });
    console.log('Screenshot saved: /tmp/snowforecast_20s.png');
    
    // Get full text content
    pageText = await page.textContent('body');
    
    console.log('\n=== CHECKING FOR VERSION STRING ===');
    const versionMatch = pageText.match(/v\d{4}\.\d{2}\.\d{2}\.\d+/);
    if (versionMatch) {
      console.log('VERSION FOUND:', versionMatch[0]);
    } else {
      console.log('VERSION NOT FOUND - checking for any version-like text...');
      const anyVersion = pageText.match(/v\d+\.\d+\.\d+[\.\d]*/);
      if (anyVersion) {
        console.log('Found similar version:', anyVersion[0]);
      }
    }
    
    console.log('\n=== CHECKING FOR LOADING STATE ===');
    if (pageText.includes('Loading forecast data')) {
      console.log('WARNING: "Loading forecast data..." text still present - data not loaded yet');
    } else {
      console.log('OK: No "Loading forecast data" text - data appears loaded');
    }
    
    console.log('\n=== CHECKING FOR DEBUG SECTION ===');
    const dbPathMatch = pageText.match(/DB:\s*([^\n]+)/);
    const fcMatch = pageText.match(/FC:\s*(\d+)/);
    const tMatch = pageText.match(/T:\s*(\d+)/);
    const runTimeMatch = pageText.match(/Run:\s*([^\n]+)/);
    
    if (dbPathMatch) console.log('DB Path:', dbPathMatch[1]);
    else console.log('DB Path: NOT FOUND');
    
    if (fcMatch) console.log('FC (Forecast Count):', fcMatch[1]);
    else console.log('FC: NOT FOUND');
    
    if (tMatch) console.log('T (Terrain Count):', tMatch[1]);
    else console.log('T: NOT FOUND');
    
    if (runTimeMatch) console.log('Run Time:', runTimeMatch[1]);
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
    }
    if (pageText.toLowerCase().includes('exception')) {
      console.log('WARNING: Page contains "exception" text');
    }
    if (pageText.toLowerCase().includes('failed')) {
      console.log('WARNING: Page contains "failed" text');
    }
    
  } catch (error) {
    console.error('Error during test:', error.message);
    await page.screenshot({ path: '/tmp/snowforecast_error.png', fullPage: true });
  }
  
  await browser.close();
  console.log('\n=== TEST COMPLETE ===');
})();
