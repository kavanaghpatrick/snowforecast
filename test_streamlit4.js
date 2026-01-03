const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    javaScriptEnabled: true,
    userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
  });
  const page = await context.newPage();
  
  console.log('Navigating to https://snowforecast.streamlit.app...');
  
  try {
    await page.goto('https://snowforecast.streamlit.app', { 
      waitUntil: 'domcontentloaded', 
      timeout: 60000 
    });
    
    console.log('Waiting 25 seconds for full load...');
    await page.waitForTimeout(25000);
    
    // Take full page screenshot
    await page.screenshot({ path: '/tmp/snowforecast_full.png', fullPage: true });
    console.log('Full page screenshot saved: /tmp/snowforecast_full.png');
    
    // Get all visible text via innerHTML parsing
    const allText = await page.evaluate(() => {
      return document.body.innerText || document.body.textContent;
    });
    
    // Also get raw HTML to see all content
    const html = await page.content();
    
    console.log('\n=== EXTRACTED PAGE TEXT ===');
    console.log(allText);
    
    // Look for specific elements
    console.log('\n=== SPECIFIC ELEMENT SEARCH ===');
    
    // Find version text
    const versionEl = await page.evaluate(() => {
      const text = document.body.innerHTML;
      const match = text.match(/v\d{4}\.\d{2}\.\d{2}\.\d+[^<]*/);
      return match ? match[0] : null;
    });
    console.log('Version element:', versionEl);
    
    // Search for debug info in HTML
    console.log('\n=== DEBUG INFO FROM HTML ===');
    const dbMatch = html.match(/DB exists[:\s]*\w+/i);
    const cloudMatch = html.match(/Cloud[:\s]*\w+/i);
    const fcMatch = html.match(/FC[:\s]*\d+/i);
    const terrainMatch = html.match(/T[:\s]*\d+/i);
    
    console.log('DB exists:', dbMatch ? dbMatch[0] : 'NOT FOUND');
    console.log('Cloud:', cloudMatch ? cloudMatch[0] : 'NOT FOUND');
    console.log('FC:', fcMatch ? fcMatch[0] : 'NOT FOUND');
    console.log('T:', terrainMatch ? terrainMatch[0] : 'NOT FOUND');
    
    // Get all metric values
    console.log('\n=== METRIC VALUES FROM PAGE ===');
    const metrics = await page.evaluate(() => {
      const results = {};
      // Look for metric containers
      const metricLabels = document.querySelectorAll('[data-testid="stMetricLabel"]');
      const metricValues = document.querySelectorAll('[data-testid="stMetricValue"]');
      
      metricLabels.forEach((label, i) => {
        if (metricValues[i]) {
          results[label.innerText] = metricValues[i].innerText;
        }
      });
      return results;
    });
    console.log('Metrics found:', JSON.stringify(metrics, null, 2));
    
    // Scroll to bottom and take another screenshot
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(1000);
    await page.screenshot({ path: '/tmp/snowforecast_bottom.png', fullPage: false });
    console.log('\nBottom screenshot saved: /tmp/snowforecast_bottom.png');
    
  } catch (error) {
    console.error('Error:', error.message);
  }
  
  await browser.close();
  console.log('\n=== TEST COMPLETE ===');
})();
