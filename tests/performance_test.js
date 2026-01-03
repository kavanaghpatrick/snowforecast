const { chromium } = require('playwright');

async function runPerformanceTest() {
  const results = {
    initialLoadTime: null,
    dataLoadTime: null,
    interactionResponseTime: null,
    consoleErrors: [],
    networkErrors: [],
    slowRequests: [],
    streamlitErrors: [],
    performanceTargets: {
      initialLoad: { target: 5000, pass: false },
      dataLoad: { target: 3000, pass: false },
      interaction: { target: 1000, pass: false }
    }
  };

  console.log('Starting performance test for https://snowforecast.streamlit.app\n');
  console.log('=' .repeat(60));

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  // Track console errors
  page.on('console', msg => {
    if (msg.type() === 'error') {
      results.consoleErrors.push({
        text: msg.text(),
        location: msg.location()
      });
    }
  });

  // Track page errors
  page.on('pageerror', error => {
    results.consoleErrors.push({
      text: error.message,
      type: 'pageerror'
    });
  });

  // Track network requests
  const pendingRequests = new Map();
  const completedRequests = [];

  page.on('request', request => {
    pendingRequests.set(request.url(), {
      url: request.url(),
      startTime: Date.now(),
      method: request.method()
    });
  });

  page.on('response', response => {
    const url = response.url();
    const requestInfo = pendingRequests.get(url);
    if (requestInfo) {
      const duration = Date.now() - requestInfo.startTime;
      const status = response.status();

      completedRequests.push({
        url: url,
        status: status,
        duration: duration,
        method: requestInfo.method
      });

      // Check for errors (4xx, 5xx)
      if (status >= 400) {
        results.networkErrors.push({
          url: url,
          status: status,
          duration: duration
        });
      }

      // Check for slow requests (>2s)
      if (duration > 2000) {
        results.slowRequests.push({
          url: url,
          duration: duration,
          status: status
        });
      }

      pendingRequests.delete(url);
    }
  });

  page.on('requestfailed', request => {
    results.networkErrors.push({
      url: request.url(),
      error: request.failure()?.errorText || 'Unknown error',
      type: 'failed'
    });
  });

  try {
    // ==========================================
    // Test 1: Initial Page Load Time
    // ==========================================
    console.log('\n[Test 1] Measuring initial page load time...');
    const navigationStart = Date.now();

    // Navigate to page
    await page.goto('https://snowforecast.streamlit.app', {
      waitUntil: 'load',
      timeout: 120000
    });

    const loadEventTime = Date.now();
    console.log(`   Load event fired at: ${loadEventTime - navigationStart}ms`);

    // Wait for network to be idle first
    await page.waitForLoadState('networkidle', { timeout: 60000 });

    const networkIdleTime = Date.now();
    console.log(`   Network idle at: ${networkIdleTime - navigationStart}ms`);

    // Find the Streamlit iframe and wait for content
    const frames = page.frames();
    let streamlitFrame = null;
    for (const frame of frames) {
      const url = frame.url();
      if (url.includes('/~/+/')) {
        streamlitFrame = frame;
        console.log(`   Found Streamlit frame: ${url.substring(0, 60)}`);
        break;
      }
    }

    if (streamlitFrame) {
      // Wait for actual content to appear in iframe
      try {
        await streamlitFrame.waitForSelector('[data-testid="stSelectbox"], text=Snow Forecast', { timeout: 60000 });
        console.log('   Content detected in Streamlit frame');
      } catch (e) {
        console.log(`   Waiting for content in frame timed out: ${e.message}`);
      }
    }

    const contentVisibleTime = Date.now();
    results.initialLoadTime = contentVisibleTime - navigationStart;
    results.performanceTargets.initialLoad.pass = results.initialLoadTime < 5000;

    console.log(`   Initial load time (to content visible): ${results.initialLoadTime}ms`);
    console.log(`   Target: <5000ms | ${results.performanceTargets.initialLoad.pass ? 'PASS' : 'FAIL'}`);

    // ==========================================
    // Test 2: Data Loading Time
    // ==========================================
    console.log('\n[Test 2] Measuring data load time...');
    const dataLoadStart = Date.now();

    // Take screenshot to verify content
    await page.screenshot({ path: '/tmp/snowforecast_content.png', fullPage: true });

    if (streamlitFrame) {
      // Wait for data indicators in iframe
      try {
        await streamlitFrame.waitForSelector('text=Snow Base', { timeout: 30000 });
        console.log('   Snow Base data visible');
      } catch (e) {
        console.log(`   Could not find Snow Base text: ${e.message}`);
      }
    }

    const dataLoadEnd = Date.now();
    results.dataLoadTime = dataLoadEnd - dataLoadStart;
    results.performanceTargets.dataLoad.pass = results.dataLoadTime < 3000;

    console.log(`   Data load time: ${results.dataLoadTime}ms`);
    console.log(`   Target: <3000ms | ${results.performanceTargets.dataLoad.pass ? 'PASS' : 'FAIL'}`);

    // ==========================================
    // Test 3: Interaction Response Time
    // ==========================================
    console.log('\n[Test 3] Measuring interaction response time...');

    // Take screenshot before interaction
    await page.screenshot({ path: '/tmp/snowforecast_before_interaction.png', fullPage: true });

    // Find selectboxes in the Streamlit frame
    let selectboxes = [];
    if (streamlitFrame) {
      selectboxes = await streamlitFrame.$$('[data-testid="stSelectbox"]');
      console.log(`   Found ${selectboxes.length} selectbox(es) in Streamlit frame`);
    }

    if (selectboxes.length === 0) {
      // Fallback: try main page
      selectboxes = await page.$$('[data-testid="stSelectbox"]');
      console.log(`   Found ${selectboxes.length} selectbox(es) in main page`);
    }

    if (selectboxes.length >= 2) {
      const skiAreaSelect = selectboxes[1];
      const interactionStart = Date.now();

      try {
        await skiAreaSelect.click();
        await page.waitForTimeout(500);

        // Look for options in frame
        let options = [];
        if (streamlitFrame) {
          options = await streamlitFrame.$$('[role="option"]');
        }
        if (options.length === 0) {
          options = await page.$$('[role="option"]');
        }
        console.log(`   Found ${options.length} options in dropdown`);

        if (options.length > 1) {
          const currentText = await skiAreaSelect.textContent();
          console.log(`   Current selection: ${currentText.trim()}`);

          await options[1].click();
          await page.waitForLoadState('networkidle', { timeout: 15000 });

          const interactionEnd = Date.now();
          results.interactionResponseTime = interactionEnd - interactionStart;
          results.performanceTargets.interaction.pass = results.interactionResponseTime < 1000;

          console.log(`   Interaction response time: ${results.interactionResponseTime}ms`);
          console.log(`   Target: <1000ms | ${results.performanceTargets.interaction.pass ? 'PASS' : 'FAIL'}`);
        } else {
          await page.keyboard.press('Escape');
          results.interactionResponseTime = 'N/A - not enough options';
          console.log('   Not enough options to test interaction');
        }
      } catch (e) {
        console.log(`   Interaction test error: ${e.message}`);
        results.interactionResponseTime = `Error: ${e.message}`;
      }
    } else if (selectboxes.length === 1) {
      const interactionStart = Date.now();
      try {
        await selectboxes[0].click();
        await page.waitForTimeout(500);

        let options = [];
        if (streamlitFrame) {
          options = await streamlitFrame.$$('[role="option"]');
        }
        if (options.length === 0) {
          options = await page.$$('[role="option"]');
        }
        console.log(`   Found ${options.length} options in dropdown`);

        if (options.length > 1) {
          await options[1].click();
          await page.waitForLoadState('networkidle', { timeout: 15000 });

          const interactionEnd = Date.now();
          results.interactionResponseTime = interactionEnd - interactionStart;
          results.performanceTargets.interaction.pass = results.interactionResponseTime < 1000;

          console.log(`   Interaction response time: ${results.interactionResponseTime}ms`);
          console.log(`   Target: <1000ms | ${results.performanceTargets.interaction.pass ? 'PASS' : 'FAIL'}`);
        } else {
          await page.keyboard.press('Escape');
          results.interactionResponseTime = 'N/A - not enough options';
        }
      } catch (e) {
        console.log(`   Interaction test error: ${e.message}`);
        results.interactionResponseTime = `Error: ${e.message}`;
      }
    } else {
      console.log('   No selectbox found for interaction test');
      results.interactionResponseTime = 'N/A - no selectbox found';
    }

    // ==========================================
    // Test 4: Check for Streamlit Error Messages
    // ==========================================
    console.log('\n[Test 4] Checking for Streamlit error messages...');

    let bodyText = '';
    if (streamlitFrame) {
      try {
        bodyText = await streamlitFrame.evaluate(() => document.body?.innerText || '');
      } catch (e) {
        bodyText = await page.evaluate(() => document.body?.innerText || '');
      }
    } else {
      bodyText = await page.evaluate(() => document.body?.innerText || '');
    }

    const errorPatterns = ['Traceback', 'Exception:', 'StreamlitAPIException'];
    for (const pattern of errorPatterns) {
      if (bodyText.includes(pattern)) {
        const lines = bodyText.split('\n');
        for (const line of lines) {
          if (line.includes(pattern)) {
            results.streamlitErrors.push(line.substring(0, 200));
          }
        }
      }
    }

    // Check for error elements in frame
    let errorElements = [];
    if (streamlitFrame) {
      errorElements = await streamlitFrame.$$('.stException, .stError, [data-testid="stException"]');
    }
    if (errorElements.length === 0) {
      errorElements = await page.$$('.stException, .stError, [data-testid="stException"]');
    }
    for (const el of errorElements) {
      const text = await el.textContent();
      results.streamlitErrors.push(text.substring(0, 200));
    }

    if (bodyText.includes('DB exists: False')) {
      console.log('   Note: DB exists: False - database may not be connected');
    }

    console.log(`   Streamlit errors found: ${results.streamlitErrors.length}`);

    // Final screenshot
    await page.screenshot({ path: '/tmp/snowforecast_performance.png', fullPage: true });
    console.log('\n   Final screenshot saved: /tmp/snowforecast_performance.png');

  } catch (error) {
    console.error(`\nTest error: ${error.message}`);
    results.testError = error.message;

    try {
      await page.screenshot({ path: '/tmp/snowforecast_error.png', fullPage: true });
      console.log('   Error screenshot saved: /tmp/snowforecast_error.png');
    } catch (e) {}
  } finally {
    await browser.close();
  }

  // ==========================================
  // Summary Report
  // ==========================================
  console.log('\n' + '=' .repeat(60));
  console.log('PERFORMANCE TEST SUMMARY');
  console.log('=' .repeat(60));

  console.log('\n--- Timing Measurements ---');
  console.log(`Initial Page Load:     ${results.initialLoadTime !== null ? results.initialLoadTime + 'ms' : 'N/A'} (target: <5000ms) ${results.performanceTargets.initialLoad.pass ? '[PASS]' : '[FAIL]'}`);
  console.log(`Data Load Time:        ${results.dataLoadTime !== null ? results.dataLoadTime + 'ms' : 'N/A'} (target: <3000ms) ${results.performanceTargets.dataLoad.pass ? '[PASS]' : '[FAIL]'}`);
  console.log(`Interaction Response:  ${typeof results.interactionResponseTime === 'number' ? results.interactionResponseTime + 'ms (target: <1000ms) ' + (results.performanceTargets.interaction.pass ? '[PASS]' : '[FAIL]') : results.interactionResponseTime}`);

  console.log('\n--- Console Errors ---');
  if (results.consoleErrors.length === 0) {
    console.log('No JavaScript console errors detected');
  } else {
    console.log(`${results.consoleErrors.length} console error(s):`);
    results.consoleErrors.slice(0, 10).forEach((err, i) => {
      console.log(`  ${i + 1}. ${err.text.substring(0, 150)}`);
    });
    if (results.consoleErrors.length > 10) {
      console.log(`  ... and ${results.consoleErrors.length - 10} more`);
    }
  }

  console.log('\n--- Network Errors (4xx/5xx) ---');
  if (results.networkErrors.length === 0) {
    console.log('No network errors detected');
  } else {
    console.log(`${results.networkErrors.length} network error(s):`);
    results.networkErrors.slice(0, 10).forEach((err, i) => {
      console.log(`  ${i + 1}. ${err.status || err.error} - ${err.url.substring(0, 80)}`);
    });
  }

  console.log('\n--- Slow Requests (>2s) ---');
  if (results.slowRequests.length === 0) {
    console.log('No slow requests detected');
  } else {
    console.log(`${results.slowRequests.length} slow request(s):`);
    results.slowRequests.sort((a, b) => b.duration - a.duration);
    results.slowRequests.slice(0, 10).forEach((req, i) => {
      console.log(`  ${i + 1}. ${req.duration}ms - ${req.url.substring(0, 70)}`);
    });
    if (results.slowRequests.length > 10) {
      console.log(`  ... and ${results.slowRequests.length - 10} more`);
    }
  }

  console.log('\n--- Streamlit Errors ---');
  if (results.streamlitErrors.length === 0) {
    console.log('No Streamlit error messages found');
  } else {
    console.log(`${results.streamlitErrors.length} Streamlit error(s):`);
    results.streamlitErrors.forEach((err, i) => {
      console.log(`  ${i + 1}. ${err}`);
    });
  }

  console.log('\n--- Overall Result ---');
  const allPassed = results.performanceTargets.initialLoad.pass &&
                    results.performanceTargets.dataLoad.pass &&
                    results.networkErrors.length === 0 &&
                    results.streamlitErrors.length === 0;

  console.log(allPassed ? 'ALL TESTS PASSED' : 'SOME TESTS FAILED OR ISSUES DETECTED');

  const totalLoadTime = (results.initialLoadTime || 0);
  console.log(`\nTotal time to content visible: ${totalLoadTime}ms`);

  console.log('\n--- Raw Results (JSON) ---');
  console.log(JSON.stringify(results, null, 2));
}

runPerformanceTest().catch(console.error);
