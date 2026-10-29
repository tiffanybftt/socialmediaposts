// Save still frames of video.html at given times (seconds) so layout can be checked by eye.
// usage: node tools/snap.js <out_dir> <t1> [t2 ...]
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

(async () => {
  const [outDir, ...times] = process.argv.slice(2);
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.on('pageerror', e => console.error('PAGE ERROR', e.message));
  await page.goto('file://' + path.resolve(__dirname, '..', 'video.html'));
  await page.evaluate(() => window.READY);
  for (const t of times) {
    await page.evaluate(t => window.render(t), parseFloat(t));
    const file = path.join(outDir, `t${String(parseFloat(t).toFixed(1)).padStart(6, '0')}.png`);
    await page.screenshot({ path: file });
    console.log(file);
  }
  await browser.close();
})();
