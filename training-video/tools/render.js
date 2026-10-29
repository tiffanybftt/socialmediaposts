// Render video.html to a silent MP4, frame by frame, across several browser workers.
// usage: node tools/render.js [fps=30] [workers=4] [out=build/video_silent.mp4] [startSec] [endSec]
const { chromium } = require('playwright');
const { spawn, execFileSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const ROOT = path.resolve(__dirname, '..');
const fps = parseInt(process.argv[2] || '30', 10);
const workers = parseInt(process.argv[3] || '4', 10);
const out = path.resolve(ROOT, process.argv[4] || 'build/video_silent.mp4');
const timeline = JSON.parse(fs.readFileSync(path.join(ROOT, 'build', 'timeline.json')));
const from = Math.floor(parseFloat(process.argv[5] || '0') * fps);
const to = Math.ceil(parseFloat(process.argv[6] || timeline.duration) * fps);
const segDir = path.join(ROOT, 'build', 'segments');
fs.rmSync(segDir, { recursive: true, force: true });
fs.mkdirSync(segDir, { recursive: true });

async function renderChunk(id, a, b) {
  const seg = path.join(segDir, `seg_${String(id).padStart(2, '0')}.mp4`);
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'mjpeg', '-i', '-',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p', '-r', String(fps), '-g', String(fps * 2), '-threads', '2', seg],
    { stdio: ['pipe', 'inherit', 'inherit'] });
  const done = new Promise((res, rej) => { ff.on('close', c => (c === 0 ? res() : rej(new Error('ffmpeg exit ' + c)))); });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.on('pageerror', e => console.error('PAGE ERROR', e.message));
  await page.goto('file://' + path.join(ROOT, 'video.html'));
  await page.evaluate(() => window.READY);
  for (let f = a; f < b; f++) {
    await page.evaluate(t => window.render(t), f / fps);
    const buf = await page.screenshot({ type: 'jpeg', quality: 94 });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if ((f - a) % 300 === 0) console.log(`worker ${id}: ${f - a}/${b - a}`);
  }
  ff.stdin.end();
  await done;
  await browser.close();
  return seg;
}

(async () => {
  const total = to - from, per = Math.ceil(total / workers);
  const jobs = [];
  for (let i = 0; i < workers; i++) {
    const a = from + i * per, b = Math.min(to, a + per);
    if (a < b) jobs.push(renderChunk(i, a, b));
  }
  const segs = (await Promise.all(jobs)).sort();
  const list = path.join(segDir, 'list.txt');
  fs.writeFileSync(list, segs.map(s => `file '${s}'`).join('\n'));
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', out]);
  console.log('wrote', out, `${total} frames @ ${fps}fps`);
})().catch(e => { console.error(e); process.exit(1); });
