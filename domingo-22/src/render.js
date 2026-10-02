const { chromium } = require('playwright-core');
const fs = require('fs');
const { spawn } = require('child_process');
(async () => {
  const tl = JSON.parse(fs.readFileSync('timeline.json', 'utf8'));
  const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || '/opt/pw-browsers/chromium-1194/chrome-linux/chrome', args: ['--allow-file-access-from-files'] });
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  page.on('console', m => console.log('page:', m.text()));
  page.on('pageerror', e => console.log('pageerror:', e.message));
  await page.goto('file://' + __dirname + '/video.html');
  await page.evaluate(tl => window.setup(tl), tl);
  const toBuf = d => Buffer.from(d.slice(d.indexOf(',') + 1), 'base64');
  const preview = process.argv[2];
  if (preview) {
    for (const s of preview.split(',')) {
      const f = Math.round(parseFloat(s) * tl.fps);
      fs.writeFileSync(`prev_${s}.png`, toBuf(await page.evaluate(f => window.frame(f), f)));
    }
  } else {
    const n = Math.round(tl.dur * tl.fps);
    const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(tl.fps), '-i', '-', '-i', 'audio.wav',
      '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
      '-shortest', '-movflags', '+faststart', 'out.mp4'], { stdio: ['pipe', 'inherit', 'inherit'] });
    for (let f = 0; f < n; f++) {
      const ok = ff.stdin.write(toBuf(await page.evaluate(f => window.frame(f), f)));
      if (!ok) await new Promise(r => ff.stdin.once('drain', r));
      if (f % 60 === 0) console.log('frame', f, '/', n);
    }
    ff.stdin.end();
    await new Promise(r => ff.on('close', r));
  }
  await browser.close();
})();
