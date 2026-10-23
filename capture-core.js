(() => {
  /* Where the Captrail model lives. Leave '' until the model is deployed. */
  const ANALYZE_ENDPOINT = '/api/analyze';

  const $ = id => document.getElementById(id);
  const video = $('video'), view = $('view'), overlay = $('overlay'), stage = $('stage'), hint = $('hint');
  const vctx = view.getContext('2d'), octx = overlay.getContext('2d');
  const store = {
    get(k) { try { return JSON.parse(localStorage.getItem('captrail.' + k)); } catch { return null; } },
    set(k, v) { try { localStorage.setItem('captrail.' + k, JSON.stringify(v)); } catch {} },
  };

  let stream = null, frames = [], sel = null, mode = 'none', pts = [], look = 'off', analyzing = false, stacking = false;
  let calib = store.get('calib'); // { umPerPx, w, h, at }

  window.CapTrailCapture = { original: async () => { if (!sel) throw new Error('Capture or select a microscope image first.'); const f=sel.calibScale && sel.pair ? sel.pair : sel; const blob=await new Promise(resolve=>f.canvas.toBlob(resolve,'image/png')); return new File([blob],(f.name||'microscope-capture').replace(/\.[^.]+$/,'')+'.png',{type:'image/png'}); } };
  /* ---------- status helpers ---------- */
  function setStatus(el, kind, text) { el.className = 'status ' + (kind || ''); el.querySelector('span').textContent = text; }
  function showHint(t) { hint.hidden = !t; hint.textContent = t || ''; }

  /* ---------- camera ---------- */
  const scopeRx = /microscope|scope|digital|usb|uvc/i, builtInRx = /facetime|integrated|built-in|front|back|virtual|obs/i;
  async function listDevices(preferId) {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
    const ds = $('devSel');
    ds.innerHTML = devs.map((d, i) => `<option value="${escHtml(d.deviceId)}">${escHtml(d.label || 'Camera ' + (i + 1))}</option>`).join('');
    ds.disabled = !devs.length;
    const saved = preferId || store.get('deviceId');
    const pick = devs.find(d => d.deviceId === saved) || devs.find(d => scopeRx.test(d.label) && !builtInRx.test(d.label)) || devs[0];
    if (pick) ds.value = pick.deviceId;
    return pick;
  }
  async function startCamera(deviceId) {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      setStatus($('connStatus'), 'err', 'Camera access needs https:// or localhost.'); return;
    }
    stopCamera();
    const [w, h] = $('resSel').value.split('x').map(Number);
    try {
      setStatus($('connStatus'), '', 'Connecting…');
      stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: w }, height: { ideal: h } }, audio: false });
      const track = stream.getVideoTracks()[0];
      const label = track.label || 'Camera';
      const id = track.getSettings().deviceId;
      // First connect only reveals device names after permission, so re-pick the microscope if we grabbed a built-in camera.
      if (!deviceId) {
        const pick = await listDevices();
        if (pick && pick.deviceId !== id) return startCamera(pick.deviceId);
      } else await listDevices(id);
      store.set('deviceId', id);
      video.srcObject = stream; await video.play();
      const s = track.getSettings();
      const looksLikeScope = scopeRx.test(label) && !builtInRx.test(label);
      setStatus($('connStatus'), looksLikeScope ? 'ok' : 'warn', looksLikeScope ? `${label} · ${s.width}×${s.height}` : `${label} · ${s.width}×${s.height}. Is this the microscope? Pick it above.`);
      $('btnConnect').textContent = 'Reconnect';
      goLive();
      setupCameraControls(track, id);
    } catch (e) {
      setStatus($('connStatus'), 'err', e.name === 'NotAllowedError' ? 'Camera permission was blocked. Allow it in the address bar.' : e.name === 'NotFoundError' ? 'No camera found. Is the microscope plugged in?' : 'Could not start: ' + e.message);
    }
  }
  function stopCamera() { if (stream) stream.getTracks().forEach(t => t.stop()); stream = null; }

  /* ---------- camera settings (like Attune's scope_settings.json) ----------
     Chrome and Edge expose UVC controls through getCapabilities()/applyConstraints(); Safari doesn't.
     Per camera we store the camera's own defaults (captured on first connect) and the chosen values,
     so reconnecting re-applies the same values instead of compounding "50% above default". */
  const CAM_SLIDERS = [
    { key: 'brightness', label: 'Brightness' },
    { key: 'exposureTime', label: 'Exposure', mode: 'exposureMode' },
    { key: 'saturation', label: 'Saturation' },
    { key: 'contrast', label: 'Contrast' },
  ];
  const CAM_KEYS = ['colorTemperature', 'exposureTime', 'saturation', 'contrast', 'brightness', 'sharpness'];
  let camTrack = null, camId = null, camCaps = {}, camApplyTimer;
  const camStore = () => store.get('camera') || {};
  const clampStep = (v, c) => {
    if (!c || c.min == null) return v;
    const st = c.step || 1, x = Math.min(c.max, Math.max(c.min, v));
    return Math.round((x - c.min) / st) * st + c.min;
  };
  async function applyCam(constraints) {
    // One control at a time: a camera that rejects one setting shouldn't block the rest.
    const failed = [];
    for (const c of constraints) {
      try { await camTrack.applyConstraints({ advanced: [c] }); } catch { failed.push(Object.keys(c).join('+')); }
    }
    return failed;
  }
  function camConstraints(v) {
    const caps = camCaps, out = [];
    if (caps.whiteBalanceMode?.includes('manual') && caps.colorTemperature && v.colorTemperature != null)
      out.push({ whiteBalanceMode: 'manual', colorTemperature: v.colorTemperature });
    if (caps.exposureMode?.includes('manual') && caps.exposureTime && v.exposureTime != null)
      out.push({ exposureMode: 'manual', exposureTime: v.exposureTime });
    for (const k of ['saturation', 'contrast', 'brightness', 'sharpness']) if (caps[k] && v[k] != null) out.push({ [k]: v[k] });
    return out;
  }
  async function setupCameraControls(track, deviceId) {
    camTrack = track; camId = deviceId || 'default';
    camCaps = typeof track.getCapabilities === 'function' ? track.getCapabilities() : {};
    const supported = CAM_KEYS.filter(k => camCaps[k] && camCaps[k].max > camCaps[k].min);
    $('camReset').disabled = !supported.length;
    if (!supported.length) {
      renderCamSliders({});
      setStatus($('camStatus'), 'warn', typeof track.getCapabilities === 'function'
        ? 'This camera doesn’t expose adjustable settings to the browser.'
        : 'This browser can’t adjust camera settings (Safari doesn’t). Use Chrome or Edge for locked color and exposure.');
      return;
    }
    const all = camStore(), mine = all[camId] || {};
    const now = track.getSettings();
    if (!mine.defaults) {
      // First connect: remember the camera's own settings, then derive the Attune-style starting values.
      mine.defaults = {};
      for (const k of CAM_KEYS) if (now[k] != null) mine.defaults[k] = now[k];
      for (const k of ['whiteBalanceMode', 'exposureMode']) if (now[k]) mine.defaults[k] = now[k];
    }
    if (!mine.values) {
      const c = camCaps, d = mine.defaults, v = {};
      if (c.colorTemperature) v.colorTemperature = clampStep(6000, c.colorTemperature);
      if (c.exposureTime) v.exposureTime = clampStep(d.exposureTime ?? (c.exposureTime.min + c.exposureTime.max) / 2, c.exposureTime);
      if (c.saturation) v.saturation = c.saturation.max;
      if (c.contrast) v.contrast = clampStep((d.contrast ?? (c.contrast.min + c.contrast.max) / 2) * 1.5, c.contrast);
      if (c.brightness) v.brightness = clampStep(c.brightness.min + (c.brightness.max - c.brightness.min) * .25, c.brightness);
      if (c.sharpness) v.sharpness = c.sharpness.max;
      mine.values = v;
    }
    all[camId] = mine; store.set('camera', all);
    renderCamSliders(mine.values);
    const failed = await applyCam(camConstraints(mine.values));
    const unsupported = ['whiteBalanceMode', 'exposureMode'].filter(k => !camCaps[k]?.includes?.('manual')).map(k => k === 'whiteBalanceMode' ? 'manual white balance' : 'manual exposure');
    const parts = [];
    if (failed.length) parts.push(`camera rejected: ${failed.join(', ')}`);
    if (unsupported.length) parts.push(`not supported: ${unsupported.join(', ')}`);
    setStatus($('camStatus'), parts.length ? 'warn' : 'ok', parts.length ? `Applied what this camera supports (${parts.join('; ')}).` : 'Settings applied: white balance 6000 K, manual exposure.');
  }
  function renderCamSliders(values) {
    $('camSliders').innerHTML = CAM_SLIDERS.map(({ key, label }) => {
      const c = camCaps[key], ok = c && c.max > c.min;
      const v = ok ? values[key] ?? c.min : '';
      return `<div class="slider${ok ? '' : ' off'}"><div class="top"><label for="cam-${key}">${label}</label><output id="camv-${key}">${ok ? v : 'not supported'}</output></div>
        <input type="range" id="cam-${key}" ${ok ? `min="${c.min}" max="${c.max}" step="${c.step || 1}" value="${v}"` : 'disabled'}></div>`;
    }).join('');
    CAM_SLIDERS.forEach(({ key, mode }) => {
      const el = $('cam-' + key);
      if (el.disabled) return;
      el.oninput = () => {
        $('camv-' + key).textContent = el.value;
        const all = camStore(), mine = all[camId] || { values: {} };
        mine.values = { ...mine.values, [key]: Number(el.value) };
        all[camId] = mine; store.set('camera', all);
        clearTimeout(camApplyTimer);
        camApplyTimer = setTimeout(async () => {
          const c = mode && camCaps[mode]?.includes('manual') ? { [mode]: 'manual', [key]: Number(el.value) } : { [key]: Number(el.value) };
          const failed = await applyCam([c]);
          if (failed.length) setStatus($('camStatus'), 'warn', `The camera rejected ${key}.`);
        }, 120);
      };
    });
  }
  $('camReset').onclick = async () => {
    if (!camTrack) return;
    const all = camStore(), mine = all[camId];
    const d = mine?.defaults || {}, c = [];
    if (camCaps.whiteBalanceMode?.includes('continuous')) c.push({ whiteBalanceMode: 'continuous' });
    if (camCaps.exposureMode?.includes('continuous')) c.push({ exposureMode: 'continuous' });
    for (const k of ['saturation', 'contrast', 'brightness', 'sharpness', 'colorTemperature']) if (camCaps[k] && d[k] != null) c.push({ [k]: d[k] });
    await applyCam(c);
    if (mine) { delete mine.values; all[camId] = mine; store.set('camera', all); }
    const now = camTrack.getSettings(), v = {};
    for (const k of CAM_KEYS) if (now[k] != null) v[k] = now[k];
    renderCamSliders(v);
    setStatus($('camStatus'), 'ok', 'Back to the camera’s defaults (auto white balance and exposure). Reconnect to re-apply the scan settings.');
  };
  $('btnConnect').onclick = () => startCamera($('devSel').disabled ? null : $('devSel').value);
  $('devSel').onchange = e => startCamera(e.target.value);
  $('resSel').onchange = () => stream && startCamera($('devSel').value);
  navigator.mediaDevices?.addEventListener?.('devicechange', () => stream && listDevices());

  /* ---------- views ---------- */
  function goLive() {
    sel = null; setMode('none');
    $('empty').hidden = true; video.hidden = false; view.hidden = overlay.hidden = true;
    stage.style.aspectRatio = (video.videoWidth || 16) + '/' + (video.videoHeight || 9);
    renderHud(); renderGallery(); syncButtons(); renderMeasures(); renderFindings(); renderSim();
  }
  function showFrame(f) {
    sel = f; setMode('none');
    $('empty').hidden = true; video.hidden = true; view.hidden = overlay.hidden = false;
    view.width = overlay.width = f.w; view.height = overlay.height = f.h;
    stage.style.aspectRatio = f.w + '/' + f.h;
    drawFrame(); renderHud(); renderGallery(); syncButtons(); renderMeasures(); renderFindings(); renderSim();
  }
  // The simulated view belongs to a Best of 8 capture (and its zoom). It is only ever drawn on its own canvas.
  function renderSim() {
    const sim = sel?.sim || sel?.pair?.sim;
    $('simFig').hidden = !sim;
    $('dual').classList.toggle('has-sim', !!sim);
    if (sim) $('simCanvas').getContext('2d').drawImage(sim, 0, 0);
  }
  function drawFrame() {
    if (!sel) return;
    vctx.drawImage(lookCanvas(sel, look), 0, 0);
    drawOverlay();
  }
  function renderHud() {
    const h = $('hud'); const bits = [];
    if (!sel && stream) bits.push('<span class="live">Live</span>');
    if (sel) bits.push(`<span>${escHtml(sel.label || (sel.source === 'scope' ? 'Captured' : 'Uploaded'))} · ${sel.w}×${sel.h}</span>`, `<span>Sharpness ${sel.score}</span>`);
    if (look !== 'off' && sel) bits.push(`<span>${LOOKS[look].label} view</span>`);
    if (sel?.note) bits.push(`<span class="warn">${escHtml(sel.note)}</span>`);
    h.innerHTML = bits.join('');
  }
  function syncButtons() {
    const live = !!stream && !sel;
    $('btnCapture').disabled = $('btnBurst').disabled = $('btnStack').disabled = !live || stacking;
    $('btnBurst').title = 'Take 8 frames, merge them, and auto enhance';
    $('btnLive').disabled = !stream || !sel;
    ['btnEnhance', 'btnMeasure', 'btnDownload', 'btnDelete', 'btnCal'].forEach(id => $(id).disabled = !sel);
    $('btnRaw').disabled = !sel?.pair || !frames.includes(sel.pair);
    $('rawLbl').textContent = sel?.calibScale ? 'Show original' : 'Show enhanced';
    $('btnAnalyze').disabled = !sel || analyzing;
    $('btnTrained').disabled = !sel || analyzing;
    $('btnEnhance').classList.toggle('on', look !== 'off');
    $('enhanceLbl').textContent = 'Enhance: ' + LOOKS[look].short;
    $('btnMeasure').classList.toggle('on', mode === 'measure');
  }

  /* ---------- capture ---------- */
  function grab() {
    const c = document.createElement('canvas'); c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext('2d').drawImage(video, 0, 0); return c;
  }
  function addFrame(canvas, source, name) {
    const f = { id: crypto.randomUUID?.() || String(Date.now() + Math.random()), canvas, w: canvas.width, h: canvas.height, source, name,
      at: new Date(), score: sharpness(canvas), measures: [], ai: null, thumb: thumbOf(canvas) };
    frames.push(f); return f;
  }
  $('btnCapture').onclick = () => { if (stream && !sel) showFrame(addFrame(grab(), 'scope')); };
  // Best of 8: grab 8 frames, align all of them to the sharpest and merge with a per-pixel median,
  // then run Auto enhance and show the result as the main capture.
  const ZOOM_MIN_SHARPNESS = 300;
  $('btnBurst').onclick = async () => {
    if (!stream || sel || stacking) return;
    stacking = true; syncButtons();
    try {
      const shots = [];
      for (let i = 0; i < 8; i++) { showHint(`Hold still… ${i + 1}/8`); shots.push(grab()); await new Promise(r => setTimeout(r, 140)); }
      const scored = shots.map(c => ({ c, score: sharpness(c) })).sort((a, b) => b.score - a.score);
      const W = scored[0].c.width, H = scored[0].c.height;
      const { canvas, shifts } = await mergeAligned(scored.map(k => pixelsOf(k.c)), W, H);
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      const orig = addFrame(canvas, 'scope', `captrail-best8-${stamp}.png`);
      orig.shifts = shifts;
      showHint('Preparing capture…'); await tick();
       // illustration only: not a frame, never saved or analyzed
      look = 'off';
      // Only enhance sharp captures; otherwise keep the original and say why.
      if (orig.score <= ZOOM_MIN_SHARPNESS) { orig.note = 'Too blurry to zoom, retake'; showHint(''); showFrame(orig); return; }
      showHint('Auto enhancing…'); await tick();
      const enh = addEnhanced(orig, autoEnhance(canvas));
      showHint('');
      showFrame(enh || orig);
    } catch (e) {
      showHint('');
      setStatus($('connStatus'), 'err', 'Capture failed: ' + e.message);
    } finally {
      stacking = false; syncButtons();
    }
  };
  $('btnLive').onclick = goLive;

  /* ---------- Stack 40: align the sharpest 20 of 40 frames and take the per-pixel median ---------- */
  const STACK_GRAB = 40, STACK_KEEP = 20, STACK_MS = 2000;
  const tick = () => new Promise(r => setTimeout(r, 0));
  // In-place iterative radix-2 complex FFT (n must be a power of two). inv = true for the inverse (unscaled).
  function fft(re, im, inv) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (inv ? 2 : -2) * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = i + k, b = a + len / 2;
          const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
          const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
        }
      }
    }
  }
  function fft2(re, im, N, inv) {
    const r = new Float64Array(N), i = new Float64Array(N);
    for (let y = 0; y < N; y++) { const o = y * N; r.set(re.subarray(o, o + N)); i.set(im.subarray(o, o + N)); fft(r, i, inv); re.set(r, o); im.set(i, o); }
    for (let x = 0; x < N; x++) {
      for (let y = 0; y < N; y++) { r[y] = re[y * N + x]; i[y] = im[y * N + x]; }
      fft(r, i, inv);
      for (let y = 0; y < N; y++) { re[y * N + x] = r[y]; im[y * N + x] = i[y]; }
    }
  }
  // Spectrum of a centered N×N grayscale patch with a Hann window (reduces edge artifacts).
  function patchSpectrum(data, W, H, N) {
    const x0 = (W - N) >> 1, y0 = (H - N) >> 1, re = new Float64Array(N * N), im = new Float64Array(N * N);
    let mean = 0;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const p = ((y0 + y) * W + x0 + x) * 4;
      mean += re[y * N + x] = .299 * data[p] + .587 * data[p + 1] + .114 * data[p + 2];
    }
    mean /= N * N;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const w = (.5 - .5 * Math.cos(2 * Math.PI * x / (N - 1))) * (.5 - .5 * Math.cos(2 * Math.PI * y / (N - 1)));
      re[y * N + x] = (re[y * N + x] - mean) * w;
    }
    fft2(re, im, N, false);
    return { re, im };
  }
  // Phase correlation: returns the integer shift (dx, dy) of a frame relative to ref, i.e. frame(x + dx, y + dy) ≈ ref(x, y).
  function phaseShift(ref, cur, N) {
    const re = new Float64Array(N * N), im = new Float64Array(N * N);
    for (let i = 0; i < N * N; i++) {
      const a = ref.re[i] * cur.re[i] + ref.im[i] * cur.im[i], b = ref.im[i] * cur.re[i] - ref.re[i] * cur.im[i];
      const m = Math.hypot(a, b) || 1; re[i] = a / m; im[i] = b / m;
    }
    fft2(re, im, N, true);
    let best = -Infinity, bi = 0;
    for (let i = 0; i < N * N; i++) if (re[i] > best) { best = re[i]; bi = i; }
    let dx = bi % N, dy = Math.floor(bi / N);
    if (dx > N / 2) dx -= N; if (dy > N / 2) dy -= N;
    return { dx: -dx, dy: -dy };
  }
  function insertionMedian(buf, n) {
    for (let i = 1; i < n; i++) { const v = buf[i]; let j = i - 1; while (j >= 0 && buf[j] > v) { buf[j + 1] = buf[j]; j--; } buf[j + 1] = v; }
    return n & 1 ? buf[n >> 1] : (buf[(n >> 1) - 1] + buf[n >> 1]) / 2;
  }
  // Align frames (RGBA arrays, sharpest first) to frame 0 with phase correlation, then take the per-pixel
  // median of the aligned frames: moving glare and noise drop out.
  async function mergeAligned(data, W, H) {
    showHint('Aligning frames…'); await tick();
    let N = 1; while (N * 2 <= Math.min(W, H, 512)) N *= 2;
    const refSpec = patchSpectrum(data[0], W, H, N), shifts = [{ dx: 0, dy: 0 }];
    for (let k = 1; k < data.length; k++) {
      shifts.push(phaseShift(refSpec, patchSpectrum(data[k], W, H, N), N));
      if (k % 4 === 0) { showHint(`Aligning frames… ${k}/${data.length - 1}`); await tick(); }
    }
    const out = new ImageData(W, H), od = out.data, buf = new Float32Array(data.length);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const o = (y * W + x) * 4;
        for (let ch = 0; ch < 3; ch++) {
          let n = 0;
          for (let k = 0; k < data.length; k++) {
            const sx = x + shifts[k].dx, sy = y + shifts[k].dy;
            if (sx >= 0 && sy >= 0 && sx < W && sy < H) buf[n++] = data[k][(sy * W + sx) * 4 + ch];
          }
          od[o + ch] = insertionMedian(buf, n);
        }
        od[o + 3] = 255;
      }
      if (y % 60 === 0) { showHint(`Merging ${data.length} frames… ${Math.round(y / H * 100)}%`); await tick(); }
    }
    const canvas = document.createElement('canvas'); canvas.width = W; canvas.height = H;
    canvas.getContext('2d').putImageData(out, 0, 0);
    return { canvas, shifts };
  }
  const pixelsOf = c => c.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, c.width, c.height).data;
  $('btnStack').onclick = async () => {
    if (!stream || sel || stacking) return;
    stacking = true; syncButtons();
    try {
      // 1-2. Grab 40 frames over ~2 s, keeping only the 20 sharpest as we go (saves memory).
      let kept = [];
      const gap = STACK_MS / STACK_GRAB;
      for (let i = 0; i < STACK_GRAB; i++) {
        const t0 = performance.now();
        showHint(`Hold still… ${i + 1}/${STACK_GRAB}`);
        const c = grab();
        kept.push({ c, score: sharpness(c) });
        if (kept.length > STACK_KEEP) { kept.sort((a, b) => b.score - a.score); kept.length = STACK_KEEP; }
        await new Promise(r => setTimeout(r, Math.max(0, gap - (performance.now() - t0))));
      }
      kept.sort((a, b) => b.score - a.score);
      const W = kept[0].c.width, H = kept[0].c.height;
      const data = kept.map(k => pixelsOf(k.c));
      kept = null;
      const { canvas: c, shifts } = await mergeAligned(data, W, H);
      // Show as a new capture with the Capillary enhance applied; Save writes _raw + _capillary.
      const f = addFrame(c, 'scope', `captrail-stacked-${new Date().toISOString().replace(/[:.]/g, '-')}.png`);
      f.label = `Stacked (${data.length} frames)`;
      f.shifts = shifts;
      look = 'capillary';
      showHint('');
      showFrame(f);
    } catch (e) {
      showHint('');
      setStatus($('connStatus'), 'err', 'Stacking failed: ' + e.message);
    } finally {
      stacking = false; syncButtons();
    }
  };
  // Uploads: every image is auto enhanced; afterwards "Download all" zips the batch (raw + enhanced).
  let lastBatch = [];
  $('fileIn').onchange = async e => {
    const files = [...e.target.files].slice(0,12);
    e.target.value = '';
    if (!files.length) return;
    let last, batch = [];
    for (let i = 0; i < files.length; i++) {
      try {
        showHint(`Enhancing ${i + 1}/${files.length}…`); await tick();
        if(files[i].size>4*1024*1024) throw new Error('Capture studio upload exceeds 4 MiB.');
        const {validateFile}=await import('./research/contract.js'); await validateFile(files[i]);
        const bmp = await createImageBitmap(files[i]);
        if(bmp.width*bmp.height>16000000||Math.min(bmp.width,bmp.height)<16){bmp.close();throw new Error('Unsupported image dimensions.');}
        const c = document.createElement('canvas'); c.width = bmp.width; c.height = bmp.height; c.getContext('2d').drawImage(bmp, 0, 0);
        const orig = addFrame(c, 'upload', files[i].name);
        const enh = addEnhanced(orig, autoEnhance(c));
        batch.push({ orig, enh });
        last = enh || orig;
      } catch (error) { setStatus($('connStatus'),'err',error.message || 'Cannot open this image.'); }
    }
    showHint('');
    lastBatch = batch;
    $('btnZip').hidden = !batch.length;
    $('zipLbl').textContent = `Download all (${batch.length})`;
    if (last) showFrame(last);
  };
  $('btnZip').onclick = async () => {
    if (!lastBatch.length) return;
    const b = $('btnZip'); b.disabled = true; showHint('Building zip…'); await tick();
    const pngBytes = cv => new Promise(r => cv.toBlob(async bl => r(new Uint8Array(await bl.arrayBuffer())), 'image/png'));
    const used = new Set(), entries = [];
    for (const { orig, enh } of lastBatch) {
      let base = (orig.name || 'image').replace(/\.[^.]+$/, ''), k = 2;
      while (used.has(base)) base = `${(orig.name || 'image').replace(/\.[^.]+$/, '')}-${k++}`;
      used.add(base);
      entries.push({ name: `${base}_raw.png`, data: await pngBytes(orig.canvas) });
      if (enh) entries.push({ name: `${base}_enhanced.png`, data: await pngBytes(enh.canvas) });
    }
    const a = document.createElement('a'); a.href = URL.createObjectURL(makeZip(entries));
    a.download = `captrail-batch-${new Date().toISOString().replace(/[:.]/g, '-')}.zip`; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    showHint(''); b.disabled = false;
  };
  // Minimal ZIP writer (stored, no compression: PNGs are already compressed).
  function makeZip(files) {
    const table = new Uint32Array(256);
    for (let i = 0; i < 256; i++) { let c = i; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; table[i] = c >>> 0; }
    const crc32 = u8 => { let c = 0xFFFFFFFF; for (let i = 0; i < u8.length; i++) c = table[(c ^ u8[i]) & 255] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; };
    const enc = new TextEncoder(), now = new Date();
    const time = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
    const date = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
    const parts = [], central = [];
    let offset = 0;
    for (const f of files) {
      const name = enc.encode(f.name), crc = crc32(f.data), size = f.data.length;
      const lh = new DataView(new ArrayBuffer(30));
      lh.setUint32(0, 0x04034b50, true); lh.setUint16(4, 20, true); lh.setUint16(6, 0x0800, true); lh.setUint16(8, 0, true);
      lh.setUint16(10, time, true); lh.setUint16(12, date, true); lh.setUint32(14, crc, true);
      lh.setUint32(18, size, true); lh.setUint32(22, size, true); lh.setUint16(26, name.length, true); lh.setUint16(28, 0, true);
      parts.push(lh.buffer, name, f.data);
      const ch = new DataView(new ArrayBuffer(46));
      ch.setUint32(0, 0x02014b50, true); ch.setUint16(4, 20, true); ch.setUint16(6, 20, true); ch.setUint16(8, 0x0800, true);
      ch.setUint16(10, 0, true); ch.setUint16(12, time, true); ch.setUint16(14, date, true); ch.setUint32(16, crc, true);
      ch.setUint32(20, size, true); ch.setUint32(24, size, true); ch.setUint16(28, name.length, true);
      ch.setUint32(42, offset, true);
      central.push(ch.buffer, name);
      offset += 30 + name.length + size;
    }
    const cdSize = central.reduce((s, x) => s + x.byteLength, 0);
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true); end.setUint16(10, files.length, true);
    end.setUint32(12, cdSize, true); end.setUint32(16, offset, true);
    return new Blob([...parts, ...central, end.buffer], { type: 'application/zip' });
  }

  /* ---------- image processing ---------- */
  // Sharpness = variance of the Laplacian on a downscaled grayscale copy. Higher is sharper.
  function sharpness(src) {
    const W = 320, H = Math.round(src.height * W / src.width);
    const c = document.createElement('canvas'); c.width = W; c.height = H;
    const x = c.getContext('2d', { willReadFrequently: true }); x.drawImage(src, 0, 0, W, H);
    const d = x.getImageData(0, 0, W, H).data, g = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++) g[i] = .299 * d[i * 4] + .587 * d[i * 4 + 1] + .114 * d[i * 4 + 2];
    let sum = 0, sum2 = 0, n = 0;
    for (let y = 1; y < H - 1; y++) for (let xx = 1; xx < W - 1; xx++) {
      const i = y * W + xx, l = g[i - 1] + g[i + 1] + g[i - W] + g[i + W] - 4 * g[i];
      sum += l; sum2 += l * l; n++;
    }
    return Math.round(sum2 / n - (sum / n) ** 2);
  }
  // Capillaries absorb green light, so the green channel shows them darkest. Stretch its contrast and sharpen.
  function grayCanvas(src) {
    const { width: W, height: H } = src;
    const c = document.createElement('canvas'); c.width = W; c.height = H;
    const x = c.getContext('2d', { willReadFrequently: true }); x.drawImage(src, 0, 0);
    const img = x.getImageData(0, 0, W, H), d = img.data, n = W * H;
    const hist = new Uint32Array(256), g = new Uint8ClampedArray(n);
    for (let i = 0; i < n; i++) { g[i] = d[i * 4 + 1]; hist[g[i]]++; }
    let lo = 0, hi = 255, acc = 0;
    for (; lo < 255; lo++) { acc += hist[lo]; if (acc > n * .01) break; }
    acc = 0; for (; hi > 0; hi--) { acc += hist[hi]; if (acc > n * .01) break; }
    const span = Math.max(1, hi - lo), s = new Float32Array(n);
    for (let i = 0; i < n; i++) s[i] = (g[i] - lo) * 255 / span;
    for (let y = 0; y < H; y++) for (let xx = 0; xx < W; xx++) {
      const i = y * W + xx;
      let v = s[i];
      if (y > 0 && y < H - 1 && xx > 0 && xx < W - 1) v = 2 * v - (s[i - 1] + s[i + 1] + s[i - W] + s[i + W] + v) / 5;
      d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v; d[i * 4 + 3] = 255;
    }
    x.putImageData(img, 0, 0); return c;
  }
  function thumbOf(src) {
    const c = document.createElement('canvas'); c.width = 240; c.height = Math.round(src.height * 240 / src.width);
    c.getContext('2d').drawImage(src, 0, 0, c.width, c.height); return c.toDataURL('image/jpeg', .8);
  }
  // Warm: make the microscope's cool blue-green frames look like clinical capillaroscopy (warm skin, red vessels).
  function warmCanvas(src) {
    const { width: W, height: H } = src;
    const c = document.createElement('canvas'); c.width = W; c.height = H;
    const x = c.getContext('2d', { willReadFrequently: true }); x.drawImage(src, 0, 0);
    const img = x.getImageData(0, 0, W, H), d = img.data, n = W * H;
    // a. Gray-world white balance: scale channels so their averages match.
    let sr = 0, sg = 0, sb = 0;
    for (let i = 0; i < d.length; i += 4) { sr += d[i]; sg += d[i + 1]; sb += d[i + 2]; }
    const mean = (sr + sg + sb) / 3;
    // b. Warm shift on top of the balance.
    const k = [mean / Math.max(sr, 1) * 1.15, mean / Math.max(sg, 1) * 0.95, mean / Math.max(sb, 1) * 0.80];
    const v = new Float32Array(n * 3), hist = new Uint32Array(256);
    for (let i = 0, j = 0; i < d.length; i += 4, j += 3) for (let ch = 0; ch < 3; ch++) {
      const val = d[i + ch] * k[ch]; v[j + ch] = val; hist[Math.min(255, Math.round(val))]++;
    }
    // c. Contrast stretch: 1st percentile -> 0, 99th -> 255. One range shared by all three channels;
    //    stretching each channel on its own would undo the white balance and the warm shift.
    let lo = 0, hi = 255, acc = 0;
    for (; lo < 255; lo++) { acc += hist[lo]; if (acc > n * 3 * .01) break; }
    acc = 0; for (; hi > 0; hi--) { acc += hist[hi]; if (acc > n * 3 * .01) break; }
    const scale = 255 / Math.max(1, hi - lo);
    // d. Clamp to 0-255 (Uint8ClampedArray rounds and clamps on write).
    for (let i = 0, j = 0; i < d.length; i += 4, j += 3) {
      d[i] = (v[j] - lo) * scale; d[i + 1] = (v[j + 1] - lo) * scale; d[i + 2] = (v[j + 2] - lo) * scale; d[i + 3] = 255;
    }
    x.putImageData(img, 0, 0); return c;
  }
  // Unsharp mask: out = orig + amount * (orig - blur), only where |orig - blur| >= threshold.
  // Gaussian blur with sigma = radius, run as two 1-D passes (horizontal, then vertical).
  function unsharpCanvas(src, { radius = 2, amount = 1.5, threshold = 3 } = {}) {
    const { width: W, height: H } = src;
    const c = document.createElement('canvas'); c.width = W; c.height = H;
    const x = c.getContext('2d', { willReadFrequently: true }); x.drawImage(src, 0, 0);
    const img = x.getImageData(0, 0, W, H), d = img.data;
    const r = Math.ceil(radius * 3), k = new Float32Array(2 * r + 1);
    let ks = 0;
    for (let i = -r; i <= r; i++) ks += k[i + r] = Math.exp(-(i * i) / (2 * radius * radius));
    for (let i = 0; i < k.length; i++) k[i] /= ks;
    const tmp = new Float32Array(W * H), blur = new Float32Array(W * H);
    for (let ch = 0; ch < 3; ch++) {
      for (let y = 0; y < H; y++) for (let xx = 0; xx < W; xx++) {
        let v = 0;
        for (let i = -r; i <= r; i++) v += d[(y * W + Math.min(W - 1, Math.max(0, xx + i))) * 4 + ch] * k[i + r];
        tmp[y * W + xx] = v;
      }
      for (let y = 0; y < H; y++) for (let xx = 0; xx < W; xx++) {
        let v = 0;
        for (let i = -r; i <= r; i++) v += tmp[Math.min(H - 1, Math.max(0, y + i)) * W + xx] * k[i + r];
        blur[y * W + xx] = v;
      }
      for (let p = 0; p < W * H; p++) {
        const o = d[p * 4 + ch], diff = o - blur[p];
        if (Math.abs(diff) >= threshold) d[p * 4 + ch] = o + amount * diff; // clamped on write
      }
    }
    x.putImageData(img, 0, 0); return c;
  }
  // Fast Gaussian blur of one channel (Float32Array, W×H) using three box blurs, O(pixels) for any sigma.
  function gaussBlur(src, W, H, sigma) {
    const n = 3, wIdeal = Math.sqrt(12 * sigma * sigma / n + 1);
    let wl = Math.floor(wIdeal); if (wl % 2 === 0) wl--;
    const m = Math.round((12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4));
    let a = Float32Array.from(src), b = new Float32Array(src.length);
    for (let pass = 0; pass < n; pass++) {
      const r = ((pass < m ? wl : wl + 2) - 1) / 2;
      boxPass(a, b, W, H, r, 1, W);  // horizontal
      boxPass(b, a, H, W, r, W, 1);  // vertical (same routine, strides swapped)
    }
    return a;
  }
  // Running-sum box blur along lines of `len` pixels; `step` moves along a line, `lineStep` between lines.
  function boxPass(src, dst, len, lines, r, step, lineStep) {
    const inv = 1 / (2 * r + 1);
    for (let l = 0; l < lines; l++) {
      const base = l * lineStep, at = i => src[base + Math.min(len - 1, Math.max(0, i)) * step];
      let acc = 0;
      for (let i = -r - 1; i < r; i++) acc += at(i);
      for (let i = 0; i < len; i++) { acc += at(i + r) - at(i - r - 1); dst[base + i * step] = acc * inv; }
    }
  }
  // Capillary: flatten uneven lighting, boost local contrast, then render in a clinical red tint and sharpen.
  function capillaryCanvas(src) {
    const { width: W, height: H } = src, n = W * H;
    const c = document.createElement('canvas'); c.width = W; c.height = H;
    const x = c.getContext('2d', { willReadFrequently: true }); x.drawImage(src, 0, 0);
    const img = x.getImageData(0, 0, W, H), d = img.data;
    // 1-2. Per channel: background = Gaussian blur (radius 25px); out = (pixel - background) * 3 + 128, clamped.
    const boosted = [];
    for (let ch = 0; ch < 3; ch++) {
      const v = new Float32Array(n);
      for (let p = 0; p < n; p++) v[p] = d[p * 4 + ch];
      const bg = gaussBlur(v, W, H, 25);
      for (let p = 0; p < n; p++) v[p] = Math.min(255, Math.max(0, (v[p] - bg[p]) * 3 + 128));
      boosted.push(v);
    }
    // 3. Grayscale, then map to a reddish tint (clamped on write).
    for (let p = 0; p < n; p++) {
      const g = .299 * boosted[0][p] + .587 * boosted[1][p] + .114 * boosted[2][p] - 128;
      d[p * 4] = 200 + .6 * g; d[p * 4 + 1] = 120 + 1.3 * g; d[p * 4 + 2] = 100 + 1.3 * g; d[p * 4 + 3] = 255;
    }
    x.putImageData(img, 0, 0);
    // 4. Unsharp mask (radius 2px, amount 1.5, threshold 3).
    return unsharpCanvas(c);
  }
  // p-th percentile (0-1) of a Float32Array, estimated from a sample of up to ~200k values.
  function percentile(arr, p) {
    const step = Math.max(1, Math.floor(arr.length / 200000)), sample = new Float32Array(Math.ceil(arr.length / step));
    for (let i = 0, j = 0; i < arr.length; i += step) sample[j++] = arr[i];
    sample.sort();
    return sample[Math.min(sample.length - 1, Math.floor(p * sample.length))];
  }
  // Glare mask: brightness above the given percentile, grown by `grow` px. Brightness is smoothed 1.5px first
  // and must also be genuinely bright (>= 215), or near-saturated (>= 250). A percentile alone always marks
  // that share of pixels even with no glare, and on a noisy capture the grown specks would cover everything.
  function growMask(m, W, H, grow) {
    if (!grow) return m;
    const n = W * H, t = new Uint8Array(n), out = new Uint8Array(n);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let v = 0; for (let k = -grow; k <= grow && !v; k++) { const xs = x + k; if (xs >= 0 && xs < W) v = m[y * W + xs]; }
      t[y * W + x] = v;
    }
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let v = 0; for (let k = -grow; k <= grow && !v; k++) { const ys = y + k; if (ys >= 0 && ys < H) v = t[ys * W + x]; }
      out[y * W + x] = v;
    }
    return out;
  }
  function glareMask(lum, W, H, pct, grow) {
    const n = W * H, sm = gaussBlur(lum, W, H, 1.5), cut = percentile(sm, pct), m = new Uint8Array(n);
    for (let p = 0; p < n; p++) m[p] = (sm[p] > cut && sm[p] >= 215) || sm[p] >= 250 ? 1 : 0;
    return growMask(m, W, H, grow);
  }
  // Lanczos (a = 3) upscale of one channel by an integer factor, as two 1-D passes.
  function lanczosUp(src, W, H, f) {
    const L = t => t === 0 ? 1 : Math.abs(t) >= 3 ? 0 : 3 * Math.sin(Math.PI * t) * Math.sin(Math.PI * t / 3) / (Math.PI * Math.PI * t * t);
    // Weights for each of the f sub-pixel phases: taps at floor(s)-2 .. floor(s)+3.
    const phases = [];
    for (let ph = 0; ph < f; ph++) {
      const sCoord = (ph + .5) / f - .5, fr = sCoord - Math.floor(sCoord);
      const w = [-2, -1, 0, 1, 2, 3].map(t => L(t - fr)), sum = w.reduce((a, b) => a + b, 0);
      phases.push({ base: Math.floor(sCoord), w: w.map(v => v / sum) });
    }
    const pass = (inp, len, lines, step, lineStep, outStep, outLineStep) => {
      const out = new Float32Array(len * f * lines);
      for (let l = 0; l < lines; l++) for (let o = 0; o < len * f; o++) {
        const i = Math.floor(o / f), { base, w } = phases[o % f];
        let v = 0;
        for (let k = 0; k < 6; k++) { const s2 = Math.min(len - 1, Math.max(0, i + base - 2 + k)); v += w[k] * inp[l * lineStep + s2 * step]; }
        out[l * outLineStep + o * outStep] = v;
      }
      return out;
    };
    const Wf = W * f;
    const h = pass(src, W, H, 1, W, 1, Wf);            // rows: W -> W*f
    return pass(h, H, Wf, Wf, 1, Wf, 1);                // columns: H -> H*f
  }
  // Fill masked pixels from their surroundings: grow inward ring by ring with the average of known
  // 4-neighbours, then smooth the filled area with ~100 diffusion passes. Only masked pixels change.
  function inpaint(chans, mask, W, H) {
    const idx = [];
    for (let p = 0; p < W * H; p++) if (mask[p]) idx.push(p);
    if (!idx.length) return;
    const known = new Uint8Array(W * H);
    for (let p = 0; p < W * H; p++) known[p] = mask[p] ? 0 : 1;
    let todo = idx.slice();
    for (let iter = 0; todo.length && iter < 5000; iter++) {
      const next = [], upd = [];
      for (const p of todo) {
        const x = p % W, nb = [];
        if (x > 0 && known[p - 1]) nb.push(p - 1);
        if (x < W - 1 && known[p + 1]) nb.push(p + 1);
        if (p >= W && known[p - W]) nb.push(p - W);
        if (p < W * (H - 1) && known[p + W]) nb.push(p + W);
        if (nb.length) upd.push([p, nb]); else next.push(p);
      }
      if (!upd.length) break;
      for (const [p, nb] of upd) { for (const c of chans) { let s = 0; for (const q of nb) s += c[q]; c[p] = s / nb.length; } }
      for (const [p] of upd) known[p] = 1;
      todo = next;
    }
    for (let it = 0; it < 100; it++) for (const p of idx) {
      const x = p % W;
      const nb = [x > 0 ? p - 1 : p, x < W - 1 ? p + 1 : p, p >= W ? p - W : p, p < W * (H - 1) ? p + W : p];
      for (const c of chans) c[p] = (c[nb[0]] + c[nb[1]] + c[nb[2]] + c[nb[3]]) / 4;
    }
  }
  function median3x3(src, W, H) {
    const out = new Float32Array(W * H), win = new Float32Array(9);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let k = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++)
        win[k++] = src[Math.min(H - 1, Math.max(0, y + dy)) * W + Math.min(W - 1, Math.max(0, x + dx))];
      out[y * W + x] = insertionMedian(win, 9);
    }
    return out;
  }
  // Auto enhance: real pixels only. Glare fill, denoise, find the capillary row, 3× Lanczos zoom on it,
  // local contrast, vessel color emphasis and light sharpening. Nothing is drawn.
  const ENHANCE_ZOOM = 3, ENHANCE_MAX_FILL = .25;
  function autoEnhance(src) {
    const { width: W, height: H } = src, n = W * H, d = pixelsOf(src);
    let R = new Float32Array(n), G = new Float32Array(n), B = new Float32Array(n);
    const lum = new Float32Array(n), sat = new Float32Array(n);
    for (let p = 0; p < n; p++) {
      const r = d[p * 4], g = d[p * 4 + 1], b = d[p * 4 + 2], mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      R[p] = r; G[p] = g; B[p] = b; lum[p] = .299 * r + .587 * g + .114 * b; sat[p] = mx ? (mx - mn) / mx : 0;
    }
    // 2. Glare: above the 95th brightness percentile (and genuinely bright), or low saturation + high
    //    brightness; grown 5px. Never fill more than 25% of the image.
    const sm = gaussBlur(lum, W, H, 1.5), cut = percentile(sm, .95), m0 = new Uint8Array(n);
    for (let p = 0; p < n; p++) m0[p] = (sm[p] > cut && sm[p] >= 215) || sm[p] >= 250 || (sat[p] < .15 && sm[p] > 200) ? 1 : 0;
    const mask = growMask(m0, W, H, 5);
    let filled = 0; for (let p = 0; p < n; p++) filled += mask[p];
    const fillFrac = filled / n;
    if (fillFrac > ENHANCE_MAX_FILL) return { canvas: null, reason: 'Too much glare, retake', fillFrac };
    inpaint([R, G, B], mask, W, H);
    // 3. Denoise: 3×3 median per channel.
    R = median3x3(R, W, H); G = median3x3(G, W, H); B = median3x3(B, W, H);
    // 4. Capillary row: lower 70%, highest mean |horizontal gradient of (R - G)| outside the glare mask,
    //    weighted by how much of the row is outside the mask and smoothed over 8 rows.
    const y0 = Math.floor(H * .3), score = new Float32Array(H);
    for (let y = y0; y < H; y++) {
      let s = 0, c = 0;
      for (let x = 0; x < W - 1; x++) {
        const p = y * W + x;
        if (mask[p] || mask[p + 1]) continue;
        s += Math.abs((R[p + 1] - G[p + 1]) - (R[p] - G[p])); c++;
      }
      score[y] = c ? (s / c) * (c / (W - 1)) : 0;
    }
    let loopY = y0, best = -1;
    for (let y = y0; y < H; y++) {
      let s = 0, c = 0;
      for (let yy = y - 4; yy < y + 4; yy++) if (yy >= y0 && yy < H) { s += score[yy]; c++; }
      if (s / c > best) { best = s / c; loopY = y; }
    }
    // 5. Crop 1/3 × 1/3 centered on the row, slid sideways to the least-filled spot (ties: most central).
    const f = ENHANCE_ZOOM, cw = Math.floor(W / f), ch = Math.floor(H / f);
    const cy = Math.min(H - ch, Math.max(0, Math.round(loopY - ch / 2)));
    const colSum = new Uint32Array(W + 1);
    for (let x = 0; x < W; x++) { let c = 0; for (let y = cy; y < cy + ch; y++) c += mask[y * W + x]; colSum[x + 1] = colSum[x] + c; }
    let cx = Math.round((W - cw) / 2), fewest = Infinity;
    for (let x = 0; x <= W - cw; x++) {
      const g = colSum[x + cw] - colSum[x];
      if (g < fewest || (g === fewest && Math.abs(x - (W - cw) / 2) < Math.abs(cx - (W - cw) / 2))) { fewest = g; cx = x; }
    }
    //    Enlarge 3× with Lanczos.
    const W2 = cw * f, H2 = ch * f, n2 = W2 * H2;
    const [r, g, b] = [R, G, B].map(chn => {
      const crop = new Float32Array(cw * ch);
      for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) crop[y * cw + x] = chn[(cy + y) * W + cx + x];
      return lanczosUp(crop, cw, ch, f);
    });
    // 6. Local contrast on luminance: L' = blur + 2 × (L - 40px blur); the change is added to every channel
    //    so hue stays the same.
    const L = new Float32Array(n2);
    for (let p = 0; p < n2; p++) L[p] = .299 * r[p] + .587 * g[p] + .114 * b[p];
    const Lb = gaussBlur(L, W2, H2, 40);
    for (let p = 0; p < n2; p++) { const dl = L[p] - Lb[p]; r[p] += dl; g[p] += dl; b[p] += dl; }
    // 7. Vessel color: e = (R - G) minus its 15px blur; R += 0.5e, G and B -= 0.3e; then saturation × 1.2.
    const red = new Float32Array(n2);
    for (let p = 0; p < n2; p++) red[p] = r[p] - g[p];
    const redBg = gaussBlur(red, W2, H2, 15);
    const out = new ImageData(W2, H2), od = out.data;
    for (let p = 0; p < n2; p++) {
      const e = red[p] - redBg[p];
      const rr = r[p] + .5 * e, gg = g[p] - .3 * e, bb = b[p] - .3 * e, Y = .299 * rr + .587 * gg + .114 * bb;
      od[p * 4] = Y + 1.2 * (rr - Y); od[p * 4 + 1] = Y + 1.2 * (gg - Y); od[p * 4 + 2] = Y + 1.2 * (bb - Y); od[p * 4 + 3] = 255;
    }
    const c = document.createElement('canvas'); c.width = W2; c.height = H2;
    c.getContext('2d').putImageData(out, 0, 0);
    // 8. Unsharp mask: radius 2px, amount 0.8.
    return { canvas: unsharpCanvas(c, { radius: 2, amount: .8, threshold: 0 }), crop: { x: cx, y: cy, w: cw, h: ch, loopY }, fillFrac, scale: 1 / f, W, H };
  }
  // Add an enhanced frame paired with its original (or a note on the original if enhancement was refused).
  function addEnhanced(orig, res) {
    if (!res.canvas) { orig.note = res.reason; return null; }
    const enh = addFrame(res.canvas, orig.source, orig.name);
    enh.calibScale = res.scale;              // 3× zoom: each enhanced pixel is a third of an original pixel
    enh.calibW = res.W; enh.calibH = res.H;  // calibration is stored for the original resolution
    enh.crop = res.crop;
    enh.pair = orig; orig.pair = enh;
    return enh;
  }

  // Simulated view: draws hairpin loops at the loop positions detected in the real capture, styled like a
  // capillaroscopy photo. It is an illustration only.
  const LOOKS = {
    off:   { label: 'Original', short: 'Off', next: 'warm' },
    warm:  { label: 'Warm', short: 'Warm', next: 'sharp', make: f => warmCanvas(f.canvas) },
    // Sharpen runs on the warm-corrected image, reusing its cached copy.
    sharp: { label: 'Warm + sharpen', short: 'Sharpen', next: 'capillary', make: f => unsharpCanvas(lookCanvas(f, 'warm')) },
    capillary: { label: 'Capillary', short: 'Capillary', next: 'gray', make: f => capillaryCanvas(f.canvas) },
    gray:  { label: 'Grayscale', short: 'Gray', next: 'off', make: f => grayCanvas(f.canvas) },
  };
  // The raw capture (f.canvas) is never modified; processed versions are cached per frame.
  function lookCanvas(f, l) {
    if (l === 'off') return f.canvas;
    f.looks ||= {};
    return (f.looks[l] ||= LOOKS[l].make(f));
  }
  $('btnEnhance').onclick = () => { look = LOOKS[look].next; drawFrame(); renderHud(); syncButtons(); };
  $('btnRaw').onclick = () => { if (sel?.pair && frames.includes(sel.pair)) showFrame(sel.pair); };

  /* ---------- gallery ---------- */
  function renderGallery() {
    const g = $('gallery');
    if (!frames.length && !stream) { g.innerHTML = '<div class="gallery-empty">Captured and uploaded images appear here, scored for sharpness.</div>'; return; }
    const best = frames.reduce((m, f) => Math.max(m, f.score), -1);
    g.innerHTML = (stream ? `<button class="thumb live-thumb${sel ? '' : ' sel'}" data-live>● Live</button>` : '') +
      frames.map(f => `<button class="thumb${sel === f ? ' sel' : ''}" data-id="${f.id}" title="${escHtml(f.name || 'Capture')}"><img src="${f.thumb}" alt=""><span class="sc">${f.score}</span>${frames.length > 1 && f.score === best ? '<span class="best">SHARPEST</span>' : ''}</button>`).join('');
    g.querySelectorAll('.thumb').forEach(b => b.onclick = () => b.dataset.live !== undefined ? goLive() : showFrame(frames.find(f => f.id === b.dataset.id)));
  }
  $('btnDelete').onclick = () => {
    if (!sel) return;
    frames = frames.filter(f => f !== sel);
    if (frames.length) showFrame(frames[frames.length - 1]);
    else if (stream) goLive();
    else { sel = null; video.hidden = view.hidden = overlay.hidden = true; $('empty').hidden = false; stage.style.aspectRatio = '16/9'; renderHud(); renderGallery(); syncButtons(); renderMeasures(); renderFindings(); renderSim(); }
  };
  // Save always downloads the untouched raw capture (for the AI dataset), plus the processed view
  // with any measurements and AI boxes drawn on it (_warm / _gray / ..., or _annotated when the mode is off).
  // Auto-enhanced pairs save both images: _raw (the capture or upload) and _enhanced (3× auto-enhanced view).
  $('btnDownload').onclick = async () => {
    if (!sel) return;
    const f = sel;
    const base = f.name ? f.name.replace(/\.[^.]+$/, '') : `captrail-${f.at.toISOString().replace(/[:.]/g, '-')}`;
    const save = (canvas, name) => new Promise(done => canvas.toBlob(b => {
      const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = name; a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000); done();
    }, 'image/png'));
    const gap = () => new Promise(r => setTimeout(r, 400)); // browsers drop back-to-back downloads
    if (f.pair) {
      const zoom = f.calibScale ? f : f.pair, orig = zoom.pair;
      await save(orig.canvas, `${base}_raw.png`);
      await gap();
      await save(zoom.canvas, `${base}_enhanced.png`);
      return;
    }
    await save(f.canvas, `${base}_raw.png`);
    const annotated = f.measures.length || (f.ai?.findings || []).length;
    if (look === 'off' && !annotated) return;
    const out = document.createElement('canvas'); out.width = f.w; out.height = f.h;
    const x = out.getContext('2d'); x.drawImage(lookCanvas(f, look), 0, 0); x.drawImage(overlay, 0, 0);
    await gap();
    await save(out, `${base}_${look === 'off' ? 'annotated' : look}.png`);
  };

  /* ---------- clicking on the image: calibrate + measure ---------- */
  function setMode(m) {
    mode = m; pts = [];
    stage.classList.toggle('picking', m !== 'none');
    showHint(m === 'calibrate' ? 'Click two tick marks on the calibration slide' : m === 'measure' ? 'Click two points to measure. Esc to stop.' : '');
    if (m !== 'calibrate') $('calForm').classList.remove('show');
    syncButtons(); drawOverlay();
  }
  $('btnMeasure').onclick = () => setMode(mode === 'measure' ? 'none' : 'measure');
  $('btnCal').onclick = () => setMode('calibrate');
  addEventListener('keydown', e => {
    if (e.key === 'Escape') setMode('none');
  });
  overlay.addEventListener('pointerdown', e => {
    if (mode === 'none' || !sel) return;
    const r = overlay.getBoundingClientRect();
    pts.push([(e.clientX - r.left) * sel.w / r.width, (e.clientY - r.top) * sel.h / r.height]);
    if (pts.length === 2) {
      const px = Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]);
      if (mode === 'measure') { sel.measures.push({ a: pts[0], b: pts[1], px }); pts = []; renderMeasures(); }
      else if (mode === 'calibrate') { showHint(''); $('calForm').classList.add('show'); $('calUm').focus(); }
    } else if (pts.length > 2) pts = [pts[2]];
    drawOverlay();
  });
  $('calSave').onclick = () => {
    const um = parseFloat($('calUm').value);
    if (pts.length !== 2 || !(um > 0)) return;
    const px = Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]);
    const scale = sel.calibScale || 1;
    calib = { umPerPx: um / px / scale, w: sel.calibW || sel.w, h: sel.calibH || sel.h, at: Date.now() };
    store.set('calib', calib); setMode('none'); renderCalib(); renderMeasures();
  };
  $('calCancel').onclick = () => setMode('none');
  $('btnCalClear').onclick = () => { calib = null; store.set('calib', null); renderCalib(); renderMeasures(); };
  // A zoomed frame's µm per pixel is the stored calibration × its calibScale (0.5 for the 2× zoom).
  function umPerPx() { return calib && sel && calib.w === (sel.calibW || sel.w) && calib.h === (sel.calibH || sel.h) ? calib.umPerPx * (sel.calibScale || 1) : null; }
  function renderCalib() {
    $('calVal').innerHTML = calib ? `${calib.umPerPx.toFixed(3)} <small>µm / px</small>` : '— <small>µm / px</small>';
    if (!calib) return setStatus($('calStatus'), '', 'Not calibrated');
    if (sel && (calib.w !== (sel.calibW || sel.w) || calib.h !== (sel.calibH || sel.h))) return setStatus($('calStatus'), 'warn', `Calibrated at ${calib.w}×${calib.h}; this image is ${sel.calibW || sel.w}×${sel.calibH || sel.h}. Recalibrate.`);
    if (sel?.calibScale) return setStatus($('calStatus'), 'ok', `Zoomed ${Math.round(1 / sel.calibScale)}×: measuring at ${umPerPx().toFixed(3)} µm/px (original ${calib.umPerPx.toFixed(3)} ÷ ${Math.round(1 / sel.calibScale)}).`);
    setStatus($('calStatus'), 'ok', `Calibrated ${new Date(calib.at).toLocaleDateString()} at ${calib.w}×${calib.h}. Recalibrate if zoom or height changes.`);
  }
  function renderMeasures() {
    renderCalib();
    const l = $('mlist'), k = umPerPx();
    if (!sel || !sel.measures.length) { l.innerHTML = '<li style="color:var(--muted)">None yet</li>'; drawOverlay(); return; }
    l.innerHTML = sel.measures.map((m, i) => `<li><span>#${i + 1}</span><span><b>${k ? (m.px * k).toFixed(1) + ' µm' : m.px.toFixed(0) + ' px'}</b>${k ? '' : ' (calibrate for µm)'}</span><button data-i="${i}" aria-label="Remove">×</button></li>`).join('');
    l.querySelectorAll('button').forEach(b => b.onclick = () => { sel.measures.splice(+b.dataset.i, 1); renderMeasures(); });
    drawOverlay();
  }
  function drawOverlay() {
    if (!sel) return;
    octx.clearRect(0, 0, overlay.width, overlay.height);
    const lw = Math.max(2, sel.w / 640), k = umPerPx();
    octx.lineWidth = lw; octx.font = `600 ${Math.round(lw * 9)}px Inter, sans-serif`;
    const dot = (p, c) => { octx.fillStyle = c; octx.beginPath(); octx.arc(p[0], p[1], lw * 2.6, 0, 7); octx.fill(); };
    const label = (t, x, y) => { const w = octx.measureText(t).width + lw * 8; octx.fillStyle = 'rgba(0,0,0,.7)'; octx.fillRect(x - w / 2, y - lw * 14, w, lw * 11); octx.fillStyle = '#fff'; octx.textAlign = 'center'; octx.fillText(t, x, y - lw * 5.5); };
    sel.measures.forEach((m, i) => {
      octx.strokeStyle = '#FFD166'; octx.beginPath(); octx.moveTo(...m.a); octx.lineTo(...m.b); octx.stroke();
      dot(m.a, '#FFD166'); dot(m.b, '#FFD166');
      label(`#${i + 1} ${k ? (m.px * k).toFixed(1) + ' µm' : m.px.toFixed(0) + ' px'}`, (m.a[0] + m.b[0]) / 2, Math.min(m.a[1], m.b[1]));
    });
    (sel.ai?.findings || []).forEach(f => {
      if (!Array.isArray(f.box)) return;
      const [x, y, w, h] = f.box;
      octx.strokeStyle = '#E0535F'; octx.strokeRect(x, y, w, h);
      label(`${f.label}${f.confidence != null ? ' ' + Math.round(f.confidence * 100) + '%' : ''}`, x + w / 2, y);
    });
    if (pts.length) {
      const c = mode === 'calibrate' ? '#7CE0A3' : '#FFD166';
      pts.forEach(p => dot(p, c));
      if (pts.length === 2) { octx.strokeStyle = c; octx.beginPath(); octx.moveTo(...pts[0]); octx.lineTo(...pts[1]); octx.stroke(); }
    }
  }

  /* ---------- AI ---------- */
  const escHtml = v => String(v).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  function renderFindings() {
    const l = $('findings'), note = $('aiNote');
    if (!ANALYZE_ENDPOINT) setStatus($('aiStatus'), 'warn', 'Model not connected yet.');
    else if (!sel?.ai && !analyzing) setStatus($('aiStatus'), '', sel ? 'Ready to analyze' : 'Select an image');
    if (!sel?.ai) { l.innerHTML = ''; note.hidden = true; return; }
    const a = sel.ai, rows = [];
    
    
    (a.findings || []).forEach(f => rows.push([f.label, f.confidence != null ? Math.round(f.confidence * 100) + '% confidence' : 'flagged']));
    if (!(a.findings || []).length) rows.push(['No patterns flagged', '']);
    l.innerHTML = rows.map(([k, v]) => `<li><b>${escHtml(k)}</b><span>${escHtml(v)}</span></li>`).join('');
    note.hidden = false;
    note.innerHTML = `<b>Note</b>${escHtml(a.note || 'No note returned.')}`;
    setStatus($('aiStatus'), 'ok', 'Experimental AI estimate. For clinician review only, not a diagnosis.');
  }
  // Vercel limits uploads to 4.5 MB, and Claude reads images at about 1600 px anyway,
  // so send a downscaled JPEG. Boxes come back in original-frame pixels via width/height.
  function uploadBlob(src) {
    const scale = Math.min(1, 1600 / Math.max(src.width, src.height));
    const c = document.createElement('canvas');
    c.width = Math.round(src.width * scale); c.height = Math.round(src.height * scale);
    c.getContext('2d').drawImage(src, 0, 0, c.width, c.height);
    return new Promise(r => c.toBlob(r, 'image/jpeg', .9));
  }
  $('btnAnalyze').onclick = async () => {
    if (!sel || analyzing) return;
    if (!$('externalReviewConsent').checked) { setStatus($('aiStatus'), 'warn', 'Separate permission to send this image to Anthropic is required.'); return; }
    if (!ANALYZE_ENDPOINT) { renderFindings(); return; }
    const f = sel, b = $('btnAnalyze');
    analyzing = true; syncButtons(); b.textContent = 'Analyzing…';
    setStatus($('aiStatus'), '', 'Analyzing… this can take up to a minute.');
    try {
      const fd = new FormData();
      fd.append('external_commentary_consent','true');
      fd.append('image', await uploadBlob(f.canvas), 'scan.jpg');
      fd.append('width', f.w); fd.append('height', f.h);
      const k = umPerPx(); if (k) fd.append('um_per_px', k);
      let res;
      try { res = await fetch(ANALYZE_ENDPOINT, { method: 'POST', headers:{'X-Captrail-Request':'1'}, body: fd }); }
      catch { throw new Error('Couldn’t reach the analysis service. Check your internet connection and try again.'); }
      let body = null;
      try { body = await res.json(); } catch {}
      if (!res.ok) throw new Error(body?.error || (res.status === 404 ? 'The analysis service isn’t deployed yet.' : 'Something went wrong. Please try again.'));
      f.ai = body;
      if (sel === f) { renderFindings(); drawOverlay(); }
    } catch (e) {
      if (sel === f) setStatus($('aiStatus'), 'err', e.message);
    } finally {
      analyzing = false; b.textContent = 'Analyze this image'; syncButtons();
    }
  };


  /* ---------- voice assistant ----------
     Listening and talking use the browser's Web Speech API. Commands run locally;
     anything else is a question sent as text (plus a little page state) to /api/voice. */
  const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
  const canTalk = 'speechSynthesis' in window;
  const vPanel = $('voicePanel'), vBtn = $('micBtn');
  let rec = null, listening = false, heard = '';
  function vStatus(t, on) {
    $('voiceStatus').querySelector('span').textContent = t;
    vPanel.classList.toggle('listening', !!on); vBtn.classList.toggle('listening', !!on);
    vBtn.setAttribute('aria-pressed', !!on);
  }
  function vLine(id, t) { const el = $(id); el.hidden = !t; el.querySelector('span').textContent = t || ''; }
  function say(t) {
    vLine('voiceReply', t); vPanel.hidden = false;
    if (!canTalk || !t) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(t); u.lang = 'en-US'; u.rate = 1.02;
    speechSynthesis.speak(u);
  }
  function listen() {
    if (!Rec || listening) return;
    if (canTalk) speechSynthesis.cancel();
    vPanel.hidden = false; heard = ''; vLine('voiceHeard', ''); vLine('voiceReply', '');
    const R = window.SpeechRecognition || window.webkitSpeechRecognition;
    rec = new R(); rec.lang = 'en-US'; rec.interimResults = true; rec.continuous = false; rec.maxAlternatives = 1;
    rec.onresult = e => {
      heard = [...e.results].map(r => r[0].transcript).join(' ').trim();
      vLine('voiceHeard', heard);
    };
    rec.onerror = e => {
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') vLine('voiceReply', 'Microphone access is blocked. Allow it in the address bar to use voice.');
      else if (e.error === 'no-speech') vLine('voiceReply', 'I didn’t hear anything. Try again.');
      else if (e.error !== 'aborted') vLine('voiceReply', 'Voice didn’t work that time. Try again.');
    };
    rec.onend = () => { listening = false; vStatus('Ready'); if (heard) handleUtterance(heard); };
    try { rec.start(); listening = true; vStatus('Listening...', true); }
    catch { vStatus('Ready'); }
  }
  function stopListening() { if (rec && listening) rec.stop(); }

  // Short spoken commands run right away; questions ("how do I…") go to the helper.
  const VOICE_CMDS = [
    ['stop', /^(stop|be quiet|quiet|stop talking|shut up|silence)$/],
    ['best8', /\bbest of (eight|8)\b/],
    ['enhanceOff', /^(turn )?(enhance|enhancement) off$|^turn off (enhance|enhancement)$/],
    ['capture', /^(capture|take a (picture|photo|pic|snapshot)|snap|take picture)( now)?$/],
    ['enhance', /^(enhance|enhancement|toggle enhance|enhance on|turn on (enhance|enhancement)|turn (enhance|enhancement) on)$/],
    ['calibrate', /^(calibrate|start calibration|open calibration|calibration)$/],
    ['analyze', /^(analy[sz]e|run analysis|analy[sz]e (this|the) (image|picture))$/],
  ];
  function matchCommand(text) {
    const t = text.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim()
      .replace(/^(hey |ok |okay )?(captrail )?/, '').replace(/^(please |can you |could you )/, '').replace(/ please$/, '').trim();
    if (/^(how|what|why|when|where|which|who|should|is|are|does|do|can i|could i|will)\b/.test(t) || t.split(' ').length > 6) return null;
    const hit = VOICE_CMDS.find(([, rx]) => rx.test(t));
    return hit ? hit[0] : null;
  }
  const needLive = () => stream ? 'Go back to the live view first.' : 'Connect the microscope first.';
  function setLookOff() { let k = 0; while (look !== 'off' && k++ < 10) $('btnEnhance').click(); }
  function runCommand(c) {
    switch (c) {
      case 'stop': if (canTalk) speechSynthesis.cancel(); vLine('voiceReply', 'Okay.'); return;
      case 'capture': if ($('btnCapture').disabled) return say(needLive()); $('btnCapture').click(); return say('Captured.');
      case 'best8': if ($('btnBurst').disabled) return say(stacking ? 'Already capturing.' : needLive()); $('btnBurst').click(); return say('Best of eight. Hold still.');
      case 'enhance':
        if (!sel) return say('Capture or select an image first.');
        if (look === 'off') { $('btnEnhance').click(); return say(`Enhance on. ${LOOKS[look].label} view.`); }
        setLookOff(); return say('Enhance off.');
      case 'enhanceOff': if (!sel) return say('Capture or select an image first.'); setLookOff(); return say('Enhance off.');
      case 'calibrate':
        if (!sel) return say('Capture the calibration card first, then say calibrate.');
        $('btnCal').click(); $('calStatus').closest('.panel').scrollIntoView({ block: 'center', behavior: 'smooth' });
        return say('Calibration on. Click the outer edges of two long lines two millimeters apart, then enter 2000.');
      case 'analyze':
        if ($('btnAnalyze').disabled) return say(analyzing ? 'Already analyzing.' : 'Capture or select an image first.');
        $('btnAnalyze').click(); return say('Analyzing.');
    }
  }
  // Only these facts go to the helper: no images, no audio.
  function voiceState() {
    const k = sel ? umPerPx() : null;
    return { connected: !!stream, image: !!sel, calibrated: !!calib, um_per_px: k ?? calib?.umPerPx ?? null,
             zoomed: !!sel?.calibScale, enhance: LOOKS[look].short };
  }
  async function ask(question) {
    vStatus('Thinking...');
    try {
      const res = await fetch('/api/voice', { method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ question, state: voiceState() }) });
      let body = null; try { body = await res.json(); } catch {}
      say(res.ok && body?.reply ? body.reply : (body?.error || (res.status === 404 ? 'The voice helper isn’t deployed here.' : 'Something went wrong. Try again.')));
    } catch { say('I couldn’t reach the voice helper. Check your internet connection.'); }
    finally { vStatus('Ready'); }
  }
  function handleUtterance(text) {
    vLine('voiceHeard', text);
    const c = matchCommand(text);
    if (c) runCommand(c); else ask(text);
  }
  if (Rec) {
    vBtn.hidden = false;
    vBtn.onclick = () => (listening ? stopListening() : listen());
  } else {
    // No speech recognition (e.g. Firefox): no mic button, just a short note.
    $('voiceNote').hidden = false; setTimeout(() => { $('voiceNote').hidden = true; }, 8000);
  }
  $('voiceClose').onclick = () => { stopListening(); if (canTalk) speechSynthesis.cancel(); vPanel.hidden = true; };
  // Space: tap = capture (as before), hold = talk.
  let spaceTimer = null, spaceTalk = false;
  const spaceTarget = e => e.code === 'Space' && !e.target?.closest?.('input,select,button,textarea,[contenteditable]');
  addEventListener('keydown', e => {
    if (!spaceTarget(e)) return;
    e.preventDefault();
    if (e.repeat || spaceTimer || spaceTalk) return;
    spaceTimer = setTimeout(() => { spaceTimer = null; if (Rec) { spaceTalk = true; listen(); } }, 350);
  });
  addEventListener('keyup', e => {
    if (!spaceTarget(e)) return;
    e.preventDefault();
    if (spaceTimer) { clearTimeout(spaceTimer); spaceTimer = null; if (!$('btnCapture').disabled) $('btnCapture').click(); }
    else if (spaceTalk) { spaceTalk = false; stopListening(); }
  });

  renderCalib(); renderFindings(); syncButtons();
  addEventListener('pagehide', stopCamera);
})();
