// app.js — Backend-only mode: displays MJPEG stream from Python Flask server
// All hand tracking, cube rendering, and gesture processing happens on the backend.

// server.py serves this page itself, so same-origin is the normal case and it
// keeps working when the backend is reached from another device on the LAN.
// Only fall back to a hardcoded host when opened straight off the filesystem.
const BACKEND_URL = location.protocol === 'file:' ? 'http://localhost:5000' : '';

const els = {
  boot: document.getElementById('boot-screen'),
  stage: document.getElementById('stage'),
  startBtn: document.getElementById('start-btn'),
  bootError: document.getElementById('boot-error'),
  backendStream: document.getElementById('backend-stream'),
  telemetryMode: document.getElementById('tel-mode'),
  telemetryState: document.getElementById('tel-state'),
  telemetryHands: document.getElementById('tel-hands'),
  telemetryFps: document.getElementById('tel-fps'),
  telemetryDetFps: document.getElementById('tel-detfps'),
  telemetryLag: document.getElementById('tel-lag'),
  telemetryNumHands: document.getElementById('tel-numhands'),
  resetBtn: document.getElementById('reset-btn'),
  legendToggle: document.getElementById('legend-toggle'),
  quitBtn: document.getElementById('quit-btn'),
  quitScreen: document.getElementById('quit-screen'),
  legendPanel: document.getElementById('legend-panel'),
};

els.startBtn.addEventListener('click', start);
els.resetBtn.addEventListener('click', async () => {
  try { await fetch(`${BACKEND_URL}/api/reset`, { method: 'POST' }); } catch (e) {}
});
els.legendToggle.addEventListener('click', () => {
  els.legendPanel.classList.toggle('open');
});

let statusTimer = null;

// Stop the backend and free the terminal. The request is expected to die
// mid-flight when the process exits, so a rejected fetch here is success.
els.quitBtn.addEventListener('click', async () => {
  if (!confirm('Stop the T-PERM server? The camera will be released and the terminal will return to a prompt.')) return;
  els.quitBtn.disabled = true;
  els.quitBtn.textContent = 'Stopping…';
  if (statusTimer) clearInterval(statusTimer);
  els.backendStream.src = '';          // drop the MJPEG stream so it stops retrying
  try {
    await fetch(`${BACKEND_URL}/api/shutdown`, { method: 'POST' });
  } catch (e) { /* connection dropped as the process exited - expected */ }
  els.quitScreen.hidden = false;
});

async function checkBackendServer() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);
    const res = await fetch(`${BACKEND_URL}/health`, { signal: controller.signal });
    clearTimeout(timeoutId);
    return res.ok;
  } catch (e) {
    return false;
  }
}

async function start() {
  els.startBtn.disabled = true;
  els.startBtn.textContent = 'Connecting to backend\u2026';
  els.bootError.textContent = '';

  const backendAvailable = await checkBackendServer();

  if (!backendAvailable) {
    els.startBtn.disabled = false;
    els.startBtn.textContent = 'Engage camera';
    els.bootError.textContent =
      'Backend server not found. Start the Python backend first: cd backend && python server.py';
    return;
  }

  if (els.telemetryMode) els.telemetryMode.textContent = 'PYTHON BACKEND';
  els.startBtn.textContent = 'Connecting to stream\u2026';

  // Display the MJPEG stream from the backend
  els.backendStream.src = `${BACKEND_URL}/video_feed`;
  els.backendStream.hidden = false;

  els.boot.hidden = true;
  els.stage.hidden = false;

  // Poll backend status telemetry
  statusTimer = setInterval(async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/status`);
      if (res.ok) {
        const data = await res.json();
        if (els.telemetryState) els.telemetryState.textContent = data.state || 'RUNNING';
        if (els.telemetryHands) els.telemetryHands.textContent = data.hands ?? '0';
        if (els.telemetryFps) els.telemetryFps.textContent = data.fps ?? '0';
        if (els.telemetryDetFps) els.telemetryDetFps.textContent = data.det_fps ?? '0';
        if (els.telemetryLag) els.telemetryLag.textContent = `${data.lag_ms ?? 0}ms`;
        if (els.telemetryNumHands) els.telemetryNumHands.textContent = data.num_hands ?? '-';
      }
    } catch (e) {}
  }, 250);
}
