#!/usr/bin/env node

'use strict';

const { execSync, spawn } = require('child_process');
const http = require('http');
const path = require('path');
const fs = require('fs');

// Under `npx` these files live in the npm cache next to this script; run from a
// clone, the same relative path applies.
const ROOT    = path.resolve(__dirname, '..');
const BACKEND = path.join(ROOT, 'backend');
const PORT    = Number(process.env.T_PERM_PORT) || 5000;
const URL     = `http://localhost:${PORT}`;

let serverExited = false;

// ── Helpers ─────────────────────────────────────────────────────────────────

function run(cmd, opts = {}) {
  return execSync(cmd, { stdio: 'inherit', ...opts });
}

function tryRun(cmd) {
  try { execSync(cmd, { stdio: 'ignore' }); return true; } catch { return false; }
}

function capture(cmd) {
  try { return execSync(cmd, { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim(); }
  catch { return null; }
}

function openBrowser(url) {
  const cmd =
    process.platform === 'win32'  ? `start "" "${url}"` :
    process.platform === 'darwin' ? `open "${url}"` :
                                    `xdg-open "${url}"`;
  try { execSync(cmd); } catch { /* not fatal - the URL is printed anyway */ }
}

function abort(msg) {
  console.error('\n' + msg + '\n');
  process.exit(1);
}

// Poll /health until the engine reports ready. The old code slept a flat 3s,
// which opened the browser onto a dead port on a slow machine and wasted time
// on a fast one.
function waitForServer(timeoutMs = 90000) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    (function poll() {
      if (serverExited) return reject(new Error('server exited during startup'));
      if (Date.now() > deadline) return reject(new Error('timed out waiting for the server'));
      const req = http.get(`${URL}/health`, (res) => {
        res.resume();
        if (res.statusCode === 200) resolve();
        else setTimeout(poll, 300);
      });
      req.on('error', () => setTimeout(poll, 300));
      req.setTimeout(1000, () => { req.destroy(); });
    })();
  });
}

// ── Step 1: Detect Python ────────────────────────────────────────────────────

const py =
  tryRun('python --version')  ? 'python'  :
  tryRun('python3 --version') ? 'python3' :
  null;

if (!py) {
  abort(
    'Python not found.\n' +
    '  Install Python 3.9+ from https://python.org and make sure it is on your PATH.'
  );
}

// ── Step 2: Confirm the MediaPipe model shipped ─────────────────────────────

const TASK = path.join(BACKEND, 'hand_landmarker.task');
if (!fs.existsSync(TASK)) {
  abort(
    `MediaPipe model missing: ${TASK}\n` +
    '  If you installed via npx:  npm cache clean --force && npx tperm-visor\n' +
    '  If you cloned the repo:    re-clone; the model is committed at\n' +
    '                             backend/hand_landmarker.task'
  );
}

// ── Step 3: Install Python dependencies, only if missing ────────────────────

// Re-running pip every launch costs seconds and writes to the user's system
// Python for no reason. Import the heavy deps first; install only if that fails.
const DEPS_OK = !process.argv.includes('--deps') &&
  tryRun(`${py} -c "import cv2, mediapipe, flask, flask_cors, OpenGL, pyglet, scipy"`);

if (!DEPS_OK) {
  console.log('\nInstalling Python dependencies (first run only, ~1-2 min)...\n');
  try {
    run(`${py} -m pip install -r "${path.join(BACKEND, 'requirements.txt')}"`);
  } catch {
    abort('pip install failed. Check the error above and retry.');
  }
}

// ── Step 4: Start the backend ────────────────────────────────────────────────

console.log('\nStarting T-PERM...\n');

const server = spawn(py, ['server.py'], {
  cwd: BACKEND,
  stdio: 'inherit',
  env: { ...process.env, PYTHONUNBUFFERED: '1', T_PERM_PORT: String(PORT) },
});

server.on('error', (err) => abort('Failed to start the Python server: ' + err.message));
server.on('exit', (code) => {
  serverExited = true;
  process.exit(code ?? 0);
});

// ── Step 5: Open the browser once the server actually answers ───────────────

waitForServer()
  .then(() => {
    console.log(`\n  T-PERM is running at ${URL}`);
    console.log('  Press the "Stop server" button in the page, or Ctrl+C here, to quit.\n');
    openBrowser(URL);
  })
  .catch((err) => {
    if (!serverExited) console.error('\n' + err.message + `\n  Try opening ${URL} manually.\n`);
  });

// ── Graceful shutdown ────────────────────────────────────────────────────────

let shuttingDown = false;
function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  server.kill('SIGINT');
  // If the engine is wedged mid-frame, do not hang the terminal forever.
  setTimeout(() => { server.kill('SIGKILL'); process.exit(0); }, 5000).unref();
}
process.on('SIGINT',  shutdown);
process.on('SIGTERM', shutdown);
