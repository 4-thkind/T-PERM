"""
hud.py
All HUD overlays — glassmorphism pill style copied from Gesture-Media-control baseline.
Confetti, state label, solved/unsolved banner, spawn ring.
"""

import random

import cv2
import numpy as np


# ── Glassmorphism helpers (from baseline) ─────────────────────────────────────

def _glass_pill(frame: np.ndarray, cx: int, cy: int, text: str,
                font_scale: float = 0.8, thickness: int = 2,
                alpha: float = 1.0, text_color=(255, 255, 255)):
    """Lightweight pill overlay — no blur, no full-frame copy."""
    if alpha < 0.01:
        return
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    px, py = 30, 15
    pill_w = text_size[0] + px * 2
    pill_h = text_size[1] + py * 2
    x1 = cx - pill_w // 2
    y1 = cy - pill_h // 2
    x2, y2 = x1 + pill_w, y1 + pill_h
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return

    # Draw a dark semi-transparent rounded rectangle (ROI-only, no full copy)
    overlay = frame[y1:y2, x1:x2].copy()
    dark = np.full_like(overlay, (18, 20, 24))
    blended = cv2.addWeighted(dark, 0.55 * alpha, overlay, 1.0 - 0.55 * alpha, 0)
    frame[y1:y2, x1:x2] = blended

    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)

    tx = cx - text_size[0] // 2
    ty = cy + text_size[1] // 2
    cv2.putText(frame, text, (tx + 1, ty + 1), font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(frame, text, (tx, ty), font, font_scale, text_color, thickness)


# ── State label (bottom centre) ───────────────────────────────────────────────

STATE_LABELS = {
    'IDLE':             'Show both hands to spawn cube',
    'SPAWN_READY':      'Spawning...',
    'HOLDING':          'Twist wrist to rotate • Pinch to turn face',
    'DRAGGING_CUBE':    'Moving cube...',
    'DRAGGING_SLICE':   'Turning layer...',
    'COMPLETION_CHECK': 'Checking...',
}

def draw_state_label(frame: np.ndarray, state: str, alpha: float = 1.0):
    h, w = frame.shape[:2]
    label = STATE_LABELS.get(state, state)
    _glass_pill(frame, w // 2, h - 55, label, font_scale=0.6, alpha=alpha)


# ── Spawn pulsing ring ────────────────────────────────────────────────────────

def draw_spawn_ring(frame: np.ndarray, cx: int, cy: int, progress: float):
    """progress 0→1 drives scale-in animation."""
    r = int(60 * progress)
    if r < 2:
        return
    alpha_ring = max(0.0, 1.0 - progress * 0.5)
    color = (0, int(255 * progress), 255)
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), r, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha_ring, frame, 1 - alpha_ring, 0, frame)


# ── Solved / unsolved banner ──────────────────────────────────────────────────

def draw_solved_banner(frame: np.ndarray, solved: bool, alpha: float = 1.0):
    h, w = frame.shape[:2]
    if solved:
        text, color = 'SOLVED!', (0, 255, 100)
    else:
        text, color = 'NOT YET — KEEP GOING', (80, 80, 255)
    _glass_pill(frame, w // 2, h // 2, text,
                font_scale=1.2, thickness=3, alpha=alpha, text_color=color)


# ── Red border flash (unsolved) ───────────────────────────────────────────────

def draw_fail_border(frame: np.ndarray, alpha: float = 0.6):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (0, 0, 220), 12)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ── Confetti ──────────────────────────────────────────────────────────────────

_confetti_particles = []

def reset_confetti(frame_w: int, n: int = 120):
    """Seed particles just above the top edge; they fall into view from there."""
    global _confetti_particles
    _confetti_particles = []
    colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255),
              (255, 255, 80), (255, 80, 255), (80, 255, 255)]
    for _ in range(n):
        _confetti_particles.append({
            'x': random.randint(0, frame_w),
            'y': random.randint(-60, 0),
            'vx': random.uniform(-2, 2),
            'vy': random.uniform(3, 9),
            'color': random.choice(colors),
            'size': random.randint(6, 16),
            'rot': random.uniform(0, 360),
            'rot_vel': random.uniform(-5, 5),
        })


def draw_confetti(frame: np.ndarray):
    h, w = frame.shape[:2]
    alive = []
    for p in _confetti_particles:
        p['x'] += p['vx']
        p['y'] += p['vy']
        p['rot'] += p['rot_vel']
        if p['y'] < h + 20:
            alive.append(p)
            s = p['size']
            cx, cy = int(p['x']), int(p['y'])
            cv2.rectangle(frame, (cx - s//2, cy - s//4),
                          (cx + s//2, cy + s//4), p['color'], -1)
    _confetti_particles[:] = alive
