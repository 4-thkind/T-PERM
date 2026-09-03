"""
gesture_engine.py
Stateless gesture classifiers + stateful detectors (throw, absent).
Pattern mirrors Gesture-Media-control baseline: deque buffers, EMA, cooldowns.
"""

from typing import Optional

import numpy as np

from hand_tracker import HandData
from utils.transforms import delta_rotation


# ── Distance helpers ──────────────────────────────────────────────────────────

def _norm_dist(lms, i: int, j: int) -> float:
    """Euclidean distance between two normalized landmarks."""
    a, b = lms[i], lms[j]
    return float(np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2))


# ── Spawn / grab ──────────────────────────────────────────────────────────────

def get_spawn_distance(h1: HandData, h2: HandData) -> float:
    return float(np.linalg.norm(h1.palm_center - h2.palm_center))


def is_cube_spawn_ready(hands: list[HandData], frame_w: int) -> bool:
    if len(hands) != 2:
        return False
    dist = get_spawn_distance(hands[0], hands[1])
    return dist <= frame_w * 0.40


def midpoint(h1: HandData, h2: HandData) -> np.ndarray:
    return (h1.palm_center + h2.palm_center) / 2.0


# ── Fist / Pinch ──────────────────────────────────────────────────────────────

def is_fist(hand: HandData) -> bool:
    """True if most fingers are tightly curled (fist)."""
    curled_count = 0
    # For each finger, if tip is closer to wrist than its MCP joint, it is curled.
    for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        d_tip = _norm_dist(hand.landmarks, tip, 0)
        d_mcp = _norm_dist(hand.landmarks, mcp, 0)
        if d_tip < d_mcp:
            curled_count += 1
    return curled_count >= 3


def is_open_palm(hand: HandData) -> bool:
    """True if all four fingers are extended (showing 5 / stop hand).
    Used as a 'lock' gesture — cube won't rotate while this hand is up."""
    extended = 0
    for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        d_tip = _norm_dist(hand.landmarks, tip, 0)
        d_mcp = _norm_dist(hand.landmarks, mcp, 0)
        if d_tip > d_mcp:
            extended += 1
    # Also require thumb to be somewhat spread (not pinching)
    thumb_idx_dist = _norm_dist(hand.landmarks, 4, 8)
    return extended >= 4 and thumb_idx_dist > 0.08


def is_pinch(hand: HandData) -> bool:
    """Thumb tip ↔ index tip closer than 0.06 normalised units."""
    d = _norm_dist(hand.landmarks, 4, 8)
    return d < 0.06


# ── Palm orientation ──────────────────────────────────────────────────────────

def is_palm_facing_down(hand: HandData) -> bool:
    if is_fist(hand) or is_pinch(hand):
        return False
    # Right hand palm down -> +Y, Left hand palm down -> -Y
    return hand.palm_normal[1] > 0.6 if hand.label == 'Right' else hand.palm_normal[1] < -0.6

def is_palm_facing_up(hand: HandData) -> bool:
    if is_fist(hand) or is_pinch(hand):
        return False
    # Right hand palm up -> -Y, Left hand palm up -> +Y
    return hand.palm_normal[1] < -0.6 if hand.label == 'Right' else hand.palm_normal[1] > 0.6


# ── Hands-absent detection ────────────────────────────────────────────────────

class AbsentDetector:
    """
    Frame-counter based. threshold_frames = 48 ≈ 0.8s @ 60fps.
    Same philosophy as baseline's gesture cooldown but inverted (counting absence).
    """

    def __init__(self, threshold_frames: int = 48):
        self.threshold = threshold_frames
        self.counter = 0

    def update(self, hands: list[HandData]) -> bool:
        if len(hands) == 0:
            self.counter += 1
        else:
            self.counter = 0
        return self.counter >= self.threshold

    def reset(self):
        self.counter = 0


# ── Wrist rotation delta ──────────────────────────────────────────────────────

def get_wrist_rotation_quat(prev_hand: Optional[HandData], curr_hand: HandData) -> Optional[np.ndarray]:
    """
    Returns delta rotation quaternion [x,y,z,w] from hand orientation change between frames.
    Uses both palm normal and finger direction for full 3D tracking.
    Returns None if prev_hand is unavailable or rotation is too small (noise).
    """
    if prev_hand is None:
        return None
    q = delta_rotation(
        prev_hand.palm_normal, curr_hand.palm_normal,
        prev_hand.finger_direction, curr_hand.finger_direction
    )
    # Ignore tiny rotations (noise floor)
    angle = 2.0 * np.degrees(np.arccos(np.clip(abs(q[3]), 0.0, 1.0)))
    if angle < 0.5:
        return None
    return q
