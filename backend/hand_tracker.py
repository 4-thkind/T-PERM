"""
hand_tracker.py
MediaPipe Hand Landmarker setup + skeleton draw + HandData extraction.
Threading model and CONNECTIONS list copied from Gesture-Media-control baseline.
"""

import time
import threading
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.transforms import compute_palm_normal, compute_finger_direction

# ── Exact CONNECTIONS from baseline (pranavpant9916-ctrl/Gesture-Media-control) ──
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# Landmark roles
IDX_WRIST      = 0
IDX_THUMB_TIP  = 4
IDX_INDEX_TIP  = 8

PALM_LANDMARKS = [0, 5, 9, 13, 17]  # for palm center average


# ── Threaded webcam stream ────────────────────────────────────────────────────

class WebcamStream:
    """Runs the webcam on a separate thread to achieve 60 fps without lag."""

    def __init__(self, src: int = 0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            self.stream = cv2.VideoCapture(src)
        # Request 720p 60fps
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.stream.set(cv2.CAP_PROP_FPS, 60)
        self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self._lock = threading.Lock()
        self._thread = None  # FIX: keep reference so we can join() on shutdown

    def start(self):
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            # FIX: if camera returns bad frames (e.g. after release is called),
            # sleep briefly instead of spinning at 100% CPU hammering a dead stream
            if not grabbed or frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self._lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def isOpened(self) -> bool:
        return self.stream.isOpened()

    def release(self):
        self.stopped = True
        # FIX: wait for the _update thread to finish its current stream.read() call
        # before releasing the camera handle.
        #
        # Without this join(), stream.release() fires while the thread is still
        # inside stream.read(). On Windows DirectShow (CAP_DSHOW), the camera
        # handle is reference-counted at the driver level — it stays open until
        # ALL threads exit their read() calls. A new process that opens the same
        # camera then gets a contested, half-initialised handle → lag and jitter.
        #
        # join(timeout=2.0) gives the thread up to 2s to finish its current read
        # (one frame at 60fps = ~16ms, so 2s is extremely conservative).
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.stream.release()


# ── HandData dataclass ────────────────────────────────────────────────────────

@dataclass
class HandData:
    label: str              # "Left" or "Right"
    landmarks: list         # all 21 raw landmark objects (normalized .x .y .z)
    palm_center: np.ndarray # pixel (x, y)
    palm_normal: np.ndarray # 3-D unit vector
    finger_direction: np.ndarray  # 3-D unit vector (wrist → middle MCP)


# ── HandTracker ───────────────────────────────────────────────────────────────

class HandTracker:
    def __init__(self, model_path: str = 'hand_landmarker.task',
                 result_callback=None):
        """result_callback: callable(result, image, timestamp_ms) — required for LIVE_STREAM."""
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            result_callback=result_callback,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect_async(self, rgb_frame: np.ndarray, timestamp_ms: int):
        """Non-blocking detect — LIVE_STREAM mode. Result arrives via callback."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.detector.detect_async(mp_image, timestamp_ms)

    def extract_hands(self, result, frame_w: int, frame_h: int) -> list[HandData]:
        hands = []
        if not result.hand_landmarks:
            return hands

        for lms, handedness in zip(result.hand_landmarks, result.handedness):
            palm_pixels = np.array(
                [[lms[i].x * frame_w, lms[i].y * frame_h] for i in PALM_LANDMARKS]
            )
            hands.append(HandData(
                label=handedness[0].category_name,  # "Left" or "Right"
                landmarks=lms,
                palm_center=palm_pixels.mean(axis=0),
                palm_normal=compute_palm_normal(lms),
                finger_direction=compute_finger_direction(lms),
            ))
        return hands

    def draw_skeleton(self, frame: np.ndarray, result) -> None:
        """
        Draw the full MediaPipe skeleton overlay.
        Stays visible in EVERY state as long as hands are in frame.
        - Bones: cyan, thickness 2
        - Joints: white, radius 4
        - Index tip (8) & thumb tip (4): red, radius 7
        - Wrist (0): amber, radius 8
        Landmarks are normalised 0-1, so this scales correctly onto the full-size
        frame even though detection ran on a 1/3-scale copy.
        """
        if not result.hand_landmarks:
            return

        h, w = frame.shape[:2]
        for lms in result.hand_landmarks:
            # Bones
            for s, e in CONNECTIONS:
                sp = (int(lms[s].x * w), int(lms[s].y * h))
                ep = (int(lms[e].x * w), int(lms[e].y * h))
                cv2.line(frame, sp, ep, (0, 255, 255), 2, cv2.LINE_AA)

            # Joints
            for idx, lm in enumerate(lms):
                px = (int(lm.x * w), int(lm.y * h))
                if idx == IDX_WRIST:
                    cv2.circle(frame, px, 8, (0, 220, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, px, 9, (0, 0, 0), 1, cv2.LINE_AA)
                elif idx in (IDX_THUMB_TIP, IDX_INDEX_TIP):
                    cv2.circle(frame, px, 7, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, px, 8, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, px, 4, (255, 255, 255), -1, cv2.LINE_AA)

    def close(self):
        self.detector.close()