"""
Find out why hand tracking is slow, on THIS machine with YOUR hand.

    python diagnose_tracking.py          (stop server.py first - it holds the camera)

Or, without stopping the server, hit http://localhost:5000/api/diagnose instead.

Everything is measured with a real hand present: with an empty frame MediaPipe
finds nothing and returns before running the landmark model, which makes it look
~10x faster than it is and is exactly how you get a wrong answer here.
"""

import statistics
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL = 'hand_landmarker.task'


def grab_frame_with_hand(seconds=8):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Cannot open the camera. Is server.py still running?")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.4))

    print("\nHold ONE hand up in front of the camera...")
    deadline = time.time() + seconds
    frame = None
    while time.time() < deadline:
        ok, f = cap.read()
        if not ok or f is None:
            continue
        f = cv2.flip(f, 1)
        rgb = cv2.cvtColor(cv2.resize(f, (f.shape[1] // 3, f.shape[0] // 3)),
                           cv2.COLOR_BGR2RGB)
        if det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).hand_landmarks:
            frame = f
            print("  hand detected - captured a %dx%d frame\n" % (f.shape[1], f.shape[0]))
            break
    cap.release()
    det.close()
    if frame is None:
        sys.exit("No hand detected in %ds. Try better lighting and rerun." % seconds)
    return frame


def bench(frame, num_hands, divisor, n=25):
    """Median ms per detection, in VIDEO mode so frame-to-frame tracking applies."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // divisor, h // divisor))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
        min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4))

    ts = 0
    for _ in range(5):                       # warm up / establish tracking
        ts += 33
        det.detect_for_video(img, ts)

    times, found = [], 0
    for _ in range(n):
        ts += 33
        t0 = time.perf_counter()
        r = det.detect_for_video(img, ts)
        times.append((time.perf_counter() - t0) * 1000)
        found += len(r.hand_landmarks)
    det.close()
    return statistics.median(times), found / n, small.shape[1], small.shape[0]


if __name__ == '__main__':
    frame = grab_frame_with_hand()

    print("%-9s %-12s %10s %10s %8s" % ("num_hands", "input", "ms/detect", "det/sec", "hands"))
    print("-" * 54)
    results = {}
    for num_hands in (2, 1):
        for divisor in (2, 3, 4, 6):
            ms, hands, iw, ih = bench(frame, num_hands, divisor)
            results[(num_hands, divisor)] = (ms, hands)
            print("%-9d %-12s %10.1f %10.1f %8.1f"
                  % (num_hands, "%dx%d" % (iw, ih), ms, 1000.0 / ms, hands))

    usable = {k: v[0] for k, v in results.items() if v[1] > 0}
    if not usable:
        sys.exit("\nNo config found a hand - results are meaningless. Rerun.")
    cur = results[(2, 3)][0]
    best = min(usable, key=usable.get)
    print("\ncurrent setting (num_hands=2, 1/3 scale): %.1f ms  ->  %.0f detections/sec"
          % (cur, 1000.0 / cur))
    print("fastest usable  (num_hands=%d, 1/%d scale): %.1f ms  ->  %.0f detections/sec  (%.2fx)"
          % (best[0], best[1], usable[best], 1000.0 / usable[best], cur / usable[best]))
