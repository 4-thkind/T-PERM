"""
server.py — AR Rubik's Cube Flask Server (Main Thread OpenGL Engine)
Runs PyOpenGL + MediaPipe AR Engine on the Main Thread (required for Windows WGL/Pyglet)
and Flask server on a background thread.
"""

import time
import threading
import signal
import socket
from enum import Enum, auto
import cv2
import numpy as np
import os
from flask import Flask, Response, jsonify, send_from_directory
from flask_cors import CORS

import pyglet
from pyglet.gl import *

from hand_tracker import HandTracker, WebcamStream
from gesture_engine import (
    AbsentDetector,
    get_spawn_distance, is_cube_spawn_ready,
    midpoint, is_pinch, is_fist, is_open_palm, get_wrist_rotation_quat,
    is_palm_facing_down, is_palm_facing_up
)
from cube.rubiks import solved_state, scramble, is_solved, apply_move, FACE_TO_MOVE
from cube.renderer import CubeRenderer
from utils.smoothing import EMA, QuatEMA
from utils.transforms import quat_multiply, cube_axes_on_screen, snap_to_nearest_90
import hud


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


class State(Enum):
    IDLE             = auto()
    SPAWN_READY      = auto()
    HOLDING          = auto()
    DRAGGING_CUBE    = auto()
    DRAGGING_SLICE   = auto()
    COMPLETION_CHECK = auto()


def ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


class AREngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_event = threading.Event()
        self.running = False
        self.state = State.IDLE
        self.hand_count = 0
        self.fps = 0
        self.reset_requested = False
        self.current_jpeg = None
        self._perf_start = time.perf_counter()  # wall-clock base for MediaPipe timestamps

    def reset_cube(self):
        with self.lock:
            self.reset_requested = True

    def run_main_loop(self):
        """Main thread loop for Windows Pyglet / OpenGL compatibility."""
        print("Initializing Pyglet OpenGL Context on Main Thread...")
        try:
            config = pyglet.gl.Config(double_buffer=False, depth_size=24, alpha_size=8)
            gl_window = pyglet.window.Window(width=1, height=1, visible=False, config=config)
        except Exception as e:
            print(f"Config fallback: {e}")
            gl_window = pyglet.window.Window(width=1, height=1, visible=False)

        gl_window.switch_to()

        print("Starting camera thread...")
        cap = WebcamStream(0).start()
        time.sleep(1.5)                         # FIX: was 0.8 — give camera driver time to settle
        if not cap.isOpened():
            print("ERROR: Cannot open camera.")
            return

        # FIX: Discard first 30 frames so camera auto-exposure/focus can stabilise.
        # These frames are always garbage (dark, over-exposed, blurry) and are the
        # #1 cause of jitter/lag on manual restart. Without this, MediaPipe fires on
        # bad frames and produces noisy detections for the first few seconds.
        print("Warming up camera (30 frames)...")
        frame = None
        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None:
                time.sleep(0.02)                # ~20ms gap — don't hammer the driver
        if frame is None:
            print("ERROR: No frames from camera.")
            cap.release()
            return
        frame_h, frame_w = frame.shape[:2]
        print(f"Camera ready: {frame_w}x{frame_h}")

        renderer = CubeRenderer(frame_w, frame_h)
        renderer.init_gl()
        # ── Async detection via LIVE_STREAM callback ─────────────────────
        _cb_lock = threading.Lock()
        _latest_hands  = [[]]
        _latest_result = [None]
        _latest_gen    = [0]

        def _on_detection(result, image, timestamp_ms):
            """Called by MediaPipe on its own thread whenever results are ready."""
            try:
                hands = tracker.extract_hands(result, frame_w, frame_h)
            except Exception:
                hands = []
            with _cb_lock:
                _latest_hands[0]  = hands
                _latest_result[0] = result
                _latest_gen[0]   += 1

        tracker = HandTracker('hand_landmarker.task', result_callback=_on_detection)
        last_processed_gen = 0
        # ────────────────────────────────────────────────────────────────────

        cube_state = solved_state()
        cube_state, _ = scramble(cube_state, n=20)

        state = State.IDLE
        cube_pos = np.array([frame_w / 2.0, frame_h / 2.0])
        cube_rotation = np.array([0.0, 0.0, 0.0, 1.0])
        cube_scale = 0.25
        spawn_scale = 0.25

        pos_ema = EMA(alpha=0.35)
        rot_ema = QuatEMA(alpha=0.25)

        spawn_frame_count = 0
        SPAWN_FRAMES = 20

        face_rot_angle = 0.0
        face_rot_face = None
        SNAP_FRAMES = 10
        snap_frame = 0
        snap_start_angle = 0.0
        snap_target_angle = 0.0
        snapping = False
        active_pointer_3d = None
        drag_start_pos = None
        drag_direction = None
        DRAG_LOCK_THRESHOLD = 15
        pinch_released = True

        prev_hands = {}
        absent_detector = AbsentDetector(threshold_frames=48)
        completion_start_time = 0.0
        completion_solved = False
        banner_alpha = 0.0
        pointers_2d = {}

        confetti_active = False
        frame_times = []
        self.running = True
        print(">>> AR Engine Ready and Processing Frames! <<<")

        try:
            while self.running:
                t_start = time.time()
                with self.lock:
                    if self.reset_requested:
                        cube_state = solved_state()
                        cube_state, _ = scramble(cube_state, n=20)
                        cube_rotation = np.array([0.0, 0.0, 0.0, 1.0])
                        rot_ema.reset()
                        self.reset_requested = False

                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                frame_h, frame_w = frame.shape[:2]

                # Submit frame to async detection (non-blocking, returns in <0.1ms).
                # 1/3 scale gives MediaPipe 9x fewer pixels to chew; landmarks are
                # normalised 0-1 so the downscale costs no accuracy in mapping back.
                # Submit every frame — LIVE_STREAM mode auto-drops when busy.
                timestamp_ms = int((time.perf_counter() - self._perf_start) * 1000)
                det_w, det_h = frame_w // 3, frame_h // 3
                small = cv2.resize(frame, (det_w, det_h))
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                try:
                    tracker.detect_async(rgb_small, timestamp_ms)
                except ValueError:
                    pass  # timestamp not monotonically increasing — skip this frame

                # Read latest results from callback
                with _cb_lock:
                    hands  = _latest_hands[0]
                    result = _latest_result[0]
                    cur_gen = _latest_gen[0]

                has_new_detection = (cur_gen != last_processed_gen)
                if has_new_detection:
                    last_processed_gen = cur_gen

                if result is not None:
                    tracker.draw_skeleton(frame, result)

                self.hand_count = len(hands)

                # State machine — only update gestures when detection has new results
                # On stale frames, we still render the cube but skip gesture logic
                if has_new_detection:
                    if state == State.IDLE:
                        absent_detector.reset()
                        spawn_frame_count = 0
                        prev_hands.clear()
                        if is_cube_spawn_ready(hands, frame_w):
                            state = State.SPAWN_READY

                    elif state == State.SPAWN_READY:
                        if len(hands) == 2:
                            mid = midpoint(hands[0], hands[1])
                            cube_pos = pos_ema.update(mid)
                            dist = get_spawn_distance(hands[0], hands[1])
                            spawn_scale = np.clip(dist / frame_w * 2.5, 0.4, 1.2)

                            spawn_frame_count += 1
                            progress = spawn_frame_count / SPAWN_FRAMES
                            cube_scale = spawn_scale * ease_out_cubic(progress)
                            cx, cy = int(cube_pos[0]), int(cube_pos[1])
                            hud.draw_spawn_ring(frame, cx, cy, ease_out_cubic(progress))

                            if spawn_frame_count >= SPAWN_FRAMES:
                                cube_scale = spawn_scale
                                state = State.HOLDING
                        else:
                            state = State.IDLE

                    elif state == State.HOLDING:
                        if absent_detector.update(hands):
                            state = State.COMPLETION_CHECK
                            completion_start_time = time.time()
                            completion_solved = is_solved(cube_state)
                            if completion_solved:
                                hud.reset_confetti(frame_w)
                                confetti_active = True
                            absent_detector.reset()

                        lock_hand = None
                        if len(hands) >= 2:
                            lock_hand = next((h for h in hands if is_open_palm(h)), None)

                        if not lock_hand:
                            palm_pitch_detected = False
                            for hand in hands:
                                if is_palm_facing_down(hand):
                                    cube_rotation = np.array([0.7071, 0.0, 0.0, 0.7071])
                                    palm_pitch_detected = True
                                    break
                                elif is_palm_facing_up(hand):
                                    cube_rotation = np.array([-0.7071, 0.0, 0.0, 0.7071])
                                    palm_pitch_detected = True
                                    break

                            if not palm_pitch_detected:
                                for hand in hands:
                                    prev = prev_hands.get(hand.label)
                                    dq = get_wrist_rotation_quat(prev, hand)
                                    if dq is not None:
                                        cube_rotation = quat_multiply(cube_rotation, dq)
                                        norm = np.linalg.norm(cube_rotation)
                                        if norm > 1e-6:
                                            cube_rotation /= norm

                        pinching_hand = next((h for h in hands if is_pinch(h)), None)
                        fist_hand = next((h for h in hands if is_fist(h)), None)

                        if not pinching_hand:
                            pinch_released = True

                        if fist_hand and not lock_hand:
                            px, py = fist_hand.palm_center
                            dist_to_center = np.hypot(px - cube_pos[0], py - cube_pos[1])
                            if dist_to_center < 180 * cube_scale / 0.25:
                                drag_start_pos = (px, py)
                                state = State.DRAGGING_CUBE
                        elif pinching_hand and not snapping and pinch_released:
                            px = pinching_hand.landmarks[8].x * frame_w
                            py = pinching_hand.landmarks[8].y * frame_h
                            # Hit-test against the PREVIOUS frame's projected pointers.
                            # Rendering here just to refresh them would cost a second full
                            # FBO draw + glReadPixels every pinch frame, and would advance
                            # rot_ema twice in one tick (making the smoothing jerk on pinch).
                            # One frame of staleness at ~45fps is ~22ms — imperceptible.
                            closest_p3d = None
                            closest_dist = float('inf')
                            for p3d, (cx, cy) in pointers_2d.items():
                                dist = np.hypot(px - cx, py - cy)
                                if dist < closest_dist:
                                    closest_dist = dist
                                    closest_p3d = p3d

                            if closest_dist < 60:
                                active_pointer_3d = closest_p3d
                                drag_start_pos = (px, py)
                                face_rot_angle = 0.0
                                pinch_released = False
                                state = State.DRAGGING_SLICE

                        prev_hands = {h.label: h for h in hands}

                    elif state == State.DRAGGING_CUBE:
                        fist_hand = next((h for h in hands if is_fist(h)), None)

                        if fist_hand:
                            px, py = fist_hand.palm_center
                            dx = px - drag_start_pos[0]
                            dy = py - drag_start_pos[1]
                            cube_pos[0] += dx
                            cube_pos[1] += dy
                            drag_start_pos = (px, py)
                        else:
                            pos_ema.value = cube_pos.copy()
                            state = State.HOLDING

                        prev_hands = {h.label: h for h in hands}

                    elif state == State.DRAGGING_SLICE:
                        pinching_hand = next((h for h in hands if is_pinch(h)), None)
                        if pinching_hand and not snapping:
                            px = pinching_hand.landmarks[8].x * frame_w
                            py = pinching_hand.landmarks[8].y * frame_h
                            dx = px - drag_start_pos[0]
                            dy = py - drag_start_pos[1]

                            if drag_direction is None:
                                smooth_q = rot_ema.update(cube_rotation)
                                if abs(dx) > DRAG_LOCK_THRESHOLD or abs(dy) > DRAG_LOCK_THRESHOLD:
                                    screen_x, screen_y = cube_axes_on_screen(smooth_q)
                                    swipe = np.array([dx, dy])
                                    proj_x = abs(np.dot(swipe, screen_x))
                                    proj_y = abs(np.dot(swipe, screen_y))
                                    if proj_x > proj_y:
                                        drag_direction = 'ROW'
                                        if active_pointer_3d[1] > 0.1: face_rot_face = 'U'
                                        elif active_pointer_3d[1] < -0.1: face_rot_face = 'D'
                                        else: face_rot_face = 'E'
                                    else:
                                        drag_direction = 'COL'
                                        if active_pointer_3d[0] > 0.1: face_rot_face = 'R'
                                        elif active_pointer_3d[0] < -0.1: face_rot_face = 'L'
                                        else: face_rot_face = 'M'

                            if drag_direction is not None and face_rot_face is not None:
                                smooth_q = rot_ema.update(cube_rotation)
                                screen_x, screen_y = cube_axes_on_screen(smooth_q)
                                swipe = np.array([dx, dy])
                                if drag_direction == 'ROW':
                                    proj = np.dot(swipe, screen_x)
                                    sign = -1.0 if face_rot_face in ('U', 'E') else 1.0
                                    face_rot_angle = sign * proj / 2.0
                                elif drag_direction == 'COL':
                                    proj = np.dot(swipe, screen_y)
                                    sign = -1.0 if face_rot_face in ('R', 'M') else 1.0
                                    face_rot_angle = sign * proj / 2.0
                        elif not snapping:
                            snap_target_angle = snap_to_nearest_90(face_rot_angle)
                            snap_start_angle = face_rot_angle
                            snap_frame = 0
                            snapping = True

                        prev_hands = {h.label: h for h in hands}

                    elif state == State.COMPLETION_CHECK:
                        elapsed = time.time() - completion_start_time
                        banner_alpha = min(elapsed / 0.4, 1.0)

                        if not completion_solved:
                            flash_alpha = max(0.0, 1.0 - elapsed / 0.8)
                            if flash_alpha > 0.01:
                                hud.draw_fail_border(frame, alpha=flash_alpha * 0.6)

                        duration = 3.0 if completion_solved else 2.0
                        if elapsed >= duration:
                            if completion_solved:
                                cube_state = solved_state()
                                cube_state, _ = scramble(cube_state, n=20)
                                cube_rotation = np.array([0.0, 0.0, 0.0, 1.0])
                                rot_ema.reset()
                            confetti_active = False
                            state = State.IDLE

                # ── Snap animation runs every frame (time-based, not detection-based) ──
                if state == State.DRAGGING_SLICE and snapping:
                    snap_frame += 1
                    t = min(snap_frame / SNAP_FRAMES, 1.0)
                    t_ease = ease_out_cubic(t)
                    face_rot_angle = snap_start_angle + (snap_target_angle - snap_start_angle) * t_ease

                    if snap_frame >= SNAP_FRAMES:
                        if face_rot_face:
                            turns = int(round(snap_target_angle / 90.0)) % 4
                            cw_move, ccw_move = FACE_TO_MOVE.get(face_rot_face, ('F', "F'"))
                            for _ in range(abs(turns)):
                                move = cw_move if turns > 0 else ccw_move
                                cube_state = apply_move(cube_state, move)

                        face_rot_face = None
                        face_rot_angle = 0.0
                        snapping = False
                        active_pointer_3d = None
                        drag_direction = None
                        state = State.HOLDING

                # ── Always render the cube at current state ──
                if state in (State.HOLDING, State.DRAGGING_CUBE, State.DRAGGING_SLICE):
                    smooth_q = rot_ema.update(cube_rotation)
                    frame, pointers_2d = renderer.render(
                        frame, cube_state, cube_pos, smooth_q,
                        cube_scale=cube_scale,
                        face_rotating=(face_rot_face, face_rot_angle) if face_rot_face else None,
                        highlighted_pointer=active_pointer_3d,
                    )
                elif state == State.COMPLETION_CHECK:
                    frame, _ = renderer.render(
                        frame, cube_state, cube_pos, rot_ema.value,
                        cube_scale=cube_scale,
                    )
                    if confetti_active:
                        hud.draw_confetti(frame)
                    hud.draw_solved_banner(frame, completion_solved, alpha=banner_alpha if has_new_detection else 1.0)

                if state != State.COMPLETION_CHECK:
                    hud.draw_state_label(frame, state.name)

                self.state = state

                # FPS calculation
                t_end = time.time()
                frame_times.append(t_end - t_start)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                avg_time = sum(frame_times) / len(frame_times) if frame_times else 0.033
                self.fps = int(1.0 / avg_time) if avg_time > 0 else 0

                # Encode frame to JPEG (quality 65 is ~2x faster than 80, still looks good)
                ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                if ok:
                    with self.lock:
                        self.current_jpeg = jpeg.tobytes()
                    self.frame_event.set()

                # Frame pacing: target ~45fps to free CPU for MediaPipe detection
                elapsed = time.time() - t_start
                target = 1.0 / 45.0
                if elapsed < target:
                    time.sleep(target - elapsed)

        finally:
            self.running = False
            renderer.cleanup()
            tracker.close()
            cap.release()
            gl_window.close()
            print("Engine stopped cleanly.")

    def get_stream(self):
        while True:
            jpeg_bytes = None
            with self.lock:
                jpeg_bytes = self.current_jpeg

            if jpeg_bytes is not None:
                header = (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(jpeg_bytes)).encode('ascii') + b'\r\n\r\n'
                )
                yield header + jpeg_bytes + b'\r\n'
            # Wait for the next frame instead of fixed sleep
            self.frame_event.wait(timeout=0.05)
            self.frame_event.clear()


engine = AREngine()


@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "name": "AR Rubiks Cube Backend Server",
        "running": engine.running
    })


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend')


@app.route('/')
def frontend_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/css/<path:filename>')
def frontend_css(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, 'css'), filename)


@app.route('/js/<path:filename>')
def frontend_js(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, 'js'), filename)


@app.route('/api/status')
def status():
    return jsonify({
        "running": engine.running,
        "state": engine.state.name,
        "hands": engine.hand_count,
        "fps": engine.fps,
    })


@app.route('/api/reset', methods=['POST', 'GET'])
def reset():
    engine.reset_cube()
    return jsonify({"status": "reset requested"})


@app.route('/video_feed')
def video_feed():
    res = Response(
        engine.get_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
    res.headers['Access-Control-Allow-Origin'] = '*'
    return res


# FIX: Use werkzeug make_server directly so we can set SO_REUSEADDR.
# Without this, the OS holds port 5000 in TIME_WAIT after shutdown and
# a quick manual restart fails to bind — causing a startup delay or crash.
def run_flask():
    from werkzeug.serving import make_server
    srv = make_server('0.0.0.0', 5000, app, threaded=True)
    srv.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.serve_forever()


# FIX: Catch Ctrl+C via signal instead of KeyboardInterrupt.
# KeyboardInterrupt fires mid-frame and can leave the camera/MediaPipe in a
# dirty state. This handler sets running=False so the main loop exits at the
# top of its next iteration and the finally block runs cleanly every time.
def _on_signal(sig, frame):
    print("\nShutdown signal received — stopping engine cleanly...")
    engine.running = False


if __name__ == '__main__':
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print("Starting Flask Server in Background Thread...")
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    time.sleep(0.5)

    print("Starting AR Engine on Main Thread...")
    print("\n  AR Rubik's Cube running at http://localhost:5000\n")
    engine.run_main_loop()