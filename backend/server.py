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
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

import pyglet
from pyglet.gl import *

from hand_tracker import HandTracker, WebcamStream
from gesture_engine import (
    AbsentDetector,
    get_spawn_distance, is_cube_spawn_ready,
    midpoint, is_pinch, is_fist, is_open_palm,
    is_palm_facing_down, is_palm_facing_up
)
from cube.rubiks import solved_state, scramble, is_solved, apply_move
from cube.renderer import CubeRenderer, LAYER_TURN_MOVE
from utils.smoothing import EMA, QuatEMA
from utils.transforms import (quat_multiply, quat_conjugate, cube_axes_on_screen,
                              snap_to_nearest_90, hand_orientation_quat)
import hud


# Anchor every asset path to this file, never to the working directory. Under
# `npx t-perm` the process is spawned from wherever the user happens to be.
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BACKEND_DIR, 'hand_landmarker.task')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Tuning knobs ─────────────────────────────────────────────────────────────
# Render/stream ceiling, and the main tuning knob for this app.
#
# The render loop and MediaPipe compete for CPU and for the GIL. Raise for a
# smoother picture; lower to hand time back to detection. Tracking lag equals one
# detection period, so TRACK FPS is what determines how well the skeleton sticks
# to your hand. Adjustable at runtime: GET/POST /api/config?target_fps=30
TARGET_FPS = 45
# JPEG quality for the MJPEG stream. 65 encodes roughly 2x faster than 80.
JPEG_QUALITY = 65
# Port to serve on. The CLI passes T_PERM_PORT so `npx t-perm` and the browser
# it opens always agree, even if the default is already taken.
PORT = int(os.environ.get('T_PERM_PORT', 5000))


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
        self.det_fps = 0
        self.lag_ms = 0
        self.target_fps = TARGET_FPS
        self.num_hands = 2        # which detector is live right now
        self._diag_frame = None   # recent raw frame, for /api/diagnose
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
        _det_times     = []        # callback arrival times, for detection-FPS
        _sent_at       = {}        # timestamp_ms -> perf_counter when submitted
        _lags          = []        # submit -> landmarks-in-hand, milliseconds

        def _on_detection(result, image, timestamp_ms):
            """Called by MediaPipe on its own thread whenever results are ready."""
            arrived = time.perf_counter()
            try:
                hands = tracker.extract_hands(result, frame_w, frame_h)
            except Exception:
                hands = []
            with _cb_lock:
                _latest_hands[0]  = hands
                _latest_result[0] = result
                _latest_gen[0]   += 1
                _det_times.append(arrived)
                if len(_det_times) > 30:
                    del _det_times[0]
                # How stale the skeleton we're about to draw actually is. Frames
                # the flow limiter dropped never call back, so their entries get
                # pruned by age rather than popped.
                started = _sent_at.pop(timestamp_ms, None)
                if started is not None:
                    _lags.append((arrived - started) * 1000.0)
                    if len(_lags) > 30:
                        del _lags[0]

        # Two detectors, differing only in num_hands, both live.
        #
        # Measured on a real frame with ONE hand up: num_hands=2 takes 143ms per
        # detection, num_hands=1 takes 66ms. The gap is the palm detector, which
        # MediaPipe re-runs every frame while short of the hand count it was
        # asked for. Input resolution changes nothing (640x360 vs 213x120 land
        # within 2ms) - only the hand count does.
        #
        # Building a detector costs a few hundred ms of model load, so make both
        # up front and switch, rather than rebuilding on transitions.
        tracker = HandTracker(MODEL_PATH, result_callback=_on_detection,
                              num_hands=2)
        tracker_solo = HandTracker(MODEL_PATH, result_callback=_on_detection,
                                   num_hands=1)
        # Which detector to run is decided by what is actually in frame, not by
        # which state we are in.
        #
        # num_hands=2 is only slow when it is SHORT of hands: MediaPipe re-runs
        # the palm detector every frame hunting for the one it cannot find. Once
        # both hands are visible it has nothing left to search for and costs
        # about the same as num_hands=1. So the expensive case is exactly "asked
        # for two, can see one".
        #
        # Rule: run the cheap solo detector while one hand is up, but re-probe
        # with the two-hand detector every PROBE_EVERY frames so a second hand
        # entering the frame is still noticed within a few hundred ms. Once two
        # hands are seen, stay on the two-hand detector until one leaves.
        # Probe every 6th frame: a second hand entering the frame is picked up
        # within ~20-70ms, while only ~18% of lone-hand frames pay the slower
        # two-hand detector. Probing every 12th halves that cost but lets the
        # gap stretch to ~156ms, which is noticeable on the spawn gesture.
        PROBE_EVERY = 6
        SOLO_GRACE = 8            # detections of 0-1 hands before dropping to solo
        two_hand_mode = True      # start wide so the spawn gesture is available
        solo_streak = 0
        probe_tick = 0
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
        # Offset between the driving hand's orientation and the cube's, captured
        # when that hand takes control. None means "re-acquire on next frame".
        grab_offset_q = None
        grab_hand_label = None
        snap_latched = False      # palm-flip snap fires once per flip, not per frame
        absent_detector = AbsentDetector(threshold_frames=48)
        completion_start_time = 0.0
        completion_solved = False
        banner_alpha = 0.0
        pointers_2d = {}

        confetti_active = False
        frame_times = []
        frames_seen = 0
        self.running = True
        print(">>> AR Engine Ready and Processing Frames! <<<")

        try:
            while self.running:
                # perf_counter, not time(): the pacing and FPS maths below
                # subtract from this, and mixing the two clocks' epochs is nonsense.
                t_start = time.perf_counter()
                with self.lock:
                    if self.reset_requested:
                        cube_state = solved_state()
                        cube_state, _ = scramble(cube_state, n=20)
                        cube_rotation = np.array([0.0, 0.0, 0.0, 1.0])
                        rot_ema.reset()
                        grab_offset_q = None
                        self.reset_requested = False

                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                frame_h, frame_w = frame.shape[:2]

                # Pick the detector from what was in frame last tick (updated
                # further down, once this frame's results have been read).
                # Timestamps come from one monotonic clock, so each detector
                # still sees a strictly increasing sequence even though it only
                # receives some of the frames.
                probe_tick += 1
                probing = (not two_hand_mode) and (probe_tick % PROBE_EVERY == 0)
                active = tracker if (two_hand_mode or probing) else tracker_solo
                self.num_hands = active.num_hands

                # Submit every frame and let MediaPipe's flow limiter drop the
                # excess. Do NOT gate submissions on a detection being in flight:
                # measured, that roughly HALVES the detection rate (13.2 -> 7.6
                # fps), because it idles the detector until the next iteration
                # instead of keeping its pipeline fed.
                #
                # The 1/3 downscale shrinks the mp.Image copy. It does NOT speed
                # up inference - MediaPipe rescales to the model's input size
                # regardless, and 640x360 vs 213x120 measured within 2ms.
                submit_t = time.perf_counter()
                timestamp_ms = int((submit_t - self._perf_start) * 1000)
                small = cv2.resize(frame, (frame_w // 3, frame_h // 3))
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                with _cb_lock:
                    _sent_at[timestamp_ms] = submit_t
                    if len(_sent_at) > 120:      # dropped frames never call back
                        for k in sorted(_sent_at)[:60]:
                            del _sent_at[k]
                try:
                    active.detect_async(rgb_small, timestamp_ms)
                except ValueError:
                    # timestamp not monotonically increasing — skip this frame
                    with _cb_lock:
                        _sent_at.pop(timestamp_ms, None)

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

                # Latch onto two-hand mode the moment a second hand appears, and
                # only fall back after SOLO_GRACE consecutive lone-hand results,
                # so one missed detection cannot drop a two-hand gesture.
                if has_new_detection:
                    if len(hands) >= 2:
                        solo_streak = 0
                        two_hand_mode = True
                    else:
                        solo_streak += 1
                        if solo_streak >= SOLO_GRACE:
                            two_hand_mode = False

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
                        if lock_hand or not hands:
                            # locked or hand gone: drop the offset so the cube is
                            # picked up from its current pose next time, instead
                            # of snapping back to where the hand left off
                            grab_offset_q = None

                        if not lock_hand:
                            # Drive the cube from where the hand actually points.
                            #
                            # This used to snap to two hard-coded quaternions the
                            # moment the palm tilted past a threshold, and
                            # otherwise integrate frame-to-frame deltas. Both
                            # fought the hand: the snaps threw the measured angle
                            # away and jumped to a canned pose, and the deltas
                            # accumulated their own error until the cube no
                            # longer corresponded to the hand at all.
                            #
                            # Instead: read the hand's ABSOLUTE orientation, and
                            # remember the offset between it and the cube at the
                            # moment control was taken. The cube is then always
                            # exactly that offset from the live hand - it tracks
                            # 1:1, holds still when the hand holds still, and
                            # cannot drift.
                            drive_hand = next(
                                (h for h in hands if not is_pinch(h) and not is_fist(h)),
                                None)

                            # Deliberate palm flip still snaps to the Top/Bottom
                            # face, but as a RE-GRAB rather than a hard pose: the
                            # cube is set to the face you asked for, then the
                            # offset is re-taken so continuous tracking carries
                            # on from there instead of sticking at a fixed pose.
                            if drive_hand is not None and not snap_latched:
                                if is_palm_facing_down(drive_hand):
                                    cube_rotation = np.array([0.7071, 0.0, 0.0, 0.7071])
                                    grab_offset_q = None
                                    snap_latched = True
                                elif is_palm_facing_up(drive_hand):
                                    cube_rotation = np.array([-0.7071, 0.0, 0.0, 0.7071])
                                    grab_offset_q = None
                                    snap_latched = True
                            if drive_hand is not None and snap_latched:
                                # only re-arm once the palm leaves the snap zone,
                                # so holding it there does not freeze the cube
                                if not (is_palm_facing_down(drive_hand)
                                        or is_palm_facing_up(drive_hand)):
                                    snap_latched = False

                            if drive_hand is not None:
                                hand_q = hand_orientation_quat(
                                    drive_hand.palm_normal, drive_hand.finger_direction)
                                if grab_offset_q is None or grab_hand_label != drive_hand.label:
                                    # Take control without teleporting the cube:
                                    # offset = current cube orientation relative
                                    # to the hand right now.
                                    grab_offset_q = quat_multiply(
                                        cube_rotation, quat_conjugate(hand_q))
                                    grab_hand_label = drive_hand.label
                                cube_rotation = quat_multiply(grab_offset_q, hand_q)
                                norm = np.linalg.norm(cube_rotation)
                                if norm > 1e-6:
                                    cube_rotation /= norm
                            else:
                                # pinching or fisted: that hand is doing something
                                # else, so re-acquire the offset when it returns
                                grab_offset_q = None

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
                            # LAYER_TURN_MOVE is the move equal to ONE +90 step of
                            # the renderer's rotation for this layer, so the state
                            # always ends up matching what was just animated.
                            # turns is taken mod 4, so a -90 drag becomes three
                            # +90 moves - same result, no sign handling needed.
                            turns = int(round(snap_target_angle / 90.0)) % 4
                            move = LAYER_TURN_MOVE[face_rot_face]
                            for _ in range(turns):
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

                ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    with self.lock:
                        self.current_jpeg = jpeg.tobytes()
                    self.frame_event.set()

                # Frame pacing. Always yield at least a sliver: if the loop can't
                # hit the target the sleep would otherwise never fire, and the
                # main thread would spin without ever handing the GIL to the
                # MediaPipe callback that has to deliver the landmarks.
                elapsed = time.perf_counter() - t_start
                target = 1.0 / max(self.target_fps, 1)
                time.sleep(max(target - elapsed, 0.001))

                # Render FPS - measured across the FULL period including the
                # sleep. Timing only the work before it reports how fast a frame
                # could have been built, not how many actually ship.
                frame_times.append(time.perf_counter() - t_start)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                avg = sum(frame_times) / len(frame_times)
                self.fps = int(1.0 / avg) if avg > 0 else 0

                # Detection FPS and skeleton staleness - the numbers that decide
                # how well tracking sticks to the hand.
                with _cb_lock:
                    span = _det_times[-1] - _det_times[0] if len(_det_times) > 1 else 0.0
                    n_det = len(_det_times)
                    lag = sorted(_lags)[len(_lags) // 2] if _lags else 0.0
                self.det_fps = int((n_det - 1) / span) if span > 0 else 0
                self.lag_ms = int(lag)

        finally:
            self.running = False
            renderer.cleanup()
            tracker.close()
            tracker_solo.close()
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


FRONTEND_DIR = os.path.join(BACKEND_DIR, '..', 'frontend')


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
        "det_fps": engine.det_fps,
        "lag_ms": engine.lag_ms,
        "target_fps": engine.target_fps,
        "num_hands": engine.num_hands,
    })


@app.route('/api/diagnose')
def diagnose():
    """Benchmark MediaPipe on a real frame from THIS camera, with YOUR hand in it.

    Visit with a hand held up; takes ~10s and eats CPU while it runs.

    hands_found matters: a config that finds no hand is meaningless, because
    MediaPipe returns early without running the landmark model at all and looks
    about 10x faster than it really is.
    """
    import statistics
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    frame = engine._diag_frame
    if frame is None:
        return jsonify({"error": "no frame captured yet - is the engine running?"}), 503

    h, w = frame.shape[:2]

    def bench(num_hands, divisor, reps=12):
        small = cv2.resize(frame, (w // divisor, h // divisor))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        det = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=mp_vision.RunningMode.VIDEO, num_hands=num_hands,
                min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4))
        ts = 0
        for _ in range(4):
            ts += 33
            det.detect_for_video(img, ts)
        times, found = [], 0
        for _ in range(reps):
            ts += 33
            t0 = time.perf_counter()
            r = det.detect_for_video(img, ts)
            times.append((time.perf_counter() - t0) * 1000.0)
            found += len(r.hand_landmarks)
        det.close()
        return {
            "num_hands": num_hands,
            "input": "%dx%d" % (small.shape[1], small.shape[0]),
            "ms_per_detect": round(statistics.median(times), 1),
            "detects_per_sec": round(1000.0 / statistics.median(times), 1),
            "hands_found": round(found / reps, 2),
        }

    runs = [bench(n, d) for n, d in ((2, 3), (1, 3), (2, 2), (2, 6), (1, 6))]
    usable = [r for r in runs if r["hands_found"] > 0]
    best = min(usable, key=lambda r: r["ms_per_detect"]) if usable else None
    current = runs[0]
    return jsonify({
        "note": "ignore any row with hands_found = 0; it never ran the landmark model",
        "source_frame": "%dx%d" % (w, h),
        "current_setting": current,
        "fastest_usable": best,
        "speedup_available": (round(current["ms_per_detect"] / best["ms_per_detect"], 2)
                              if best else None),
        "results": runs,
    })


@app.route('/api/config', methods=['GET', 'POST'])
def config():
    """Live-tune the render cap without a restart: /api/config?target_fps=30

    Render rate and tracking rate trade against each other - they share the CPU
    and the GIL. Sweep this while watching TRACK FPS to find the knee.
    """
    raw = request.args.get('target_fps')
    if raw is not None:
        try:
            engine.target_fps = max(1, min(240, int(raw)))
        except ValueError:
            return jsonify({"error": "target_fps must be an integer"}), 400
    return jsonify({"target_fps": engine.target_fps})


@app.route('/api/shutdown', methods=['POST'])
def shutdown():
    """Stop the engine and exit the process, so the terminal returns to a prompt.

    POST only: a GET here would let a stray browser prefetch or a page reload
    kill the app. Clearing engine.running lets the main loop fall out of its
    `while` and run its finally block, which releases the camera and GL context
    cleanly. The exit itself is deferred to a daemon thread so this request can
    still return 200 before the interpreter goes away.
    """
    engine.running = False

    def _exit_soon():
        time.sleep(0.6)      # let the main loop finish its cleanup first
        os._exit(0)          # hard exit: Flask runs on a daemon thread

    threading.Thread(target=_exit_soon, daemon=True).start()
    return jsonify({"status": "shutting down"})


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
# Without this, the OS holds the port in TIME_WAIT after shutdown and
# a quick manual restart fails to bind — causing a startup delay or crash.
def run_flask():
    from werkzeug.serving import make_server
    srv = make_server('0.0.0.0', PORT, app, threaded=True)
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

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH}")
        raise SystemExit(1)

    print("Starting Flask Server in Background Thread...")
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    time.sleep(0.5)

    print("Starting AR Engine on Main Thread...")
    print(f"\n  T-PERM running at http://localhost:{PORT}\n")
    engine.run_main_loop()