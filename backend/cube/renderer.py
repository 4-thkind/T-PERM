"""
cube/renderer.py
Renders the Rubik's cube via PyOpenGL into an offscreen framebuffer, then
composites the cube's bounding box onto the OpenCV camera frame.
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from typing import Optional

from cube.rubiks import FACE_COLORS_GL

# ── Geometry Generation ───────────────────────────────────────────────────────

# Generate 26 physical sub-cubes
_BLOCK_CENTERS = []
for x in [-1.0, 0.0, 1.0]:
    for y in [-1.0, 0.0, 1.0]:
        for z in [-1.0, 0.0, 1.0]:
            if x == 0 and y == 0 and z == 0: continue
            _BLOCK_CENTERS.append((x, y, z))

# Sticker grid geometry, one entry per face: outward normal, and the grid's
# "right" and "down" directions as seen from OUTSIDE that face.
#
# These must match the sticker index convention in rubiks.py exactly: index 0 is
# the top-left sticker viewed from outside, running row-major. U is oriented with
# its row 0 against the B face and row 2 against F; D is the mirror, row 0 against
# F. Getting U, R or B mirrored here does not show up on a solved cube - every
# sticker on a face is the same colour - but it silently corrupts every turn,
# because the renderer then moves stickers somewhere the permutation in rubiks.py
# did not put them. test_slice_moves.py pins all six faces down.
_STICKER_SIZE = 0.85
_FACE_PLANE   = 1.51      # distance from cube centre to the sticker plane

_V = lambda *a: np.array(a, dtype=float)
_FACE_AXES = {
    'F': (_V(0, 0, 1),  _V(1, 0, 0),  _V(0, -1, 0)),
    'B': (_V(0, 0, -1), _V(-1, 0, 0), _V(0, -1, 0)),
    'R': (_V(1, 0, 0),  _V(0, 0, -1), _V(0, -1, 0)),
    'L': (_V(-1, 0, 0), _V(0, 0, 1),  _V(0, -1, 0)),
    'U': (_V(0, 1, 0),  _V(1, 0, 0),  _V(0, 0, 1)),
    'D': (_V(0, -1, 0), _V(1, 0, 0),  _V(0, 0, -1)),
}


def _sticker_quads_for_face(face: str):
    normal, right, down = _FACE_AXES[face]
    half = _STICKER_SIZE / 2.0
    quads = []
    for row in range(3):
        for col in range(3):
            centre = normal * _FACE_PLANE + right * (col - 1) + down * (row - 1)
            quads.append([
                tuple(centre - right * half - down * half),
                tuple(centre + right * half - down * half),
                tuple(centre + right * half + down * half),
                tuple(centre - right * half + down * half),
            ])
    return quads

_STICKER_QUADS = {face: _sticker_quads_for_face(face) for face in 'UDFBLR'}

# Map each of the 54 stickers to its parent block (x,y,z)
_STICKER_TO_BLOCK = {}
for face, quads in _STICKER_QUADS.items():
    _STICKER_TO_BLOCK[face] = []
    for q in quads:
        # Calculate center of quad
        cx = sum(v[0] for v in q) / 4.0
        cy = sum(v[1] for v in q) / 4.0
        cz = sum(v[2] for v in q) / 4.0
        bx = round(cx)
        by = round(cy)
        bz = round(cz)
        _STICKER_TO_BLOCK[face].append((bx, by, bz))

def _draw_subcube_body(size=0.96):
    """Draws a black box centered at origin."""
    hs = size / 2.0
    glBegin(GL_QUADS)
    # F
    glNormal3f(0, 0, 1); glVertex3f(-hs, -hs, hs); glVertex3f(hs, -hs, hs); glVertex3f(hs, hs, hs); glVertex3f(-hs, hs, hs)
    # B
    glNormal3f(0, 0, -1); glVertex3f(hs, -hs, -hs); glVertex3f(-hs, -hs, -hs); glVertex3f(-hs, hs, -hs); glVertex3f(hs, hs, -hs)
    # U
    glNormal3f(0, 1, 0); glVertex3f(-hs, hs, hs); glVertex3f(hs, hs, hs); glVertex3f(hs, hs, -hs); glVertex3f(-hs, hs, -hs)
    # D
    glNormal3f(0, -1, 0); glVertex3f(-hs, -hs, -hs); glVertex3f(hs, -hs, -hs); glVertex3f(hs, -hs, hs); glVertex3f(-hs, -hs, hs)
    # R
    glNormal3f(1, 0, 0); glVertex3f(hs, -hs, hs); glVertex3f(hs, -hs, -hs); glVertex3f(hs, hs, -hs); glVertex3f(hs, hs, hs)
    # L
    glNormal3f(-1, 0, 0); glVertex3f(-hs, -hs, -hs); glVertex3f(-hs, -hs, hs); glVertex3f(-hs, hs, hs); glVertex3f(-hs, hs, -hs)
    glEnd()

# Corners of a model-space box containing the cube in ANY slice rotation. The
# furthest geometry is a pointer at (1.6, 1.6, 1.6), norm 2.77; rotating a layer
# about an axis preserves distance from the origin, so half-extent 2.8 bounds it.
_BOUND_CORNERS = [(x, y, z) for x in (-2.8, 2.8)
                            for y in (-2.8, 2.8)
                            for z in (-2.8, 2.8)]

_POINTERS_3D = []
for _x in [-1.6, 0.0, 1.6]:
    for _y in [-1.6, 0.0, 1.6]:
        for _z in [-1.6, 0.0, 1.6]:
            if _x == 0 and _y == 0 and _z == 0: continue
            _POINTERS_3D.append((_x, _y, _z))

def _is_in_layer(face_key, bx, by, bz):
    """Check if a block/pointer belongs to the rotating layer."""
    return (
        (face_key == 'U' and by > 0.5) or (face_key == 'D' and by < -0.5) or
        (face_key == 'R' and bx > 0.5) or (face_key == 'L' and bx < -0.5) or
        (face_key == 'F' and bz > 0.5) or (face_key == 'B' and bz < -0.5) or
        (face_key == 'M' and abs(bx) < 0.5) or  # middle column
        (face_key == 'E' and abs(by) < 0.5)      # middle row
    )

_FACE_NORMALS = {
    'U': (0, 1, 0), 'D': (0, -1, 0), 'F': (0, 0, 1),
    'B': (0, 0, -1), 'R': (1, 0, 0), 'L': (-1, 0, 0),
}

# The cube move equal to ONE +90 step of this renderer's layer rotation.
#
# A layer spins about a world axis, but move names are face-local: the same
# world rotation about +Y is U for the top layer and D' for the bottom, because
# D is defined clockwise viewed from below. So +90 means clockwise only for the
# faces on the positive side of their axis (U, R, F); the rest take the prime.
# Verified exhaustively in test_slice_moves.py.
LAYER_TURN_MOVE = {
    'U': 'U',  'R': 'R',  'F': 'F',
    'D': "D'", 'L': "L'", 'B': "B'", 'M': "M'", 'E': "E'",
}


def _get_rotation_axis(face_key):
    """Return the GL rotation axis for a face."""
    if face_key in ('R', 'L', 'M'):
        return (1, 0, 0)
    elif face_key in ('U', 'D', 'E'):
        return (0, 1, 0)
    elif face_key in ('F', 'B'):
        return (0, 0, 1)
    return (0, 1, 0)

class CubeRenderer:
    def __init__(self, frame_w: int, frame_h: int):
        self.w = frame_w
        self.h = frame_h
        self._fbo = None
        self._tex = None
        self._depth_rb = None
        self._ready = False
        self._body_list = None      # display list: one cubie body
        self._sticker_base = None   # 54 consecutive display lists, one per sticker

    def init_gl(self):
        self._fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)

        self._tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.w, self.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._tex, 0)

        self._depth_rb = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self._depth_rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self.w, self.h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._depth_rb)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status == GL_FRAMEBUFFER_COMPLETE:
            self._ready = True

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [2.0, 4.0, 3.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.9, 0.9, 0.9, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.35, 0.35, 0.35, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._build_display_lists()

    def _build_display_lists(self):
        """Bake the static geometry into display lists.

        Drawn immediately, one frame is ~1210 individual PyOpenGL calls (26 cubie
        bodies at ~32 each, 54 stickers at ~7). Every one acquires the GIL that
        the MediaPipe result callback also needs. The geometry never changes -
        only its colour and the slice transform do - so compile it once and
        replay it with ~294 calls.
        """
        self._body_list = glGenLists(1)
        glNewList(self._body_list, GL_COMPILE)
        _draw_subcube_body(0.96)
        glEndList()

        self._sticker_base = glGenLists(54)
        for i, (face, quad) in enumerate(
            (f, q) for f in 'UDFBLR' for q in _STICKER_QUADS[f]
        ):
            glNewList(self._sticker_base + i, GL_COMPILE)
            glBegin(GL_QUADS)
            glNormal3f(*_FACE_NORMALS[face])
            for v in quad:
                glVertex3f(*v)
            glEnd()
            glEndList()

    # Sticker i of face F lives at _sticker_base + _STICKER_LIST_OFFSET[F] + i
    _STICKER_LIST_OFFSET = {face: n * 9 for n, face in enumerate('UDFBLR')}

    def render(self, cv_frame: np.ndarray, cube_state: dict, cube_pos_px: np.ndarray,
               cube_rotation_q: np.ndarray, cube_scale: float = 1.0,
               face_rotating: Optional[tuple] = None,
               highlighted_pointer: Optional[tuple] = None) -> tuple[np.ndarray, dict]:
        if not self._ready:
            return cv_frame, {}

        h, w = cv_frame.shape[:2]
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glViewport(0, 0, w, h)
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h, 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        cx_px, cy_px = cube_pos_px
        z_dist = 5.0
        fov_y_rad = np.radians(45.0)
        plane_h = 2.0 * z_dist * np.tan(fov_y_rad / 2.0)
        plane_w = plane_h * (w / h)
        cx_ndc = (cx_px / w) * 2.0 - 1.0
        cy_ndc = 1.0 - (cy_px / h) * 2.0
        tx = cx_ndc * (plane_w / 2.0)
        ty = cy_ndc * (plane_h / 2.0)
        tz = -z_dist

        glTranslatef(tx, ty, tz)
        
        base_scale = 0.35 * cube_scale
        glScalef(base_scale, base_scale, base_scale)

        def quat_to_matrix(q):
            w, x, y, z = q
            return np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w,     0],
                [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,     0],
                [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y, 0],
                [0,                 0,                 0,                 1]
            ], dtype=np.float32).T

        mat = quat_to_matrix(cube_rotation_q)
        glMultMatrixf(mat)

        rot_face, rot_angle = face_rotating if face_rotating else (None, 0.0)
        rot_axis = _get_rotation_axis(rot_face) if rot_face else None

        # Draw the 26 blocks
        glColor4f(0.05, 0.05, 0.05, 1.0)
        for block in _BLOCK_CENTERS:
            glPushMatrix()
            if rot_face and _is_in_layer(rot_face, *block):
                glRotatef(-rot_angle, *rot_axis)
            glTranslatef(*block)
            glCallList(self._body_list)
            glPopMatrix()

        # Draw Stickers
        for face_key, quads in _STICKER_QUADS.items():
            colors = cube_state[face_key]
            block_map = _STICKER_TO_BLOCK[face_key]
            list_base = self._sticker_base + self._STICKER_LIST_OFFSET[face_key]
            for i in range(len(quads)):
                glPushMatrix()
                if rot_face and _is_in_layer(rot_face, *block_map[i]):
                    glRotatef(-rot_angle, *rot_axis)
                glColor4f(*FACE_COLORS_GL[colors[i]], 1.0)
                glCallList(list_base + i)
                glPopMatrix()

        # Draw and project pointers
        pointers_2d = {}
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        for p3d in _POINTERS_3D:
            # Highlight the grabbed pointer in red, others white
            if highlighted_pointer and p3d == highlighted_pointer:
                glColor4f(1.0, 0.15, 0.15, 1.0)  # bright red
                glPointSize(12.0)
            else:
                glColor4f(1.0, 1.0, 1.0, 0.9)
                glPointSize(6.0)
            # We must apply slice rotation to pointers too!
            glPushMatrix()
            if face_rotating:
                rot_face, angle_deg = face_rotating
                bx, by, bz = p3d
                if _is_in_layer(rot_face, bx, by, bz):
                    ax = _get_rotation_axis(rot_face)
                    glRotatef(-angle_deg, *ax)
            glBegin(GL_POINTS)
            glVertex3f(*p3d)
            glEnd()
            glPopMatrix()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        for p3d in _POINTERS_3D:
            try:
                # Need to manually project with slice rotation applied to get accurate 2D pointers
                # But it's easier to just use the un-rotated pointer for the interaction hit-test
                # Since when you drag, the slice rotates, but your finger stays near the pointer.
                win_x, win_y, win_z = gluProject(p3d[0], p3d[1], p3d[2], modelview, projection, viewport)
                if 0 <= win_z <= 1:
                    cv_y = h - win_y
                    pointers_2d[p3d] = (win_x, cv_y)
            except Exception:
                pass

        # Read back only the cube's bounding box, not the whole frame. A full 720p
        # RGBA readback is 3.7 MB across the bus every tick and dominates the
        # frame; the cube usually covers a few percent of the screen.
        xs, ys = [], []
        for c in _BOUND_CORNERS:
            try:
                wx, wy, _ = gluProject(c[0], c[1], c[2], modelview, projection, viewport)
            except Exception:
                xs = []
                break
            xs.append(wx)
            ys.append(wy)

        if xs:
            x0 = max(0, int(np.floor(min(xs))))
            x1 = min(w, int(np.ceil(max(xs))) + 1)
            y0 = max(0, int(np.floor(min(ys))))      # GL coords, origin bottom-left
            y1 = min(h, int(np.ceil(max(ys))) + 1)
        else:                                        # projection failed - be safe
            x0, y0, x1, y1 = 0, 0, w, h

        if x1 <= x0 or y1 <= y0:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            return cv_frame, pointers_2d             # cube entirely off-screen

        bw, bh = x1 - x0, y1 - y0
        pixels = glReadPixels(x0, y0, bw, bh, GL_BGRA, GL_UNSIGNED_BYTE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # GL's origin is bottom-left, OpenCV's top-left. ::-1 is a reverse-strided
        # view, so it flips without copying like cv2.flip did.
        gl_img = np.frombuffer(pixels, dtype=np.uint8).reshape(bh, bw, 4)[::-1]
        roi = cv_frame[h - y1:h - y0, x0:x1]

        # `where=` broadcasts the mask over the channel axis, so there is no
        # 3-channel temporary, and no np.any() pre-scan of the whole frame.
        np.copyto(roi, gl_img[:, :, :3], where=gl_img[:, :, 3:4] > 0)

        return cv_frame, pointers_2d

    def cleanup(self):
        if self._body_list:
            glDeleteLists(self._body_list, 1)
            self._body_list = None
        if self._sticker_base:
            glDeleteLists(self._sticker_base, 54)
            self._sticker_base = None
        if self._fbo:
            glDeleteFramebuffers(1, [self._fbo])
            self._fbo = None
        if self._tex:
            glDeleteTextures(1, [self._tex])
            self._tex = None
        if self._depth_rb:
            glDeleteRenderbuffers(1, [self._depth_rb])
            self._depth_rb = None
