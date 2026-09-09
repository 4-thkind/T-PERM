import numpy as np
from scipy.spatial.transform import Rotation


def compute_palm_normal(landmarks) -> np.ndarray:
    """
    Cross-product of wrist→index_MCP and wrist→pinky_MCP vectors.
    Gives the 3-D outward-facing normal of the palm.
    landmarks: list of 21 (x,y,z) normalized tuples/objects with .x .y .z
    """
    def lm(idx):
        p = landmarks[idx]
        return np.array([p.x, p.y, p.z])

    v1 = lm(5) - lm(0)   # wrist → index MCP
    v2 = lm(17) - lm(0)  # wrist → pinky MCP
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])


def compute_finger_direction(landmarks) -> np.ndarray:
    """
    Direction from wrist to middle finger MCP.
    Combined with palm_normal this gives the full hand orientation frame.
    """
    def lm(idx):
        p = landmarks[idx]
        return np.array([p.x, p.y, p.z])

    v = lm(9) - lm(0)   # wrist → middle MCP
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0])


def delta_rotation(prev_normal: np.ndarray, curr_normal: np.ndarray,
                   prev_finger: np.ndarray = None, curr_finger: np.ndarray = None) -> np.ndarray:
    """
    Rotation (as quaternion [x,y,z,w]) that takes prev orientation to curr.
    If finger directions are provided, uses both vectors for full 3D tracking.
    Otherwise falls back to single-vector alignment (yaw only).
    """
    try:
        if prev_finger is not None and curr_finger is not None:
            # Full 3D: align two vectors (palm normal + finger direction)
            R, _ = Rotation.align_vectors(
                [curr_normal, curr_finger],
                [prev_normal, prev_finger],
                weights=[1.0, 0.7]  # normal is primary, finger is secondary
            )
        else:
            R, _ = Rotation.align_vectors([curr_normal], [prev_normal])
        return R.as_quat()  # [x, y, z, w]
    except Exception:
        return np.array([0.0, 0.0, 0.0, 1.0])


def hand_orientation_quat(palm_normal: np.ndarray,
                          finger_direction: np.ndarray) -> np.ndarray:
    """Absolute orientation of a hand as a quaternion [x, y, z, w].

    Builds an orthonormal frame from the two hand vectors and returns the
    rotation that carries the reference frame onto it. Absolute, not a
    frame-to-frame delta: the cube can then be driven straight from where the
    hand actually points, instead of integrating deltas whose small errors
    accumulate until the cube no longer corresponds to the hand at all.
    """
    forward = np.asarray(palm_normal, dtype=float)
    n = np.linalg.norm(forward)
    if n < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])
    forward = forward / n

    # Gram-Schmidt the finger direction against the palm normal so the two axes
    # are exactly orthogonal even though the measured vectors never quite are.
    up = np.asarray(finger_direction, dtype=float)
    up = up - np.dot(up, forward) * forward
    n = np.linalg.norm(up)
    if n < 1e-6:                      # degenerate: fingers along the normal
        return np.array([0.0, 0.0, 0.0, 1.0])
    up = up / n

    right = np.cross(up, forward)
    n = np.linalg.norm(right)
    if n < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])
    right = right / n

    return Rotation.from_matrix(np.column_stack((right, up, forward))).as_quat()


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion [x, y, z, w]."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [x,y,z,w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def snap_to_nearest_90(angle_degrees: float) -> float:
    """Round to nearest multiple of 90°."""
    return round(angle_degrees / 90.0) * 90.0


def cube_axes_on_screen(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given cube rotation quaternion, return the cube's local X and Y
    axes projected to 2D screen space (as unit vectors in pixel-direction).
    
    Uses the SAME quaternion convention as the renderer: [x, y, z, w].

    Returns (screen_x, screen_y) — each is a 2D numpy array.
    """
    x, y, z, w = q
    
    # Rotation matrix columns (local axes in world space)
    # Same formula as renderer.py's quat_to_matrix
    local_x = np.array([
        1 - 2*y*y - 2*z*z,
        2*x*y + 2*z*w,
        2*x*z - 2*y*w
    ])
    local_y = np.array([
        2*x*y - 2*z*w,
        1 - 2*x*x - 2*z*z,
        2*y*z + 2*x*w
    ])
    
    # Project to screen: take x,y components, flip y for pixel coords
    sx = np.array([local_x[0], -local_x[1]])
    sy = np.array([local_y[0], -local_y[1]])
    
    # Normalize
    nx = np.linalg.norm(sx)
    ny = np.linalg.norm(sy)
    if nx > 1e-6: sx /= nx
    if ny > 1e-6: sy /= ny
    
    return sx, sy
