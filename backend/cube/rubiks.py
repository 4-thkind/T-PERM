"""
cube/rubiks.py
3×3 Rubik's cube state — 6 faces × 9 stickers, all 18 standard moves.
Pure functions: state in → state out. Immutable per move.
"""

import copy
import random

# Face colour constants (OpenGL RGB floats, also used for CV drawing)
FACE_COLORS_GL = {
    'U': (1.00, 1.00, 1.00),  # white  — top
    'D': (1.00, 1.00, 0.00),  # yellow — bottom
    'F': (1.00, 0.50, 0.00),  # orange — front
    'B': (0.90, 0.10, 0.10),  # red    — back
    'L': (0.10, 0.45, 1.00),  # blue   — left
    'R': (0.10, 0.80, 0.10),  # green  — right
}

FACES = ['U', 'D', 'F', 'B', 'L', 'R']

# Sticker indices layout (face viewed from outside, row-major):
#   0 1 2
#   3 4 5
#   6 7 8
# [4] is always the centre (fixed colour)


def solved_state() -> dict:
    """Return a fully solved cube state."""
    return {face: [face] * 9 for face in FACES}


def _rotate_face_cw(state: dict, face: str) -> dict:
    """Rotate a single face clockwise (sticker rearrangement only, no adjacents)."""
    s = state[face]
    state[face] = [s[6], s[3], s[0],
                   s[7], s[4], s[1],
                   s[8], s[5], s[2]]
    return state


def _rotate_face_ccw(state: dict, face: str) -> dict:
    s = state[face]
    state[face] = [s[2], s[5], s[8],
                   s[1], s[4], s[7],
                   s[0], s[3], s[6]]
    return state


# ── Move implementations ──────────────────────────────────────────────────────
# Each move: deepcopy → transform → return

def _apply_U(s):
    s = copy.deepcopy(s)
    s = _rotate_face_cw(s, 'U')
    t = s['F'][:3]
    s['F'][:3] = s['R'][:3]
    s['R'][:3] = s['B'][:3]
    s['B'][:3] = s['L'][:3]
    s['L'][:3] = t
    return s

def _apply_U_prime(s):
    s = copy.deepcopy(s)
    s = _rotate_face_ccw(s, 'U')
    t = s['F'][:3]
    s['F'][:3] = s['L'][:3]
    s['L'][:3] = s['B'][:3]
    s['B'][:3] = s['R'][:3]
    s['R'][:3] = t
    return s

def _apply_D(s):
    s = copy.deepcopy(s)
    s = _rotate_face_cw(s, 'D')
    t = s['F'][6:]
    s['F'][6:] = s['L'][6:]
    s['L'][6:] = s['B'][6:]
    s['B'][6:] = s['R'][6:]
    s['R'][6:] = t
    return s

def _apply_D_prime(s):
    s = copy.deepcopy(s)
    s = _rotate_face_ccw(s, 'D')
    t = s['F'][6:]
    s['F'][6:] = s['R'][6:]
    s['R'][6:] = s['B'][6:]
    s['B'][6:] = s['L'][6:]
    s['L'][6:] = t
    return s

def _col(face_arr, c):
    return [face_arr[c], face_arr[c+3], face_arr[c+6]]

def _set_col(face_arr, c, vals):
    face_arr[c], face_arr[c+3], face_arr[c+6] = vals

def _apply_R(s):
    s = copy.deepcopy(s)
    s = _rotate_face_cw(s, 'R')
    t = _col(s['F'], 2)
    _set_col(s['F'], 2, _col(s['D'], 2))
    _set_col(s['D'], 2, list(reversed(_col(s['B'], 0))))
    _set_col(s['B'], 0, list(reversed(_col(s['U'], 2))))
    _set_col(s['U'], 2, t)
    return s

def _apply_R_prime(s):
    s = copy.deepcopy(s)
    s = _rotate_face_ccw(s, 'R')
    t = _col(s['F'], 2)
    _set_col(s['F'], 2, _col(s['U'], 2))
    _set_col(s['U'], 2, list(reversed(_col(s['B'], 0))))
    _set_col(s['B'], 0, list(reversed(_col(s['D'], 2))))
    _set_col(s['D'], 2, t)
    return s

def _apply_L(s):
    s = copy.deepcopy(s)
    s = _rotate_face_cw(s, 'L')
    t = _col(s['F'], 0)
    _set_col(s['F'], 0, _col(s['U'], 0))
    _set_col(s['U'], 0, list(reversed(_col(s['B'], 2))))
    _set_col(s['B'], 2, list(reversed(_col(s['D'], 0))))
    _set_col(s['D'], 0, t)
    return s

def _apply_L_prime(s):
    s = copy.deepcopy(s)
    s = _rotate_face_ccw(s, 'L')
    t = _col(s['F'], 0)
    _set_col(s['F'], 0, _col(s['D'], 0))
    _set_col(s['D'], 0, list(reversed(_col(s['B'], 2))))
    _set_col(s['B'], 2, list(reversed(_col(s['U'], 0))))
    _set_col(s['U'], 0, t)
    return s

def _apply_F(s):
    s = copy.deepcopy(s)
    s = _rotate_face_cw(s, 'F')
    t = s['U'][6:]
    s['U'][6:] = list(reversed(_col(s['L'], 2)))
    _set_col(s['L'], 2, s['D'][:3])
    s['D'][:3] = list(reversed(_col(s['R'], 0)))
    _set_col(s['R'], 0, t)
    return s

def _apply_F_prime(s):
    s = copy.deepcopy(s)
    s = _rotate_face_ccw(s, 'F')
    t = s['U'][6:]
    s['U'][6:] = _col(s['R'], 0)
    _set_col(s['R'], 0, list(reversed(s['D'][:3])))
    s['D'][:3] = _col(s['L'], 2)
    _set_col(s['L'], 2, list(reversed(t)))
    return s

def _apply_B(s):
    s = copy.deepcopy(s)
    s = _rotate_face_cw(s, 'B')
    t = s['U'][:3]
    s['U'][:3] = _col(s['R'], 2)
    _set_col(s['R'], 2, list(reversed(s['D'][6:])))
    s['D'][6:] = _col(s['L'], 0)
    _set_col(s['L'], 0, list(reversed(t)))
    return s

def _apply_B_prime(s):
    s = copy.deepcopy(s)
    s = _rotate_face_ccw(s, 'B')
    t = s['U'][:3]
    s['U'][:3] = list(reversed(_col(s['L'], 0)))
    _set_col(s['L'], 0, s['D'][6:])
    s['D'][6:] = list(reversed(_col(s['R'], 2)))
    _set_col(s['R'], 2, t)
    return s


def _apply_M(s):
    """Middle layer between L and R (follows L direction: top→front→bottom→back)."""
    s = copy.deepcopy(s)
    t = _col(s['F'], 1)
    _set_col(s['F'], 1, _col(s['U'], 1))
    _set_col(s['U'], 1, list(reversed(_col(s['B'], 1))))
    _set_col(s['B'], 1, list(reversed(_col(s['D'], 1))))
    _set_col(s['D'], 1, t)
    return s

def _apply_M_prime(s):
    s = copy.deepcopy(s)
    t = _col(s['F'], 1)
    _set_col(s['F'], 1, _col(s['D'], 1))
    _set_col(s['D'], 1, list(reversed(_col(s['B'], 1))))
    _set_col(s['B'], 1, list(reversed(_col(s['U'], 1))))
    _set_col(s['U'], 1, t)
    return s

def _apply_E(s):
    """Middle layer between U and D (follows D direction: front→left→back→right)."""
    s = copy.deepcopy(s)
    t = s['F'][3:6]
    s['F'][3:6] = s['L'][3:6]
    s['L'][3:6] = s['B'][3:6]
    s['B'][3:6] = s['R'][3:6]
    s['R'][3:6] = t
    return s

def _apply_E_prime(s):
    s = copy.deepcopy(s)
    t = s['F'][3:6]
    s['F'][3:6] = s['R'][3:6]
    s['R'][3:6] = s['B'][3:6]
    s['B'][3:6] = s['L'][3:6]
    s['L'][3:6] = t
    return s


MOVES = {
    'U':  _apply_U,       "U'": _apply_U_prime,
    'U2': lambda s: _apply_U(_apply_U(s)),
    'D':  _apply_D,       "D'": _apply_D_prime,
    'D2': lambda s: _apply_D(_apply_D(s)),
    'R':  _apply_R,       "R'": _apply_R_prime,
    'R2': lambda s: _apply_R(_apply_R(s)),
    'L':  _apply_L,       "L'": _apply_L_prime,
    'L2': lambda s: _apply_L(_apply_L(s)),
    'F':  _apply_F,       "F'": _apply_F_prime,
    'F2': lambda s: _apply_F(_apply_F(s)),
    'B':  _apply_B,       "B'": _apply_B_prime,
    'B2': lambda s: _apply_B(_apply_B(s)),
    'M':  _apply_M,       "M'": _apply_M_prime,
    'E':  _apply_E,       "E'": _apply_E_prime,
}

ALL_MOVES = list(MOVES.keys())


def apply_move(state: dict, move: str) -> dict:
    if move not in MOVES:
        raise ValueError(f"Unknown move: {move}")
    return MOVES[move](state)


def scramble(state: dict, n: int = 20) -> tuple[dict, list[str]]:
    """Apply n random moves. Returns (new_state, move_sequence)."""
    seq = []
    prev = None
    for _ in range(n):
        choices = [m for m in ALL_MOVES if m[0] != (prev[0] if prev else None)]
        move = random.choice(choices)
        state = apply_move(state, move)
        seq.append(move)
        prev = move
    return state, seq


def is_solved(state: dict) -> bool:
    return all(len(set(stickers)) == 1 for stickers in state.values())
