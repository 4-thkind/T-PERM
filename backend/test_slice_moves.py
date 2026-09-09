"""
Pins the renderer's layer animation to the permutation engine. Run it:
    python test_slice_moves.py

This is the guard for a bug that is invisible on a solved cube. Every sticker on
a face is the same colour there, so a mirrored sticker grid or a backwards turn
direction looks perfectly fine until you actually turn a layer on a scrambled
cube - and then colours land in the wrong places.

The check needs no OpenGL context: it rotates each sticker quad the way the
renderer does, sees which slot it lands in, and compares that permutation
against apply_move().
"""

import numpy as np
from scipy.spatial.transform import Rotation

from cube.renderer import (
    LAYER_TURN_MOVE, _STICKER_QUADS, _STICKER_TO_BLOCK,
    _get_rotation_axis, _is_in_layer,
)
from cube.rubiks import FACES, MOVES, apply_move

LAYER_KEYS = ('U', 'D', 'L', 'R', 'F', 'B', 'M', 'E')

# centre of every sticker quad, keyed by (face, index)
CENTRE = {(f, i): np.array(q, float).mean(axis=0)
          for f, quads in _STICKER_QUADS.items() for i, q in enumerate(quads)}

# a state whose every sticker is uniquely identifiable
LABELLED = {f: [(f, i) for i in range(9)] for f in FACES}


def visual_permutation(layer_key, angle_deg):
    """State after the renderer's glRotatef(-angle_deg, axis) on this layer."""
    axis = np.array(_get_rotation_axis(layer_key), float)
    rot = Rotation.from_rotvec(np.radians(-angle_deg) * axis)   # renderer negates
    out = {f: [None] * 9 for f in FACES}
    for (face, i), centre in CENTRE.items():
        if _is_in_layer(layer_key, *_STICKER_TO_BLOCK[face][i]):
            moved = rot.apply(centre)
            dest = min(CENTRE, key=lambda k: np.linalg.norm(CENTRE[k] - moved))
            assert np.linalg.norm(CENTRE[dest] - moved) < 0.2, \
                f"{layer_key} {angle_deg}: {face}[{i}] landed off-grid"
        else:
            dest = (face, i)
        out[dest[0]][dest[1]] = LABELLED[face][i]
    assert all(all(slot) for slot in out.values()), \
        f"{layer_key} {angle_deg} is not a permutation - two stickers collided"
    return out


def test_every_layer_rotation_is_exactly_one_move():
    """A 90 degree animation must equal some single move - no more, no less.

    This is what catches a mirrored sticker grid: if a face's grid is flipped,
    its own 3x3 spins one way while the side stickers cycle the other, and the
    result is not any legal move at all.
    """
    singles = [m for m in MOVES if not m.endswith('2')]
    for key in LAYER_KEYS:
        for angle in (90.0, -90.0):
            perm = visual_permutation(key, angle)
            matches = [m for m in singles if apply_move(LABELLED, m) == perm]
            assert len(matches) == 1, (
                f"{key} {angle:+.0f} matched {matches or 'NO move'}; "
                "sticker grid orientation is wrong")


def test_committed_move_matches_the_animation():
    """LAYER_TURN_MOVE must be the move the +90 animation actually performs."""
    for key in LAYER_KEYS:
        expected = apply_move(LABELLED, LAYER_TURN_MOVE[key])
        assert visual_permutation(key, 90.0) == expected, (
            f"{key}: animation does not match LAYER_TURN_MOVE[{key}]"
            f" = {LAYER_TURN_MOVE[key]}")


def test_negative_drag_commits_as_three_positive_turns():
    """server.py takes turns mod 4, so -90 becomes three +90 moves."""
    for key in LAYER_KEYS:
        state = LABELLED
        for _ in range(int(round(-90.0 / 90.0)) % 4):
            state = apply_move(state, LAYER_TURN_MOVE[key])
        assert state == visual_permutation(key, -90.0), \
            f"{key}: a -90 drag would commit the wrong turn"


def test_full_turn_returns_to_start():
    for key in LAYER_KEYS:
        state = LABELLED
        for _ in range(4):
            state = apply_move(state, LAYER_TURN_MOVE[key])
        assert state == LABELLED, f"{key} x4 is not the identity"


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed.")
