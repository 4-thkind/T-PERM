"""
Self-check for the cube permutation engine. No framework — just run it:
    python test_rubiks.py

These four invariants between them catch essentially any wrong sticker index:
a bad index either loses/duplicates a colour (sticker census), breaks the
4-fold symmetry of a quarter turn (order 4), or breaks move/inverse pairing.
"""

from collections import Counter

from cube.rubiks import (
    ALL_MOVES, FACES, MOVES, apply_move, is_solved, scramble, solved_state,
)

# The 90° turns — M2/E2/U2 etc. are compositions and are covered via those.
QUARTER_TURNS = [m for m in ALL_MOVES if not m.endswith('2')]


def test_solved_state():
    s = solved_state()
    assert is_solved(s)
    assert set(s) == set(FACES)
    assert all(len(v) == 9 for v in s.values())


def test_sticker_census_is_preserved():
    """Any sequence of moves permutes stickers — it never creates or destroys one."""
    s, seq = scramble(solved_state(), n=50)
    census = Counter(c for face in s.values() for c in face)
    assert census == Counter({f: 9 for f in FACES}), f"{seq} broke the census: {census}"


def test_quarter_turn_has_order_four():
    for move in QUARTER_TURNS:
        s = solved_state()
        for _ in range(4):
            s = apply_move(s, move)
        assert is_solved(s), f"{move} x4 did not return to solved"


def test_move_then_inverse_is_identity():
    for move in QUARTER_TURNS:
        inverse = move[:-1] if move.endswith("'") else move + "'"
        assert inverse in MOVES, f"no inverse defined for {move}"
        assert is_solved(apply_move(apply_move(solved_state(), move), inverse)), \
            f"{move} then {inverse} did not cancel"


def test_double_turn_equals_two_quarter_turns():
    for move in (m for m in ALL_MOVES if m.endswith('2')):
        base = move[0]
        assert apply_move(solved_state(), move) == \
            apply_move(apply_move(solved_state(), base), base), f"{move} != {base} {base}"


def test_sexy_move_has_order_six():
    """(R U R' U') repeated 6x is the identity — the classic engine sanity check."""
    s = solved_state()
    for _ in range(6):
        for move in ("R", "U", "R'", "U'"):
            s = apply_move(s, move)
    assert is_solved(s), "sexy move x6 did not return to solved"


def test_apply_move_does_not_mutate_input():
    s = solved_state()
    s['U'][0] = 'X'          # marker so we'd notice an in-place edit
    before = {f: list(v) for f, v in s.items()}
    apply_move(s, 'R')
    assert s == before, "apply_move mutated its input"


def test_scramble_actually_scrambles():
    s, seq = scramble(solved_state(), n=20)
    assert len(seq) == 20
    assert not is_solved(s), "20 random moves left the cube solved"
    assert all(a[0] != b[0] for a, b in zip(seq, seq[1:])), \
        "scramble produced consecutive moves on the same face"


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed.")
