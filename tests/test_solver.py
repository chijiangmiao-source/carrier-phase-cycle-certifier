"""Solver tests: known answers, brute-force cross-checks, scale limits."""
from __future__ import annotations

import random
import time

import pytest

from app.errors import DomainError
from app.solver import MAX_CANDIDATES, _transition, build_candidates, solve
from tests.brute import brute_solve

# ---------------------------------------------------------------------------
# Known-answer cases
# ---------------------------------------------------------------------------


def test_unique_nonzero_cost():
    # d1 = 1 (only value == 1 mod 5 in [1, 1]), d2 = 6 -> cost |6 - 1| = 5
    assert solve(5, 3, [0, 1, 2], 0, [1, 6], [1, 6]) == {
        "status": "unique",
        "x": [0, 1, 7],
        "cost": 5,
        "increments": [1, 6],
    }


def test_ambiguous_two_witnesses_ordered():
    # C1 = C2 = {1, 6}; optima (1, 1) and (6, 6), both cost 0
    assert solve(5, 3, [0, 1, 2], 0, [0, 0], [10, 10]) == {
        "status": "ambiguous",
        "cost": 0,
        "witnesses": [
            {"x": [0, 1, 2], "increments": [1, 1], "cost": 0},
            {"x": [0, 6, 12], "increments": [6, 6], "cost": 0},
        ],
    }


def test_count_truncated_at_two_witnesses():
    # Three optimal sequences (0,0), (3,3), (6,6): only the two
    # lexicographically smallest witnesses may be returned.
    result = solve(3, 3, [0, 0, 0], 0, [0, 0], [6, 6])
    assert result == {
        "status": "ambiguous",
        "cost": 0,
        "witnesses": [
            {"x": [0, 0, 0], "increments": [0, 0], "cost": 0},
            {"x": [0, 3, 6], "increments": [3, 3], "cost": 0},
        ],
    }


def test_negative_start_and_increments():
    # C1 = {-9, -4} (== 1 mod 5), C2 = {-6, -1} (== 4 mod 5)
    # costs: 3, 8, 2, 3 -> unique optimum (-4, -6)
    assert solve(5, 3, [0, 1, 0], -10, [-9, -9], [-1, -1]) == {
        "status": "unique",
        "x": [-10, -14, -20],
        "cost": 2,
        "increments": [-4, -6],
    }


def test_n_equals_one():
    assert solve(5, 1, [3], -7, [], []) == {
        "status": "unique", "x": [-7], "cost": 0, "increments": []}


def test_n_equals_two_cost_zero_ambiguous():
    assert solve(5, 2, [0, 1], 0, [0], [10]) == {
        "status": "ambiguous",
        "cost": 0,
        "witnesses": [
            {"x": [0, 1], "increments": [1], "cost": 0},
            {"x": [0, 6], "increments": [6], "cost": 0},
        ],
    }


def test_n_equals_two_unique():
    assert solve(5, 2, [0, 1], 0, [1], [1]) == {
        "status": "unique", "x": [0, 1], "cost": 0, "increments": [1]}


def test_impossible_reports_first_empty_epoch():
    # epoch 2 ([3,4], need == 1 mod 5) and epoch 3 ([0,0]) are both empty;
    # the earliest one must be reported.
    assert solve(5, 4, [0, 1, 2, 3], 0, [0, 3, 0], [1, 4, 0]) == {
        "status": "impossible", "first_empty_index": 2}


def test_max_modulus_and_exactly_64_candidates():
    M = 10**9
    result = solve(M, 2, [0, 123456789], 0, [0], [63 * M + 123456789])
    assert result["status"] == "ambiguous"
    assert result["cost"] == 0
    first, second = result["witnesses"]
    assert first["increments"] == [123456789]
    assert second["increments"] == [123456789 + M]


# ---------------------------------------------------------------------------
# Domain errors
# ---------------------------------------------------------------------------


def test_error_invalid_remainder():
    with pytest.raises(DomainError) as exc:
        solve(5, 1, [5], 0, [], [])
    assert exc.value.code == "INVALID_REMAINDER"
    assert exc.value.index == 0


def test_error_invalid_start():
    with pytest.raises(DomainError) as exc:
        solve(5, 1, [0], 1, [], [])
    assert exc.value.code == "INVALID_START"


def test_error_invalid_interval():
    with pytest.raises(DomainError) as exc:
        solve(5, 2, [0, 1], 0, [3], [1])
    assert exc.value.code == "INVALID_INTERVAL"
    assert exc.value.index == 1


def test_error_length_mismatch():
    with pytest.raises(DomainError) as exc:
        solve(5, 3, [0, 1], 0, [0, 0], [1, 1])
    assert exc.value.code == "LENGTH_MISMATCH"
    with pytest.raises(DomainError) as exc:
        solve(5, 2, [0, 1], 0, [], [])
    assert exc.value.code == "LENGTH_MISMATCH"


def test_error_too_many_candidates():
    M = 10**9
    with pytest.raises(DomainError) as exc:
        solve(M, 2, [0, 123456789], 0, [0], [64 * M + 123456789])
    assert exc.value.code == "TOO_MANY_CANDIDATES"
    assert exc.value.details["count"] == MAX_CANDIDATES + 1


def test_error_precedence_over_impossible():
    # Epoch 1 has no candidate; epoch 2 violates the 64-candidate limit.
    # Malformed input must be reported as an error, not as impossibility.
    with pytest.raises(DomainError) as exc:
        solve(5, 3, [0, 1, 2], 0, [2, 0], [3, 1000])
    assert exc.value.code == "TOO_MANY_CANDIDATES"
    assert exc.value.index == 2


# ---------------------------------------------------------------------------
# Transition cross-check against an O(K^2) reference
# ---------------------------------------------------------------------------


def _transition_quadratic(queries, sources, h, hc):
    g, gc = [], []
    for q in queries:
        best = None
        cnt = 0
        for s, hv, cv in zip(sources, h, hc):
            v = abs(q - s) + hv
            if best is None or v < best:
                best, cnt = v, cv
            elif v == best:
                cnt = min(2, cnt + cv)
        g.append(best)
        gc.append(cnt)
    return g, gc


def test_transition_matches_quadratic_reference():
    rng = random.Random(20260915)
    for _ in range(2000):
        kq = rng.randint(1, 8)
        ks = rng.randint(1, 8)
        queries = sorted(rng.randint(-50, 50) for _ in range(kq))
        sources = sorted(rng.randint(-50, 50) for _ in range(ks))
        h = [rng.randint(0, 100) for _ in range(ks)]
        hc = [rng.randint(1, 2) for _ in range(ks)]
        assert _transition(queries, sources, h, hc) == \
            _transition_quadratic(queries, sources, h, hc)


# ---------------------------------------------------------------------------
# Brute-force cross-check on random small instances
# ---------------------------------------------------------------------------


def _random_case(rng: random.Random, feasible_bias: bool):
    n = rng.randint(1, 6)
    M = rng.randint(2, 9)
    r = [rng.randrange(M) for _ in range(n)]
    x0 = r[0] + rng.randint(-3, 3) * M
    lo, hi = [], []
    for i in range(1, n):
        if feasible_bias:
            # build an interval around a randomly chosen feasible increment
            t = (r[i] - r[i - 1]) % M
            d = rng.randint(-15, 15)
            d += (t - d) % M
            a = d - rng.randint(0, 2) * M
            b = d + rng.randint(0, 2) * M
        else:
            a = rng.randint(-20, 20)
            b = a + rng.randint(0, 4 * M - 1)
        lo.append(a)
        hi.append(b)
    return M, n, r, x0, lo, hi


@pytest.mark.parametrize("seed", range(400))
def test_random_cases_match_brute_force(seed: int):
    rng = random.Random(seed)
    M, n, r, x0, lo, hi = _random_case(rng, feasible_bias=False)
    assert solve(M, n, r, x0, lo, hi) == brute_solve(M, n, r, x0, lo, hi)


@pytest.mark.parametrize("seed", range(400))
def test_random_feasible_cases_match_brute_force(seed: int):
    rng = random.Random(10_000 + seed)
    M, n, r, x0, lo, hi = _random_case(rng, feasible_bias=True)
    assert solve(M, n, r, x0, lo, hi) == brute_solve(M, n, r, x0, lo, hi)


# ---------------------------------------------------------------------------
# Witness invariants + independent cost recomputation at medium scale
# ---------------------------------------------------------------------------


def _check_witness(M, n, r, x0, lo, hi, witness):
    x = witness["x"]
    inc = witness["increments"]
    assert len(x) == n
    assert len(inc) == n - 1
    assert x[0] == x0
    for i in range(n):
        assert x[i] % M == r[i]
    for i in range(1, n):
        assert x[i] - x[i - 1] == inc[i - 1]
        assert lo[i - 1] <= inc[i - 1] <= hi[i - 1]
    cost = sum(abs(inc[i] - inc[i - 1]) for i in range(1, len(inc)))
    assert cost == witness["cost"]


def _quadratic_optimal_cost(M, n, r, lo, hi):
    cands, first_empty = build_candidates(M, n, r, lo, hi)
    assert first_empty is None
    g = [0] * len(cands[n - 1])
    for i in range(n - 2, 0, -1):
        src = cands[i + 1]
        g = [min(abs(c - s) + gv for s, gv in zip(src, g))
             for c in cands[i]]
    return min(g)


def test_medium_scale_matches_quadratic_dp():
    rng = random.Random(777)
    n, M = 1500, 10**6
    r = [rng.randrange(M) for _ in range(n)]
    x0 = r[0] - 2 * M
    lo, hi = [], []
    for _ in range(n - 1):
        a = rng.randint(-10**9, 10**9)
        lo.append(a)
        hi.append(a + 64 * M - 1)  # exactly 64 candidates per epoch
    result = solve(M, n, r, x0, lo, hi)
    assert result["status"] in ("unique", "ambiguous")
    assert result["cost"] == _quadratic_optimal_cost(M, n, r, lo, hi)
    witnesses = ([{"x": result["x"], "increments": result["increments"],
                   "cost": result["cost"]}]
                 if result["status"] == "unique" else result["witnesses"])
    for w in witnesses:
        _check_witness(M, n, r, x0, lo, hi, w)
    if result["status"] == "ambiguous":
        w1, w2 = result["witnesses"]
        assert w1["x"] != w2["x"]
        assert w1["x"] < w2["x"]  # lexicographic order, ascending


# ---------------------------------------------------------------------------
# Full-scale limit: n = 20000, 64 candidates per epoch
# ---------------------------------------------------------------------------


def test_full_scale_limit():
    rng = random.Random(2026)
    n, M = 20_000, 10**9
    r = [rng.randrange(M) for _ in range(n)]
    x0 = r[0] - 3 * M  # negative start is legal
    lo, hi = [], []
    for _ in range(n - 1):
        a = rng.randint(-10**12, 10**12)
        lo.append(a)
        hi.append(a + 64 * M - 1)
    started = time.monotonic()
    result = solve(M, n, r, x0, lo, hi)
    elapsed = time.monotonic() - started
    assert elapsed < 60, f"solver too slow at the scale limit: {elapsed:.1f}s"
    assert result["status"] in ("unique", "ambiguous")
    witnesses = ([{"x": result["x"], "increments": result["increments"],
                   "cost": result["cost"]}]
                 if result["status"] == "unique" else result["witnesses"])
    for w in witnesses:
        _check_witness(M, n, r, x0, lo, hi, w)
    # deterministic: identical second run, identical result
    assert solve(M, n, r, x0, lo, hi) == result
