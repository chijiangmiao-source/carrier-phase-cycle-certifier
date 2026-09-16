"""Windowed what-if analysis: known answers, cross-checks, scale limits."""
from __future__ import annotations

import random
import time

import pytest

from app.analyze import Window, analyze_windows
from app.errors import DomainError
from app.solver import solve
from tests.brute import brute_solve


def _apply_window(lo, hi, w: Window):
    """Return the lo/hi arrays of the instance modified by window ``w``."""
    return (lo[:w.start - 1] + list(w.lo) + lo[w.end:],
            hi[:w.start - 1] + list(w.hi) + hi[w.end:])


# ---------------------------------------------------------------------------
# Known-answer cases
# ---------------------------------------------------------------------------


def test_known_mixed_scenarios():
    result = analyze_windows(5, 3, [0, 1, 2], 0, [1, 6], [1, 6], [
        Window(1, 1, [0], [10]),        # C1={1,6}, C2={6} -> (6,6) cost 0
        Window(2, 2, [0], [10]),        # C1={1}, C2={1,6} -> (1,1) cost 0
        Window(1, 2, [0, 0], [10, 10]),  # both free -> ambiguous, cost 0
        Window(1, 1, [3], [4]),         # no candidate == 1 mod 5 -> empty
        Window(2, 2, [8], [3]),         # lo > hi -> rejected scenario
        Window(1, 1, [0], [1000]),      # 200 candidates -> rejected scenario
    ])
    assert result["baseline"] == {"status": "unique", "cost": 5}
    scen = result["scenarios"]
    assert [s["index"] for s in scen] == list(range(6))
    assert scen[0] == {"index": 0, "status": "unique", "cost": 0, "delta": -5}
    assert scen[1] == {"index": 1, "status": "unique", "cost": 0, "delta": -5}
    assert scen[2] == {"index": 2, "status": "ambiguous", "cost": 0,
                       "delta": -5}
    assert scen[3] == {"index": 3, "status": "impossible",
                       "first_empty_index": 1}
    assert scen[4]["status"] == "error"
    assert scen[4]["error"]["code"] == "INVALID_INTERVAL"
    assert scen[4]["error"]["index"] == 2
    assert scen[5]["status"] == "error"
    assert scen[5]["error"]["code"] == "TOO_MANY_CANDIDATES"
    assert scen[5]["error"]["index"] == 1
    assert scen[5]["error"]["details"] == {"count": 200, "max": 64}


def test_empty_candidate_repair_and_creation():
    # Baseline epoch 2 interval [3, 4] has no candidate (need == 1 mod 5).
    result = analyze_windows(5, 3, [0, 1, 2], 0, [0, 3], [1, 4], [
        Window(2, 2, [1], [1]),   # repairs epoch 2 -> d = (1, 1), cost 0
        Window(1, 1, [1], [1]),   # epoch 2 untouched, still empty
        Window(2, 2, [3], [4]),   # replacement is empty as well
        Window(1, 1, [0], [0]),   # window itself manufactures an empty epoch
    ])
    assert result["baseline"] == {"status": "impossible",
                                  "first_empty_index": 2}
    scen = result["scenarios"]
    # baseline infeasible -> no delta anywhere
    assert scen[0] == {"index": 0, "status": "unique", "cost": 0}
    assert scen[1] == {"index": 1, "status": "impossible",
                       "first_empty_index": 2}
    assert scen[2] == {"index": 2, "status": "impossible",
                       "first_empty_index": 2}
    assert scen[3] == {"index": 3, "status": "impossible",
                       "first_empty_index": 1}

    # Manufacturing an empty epoch inside a feasible baseline.
    result = analyze_windows(5, 3, [0, 1, 2], 0, [1, 6], [1, 6],
                             [Window(1, 1, [3], [4])])
    assert result["baseline"] == {"status": "unique", "cost": 5}
    assert result["scenarios"][0] == {"index": 0, "status": "impossible",
                                      "first_empty_index": 1}


def test_baseline_ambiguous_delta():
    result = analyze_windows(5, 3, [0, 1, 2], 0, [0, 0], [10, 10], [
        Window(1, 1, [1], [1]),        # C1={1}: (1,1) cost 0, unique
        Window(1, 2, [1, 6], [1, 6]),  # forced (1,6): cost 5
    ])
    assert result["baseline"] == {"status": "ambiguous", "cost": 0}
    assert result["scenarios"][0] == {"index": 0, "status": "unique",
                                      "cost": 0, "delta": 0}
    assert result["scenarios"][1] == {"index": 1, "status": "unique",
                                      "cost": 5, "delta": 5}


def test_single_epoch_baseline_no_windows():
    assert analyze_windows(5, 1, [3], -7, [], [], []) == {
        "baseline": {"status": "unique", "cost": 0}, "scenarios": []}


def test_negative_increments_in_windows():
    # C1 = {-9, -4} (== 1 mod 5), C2 = {-6, -1} (== 4 mod 5);
    # window forces d1 = -9 -> costs (-9,-6)=3, (-9,-1)=8 -> unique cost 3.
    result = analyze_windows(5, 3, [0, 1, 0], -10, [-9, -9], [-1, -1], [
        Window(1, 1, [-9], [-9]),
        Window(2, 2, [-6], [-6]),  # forces optimum (-4,-6) cost 2
    ])
    assert result["baseline"] == {"status": "unique", "cost": 2}
    assert result["scenarios"][0] == {"index": 0, "status": "unique",
                                      "cost": 3, "delta": 1}
    assert result["scenarios"][1] == {"index": 1, "status": "unique",
                                      "cost": 2, "delta": 0}


def test_window_covering_all_epochs():
    # Replacing every epoch must reproduce a fresh solve of the new instance.
    lo, hi = [0, 3, -5], [10, 4, 5]
    windows = [Window(1, 3, [1, 1, 1], [1, 1, 1])]
    result = analyze_windows(5, 4, [0, 1, 2, 3], 0, lo, hi, windows)
    expected = solve(5, 4, [0, 1, 2, 3], 0, [1, 1, 1], [1, 1, 1])
    entry = result["scenarios"][0]
    assert entry["status"] == expected["status"] == "unique"
    assert entry["cost"] == expected["cost"] == 0


def test_multiple_empty_epochs_earliest_reported():
    # Baseline epochs 2 and 4 are both empty; a window covering only epoch 2
    # must still report epoch 4, and vice versa.
    lo = [0, 3, 0, 3]
    hi = [1, 4, 1, 4]
    result = analyze_windows(5, 5, [0, 1, 2, 3, 4], 0, lo, hi, [
        Window(2, 2, [1], [1]),          # fixes epoch 2; epoch 4 remains
        Window(4, 4, [6], [6]),          # fixes epoch 4; epoch 2 remains
        Window(2, 4, [1, 1, 6], [1, 1, 6]),  # fixes both -> feasible
    ])
    assert result["baseline"] == {"status": "impossible",
                                  "first_empty_index": 2}
    scen = result["scenarios"]
    assert scen[0] == {"index": 0, "status": "impossible",
                       "first_empty_index": 4}
    assert scen[1] == {"index": 1, "status": "impossible",
                       "first_empty_index": 2}
    # forced increments (1, 1, 1, 6): cost = |6 - 1| = 5, unique
    assert scen[2] == {"index": 2, "status": "unique", "cost": 5}


def test_scenario_error_isolation_and_order():
    # An invalid scenario never affects its neighbours; first violation wins.
    result = analyze_windows(5, 4, [0, 1, 2, 3], 0, [1, 1, 1], [1, 1, 1], [
        Window(1, 1, [1], [1]),              # ok
        Window(2, 3, [5, 0], [2, 0]),        # INVALID_INTERVAL at epoch 2
        Window(2, 3, [0, 5], [0, 2]),        # INVALID_INTERVAL at epoch 3
        Window(1, 1, [0], [1000]),           # TOO_MANY_CANDIDATES at epoch 1
        Window(3, 3, [1], [1]),              # ok
    ])
    scen = result["scenarios"]
    assert [s["index"] for s in scen] == list(range(5))
    assert scen[0]["status"] == "unique" and scen[0]["cost"] == 0
    assert scen[1]["error"]["code"] == "INVALID_INTERVAL"
    assert scen[1]["error"]["index"] == 2
    assert scen[2]["error"]["code"] == "INVALID_INTERVAL"
    assert scen[2]["error"]["index"] == 3
    assert scen[3]["error"]["code"] == "TOO_MANY_CANDIDATES"
    assert scen[3]["error"]["index"] == 1
    assert scen[4]["status"] == "unique" and scen[4]["cost"] == 0


def test_interval_error_checked_before_candidate_overflow():
    # A window whose interval is both inverted and (if valid) too wide must
    # report INVALID_INTERVAL, mirroring the solve error priority.
    result = analyze_windows(5, 2, [0, 1], 0, [1], [1],
                             [Window(1, 1, [1000], [0])])
    assert result["scenarios"][0]["error"]["code"] == "INVALID_INTERVAL"


# ---------------------------------------------------------------------------
# Baseline domain errors propagate exactly like solve
# ---------------------------------------------------------------------------


def test_baseline_domain_errors_raise_like_solve():
    with pytest.raises(DomainError) as exc:
        analyze_windows(5, 2, [0, 1], 0, [3], [1], [Window(1, 1, [1], [1])])
    assert exc.value.code == "INVALID_INTERVAL"
    with pytest.raises(DomainError) as exc:
        analyze_windows(5, 2, [0, 1], 0, [0], [1000], [Window(1, 1, [1], [1])])
    assert exc.value.code == "TOO_MANY_CANDIDATES"
    with pytest.raises(DomainError) as exc:
        analyze_windows(5, 1, [9], 0, [], [], [])
    assert exc.value.code == "INVALID_REMAINDER"
    with pytest.raises(DomainError) as exc:
        analyze_windows(5, 2, [0, 1], 0, [], [], [])
    assert exc.value.code == "LENGTH_MISMATCH"


# ---------------------------------------------------------------------------
# Brute-force cross-check on random small instances
# ---------------------------------------------------------------------------


def _random_windows(rng: random.Random, n: int, M: int, count: int):
    windows = []
    for _ in range(count):
        a = rng.randint(1, n - 1)
        b = rng.randint(a, n - 1)
        lo, hi = [], []
        for _ in range(b - a + 1):
            x = rng.randint(-15, 15)
            lo.append(x)
            hi.append(x + rng.randint(0, 3 * M - 1))
        windows.append(Window(a, b, lo, hi))
    return windows


def _check_against_reference(M, n, r, x0, lo, hi, windows, reference, result):
    """Compare one analyze_windows result against a per-scenario reference
    (brute force or solve) evaluated on each modified instance."""
    base = reference(M, n, r, x0, lo, hi)
    if base["status"] == "impossible":
        assert result["baseline"] == {
            "status": "impossible",
            "first_empty_index": base["first_empty_index"]}
        base_cost = None
    else:
        assert result["baseline"] == {"status": base["status"],
                                      "cost": base["cost"]}
        base_cost = base["cost"]

    assert [s["index"] for s in result["scenarios"]] == \
        list(range(len(windows)))
    for entry, w in zip(result["scenarios"], windows):
        lo2, hi2 = _apply_window(lo, hi, w)
        expected = reference(M, n, r, x0, lo2, hi2)
        assert entry["status"] == expected["status"]
        if expected["status"] == "impossible":
            assert entry["first_empty_index"] == \
                expected["first_empty_index"]
        else:
            assert entry["cost"] == expected["cost"]
            if base_cost is None:
                assert "delta" not in entry
            else:
                assert entry["delta"] == expected["cost"] - base_cost


@pytest.mark.parametrize("seed", range(300))
def test_random_scenarios_match_brute_force(seed: int):
    rng = random.Random(50_000 + seed)
    n = rng.randint(2, 6)
    M = rng.randint(2, 9)
    r = [rng.randrange(M) for _ in range(n)]
    x0 = r[0] + rng.randint(-3, 3) * M
    lo, hi = [], []
    for _ in range(n - 1):
        a = rng.randint(-15, 15)
        lo.append(a)
        hi.append(a + rng.randint(0, 3 * M - 1))
    windows = _random_windows(rng, n, M, rng.randint(1, 4))
    result = analyze_windows(M, n, r, x0, lo, hi, windows)
    _check_against_reference(M, n, r, x0, lo, hi, windows, brute_solve,
                             result)


# ---------------------------------------------------------------------------
# Medium-scale consistency against solve on each modified instance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_medium_scenarios_match_solve(seed: int):
    rng = random.Random(90_000 + seed)
    n = rng.randint(50, 200)
    M = rng.randint(2, 10**6)
    r = [rng.randrange(M) for _ in range(n)]
    x0 = r[0] - rng.randint(0, 5) * M
    lo, hi = [], []
    for _ in range(n - 1):
        a = rng.randint(-10**6, 10**6)
        lo.append(a)
        hi.append(a + rng.randint(0, 8 * M))  # up to 9 candidates per epoch
    windows = _random_windows(rng, n, M, rng.randint(1, 6))
    result = analyze_windows(M, n, r, x0, lo, hi, windows)
    _check_against_reference(M, n, r, x0, lo, hi, windows, solve, result)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_repeat():
    args = (5, 4, [0, 1, 2, 3], 0, [0, 3, 0], [10, 4, 10])
    windows = [Window(1, 2, [0, 0], [10, 10]), Window(3, 3, [2], [2]),
               Window(2, 2, [9], [1])]
    assert analyze_windows(*args, windows) == analyze_windows(*args, windows)


# ---------------------------------------------------------------------------
# Full-scale limit: n = 20000 epochs, P = 20000 replaced points
# ---------------------------------------------------------------------------


def _full_scale_instance():
    rng = random.Random(2026)
    n, M = 20_000, 10**9
    r = [rng.randrange(M) for _ in range(n)]
    x0 = r[0] - 3 * M
    lo, hi = [], []
    for _ in range(n - 1):
        a = rng.randint(-10**12, 10**12)
        lo.append(a)
        hi.append(a + 64 * M - 1)  # 64 candidates per epoch
    return M, n, r, x0, lo, hi


def test_full_scale_many_single_epoch_windows():
    # 20000 scenarios of one replaced epoch each (P = 20000): a degraded
    # scenarios x full-length implementation would need hours, the spliced
    # DP stays in the O((n+P)*K) regime.
    M, n, r, x0, lo, hi = _full_scale_instance()
    rng = random.Random(99)
    windows = []
    for i in range(1, n):
        a = rng.randint(-10**12, 10**12)
        windows.append(Window(i, i, [a], [a + 64 * M - 1]))
    a = rng.randint(-10**12, 10**12)
    windows.append(Window(1, 1, [a], [a + 64 * M - 1]))  # P = 20000 total

    started = time.monotonic()
    result = analyze_windows(M, n, r, x0, lo, hi, windows)
    elapsed = time.monotonic() - started
    assert elapsed < 60, f"window analysis degraded: {elapsed:.1f}s"

    assert result["baseline"]["status"] in ("unique", "ambiguous")
    base_cost = result["baseline"]["cost"]
    scen = result["scenarios"]
    assert [s["index"] for s in scen] == list(range(20_000))
    for entry in scen:
        assert entry["status"] in ("unique", "ambiguous")
        assert entry["delta"] == entry["cost"] - base_cost

    # Spot-check a few scenarios against a full solve of the modified
    # instance (allowed for the *checker*, never inside the endpoint).
    for idx in (0, 10_000, 19_999):
        lo2, hi2 = _apply_window(lo, hi, windows[idx])
        expected = solve(M, n, r, x0, lo2, hi2)
        assert scen[idx]["status"] == expected["status"]
        assert scen[idx]["cost"] == expected["cost"]


def test_full_scale_wide_windows():
    # 200 scenarios of 100 replaced epochs each (P = 20000): exercises the
    # in-window DP, not just the boundary splicing.
    M, n, r, x0, lo, hi = _full_scale_instance()
    rng = random.Random(77)
    windows = []
    for k in range(200):
        a = 1 + (k * 100) % (n - 101)
        wl = [rng.randint(-10**12, 10**12) for _ in range(100)]
        windows.append(Window(a, a + 99, wl, [x + 64 * M - 1 for x in wl]))

    started = time.monotonic()
    result = analyze_windows(M, n, r, x0, lo, hi, windows)
    elapsed = time.monotonic() - started
    assert elapsed < 60, f"window analysis degraded: {elapsed:.1f}s"
    assert len(result["scenarios"]) == 200
    for entry in result["scenarios"]:
        assert entry["status"] in ("unique", "ambiguous")

    lo2, hi2 = _apply_window(lo, hi, windows[123])
    expected = solve(M, n, r, x0, lo2, hi2)
    assert result["scenarios"][123]["cost"] == expected["cost"]
