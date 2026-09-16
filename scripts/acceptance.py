"""Black-box HTTP acceptance checks against a running API instance.

Covers: health, known answers for all three statuses, brute-force
cross-checks on random small instances, an independent O(n*K^2) cost
recomputation on a larger instance, structured error envelopes and
byte-level determinism of repeated requests — for both ``POST /solve`` and
the windowed what-if batch endpoint ``POST /analyze-windows`` (per-scenario
oracle cross-checks, empty-candidate repair/creation, scenario error
isolation, and a full-scale n = P = 20000 batch that must not degrade to
scenarios x full length).
"""
from __future__ import annotations

import os
import random
import sys
import time

import httpx

from scripts.oracle import oracle_solve

BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

_failures: list[str] = []


def check(name: str, condition: bool, extra: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}"
          + (f" :: {extra}" if extra and not condition else ""), flush=True)
    if not condition:
        _failures.append(name)


def wait_ready() -> bool:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            with httpx.Client(base_url=BASE, timeout=5.0) as client:
                if client.get("/health").status_code == 200:
                    return True
        except httpx.HTTPError:
            time.sleep(0.5)
    return False


def _check_witness(M, n, r, x0, lo, hi, witness) -> bool:
    x, inc, cost = witness["x"], witness["increments"], witness["cost"]
    if len(x) != n or len(inc) != n - 1 or x[0] != x0:
        return False
    if any(x[i] % M != r[i] for i in range(n)):
        return False
    for i in range(1, n):
        if x[i] - x[i - 1] != inc[i - 1]:
            return False
        if not lo[i - 1] <= inc[i - 1] <= hi[i - 1]:
            return False
    recomputed = sum(abs(inc[i] - inc[i - 1]) for i in range(1, len(inc)))
    return recomputed == cost


def _quadratic_optimal_cost(M, n, r, lo, hi) -> int:
    """Independent O(n*K^2) recomputation of the optimal cost."""
    rows = []
    for i in range(1, n):
        target = (r[i] - r[i - 1]) % M
        start = lo[i - 1] + ((target - lo[i - 1]) % M)
        rows.append([start + k * M
                     for k in range((hi[i - 1] - start) // M + 1)])
    g = [0] * len(rows[-1])
    for i in range(len(rows) - 2, -1, -1):
        src = rows[i + 1]
        g = [min(abs(c - s) + gv for s, gv in zip(src, g)) for c in rows[i]]
    return min(g)


def _expected_analyze(M, n, r, x0, lo, hi, windows) -> dict:
    """Expected analyze-windows response, recomputed per scenario with the
    independent brute-force oracle on each modified instance.  ``windows``
    are ``(start, end, lo, hi)`` tuples and must be well-formed."""
    base = oracle_solve(M, n, r, x0, lo, hi)
    if base["status"] == "impossible":
        baseline = {"status": "impossible",
                    "first_empty_index": base["first_empty_index"]}
        base_cost = None
    else:
        baseline = {"status": base["status"], "cost": base["cost"]}
        base_cost = base["cost"]
    scenarios = []
    for idx, (a, b, wlo, whi) in enumerate(windows):
        lo2 = lo[:a - 1] + wlo + lo[b:]
        hi2 = hi[:a - 1] + whi + hi[b:]
        res = oracle_solve(M, n, r, x0, lo2, hi2)
        if res["status"] == "impossible":
            scenarios.append({"index": idx, "status": "impossible",
                              "first_empty_index": res["first_empty_index"]})
        else:
            entry = {"index": idx, "status": res["status"],
                     "cost": res["cost"]}
            if base_cost is not None:
                entry["delta"] = res["cost"] - base_cost
            scenarios.append(entry)
    return {"baseline": baseline, "scenarios": scenarios}


def main() -> int:
    if not wait_ready():
        print("[FAIL] API did not become healthy within 60s", flush=True)
        return 1

    with httpx.Client(base_url=BASE, timeout=httpx.Timeout(120.0)) as client:
        def post(payload):
            return client.post("/solve", json=payload)

        # -- health ------------------------------------------------------
        resp = client.get("/health")
        check("health endpoint", resp.status_code == 200
              and resp.json() == {"status": "ok"})

        # -- known answers ------------------------------------------------
        resp = post({"M": 5, "n": 3, "r": [0, 1, 2], "x0": 0,
                     "lo": [1, 6], "hi": [1, 6]})
        check("known unique answer", resp.status_code == 200 and resp.json() == {
            "status": "unique", "x": [0, 1, 7], "cost": 5,
            "increments": [1, 6]}, resp.text)

        resp = post({"M": 5, "n": 3, "r": [0, 1, 2], "x0": 0,
                     "lo": [0, 0], "hi": [10, 10]})
        check("known ambiguous answer", resp.status_code == 200
              and resp.json() == {
                  "status": "ambiguous", "cost": 0,
                  "witnesses": [
                      {"x": [0, 1, 2], "increments": [1, 1], "cost": 0},
                      {"x": [0, 6, 12], "increments": [6, 6], "cost": 0},
                  ]}, resp.text)

        resp = post({"M": 5, "n": 3, "r": [0, 1, 2], "x0": 0,
                     "lo": [0, 3], "hi": [1, 4]})
        check("known impossible answer", resp.status_code == 200
              and resp.json() == {"status": "impossible",
                                  "first_empty_index": 2}, resp.text)

        # -- arbitrary-precision integers (> 4300 decimal digits) ----------
        big = 10**5000
        resp = post({"M": 10**9, "n": 2, "r": [0, 123456], "x0": big,
                     "lo": [big + 123456], "hi": [big + 123456]})
        check("huge integers (>4300 digits) solved exactly",
              resp.status_code == 200 and resp.json() == {
                  "status": "unique",
                  "x": [big, 2 * big + 123456],
                  "cost": 0,
                  "increments": [big + 123456],
              }, resp.text[:200])

        # -- brute-force cross-check on random small instances ------------
        rng = random.Random(20260915)
        mismatches = 0
        for trial in range(60):
            n = rng.randint(1, 5)
            M = rng.randint(2, 8)
            r = [rng.randrange(M) for _ in range(n)]
            x0 = r[0] + rng.randint(-2, 2) * M
            lo, hi = [], []
            for _ in range(n - 1):
                a = rng.randint(-12, 12)
                lo.append(a)
                hi.append(a + rng.randint(0, 3 * M - 1))
            payload = {"M": M, "n": n, "r": r, "x0": x0, "lo": lo, "hi": hi}
            resp = post(payload)
            expected = oracle_solve(M, n, r, x0, lo, hi)
            if resp.status_code != 200 or resp.json() != expected:
                mismatches += 1
                print(f"  mismatch on {payload}: got {resp.text}, "
                      f"want {expected}", flush=True)
        check("60 random instances match brute-force oracle", mismatches == 0)

        # -- larger instance: invariants + independent cost recomputation -
        n, M = 800, 10**6
        r = [rng.randrange(M) for _ in range(n)]
        x0 = r[0] - 5 * M
        lo, hi = [], []
        for _ in range(n - 1):
            a = rng.randint(-10**8, 10**8)
            lo.append(a)
            hi.append(a + 16 * M - 1)  # 16 candidates per epoch
        resp = post({"M": M, "n": n, "r": r, "x0": x0, "lo": lo, "hi": hi})
        ok = resp.status_code == 200
        if ok:
            body = resp.json()
            ok = body["status"] in ("unique", "ambiguous") \
                and body["cost"] == _quadratic_optimal_cost(M, n, r, lo, hi)
            if ok:
                witnesses = ([{"x": body["x"], "increments": body["increments"],
                               "cost": body["cost"]}]
                             if body["status"] == "unique"
                             else body["witnesses"])
                ok = all(_check_witness(M, n, r, x0, lo, hi, w)
                         for w in witnesses)
                if body["status"] == "ambiguous":
                    w1, w2 = body["witnesses"]
                    ok = ok and w1["x"] < w2["x"] \
                        and w1["cost"] == w2["cost"] == body["cost"]
        check("800-epoch instance: invariants + O(nK^2) cost recomputation",
              ok, resp.text[:400])

        # -- structured errors --------------------------------------------
        error_cases = [
            ({"M": 5, "n": 1, "r": [7], "x0": 0, "lo": [], "hi": []},
             "INVALID_REMAINDER"),
            ({"M": 5, "n": 1, "r": [1], "x0": 0, "lo": [], "hi": []},
             "INVALID_START"),
            ({"M": 5, "n": 2, "r": [0, 1], "x0": 0, "lo": [4], "hi": [1]},
             "INVALID_INTERVAL"),
            ({"M": 5, "n": 2, "r": [0, 1], "x0": 0, "lo": [0], "hi": [400]},
             "TOO_MANY_CANDIDATES"),
            ({"M": 5, "n": 2, "r": [0], "x0": 0, "lo": [0], "hi": [1]},
             "LENGTH_MISMATCH"),
            ({"M": 5.5, "n": 1, "r": [0], "x0": 0, "lo": [], "hi": []},
             "VALIDATION_ERROR"),
        ]
        for payload, code in error_cases:
            resp = post(payload)
            got = None
            try:
                got = resp.json()["error"]["code"]
            except (ValueError, KeyError):
                pass
            check(f"error envelope {code}",
                  resp.status_code == 422 and got == code, resp.text)

        # -- byte-level determinism ----------------------------------------
        payload = {"M": 5, "n": 3, "r": [0, 1, 2], "x0": 0,
                   "lo": [0, 0], "hi": [10, 10]}
        first, second = post(payload), post(payload)
        check("repeated requests are byte-identical",
              first.status_code == second.status_code
              and first.content == second.content)

        # -- analyze-windows: known mixed answer ---------------------------
        mixed = {"M": 5, "n": 3, "r": [0, 1, 2], "x0": 0,
                 "lo": [1, 6], "hi": [1, 6],
                 "scenarios": [
                     {"start": 1, "end": 1, "lo": [0], "hi": [10]},
                     {"start": 2, "end": 2, "lo": [0], "hi": [10]},
                     {"start": 1, "end": 2, "lo": [0, 0], "hi": [10, 10]},
                     {"start": 1, "end": 1, "lo": [3], "hi": [4]},
                     {"start": 2, "end": 2, "lo": [8], "hi": [3]},
                     {"start": 1, "end": 1, "lo": [0], "hi": [1000]},
                 ]}
        resp = client.post("/analyze-windows", json=mixed)
        ok = resp.status_code == 200
        if ok:
            body = resp.json()
            ok = body["baseline"] == {"status": "unique", "cost": 5} \
                and [s["index"] for s in body["scenarios"]] == list(range(6))
            if ok:
                scen = body["scenarios"]
                ok = scen[0] == {"index": 0, "status": "unique", "cost": 0,
                                 "delta": -5} \
                    and scen[1] == {"index": 1, "status": "unique", "cost": 0,
                                    "delta": -5} \
                    and scen[2] == {"index": 2, "status": "ambiguous",
                                    "cost": 0, "delta": -5} \
                    and scen[3] == {"index": 3, "status": "impossible",
                                    "first_empty_index": 1} \
                    and scen[4]["status"] == "error" \
                    and scen[4]["error"]["code"] == "INVALID_INTERVAL" \
                    and scen[4]["error"]["index"] == 2 \
                    and scen[5]["status"] == "error" \
                    and scen[5]["error"]["code"] == "TOO_MANY_CANDIDATES" \
                    and scen[5]["error"]["index"] == 1
        check("analyze-windows known mixed answer (incl. error isolation)",
              ok, resp.text[:600])

        # -- analyze-windows: empty-candidate repair and creation ----------
        resp = client.post("/analyze-windows", json={
            "M": 5, "n": 3, "r": [0, 1, 2], "x0": 0,
            "lo": [0, 3], "hi": [1, 4],
            "scenarios": [{"start": 2, "end": 2, "lo": [1], "hi": [1]},
                          {"start": 1, "end": 1, "lo": [1], "hi": [1]},
                          {"start": 1, "end": 1, "lo": [0], "hi": [0]}]})
        ok = resp.status_code == 200 and resp.json() == {
            "baseline": {"status": "impossible", "first_empty_index": 2},
            "scenarios": [
                {"index": 0, "status": "unique", "cost": 0},
                {"index": 1, "status": "impossible", "first_empty_index": 2},
                {"index": 2, "status": "impossible", "first_empty_index": 1},
            ]}
        check("analyze-windows empty-candidate repair/creation, no delta "
              "when baseline infeasible", ok, resp.text[:400])

        # -- analyze-windows: per-scenario oracle cross-check --------------
        mismatches = 0
        for trial in range(40):
            n = rng.randint(2, 6)
            M = rng.randint(2, 8)
            r = [rng.randrange(M) for _ in range(n)]
            x0 = r[0] + rng.randint(-2, 2) * M
            lo, hi = [], []
            for _ in range(n - 1):
                a = rng.randint(-12, 12)
                lo.append(a)
                hi.append(a + rng.randint(0, 3 * M - 1))
            windows = []
            for _ in range(rng.randint(1, 4)):
                a = rng.randint(1, n - 1)
                b = rng.randint(a, n - 1)
                wlo, whi = [], []
                for _ in range(b - a + 1):
                    x = rng.randint(-12, 12)
                    wlo.append(x)
                    whi.append(x + rng.randint(0, 3 * M - 1))
                windows.append((a, b, wlo, whi))
            payload = {"M": M, "n": n, "r": r, "x0": x0, "lo": lo, "hi": hi,
                       "scenarios": [{"start": a, "end": b, "lo": wl,
                                      "hi": wh}
                                     for a, b, wl, wh in windows]}
            resp = client.post("/analyze-windows", json=payload)
            expected = _expected_analyze(M, n, r, x0, lo, hi, windows)
            if resp.status_code != 200 or resp.json() != expected:
                mismatches += 1
                print(f"  analyze mismatch on {payload}: got "
                      f"{resp.text[:300]}, want {expected}", flush=True)
        check("40 random analyze-windows batches match per-scenario oracle",
              mismatches == 0)

        # -- analyze-windows: byte-level determinism -----------------------
        first = client.post("/analyze-windows", json=mixed)
        second = client.post("/analyze-windows", json=mixed)
        check("analyze-windows repeated requests are byte-identical",
              first.status_code == second.status_code == 200
              and first.content == second.content)

        # -- analyze-windows: full scale, n = P = 20000 --------------------
        n, M = 20_000, 10**9
        r = [rng.randrange(M) for _ in range(n)]
        x0 = r[0] - 3 * M
        lo, hi = [], []
        for _ in range(n - 1):
            a = rng.randint(-10**12, 10**12)
            lo.append(a)
            hi.append(a + 64 * M - 1)  # 64 candidates per epoch
        windows = []
        for i in range(1, n):
            a = rng.randint(-10**12, 10**12)
            windows.append((i, i, [a], [a + 64 * M - 1]))
        a = rng.randint(-10**12, 10**12)
        windows.append((1, 1, [a], [a + 64 * M - 1]))  # P = 20000 total
        payload = {"M": M, "n": n, "r": r, "x0": x0, "lo": lo, "hi": hi,
                   "scenarios": [{"start": a_, "end": b_, "lo": wl, "hi": wh}
                                 for a_, b_, wl, wh in windows]}
        started = time.monotonic()
        resp = client.post("/analyze-windows", json=payload)
        elapsed = time.monotonic() - started
        ok = resp.status_code == 200 and elapsed < 60
        if ok:
            body = resp.json()
            ok = body["baseline"]["status"] in ("unique", "ambiguous") \
                and [s["index"] for s in body["scenarios"]] == \
                list(range(20_000)) \
                and all(s["status"] in ("unique", "ambiguous")
                        and s["delta"] == s["cost"] - body["baseline"]["cost"]
                        for s in body["scenarios"])
        check(f"analyze-windows full scale n=P=20000 in {elapsed:.1f}s "
              "(no scenarios x length degradation)", ok,
              resp.text[:200] if not ok else "")
        # spot-check three scenarios with full /solve runs on the modified
        # instances (the checker may re-solve; the endpoint must not)
        if resp.status_code == 200:
            body = resp.json()
            spot_ok = True
            for idx in (0, 10_000, 19_999):
                a_, b_, wl, wh = windows[idx]
                lo2 = lo[:a_ - 1] + wl + lo[b_:]
                hi2 = hi[:a_ - 1] + wh + hi[b_:]
                ref = post({"M": M, "n": n, "r": r, "x0": x0,
                            "lo": lo2, "hi": hi2})
                entry = body["scenarios"][idx]
                spot_ok = spot_ok and ref.status_code == 200 \
                    and entry["status"] == ref.json()["status"] \
                    and entry["cost"] == ref.json()["cost"]
            check("analyze-windows full-scale spot checks match /solve",
                  spot_ok)

    if _failures:
        print(f"acceptance: {len(_failures)} check(s) failed: "
              f"{_failures}", flush=True)
        return 1
    print("acceptance: all checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
