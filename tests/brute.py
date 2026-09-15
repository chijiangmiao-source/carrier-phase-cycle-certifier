"""Independent brute-force reference implementation.

Enumerates every feasible increment sequence with ``itertools.product`` and
reproduces the exact response contract of :func:`app.solver.solve`.  Used to
cross-check the DP solver on small instances.  Only suitable for tiny inputs
(exponential in ``n``).
"""
from __future__ import annotations

import itertools


def _candidates(M: int, n: int, r: list[int],
                lo: list[int], hi: list[int]) -> list[list[int]]:
    cands: list[list[int]] = []
    for i in range(1, n):
        target = (r[i] - r[i - 1]) % M
        cands.append([d for d in range(lo[i - 1], hi[i - 1] + 1)
                      if d % M == target])
    return cands


def _cost(inc: tuple[int, ...]) -> int:
    return sum(abs(inc[j] - inc[j - 1]) for j in range(1, len(inc)))


def _witness(x0: int, inc: tuple[int, ...], cost: int) -> dict:
    x = [x0]
    for d in inc:
        x.append(x[-1] + d)
    return {"x": x, "increments": list(inc), "cost": cost}


def brute_solve(M: int, n: int, r: list[int], x0: int,
                lo: list[int], hi: list[int]) -> dict:
    cands = _candidates(M, n, r, lo, hi)
    for i, cs in enumerate(cands, start=1):
        if not cs:
            return {"status": "impossible", "first_empty_index": i}
    if n == 1:
        return {"status": "unique", "x": [x0], "cost": 0, "increments": []}

    best: int | None = None
    seqs: list[tuple[int, ...]] = []
    for combo in itertools.product(*cands):
        cost = _cost(combo)
        if best is None or cost < best:
            best = cost
            seqs = [combo]
        elif cost == best:
            seqs.append(combo)

    assert best is not None
    seqs.sort()  # lexicographic on increments == lexicographic on x (x0 fixed)
    if len(seqs) == 1:
        w = _witness(x0, seqs[0], best)
        return {"status": "unique", "x": w["x"], "cost": best,
                "increments": w["increments"]}
    return {
        "status": "ambiguous",
        "cost": best,
        "witnesses": [_witness(x0, seqs[0], best), _witness(x0, seqs[1], best)],
    }
