"""Brute-force oracle used by the acceptance checks.

Deliberately written as a straightforward enumeration (independent of the
solver's DP) so the HTTP results are recomputed from first principles.
"""
from __future__ import annotations

import itertools


def oracle_solve(M: int, n: int, r: list[int], x0: int,
                 lo: list[int], hi: list[int]) -> dict:
    cands = []
    for i in range(1, n):
        target = (r[i] - r[i - 1]) % M
        cands.append([d for d in range(lo[i - 1], hi[i - 1] + 1)
                      if d % M == target])
    for i, cs in enumerate(cands, start=1):
        if not cs:
            return {"status": "impossible", "first_empty_index": i}
    if n == 1:
        return {"status": "unique", "x": [x0], "cost": 0, "increments": []}

    best = None
    seqs: list[tuple[int, ...]] = []
    for combo in itertools.product(*cands):
        cost = sum(abs(combo[j] - combo[j - 1]) for j in range(1, len(combo)))
        if best is None or cost < best:
            best = cost
            seqs = [combo]
        elif cost == best:
            seqs.append(combo)
    seqs.sort()

    def witness(inc: tuple[int, ...]) -> dict:
        x = [x0]
        for d in inc:
            x.append(x[-1] + d)
        return {"x": x, "increments": list(inc), "cost": best}

    if len(seqs) == 1:
        w = witness(seqs[0])
        return {"status": "unique", "x": w["x"], "cost": best,
                "increments": w["increments"]}
    return {"status": "ambiguous", "cost": best,
            "witnesses": [witness(seqs[0]), witness(seqs[1])]}
