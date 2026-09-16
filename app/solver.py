"""Exact integer DP solver for the modular carrier-phase unwrapping problem.

Problem
-------
A time-transfer station broadcasts carrier-phase remainders ``r_i`` modulo
``M`` for epochs ``0 .. n-1``.  The true integer timeline ``x_0 .. x_{n-1}``
must satisfy

* ``x_i == r_i (mod M)``            (phase consistency),
* ``x_0`` is given and ``x_0 == r_0 (mod M)``,
* every increment ``d_i = x_i - x_{i-1}`` lies in a closed interval
  ``[lo_i, hi_i]`` for ``i = 1 .. n-1``.

Among all feasible timelines the one minimising the total absolute increment
variation

    cost = sum_{i=2}^{n-1} |d_i - d_{i-1}|          (0 when n < 3)

is requested.  The number of optimal timelines is truncated at 2: the solver
reports ``unique`` (exactly one), ``ambiguous`` (two or more, together with
the two lexicographically smallest timelines by ``x``) or ``impossible``
(some epoch has no candidate increment at all).

Algorithm
---------
Because ``x_i == r_i (mod M)`` and ``x_{i-1} == r_{i-1} (mod M)``, a feasible
increment at epoch ``i`` is any ``d`` in ``[lo_i, hi_i]`` with
``d == r_i - r_{i-1} (mod M)``; the contract guarantees at most 64 such
candidates per epoch.  With ``C_i`` denoting the sorted candidate list, a
backward dynamic program computes, for every ``c`` in ``C_i``,

    g_i(c) = min over c' in C_{i+1} of |c' - c| + g_{i+1}(c')

which is exactly the 1-D L1 distance transform of the row ``g_{i+1}``
evaluated on ``C_i``.  The transform is evaluated with two monotone sweeps
(left sources ``<= c`` and right sources ``> c``), each O(|C_i| + |C_{i+1}|),
so the whole DP costs O(n * K) with K <= 64 — about 2.6e6 elementary integer
operations at the n = 20000 limit.  Alongside the values the same sweeps
propagate the number of optimal continuations, truncated at 2.

Witnesses are reconstructed greedily from the stored ``g`` rows: the first
witness always takes the smallest candidate that can still attain the optimum,
and the second witness is found by scanning, from the last epoch backwards,
for the latest position where a larger candidate still allows an optimal
completion.  Since ``x_0`` is fixed, lexicographic order on ``x`` coincides
with lexicographic order on the increment sequence.

Everything is exact integer arithmetic — no floats are used anywhere, and
negative ``x`` / negative increments are fully supported.
"""
from __future__ import annotations

from .errors import DomainError

MAX_CANDIDATES = 64
COUNT_CAP = 2

# ---------------------------------------------------------------------------
# Domain validation
# ---------------------------------------------------------------------------


def validate_domain(M: int, n: int, r: list[int], x0: int,
                    lo: list[int], hi: list[int]) -> None:
    """Raise :class:`DomainError` on any semantically invalid input.

    Schema-level properties (types, ``M``/``n`` ranges) are enforced by the
    API layer; everything expressible only with the whole payload is checked
    here, in a deterministic order.
    """
    if len(r) != n:
        raise DomainError(
            "LENGTH_MISMATCH",
            f"len(r) is {len(r)} but n is {n}",
            details={"field": "r", "expected": n, "actual": len(r)},
        )
    want = n - 1
    if len(lo) != want:
        raise DomainError(
            "LENGTH_MISMATCH",
            f"len(lo) is {len(lo)} but n-1 is {want}",
            details={"field": "lo", "expected": want, "actual": len(lo)},
        )
    if len(hi) != want:
        raise DomainError(
            "LENGTH_MISMATCH",
            f"len(hi) is {len(hi)} but n-1 is {want}",
            details={"field": "hi", "expected": want, "actual": len(hi)},
        )
    for i, value in enumerate(r):
        if not 0 <= value < M:
            raise DomainError(
                "INVALID_REMAINDER",
                f"r[{i}] = {value} violates 0 <= r_i < M (M = {M})",
                index=i,
            )
    if x0 % M != r[0]:
        raise DomainError(
            "INVALID_START",
            f"x0 = {x0} is not congruent to r[0] = {r[0]} modulo M = {M}",
            index=0,
        )
    for i in range(1, n):
        if lo[i - 1] > hi[i - 1]:
            raise DomainError(
                "INVALID_INTERVAL",
                f"empty interval at epoch {i}: lo = {lo[i - 1]} > hi = {hi[i - 1]}",
                index=i,
            )


def _interval_candidate_span(M: int, target: int, lo_i: int, hi_i: int
                             ) -> tuple[int, int]:
    """Smallest integer ``>= lo_i`` congruent to ``target (mod M)`` and how
    many such integers lie in the closed interval ``[lo_i, hi_i]`` (0 when
    none).

    Pure integer arithmetic, so negative bounds behave correctly.  The count
    is returned instead of the list so callers can enforce the
    ``MAX_CANDIDATES`` limit before materialising anything.
    """
    start = lo_i + ((target - lo_i) % M)
    if start > hi_i:
        return start, 0
    return start, (hi_i - start) // M + 1


def build_candidates(M: int, n: int, r: list[int],
                     lo: list[int], hi: list[int]) -> tuple[list, int | None]:
    """Build the sorted candidate increment list per epoch.

    Returns ``(cands, first_empty)`` where ``cands[i]`` (``1 <= i <= n-1``) is
    the ascending list of feasible increments at epoch ``i`` and
    ``first_empty`` is the smallest epoch with no candidate (``None`` if every
    epoch has at least one).

    A candidate at epoch ``i`` is any integer ``d`` with
    ``lo_i <= d <= hi_i`` and ``d == r_i - r_{i-1} (mod M)``.  The smallest
    such ``d >= lo_i`` is ``lo_i + ((r_i - r_{i-1} - lo_i) mod M)`` and the
    rest follow in steps of ``M``.

    Raises ``DomainError(TOO_MANY_CANDIDATES)`` if any interval holds more
    than ``MAX_CANDIDATES`` candidates.  All intervals are checked for this
    violation even after an empty interval has been seen, so malformed input
    is always reported as an error rather than as impossibility.
    """
    cands: list = [None] * n
    first_empty: int | None = None
    for i in range(1, n):
        target = (r[i] - r[i - 1]) % M
        start, count = _interval_candidate_span(M, target, lo[i - 1],
                                                hi[i - 1])
        if count == 0:
            if first_empty is None:
                first_empty = i
            continue
        if count > MAX_CANDIDATES:
            raise DomainError(
                "TOO_MANY_CANDIDATES",
                f"interval at epoch {i} holds {count} modular candidates "
                f"(limit {MAX_CANDIDATES})",
                index=i,
                details={"count": count, "max": MAX_CANDIDATES},
            )
        cands[i] = [start + k * M for k in range(count)]
    return cands, first_empty


# ---------------------------------------------------------------------------
# Core DP: one L1 distance-transform step with truncated counting
# ---------------------------------------------------------------------------


def _transition(queries: list[int], sources: list[int],
                h: list[int], hc: list[int]) -> tuple[list, list]:
    """One backward-DP step.

    ``queries`` are the sorted candidates at epoch ``i``; ``sources`` the
    sorted candidates at epoch ``i+1``; ``h[j]`` is the optimal suffix cost
    ``g_{i+1}(sources[j])`` and ``hc[j]`` the truncated (<= 2) number of
    optimal continuations from ``sources[j]``.

    Returns ``(g, gc)`` with, for every query ``c``,

    * ``g[t]  = min_j |c - sources[j]| + h[j]``
    * ``gc[t] = min(2, sum of hc[j] over the sources j attaining the min)``

    Runs in O(len(queries) + len(sources)) via two monotone sweeps: sources
    ``<= c`` contribute ``c + min(h[j] - sources[j])`` and sources ``> c``
    contribute ``-c + min(h[j] + sources[j])``.  The two source sets are
    disjoint, so counts never double-count a source state.
    """
    m = len(queries)
    k = len(sources)

    left_v: list = [None] * m
    left_c = [0] * m
    j = 0
    best = 0
    bestc = 0
    have = False
    for t in range(m):
        qt = queries[t]
        while j < k and sources[j] <= qt:
            v = h[j] - sources[j]
            c = hc[j]
            if not have or v < best:
                best = v
                bestc = c
                have = True
            elif v == best:
                s = bestc + c
                bestc = COUNT_CAP if s > COUNT_CAP else s
            j += 1
        if have:
            left_v[t] = best + qt
            left_c[t] = bestc

    right_v: list = [None] * m
    right_c = [0] * m
    j = k - 1
    best = 0
    bestc = 0
    have = False
    for t in range(m - 1, -1, -1):
        qt = queries[t]
        while j >= 0 and sources[j] > qt:
            v = h[j] + sources[j]
            c = hc[j]
            if not have or v < best:
                best = v
                bestc = c
                have = True
            elif v == best:
                s = bestc + c
                bestc = COUNT_CAP if s > COUNT_CAP else s
            j -= 1
        if have:
            right_v[t] = best - qt
            right_c[t] = bestc

    g = [0] * m
    gc = [0] * m
    for t in range(m):
        lv = left_v[t]
        rv = right_v[t]
        if lv is None:
            g[t] = rv
            gc[t] = right_c[t]
        elif rv is None:
            g[t] = lv
            gc[t] = left_c[t]
        elif lv < rv:
            g[t] = lv
            gc[t] = left_c[t]
        elif rv < lv:
            g[t] = rv
            gc[t] = right_c[t]
        else:
            g[t] = lv
            s = left_c[t] + right_c[t]
            gc[t] = COUNT_CAP if s > COUNT_CAP else s
    return g, gc


# ---------------------------------------------------------------------------
# Witness reconstruction
# ---------------------------------------------------------------------------


def _greedy_from(n: int, cands: list, g_rows: list,
                 prefix: list[int], pos: int, idx: int) -> list[int]:
    """Build the lexicographically smallest optimal increment sequence whose
    first ``pos - 1`` increments equal ``prefix[1:pos]`` and whose increment at
    ``pos`` is ``cands[pos][idx]``.

    At every later epoch the smallest candidate ``c`` satisfying the optimal
    substructure equation ``|c - prev| + g_i(c) == g_{i-1}(prev)`` is chosen.
    """
    inc = [0] * n
    inc[1:pos] = prefix[1:pos]
    inc[pos] = cands[pos][idx]
    for i in range(pos + 1, n):
        prev = inc[i - 1]
        target = g_rows[i - 1][idx]
        row = g_rows[i]
        ci = cands[i]
        for t in range(len(ci)):
            c = ci[t]
            d = c - prev
            if d < 0:
                d = -d
            if d + row[t] == target:
                idx = t
                break
        inc[i] = ci[idx]
    return inc


def _second_witness(M: int, n: int, cands: list, g_rows: list,
                    first_inc: list[int], optimal: int) -> list[int] | None:
    """Increment sequence of the second lexicographically smallest optimal
    timeline, or ``None`` if the optimum is unique.

    Any optimal sequence different from the first witness deviates from it at
    some earliest epoch ``i`` with a larger candidate.  The second witness is
    therefore the deviation with the *latest* possible epoch, the *smallest*
    larger candidate there, and the lexicographically smallest optimal
    completion afterwards.
    """
    for i in range(n - 1, 0, -1):
        ci = cands[i]
        k0 = (first_inc[i] - ci[0]) // M + 1  # first candidate > first_inc[i]
        if k0 >= len(ci):
            continue
        if i == 1:
            row = g_rows[1]
            for k in range(k0, len(ci)):
                if row[k] == optimal:
                    return _greedy_from(n, cands, g_rows, first_inc, 1, k)
        else:
            prev = first_inc[i - 1]
            pidx = (prev - cands[i - 1][0]) // M
            target = g_rows[i - 1][pidx]
            row = g_rows[i]
            for k in range(k0, len(ci)):
                c = ci[k]
                d = c - prev
                if d < 0:
                    d = -d
                if d + row[k] == target:
                    return _greedy_from(n, cands, g_rows, first_inc, i, k)
    return None


def _x_from_inc(x0: int, inc: list[int], n: int) -> list[int]:
    x = [0] * n
    x[0] = x0
    s = x0
    for i in range(1, n):
        s += inc[i]
        x[i] = s
    return x


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def solve(M: int, n: int, r: list[int], x0: int,
          lo: list[int], hi: list[int]) -> dict:
    """Solve one unwrapping instance.

    Returns a JSON-serialisable dict:

    * ``{"status": "unique", "x": [...], "cost": c, "increments": [...]}``
    * ``{"status": "ambiguous", "cost": c, "witnesses": [w1, w2]}`` with each
      witness ``{"x": [...], "increments": [...], "cost": c}`` ordered by
      lexicographically increasing ``x``
    * ``{"status": "impossible", "first_empty_index": i}``

    Raises :class:`DomainError` for semantically invalid input.
    """
    validate_domain(M, n, r, x0, lo, hi)
    cands, first_empty = build_candidates(M, n, r, lo, hi)
    if first_empty is not None:
        return {"status": "impossible", "first_empty_index": first_empty}
    if n == 1:
        return {"status": "unique", "x": [x0], "cost": 0, "increments": []}

    # Backward DP.  g_rows[i][k] = optimal suffix cost when d_i = cands[i][k].
    # Counts are only needed while sweeping, so just the current row is kept.
    g_rows: list = [None] * n
    g_rows[n - 1] = [0] * len(cands[n - 1])
    gc = [1] * len(cands[n - 1])
    for i in range(n - 2, 0, -1):
        g, gc = _transition(cands[i], cands[i + 1], g_rows[i + 1], gc)
        g_rows[i] = g

    row1 = g_rows[1]
    optimal = min(row1)
    total = 0
    for v, c in zip(row1, gc):
        if v == optimal:
            total += c
            if total >= COUNT_CAP:
                total = COUNT_CAP
                break

    first_inc = _greedy_from(n, cands, g_rows, [0] * n, 1,
                             row1.index(optimal))
    if total == 1:
        return {
            "status": "unique",
            "x": _x_from_inc(x0, first_inc, n),
            "cost": optimal,
            "increments": first_inc[1:],
        }

    second_inc = _second_witness(M, n, cands, g_rows, first_inc, optimal)
    if second_inc is None:  # pragma: no cover - defensive, cannot happen
        raise RuntimeError(
            "counted >= 2 optimal timelines but found no second witness")
    witnesses = [
        {"x": _x_from_inc(x0, first_inc, n),
         "increments": first_inc[1:], "cost": optimal},
        {"x": _x_from_inc(x0, second_inc, n),
         "increments": second_inc[1:], "cost": optimal},
    ]
    return {"status": "ambiguous", "cost": optimal, "witnesses": witnesses}
