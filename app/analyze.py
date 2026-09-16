"""Windowed what-if analysis on top of the exact integer DP solver.

Companion of ``POST /analyze-windows``: given one baseline instance and a
batch of independent scenarios — each replacing the increment intervals of a
single contiguous epoch window — report the baseline verdict and one compact
verdict per scenario, without re-solving the whole timeline per scenario.

Algorithm
---------
The baseline is swept once in **both** directions:

* backward rows ``g_i(c)`` — optimal suffix cost when ``d_i = c`` (the same
  L1 distance-transform step as :func:`app.solver.solve`), and
* forward rows ``f_i(c)`` — optimal prefix cost when ``d_i = c``,

each carrying the number of optimal continuations truncated at 2.  Only rows
sitting at scenario window boundaries are cached, so the baseline phase costs
``O(n*K)`` time and ``O((n + B)*K)`` memory with ``B`` cached boundary rows.

A scenario replacing epochs ``a .. b`` is then evaluated by recomputing the
DP **inside the window only** and splicing the cached prefix/suffix rows:

* seed ``h_a`` from ``f_{a-1}`` (or from zeros when ``a = 1``),
* sweep the replacement candidates forward to ``h_b``,
* combine with ``g_{b+1}`` (unless ``b = n - 1``):

  ``cost = min_c h_b(c) + min_{c'} |c' - c| + g_{b+1}(c')``

  with the truncated counts multiplied per boundary candidate and summed
  over the argmin (capping at 2 is preserved by products and sums).

Every step reuses the solver's candidate generation and truncated-counting
transition, so a scenario touching ``w`` epochs costs ``O(w*K)`` and the
whole batch stays ``O((n + P)*K)`` with ``P`` the total number of replaced
points — never ``scenarios * n``, and :func:`app.solver.solve` is never
called per scenario.

Empty candidate epochs are not errors: the baseline's empty epochs are
collected once, and a scenario is impossible exactly when an empty epoch
survives outside its window or its own replacement intervals produce one;
the earliest such epoch is reported.  Malformed scenario intervals
(``lo > hi``) and windows holding more than 64 modular candidates reject
only their own scenario — the rest of the batch is unaffected.
"""
from __future__ import annotations

from bisect import bisect_right
from typing import NamedTuple

from .errors import DomainError
from .solver import (
    COUNT_CAP,
    MAX_CANDIDATES,
    _interval_candidate_span,
    _transition,
    build_candidates,
    validate_domain,
)

#: Upper bound on the total number of replaced points over all scenarios
#: (mirrored by the request schema).
MAX_TOTAL_REPLACEMENTS = 20_000


class Window(NamedTuple):
    """One what-if scenario: replace the increment intervals of the closed
    epoch window ``start .. end`` (increment indices, ``1 <= start <= end <=
    n - 1``).  ``lo``/``hi`` must each hold exactly ``end - start + 1``
    values; this is enforced by the API schema."""

    start: int
    end: int
    lo: list[int]
    hi: list[int]


def _min_with_count(values: list[int], counts) -> tuple[int, int]:
    """Minimum value and the truncated (<= 2) total count of the entries
    attaining it — the same capping convention as the solver."""
    best = min(values)
    total = 0
    for v, c in zip(values, counts):
        if v == best:
            total += c
            if total >= COUNT_CAP:
                total = COUNT_CAP
                break
    return best, total


def _scenario_error(index: int, exc: DomainError) -> dict:
    """Isolated per-scenario rejection, reusing the domain-error envelope
    shape (``{"code", "message", "index"[, "details"]}``)."""
    return {"index": index, "status": "error", "error": exc.payload()["error"]}


def analyze_windows(M: int, n: int, r: list[int], x0: int,
                    lo: list[int], hi: list[int],
                    windows: list[Window]) -> dict:
    """Evaluate one baseline instance plus a batch of window replacements.

    Returns ``{"baseline": ..., "scenarios": [...]}``:

    * baseline — ``{"status": "unique"|"ambiguous", "cost": c}`` or
      ``{"status": "impossible", "first_empty_index": i}``;
    * one entry per scenario, in input order —
      ``{"index", "status": "unique"|"ambiguous", "cost", "delta"}`` with
      ``delta = cost - baseline cost`` present only when the baseline is
      feasible, ``{"index", "status": "impossible", "first_empty_index"}``,
      or ``{"index", "status": "error", "error": {...}}`` for a rejected
      scenario.

    Raises :class:`DomainError` when the *baseline* request itself is
    invalid, with the same codes and priority as :func:`app.solver.solve`.
    """
    validate_domain(M, n, r, x0, lo, hi)
    cands, first_empty = build_candidates(M, n, r, lo, hi)
    # Epochs whose baseline interval has no candidate, ascending.
    empty_epochs = [i for i in range(1, n) if cands[i] is None]

    # Boundary rows the scenarios may splice into.
    need_f: set[int] = set()
    need_g: set[int] = set()
    for w in windows:
        if w.start >= 2:
            need_f.add(w.start - 1)
        if w.end <= n - 2:
            need_g.add(w.end + 1)
    if n >= 2:
        need_g.add(1)  # baseline verdict

    # Backward sweep over the baseline: g_rows[i] = (values, counts) with
    # g_i(c) the optimal suffix cost when d_i = c.  A row exists only when
    # every epoch from i to n-1 has candidates; counts are stored as bytes
    # (values 1 or 2) to keep the cache compact.
    g_rows: dict[int, tuple[list[int], bytes]] = {}
    if n >= 2:
        cur: tuple[list[int], list[int]] | None = None
        if cands[n - 1] is not None:
            cur = ([0] * len(cands[n - 1]), [1] * len(cands[n - 1]))
            if n - 1 in need_g:
                g_rows[n - 1] = (cur[0], bytes(cur[1]))
        for i in range(n - 2, 0, -1):
            if cur is not None and cands[i] is not None:
                cur = _transition(cands[i], cands[i + 1], cur[0], cur[1])
            else:
                cur = None
            if cur is not None and i in need_g:
                g_rows[i] = (cur[0], bytes(cur[1]))

    # Forward sweep over the baseline: f_rows[i] = (values, counts) with
    # f_i(c) the optimal prefix cost when d_i = c.  A row exists only when
    # every epoch from 1 to i has candidates.
    f_rows: dict[int, tuple[list[int], bytes]] = {}
    if n >= 2:
        cur = None
        if cands[1] is not None:
            cur = ([0] * len(cands[1]), [1] * len(cands[1]))
            if 1 in need_f:
                f_rows[1] = (cur[0], bytes(cur[1]))
        for i in range(2, n):
            if cur is not None and cands[i] is not None:
                cur = _transition(cands[i], cands[i - 1], cur[0], cur[1])
            else:
                cur = None
            if cur is not None and i in need_f:
                f_rows[i] = (cur[0], bytes(cur[1]))

    # Baseline verdict (compact: no witnesses are needed for comparisons).
    if first_empty is not None:
        baseline: dict = {"status": "impossible",
                          "first_empty_index": first_empty}
        base_cost: int | None = None
    elif n == 1:
        baseline = {"status": "unique", "cost": 0}
        base_cost = 0
    else:
        base_cost, base_total = _min_with_count(*g_rows[1])
        baseline = {"status": "unique" if base_total == 1 else "ambiguous",
                    "cost": base_cost}

    results: list[dict] = []
    for index, w in enumerate(windows):
        a, b = w.start, w.end
        width = b - a + 1

        # 1) Replacement intervals must be non-empty (isolated rejection).
        bad = next((a + j for j in range(width) if w.lo[j] > w.hi[j]), None)
        if bad is not None:
            j = bad - a
            results.append(_scenario_error(index, DomainError(
                "INVALID_INTERVAL",
                f"empty interval at epoch {bad}: "
                f"lo = {w.lo[j]} > hi = {w.hi[j]}",
                index=bad)))
            continue

        # 2) Replacement candidates (isolated rejection when a window
        #    interval holds more than MAX_CANDIDATES of them).
        win: list = [None] * width
        win_empty: int | None = None
        overflow: DomainError | None = None
        for j in range(width):
            i = a + j
            target = (r[i] - r[i - 1]) % M
            start, count = _interval_candidate_span(M, target,
                                                    w.lo[j], w.hi[j])
            if count == 0:
                if win_empty is None:
                    win_empty = i
                continue
            if count > MAX_CANDIDATES:
                overflow = DomainError(
                    "TOO_MANY_CANDIDATES",
                    f"interval at epoch {i} holds {count} modular candidates "
                    f"(limit {MAX_CANDIDATES})",
                    index=i,
                    details={"count": count, "max": MAX_CANDIDATES})
                break
            win[j] = [start + k * M for k in range(count)]
        if overflow is not None:
            results.append(_scenario_error(index, overflow))
            continue

        # 3) Feasibility: an empty epoch outside the window survives the
        #    replacement, or the window itself produced one.
        outside: int | None = None
        if empty_epochs:
            if empty_epochs[0] < a:
                outside = empty_epochs[0]
            else:
                k = bisect_right(empty_epochs, b)
                if k < len(empty_epochs):
                    outside = empty_epochs[k]
        empties = [e for e in (outside, win_empty) if e is not None]
        if empties:
            results.append({"index": index, "status": "impossible",
                            "first_empty_index": min(empties)})
            continue

        # 4) Splice: seed from the cached prefix boundary, sweep the window,
        #    then combine with the cached suffix boundary.
        if a == 1:
            h_val = [0] * len(win[0])
            h_cnt = [1] * len(win[0])
        else:
            boundary = f_rows.get(a - 1)
            if boundary is None:  # pragma: no cover - feasible => cached
                raise RuntimeError(
                    "missing cached prefix boundary row for feasible window")
            h_val, h_cnt = _transition(win[0], cands[a - 1],
                                       boundary[0], boundary[1])
        for j in range(1, width):
            h_val, h_cnt = _transition(win[j], win[j - 1], h_val, h_cnt)

        if b == n - 1:
            optimal, total = _min_with_count(h_val, h_cnt)
        else:
            boundary = g_rows.get(b + 1)
            if boundary is None:  # pragma: no cover - feasible => cached
                raise RuntimeError(
                    "missing cached suffix boundary row for feasible window")
            t_val, t_cnt = _transition(win[width - 1], cands[b + 1],
                                       boundary[0], boundary[1])
            optimal = None
            total = 0
            for hv, hc, tv, tc in zip(h_val, h_cnt, t_val, t_cnt):
                v = hv + tv
                c = hc * tc
                if c > COUNT_CAP:
                    c = COUNT_CAP
                if optimal is None or v < optimal:
                    optimal = v
                    total = c
                elif v == optimal:
                    total += c
                    if total > COUNT_CAP:
                        total = COUNT_CAP

        entry = {"index": index,
                 "status": "unique" if total == 1 else "ambiguous",
                 "cost": optimal}
        if base_cost is not None:
            entry["delta"] = optimal - base_cost
        results.append(entry)

    return {"baseline": baseline, "scenarios": results}
