"""HTTP contract for POST /analyze-windows: responses, isolated scenario
errors, schema validation and byte-level determinism."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALIDATION_ERROR = "VALIDATION_ERROR"

BASE = {"M": 5, "n": 3, "r": [0, 1, 2], "x0": 0, "lo": [1, 6], "hi": [1, 6]}

MIXED = dict(BASE, scenarios=[
    {"start": 1, "end": 1, "lo": [0], "hi": [10]},
    {"start": 2, "end": 2, "lo": [0], "hi": [10]},
    {"start": 1, "end": 2, "lo": [0, 0], "hi": [10, 10]},
    {"start": 1, "end": 1, "lo": [3], "hi": [4]},
    {"start": 2, "end": 2, "lo": [8], "hi": [3]},
    {"start": 1, "end": 1, "lo": [0], "hi": [1000]},
])


def _post(payload):
    return client.post("/analyze-windows", json=payload)


def _error_code(resp):
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert set(body) == {"error"}
    assert {"code", "message"} <= set(body["error"])
    return body["error"]["code"]


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_known_mixed_response():
    resp = _post(MIXED)
    assert resp.status_code == 200
    body = resp.json()
    assert body["baseline"] == {"status": "unique", "cost": 5}
    scen = body["scenarios"]
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


def test_empty_scenarios_list():
    resp = _post(dict(BASE, scenarios=[]))
    assert resp.status_code == 200
    assert resp.json() == {"baseline": {"status": "unique", "cost": 5},
                           "scenarios": []}


def test_single_epoch_baseline():
    resp = _post({"M": 5, "n": 1, "r": [3], "x0": -7, "lo": [], "hi": [],
                  "scenarios": []})
    assert resp.status_code == 200
    assert resp.json() == {"baseline": {"status": "unique", "cost": 0},
                           "scenarios": []}


def test_baseline_impossible_no_delta_keys():
    resp = _post({"M": 5, "n": 3, "r": [0, 1, 2], "x0": 0,
                  "lo": [0, 3], "hi": [1, 4],
                  "scenarios": [{"start": 2, "end": 2, "lo": [1], "hi": [1]},
                                {"start": 1, "end": 1, "lo": [1], "hi": [1]}]})
    assert resp.status_code == 200
    assert resp.json() == {
        "baseline": {"status": "impossible", "first_empty_index": 2},
        "scenarios": [
            {"index": 0, "status": "unique", "cost": 0},
            {"index": 1, "status": "impossible", "first_empty_index": 2},
        ],
    }


def test_repeated_requests_are_byte_identical():
    first, second = _post(MIXED), _post(MIXED)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    bad = dict(BASE, scenarios=[{"start": 2, "end": 1, "lo": [], "hi": []}])
    assert _post(bad).content == _post(bad).content


# ---------------------------------------------------------------------------
# Baseline domain errors: whole request fails like /solve
# ---------------------------------------------------------------------------


def test_baseline_domain_errors():
    one_window = [{"start": 1, "end": 1, "lo": [1], "hi": [1]}]
    cases = [
        ({"M": 5, "n": 1, "r": [7], "x0": 0, "lo": [], "hi": [],
          "scenarios": []}, "INVALID_REMAINDER"),
        ({"M": 5, "n": 1, "r": [1], "x0": 0, "lo": [], "hi": [],
          "scenarios": []}, "INVALID_START"),
        ({"M": 5, "n": 2, "r": [0, 1], "x0": 0, "lo": [4], "hi": [1],
          "scenarios": one_window}, "INVALID_INTERVAL"),
        ({"M": 5, "n": 2, "r": [0, 1], "x0": 0, "lo": [0], "hi": [400],
          "scenarios": one_window}, "TOO_MANY_CANDIDATES"),
        ({"M": 5, "n": 2, "r": [0], "x0": 0, "lo": [0], "hi": [1],
          "scenarios": one_window}, "LENGTH_MISMATCH"),
    ]
    for payload, code in cases:
        assert _error_code(_post(payload)) == code, payload


def test_baseline_error_beats_scenario_error():
    # The baseline itself is invalid (lo > hi) and the scenario is too:
    # the whole request is rejected with the baseline's domain error.
    resp = _post({"M": 5, "n": 2, "r": [0, 1], "x0": 0, "lo": [4], "hi": [1],
                  "scenarios": [{"start": 1, "end": 1, "lo": [2],
                                 "hi": [0]}]})
    assert _error_code(resp) == "INVALID_INTERVAL"


# ---------------------------------------------------------------------------
# Schema errors (window structure, types, totals)
# ---------------------------------------------------------------------------


def test_schema_error_window_structure():
    valid = {"start": 1, "end": 1, "lo": [1], "hi": [1]}
    bad_windows = [
        {"start": 0, "end": 1, "lo": [1, 1], "hi": [1, 1]},   # start < 1
        {"start": 2, "end": 1, "lo": [], "hi": []},           # end < start
        {"start": 1, "end": 2, "lo": [1], "hi": [1, 1]},      # lo too short
        {"start": 1, "end": 2, "lo": [1, 1], "hi": [1]},      # hi too short
        {"start": 1, "end": 3, "lo": [1, 1, 1], "hi": [1, 1, 1]},  # end > n-1
        dict(valid, extra=1),                                  # unknown field
    ]
    for window in bad_windows:
        resp = _post(dict(BASE, scenarios=[window]))
        assert _error_code(resp) == VALIDATION_ERROR, window


def test_schema_error_scenario_types():
    bad_windows = [
        {"start": 1.0, "end": 1, "lo": [1], "hi": [1]},   # float
        {"start": 1, "end": True, "lo": [1], "hi": [1]},  # bool
        {"start": 1, "end": 1, "lo": ["1"], "hi": [1]},   # numeric string
        {"start": 1, "end": 1, "lo": [1.5], "hi": [1]},   # float bound
    ]
    for window in bad_windows:
        resp = _post(dict(BASE, scenarios=[window]))
        assert _error_code(resp) == VALIDATION_ERROR, window


def test_schema_error_missing_and_wrong_scenarios_field():
    assert _error_code(_post(BASE)) == VALIDATION_ERROR  # scenarios missing
    assert _error_code(_post(dict(BASE, scenarios={}))) == VALIDATION_ERROR
    assert _error_code(_post(dict(BASE, scenarios=[1]))) == VALIDATION_ERROR


def test_schema_error_total_replacement_points_exceeded():
    n = 20_000
    payload = {
        "M": 5, "n": n, "r": [0] * n, "x0": 0,
        "lo": [0] * (n - 1), "hi": [0] * (n - 1),
        "scenarios": [
            {"start": 1, "end": n - 1, "lo": [0] * (n - 1),
             "hi": [0] * (n - 1)},          # 19999 points
            {"start": 1, "end": 2, "lo": [0, 0], "hi": [0, 0]},  # +2 = 20001
        ],
    }
    assert _error_code(_post(payload)) == VALIDATION_ERROR


def test_method_not_allowed():
    assert client.get("/analyze-windows").status_code == 405


# ---------------------------------------------------------------------------
# Scenario error isolation over HTTP
# ---------------------------------------------------------------------------


def test_scenario_errors_are_isolated_and_ordered():
    resp = _post({"M": 5, "n": 4, "r": [0, 1, 2, 3], "x0": 0,
                  "lo": [1, 1, 1], "hi": [1, 1, 1],
                  "scenarios": [
                      {"start": 1, "end": 1, "lo": [1], "hi": [1]},
                      {"start": 2, "end": 3, "lo": [5, 0], "hi": [2, 0]},
                      {"start": 1, "end": 1, "lo": [0], "hi": [1000]},
                      {"start": 3, "end": 3, "lo": [1], "hi": [1]},
                  ]})
    assert resp.status_code == 200
    scen = resp.json()["scenarios"]
    assert [s["index"] for s in scen] == [0, 1, 2, 3]
    assert scen[0] == {"index": 0, "status": "unique", "cost": 0, "delta": 0}
    assert scen[1]["status"] == "error"
    assert scen[1]["error"]["code"] == "INVALID_INTERVAL"
    assert scen[1]["error"]["index"] == 2
    assert scen[2]["status"] == "error"
    assert scen[2]["error"]["code"] == "TOO_MANY_CANDIDATES"
    assert scen[2]["error"]["index"] == 1
    assert scen[3] == {"index": 3, "status": "unique", "cost": 0, "delta": 0}


# ---------------------------------------------------------------------------
# Arbitrary-precision integers (> 4300 decimal digits)
# ---------------------------------------------------------------------------


def test_huge_integers_in_windows_solved_exactly():
    big = 10**5000
    resp = _post({"M": 10**9, "n": 2, "r": [0, 123456], "x0": big,
                  "lo": [big + 123456], "hi": [big + 123456],
                  "scenarios": [
                      {"start": 1, "end": 1, "lo": [big + 123456],
                       "hi": [big + 123456 + 10**9]},
                  ]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["baseline"] == {"status": "unique", "cost": 0}
    # two candidates (big+123456, big+123456+M) -> ambiguous, cost 0
    assert body["scenarios"][0] == {"index": 0, "status": "ambiguous",
                                    "cost": 0, "delta": 0}
