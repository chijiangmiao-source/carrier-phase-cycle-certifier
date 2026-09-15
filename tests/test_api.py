"""HTTP API tests: contracts, structured errors, byte-level determinism."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

UNIQUE_CASE = {"M": 5, "n": 3, "r": [0, 1, 2], "x0": 0, "lo": [1, 6],
               "hi": [1, 6]}
AMBIGUOUS_CASE = {"M": 5, "n": 3, "r": [0, 1, 2], "x0": 0, "lo": [0, 0],
                  "hi": [10, 10]}
IMPOSSIBLE_CASE = {"M": 5, "n": 3, "r": [0, 1, 2], "x0": 0, "lo": [0, 3],
                   "hi": [1, 4]}


def _post(payload):
    return client.post("/solve", json=payload)


def _error_code(resp):
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert set(body) == {"error"}
    assert {"code", "message"} <= set(body["error"])
    return body["error"]["code"]


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_unique_response():
    resp = _post(UNIQUE_CASE)
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "unique",
        "x": [0, 1, 7],
        "cost": 5,
        "increments": [1, 6],
    }


def test_ambiguous_response():
    resp = _post(AMBIGUOUS_CASE)
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ambiguous",
        "cost": 0,
        "witnesses": [
            {"x": [0, 1, 2], "increments": [1, 1], "cost": 0},
            {"x": [0, 6, 12], "increments": [6, 6], "cost": 0},
        ],
    }


def test_impossible_response():
    resp = _post(IMPOSSIBLE_CASE)
    assert resp.status_code == 200
    assert resp.json() == {"status": "impossible", "first_empty_index": 2}


def test_single_epoch_response():
    resp = _post({"M": 5, "n": 1, "r": [3], "x0": -7, "lo": [], "hi": []})
    assert resp.status_code == 200
    assert resp.json() == {"status": "unique", "x": [-7], "cost": 0,
                           "increments": []}


def test_repeated_requests_are_byte_identical():
    first = _post(AMBIGUOUS_CASE)
    second = _post(AMBIGUOUS_CASE)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    # error responses are deterministic as well
    bad = {"M": 5, "n": 1, "r": [9], "x0": 0, "lo": [], "hi": []}
    assert _post(bad).content == _post(bad).content


# ---------------------------------------------------------------------------
# Domain errors (HTTP 422, structured envelope)
# ---------------------------------------------------------------------------


def test_error_invalid_remainder():
    assert _error_code(_post({"M": 5, "n": 1, "r": [5], "x0": 0,
                              "lo": [], "hi": []})) == "INVALID_REMAINDER"
    assert _error_code(_post({"M": 5, "n": 1, "r": [-1], "x0": 0,
                              "lo": [], "hi": []})) == "INVALID_REMAINDER"


def test_error_invalid_start():
    assert _error_code(_post({"M": 5, "n": 1, "r": [1], "x0": 0,
                              "lo": [], "hi": []})) == "INVALID_START"


def test_error_invalid_interval():
    assert _error_code(_post({"M": 5, "n": 2, "r": [0, 1], "x0": 0,
                              "lo": [3], "hi": [1]})) == "INVALID_INTERVAL"


def test_error_length_mismatch():
    assert _error_code(_post({"M": 5, "n": 3, "r": [0, 1], "x0": 0,
                              "lo": [0, 0], "hi": [1, 1]})) == "LENGTH_MISMATCH"
    assert _error_code(_post({"M": 5, "n": 2, "r": [0, 1], "x0": 0,
                              "lo": [], "hi": []})) == "LENGTH_MISMATCH"


def test_error_too_many_candidates():
    assert _error_code(_post({"M": 5, "n": 2, "r": [0, 1], "x0": 0,
                              "lo": [0], "hi": [1000]})) == "TOO_MANY_CANDIDATES"


def test_error_body_is_structured_and_deterministic():
    resp = _post({"M": 5, "n": 2, "r": [0, 1], "x0": 0, "lo": [3], "hi": [1]})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "INVALID_INTERVAL"
    assert body["error"]["index"] == 1
    assert isinstance(body["error"]["message"], str)


# ---------------------------------------------------------------------------
# Schema errors (types, ranges, unknown fields)
# ---------------------------------------------------------------------------


def test_schema_error_out_of_range_parameters():
    assert _error_code(_post({"M": 1, "n": 1, "r": [0], "x0": 0,
                              "lo": [], "hi": []})) == "VALIDATION_ERROR"
    assert _error_code(_post({"M": 10**9 + 1, "n": 1, "r": [0], "x0": 0,
                              "lo": [], "hi": []})) == "VALIDATION_ERROR"
    assert _error_code(_post({"M": 5, "n": 0, "r": [], "x0": 0,
                              "lo": [], "hi": []})) == "VALIDATION_ERROR"
    assert _error_code(_post({"M": 5, "n": 20001, "r": [], "x0": 0,
                              "lo": [], "hi": []})) == "VALIDATION_ERROR"


def test_schema_error_floats_rejected():
    assert _error_code(_post({"M": 5.0, "n": 1, "r": [0], "x0": 0,
                              "lo": [], "hi": []})) == "VALIDATION_ERROR"
    assert _error_code(_post({"M": 5, "n": 1, "r": [0.5], "x0": 0,
                              "lo": [], "hi": []})) == "VALIDATION_ERROR"
    assert _error_code(_post({"M": 5, "n": 2, "r": [0, 1], "x0": 0,
                              "lo": [1.0], "hi": [1]})) == "VALIDATION_ERROR"


def test_schema_error_wrong_types_rejected():
    assert _error_code(_post({"M": "5", "n": 1, "r": [0], "x0": 0,
                              "lo": [], "hi": []})) == "VALIDATION_ERROR"
    assert _error_code(_post({"M": True, "n": 1, "r": [0], "x0": 0,
                              "lo": [], "hi": []})) == "VALIDATION_ERROR"
    assert _error_code(_post({"M": 5, "n": 1, "r": [0], "x0": None,
                              "lo": [], "hi": []})) == "VALIDATION_ERROR"


def test_schema_error_unknown_and_missing_fields():
    payload = dict(UNIQUE_CASE)
    payload["extra"] = 1
    assert _error_code(_post(payload)) == "VALIDATION_ERROR"
    payload = dict(UNIQUE_CASE)
    del payload["hi"]
    assert _error_code(_post(payload)) == "VALIDATION_ERROR"


def test_schema_error_malformed_json():
    resp = client.post("/solve", content=b'{"M": 5,',
                       headers={"content-type": "application/json"})
    assert _error_code(resp) == "VALIDATION_ERROR"


def test_schema_error_non_finite_numbers_rejected():
    resp = client.post(
        "/solve",
        content=b'{"M": NaN, "n": 1, "r": [0], "x0": 0, "lo": [], "hi": []}',
        headers={"content-type": "application/json"},
    )
    assert _error_code(resp) == "VALIDATION_ERROR"


def test_method_not_allowed():
    assert client.get("/solve").status_code == 405


# ---------------------------------------------------------------------------
# Arbitrary-precision integers (> 4300 decimal digits)
# ---------------------------------------------------------------------------


def test_huge_integers_solved_exactly_over_http():
    big = 10**5000  # 5001 digits, beyond CPython's 4300-digit str limit
    resp = _post({"M": 10**9, "n": 2, "r": [0, 123456], "x0": big,
                  "lo": [big + 123456], "hi": [big + 123456]})
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "unique",
        "x": [big, 2 * big + 123456],
        "cost": 0,
        "increments": [big + 123456],
    }


def test_huge_integer_domain_error_is_structured_not_500():
    big = 10**5000
    resp = _post({"M": 10**9, "n": 1, "r": [0], "x0": big + 1,
                  "lo": [], "hi": []})
    assert _error_code(resp) == "INVALID_START"


def test_huge_interval_endpoint_domain_error_is_structured():
    big = 10**5000
    resp = _post({"M": 10**9, "n": 2, "r": [0, 1], "x0": 0,
                  "lo": [big], "hi": [big - 1]})
    assert _error_code(resp) == "INVALID_INTERVAL"
