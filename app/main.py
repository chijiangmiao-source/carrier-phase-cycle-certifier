"""FastAPI application exposing the unwrapping solver over HTTP.

Endpoints
---------
* ``POST /solve``  — solve one instance; always answers deterministically.
* ``GET  /health`` — liveness probe used by the compose healthcheck.

All responses are plain JSON produced from dicts in a fixed key order, so a
repeated request body yields byte-identical response bytes.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .errors import DomainError
from .schemas import SolveRequest
from .solver import solve

app = FastAPI(
    title="carrier-phase-unwrapper",
    version="1.0.0",
    description="Determines whether the integer-cycle timeline of a "
                "modulo-M carrier-phase station is unique.",
)


@app.exception_handler(DomainError)
async def domain_error_handler(_request, exc: DomainError) -> JSONResponse:
    return JSONResponse(status_code=422, content=exc.payload())


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request, exc: RequestValidationError
                                   ) -> JSONResponse:
    details = [
        {
            "loc": [str(part) for part in err.get("loc", ())],
            "type": err.get("type", ""),
            "msg": err.get("msg", ""),
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "request body failed schema validation",
                "details": details,
            }
        },
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/solve")
def solve_endpoint(request: SolveRequest) -> dict:
    return solve(request.M, request.n, request.r, request.x0,
                 request.lo, request.hi)
