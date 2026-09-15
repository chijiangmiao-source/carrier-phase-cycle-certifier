"""Structured domain errors.

Every semantically invalid request is reported with a stable machine-readable
code so clients can react programmatically.  All domain errors map to HTTP 422
and are rendered as ``{"error": {...}}`` envelopes by the API layer.
"""
from __future__ import annotations


class DomainError(Exception):
    """A semantically invalid solve request.

    Attributes:
        code: stable machine-readable error code.
        message: human-readable explanation (deterministic for a given input).
        index: optional time index the error refers to.  Remainders use the
            epoch index ``0 .. n-1``; increment intervals use the increment
            index ``1 .. n-1`` (``lo[i-1]``/``hi[i-1]`` in the request arrays).
        details: optional extra structured payload.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        index: int | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.index = index
        self.details = details or {}

    def payload(self) -> dict:
        error: dict = {"code": self.code, "message": self.message}
        if self.index is not None:
            error["index"] = self.index
        if self.details:
            error["details"] = self.details
        return {"error": error}
