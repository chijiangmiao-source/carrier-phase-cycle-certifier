"""Request schema for the solve endpoint.

Strict mode is used throughout: JSON floats, booleans and numeric strings are
rejected instead of being silently coerced, which keeps the whole pipeline
free of floating-point values.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class SolveRequest(BaseModel):
    """One carrier-phase unwrapping instance.

    ``lo``/``hi`` hold the closed increment intervals: ``lo[k]``/``hi[k]`` are
    the bounds for the increment ``d_i = x_i - x_{i-1}`` at epoch ``i = k+1``,
    so both arrays must have length ``n - 1``.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    M: Annotated[int, Field(ge=2, le=1_000_000_000,
                            description="carrier-phase modulus")]
    n: Annotated[int, Field(ge=1, le=20_000,
                            description="number of epochs")]
    r: Annotated[list[int], Field(
        description="phase remainders, n values with 0 <= r_i < M")]
    x0: Annotated[int, Field(
        description="known start value, must satisfy x0 == r[0] (mod M); "
                    "may be negative")]
    lo: Annotated[list[int], Field(
        description="lower increment bounds, n-1 values")]
    hi: Annotated[list[int], Field(
        description="upper increment bounds, n-1 values")]
