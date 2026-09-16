"""Request schemas for the solve and analyze-windows endpoints.

Strict mode is used throughout: JSON floats, booleans and numeric strings are
rejected instead of being silently coerced, which keeps the whole pipeline
free of floating-point values.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Upper bound on the total number of replaced points over all scenarios of
#: one analyze-windows request (mirrors ``app.analyze``).
MAX_TOTAL_REPLACEMENTS = 20_000


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


class WindowScenario(BaseModel):
    """One what-if scenario: replace the increment intervals of the closed
    epoch window ``start .. end`` (increment indices, 1-based).  ``lo``/``hi``
    hold the replacement bounds, exactly ``end - start + 1`` values each.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    start: Annotated[int, Field(ge=1,
                                description="first replaced epoch "
                                            "(increment index)")]
    end: Annotated[int, Field(ge=1,
                              description="last replaced epoch (inclusive)")]
    lo: Annotated[list[int], Field(
        description="replacement lower bounds, end-start+1 values")]
    hi: Annotated[list[int], Field(
        description="replacement upper bounds, end-start+1 values")]

    @model_validator(mode="after")
    def _window_consistent(self) -> "WindowScenario":
        if self.end < self.start:
            raise ValueError(
                f"end ({self.end}) must be >= start ({self.start})")
        want = self.end - self.start + 1
        if len(self.lo) != want or len(self.hi) != want:
            raise ValueError(
                f"lo/hi must hold exactly end-start+1 = {want} values")
        return self


class AnalyzeWindowsRequest(SolveRequest):
    """One baseline instance plus independent windowed what-if scenarios,
    evaluated and reported in input order."""

    scenarios: Annotated[list[WindowScenario], Field(
        description="independent window replacements; the total number of "
                    "replaced points must not exceed 20000")]

    @model_validator(mode="after")
    def _scenarios_in_range(self) -> "AnalyzeWindowsRequest":
        total = 0
        for idx, sc in enumerate(self.scenarios):
            if sc.end > self.n - 1:
                raise ValueError(
                    f"scenarios[{idx}]: end = {sc.end} exceeds "
                    f"n-1 = {self.n - 1}")
            total += sc.end - sc.start + 1
        if total > MAX_TOTAL_REPLACEMENTS:
            raise ValueError(
                f"scenarios replace {total} points in total, limit is "
                f"{MAX_TOTAL_REPLACEMENTS}")
        return self
