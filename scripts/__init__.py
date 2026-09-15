"""Tooling package: brute-force oracle, HTTP acceptance, one-shot verify.

Acceptance traffic carries arbitrary-precision integers, so lift CPython's
default 4300-decimal-digit limit on int<->str conversion in this process as
well (mirrors ``app.__init__``).
"""
from __future__ import annotations

import sys

sys.set_int_max_str_digits(0)
