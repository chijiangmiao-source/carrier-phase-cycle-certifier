"""Carrier-phase integer-cycle unwrapping service.

The start value and interval endpoints are arbitrary-precision integers, so
lift CPython's default 4300-decimal-digit limit on int<->str conversion
(CVE-2020-10735 mitigation): JSON request parsing, JSON response
serialisation and error-message formatting must all handle integers of any
size exactly.
"""
from __future__ import annotations

import sys

sys.set_int_max_str_digits(0)
