"""One-shot acceptance service.

Runs the pytest suite first, then performs black-box HTTP acceptance checks
against the live API (known answers, brute-force cross-checks, structured
errors, byte-level determinism).  Exits 0 only when everything passes, so it
can be used as a one-shot ``verify`` compose service.
"""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    print("=== verify: unit tests (pytest) ===", flush=True)
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"])
    if proc.returncode != 0:
        print("=== verify: FAILED (unit tests) ===", flush=True)
        return proc.returncode

    base = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
    print(f"=== verify: HTTP acceptance against {base} ===", flush=True)
    from scripts.acceptance import main as acceptance_main

    rc = acceptance_main()
    print(f"=== verify: {'OK' if rc == 0 else 'FAILED'} ===", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
