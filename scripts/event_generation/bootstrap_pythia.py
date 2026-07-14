#!/usr/bin/env python3
"""Check the pinned HADROS3 PYTHIA 8/HepMC3 toolchain."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hadros3.event_generation import backend_availability


if __name__ == "__main__":
    state = backend_availability()
    print(json.dumps(state, indent=2, sort_keys=True))
    raise SystemExit(0 if state["available"] else 1)
