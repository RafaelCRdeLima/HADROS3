#!/usr/bin/env python3
"""Install, inspect, build, and smoke-test the pinned H3-W11 Geant4 backend."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hadros3.environment import env_file_values, geant4_prefix  # noqa: E402

PREFIX = geant4_prefix()
# `make setup` records which environment this checkout uses; default to the
# canonical name from environment.yml when nothing has been configured yet.
ENV_NAME = os.environ.get("HADROS3_CONDA_ENV") or env_file_values().get("HADROS3_CONDA_ENV") or "hadros3"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["install", "inspect", "build", "smoke"])
    args = parser.parse_args()
    if args.action == "install":
        run(["micromamba", "install", "-n", ENV_NAME, "-c", "conda-forge", "geant4=11.4.2", "cmake", "ninja", "-y"])
    elif args.action == "build":
        run(["make", "bin/hadros3_geant4_transport"])
    elif args.action == "smoke":
        run(["make", "geant4-validate"])
    else:
        from hadros3.geant4_transport import backend_availability
        availability = backend_availability(PREFIX)
        print(json.dumps(availability, indent=2, sort_keys=True))
        return 0 if availability["available"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
