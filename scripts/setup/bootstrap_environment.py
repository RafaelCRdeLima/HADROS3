#!/usr/bin/env python3
"""Resolve (or create) a working HADROS3 environment for THIS machine.

The script is deliberately written against the Python standard library only:
it is the bootstrap step, so it must run with whatever ``python3`` a freshly
cloned checkout finds in ``PATH``.

It resolves, in this order:

1. an existing conda/micromamba environment (``dis``, then ``hadros3``, then
   ``$CONDA_PREFIX``);
2. a new conda/micromamba environment created from ``environment.yml``
   (skipped with ``--light`` or when no conda-like manager is installed);
3. a local ``.venv`` with the pure-Python layer from ``requirements-dev.txt``.

Whatever it resolves is written to ``.hadros3-env.mk``, which the Makefile
includes.  Nothing else in the repository hardcodes an interpreter or a
library prefix.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".hadros3-env.mk"
ENVIRONMENT_YML = ROOT / "environment.yml"
REQUIREMENTS = ROOT / "requirements-dev.txt"
VENV_DIR = ROOT / ".venv"

DEFAULT_ENV_NAME = os.environ.get("HADROS3_CONDA_ENV", "dis")
KNOWN_ENV_NAMES = ["dis", "hadros3"]
MANAGERS = ["micromamba", "mamba", "conda"]


def info(message: str) -> None:
    print("[setup] " + message, flush=True)


def run(command: list[str], **kwargs: object) -> int:
    info("$ " + " ".join(command))
    return subprocess.call(command, **kwargs)  # type: ignore[arg-type]


def find_manager() -> str | None:
    """Locate micromamba/mamba/conda, including common non-PATH install spots."""
    for name in MANAGERS:
        found = shutil.which(name)
        if found:
            return found
    extra = [
        ROOT / ".tools" / "bin" / "micromamba",
        Path.home() / ".local" / "bin" / "micromamba",
        Path.home() / "micromamba" / "bin" / "micromamba",
        Path.home() / "bin" / "micromamba",
        Path.home() / "miniforge3" / "bin" / "conda",
        Path.home() / "miniconda3" / "bin" / "conda",
        Path.home() / "anaconda3" / "bin" / "conda",
    ]
    for candidate in extra:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def list_envs(manager: str) -> dict[str, Path]:
    """Map environment name -> prefix, tolerating every manager's quirks."""
    envs: dict[str, Path] = {}
    try:
        raw = subprocess.check_output(
            [manager, "env", "list", "--json"], stderr=subprocess.DEVNULL, text=True, timeout=120
        )
        for prefix in json.loads(raw).get("envs", []):
            path = Path(prefix)
            envs.setdefault(path.name, path)
    except Exception:
        pass
    roots = [
        os.environ.get("MAMBA_ROOT_PREFIX"),
        os.environ.get("CONDA_ROOT"),
        str(Path.home() / "micromamba"),
        str(Path.home() / "miniforge3"),
        str(Path.home() / "miniconda3"),
        str(Path.home() / "anaconda3"),
    ]
    for root in roots:
        if not root:
            continue
        env_root = Path(root) / "envs"
        if not env_root.is_dir():
            continue
        for path in sorted(env_root.iterdir()):
            if (path / "bin" / "python").exists():
                envs.setdefault(path.name, path)
    return envs


def resolve_existing_env(manager: str, requested: str) -> tuple[str, Path] | None:
    envs = list_envs(manager)
    for name in [requested] + [n for n in KNOWN_ENV_NAMES if n != requested]:
        prefix = envs.get(name)
        if prefix is not None and (prefix / "bin" / "python").exists():
            return name, prefix
    active = os.environ.get("CONDA_PREFIX")
    if active and (Path(active) / "bin" / "python").exists():
        return Path(active).name, Path(active)
    return None


def create_env(manager: str, name: str) -> tuple[str, Path] | None:
    if not ENVIRONMENT_YML.exists():
        info("environment.yml is missing; cannot create a conda environment.")
        return None
    info("creating conda environment '%s' from environment.yml (this takes a few minutes)" % name)
    tool = Path(manager).name
    if tool == "micromamba":
        command = [manager, "create", "-y", "-n", name, "-f", str(ENVIRONMENT_YML)]
    else:
        command = [manager, "env", "create", "-n", name, "-f", str(ENVIRONMENT_YML)]
    if run(command) != 0:
        info("environment creation failed; falling back to the pip/venv layer.")
        return None
    return resolve_existing_env(manager, name)


def create_venv() -> Path | None:
    python = VENV_DIR / "bin" / "python"
    if not python.exists():
        info("creating local virtual environment at .venv")
        if run([sys.executable, "-m", "venv", str(VENV_DIR)]) != 0:
            info("could not create .venv (is python3-venv installed?)")
            return None
    if run([str(python), "-m", "pip", "install", "--upgrade", "pip"]) != 0:
        info("warning: could not upgrade pip inside .venv")
    if run([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)]) != 0:
        info("installing the Python dependencies failed.")
        return None
    return python


def conda_runner(manager: str, name: str, prefix: Path) -> str:
    """Prefer `<manager> run -n <env>`: it applies the activation scripts that
    export the Geant4 dataset variables. Fall back to the raw interpreter."""
    tool = Path(manager).name
    if tool in {"micromamba", "mamba", "conda"}:
        return "%s run -n %s python" % (manager, name)
    return str(prefix / "bin" / "python")


def detect_pythia8(prefix: Path | None) -> Path | None:
    if prefix is None:
        return None
    return prefix if (prefix / "include" / "Pythia8" / "Pythia.h").exists() else None


def detect_geant4(prefix: Path | None) -> Path | None:
    if prefix is None:
        return None
    for libdir in ("lib", "lib64"):
        if (prefix / libdir / "cmake" / "Geant4").is_dir():
            return prefix
    return None


def detect_conda_cxx(prefix: Path | None) -> str | None:
    if prefix is None:
        return None
    matches = sorted((prefix / "bin").glob("*-linux-gnu-c++")) if (prefix / "bin").is_dir() else []
    return str(matches[0]) if matches else None


def write_env_file(lines: list[str]) -> None:
    body = "\n".join(
        [
            "# Generated by `make setup` / `make install` -- do not edit by hand, do not commit.",
            "# Regenerate at any time with: make setup",
            "",
        ]
        + lines
        + [""]
    )
    ENV_FILE.write_text(body, encoding="utf-8")
    info("wrote %s" % ENV_FILE.relative_to(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Configure the HADROS3 build environment.")
    parser.add_argument(
        "--light",
        action="store_true",
        help="Skip conda entirely and configure only the pure-Python layer in .venv.",
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="Only detect what already exists; never create an environment.",
    )
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME, help="Conda environment name (default: dis).")
    args = parser.parse_args()

    lines: list[str] = []
    kind = "system"
    python_cmd = None
    prefix: Path | None = None

    manager = None if args.light else find_manager()
    if manager is not None:
        info("found environment manager: %s" % manager)
        resolved = resolve_existing_env(manager, args.env_name)
        if resolved is None and not args.no_create:
            resolved = create_env(manager, args.env_name)
        if resolved is not None:
            name, prefix = resolved
            python_cmd = conda_runner(manager, name, prefix)
            kind = "conda"
            info("using conda environment '%s' at %s" % (name, prefix))
            lines.append("HADROS3_CONDA_ENV := %s" % name)
            lines.append("HADROS3_CONDA_PREFIX := %s" % prefix)
    elif not args.light:
        info("no micromamba/mamba/conda found; using the pip/venv layer.")
        info("PYTHIA 8 and Geant4 are not installable from PyPI and will stay unavailable.")

    if python_cmd is None:
        if args.no_create:
            existing = VENV_DIR / "bin" / "python"
            venv_python = existing if existing.exists() else None
        else:
            venv_python = create_venv()
        if venv_python is not None:
            python_cmd = str(venv_python)
            kind = "venv"
        else:
            python_cmd = sys.executable
            kind = "system"
            info("falling back to the current interpreter: %s" % sys.executable)

    lines.insert(0, "HADROS3_ENV_KIND := %s" % kind)
    lines.append("PYTHON := %s" % python_cmd)
    lines.append("PIP := $(PYTHON) -m pip")

    pythia8 = detect_pythia8(prefix)
    if pythia8 is not None:
        lines.append("PYTHIA8_PREFIX := %s" % pythia8)
        conda_cxx = detect_conda_cxx(pythia8)
        if conda_cxx is not None:
            lines.append("PYTHIA8_CXX := %s" % conda_cxx)

    geant4 = detect_geant4(prefix)
    if geant4 is not None:
        lines.append("GEANT4_PREFIX := %s" % geant4)
        cmake = geant4 / "bin" / "cmake"
        lines.append("GEANT4_CMAKE := %s" % (cmake if cmake.exists() else shutil.which("cmake") or "cmake"))

    nvcc = shutil.which("nvcc")
    if nvcc is None and prefix is not None and (prefix / "bin" / "nvcc").exists():
        nvcc = str(prefix / "bin" / "nvcc")
    if nvcc is not None:
        lines.append("NVCC := %s" % nvcc)

    write_env_file(lines)

    print()
    info("environment kind: %s" % kind)
    info("python: %s" % python_cmd)
    info("PYTHIA 8: %s" % (pythia8 or "not available"))
    info("Geant4:   %s" % (geant4 or "not available"))
    info("nvcc:     %s" % (nvcc or "not available"))
    print()
    info("next step: make doctor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
