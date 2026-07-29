#!/usr/bin/env python3
"""Report what HADROS3 can and cannot do on THIS machine, and how to fix it.

Standard library only: `make doctor` has to work before anything is installed.

    make doctor            human readable capability table
    make doctor ARGS=--json    machine readable report
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".hadros3-env.mk"

APT = "sudo apt install"

REQUIRED_MODULES = ["numpy", "matplotlib", "jsonschema", "pytest"]

CORE_BINARIES = [
    "hadros3_forward_geodesics",
    "hadros3_dis_sampler",
    "hadros3_observer_bridge",
    "hadros3_powheg_driver",
]
OPTIONAL_BINARIES = [
    "hadros3_event_generator",
    "hadros3_geant4_transport",
    "hadros3_geodesic_preview_cuda",
]

OK = "OK"
MISSING = "MISSING"
ABSENT = "ABSENT"

PROBE_TEMPLATE = """
import json, sys

modules = {}
for name in %r:
    try:
        __import__(name)
        modules[name] = True
    except Exception:
        modules[name] = False
print(json.dumps({"version": sys.version.split()[0], "executable": sys.executable, "modules": modules}))
"""


class Report:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []
        self.required_failures = 0

    def add(self, group: str, item: str, status: str, detail: str = "", hint: str = "", required: bool = False) -> None:
        self.rows.append(
            {"group": group, "item": item, "status": status, "detail": detail, "hint": hint, "required": str(required)}
        )
        if required and status != OK:
            self.required_failures += 1

    def render(self) -> str:
        width = max(len(row["item"]) for row in self.rows) + 2
        out: list[str] = []
        group = None
        for row in self.rows:
            if row["group"] != group:
                group = row["group"]
                out.append("")
                out.append(group)
                out.append("-" * len(group))
            mark = {OK: "  ok  ", MISSING: " MISS ", ABSENT: "  --  "}[row["status"]]
            line = "[%s] %s%s" % (mark, row["item"].ljust(width), row["detail"])
            out.append(line.rstrip())
            if row["hint"] and row["status"] != OK:
                out.append(" " * 9 + "-> " + row["hint"])
        return "\n".join(out)


def read_makefile_defaults() -> dict[str, str]:
    """Before `make setup` has run, ask the Makefile what it would resolve."""
    values: dict[str, str] = {}
    raw = command_output(["make", "--no-print-directory", "-C", str(ROOT), "print-env"], timeout=120)
    if not raw:
        return values
    for line in raw.splitlines():
        key, _, value = line.partition("=")
        if value.strip():
            values[key.strip()] = value.strip()
    return values


def read_env_file() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_FILE.exists():
        return read_makefile_defaults()
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":=" not in line:
            continue
        key, _, value = line.partition(":=")
        values[key.strip()] = value.strip()
    return values


def command_output(command: list[str], timeout: int = 60) -> str | None:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=timeout).strip()
    except Exception:
        return None


def python_command(env: dict[str, str]) -> list[str] | None:
    raw = env.get("PYTHON")
    if raw:
        raw = raw.replace("$(PYTHON)", "").strip()
        parts = shlex.split(raw)
        if parts:
            return parts
    venv = ROOT / ".venv" / "bin" / "python"
    if venv.exists():
        return [str(venv)]
    found = shutil.which("python3")
    return [found] if found else None


def check_platform(report: Report) -> None:
    report.add("System", "os", OK, "%s %s" % (platform.system(), platform.release()))
    libc = platform.libc_ver()
    detail = " ".join(part for part in libc if part) or "unknown"
    report.add("System", "libc", OK, detail)
    report.add("System", "cpu", OK, platform.machine())


def check_toolchain(report: Report, env: dict[str, str]) -> None:
    # cmake/ninja usually live inside the conda environment rather than in PATH.
    prefixes = [env.get("HADROS3_CONDA_PREFIX"), env.get("GEANT4_PREFIX"), env.get("PYTHIA8_PREFIX")]
    extra_bins = [Path(prefix) / "bin" for prefix in prefixes if prefix]

    for tool, hint, required in [
        ("make", "%s make" % APT, True),
        ("g++", "%s build-essential" % APT, True),
        ("pkg-config", "%s pkg-config" % APT, False),
        ("cmake", "needed only for the Geant4 backend", False),
        ("ninja", "needed only for the Geant4 backend", False),
    ]:
        path = shutil.which(tool)
        if path is None:
            for bindir in extra_bins:
                candidate = bindir / tool
                if candidate.exists():
                    path = str(candidate)
                    break
        if path:
            version = (command_output([path, "--version"]) or "").splitlines()
            report.add("Toolchain", tool, OK, version[0] if version else path)
        else:
            report.add("Toolchain", tool, MISSING if required else ABSENT, "not in PATH", hint, required)


def check_python(report: Report, env: dict[str, str]) -> None:
    kind = env.get("HADROS3_ENV_KIND", "not configured")
    if ENV_FILE.exists():
        report.add("Python", "environment", OK, "%s (.hadros3-env.mk)" % kind)
    else:
        report.add(
            "Python",
            "environment",
            MISSING,
            "no .hadros3-env.mk",
            "run: make setup",
            True,
        )
    command = python_command(env)
    if command is None:
        report.add("Python", "interpreter", MISSING, "no python3 found", "%s python3" % APT, True)
        return
    probe = PROBE_TEMPLATE % (REQUIRED_MODULES,)
    raw = command_output(command + ["-c", probe], timeout=180)
    if raw is None:
        report.add(
            "Python",
            "interpreter",
            MISSING,
            " ".join(command) + " does not run",
            "run: make setup",
            True,
        )
        return
    try:
        data = json.loads(raw.splitlines()[-1])
    except Exception:
        report.add("Python", "interpreter", MISSING, "unexpected probe output", "run: make setup", True)
        return
    report.add("Python", "interpreter", OK, "%s (%s)" % (data["version"], data["executable"]))
    for name, present in sorted(data["modules"].items()):
        report.add(
            "Python",
            "module %s" % name,
            OK if present else MISSING,
            "" if present else "not importable",
            "run: make setup",
            True,
        )


def check_physics_backends(report: Report, env: dict[str, str]) -> None:
    pythia8 = env.get("PYTHIA8_PREFIX")
    if pythia8 and (Path(pythia8) / "include" / "Pythia8" / "Pythia.h").exists():
        report.add("Physics backends", "PYTHIA 8", OK, pythia8)
    else:
        report.add(
            "Physics backends",
            "PYTHIA 8",
            ABSENT,
            "headers not found",
            "conda-forge only: make setup (needs micromamba/conda)",
        )
    geant4 = env.get("GEANT4_PREFIX")
    found_geant4 = False
    if geant4:
        for libdir in ("lib", "lib64"):
            if (Path(geant4) / libdir / "cmake" / "Geant4").is_dir():
                found_geant4 = True
    if found_geant4:
        report.add("Physics backends", "Geant4", OK, geant4)
    else:
        report.add(
            "Physics backends",
            "Geant4",
            ABSENT,
            "CMake package not found",
            "conda-forge only: make setup (needs micromamba/conda)",
        )


def check_cuda(report: Report, env: dict[str, str]) -> None:
    nvcc = env.get("NVCC") or shutil.which("nvcc")
    if nvcc and Path(nvcc).exists():
        version = command_output([nvcc, "--version"]) or ""
        release = [line for line in version.splitlines() if "release" in line]
        report.add("CUDA camera preview", "nvcc", OK, release[0].strip() if release else nvcc)
    else:
        report.add(
            "CUDA camera preview",
            "nvcc",
            ABSENT,
            "not in PATH",
            "%s nvidia-cuda-toolkit (or install the NVIDIA CUDA toolkit)" % APT,
        )
    smi = shutil.which("nvidia-smi")
    if smi:
        gpu = command_output([smi, "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"])
        if gpu:
            report.add("CUDA camera preview", "GPU", OK, gpu.splitlines()[0])
        else:
            report.add("CUDA camera preview", "GPU", ABSENT, "nvidia-smi failed", "check the NVIDIA driver install")
    else:
        report.add(
            "CUDA camera preview",
            "GPU",
            ABSENT,
            "no nvidia-smi",
            "no NVIDIA driver: the interactive camera preview cannot run here",
        )
    if shutil.which("pkg-config") and subprocess.call(
        ["pkg-config", "--exists", "glfw3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ) == 0:
        report.add("CUDA camera preview", "glfw3", OK, command_output(["pkg-config", "--modversion", "glfw3"]) or "")
    else:
        report.add(
            "CUDA camera preview",
            "glfw3",
            ABSENT,
            "development package not found",
            "%s libglfw3-dev (without it the preview builds headless, no window)" % APT,
        )


def check_binaries(report: Report) -> None:
    for name in CORE_BINARIES:
        path = ROOT / "bin" / name
        report.add(
            "Compiled backends",
            name,
            OK if path.exists() else MISSING,
            str(path.relative_to(ROOT)) if path.exists() else "not built",
            "run: make cpp-core",
            True,
        )
    for name in OPTIONAL_BINARIES:
        path = ROOT / "bin" / name
        hint = {
            "hadros3_event_generator": "run: make hadros3-event-generator (needs PYTHIA 8)",
            "hadros3_geant4_transport": "run: make geant4-build (needs Geant4)",
            "hadros3_geodesic_preview_cuda": "run: make hadros3-geodesic-preview-cuda (needs nvcc)",
        }[name]
        report.add(
            "Compiled backends",
            name,
            OK if path.exists() else ABSENT,
            str(path.relative_to(ROOT)) if path.exists() else "not built",
            hint,
        )


def check_assets(report: Report) -> None:
    sky = ROOT / "assets" / "sky" / "eso0932a.ppm"
    report.add(
        "Assets",
        "sky texture",
        OK if sky.exists() else ABSENT,
        "assets/sky/eso0932a.ppm" if sky.exists() else "missing",
        "the camera preview falls back to a procedural sky",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Report HADROS3 capabilities on this machine.")
    parser.add_argument("--json", action="store_true", help="Emit the raw report as JSON.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when a required item is missing.")
    args = parser.parse_args()

    env = read_env_file()
    report = Report()
    check_platform(report)
    check_python(report, env)
    check_toolchain(report, env)
    check_binaries(report)
    check_physics_backends(report, env)
    check_cuda(report, env)
    check_assets(report)

    if args.json:
        print(json.dumps({"rows": report.rows, "required_failures": report.required_failures}, indent=2))
    else:
        print("HADROS3 doctor -- %s" % ROOT)
        print(report.render())
        print()
        print("Legend: [  ok  ] available   [ MISS ] required and missing   [  --  ] optional, not installed")
        if report.required_failures:
            print()
            print("%d required item(s) missing. Start with: make setup && make cpp-core" % report.required_failures)
        else:
            print()
            print("Everything required for the core HADROS3 workflow is present.")

    if args.strict and report.required_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
