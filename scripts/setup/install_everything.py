#!/usr/bin/env python3
"""One-command HADROS3 installation: `make install`.

Takes a machine that has nothing but a shell and a C++ compiler and leaves it
with a fully working HADROS3 checkout:

1. system packages (compiler, pkg-config, GLFW, git) through apt/dnf/pacman;
2. micromamba, downloaded into .tools/bin when it is not already installed;
3. the ``dis`` conda environment from environment.yml (Python stack, PYTHIA 8,
   HepMC3, Geant4, LHAPDF, gfortran, cmake, ninja);
4. the CUDA compiler from conda-forge when an NVIDIA GPU is present;
5. .hadros3-env.mk, so the Makefile and the Python layer agree on the paths;
6. every C++/CUDA backend the machine can build;
7. the POWHEG-BOX-RES DIS source (cloned and built);
8. a final `make doctor` report.

Standard library only: it must run before anything is installed.

Every step is independent and idempotent. A step that fails is reported and the
installation continues, so the final doctor report always tells the whole truth.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS_BIN = ROOT / ".tools" / "bin"
ENVIRONMENT_YML = ROOT / "environment.yml"

DEFAULT_ENV_NAME = os.environ.get("HADROS3_CONDA_ENV", "dis")
MAMBA_ROOT = Path(os.environ.get("MAMBA_ROOT_PREFIX", str(Path.home() / "micromamba")))

MICROMAMBA_PLATFORM = {
    ("Linux", "x86_64"): "linux-64",
    ("Linux", "aarch64"): "linux-aarch64",
    ("Linux", "ppc64le"): "linux-ppc64le",
    ("Darwin", "x86_64"): "osx-64",
    ("Darwin", "arm64"): "osx-arm64",
}

APT_PACKAGES = ["build-essential", "pkg-config", "libglfw3-dev", "libgl1-mesa-dev", "git", "ca-certificates", "bzip2"]
DNF_PACKAGES = ["gcc-c++", "make", "pkgconf-pkg-config", "glfw-devel", "mesa-libGL-devel", "git", "bzip2"]
PACMAN_PACKAGES = ["base-devel", "pkgconf", "glfw", "mesa", "git"]

results: list[tuple[str, str, str]] = []


def banner(title: str) -> None:
    print("\n" + "=" * 72, flush=True)
    print("== " + title, flush=True)
    print("=" * 72, flush=True)


def note(message: str) -> None:
    print("[install] " + message, flush=True)


def record(step: str, status: str, detail: str = "") -> None:
    results.append((step, status, detail))


def run(command: list[str], **kwargs: object) -> int:
    note("$ " + " ".join(str(part) for part in command))
    return subprocess.call(command, **kwargs)  # type: ignore[arg-type]


def command_output(command: list[str], timeout: int = 120) -> str | None:
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True, timeout=timeout).strip()
    except Exception:
        return None


# --------------------------------------------------------------------------
# 1. system packages
# --------------------------------------------------------------------------


def missing_apt_packages(packages: list[str]) -> list[str]:
    missing = []
    for name in packages:
        status = command_output(["dpkg-query", "-W", "-f=${Status}", name])
        if status is None or "install ok installed" not in status:
            missing.append(name)
    return missing


def sudo_prefix() -> list[str]:
    if os.geteuid() == 0:
        return []
    if shutil.which("sudo") is None:
        return []
    return ["sudo"]


def install_system_packages(allow_sudo: bool) -> None:
    banner("Step 1/8 -- system packages")
    if shutil.which("apt-get"):
        missing = missing_apt_packages(APT_PACKAGES)
        if not missing:
            note("all system packages already installed")
            record("system packages", "ok", "nothing to install")
            return
        note("missing: " + " ".join(missing))
        prefix = sudo_prefix()
        if not allow_sudo or (prefix and shutil.which("sudo") is None):
            note("skipping the system install; run this yourself:")
            note("  sudo apt-get install -y " + " ".join(missing))
            record("system packages", "skipped", "missing: " + " ".join(missing))
            return
        if prefix:
            note("this needs administrator rights; sudo may ask for your password")
        run(prefix + ["apt-get", "update"])
        code = run(prefix + ["apt-get", "install", "-y"] + missing)
        record("system packages", "ok" if code == 0 else "failed", " ".join(missing))
        return

    for manager, packages, install in (
        ("dnf", DNF_PACKAGES, ["dnf", "install", "-y"]),
        ("pacman", PACMAN_PACKAGES, ["pacman", "-S", "--needed", "--noconfirm"]),
    ):
        if shutil.which(manager):
            if not allow_sudo:
                note("run yourself: %s %s" % (" ".join(install), " ".join(packages)))
                record("system packages", "skipped", manager)
                return
            code = run(sudo_prefix() + install + packages)
            record("system packages", "ok" if code == 0 else "failed", manager)
            return

    note("no supported package manager found; make sure a C++ compiler and GLFW are installed")
    record("system packages", "skipped", "unknown package manager")


# --------------------------------------------------------------------------
# 2. micromamba
# --------------------------------------------------------------------------


def find_micromamba() -> str | None:
    for candidate in [
        shutil.which("micromamba"),
        str(TOOLS_BIN / "micromamba"),
        str(Path.home() / ".local" / "bin" / "micromamba"),
        str(Path.home() / "micromamba" / "bin" / "micromamba"),
        str(Path.home() / "bin" / "micromamba"),
    ]:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def download_micromamba() -> str | None:
    key = (platform.system(), platform.machine())
    target = MICROMAMBA_PLATFORM.get(key)
    if target is None:
        note("no micromamba build known for %s/%s" % key)
        return None
    url = "https://micro.mamba.pm/api/micromamba/%s/latest" % target
    TOOLS_BIN.mkdir(parents=True, exist_ok=True)
    destination = TOOLS_BIN / "micromamba"
    note("downloading micromamba from %s" % url)
    try:
        with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310 - fixed vendor URL
            payload = response.read()
        with tempfile.TemporaryDirectory() as workdir:
            archive = Path(workdir) / "micromamba.tar.bz2"
            archive.write_bytes(payload)
            with tarfile.open(archive, "r:bz2") as tar:
                member = next((item for item in tar.getmembers() if item.name.endswith("bin/micromamba")), None)
                if member is None:
                    note("unexpected micromamba archive layout")
                    return None
                extracted = tar.extractfile(member)
                if extracted is None:
                    return None
                destination.write_bytes(extracted.read())
        destination.chmod(0o755)
    except Exception as error:  # network, TLS, tar -- all equally fatal for this step
        note("micromamba download failed: %s" % error)
        return None
    note("micromamba installed at %s" % destination)
    return str(destination)


def ensure_micromamba() -> str | None:
    banner("Step 2/8 -- micromamba")
    found = find_micromamba()
    if found:
        note("found: %s" % found)
        record("micromamba", "ok", found)
        return found
    downloaded = download_micromamba()
    record("micromamba", "ok" if downloaded else "failed", downloaded or "download failed")
    return downloaded


# --------------------------------------------------------------------------
# 3. conda environment
# --------------------------------------------------------------------------


def mamba_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MAMBA_ROOT_PREFIX", str(MAMBA_ROOT))
    return env


def env_prefix(name: str) -> Path | None:
    for root in [MAMBA_ROOT, Path.home() / "micromamba", Path.home() / "miniforge3", Path.home() / "miniconda3"]:
        candidate = root / "envs" / name
        if (candidate / "bin" / "python").exists():
            return candidate
    return None


def create_environment(manager: str, name: str) -> Path | None:
    banner("Step 3/8 -- conda environment '%s'" % name)
    env = mamba_env()
    existing = env_prefix(name)
    if existing is not None:
        note("environment already exists at %s; updating it from environment.yml" % existing)
        run([manager, "install", "-y", "-n", name, "-f", str(ENVIRONMENT_YML)], env=env)
    else:
        note("creating '%s' from environment.yml" % name)
        note("this downloads a few GB (Geant4 datasets included) and takes several minutes")
        code = run([manager, "create", "-y", "-n", name, "-f", str(ENVIRONMENT_YML)], env=env)
        if code != 0:
            note("environment creation failed")
            record("conda environment", "failed", name)
            return None
    prefix = env_prefix(name)
    record("conda environment", "ok" if prefix else "failed", str(prefix or name))
    return prefix


# --------------------------------------------------------------------------
# 4. CUDA compiler
# --------------------------------------------------------------------------


def install_cuda_compiler(manager: str, name: str, prefix: Path | None) -> None:
    banner("Step 4/8 -- CUDA compiler")
    if prefix is not None and (prefix / "bin" / "nvcc").exists():
        note("nvcc already present in the environment")
        record("cuda compiler", "ok", str(prefix / "bin" / "nvcc"))
        return
    if shutil.which("nvcc"):
        note("nvcc already present in PATH")
        record("cuda compiler", "ok", shutil.which("nvcc") or "")
        return
    if shutil.which("nvidia-smi") is None:
        note("no NVIDIA driver found: this machine cannot run the interactive CUDA camera preview")
        note("everything else (web shell, geometry, static preview, physics backends) still works")
        record("cuda compiler", "skipped", "no NVIDIA GPU")
        return
    gpu = command_output(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"])
    note("NVIDIA GPU detected: %s" % (gpu or "unknown"))
    note("installing the CUDA compiler from conda-forge (no root needed)")
    code = run(
        [manager, "install", "-y", "-n", name, "-c", "conda-forge", "cuda-nvcc", "cuda-cudart-dev"],
        env=mamba_env(),
    )
    record("cuda compiler", "ok" if code == 0 else "failed", gpu or "")


# --------------------------------------------------------------------------
# 5-8. environment file, builds, POWHEG, doctor
# --------------------------------------------------------------------------


def write_environment_file(name: str) -> None:
    banner("Step 5/8 -- resolving paths (.hadros3-env.mk)")
    code = run(
        [sys.executable, str(ROOT / "scripts" / "setup" / "bootstrap_environment.py"), "--env-name", name, "--no-create"],
        cwd=ROOT,
    )
    record("path resolution", "ok" if code == 0 else "failed", ".hadros3-env.mk")


def build_backends() -> None:
    banner("Step 6/8 -- building the backends")
    code = run(["make", "all"], cwd=ROOT)
    record("backends", "ok" if code == 0 else "failed", "make all")


def install_powheg(skip: bool) -> None:
    banner("Step 7/8 -- POWHEG-BOX-RES DIS")
    if skip:
        note("skipped (HADROS3_SKIP_POWHEG=1)")
        record("powheg", "skipped", "requested")
        return
    note("cloning and building POWHEG; this is the long tail of the install")
    if run(["make", "powheg-fetch"], cwd=ROOT) != 0:
        note("POWHEG fetch failed (network or GitLab); the rest of HADROS3 is unaffected")
        record("powheg", "failed", "fetch")
        return
    code = run(["make", "powheg-build"], cwd=ROOT)
    record("powheg", "ok" if code == 0 else "failed", "build")


def final_report() -> int:
    banner("Step 8/8 -- capability report")
    run(["make", "doctor"], cwd=ROOT)

    print("\n" + "=" * 72)
    print("== HADROS3 installation summary")
    print("=" * 72)
    width = max(len(step) for step, _, _ in results) + 2
    for step, status, detail in results:
        print("  %s %s %s" % (step.ljust(width), status.ljust(9), detail))
    failed = [step for step, status, _ in results if status == "failed"]
    print()
    if failed:
        print("Some steps failed: %s" % ", ".join(failed))
        print("The doctor table above shows what is usable right now.")
        print("Re-running 'make install' is safe and only redoes what is missing.")
        return 1
    print("HADROS3 is installed. Start the dashboard with:  make hadros-web")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Install everything HADROS3 needs on this machine.")
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME, help="Conda environment name (default: dis).")
    parser.add_argument("--no-sudo", action="store_true", help="Never invoke sudo; print the system packages instead.")
    parser.add_argument("--skip-powheg", action="store_true", help="Skip cloning and building POWHEG-BOX-RES.")
    args = parser.parse_args()

    allow_sudo = not (args.no_sudo or os.environ.get("HADROS3_NO_SUDO") == "1")
    skip_powheg = args.skip_powheg or os.environ.get("HADROS3_SKIP_POWHEG") == "1"

    banner("HADROS3 -- full installation")
    note("repository: %s" % ROOT)
    note("environment: %s (conda root %s)" % (args.env_name, MAMBA_ROOT))
    note("this takes 15-40 minutes on a first run and downloads several GB")

    install_system_packages(allow_sudo)

    manager = ensure_micromamba()
    prefix = None
    if manager is None:
        note("without micromamba the physics backends cannot be installed; continuing with the Python-only layer")
        record("conda environment", "skipped", "no micromamba")
        record("cuda compiler", "skipped", "no micromamba")
        run([sys.executable, str(ROOT / "scripts" / "setup" / "bootstrap_environment.py"), "--light"], cwd=ROOT)
    else:
        prefix = create_environment(manager, args.env_name)
        install_cuda_compiler(manager, args.env_name, prefix)
        write_environment_file(args.env_name)

    build_backends()
    install_powheg(skip_powheg)
    return final_report()


if __name__ == "__main__":
    raise SystemExit(main())
