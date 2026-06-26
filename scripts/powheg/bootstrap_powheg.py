#!/usr/bin/env python3
"""Bootstrap the self-contained HADROS3 POWHEG DIS smoke environment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
POWHEG_URL = "https://gitlab.com/POWHEG-BOX/RES/POWHEG-BOX-RES.git"
POWHEG_COMMIT = "1a31cef9bc594e7f59ac94485a107512030dd1b1"
POWHEG_DIS_URL = "https://gitlab.com/POWHEG-BOX/RES/User-Processes/DIS.git"
POWHEG_DIS_COMMIT = "29f394adf9e958c307812fd9e2ce61d368461e96"
POWHEG_ROOT = ROOT / "external" / "powheg"
POWHEG_SOURCE = POWHEG_ROOT / "POWHEG-BOX-RES"
POWHEG_BUILD = POWHEG_ROOT / "build" / "DIS"
POWHEG_BINARY = POWHEG_BUILD / "pwhg_main"
DEFAULT_LOCAL_SOURCE = ROOT.parent / "HADROS-CASCADE" / "external_cache" / "POWHEG-BOX-RES"
SMOKE_DIR = ROOT / "tmp" / "powheg_smoke"
SMOKE_EVENTS = 2
SMOKE_ENERGY_GEV = 1.0e9
NUCLEON_MASS_GEV = 0.938272


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, log: Path | None = None) -> None:
    if log is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        subprocess.run(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    ignored = shutil.ignore_patterns(
        ".git",
        "*.o",
        "*.a",
        "*.so",
        "*.mod",
        "pwhg_main",
        "pwgevents*.lhe",
        "*.log",
    )
    shutil.copytree(source, target, symlinks=True, ignore=ignored)


def git_output(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def git_commit_if_repo(path: Path) -> str:
    if not ((path / ".git").exists() or (path / ".git").is_file()):
        return ""
    top = git_output(["rev-parse", "--show-toplevel"], path)
    if top and Path(top).resolve() != path.resolve():
        return ""
    return git_output(["rev-parse", "HEAD"], path)


def source_metadata(source: Path) -> dict[str, Any]:
    dis = source / "DIS"
    return {
        "powheg_source_url": POWHEG_URL,
        "powheg_source_commit_pinned": POWHEG_COMMIT,
        "powheg_dis_source_url": POWHEG_DIS_URL,
        "powheg_dis_commit_pinned": POWHEG_DIS_COMMIT,
        "powheg_source_commit_detected": git_commit_if_repo(source),
        "powheg_dis_commit_detected": git_commit_if_repo(dis) if dis.exists() else "",
        "source_path": str(source),
        "dis_makefile_exists": (dis / "Makefile").exists(),
    }


def fetch(args: argparse.Namespace) -> None:
    del args
    POWHEG_ROOT.mkdir(parents=True, exist_ok=True)
    if (POWHEG_SOURCE / "DIS" / "Makefile").exists():
        print(f"[powheg-fetch] POWHEG source already present: {POWHEG_SOURCE}")
    else:
        explicit = os.environ.get("HADROS3_POWHEG_SOURCE", "").strip()
        source = Path(explicit).expanduser() if explicit else DEFAULT_LOCAL_SOURCE
        if source.exists() and (source / "DIS" / "Makefile").exists():
            print(f"[powheg-fetch] Copying local POWHEG source from {source}")
            copy_tree(source, POWHEG_SOURCE)
        else:
            print(f"[powheg-fetch] Cloning pinned POWHEG source from {POWHEG_URL}")
            run(["git", "clone", "--recursive", POWHEG_URL, str(POWHEG_SOURCE)])
            run(["git", "checkout", POWHEG_COMMIT], cwd=POWHEG_SOURCE)
            run(["git", "submodule", "update", "--init", "--recursive"], cwd=POWHEG_SOURCE)
            dis_commit = git_output(["rev-parse", "HEAD"], POWHEG_SOURCE / "DIS")
            if dis_commit and dis_commit != POWHEG_DIS_COMMIT:
                run(["git", "checkout", POWHEG_DIS_COMMIT], cwd=POWHEG_SOURCE / "DIS")
    meta = source_metadata(POWHEG_SOURCE)
    if not meta["dis_makefile_exists"]:
        raise SystemExit(f"POWHEG DIS Makefile missing after fetch: {POWHEG_SOURCE / 'DIS' / 'Makefile'}")
    (POWHEG_ROOT / "POWHEG_SOURCE.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))


def command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def toolchain_env() -> dict[str, str]:
    env = dict(os.environ)
    if command_exists("gfortran"):
        return env
    candidates = [
        Path.home() / "micromamba" / "envs" / "hadros-cascade" / "bin",
        Path.home() / "micromamba" / "envs" / "dis" / "bin",
    ]
    for candidate in candidates:
        if (candidate / "gfortran").exists():
            env["PATH"] = str(candidate) + os.pathsep + env.get("PATH", "")
            return env
    return env


def find_lhapdf_config() -> str:
    explicit = os.environ.get("LHAPDF_CONFIG", "").strip()
    if explicit:
        return explicit
    found = shutil.which("lhapdf-config")
    if found:
        return found
    candidates = [
        Path.home() / "micromamba" / "envs" / "dis" / "bin" / "lhapdf-config",
        Path.home() / "micromamba" / "envs" / "hadros-cascade" / "bin" / "lhapdf-config",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise SystemExit("LHAPDF_CONFIG not found. Install LHAPDF or set LHAPDF_CONFIG=/path/to/lhapdf-config.")


def build(args: argparse.Namespace) -> None:
    del args
    if not (POWHEG_SOURCE / "DIS" / "Makefile").exists():
        fetch(argparse.Namespace())

    lhapdf_config = find_lhapdf_config()
    build_env = toolchain_env()
    if shutil.which("gfortran", path=build_env.get("PATH", "")) is None:
        raise SystemExit("gfortran not found. Install a Fortran compiler or provide the hadros-cascade micromamba environment.")

    build_root = Path(os.environ.get("HADROS3_POWHEG_BUILD_STAGING", "/tmp/hadros3_powheg_build")).resolve()
    staging = build_root / "POWHEG-BOX-RES"
    if staging.exists():
        shutil.rmtree(staging)
    build_root.mkdir(parents=True, exist_ok=True)
    print(f"[powheg-build] Staging source in {staging}")
    copy_tree(POWHEG_SOURCE, staging)

    cmd = ["make", "DEBUG=", f"LHAPDF_CONFIG={lhapdf_config}", "pwhg_main"]
    log = POWHEG_ROOT / "build" / "powheg_build.log"
    start = time.time()
    try:
        run(cmd, cwd=staging / "DIS", env=build_env, log=log)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"POWHEG build failed with exit code {exc.returncode}. See {log}") from exc

    built = staging / "DIS" / "pwhg_main"
    if not built.exists():
        raise SystemExit(f"POWHEG build did not create {built}")
    POWHEG_BUILD.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, POWHEG_BINARY)
    POWHEG_BINARY.chmod(0o755)
    summary = {
        **source_metadata(POWHEG_SOURCE),
        "build_status": "ok",
        "build_staging": str(staging),
        "build_log": str(log),
        "lhapdf_config": lhapdf_config,
        "make_command": " ".join(cmd),
        "pwhg_main": str(POWHEG_BINARY),
        "runtime_self_contained_within_hadros3": True,
        "runtime_uses_hadros_or_hadros_cascade_paths": False,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    (POWHEG_ROOT / "build" / "powheg_build_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def lhapdf_env(lhapdf_config: str) -> dict[str, str]:
    env = dict(os.environ)
    prefix = Path(lhapdf_config).resolve().parents[1]
    lib_paths = [str(prefix / "lib")]
    for candidate in [
        Path.home() / "micromamba" / "envs" / "hadros-cascade" / "lib",
        Path.home() / "micromamba" / "envs" / "dis" / "lib",
    ]:
        if candidate.exists():
            lib_paths.append(str(candidate))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_paths + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    try:
        datadir = subprocess.check_output([lhapdf_config, "--datadir"], text=True).strip()
        if datadir:
            env["LHAPDF_DATA_PATH"] = datadir
    except Exception:
        pass
    return env


def smoke_card() -> str:
    return f"""! HADROS3 POWHEG DIS smoke card.
! H3-W9 bootstrap only: no ObserverBridge coupling, PYTHIA, GEANT4, photon transport, or spectra.
LOevents 1
numevts {SMOKE_EVENTS}
ih1 12
ih2 1
ebeam1 {SMOKE_ENERGY_GEV:.10E}
ebeam2 {NUCLEON_MASS_GEV:.6f}d0
bornktmin 0d0
bornsuppfact 0d0
Qmin 10d0
Qmax 6.1262451796D+04
xmin 0d0
xmax 1d0
ymin 0d0
ymax 1d0
q2suppr 200d0
lhans1 303400
lhans2 303400
alphas_from_pdf 1
renscfact 1d0
facscfact 1d0
use-old-grid 0
use-old-ubound 0
ncall1 100
itmx1 1
ncall2 200
itmx2 1
foldcsi 1
foldy 1
foldphi 1
nubound 100
iupperfsr 1
fastbtlbound 1
storemintupb 1
ubexcess_correct 1
storeinfo_rwgt 1
hdamp 0
bornzerodamp 1
withnegweights 1
flg_jacsing 1
testplots 0
xupbound 2d0
iseed 12345
manyseeds 0
doublefsr 0
runningscales 1
olddij 0
channel_type 3
vtype 2
smartsig 1
nores 1
parallelstage 0
xgriditeration 1
py8QED 0
py8MPI 1
py8had 2
py8shower 1
colltest 0
softtest 0
"""


def smoke(args: argparse.Namespace) -> None:
    del args
    if not POWHEG_BINARY.exists():
        build(argparse.Namespace())
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    for child in SMOKE_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    card = SMOKE_DIR / "powheg.input"
    log = SMOKE_DIR / "powheg.log"
    lhe = SMOKE_DIR / "pwgevents.lhe"
    card.write_text(smoke_card(), encoding="utf-8")

    lhapdf_config = find_lhapdf_config()
    env = lhapdf_env(lhapdf_config)
    start = time.time()
    try:
        run([str(POWHEG_BINARY)], cwd=SMOKE_DIR, env=env, log=log)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"POWHEG smoke failed with exit code {exc.returncode}. See {log}") from exc

    if not lhe.exists() or lhe.stat().st_size == 0:
        raise SystemExit(f"POWHEG smoke did not create a non-empty LHE file: {lhe}")
    text = lhe.read_text(encoding="utf-8", errors="replace")
    event_count = text.count("<event>")
    ok = "<LesHouchesEvents" in text and "</LesHouchesEvents>" in text and event_count > 0
    summary = {
        **source_metadata(POWHEG_SOURCE),
        "smoke_status": "ok" if ok else "invalid_lhe",
        "powheg_invoked": True,
        "powheg_executable": str(POWHEG_BINARY),
        "powheg_runtime_self_contained": True,
        "runtime_uses_hadros_or_hadros_cascade_paths": False,
        "pythia_invoked": False,
        "geant4_invoked": False,
        "photon_transport_invoked": False,
        "numevts_requested": SMOKE_EVENTS,
        "lhe_event_count": event_count,
        "powheg_input": str(card),
        "powheg_log": str(log),
        "lhe_output": str(lhe),
        "elapsed_seconds": round(time.time() - start, 3),
    }
    (SMOKE_DIR / "powheg_smoke_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not ok:
        raise SystemExit(f"POWHEG smoke LHE validation failed: {lhe}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch")
    sub.add_parser("build")
    sub.add_parser("smoke")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch(args)
    elif args.command == "build":
        build(args)
    elif args.command == "smoke":
        smoke(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
