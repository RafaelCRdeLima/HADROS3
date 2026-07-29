"""H3-W10 POWHEG LHE to PYTHIA 8/HepMC3 event generation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import resource
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import validate_values
from .environment import pythia8_prefix
from .paths import clear_event_generation_outputs, event_generation_dir, powheg_dir
from .powheg import lhe_weight_statistics, parse_lhe_particles


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "bin" / "hadros3_event_generator"
PYTHIA_PREFIX = pythia8_prefix()
PYTHIA_CONFIG = PYTHIA_PREFIX / "bin" / "pythia8-config"
PYTHIA_DATA = PYTHIA_PREFIX / "share" / "Pythia8" / "xmldoc"
HEPMC_CONFIG = PYTHIA_PREFIX / "share" / "HepMC3" / "cmake" / "HepMC3Config-version.cmake"

_PDG_SYMBOLS = {
    1: r"$d$", -1: r"$\bar{d}$", 2: r"$u$", -2: r"$\bar{u}$",
    3: r"$s$", -3: r"$\bar{s}$", 4: r"$c$", -4: r"$\bar{c}$",
    5: r"$b$", -5: r"$\bar{b}$", 6: r"$t$", -6: r"$\bar{t}$",
    11: r"$e^{-}$", -11: r"$e^{+}$", 12: r"$\nu_e$", -12: r"$\bar{\nu}_e$",
    13: r"$\mu^{-}$", -13: r"$\mu^{+}$", 14: r"$\nu_\mu$", -14: r"$\bar{\nu}_\mu$",
    15: r"$\tau^{-}$", -15: r"$\tau^{+}$", 16: r"$\nu_\tau$", -16: r"$\bar{\nu}_\tau$",
    21: r"$g$", 22: r"$\gamma$", 23: r"$Z^0$", 24: r"$W^{+}$", -24: r"$W^{-}$",
    111: r"$\pi^0$", 211: r"$\pi^{+}$", -211: r"$\pi^{-}$",
    221: r"$\eta$", 331: r"$\eta'$", 113: r"$\rho^0$", 213: r"$\rho^{+}$", -213: r"$\rho^{-}$",
    223: r"$\omega$", 333: r"$\phi$", 130: r"$K_L^0$", 310: r"$K_S^0$",
    311: r"$K^0$", -311: r"$\bar{K}^0$", 321: r"$K^{+}$", -321: r"$K^{-}$",
    2112: r"$n$", -2112: r"$\bar{n}$", 2212: r"$p$", -2212: r"$\bar{p}$",
    3122: r"$\Lambda^0$", -3122: r"$\bar{\Lambda}^0$",
    3222: r"$\Sigma^{+}$", 3212: r"$\Sigma^0$", 3112: r"$\Sigma^{-}$",
    -3222: r"$\bar{\Sigma}^{-}$", -3212: r"$\bar{\Sigma}^0$", -3112: r"$\bar{\Sigma}^{+}$",
    3322: r"$\Xi^0$", 3312: r"$\Xi^{-}$", -3322: r"$\bar{\Xi}^0$", -3312: r"$\bar{\Xi}^{+}$",
    3334: r"$\Omega^{-}$", -3334: r"$\bar{\Omega}^{+}$",
    411: r"$D^{+}$", -411: r"$D^{-}$", 421: r"$D^0$", -421: r"$\bar{D}^0$",
    431: r"$D_s^{+}$", -431: r"$D_s^{-}$", 511: r"$B^0$", -511: r"$\bar{B}^0$",
    521: r"$B^{+}$", -521: r"$B^{-}$", 531: r"$B_s^0$", -531: r"$\bar{B}_s^0$",
    1000010020: r"$^2\mathrm{H}$", 1000010030: r"$^3\mathrm{H}$", 1000020030: r"$^3\mathrm{He}$", 1000020040: r"$\alpha$",
}


def pdg_symbol(pdg_id: int) -> str:
    """Return a publication-style particle symbol while retaining PDG in data."""
    return _PDG_SYMBOLS.get(int(pdg_id), rf"$\mathrm{{PDG}}\,{int(pdg_id)}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def backend_availability() -> dict[str, Any]:
    pythia_version = None
    if PYTHIA_CONFIG.exists():
        try:
            pythia_version = subprocess.check_output([str(PYTHIA_CONFIG), "--version"], text=True).strip()
        except Exception:
            pass
    hepmc_version = None
    if HEPMC_CONFIG.exists():
        for line in HEPMC_CONFIG.read_text(encoding="utf-8").splitlines():
            if line.startswith("set(PACKAGE_VERSION "):
                hepmc_version = line.split('"')[1]
                break
    compiler_version = None
    compilers = sorted((PYTHIA_PREFIX / "bin").glob("*-linux-gnu-c++")) if (PYTHIA_PREFIX / "bin").is_dir() else []
    compiler = compilers[0] if compilers else PYTHIA_PREFIX / "bin" / "x86_64-conda-linux-gnu-c++"
    if compiler.exists():
        try:
            compiler_version = subprocess.check_output([str(compiler), "--version"], text=True).splitlines()[0].strip()
        except Exception:
            pass
    return {
        "available": bool(BACKEND.exists() and pythia_version and hepmc_version and PYTHIA_DATA.exists()),
        "backend_executable": str(BACKEND),
        "backend_sha256": _sha256(BACKEND) if BACKEND.exists() else None,
        "pythia_version": pythia_version,
        "hepmc3_version": hepmc_version,
        "compiler": str(compiler),
        "compiler_version": compiler_version,
        "pythia_data": str(PYTHIA_DATA),
    }


def _inputs(run_output_dir: Path, max_requests: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = powheg_dir(run_output_dir)
    summary_path = directory / "powheg_summary.json"
    requests_path = directory / "powheg_event_requests.jsonl"
    if not summary_path.exists() or not requests_path.exists():
        raise FileNotFoundError("POWHEG summary/requests not found; run real POWHEG first")
    summary = _read_json(summary_path)
    if str(summary.get("powheg_run_mode")) not in {"real_smoke", "real_free"} or not bool(summary.get("powheg_lhe_generated")):
        raise ValueError("Event Generation requires validated POWHEG real_smoke or real_free LHE")
    requests = [row for row in _read_jsonl(requests_path) if row.get("powheg_lhe_generated") or row.get("powheg_lhe_path")]
    if not requests:
        raise ValueError("POWHEG declared no generated LHE requests")
    return summary, requests[:max_requests]


def _job_input(run_output_dir: Path, request: dict[str, Any]) -> Path:
    raw = request.get("powheg_lhe_path")
    if not raw:
        raise ValueError(f"request {request.get('powheg_request_id')} has no powheg_lhe_path")
    path = Path(str(raw))
    if not path.is_absolute():
        path = run_output_dir / path
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"LHE is absent or empty: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if "<LesHouchesEvents" not in text or "</LesHouchesEvents>" not in text or "<event>" not in text:
        raise ValueError(f"invalid or truncated LHE: {path}")
    return path


def _parton_check(lhe: Path, request: dict[str, Any], job_dir: Path, max_events: int, seed: int) -> dict[str, Any]:
    request_id = str(request["powheg_request_id"])
    particles, events = parse_lhe_particles(lhe, powheg_job_id=request_id)
    events = events[:max_events]
    selected_ids = {int(row["lhe_event_index"]) for row in events}
    particles = [row for row in particles if int(row["lhe_event_index"]) in selected_ids and int(row["status"]) == 1]
    event_rows: list[dict[str, Any]] = []
    for event in events:
        event_rows.append(
            {
                "powheg_request_id": request_id,
                "lhe_event_index": event["lhe_event_index"],
                "event_generation_event_id": f"{request_id}:{event['lhe_event_index']}",
                "seed": seed,
                "xwgtup": event["event_weight"],
                "scalup_gev": event["event_scale_gev"],
                "idwtup": event.get("idwtup"),
                "xsecup_total_pb": event.get("xsecup_total_pb"),
                "n_final_particles": event["n_final_state"],
                "n_final_partons": sum(1 for p in particles if p["lhe_event_index"] == event["lhe_event_index"] and p["particle_category"] in {"quark", "gluon"}),
                "n_final_hadrons": 0,
                "four_momentum_residual_relative": event["four_momentum_residual_relative"],
                "shower_invoked": False,
                "hadronization_invoked": False,
                "decays_invoked": False,
                "matching_policy": "off",
                "generator_frame": "local_matter_tetrad",
                "status": "ok",
            }
        )
    particle_rows = []
    for particle in particles:
        particle_rows.append(
            {
                "event_generation_event_id": f"{request_id}:{particle['lhe_event_index']}",
                "particle_index": particle["particle_index"],
                "pdg_id": particle["pdg_id"],
                "status": particle["status"],
                "mother1": particle["mother1"],
                "mother2": particle["mother2"],
                "color1": particle["color1"],
                "color2": particle["color2"],
                "px_gev": particle["px_gev"],
                "py_gev": particle["py_gev"],
                "pz_gev": particle["pz_gev"],
                "energy_gev": particle["energy_gev"],
                "mass_gev": particle["mass_gev"],
                "generator_frame": "local_matter_tetrad",
                "momentum_unit": "GeV",
                "length_unit": "mm",
            }
        )
    _write_jsonl(job_dir / "events_summary.jsonl", event_rows)
    _write_jsonl(job_dir / "final_particles.jsonl", particle_rows)
    backend_summary = {
        "status": "ok",
        "backend": "lhe_canonical_parser",
        "n_events_generated": len(event_rows),
        "n_event_failures": 0,
        "four_momentum_residual_relative_max": max((row["four_momentum_residual_relative"] for row in event_rows), default=0.0),
        "onshell_residual_relative_max": 0.0,
    }
    _write_json(job_dir / "backend_summary.json", backend_summary)
    return backend_summary


def _canonicalize_lhe_beam_frame(source: Path, target: Path) -> dict[str, Any]:
    """Boost each event to the declared LHE lepton-beam frame.

    The fixed-target POWHEG lab reboost is ill-conditioned at UHE and can leave
    the incoming lepton at a slightly different energy from EBMUP. A common
    longitudinal Lorentz boost of the complete event preserves every invariant
    and restores the frame required by an external shower.
    """
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    in_init = False
    beam_energy: float | None = None
    target_mass: float | None = None
    in_event = False
    remaining = 0
    particle_indices: list[int] = []
    event_groups: list[list[int]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "<init>":
            in_init = True
            continue
        if in_init and stripped and not stripped.startswith("<"):
            init_fields = stripped.split()
            beam_energy = float(init_fields[2].replace("D", "E").replace("d", "e"))
            target_mass = float(init_fields[3].replace("D", "E").replace("d", "e"))
            in_init = False
        if stripped == "<event>":
            in_event = True
            remaining = -1
            particle_indices = []
            continue
        if not in_event or not stripped or stripped.startswith("#"):
            continue
        if remaining == -1:
            remaining = int(float(stripped.split()[0]))
            continue
        if remaining > 0:
            particle_indices.append(index)
            remaining -= 1
            if remaining == 0:
                event_groups.append(particle_indices)
                in_event = False
    if beam_energy is None or beam_energy <= 0:
        raise ValueError(f"cannot read positive LHE beam energy from {source}")
    corrections: list[float] = []
    for group in event_groups:
        incoming_energy = None
        for index in group:
            fields = lines[index].split()
            if int(fields[1]) == -1 and 11 <= abs(int(fields[0])) <= 16 and float(fields[8].replace("D", "E")) > 0:
                incoming_energy = float(fields[9].replace("D", "E"))
                break
        if incoming_energy is None or incoming_energy <= 0:
            raise ValueError("LHE event has no positive-z incoming lepton")
        rapidity = math.log(incoming_energy / beam_energy)
        corrections.append(abs(incoming_energy - beam_energy) / beam_energy)
        ch = math.cosh(rapidity)
        sh = math.sinh(rapidity)
        for index in group:
            fields = lines[index].split()
            pz = float(fields[8].replace("D", "E").replace("d", "e"))
            energy = float(fields[9].replace("D", "E").replace("d", "e"))
            fields[8] = f"{ch * pz - sh * energy:.17E}"
            fields[9] = f"{ch * energy - sh * pz:.17E}"
            lines[index] = " " + " ".join(fields)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if target_mass is None or target_mass <= 0.0:
        raise ValueError(f"cannot read positive LHE target mass from {source}")
    return {"method": "common_longitudinal_lorentz_boost_to_ebmup", "beam_energy_gev": beam_energy, "target_mass_gev": target_mass, "n_events": len(event_groups), "max_input_beam_mismatch_relative": max(corrections, default=0.0)}


def _run_backend(lhe: Path, request: dict[str, Any], job_dir: Path, cfg: dict[str, Any], mode: str, seed: int, max_events: int) -> dict[str, Any]:
    local_lhe = job_dir / "input.lhe"
    frame_correction = _canonicalize_lhe_beam_frame(lhe, local_lhe)
    command = [
        str(BACKEND), "--lhe", local_lhe.name, "--output-dir", str(job_dir), "--request-id", str(request["powheg_request_id"]),
        "--mode", mode, "--seed", str(seed), "--max-events", str(max_events),
        "--isr", str(bool(cfg["isr_enabled"])).lower(), "--fsr", str(bool(cfg["fsr_enabled"])).lower(),
        "--hadronization", str(bool(cfg["hadronization_enabled"])).lower(), "--decays", str(bool(cfg["decays_enabled"])).lower(),
        "--mpi", str(bool(cfg["mpi_enabled"])).lower(), "--write-hepmc", str(bool(cfg["write_hepmc3"])).lower(),
        "--target-lepton-energy", str(frame_correction["beam_energy_gev"]),
        "--target-mass", str(frame_correction["target_mass_gev"]),
    ]
    env = dict(os.environ)
    env["PYTHIA8DATA"] = str(PYTHIA_DATA)
    env["LD_LIBRARY_PATH"] = str(PYTHIA_PREFIX / "lib") + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    effective_settings = [
        f"Beams:LHEF = {local_lhe.name}",
        "Beams:frameType = 4",
        f"2212:m0 = {frame_correction['target_mass_gev']:.17g}",
        "PDF:lepton = off",
        "Random:setSeed = on",
        f"Random:seed = {seed}",
        f"PartonLevel:ISR = {'on' if cfg['isr_enabled'] else 'off'}",
        f"PartonLevel:FSR = {'on' if cfg['fsr_enabled'] else 'off'}",
        f"PartonLevel:MPI = {'on' if cfg['mpi_enabled'] else 'off'}",
        f"HadronLevel:Hadronize = {'on' if cfg['hadronization_enabled'] else 'off'}",
        f"HadronLevel:Decay = {'on' if cfg['decays_enabled'] else 'off'}",
        "POWHEG:veto = 1",
        "POWHEG:pTdef = 1",
        "POWHEG:pTemt = 0",
        "POWHEG:emitted = 0",
        "POWHEG:pThard = 0",
        "POWHEG:MPIveto = 0",
    ]
    (job_dir / "pythia.cmnd").write_text("\n".join(effective_settings) + "\n", encoding="utf-8")
    log_path = job_dir / "event_generation.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=job_dir, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = _read_json(job_dir / "backend_summary.json")
    summary["lhe_frame_correction"] = frame_correction
    _write_json(job_dir / "backend_summary.json", summary)
    return summary


def _merge_hepmc(paths: list[Path], output: Path) -> None:
    if not paths:
        return
    run_header: list[str] = []
    event_lines: list[str] = []
    next_event_number = 0
    for path_index, path in enumerate(paths):
        content = [
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            and not line.startswith("HepMC::Version")
            and "START_EVENT_LISTING" not in line
            and "END_EVENT_LISTING" not in line
        ]
        try:
            first_event = next(index for index, line in enumerate(content) if line.startswith("E "))
        except StopIteration as exc:
            raise RuntimeError(f"HepMC3 input has no event records: {path}") from exc
        # HepMC3 run information (notably the W weight-name record) must occur
        # exactly once. Repeating it between events makes ReaderAscii reject the
        # otherwise valid aggregate file.
        if path_index == 0:
            run_header = content[:first_event]
        for line in content[first_event:]:
            if line.startswith("E "):
                fields = line.split(" ", 2)
                if len(fields) < 3:
                    raise RuntimeError(f"malformed HepMC3 event record in {path}: {line}")
                line = f"E {next_event_number} {fields[2]}"
                next_event_number += 1
            event_lines.append(line)
    lines = run_header + event_lines
    output.write_text(
        "HepMC::Version 3.03.01\nHepMC::Asciiv3-START_EVENT_LISTING\n"
        + "\n".join(lines)
        + "\nHepMC::Asciiv3-END_EVENT_LISTING\n",
        encoding="utf-8",
    )


def _plots(events: list[dict[str, Any]], particles: list[dict[str, Any]], output_dir: Path) -> None:
    specs = [
        ("event_generation_multiplicity.png", [int(row.get("n_final_particles", 0)) for row in events], "Final-state multiplicity", "particles/event"),
        ("event_generation_energy_spectrum.png", [float(row.get("energy_gev", 0.0)) for row in particles if float(row.get("energy_gev", 0.0)) > 0], "Final-particle energy", "energy [GeV]"),
        ("event_generation_conservation.png", [float(row.get("four_momentum_residual_relative", 0.0)) for row in events], "Four-momentum residual", "relative residual"),
    ]
    for filename, values, title, xlabel in specs:
        fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
        if values:
            ax.hist(values, bins=min(30, max(3, len(values))), color="#2563eb", alpha=0.82)
            if filename != "event_generation_multiplicity.png" and all(value > 0 for value in values):
                ax.set_xscale("log")
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(output_dir / filename)
        plt.close(fig)
    counts = Counter(int(row["pdg_id"]) for row in particles)
    items = counts.most_common(16)
    fig, ax = plt.subplots(figsize=(max(7.2, 0.62 * len(items)), 4.4), dpi=180)
    positions = list(range(len(items)))
    ax.bar(positions, [value for _, value in items], color="#059669")
    ax.set_xticks(positions, [pdg_symbol(key) for key, _ in items])
    ax.set_title("Final-particle species")
    ax.set_xlabel("particle species")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=0, labelsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "event_generation_species.png")
    plt.close(fig)


def _final_particle_class(row: dict[str, Any]) -> str:
    pdg = abs(int(row.get("pdg_id", 0)))
    if pdg in {12, 14, 16}:
        return "neutrino"
    if pdg in {11, 13, 15}:
        return "charged_lepton"
    if pdg == 22:
        return "photon"
    if bool(row.get("is_hadron")):
        return "meson" if 100 <= pdg < 1000 else "baryon"
    if bool(row.get("is_parton")):
        return "parton"
    return "other"


def generate_event_generation_products(values: dict[str, dict[str, Any]], *, run_output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    run_output_dir = run_output_dir.resolve()
    problems = validate_values(values)
    if problems:
        raise ValueError("Invalid HADROS3 configuration:\n- " + "\n- ".join(problems))
    cfg = values["event_generation"]
    mode = str(cfg["mode"])
    if mode == "disabled":
        raise ValueError("event_generation.mode=disabled; choose dry_run, parton_check, real_smoke, or real_free")
    availability = backend_availability()
    if not availability["available"]:
        raise RuntimeError(f"Event Generation backend unavailable: {availability}")
    max_requests = 1 if mode == "real_smoke" else int(float(cfg["max_requests"]))
    max_events = min(2, int(float(cfg["max_events_per_request"]))) if mode == "real_smoke" else int(float(cfg["max_events_per_request"]))
    powheg_summary, requests = _inputs(run_output_dir, max_requests)

    clear_event_generation_outputs(run_output_dir)
    final_dir = event_generation_dir(run_output_dir)
    staging = final_dir / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    jobs_dir = staging / "jobs"
    jobs_dir.mkdir()
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    manifest_jobs: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_particles: list[dict[str, Any]] = []
    hepmc_paths: list[Path] = []
    job_rows: list[dict[str, Any]] = []
    for request_index, request in enumerate(requests, start=1):
        request_id = str(request["powheg_request_id"])
        lhe = _job_input(run_output_dir, request)
        before_hash = _sha256(lhe)
        input_particles, input_events = parse_lhe_particles(lhe, powheg_job_id=request_id)
        numeric_event_fields = ("event_weight", "event_scale_gev", "four_momentum_residual_relative")
        numeric_particle_fields = ("px_gev", "py_gev", "pz_gev", "energy_gev", "mass_gev")
        if any(not math.isfinite(float(row[key])) for row in input_events for key in numeric_event_fields):
            raise ValueError(f"non-finite LHE event field in {lhe}")
        if any(not math.isfinite(float(row[key])) for row in input_particles for key in numeric_particle_fields):
            raise ValueError(f"non-finite LHE particle field in {lhe}")
        expected = min(max_events, len(input_events))
        if expected <= 0:
            raise ValueError(f"no parseable LHE events in {lhe}")
        job_dir = jobs_dir / request_id
        job_dir.mkdir()
        seed = (int(float(cfg["random_seed"])) + request_index * 100003) % 900000000 or 1
        if mode == "dry_run":
            backend_summary = {"status": "ready", "n_events_generated": 0, "four_momentum_residual_relative_max": 0.0, "onshell_residual_relative_max": 0.0}
            _write_json(job_dir / "backend_summary.json", backend_summary)
        elif mode == "parton_check":
            backend_summary = _parton_check(lhe, request, job_dir, expected, seed)
        else:
            backend_summary = _run_backend(lhe, request, job_dir, cfg, mode, seed, expected)
        after_hash = _sha256(lhe)
        if before_hash != after_hash:
            raise RuntimeError(f"upstream LHE was modified: {lhe}")
        if mode != "dry_run" and int(backend_summary["n_events_generated"]) != expected:
            raise RuntimeError(f"event cardinality mismatch for {request_id}: expected {expected}, got {backend_summary['n_events_generated']}")
        events = _read_jsonl(job_dir / "events_summary.jsonl") if (job_dir / "events_summary.jsonl").exists() else []
        particles = _read_jsonl(job_dir / "final_particles.jsonl") if (job_dir / "final_particles.jsonl").exists() else []
        for event in events:
            input_event = next((row for row in input_events if int(row["lhe_event_index"]) == int(event["lhe_event_index"])), None)
            if input_event:
                event["xwgtup"] = input_event["event_weight"]
                event["scalup_gev"] = input_event["event_scale_gev"]
                event["idwtup"] = input_event.get("idwtup")
                event["xsecup_total_pb"] = input_event.get("xsecup_total_pb")
            event.update({
                "interaction_id": request.get("interaction_id"), "source_sample_id": request.get("source_sample_id"),
                "primary_branch_id": request.get("primary_branch_id"), "physics_weight": request.get("physics_weight"),
                "observer_weight": request.get("observer_weight"), "final_observation_score": request.get("final_observation_score"),
            })
        all_events.extend(events)
        all_particles.extend(particles)
        if (job_dir / "events.hepmc3").exists():
            hepmc_paths.append(job_dir / "events.hepmc3")
        job_row = {"powheg_request_id": request_id, "lhe_path": str(lhe.relative_to(run_output_dir)), "lhe_sha256": before_hash, "seed": seed, "expected_events": expected, **backend_summary}
        job_rows.append(job_row)
        manifest_jobs.append({
            "powheg_request_id": request_id,
            "lhe_path": str(lhe.relative_to(run_output_dir)),
            "lhe_sha256": before_hash,
            "seed": seed,
            "expected_events": expected,
            "status": str(backend_summary["status"]),
            "effective_settings_path": f"jobs/{request_id}/pythia.cmnd" if mode in {"real_smoke", "real_free"} else None,
        })

    _write_jsonl(staging / "event_generation_jobs.jsonl", job_rows)
    _write_jsonl(staging / "event_generation_events_summary.jsonl", all_events)
    _write_jsonl(staging / "event_generation_final_particles.jsonl", all_particles)
    if hepmc_paths:
        _merge_hepmc(hepmc_paths, staging / "event_generation_events.hepmc3")
    weights = [float(row.get("xwgtup", 0.0)) for row in all_events]
    idwtup = next((row.get("idwtup") for row in all_events if row.get("idwtup") is not None), None)
    xsecup = next((row.get("xsecup_total_pb") for row in all_events if row.get("xsecup_total_pb") is not None), None)
    weight_summary = lhe_weight_statistics(weights, idwtup=idwtup, xsecup_total_pb=xsecup)
    max_four = max((float(row.get("four_momentum_residual_relative", 0.0)) for row in all_events), default=0.0)
    max_target_rest = max((float(row.get("target_rest_frame_residual", 0.0)) for row in all_events), default=0.0)
    max_shell = max((float(row.get("onshell_residual_relative_max", 0.0)) for row in job_rows), default=0.0)
    final_partons = sum(1 for row in all_particles if row.get("is_parton"))
    final_hadrons = sum(1 for row in all_particles if row.get("is_hadron"))
    real = mode in {"real_smoke", "real_free"}
    validation = {
        "status": "ok" if mode == "dry_run" or (all_events and max_four <= 5e-8 and max_shell <= 1e-8 and max_target_rest <= 1e-9) else "failed",
        "cardinality_pass": mode == "dry_run" or len(all_events) == sum(int(row["expected_events"]) for row in job_rows),
        "unique_event_ids_pass": len({row["event_generation_event_id"] for row in all_events}) == len(all_events),
        "four_momentum_conservation_tolerance": 5e-8,
        "four_momentum_residual_relative_max": max_four,
        "four_momentum_conservation_pass": mode == "dry_run" or max_four <= 5e-8,
        "target_rest_frame_tolerance": 1e-9,
        "target_rest_frame_residual_max": max_target_rest,
        "target_rest_frame_pass": mode == "dry_run" or max_target_rest <= 1e-9,
        "onshell_tolerance": 1e-8,
        "onshell_residual_relative_max": max_shell,
        "onshell_pass": mode == "dry_run" or max_shell <= 1e-8,
        "charge_conservation_pass": mode == "dry_run" or all(bool(row.get("charge_conservation_pass", True)) for row in all_events),
        "matching_scale_pass": mode in {"dry_run", "parton_check"} or all(bool(row.get("matching_scale_pass", False)) for row in all_events),
        "hadronization_content_pass": not (real and bool(cfg["hadronization_enabled"])) or (final_partons == 0 and final_hadrons > 0),
        "upstream_hashes_unchanged": True,
    }
    if validation["status"] != "ok" or not all(value for key, value in validation.items() if key.endswith("_pass")):
        raise RuntimeError(f"Event Generation numerical validation failed: {validation}")
    manifest = {
        "stage": "H3-W10", "backend": "pythia8", "generator_frame": "local_matter_tetrad", "momentum_unit": "GeV", "length_unit": "mm",
        "configuration_sha256": config_hash, "configuration": cfg, "dependencies": availability, "jobs": manifest_jobs,
        "matching_policy": "off" if mode in {"dry_run", "parton_check"} else ("powheg_isr_fsr_vetoed_scalup" if cfg["isr_enabled"] else "powheg_fsr_vetoed_scalup_no_isr_fixed_target"),
    }
    _write_json(staging / "event_generation_manifest.json", manifest)
    _write_json(staging / "event_generation_validation_report.json", validation)
    content = Counter(str(row.get("pdg_id")) for row in all_particles)
    classes = Counter(_final_particle_class(row) for row in all_particles)
    _write_json(staging / "event_generation_particle_content.json", {
        "counts_by_pdg_id": dict(sorted(content.items())),
        "symbols_by_pdg_id": {key: pdg_symbol(int(key)) for key in sorted(content, key=int)},
        "counts_by_particle_class": dict(sorted(classes.items())),
        "unknown_or_unclassified_particles": classes.get("other", 0),
        "n_particles": len(all_particles),
    })
    if mode != "dry_run":
        _plots(all_events, all_particles, staging)
    viewer = "<!doctype html><meta charset='utf-8'><title>HADROS3 Event Generation</title><h1>Event Generation</h1><pre id='o'></pre><script>fetch('event_generation_events_summary.jsonl').then(r=>r.text()).then(t=>o.textContent=t)</script>"
    (staging / "event_generation_event_view.html").write_text(viewer, encoding="utf-8")
    summary = {
        "status": "ok", "stage": "H3-W10", "event_generation_mode": mode, **availability,
        "powheg_run_mode": powheg_summary.get("powheg_run_mode"), "n_requests": len(requests), "n_events_generated": len(all_events),
        "n_final_particles": len(all_particles), "n_final_partons": final_partons, "n_final_hadrons": final_hadrons,
        "pythia_invoked": real, "shower_invoked": real and (bool(cfg["isr_enabled"]) or bool(cfg["fsr_enabled"])),
        "hadronization_invoked": real and bool(cfg["hadronization_enabled"]), "decays_invoked": real and bool(cfg["decays_enabled"]),
        "geant4_invoked": False, "photon_transport_invoked": False, "spectra_invoked": False,
        "expensive_event_generation_invoked": real, "weight_statistics": weight_summary, "validation": validation,
    }
    runtime_seconds = time.perf_counter() - started
    hepmc_path = staging / "event_generation_events.hepmc3"
    hepmc_bytes = hepmc_path.stat().st_size if hepmc_path.exists() else 0
    summary.update({
        "runtime_seconds": runtime_seconds,
        "events_per_second": (len(all_events) / runtime_seconds) if runtime_seconds > 0 else None,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "hepmc3_bytes": hepmc_bytes,
        "hepmc3_bytes_per_event": (hepmc_bytes / len(all_events)) if all_events else None,
        "estimated_hepmc3_bytes_before_run": sum(int(row["expected_events"]) for row in job_rows) * 250000,
    })
    products = {path.stem: str(final_dir / path.name) for path in staging.iterdir() if path.is_file()}
    products.update({"event_generation_summary": str(final_dir / "event_generation_summary.json"), "event_generation_summary_csv": str(final_dir / "event_generation_summary.csv")})
    summary["products"] = products
    _write_json(staging / "event_generation_summary.json", summary)
    with (staging / "event_generation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key in ("status", "event_generation_mode", "n_requests", "n_events_generated", "n_final_particles", "n_final_hadrons", "runtime_seconds", "events_per_second", "peak_rss_mib", "hepmc3_bytes"):
            writer.writerow([key, summary[key]])
    for child in list(staging.iterdir()):
        child.rename(final_dir / child.name)
    staging.rmdir()
    return summary
