"""H3-W9 POWHEG orchestration and diagnostics."""

from __future__ import annotations

import json
import csv
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import validate_values
from .paths import clear_powheg_outputs, observer_bridge_dir, powheg_dir, run_metadata_dir
from .provenance import write_json


ROOT = Path(__file__).resolve().parents[1]
POWHEG_CPP_EXECUTABLE = ROOT / "bin" / "hadros3_powheg_driver"
POWHEG_BINARY = ROOT / "external" / "powheg" / "build" / "DIS" / "pwhg_main"

PDG_NAMES = {
    -16: "anti_nu_tau",
    -15: "tau+",
    -14: "anti_nu_mu",
    -13: "mu+",
    -12: "anti_nu_e",
    -11: "e+",
    -6: "tbar",
    -5: "bbar",
    -4: "cbar",
    -3: "sbar",
    -2: "ubar",
    -1: "dbar",
    1: "d",
    2: "u",
    3: "s",
    4: "c",
    5: "b",
    6: "t",
    11: "e-",
    12: "nu_e",
    13: "mu-",
    14: "nu_mu",
    15: "tau-",
    16: "nu_tau",
    21: "g",
    22: "gamma",
    23: "Z0",
    24: "W+",
    -24: "W-",
    90: "system",
    91: "cluster",
    92: "string",
    2212: "p",
    -2212: "pbar",
    2112: "n",
    -2112: "nbar",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _particle_name(pdg_id: int) -> str:
    if pdg_id in PDG_NAMES:
        return PDG_NAMES[pdg_id]
    sign = "anti_" if pdg_id < 0 else ""
    return f"{sign}pdg_{abs(pdg_id)}"


def _parse_lhe_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _eta(px: float, py: float, pz: float) -> float | None:
    pt = math.hypot(px, py)
    if pt <= 0.0:
        if pz > 0.0:
            return None
        if pz < 0.0:
            return None
        return 0.0
    return math.asinh(pz / pt)


def parse_lhe_particles(lhe_path: Path, *, powheg_job_id: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse a compact subset of LHE event records into particle/event rows."""
    text = lhe_path.read_text(encoding="utf-8", errors="replace")
    job_id = powheg_job_id or lhe_path.parent.name
    particles: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    event_index = 0
    cursor = 0
    while True:
        start = text.find("<event>", cursor)
        if start < 0:
            break
        end = text.find("</event>", start)
        if end < 0:
            break
        event_index += 1
        block = text[start + len("<event>") : end]
        lines = [line.strip() for line in block.splitlines() if line.strip() and not line.strip().startswith("#")]
        cursor = end + len("</event>")
        if not lines:
            continue
        try:
            n_particles = int(float(lines[0].split()[0]))
        except (IndexError, ValueError):
            n_particles = max(0, len(lines) - 1)
        event_particles: list[dict[str, Any]] = []
        for particle_index, line in enumerate(lines[1 : 1 + n_particles], start=1):
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                pdg_id = int(parts[0])
                status = int(parts[1])
                mother1 = int(parts[2])
                mother2 = int(parts[3])
                color1 = int(parts[4])
                color2 = int(parts[5])
                px = _parse_lhe_float(parts[6])
                py = _parse_lhe_float(parts[7])
                pz = _parse_lhe_float(parts[8])
                energy = _parse_lhe_float(parts[9])
                mass = _parse_lhe_float(parts[10])
                lifetime = _parse_lhe_float(parts[11])
                spin = _parse_lhe_float(parts[12])
            except ValueError:
                continue
            pt = math.hypot(px, py)
            p_abs = math.sqrt(px * px + py * py + pz * pz)
            row = {
                "powheg_job_id": job_id,
                "lhe_event_index": event_index,
                "particle_index": particle_index,
                "pdg_id": pdg_id,
                "particle_name": _particle_name(pdg_id),
                "status": status,
                "mother1": mother1,
                "mother2": mother2,
                "color1": color1,
                "color2": color2,
                "px_gev": px,
                "py_gev": py,
                "pz_gev": pz,
                "energy_gev": energy,
                "mass_gev": mass,
                "lifetime": lifetime,
                "spin": spin,
                "pt_gev": pt,
                "p_abs_gev": p_abs,
                "eta": _eta(px, py, pz),
                "phi": math.atan2(py, px),
            }
            event_particles.append(row)
            particles.append(row)
        final_particles = [row for row in event_particles if int(row["status"]) == 1]
        initial_particles = [row for row in event_particles if int(row["status"]) == -1]
        events.append(
            {
                "powheg_job_id": job_id,
                "lhe_event_index": event_index,
                "n_particles": len(event_particles),
                "n_initial_state": len(initial_particles),
                "n_final_state": len(final_particles),
                "sum_final_energy_gev": sum(float(row["energy_gev"]) for row in final_particles),
                "sum_final_px_gev": sum(float(row["px_gev"]) for row in final_particles),
                "sum_final_py_gev": sum(float(row["py_gev"]) for row in final_particles),
                "sum_final_pz_gev": sum(float(row["pz_gev"]) for row in final_particles),
                "pdg_ids": [int(row["pdg_id"]) for row in event_particles],
                "particle_names": [str(row["particle_name"]) for row in event_particles],
            }
        )
    return particles, events


def _runtime_config_path(values: dict[str, dict[str, Any]], run_output_dir: Path) -> Path:
    config_path = run_metadata_dir(run_output_dir) / "hadros3_config.json"
    write_json(config_path, {"hadros3_values": values})
    return config_path


def _draw_card_preview(card_path: Path, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    card_text = card_path.read_text(encoding="utf-8") if card_path.exists() else "No POWHEG card generated."
    lines = card_text.splitlines()[:36]
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=150)
    ax.set_facecolor("#111827")
    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.0,
        color="#e5e7eb",
    )
    ax.set_axis_off()
    ax.set_title("POWHEG input card preview", color="#111827")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_energy_distribution(requests: list[dict[str, Any]], path: Path) -> None:
    energies = [float(row.get("interaction_E_nu_local_gev", 0.0)) for row in requests]
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    if energies:
        ax.hist(energies, bins=min(20, max(3, len(energies))), color="#2563eb", alpha=0.82)
        ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "No POWHEG requests", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("interaction_E_nu_local_gev")
    ax.set_ylabel("POWHEG jobs")
    ax.set_title("POWHEG request energy distribution")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_job_summary(summary: dict[str, Any], path: Path) -> None:
    labels = ["candidates", "jobs", "cards", "LHE"]
    values = [
        int(summary.get("n_candidates_input", 0)),
        int(summary.get("powheg_jobs_prepared", 0)),
        int(summary.get("powheg_cards_generated", 0)),
        int(summary.get("n_lhe_events", 0)),
    ]
    colors = ["#64748b", "#0f766e", "#2563eb", "#dc2626"]
    run_mode = str(summary.get("powheg_run_mode", "dry_run"))
    lhe_generated = bool(summary.get("powheg_lhe_generated", False))
    pwhg_state = "executed" if bool(summary.get("pwhg_main_executed", False)) else "NOT executed"
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("count")
    ax.set_title("POWHEG job summary")
    ax.text(
        0.02,
        0.95,
        f"{run_mode}\npwhg_main {pwhg_state}\nLHE generated: {'YES' if lhe_generated else 'NO'}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _aggregate_particle_summary(particles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in particles:
        grouped.setdefault(int(row["pdg_id"]), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for pdg_id in sorted(grouped, key=lambda item: (-len(grouped[item]), item)):
        rows = grouped[pdg_id]
        final_rows = [row for row in rows if int(row["status"]) == 1]
        initial_rows = [row for row in rows if int(row["status"]) == -1]
        energies = [float(row["energy_gev"]) for row in rows]
        pts = [float(row["pt_gev"]) for row in rows]
        summary_rows.append(
            {
                "pdg_id": pdg_id,
                "particle_name": _particle_name(pdg_id),
                "count": len(rows),
                "final_state_count": len(final_rows),
                "initial_state_count": len(initial_rows),
                "mean_energy_gev": sum(energies) / len(energies) if energies else 0.0,
                "max_energy_gev": max(energies) if energies else 0.0,
                "mean_pt_gev": sum(pts) / len(pts) if pts else 0.0,
                "max_pt_gev": max(pts) if pts else 0.0,
            }
        )
    return summary_rows


def _write_particle_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "pdg_id",
        "particle_name",
        "count",
        "final_state_count",
        "initial_state_count",
        "mean_energy_gev",
        "max_energy_gev",
        "mean_pt_gev",
        "max_pt_gev",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _draw_lhe_particle_histogram(summary_rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    if summary_rows:
        labels = [f"{row['particle_name']}\n({row['pdg_id']})" for row in summary_rows]
        counts = [int(row["count"]) for row in summary_rows]
        ax.bar(labels, counts, color="#7c3aed", alpha=0.82)
        ax.tick_params(axis="x", labelrotation=35)
    else:
        ax.text(0.5, 0.5, "No LHE particles parsed", transform=ax.transAxes, ha="center", va="center")
    ax.set_ylabel("particle count")
    ax.set_title("POWHEG LHE hard-process particle content")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_lhe_energy_spectrum(particles: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    initial = [float(row["energy_gev"]) for row in particles if int(row["status"]) == -1 and float(row["energy_gev"]) > 0.0]
    final = [float(row["energy_gev"]) for row in particles if int(row["status"]) == 1 and float(row["energy_gev"]) > 0.0]
    data = []
    labels = []
    if initial:
        data.append(initial)
        labels.append("initial state")
    if final:
        data.append(final)
        labels.append("final state")
    if data:
        ax.hist(data, bins=min(20, max(3, len(initial) + len(final))), label=labels, alpha=0.72)
        ax.set_xscale("log")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No positive LHE particle energies", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("energy [GeV]")
    ax.set_ylabel("particles")
    ax.set_title("POWHEG LHE particle energy spectrum")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_lhe_momentum_spectrum(particles: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    p_abs = [float(row["p_abs_gev"]) for row in particles if float(row["p_abs_gev"]) > 0.0]
    pt = [float(row["pt_gev"]) for row in particles if float(row["pt_gev"]) > 0.0]
    data = []
    labels = []
    if p_abs:
        data.append(p_abs)
        labels.append("|p|")
    if pt:
        data.append(pt)
        labels.append("pT")
    if data:
        ax.hist(data, bins=min(20, max(3, len(p_abs))), label=labels, alpha=0.72)
        ax.set_xscale("log")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No positive LHE particle momenta", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("momentum [GeV]")
    ax.set_ylabel("particles")
    ax.set_title("POWHEG LHE particle momentum spectrum")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def generate_lhe_diagnostics(lhe_path: Path, output_dir: Path, *, powheg_job_id: str | None = None) -> dict[str, Any]:
    particles, events = parse_lhe_particles(lhe_path, powheg_job_id=powheg_job_id)
    particle_summary = _aggregate_particle_summary(particles)
    particles_path = output_dir / "powheg_lhe_particles.jsonl"
    events_path = output_dir / "powheg_lhe_events_summary.jsonl"
    summary_json_path = output_dir / "powheg_lhe_particle_summary.json"
    summary_csv_path = output_dir / "powheg_lhe_particle_summary.csv"
    histogram_path = output_dir / "powheg_lhe_particle_histogram.png"
    energy_path = output_dir / "powheg_lhe_energy_spectrum.png"
    momentum_path = output_dir / "powheg_lhe_momentum_spectrum.png"
    _write_jsonl(particles_path, particles)
    _write_jsonl(events_path, events)
    write_json(summary_json_path, particle_summary)
    _write_particle_summary_csv(summary_csv_path, particle_summary)
    _draw_lhe_particle_histogram(particle_summary, histogram_path)
    _draw_lhe_energy_spectrum(particles, energy_path)
    _draw_lhe_momentum_spectrum(particles, momentum_path)
    unique_types = sorted({int(row["pdg_id"]) for row in particles})
    return {
        "lhe_parser_invoked": True,
        "lhe_particles_are_hard_process": True,
        "hadronization_invoked": False,
        "pythia_invoked": False,
        "geant4_invoked": False,
        "photon_transport_invoked": False,
        "n_lhe_events_parsed": len(events),
        "n_lhe_particles": len(particles),
        "n_final_state_particles": sum(1 for row in particles if int(row["status"]) == 1),
        "unique_particle_types": len(unique_types),
        "unique_pdg_ids": unique_types,
        "particle_summary": particle_summary,
        "products": {
            "powheg_lhe_particles": str(particles_path),
            "powheg_lhe_events_summary": str(events_path),
            "powheg_lhe_particle_summary_csv": str(summary_csv_path),
            "powheg_lhe_particle_summary_json": str(summary_json_path),
            "powheg_lhe_particle_histogram": str(histogram_path),
            "powheg_lhe_energy_spectrum": str(energy_path),
            "powheg_lhe_momentum_spectrum": str(momentum_path),
        },
    }


def _count_lhe_events(text: str) -> int:
    return text.count("<event>")


def _powheg_runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    lib_paths: list[str] = []
    explicit = env.get("LHAPDF_CONFIG", "").strip()
    lhapdf_config = Path(explicit) if explicit else None
    if lhapdf_config is None:
        found = shutil.which("lhapdf-config")
        lhapdf_config = Path(found) if found else None
    if lhapdf_config and lhapdf_config.exists():
        prefix = lhapdf_config.resolve().parents[1]
        lib_paths.append(str(prefix / "lib"))
        try:
            datadir = subprocess.check_output([str(lhapdf_config), "--datadir"], text=True, stderr=subprocess.DEVNULL).strip()
            if datadir:
                env["LHAPDF_DATA_PATH"] = datadir
        except Exception:
            pass
    dis_lib = Path.home() / "micromamba" / "envs" / "dis" / "lib"
    if dis_lib.exists():
        lib_paths.append(str(dis_lib))
    if lib_paths:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_paths + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    return env


def _run_real_smoke(output_dir: Path, requests: list[dict[str, Any]]) -> dict[str, Any]:
    if not requests:
        raise RuntimeError("POWHEG real_smoke requested, but no POWHEG request was prepared.")
    if len(requests) != 1:
        raise RuntimeError(f"POWHEG real_smoke must prepare exactly one request; got {len(requests)}.")
    if not POWHEG_BINARY.exists():
        raise FileNotFoundError(f"Local POWHEG pwhg_main not found: {POWHEG_BINARY}. Run make powheg-fetch && make powheg-build first.")

    request = requests[0]
    request_id = str(request["powheg_request_id"])
    card_path = output_dir.parent / str(request["powheg_input_path"])
    if not card_path.exists():
        raise FileNotFoundError(f"POWHEG input card not found for real_smoke: {card_path}")

    work_dir = output_dir / "powheg_work" / request_id
    lhe_dir = output_dir / "powheg_lhe" / request_id
    log_dir = output_dir / "powheg_run_logs" / request_id
    work_dir.mkdir(parents=True, exist_ok=True)
    lhe_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(card_path, work_dir / "powheg.input")
    log_path = log_dir / "powheg.log"
    produced_lhe = work_dir / "pwgevents.lhe"
    final_lhe = lhe_dir / "pwgevents.lhe"

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {POWHEG_BINARY}\n")
        handle.write(f"cwd={work_dir}\n")
        handle.flush()
        try:
            subprocess.run([str(POWHEG_BINARY)], cwd=work_dir, env=_powheg_runtime_env(), stdout=handle, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"POWHEG real_smoke failed with exit code {exc.returncode}. See {log_path}") from exc

    if not produced_lhe.exists() or produced_lhe.stat().st_size == 0:
        raise RuntimeError(f"POWHEG real_smoke did not create a non-empty LHE file: {produced_lhe}. See {log_path}")
    text = produced_lhe.read_text(encoding="utf-8", errors="replace")
    n_events = _count_lhe_events(text)
    valid = "<LesHouchesEvents" in text and "</LesHouchesEvents>" in text and n_events > 0
    if not valid:
        raise RuntimeError(f"POWHEG real_smoke produced an invalid LHE file: {produced_lhe}. See {log_path}")
    shutil.copy2(produced_lhe, final_lhe)

    request.update(
        {
            "powheg_status": "real_smoke_lhe_generated",
            "powheg_invoked": True,
            "pwhg_main_executed": True,
            "powheg_lhe_generated": True,
            "powheg_lhe_path": str(final_lhe.relative_to(output_dir.parent)),
            "powheg_log_path": str(log_path.relative_to(output_dir.parent)),
            "n_lhe_events": n_events,
        }
    )
    return {
        "powheg_validation_status": "ok",
        "powheg_request_id": request_id,
        "run_mode": "real_smoke",
        "powheg_run_mode": "real_smoke",
        "pwhg_main": str(POWHEG_BINARY),
        "pwhg_main_executed": True,
        "powheg_invoked": True,
        "powheg_lhe_generated": True,
        "lhe_found": True,
        "lhe_valid": True,
        "n_lhe_events": n_events,
        "powheg_lhe_path": str(final_lhe),
        "powheg_log_path": str(log_path),
        "powheg_work_dir": str(work_dir),
        "pythia_invoked": False,
        "geant4_invoked": False,
        "photon_transport_invoked": False,
        "spectra_invoked": False,
    }


def _augment_summary(summary: dict[str, Any], output_dir: Path, requests: list[dict[str, Any]]) -> dict[str, Any]:
    products = dict(summary.get("products", {}))
    products.update(
        {
            "powheg_event_requests": str(output_dir / "powheg_event_requests.jsonl"),
            "powheg_summary_json": str(output_dir / "powheg_summary.json"),
            "powheg_summary_csv": str(output_dir / "powheg_summary.csv"),
            "powheg_report": str(output_dir / "powheg_report.json"),
            "powheg_card_preview": str(output_dir / "powheg_card_preview.png"),
            "powheg_energy_distribution": str(output_dir / "powheg_energy_distribution.png"),
            "powheg_job_summary": str(output_dir / "powheg_job_summary.png"),
            "powheg_validation_report": str(output_dir / "powheg_validation_report.json"),
            "powheg_lhe_particles": str(output_dir / "powheg_lhe_particles.jsonl"),
            "powheg_lhe_events_summary": str(output_dir / "powheg_lhe_events_summary.jsonl"),
            "powheg_lhe_particle_summary_csv": str(output_dir / "powheg_lhe_particle_summary.csv"),
            "powheg_lhe_particle_summary_json": str(output_dir / "powheg_lhe_particle_summary.json"),
            "powheg_lhe_particle_histogram": str(output_dir / "powheg_lhe_particle_histogram.png"),
            "powheg_lhe_energy_spectrum": str(output_dir / "powheg_lhe_energy_spectrum.png"),
            "powheg_lhe_momentum_spectrum": str(output_dir / "powheg_lhe_momentum_spectrum.png"),
        }
    )
    run_mode = str(summary.get("powheg_run_mode", "dry_run"))
    is_real_smoke = run_mode == "real_smoke"
    summary.update(
        {
            "products": products,
            "powheg_dry_run_invoked": run_mode == "dry_run",
            "powheg_real_smoke_invoked": is_real_smoke,
            "powheg_invoked": False,
            "pwhg_main_executed": False,
            "powheg_lhe_generated": False,
            "lhe_found": False,
            "powheg_card_preview_generated": True,
            "powheg_energy_distribution_generated": True,
            "powheg_job_summary_generated": True,
            "powheg_event_requests_generated": True,
            "powheg_input_cards_generated": int(summary.get("powheg_cards_generated", len(requests))),
            "lhe_parser_invoked": False,
            "lhe_particles_are_hard_process": True,
            "hadronization_invoked": False,
            "n_lhe_particles": 0,
            "n_final_state_particles": 0,
            "unique_particle_types": 0,
            "powheg_lhe_products_generated": False,
            "powheg_lhe_message": "No LHE available: POWHEG dry run only.",
            "pythia_invoked": False,
            "geant4_invoked": False,
            "photon_transport_invoked": False,
            "spectra_invoked": False,
            "expensive_event_generation_invoked": False,
        }
    )
    return summary


def generate_powheg_products(values: dict[str, dict[str, Any]], *, run_output_dir: Path) -> dict[str, Any]:
    problems = validate_values(values)
    if problems:
        raise ValueError("Invalid HADROS3 configuration:\n- " + "\n- ".join(problems))
    if not POWHEG_CPP_EXECUTABLE.exists():
        raise FileNotFoundError(f"POWHEG dry-run C++ backend not found: {POWHEG_CPP_EXECUTABLE}")

    ranked_path = observer_bridge_dir(run_output_dir) / "observer_bridge_ranked_events.jsonl"
    if not ranked_path.exists():
        raise FileNotFoundError(f"Observer Bridge ranked events not found: {ranked_path}")

    output_dir = powheg_dir(run_output_dir)
    clear_powheg_outputs(run_output_dir)
    _runtime_config_path(values, run_output_dir)
    subprocess.run(
        [str(POWHEG_CPP_EXECUTABLE), "--run-output", str(run_output_dir)],
        cwd=ROOT,
        check=True,
    )

    requests_path = output_dir / "powheg_event_requests.jsonl"
    summary_path = output_dir / "powheg_summary.json"
    report_path = output_dir / "powheg_report.json"
    validation_report_path = output_dir / "powheg_validation_report.json"
    requests = _read_jsonl(requests_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_mode = str(summary.get("powheg_run_mode", values.get("powheg", {}).get("run_mode", "dry_run")))

    first_card = output_dir / "powheg_input_cards" / (requests[0]["powheg_request_id"] if requests else "H3PWHG-000001") / "powheg.input"
    _draw_card_preview(first_card, output_dir / "powheg_card_preview.png")
    _draw_energy_distribution(requests, output_dir / "powheg_energy_distribution.png")

    summary = _augment_summary(summary, output_dir, requests)
    validation_report: dict[str, Any] = {
        "powheg_validation_status": "not_run",
        "run_mode": run_mode,
        "powheg_run_mode": run_mode,
        "powheg_invoked": False,
        "pwhg_main_executed": False,
        "powheg_lhe_generated": False,
        "lhe_found": False,
        "lhe_valid": False,
        "n_lhe_events": 0,
        "pythia_invoked": False,
        "geant4_invoked": False,
        "photon_transport_invoked": False,
        "spectra_invoked": False,
        "lhe_parser_invoked": False,
        "lhe_particles_are_hard_process": True,
        "hadronization_invoked": False,
    }
    if run_mode == "real_smoke":
        validation_report = _run_real_smoke(output_dir, requests)
        lhe_diagnostics = generate_lhe_diagnostics(Path(validation_report["powheg_lhe_path"]), output_dir, powheg_job_id=str(validation_report["powheg_request_id"]))
        summary.update(
            {
                "stage_name": "H3-W9b POWHEG Real Run Smoke Mode",
                "powheg_dry_run_invoked": False,
                "powheg_real_smoke_invoked": True,
                "powheg_invoked": True,
                "pwhg_main_executed": True,
                "powheg_lhe_generated": True,
                "lhe_found": True,
                "lhe_valid": True,
                "n_lhe_events": int(validation_report["n_lhe_events"]),
                "n_powheg_jobs": 1,
                "powheg_jobs_prepared": 1,
                "powheg_cards_generated": 1,
                "powheg_validation_report_generated": True,
                "powheg_validation_report": str(validation_report_path),
                "powheg_lhe_path": validation_report["powheg_lhe_path"],
                "powheg_log_path": validation_report["powheg_log_path"],
                "lhe_parser_invoked": True,
                "lhe_particles_are_hard_process": True,
                "hadronization_invoked": False,
                "n_lhe_events_parsed": int(lhe_diagnostics["n_lhe_events_parsed"]),
                "n_lhe_particles": int(lhe_diagnostics["n_lhe_particles"]),
                "n_final_state_particles": int(lhe_diagnostics["n_final_state_particles"]),
                "unique_particle_types": int(lhe_diagnostics["unique_particle_types"]),
                "unique_pdg_ids": lhe_diagnostics["unique_pdg_ids"],
                "powheg_lhe_products_generated": True,
                "powheg_lhe_particle_summary": lhe_diagnostics["particle_summary"],
                "powheg_lhe_message": "These are POWHEG hard-process/LHE particles. They are not hadronized final-state particles. PYTHIA has not been invoked.",
                "pythia_invoked": False,
                "geant4_invoked": False,
                "photon_transport_invoked": False,
                "spectra_invoked": False,
                "expensive_event_generation_invoked": False,
            }
        )
        summary["products"].update(lhe_diagnostics["products"])
        validation_report.update(
            {
                "lhe_parser_invoked": True,
                "lhe_particles_are_hard_process": True,
                "hadronization_invoked": False,
                "n_lhe_events_parsed": int(lhe_diagnostics["n_lhe_events_parsed"]),
                "n_lhe_particles": int(lhe_diagnostics["n_lhe_particles"]),
                "n_final_state_particles": int(lhe_diagnostics["n_final_state_particles"]),
                "unique_particle_types": int(lhe_diagnostics["unique_particle_types"]),
            }
        )
        _write_jsonl(requests_path, requests)
    else:
        summary["powheg_validation_report_generated"] = True

    write_json(validation_report_path, validation_report)
    _draw_job_summary(summary, output_dir / "powheg_job_summary.png")
    write_json(summary_path, summary)
    write_json(report_path, summary)
    return summary
