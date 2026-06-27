"""H3-W9 POWHEG orchestration and diagnostics."""

from __future__ import annotations

import json
import csv
import html
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

PDG_LATEX_NAMES = {
    -16: r"\bar{\nu}_{\tau}",
    -15: r"\tau^{+}",
    -14: r"\bar{\nu}_{\mu}",
    -13: r"\mu^{+}",
    -12: r"\bar{\nu}_{e}",
    -11: r"e^{+}",
    -6: r"\bar{t}",
    -5: r"\bar{b}",
    -4: r"\bar{c}",
    -3: r"\bar{s}",
    -2: r"\bar{u}",
    -1: r"\bar{d}",
    1: "d",
    2: "u",
    3: "s",
    4: "c",
    5: "b",
    6: "t",
    11: r"e^{-}",
    12: r"\nu_{e}",
    13: r"\mu^{-}",
    14: r"\nu_{\mu}",
    15: r"\tau^{-}",
    16: r"\nu_{\tau}",
    21: "g",
    22: r"\gamma",
    23: "Z^{0}",
    24: "W^{+}",
    -24: "W^{-}",
    90: r"\mathrm{system}",
    91: r"\mathrm{cluster}",
    92: r"\mathrm{string}",
    2212: "p",
    -2212: r"\bar{p}",
    2112: "n",
    -2112: r"\bar{n}",
}

PDG_DISPLAY_NAMES = {
    -16: "ν̄τ",
    -15: "τ⁺",
    -14: "ν̄μ",
    -13: "μ⁺",
    -12: "ν̄ₑ",
    -11: "e⁺",
    -6: "t̄",
    -5: "b̄",
    -4: "c̄",
    -3: "s̄",
    -2: "ū",
    -1: "d̄",
    1: "d",
    2: "u",
    3: "s",
    4: "c",
    5: "b",
    6: "t",
    11: "e⁻",
    12: "νₑ",
    13: "μ⁻",
    14: "νμ",
    15: "τ⁻",
    16: "ντ",
    21: "g",
    22: "γ",
    23: "Z⁰",
    24: "W⁺",
    -24: "W⁻",
    90: "system",
    91: "cluster",
    92: "string",
    2212: "p",
    -2212: "p̄",
    2112: "n",
    -2112: "n̄",
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


def _particle_latex_name(pdg_id: int) -> str:
    if pdg_id in PDG_LATEX_NAMES:
        return PDG_LATEX_NAMES[pdg_id]
    if pdg_id < 0:
        return rf"\overline{{\mathrm{{PDG}}\,{abs(pdg_id)}}}"
    return rf"\mathrm{{PDG}}\,{pdg_id}"


def _particle_display_name(pdg_id: int) -> str:
    if pdg_id in PDG_DISPLAY_NAMES:
        return PDG_DISPLAY_NAMES[pdg_id]
    if pdg_id < 0:
        return f"PDG {pdg_id}"
    return f"PDG {pdg_id}"


def _particle_category(pdg_id: int) -> str:
    apdg = abs(pdg_id)
    if apdg in {11, 12, 13, 14, 15, 16}:
        return "lepton"
    if 1 <= apdg <= 6:
        return "quark"
    if apdg == 21:
        return "gluon"
    if apdg in {22, 23, 24}:
        return "boson"
    if apdg in {2212, 2112}:
        return "hadron"
    return "other"


def _state_name(status: int) -> str:
    if status == -1:
        return "incoming"
    if status == 1:
        return "outgoing"
    return "intermediate"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


def _sqrt_s_gev(initial_particles: list[dict[str, Any]]) -> float:
    energy = sum(float(row["energy_gev"]) for row in initial_particles)
    px = sum(float(row["px_gev"]) for row in initial_particles)
    py = sum(float(row["py_gev"]) for row in initial_particles)
    pz = sum(float(row["pz_gev"]) for row in initial_particles)
    s = energy * energy - px * px - py * py - pz * pz
    return math.sqrt(max(s, 0.0))


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
        header_parts = lines[0].split()
        try:
            n_particles = int(float(header_parts[0]))
        except (IndexError, ValueError):
            n_particles = max(0, len(lines) - 1)
        try:
            event_weight = _parse_lhe_float(header_parts[2])
        except (IndexError, ValueError):
            event_weight = 0.0
        try:
            event_scale_gev = _parse_lhe_float(header_parts[3])
        except (IndexError, ValueError):
            event_scale_gev = 0.0
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
                "particle_latex": _particle_latex_name(pdg_id),
                "particle_display": _particle_display_name(pdg_id),
                "particle_category": _particle_category(pdg_id),
                "state": _state_name(status),
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
        incoming_names = [str(row["particle_name"]) for row in initial_particles]
        outgoing_names = [str(row["particle_name"]) for row in final_particles]
        incoming_display_names = [str(row["particle_display"]) for row in initial_particles]
        outgoing_display_names = [str(row["particle_display"]) for row in final_particles]
        events.append(
            {
                "powheg_job_id": job_id,
                "lhe_event_index": event_index,
                "event_id": f"{job_id}:{event_index}",
                "event_weight": event_weight,
                "event_scale_gev": event_scale_gev,
                "sqrt_s_gev": _sqrt_s_gev(initial_particles),
                "n_particles": len(event_particles),
                "n_initial_state": len(initial_particles),
                "n_final_state": len(final_particles),
                "sum_final_energy_gev": sum(float(row["energy_gev"]) for row in final_particles),
                "sum_final_px_gev": sum(float(row["px_gev"]) for row in final_particles),
                "sum_final_py_gev": sum(float(row["py_gev"]) for row in final_particles),
                "sum_final_pz_gev": sum(float(row["pz_gev"]) for row in final_particles),
                "pdg_ids": [int(row["pdg_id"]) for row in event_particles],
                "particle_names": [str(row["particle_name"]) for row in event_particles],
                "particle_display_names": [str(row["particle_display"]) for row in event_particles],
                "incoming_particles": incoming_names,
                "outgoing_particles": outgoing_names,
                "incoming_particles_display": incoming_display_names,
                "outgoing_particles_display": outgoing_display_names,
                "raw_lhe_event": text[start : end + len("</event>")],
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
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=300)
    if energies:
        scaled = [energy / 1.0e9 for energy in energies]
        ax.hist(scaled, bins=min(20, max(3, len(scaled))), color="#2563eb", alpha=0.82)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.tick_params(axis="x", labelrotation=0)
        sorted_energies = sorted(energies)
        median = sorted_energies[len(sorted_energies) // 2] if len(sorted_energies) % 2 else 0.5 * (sorted_energies[len(sorted_energies) // 2 - 1] + sorted_energies[len(sorted_energies) // 2])
        mean = _mean(energies)
        std = math.sqrt(_mean([(energy - mean) ** 2 for energy in energies]))
        ax.text(
            0.98,
            0.95,
            "\n".join(
                [
                    f"N candidates = {len(energies)}",
                    f"mean = {mean / 1.0e9:.3f} x 10^9 GeV",
                    f"median = {median / 1.0e9:.3f} x 10^9 GeV",
                    f"std = {std / 1.0e9:.3f} x 10^9 GeV",
                    f"range = [{min(energies) / 1.0e9:.3f}, {max(energies) / 1.0e9:.3f}]",
                ]
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
        )
    else:
        ax.text(0.5, 0.5, "No POWHEG requests", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel(r"$E_{\nu,\mathrm{local}}$ [$10^9$ GeV]")
    ax.set_ylabel("POWHEG jobs")
    ax.set_title("Local neutrino energy distribution submitted to POWHEG")
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
                "particle_latex": _particle_latex_name(pdg_id),
                "particle_display": _particle_display_name(pdg_id),
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
        "particle_display",
        "particle_latex",
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


def _write_particle_table_csv(path: Path, particles: list[dict[str, Any]]) -> None:
    fieldnames = [
        "powheg_job_id",
        "lhe_event_index",
        "particle_index",
        "particle_name",
        "particle_display",
        "particle_latex",
        "pdg_id",
        "status",
        "state",
        "mother1",
        "mother2",
        "px_gev",
        "py_gev",
        "pz_gev",
        "energy_gev",
        "mass_gev",
        "pt_gev",
        "eta",
        "phi",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in particles:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_particle_table_html(path: Path, particles: list[dict[str, Any]]) -> None:
    columns = [
        "particle_display",
        "particle_name",
        "pdg_id",
        "status",
        "state",
        "mother1",
        "mother2",
        "px_gev",
        "py_gev",
        "pz_gev",
        "energy_gev",
        "mass_gev",
        "pt_gev",
        "eta",
        "phi",
    ]

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6e}"
        return str(value)

    rows_html = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(fmt(row.get(column, '')))}</td>" for column in columns) + "</tr>"
        for row in particles
    )
    header_html = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>POWHEG Particle Table</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f8fafc; color: #0f172a; }}
    main {{ padding: 14px; }}
    h1 {{ margin: 0 0 8px; font-size: 20px; }}
    p {{ margin: 0 0 12px; color: #475569; }}
    .table-wrap {{ overflow: auto; border: 1px solid #cbd5e1; border-radius: 8px; background: white; max-height: 520px; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 1080px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 7px 9px; text-align: right; white-space: nowrap; }}
    th {{ position: sticky; top: 0; background: #eaf0f7; color: #0f172a; z-index: 1; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(5), td:nth-child(5) {{ text-align: left; }}
    tbody tr:nth-child(even) {{ background: #f8fafc; }}
  </style>
</head>
<body>
<main>
  <h1>POWHEG Particle Table</h1>
  <p>Automatically generated in <code>POWHEG/</code>. The CSV audit file is saved separately as <code>powheg_particle_table.csv</code>.</p>
  <div class="table-wrap">
    <table>
      <thead><tr>{header_html}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def _write_event_summary_table_csv(path: Path, events: list[dict[str, Any]]) -> None:
    fieldnames = [
        "event",
        "powheg_job_id",
        "lhe_event_index",
        "incoming",
        "outgoing",
        "weight",
        "sqrt_s_gev",
        "n_final_particles",
        "sum_final_energy_gev",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "event": event.get("event_id", ""),
                    "powheg_job_id": event.get("powheg_job_id", ""),
                    "lhe_event_index": event.get("lhe_event_index", ""),
                    "incoming": ", ".join(event.get("incoming_particles_display", event.get("incoming_particles", []))),
                    "outgoing": ", ".join(event.get("outgoing_particles_display", event.get("outgoing_particles", []))),
                    "weight": event.get("event_weight", 0.0),
                    "sqrt_s_gev": event.get("sqrt_s_gev", 0.0),
                    "n_final_particles": event.get("n_final_state", 0),
                    "sum_final_energy_gev": event.get("sum_final_energy_gev", 0.0),
                }
            )


def _physics_summary(particles: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    incoming = [row for row in particles if int(row["status"]) == -1]
    outgoing = [row for row in particles if int(row["status"]) == 1]
    final_energies = [float(row["energy_gev"]) for row in outgoing]
    pts = [float(row["pt_gev"]) for row in outgoing]
    weights = [float(event.get("event_weight", 0.0)) for event in events]
    multiplicities = [float(event.get("n_final_state", 0)) for event in events]
    return {
        "incoming_particles": sorted({_particle_display_name(int(row["pdg_id"])) for row in incoming}),
        "outgoing_particles": sorted({_particle_display_name(int(row["pdg_id"])) for row in outgoing}),
        "unique_particle_species": sorted({_particle_display_name(int(row["pdg_id"])) for row in particles}),
        "incoming_particles_raw": sorted({_particle_name(int(row["pdg_id"])) for row in incoming}),
        "outgoing_particles_raw": sorted({_particle_name(int(row["pdg_id"])) for row in outgoing}),
        "unique_particle_species_raw": sorted({_particle_name(int(row["pdg_id"])) for row in particles}),
        "average_event_energy_gev": _mean([float(event.get("sum_final_energy_gev", 0.0)) for event in events]),
        "average_event_weight": _mean(weights),
        "average_pt_gev": _mean(pts),
        "average_multiplicity": _mean(multiplicities),
        "max_final_energy_gev": max(final_energies) if final_energies else 0.0,
        "n_incoming_particles": len(incoming),
        "n_outgoing_particles": len(outgoing),
    }


def _draw_lhe_particle_histogram(summary_rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=300)
    if summary_rows:
        labels = [f"${_particle_latex_name(int(row['pdg_id']))}$\nPDG {row['pdg_id']}" for row in summary_rows]
        counts = [int(row["count"]) for row in summary_rows]
        colors = {
            "lepton": "#1f77b4",
            "quark": "#d62728",
            "gluon": "#2ca02c",
            "boson": "#9467bd",
            "hadron": "#8c564b",
            "other": "#7f7f7f",
        }
        bar_colors = [colors.get(_particle_category(int(row["pdg_id"])), "#7f7f7f") for row in summary_rows]
        ax.bar(labels, counts, color=bar_colors, alpha=0.88, edgecolor="#111827", linewidth=0.5)
        ax.tick_params(axis="x", labelrotation=35)
        total = sum(counts)
        table_rows = [
            [
                _particle_display_name(int(row["pdg_id"])),
                str(row["pdg_id"]),
                str(row["count"]),
                str(row["initial_state_count"]),
                str(row["final_state_count"]),
                f"{(100.0 * int(row['count']) / total) if total else 0.0:.1f}%",
            ]
            for row in summary_rows[:8]
        ]
        table = ax.table(
            cellText=table_rows,
            colLabels=["Particle", "PDG", "Count", "Initial", "Final", "Fraction"],
            loc="upper right",
            bbox=[0.52, 0.43, 0.46, 0.50],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.8)
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=color, label=label)
            for label, color in [("leptons", "#1f77b4"), ("quarks", "#d62728"), ("gluons", "#2ca02c"), ("bosons", "#9467bd")]
        ]
        ax.legend(handles=handles, frameon=False, loc="upper left")
    else:
        ax.text(0.5, 0.5, "No LHE particles parsed", transform=ax.transAxes, ha="center", va="center")
    ax.set_ylabel("particle count")
    ax.set_title("POWHEG LHE hard-process particle content")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_lhe_energy_spectrum(particles: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.2), dpi=300, sharex=True)
    initial = [float(row["energy_gev"]) for row in particles if int(row["status"]) == -1 and float(row["energy_gev"]) > 0.0]
    final = [float(row["energy_gev"]) for row in particles if int(row["status"]) == 1 and float(row["energy_gev"]) > 0.0]
    for ax, data, title, color in [
        (axes[0], initial, "Incoming particles", "#1f77b4"),
        (axes[1], final, "Outgoing hard-process particles", "#d62728"),
    ]:
        if data:
            ax.hist(data, bins=min(18, max(3, len(data))), color=color, alpha=0.78, edgecolor="#111827", linewidth=0.4)
            ax.set_xscale("log")
            ax.text(
                0.98,
                0.92,
                f"N={len(data)}\nmean={_mean(data):.3e} GeV\nmax={max(data):.3e} GeV",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
            )
        else:
            ax.text(0.5, 0.5, "No positive energies", transform=ax.transAxes, ha="center", va="center")
        ax.set_ylabel("particles")
        ax.set_title(title, loc="left", fontsize=10)
    axes[-1].set_xlabel("energy [GeV]")
    fig.suptitle("POWHEG LHE particle energy spectrum", y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_lhe_momentum_spectrum(particles: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.2), dpi=300, sharex=False)
    p_abs = [float(row["p_abs_gev"]) for row in particles if float(row["p_abs_gev"]) > 0.0]
    pt = [float(row["pt_gev"]) for row in particles if float(row["pt_gev"]) > 0.0]
    for ax, data, title, color in [
        (axes[0], p_abs, "Momentum magnitude |p|", "#2ca02c"),
        (axes[1], pt, "Transverse momentum $p_T$", "#ff7f0e"),
    ]:
        if data:
            ax.hist(data, bins=min(18, max(3, len(data))), color=color, alpha=0.78, edgecolor="#111827", linewidth=0.4)
            ax.set_xscale("log")
            ax.text(
                0.98,
                0.92,
                f"N={len(data)}\nmean={_mean(data):.3e}\nRMS={_rms(data):.3e}\nmin={min(data):.3e}\nmax={max(data):.3e}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
            )
        else:
            ax.text(0.5, 0.5, "No positive momenta", transform=ax.transAxes, ha="center", va="center")
        ax.set_xlabel("momentum [GeV]")
        ax.set_ylabel("particles")
        ax.set_title(title, loc="left", fontsize=10)
    fig.suptitle("POWHEG LHE particle momentum spectra", y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_hard_process_event_display(events: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = events[0] if events else {}
    incoming = list(event.get("incoming_particles_display", event.get("incoming_particles", [])))
    outgoing = list(event.get("outgoing_particles_display", event.get("outgoing_particles", [])))
    outgoing_raw = list(event.get("outgoing_particles", []))
    if len(outgoing_raw) != len(outgoing):
        outgoing_raw = list(outgoing)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Hard Process Event Display", loc="left", fontsize=14, fontweight="bold")
    ax.text(0.02, 0.91, f"Event: {event.get('event_id', 'none')}", fontsize=9, color="#475569")
    if not event:
        ax.text(0.5, 0.5, "No LHE event available", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    def draw_node(x: float, y: float, label: str, color: str) -> None:
        ax.scatter([x], [y], s=1800, color=color, edgecolor="#111827", linewidth=1.0, zorder=3)
        ax.text(x, y, label, ha="center", va="center", fontsize=10, color="white", zorder=4)

    incoming_y = [0.68, 0.32] if len(incoming) <= 2 else [0.76 - i * 0.18 for i in range(len(incoming))]
    outgoing_y = [0.78 - i * (0.56 / max(1, len(outgoing) - 1)) for i in range(len(outgoing))] if outgoing else []
    for y, name in zip(incoming_y, incoming):
        draw_node(0.16, y, name, "#1f77b4")
        ax.annotate("", xy=(0.45, 0.50), xytext=(0.24, y), arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#334155"})
    ax.scatter([0.50], [0.50], s=2600, color="#111827", edgecolor="#64748b", linewidth=1.2, zorder=3)
    ax.text(0.50, 0.50, "DIS\nhard\nvertex", ha="center", va="center", fontsize=10, color="white", zorder=4)
    for y, name, raw_name in zip(outgoing_y, outgoing, outgoing_raw):
        draw_node(0.84, y, name, "#d62728" if raw_name in {"d", "u", "s", "c", "b", "t", "dbar", "ubar", "sbar", "cbar", "bbar", "tbar"} else "#2ca02c")
        ax.annotate("", xy=(0.76, y), xytext=(0.55, 0.50), arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#334155"})
    ax.text(
        0.02,
        0.06,
        f"weight={float(event.get('event_weight', 0.0)):.3e}   sqrt(s)={float(event.get('sqrt_s_gev', 0.0)):.3e} GeV   final particles={event.get('n_final_state', 0)}",
        fontsize=9,
        color="#334155",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_hard_process_event_display_viewer(path: Path, events: list[dict[str, Any]]) -> None:
    payload: list[dict[str, Any]] = []
    for event in events:
        incoming_display = list(event.get("incoming_particles_display", event.get("incoming_particles", [])))
        outgoing_display = list(event.get("outgoing_particles_display", event.get("outgoing_particles", [])))
        incoming_raw = list(event.get("incoming_particles", incoming_display))
        outgoing_raw = list(event.get("outgoing_particles", outgoing_display))
        payload.append(
            {
                "event_id": event.get("event_id", f"event-{event.get('lhe_event_index', len(payload) + 1)}"),
                "lhe_event_index": event.get("lhe_event_index", len(payload) + 1),
                "event_weight": float(event.get("event_weight", 0.0)),
                "sqrt_s_gev": float(event.get("sqrt_s_gev", 0.0)),
                "n_final_state": int(event.get("n_final_state", 0)),
                "incoming": [
                    {"label": str(label), "raw": str(raw)}
                    for label, raw in zip(incoming_display, incoming_raw, strict=False)
                ],
                "outgoing": [
                    {"label": str(label), "raw": str(raw)}
                    for label, raw in zip(outgoing_display, outgoing_raw, strict=False)
                ],
            }
        )
    event_json = json.dumps(payload, ensure_ascii=False)
    html_text = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>POWHEG Hard Process Event Display</title>
  <style>
    :root { color-scheme: light; }
    body { margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #ffffff; color: #0f172a; }
    main { padding: 14px 16px 16px; }
    svg { width: 100%; height: auto; display: block; background: #ffffff; border: 1px solid #d7dee8; border-radius: 8px; }
    .controls { display: flex; align-items: center; gap: 10px; margin-top: 10px; flex-wrap: wrap; }
    label { font-weight: 700; color: #0f172a; }
    select { min-width: 280px; max-width: 100%; border: 1px solid #94a3b8; border-radius: 6px; padding: 7px 9px; background: white; color: #0f172a; }
    .meta { color: #475569; font-size: 13px; }
  </style>
</head>
<body>
<main>
  <svg id="display" viewBox="0 0 1200 720" role="img" aria-label="POWHEG hard process event display"></svg>
  <div class="controls">
    <label for="event-select">Hard process event</label>
    <select id="event-select"></select>
    <span id="event-meta" class="meta"></span>
  </div>
</main>
<script>
const events = __EVENTS__;
const svg = document.getElementById("display");
const select = document.getElementById("event-select");
const meta = document.getElementById("event-meta");
const quarks = new Set(["d","u","s","c","b","t","dbar","ubar","sbar","cbar","bbar","tbar"]);
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function yPositions(count, top, bottom) {
  if (!count) return [];
  if (count === 1) return [(top + bottom) / 2];
  return Array.from({length: count}, (_, i) => top + i * ((bottom - top) / (count - 1)));
}
function node(x, y, label, color) {
  return `<circle cx="${x}" cy="${y}" r="46" fill="${color}" stroke="#111827" stroke-width="3"/>` +
    `<text x="${x}" y="${y + 9}" text-anchor="middle" font-size="28" fill="white">${esc(label)}</text>`;
}
function arrow(x1, y1, x2, y2) {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#334155" stroke-width="5" stroke-linecap="round" marker-end="url(#arrow)"/>`;
}
function render(index) {
  const event = events[index];
  if (!event) {
    svg.innerHTML = `<text x="600" y="360" text-anchor="middle" font-size="28" fill="#475569">No LHE event available</text>`;
    meta.textContent = "No events";
    return;
  }
  const incoming = event.incoming || [];
  const outgoing = event.outgoing || [];
  const inY = yPositions(incoming.length, 255, 465);
  const outY = yPositions(outgoing.length, 190, 530);
  let parts = `
    <defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#334155"/></marker></defs>
    <text x="34" y="54" font-size="36" font-weight="800" fill="#0f172a">Hard Process Event Display</text>
    <text x="34" y="112" font-size="22" fill="#475569">Event: ${esc(event.event_id)}</text>
    <circle cx="600" cy="360" r="58" fill="#111827" stroke="#64748b" stroke-width="4"/>
    <text x="600" y="344" text-anchor="middle" font-size="22" fill="white">DIS</text>
    <text x="600" y="374" text-anchor="middle" font-size="22" fill="white">hard</text>
    <text x="600" y="404" text-anchor="middle" font-size="22" fill="white">vertex</text>`;
  incoming.forEach((particle, i) => {
    const y = inY[i];
    parts += node(200, y, particle.label, "#1f77b4");
    parts += arrow(300, y, 535, 360);
  });
  outgoing.forEach((particle, i) => {
    const y = outY[i];
    const color = quarks.has(particle.raw) ? "#d62728" : "#16a34a";
    parts += arrow(665, 360, 900, y);
    parts += node(1000, y, particle.label, color);
  });
  parts += `<text x="34" y="674" font-size="22" fill="#334155">weight=${Number(event.event_weight || 0).toExponential(3)}   sqrt(s)=${Number(event.sqrt_s_gev || 0).toExponential(3)} GeV   final particles=${event.n_final_state || 0}</text>`;
  svg.innerHTML = parts;
  meta.textContent = `Event ${index + 1} / ${events.length}`;
}
events.forEach((event, index) => {
  const option = document.createElement("option");
  option.value = String(index);
  option.textContent = `${index + 1}. ${event.event_id} — ${event.n_final_state || 0} final particles`;
  select.appendChild(option);
});
select.onchange = () => render(Number(select.value || 0));
render(0);
</script>
</body>
</html>
""".replace("__EVENTS__", event_json)
    path.write_text(html_text, encoding="utf-8")


def _write_lhe_event_viewer(path: Path, events: list[dict[str, Any]]) -> None:
    event_blocks = [str(event.get("raw_lhe_event", "")) for event in events]
    event_json = json.dumps(event_blocks)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>POWHEG Raw LHE Event Viewer</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
    main {{ max-width: 980px; margin: 0 auto; }}
    button {{ border: 1px solid #94a3b8; background: white; border-radius: 6px; padding: 8px 12px; margin-right: 8px; cursor: pointer; }}
    pre {{ background: #0b1020; color: #e5e7eb; padding: 16px; border-radius: 8px; overflow: auto; line-height: 1.35; }}
    .tag {{ color: #93c5fd; }}
    .meta {{ color: #475569; }}
  </style>
</head>
<body>
<main>
  <h1>Raw LHE Event</h1>
  <p class="meta">Shows one <code>&lt;event&gt;</code> block at a time. This is hard-process LHE content, not hadronized final-state particles.</p>
  <p><button id="prev">Previous Event</button><button id="next">Next Event</button> <span id="label"></span></p>
  <pre id="event"></pre>
</main>
<script>
const events = {event_json};
let index = 0;
function esc(s) {{ return s.replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c])); }}
function render() {{
  const block = events[index] || "No LHE event available.";
  document.getElementById("label").textContent = events.length ? `Event ${{index + 1}} / ${{events.length}}` : "No events";
  document.getElementById("event").innerHTML = esc(block).replace(/(&lt;\\/?event&gt;)/g, '<span class="tag">$1</span>');
}}
document.getElementById("prev").onclick = () => {{ if (events.length) {{ index = (index + events.length - 1) % events.length; render(); }} }};
document.getElementById("next").onclick = () => {{ if (events.length) {{ index = (index + 1) % events.length; render(); }} }};
render();
</script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def _write_particle_content_report(path: Path, particles: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    unique_pdgs = sorted({int(row["pdg_id"]) for row in particles})
    quark_pdgs = sorted({pdg for pdg in unique_pdgs if 1 <= abs(pdg) <= 6})
    report = {
        "process": "POWHEG nudis hard-process LHE",
        "lhe_particles_are_hard_process": True,
        "hadronization_invoked": False,
        "pythia_invoked": False,
        "geant4_invoked": False,
        "photon_transport_invoked": False,
        "unique_pdg_ids": unique_pdgs,
        "quark_pdg_ids": quark_pdgs,
        "why_u_d_c_s_can_appear": (
            "The POWHEG DIS process samples partonic initial states from the configured PDFs and may produce quark flavors allowed by "
            "the neutrino DIS hard process. In charged-current neutrino scattering the flavor structure includes weak charged-current "
            "transitions; CKM-weighted channels and PDF flavor content can make u, d, s, and c quarks or antiquarks appear in the LHE "
            "hard process. These are parton-level hard-process records, not hadronized particles."
        ),
        "cc_nc_note": (
            "The current HADROS3 POWHEG card records channel_type and vtype from the configured nudis template. Inspect each generated "
            "powheg.input for the exact CC/NC configuration used by that job."
        ),
        "pdf_note": "Incoming partons are sampled from the configured LHAPDF set; the observed flavor mix reflects PDF support and POWHEG matrix-element channel selection.",
        "n_events": len(events),
        "n_particles": len(particles),
    }
    write_json(path, report)
    return report


def _write_lhe_diagnostics(particles: list[dict[str, Any]], events: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    particle_summary = _aggregate_particle_summary(particles)
    particles_path = output_dir / "powheg_lhe_particles.jsonl"
    events_path = output_dir / "powheg_lhe_events_summary.jsonl"
    summary_json_path = output_dir / "powheg_lhe_particle_summary.json"
    summary_csv_path = output_dir / "powheg_lhe_particle_summary.csv"
    histogram_path = output_dir / "powheg_lhe_particle_histogram.png"
    energy_path = output_dir / "powheg_lhe_energy_spectrum.png"
    momentum_path = output_dir / "powheg_lhe_momentum_spectrum.png"
    event_display_path = output_dir / "powheg_hard_process_event_display.png"
    event_display_viewer_path = output_dir / "powheg_hard_process_event_display_view.html"
    event_summary_table_path = output_dir / "powheg_event_summary_table.csv"
    particle_table_path = output_dir / "powheg_particle_table.csv"
    particle_table_html_path = output_dir / "powheg_particle_table.html"
    content_report_path = output_dir / "powheg_particle_content_report.json"
    event_viewer_path = output_dir / "powheg_lhe_event_view.html"
    _write_jsonl(particles_path, particles)
    _write_jsonl(events_path, events)
    write_json(summary_json_path, particle_summary)
    _write_particle_summary_csv(summary_csv_path, particle_summary)
    _write_event_summary_table_csv(event_summary_table_path, events)
    _write_particle_table_csv(particle_table_path, particles)
    _write_particle_table_html(particle_table_html_path, particles)
    _draw_lhe_particle_histogram(particle_summary, histogram_path)
    _draw_lhe_energy_spectrum(particles, energy_path)
    _draw_lhe_momentum_spectrum(particles, momentum_path)
    _draw_hard_process_event_display(events, event_display_path)
    _write_hard_process_event_display_viewer(event_display_viewer_path, events)
    _write_lhe_event_viewer(event_viewer_path, events)
    content_report = _write_particle_content_report(content_report_path, particles, events)
    physics_summary = _physics_summary(particles, events)
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
        "physics_summary": physics_summary,
        "particle_content_report": content_report,
        "products": {
            "powheg_lhe_particles": str(particles_path),
            "powheg_lhe_events_summary": str(events_path),
            "powheg_lhe_particle_summary_csv": str(summary_csv_path),
            "powheg_lhe_particle_summary_json": str(summary_json_path),
            "powheg_lhe_particle_histogram": str(histogram_path),
            "powheg_lhe_energy_spectrum": str(energy_path),
            "powheg_lhe_momentum_spectrum": str(momentum_path),
            "powheg_hard_process_event_display": str(event_display_path),
            "powheg_hard_process_event_display_view": str(event_display_viewer_path),
            "powheg_event_summary_table": str(event_summary_table_path),
            "powheg_particle_table": str(particle_table_path),
            "powheg_particle_table_html": str(particle_table_html_path),
            "powheg_particle_content_report": str(content_report_path),
            "powheg_lhe_event_view": str(event_viewer_path),
        },
    }


def generate_lhe_diagnostics(lhe_path: Path, output_dir: Path, *, powheg_job_id: str | None = None) -> dict[str, Any]:
    particles, events = parse_lhe_particles(lhe_path, powheg_job_id=powheg_job_id)
    return _write_lhe_diagnostics(particles, events, output_dir)


def generate_lhe_diagnostics_for_paths(lhe_paths: list[Path], output_dir: Path) -> dict[str, Any]:
    particles: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for lhe_path in lhe_paths:
        job_particles, job_events = parse_lhe_particles(lhe_path, powheg_job_id=lhe_path.parent.name)
        particles.extend(job_particles)
        events.extend(job_events)
    return _write_lhe_diagnostics(particles, events, output_dir)


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


def _run_powheg_request(output_dir: Path, request: dict[str, Any], run_mode: str) -> dict[str, Any]:
    request_id = str(request["powheg_request_id"])
    card_path = output_dir.parent / str(request["powheg_input_path"])
    if not card_path.exists():
        raise FileNotFoundError(f"POWHEG input card not found for {run_mode}: {card_path}")

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
            raise RuntimeError(f"POWHEG {run_mode} failed for {request_id} with exit code {exc.returncode}. See {log_path}") from exc

    if not produced_lhe.exists() or produced_lhe.stat().st_size == 0:
        raise RuntimeError(f"POWHEG {run_mode} did not create a non-empty LHE file for {request_id}: {produced_lhe}. See {log_path}")
    text = produced_lhe.read_text(encoding="utf-8", errors="replace")
    n_events = _count_lhe_events(text)
    valid = "<LesHouchesEvents" in text and "</LesHouchesEvents>" in text and n_events > 0
    if not valid:
        raise RuntimeError(f"POWHEG {run_mode} produced an invalid LHE file for {request_id}: {produced_lhe}. See {log_path}")
    shutil.copy2(produced_lhe, final_lhe)

    request.update(
        {
            "powheg_status": f"{run_mode}_lhe_generated",
            "powheg_invoked": True,
            "pwhg_main_executed": True,
            "powheg_lhe_generated": True,
            "powheg_lhe_path": str(final_lhe.relative_to(output_dir.parent)),
            "powheg_log_path": str(log_path.relative_to(output_dir.parent)),
            "n_lhe_events": n_events,
        }
    )
    return {
        "powheg_request_id": request_id,
        "powheg_lhe_path": str(final_lhe),
        "powheg_log_path": str(log_path),
        "powheg_work_dir": str(work_dir),
        "n_lhe_events": n_events,
        "lhe_valid": True,
    }


def _run_real_powheg(output_dir: Path, requests: list[dict[str, Any]], *, run_mode: str) -> dict[str, Any]:
    if not requests:
        raise RuntimeError(f"POWHEG {run_mode} requested, but no POWHEG request was prepared.")
    if run_mode == "real_smoke" and len(requests) != 1:
        raise RuntimeError(f"POWHEG real_smoke must prepare exactly one request; got {len(requests)}.")
    if run_mode not in {"real_smoke", "real_free"}:
        raise ValueError(f"unsupported real POWHEG run_mode: {run_mode}")
    if not POWHEG_BINARY.exists():
        raise FileNotFoundError(f"Local POWHEG pwhg_main not found: {POWHEG_BINARY}. Run make powheg-fetch && make powheg-build first.")

    jobs = [_run_powheg_request(output_dir, request, run_mode) for request in requests]
    total_events = sum(int(job["n_lhe_events"]) for job in jobs)
    return {
        "powheg_validation_status": "ok",
        "powheg_request_id": str(requests[0]["powheg_request_id"]),
        "run_mode": run_mode,
        "powheg_run_mode": run_mode,
        "pwhg_main": str(POWHEG_BINARY),
        "pwhg_main_executed": True,
        "powheg_invoked": True,
        "powheg_lhe_generated": True,
        "lhe_found": True,
        "lhe_valid": True,
        "n_lhe_events": total_events,
        "n_powheg_jobs_run": len(jobs),
        "powheg_lhe_path": str(jobs[0]["powheg_lhe_path"]),
        "powheg_log_path": str(jobs[0]["powheg_log_path"]),
        "powheg_work_dir": str(jobs[0]["powheg_work_dir"]),
        "powheg_jobs": jobs,
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
            "powheg_hard_process_event_display": str(output_dir / "powheg_hard_process_event_display.png"),
            "powheg_hard_process_event_display_view": str(output_dir / "powheg_hard_process_event_display_view.html"),
            "powheg_event_summary_table": str(output_dir / "powheg_event_summary_table.csv"),
            "powheg_particle_table": str(output_dir / "powheg_particle_table.csv"),
            "powheg_particle_table_html": str(output_dir / "powheg_particle_table.html"),
            "powheg_particle_content_report": str(output_dir / "powheg_particle_content_report.json"),
            "powheg_lhe_event_view": str(output_dir / "powheg_lhe_event_view.html"),
        }
    )
    run_mode = str(summary.get("powheg_run_mode", "dry_run"))
    is_real_smoke = run_mode == "real_smoke"
    is_real_free = run_mode == "real_free"
    summary.update(
        {
            "products": products,
            "powheg_dry_run_invoked": run_mode == "dry_run",
            "powheg_real_smoke_invoked": is_real_smoke,
            "powheg_real_free_invoked": is_real_free,
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
            "powheg_physics_summary": {},
            "powheg_particle_content_report_generated": False,
            "powheg_hard_process_event_display_generated": False,
            "powheg_hard_process_event_display_view_generated": False,
            "powheg_lhe_event_view_generated": False,
            "n_powheg_jobs_requested": int(summary.get("max_powheg_events", len(requests))),
            "n_powheg_jobs_run": 0,
            "events_per_candidate_requested": int(summary.get("events_per_candidate", 0)),
            "n_lhe_events_total": 0,
            "real_free_mode": is_real_free,
            "real_smoke_safety_clamp": is_real_smoke,
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
    if run_mode in {"real_smoke", "real_free"}:
        validation_report = _run_real_powheg(output_dir, requests, run_mode=run_mode)
        lhe_paths = [Path(job["powheg_lhe_path"]) for job in validation_report["powheg_jobs"]]
        lhe_diagnostics = generate_lhe_diagnostics_for_paths(lhe_paths, output_dir)
        real_free = run_mode == "real_free"
        summary.update(
            {
                "stage_name": "H3-W9b POWHEG Real Free Mode" if real_free else "H3-W9b POWHEG Real Run Smoke Mode",
                "powheg_dry_run_invoked": False,
                "powheg_real_smoke_invoked": not real_free,
                "powheg_real_free_invoked": real_free,
                "powheg_invoked": True,
                "pwhg_main_executed": True,
                "powheg_lhe_generated": True,
                "lhe_found": True,
                "lhe_valid": True,
                "n_lhe_events": int(validation_report["n_lhe_events"]),
                "n_lhe_events_total": int(validation_report["n_lhe_events"]),
                "n_powheg_jobs": len(requests),
                "n_powheg_jobs_requested": int(summary.get("max_powheg_events", len(requests))),
                "n_powheg_jobs_run": int(validation_report["n_powheg_jobs_run"]),
                "events_per_candidate_requested": int(summary.get("events_per_candidate", 0)),
                "powheg_jobs_prepared": len(requests),
                "powheg_cards_generated": len(requests),
                "powheg_validation_report_generated": True,
                "powheg_validation_report": str(validation_report_path),
                "powheg_lhe_path": validation_report["powheg_lhe_path"],
                "powheg_log_path": validation_report["powheg_log_path"],
                "powheg_lhe_paths": [job["powheg_lhe_path"] for job in validation_report["powheg_jobs"]],
                "powheg_log_paths": [job["powheg_log_path"] for job in validation_report["powheg_jobs"]],
                "real_free_mode": real_free,
                "real_smoke_safety_clamp": not real_free,
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
                "powheg_physics_summary": lhe_diagnostics["physics_summary"],
                "powheg_particle_content_report": lhe_diagnostics["particle_content_report"],
                "powheg_particle_content_report_generated": True,
                "powheg_hard_process_event_display_generated": True,
                "powheg_hard_process_event_display_view_generated": True,
                "powheg_lhe_event_view_generated": True,
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
                "n_powheg_jobs_requested": int(summary.get("n_powheg_jobs_requested", len(requests))),
                "n_powheg_jobs_run": int(validation_report["n_powheg_jobs_run"]),
                "events_per_candidate_requested": int(summary.get("events_per_candidate_requested", 0)),
                "n_lhe_events_total": int(validation_report["n_lhe_events"]),
                "real_free_mode": real_free,
                "real_smoke_safety_clamp": not real_free,
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
