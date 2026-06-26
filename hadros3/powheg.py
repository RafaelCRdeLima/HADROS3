"""H3-W9a POWHEG dry-run orchestration and diagnostics."""

from __future__ import annotations

import json
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


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
    ax.set_title("POWHEG dry-run input card preview", color="#111827")
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
        ax.text(0.5, 0.5, "No POWHEG dry-run requests", transform=ax.transAxes, ha="center", va="center")
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
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("count")
    ax.set_title("POWHEG dry-run job summary")
    ax.text(
        0.02,
        0.95,
        "Dry Run\npwhg_main NOT executed\nLHE generated: NO",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


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
        }
    )
    summary.update(
        {
            "products": products,
            "powheg_dry_run_invoked": True,
            "powheg_invoked": False,
            "pwhg_main_executed": False,
            "powheg_lhe_generated": False,
            "lhe_found": False,
            "powheg_card_preview_generated": True,
            "powheg_energy_distribution_generated": True,
            "powheg_job_summary_generated": True,
            "powheg_event_requests_generated": True,
            "powheg_input_cards_generated": int(summary.get("powheg_cards_generated", len(requests))),
            "pythia_invoked": False,
            "geant4_invoked": False,
            "photon_transport_invoked": False,
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
    requests = _read_jsonl(requests_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    first_card = output_dir / "powheg_input_cards" / (requests[0]["powheg_request_id"] if requests else "H3PWHG-000001") / "powheg.input"
    _draw_card_preview(first_card, output_dir / "powheg_card_preview.png")
    _draw_energy_distribution(requests, output_dir / "powheg_energy_distribution.png")
    _draw_job_summary(summary, output_dir / "powheg_job_summary.png")

    summary = _augment_summary(summary, output_dir, requests)
    write_json(summary_path, summary)
    write_json(report_path, summary)
    return summary
