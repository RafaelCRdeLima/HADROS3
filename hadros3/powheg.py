"""H3-W9 POWHEG orchestration and diagnostics."""

from __future__ import annotations

import json
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
    }
    if run_mode == "real_smoke":
        validation_report = _run_real_smoke(output_dir, requests)
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
                "pythia_invoked": False,
                "geant4_invoked": False,
                "photon_transport_invoked": False,
                "spectra_invoked": False,
                "expensive_event_generation_invoked": False,
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
