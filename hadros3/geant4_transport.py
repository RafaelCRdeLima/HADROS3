"""H3-W11 local-material transport with Geant4 and HepMC3."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .environment import geant4_prefix
from .paths import event_generation_dir, geant4_dir
from .geant4_visualization import write_geant4_visualizations


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = geant4_prefix()
DATASETS = {
    "G4ABLADATA": "ABLA3.3",
    "G4CHANNELINGDATA": "CHANNELING2.0",
    "G4LEDATA": "EMLOW8.8",
    "G4ENSDFSTATEDATA": "ENSDFSTATE3.0",
    "G4INCLDATA": "INCL1.3",
    "G4NEUTRONHPDATA": "NDL4.7.1",
    "G4PARTICLEXSDATA": "PARTICLEXS4.2",
    "G4LEVELGAMMADATA": "PhotonEvaporation6.1.2",
    "G4PIIDATA": "PII1.3",
    "G4RADIOACTIVEDATA": "RadioactiveDecay6.1.2",
    "G4REALSURFACEDATA": "RealSurface2.2",
    "G4SAIDXSDATA": "SAIDDATA2.0",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _binary() -> Path:
    installed = ROOT / "bin" / "hadros3_geant4_transport"
    built = ROOT / "build" / "geant4" / "hadros3_geant4_transport"
    return installed if installed.exists() else built


def geant4_environment(prefix: Path | None = None) -> dict[str, str]:
    prefix = prefix or DEFAULT_PREFIX
    env = dict(os.environ)
    env["CONDA_PREFIX"] = str(prefix)
    env["PATH"] = str(prefix / "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(prefix / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    data_root = prefix / "share" / "Geant4" / "data"
    for variable, directory in DATASETS.items():
        env[variable] = str(data_root / directory)
    return env


def backend_availability(prefix: Path | None = None) -> dict[str, Any]:
    prefix = prefix or DEFAULT_PREFIX
    executable = _binary()
    config = prefix / "bin" / "geant4-config"
    version: str | None = None
    if config.exists():
        result = subprocess.run([str(config), "--version"], text=True, capture_output=True, check=False)
        version = result.stdout.strip() or None
    data_root = prefix / "share" / "Geant4" / "data"
    datasets = {
        variable: {"directory": directory, "path": str(data_root / directory), "available": (data_root / directory).is_dir()}
        for variable, directory in DATASETS.items()
    }
    return {
        "available": executable.is_file() and os.access(executable, os.X_OK) and version == "11.4.2" and all(v["available"] for v in datasets.values()),
        "backend_executable": str(executable),
        "backend_sha256": _sha256(executable) if executable.is_file() else None,
        "geant4_config": str(config),
        "geant4_version": version,
        "hepmc3_version": "3.3.1",
        "prefix": str(prefix),
        "datasets": datasets,
        "multithreading_available": True,
    }


def _config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _plots(events: list[dict[str, Any]], escaped: list[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    x = list(range(len(events)))
    ax.bar(x, [float(row.get("deposited_gev", 0.0)) for row in events], label="deposited")
    ax.bar(x, [float(row.get("escaped_total_gev", 0.0)) for row in events], bottom=[float(row.get("deposited_gev", 0.0)) for row in events], label="escaped")
    ax.set_xlabel("Geant4 event")
    ax.set_ylabel("energy [GeV]")
    ax.set_title("H3-W11 energy ledger")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "geant4_energy_balance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    energies = [float(row["energy_local_gev"]) for row in escaped if float(row.get("energy_local_gev", 0.0)) > 0.0]
    if energies:
        ax.hist(energies, bins=min(40, max(5, len(energies))))
        if max(energies) / max(min(energies), 1e-300) > 100:
            ax.set_xscale("log")
    ax.set_xlabel("escaping total energy [GeV]")
    ax.set_ylabel("count")
    ax.set_title("H3-W11 escaping particles")
    fig.tight_layout()
    fig.savefig(output / "geant4_escape_spectrum.png", dpi=150)
    plt.close(fig)


def _domain_violations(import_report: dict[str, Any]) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"unsupported_energy event=(?P<event>\d+) particle=(?P<particle>\d+) "
        r"pdg=(?P<pdg>-?\d+) energy_gev=(?P<energy>[0-9.eE+-]+) maximum_gev=(?P<maximum>[0-9.eE+-]+)"
    )
    rows: list[dict[str, Any]] = []
    for violation in import_report.get("violations", []):
        match = pattern.fullmatch(str(violation))
        if match:
            rows.append({
                "event_id": int(match.group("event")),
                "particle_id": int(match.group("particle")),
                "pdg_id": int(match.group("pdg")),
                "energy_gev": float(match.group("energy")),
                "validated_maximum_energy_gev": float(match.group("maximum")),
                "violation": str(violation),
            })
        elif str(violation).startswith("unsupported_"):
            rows.append({"violation": str(violation)})
    return rows


def _domain_audit_plot(rows: list[dict[str, Any]], output: Path) -> None:
    numeric = [row for row in rows if "energy_gev" in row]
    if not numeric:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    energies = [float(row["energy_gev"]) for row in numeric]
    ceiling = float(numeric[0]["validated_maximum_energy_gev"])
    labels = [f"{row['particle_id']} ({row['pdg_id']})" for row in numeric]
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.28 * len(numeric))))
    positions = list(range(len(numeric)))
    ax.barh(positions, energies, color="#c2410c")
    ax.axvline(ceiling, color="#1d4ed8", linestyle="--", linewidth=2, label=f"validated ceiling = {ceiling:g} GeV")
    ax.set_yticks(positions, labels=labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("input total energy [GeV]")
    ax.set_ylabel("HepMC3 particle ID (PDG)")
    ax.set_title("H3-W11 domain audit — particles refused before transport")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "geant4_domain_audit.png", dpi=160)
    plt.close(fig)

def _enrich_rows(
    raw_events: list[dict[str, Any]], raw_escaped: list[dict[str, Any]], raw_steps: list[dict[str, Any]],
    upstream_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_index = {index: row for index, row in enumerate(upstream_events)}
    fields = ("event_generation_event_id", "powheg_request_id", "lhe_event_index", "interaction_id", "xwgtup", "physics_weight", "observer_weight", "final_observation_score")
    for row in raw_events + raw_escaped + raw_steps:
        upstream = by_index.get(int(row["geant4_event_id"]), {})
        row.update({key: upstream.get(key) for key in fields})
        row["generator_frame"] = "local_matter_tetrad"
        row["momentum_unit"] = "GeV"
        row["length_unit"] = "mm"
    return raw_events, raw_escaped, raw_steps


def _resolve_density(
    cfg: dict[str, Any], run_output_dir: Path, upstream_events: list[dict[str, Any]], *, input_override: Path | None,
) -> tuple[float, str, list[dict[str, Any]]]:
    configured = float(cfg.get("density_g_cm3", 1.0))
    source = str(cfg.get("density_source", "dis_vertex_local"))
    if source == "configured_fixture" or input_override:
        return configured, "configured_fixture", []
    if source != "dis_vertex_local":
        raise ValueError(f"unsupported GEANT4 density source: {source}")
    rows = _read_jsonl(run_output_dir / "DIS" / "dis_accepted_interactions.jsonl")
    by_id = {str(row.get("interaction_id")): row for row in rows}
    selected: list[dict[str, Any]] = []
    densities: list[float] = []
    for event in upstream_events[: int(float(cfg.get("max_events", 2)))]:
        interaction_id = str(event.get("interaction_id"))
        row = by_id.get(interaction_id)
        if row is None:
            raise RuntimeError(f"no DIS interaction metadata found for H3-W10 event {event.get('event_generation_event_id')} ({interaction_id})")
        density = float(row.get("interaction_rho_g_cm3", row.get("interaction_point_rho_g_cm3", 0.0)))
        if not math.isfinite(density) or density <= 0.0:
            raise RuntimeError(f"invalid local DIS density for {interaction_id}: {density}")
        densities.append(density)
        selected.append({"event_generation_event_id": event.get("event_generation_event_id"), "interaction_id": interaction_id, "density_g_cm3": density})
    if not densities:
        raise RuntimeError("density_source=dis_vertex_local requires H3-W10 event summaries and DIS interaction metadata")
    reference = densities[0]
    if any(abs(value - reference) > 1e-12 * max(1.0, abs(reference)) for value in densities[1:]):
        raise RuntimeError("selected H3-W10 events have different local densities; H3-W11 v1 requires one homogeneous-material job per density")
    return reference, "dis_vertex_local", selected


def _generate_single_geant4_products(values: dict[str, dict[str, Any]], *, run_output_dir: Path, input_override: Path | None = None) -> dict[str, Any]:
    """Run an explicit H3-W11 request and atomically publish its products."""
    started = time.perf_counter()
    cfg = dict(values.get("geant4", {}))
    mode = str(cfg.get("mode", "disabled"))
    if mode == "disabled":
        raise ValueError("GEANT4 mode is disabled; choose an explicit run mode and click Run")
    allowed = {"environment_check", "import_check", "vacuum_smoke", "material_smoke", "real_free"}
    if mode not in allowed:
        raise ValueError(f"unsupported GEANT4 mode: {mode}")

    availability = backend_availability()
    if not availability["available"]:
        raise RuntimeError(f"GEANT4 backend or required datasets are unavailable: {availability}")

    event_dir = event_generation_dir(run_output_dir)
    input_path = input_override or event_dir / "event_generation_events.hepmc3"
    uses_environment_fixture = False
    if mode == "environment_check" and not input_path.exists():
        input_path = ROOT / "tests" / "fixtures" / "geant4" / "six_muons_vacuum.hepmc3"
        uses_environment_fixture = True
    upstream_manifest_path = event_dir / "event_generation_manifest.json"
    upstream_summary_path = event_dir / "event_generation_summary.json"
    upstream_events_path = event_dir / "event_generation_events_summary.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"H3-W11 requires H3-W10 HepMC3 input: {input_path}")
    upstream_manifest = json.loads(upstream_manifest_path.read_text(encoding="utf-8")) if upstream_manifest_path.exists() else {}
    if not input_override and not uses_environment_fixture:
        if upstream_manifest.get("stage") != "H3-W10":
            raise RuntimeError("GEANT4 requires an H3-W10 manifest")
        if upstream_manifest.get("generator_frame") != "local_matter_tetrad" or upstream_manifest.get("momentum_unit") != "GeV" or upstream_manifest.get("length_unit") != "mm":
            raise RuntimeError("H3-W10 frame/units contract is incompatible with H3-W11")
        if upstream_summary_path.exists() and json.loads(upstream_summary_path.read_text(encoding="utf-8")).get("status") != "ok":
            raise RuntimeError("H3-W10 summary is not valid")

    upstream_events = _read_jsonl(upstream_events_path)
    density_override = input_override or (input_path if uses_environment_fixture else None)
    resolved_density, resolved_density_source, density_records = _resolve_density(
        cfg, run_output_dir, upstream_events, input_override=density_override,
    )
    input_hash_before = _sha256(input_path)
    final_dir = geant4_dir(run_output_dir)
    staging = final_dir.parent / f".{final_dir.name}.staging-{os.getpid()}-{time.time_ns()}"
    staging.mkdir(parents=True, exist_ok=False)
    backend_dir = staging / "backend"
    backend_dir.mkdir()
    effective_backend_mode = "import_check" if mode == "environment_check" else mode
    command = [
        str(Path(availability["backend_executable"])), "--input", str(input_path), "--output-dir", str(backend_dir),
        "--mode", effective_backend_mode,
        "--material", str(cfg.get("material", "HADROS3_H_HE")),
        "--density-g-cm3", str(resolved_density),
        "--hydrogen-mass-fraction", str(float(cfg.get("hydrogen_mass_fraction", 0.75))),
        "--half-size-mm", str(float(cfg.get("patch_half_size_mm", 10.0))),
        "--world-margin-mm", str(float(cfg.get("world_margin_mm", 10.0))),
        "--production-cut-mm", str(float(cfg.get("production_cut_mm", 0.1))),
        "--max-energy-gev", str(float(cfg.get("validated_maximum_energy_gev", 1.0e5))),
        "--max-events", str(int(float(cfg.get("max_events", 2)))),
        "--max-recorded-steps", str(int(float(cfg.get("max_recorded_steps", 50000)))),
        "--seed", str(int(float(cfg.get("random_seed", 59001)))),
    ]
    result = subprocess.run(command, cwd=ROOT, env=geant4_environment(), text=True, capture_output=True, check=False)
    (staging / "geant4_stdout.log").write_text(result.stdout, encoding="utf-8")
    (staging / "geant4_stderr.log").write_text(result.stderr, encoding="utf-8")
    import_report_path = backend_dir / "geant4_import_report.json"
    import_report = json.loads(import_report_path.read_text(encoding="utf-8")) if import_report_path.exists() else {}

    if result.returncode not in {0, 3}:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(f"GEANT4 backend failed with exit {result.returncode}: {result.stderr[-2000:]}")

    status = "unsupported_domain" if result.returncode == 3 else "ok"
    transported = effective_backend_mode != "import_check" and status == "ok"
    raw_events = _read_jsonl(backend_dir / "geant4_events_raw.jsonl")
    raw_escaped = _read_jsonl(backend_dir / "geant4_escaped_particles_raw.jsonl")
    raw_steps = _read_jsonl(backend_dir / "geant4_steps_raw.jsonl")
    events, escaped, steps = _enrich_rows(raw_events, raw_escaped, raw_steps, upstream_events)
    _write_jsonl(staging / "geant4_events_summary.jsonl", events)
    _write_jsonl(staging / "geant4_escaped_particles.jsonl", escaped)
    _write_jsonl(staging / "geant4_steps.jsonl", steps)
    domain_violations = _domain_violations(import_report)
    _write_jsonl(staging / "geant4_unsupported_particles.jsonl", domain_violations)
    shutil.copy2(import_report_path, staging / "geant4_import_report.json")
    backend_summary_path = backend_dir / "geant4_backend_summary.json"
    backend_summary = json.loads(backend_summary_path.read_text(encoding="utf-8")) if backend_summary_path.exists() else {}

    tolerance = 1.0e-6 if mode in {"vacuum_smoke", "material_smoke"} else 1.0e-4
    max_residual = float(backend_summary.get("max_abs_normalized_unexplained_residual", 0.0))
    max_raw_balance = float(backend_summary.get("max_abs_normalized_raw_energy_balance", 0.0))
    validation = {
        "status": status,
        "environment_pass": availability["available"],
        "input_contract_pass": import_report.get("status") in {"ok", "unsupported_domain"},
        "domain_pass": status == "ok",
        "unsupported_energy_particles": int(import_report.get("unsupported_energy", 0)),
        "unsupported_species": int(import_report.get("unsupported_species", 0)),
        "energy_ledger_tolerance": tolerance,
        "max_abs_normalized_unexplained_residual": max_residual if transported else None,
        "max_abs_normalized_raw_energy_balance": max_raw_balance if transported else None,
        "material_rest_mass_and_binding_exchange_inferred": transported and mode in {"material_smoke", "real_free"},
        "energy_ledger_interpretation": ("strict_total_energy_vacuum" if mode == "vacuum_smoke" else "material_exchange_term_includes_rest_mass_and_nuclear_binding_reservoir"),
        "energy_ledger_pass": (not transported) or max_residual <= tolerance,
        "cardinality_pass": (not transported) or len(events) == int(backend_summary.get("events_transported", -1)),
        "escaped_records_pass": (not transported) or len(escaped) == int(backend_summary.get("escaped_particles", -1)),
        "recorded_steps_pass": (not transported) or len(steps) == int(backend_summary.get("recorded_steps", -1)),
        "upstream_hash_unchanged": input_hash_before == _sha256(input_path),
    }
    if status == "ok" and not all(value for key, value in validation.items() if key.endswith("_pass")):
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(f"H3-W11 numerical validation failed: {validation}")

    environment_manifest = availability
    _write_json(staging / "geant4_environment_manifest.json", environment_manifest)
    config_hash = _config_hash(cfg)
    manifest = {
        "stage": "H3-W11",
        "status": status,
        "configuration": cfg,
        "configuration_sha256": config_hash,
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "upstream_manifest_sha256": _sha256(upstream_manifest_path) if upstream_manifest_path.exists() else None,
        "generator_frame": "local_matter_tetrad",
        "momentum_unit": "GeV",
        "length_unit": "mm",
        "physics_list": "FTFP_BERT",
        "physics_domain_policy": "fail_stage",
        "resolved_density_source": resolved_density_source,
        "resolved_density_g_cm3": resolved_density,
        "density_applied_to_material": str(cfg.get("material", "HADROS3_H_HE")) == "HADROS3_H_HE",
        "material_density_policy": ("configured local density applied" if str(cfg.get("material", "HADROS3_H_HE")) == "HADROS3_H_HE" else "built-in Geant4 NIST material density"),
        "event_density_records": density_records,
        "dependencies": availability,
        "command": command,
    }
    _write_json(staging / "geant4_manifest.json", manifest)
    _write_json(staging / "geant4_validation_report.json", validation)
    if transported and not bool(cfg.get("_defer_presentation", False)):
        _plots(events, escaped, staging)
        sites = write_geant4_visualizations(
            values, run_output_dir, staging, events, escaped, steps,
            material=str(cfg.get("material", "HADROS3_H_HE")), density_g_cm3=resolved_density,
        )
    else:
        sites = []
    if status == "unsupported_domain":
        _domain_audit_plot(domain_violations, staging)
        viewer = "<!doctype html><meta charset='utf-8'><title>HADROS3 GEANT4 domain audit</title><h1>H3-W11 transport not started</h1><p>The input is outside the validated physics domain. The rows below are the audit result; no transported event exists.</p><pre id='o'></pre><script>fetch('geant4_unsupported_particles.jsonl').then(r=>r.text()).then(t=>o.textContent=t)</script>"
        (staging / "geant4_event_view.html").write_text(viewer, encoding="utf-8")

    summary = {
        "status": status,
        "stage": "H3-W11",
        "geant4_mode": mode,
        "geant4_invoked": transported,
        "geant4_process_running": False,
        "photon_transport_invoked": False,
        "spectra_invoked": False,
        "events_imported": int(import_report.get("events", 0)),
        "events_transported": int(backend_summary.get("events_transported", 0)),
        "final_primaries": int(import_report.get("final_particles", 0)),
        "unsupported_energy_particles": int(import_report.get("unsupported_energy", 0)),
        "unsupported_species": int(import_report.get("unsupported_species", 0)),
        "domain_violations": domain_violations,
        "maximum_input_energy_gev": import_report.get("maximum_energy_gev"),
        "validated_maximum_energy_gev": import_report.get("validated_maximum_energy_gev"),
        "total_steps": int(backend_summary.get("total_steps", 0)),
        "recorded_steps": int(backend_summary.get("recorded_steps", 0)),
        "steps_truncated": bool(backend_summary.get("steps_truncated", False)),
        "geant4_sites": len(sites),
        "material": str(cfg.get("material", "HADROS3_H_HE")),
        "density_applied_to_material": str(cfg.get("material", "HADROS3_H_HE")) == "HADROS3_H_HE",
        "escaped_particles": int(backend_summary.get("escaped_particles", 0)),
        "deposited_gev": float(backend_summary.get("deposited_gev", 0.0)),
        "escaped_total_gev": float(backend_summary.get("escaped_total_gev", 0.0)),
        "validation": validation,
        "runtime_seconds": time.perf_counter() - started,
        "message": ("Input is outside the validated Geant4 physics domain; no transport was started." if status == "unsupported_domain" else "GEANT4 run completed and is not currently running."),
    }
    product_names = [
        "geant4_summary.json", "geant4_summary.csv", "geant4_manifest.json", "geant4_environment_manifest.json",
        "geant4_validation_report.json", "geant4_import_report.json", "geant4_events_summary.jsonl",
        "geant4_escaped_particles.jsonl", "geant4_steps.jsonl", "geant4_sites.json",
        "geant4_unsupported_particles.jsonl", "geant4_energy_balance.png", "geant4_escape_spectrum.png",
        "geant4_domain_audit.png", "geant4_macro_sites_3d.html", "geant4_event_view.html",
        "geant4_stdout.log", "geant4_stderr.log",
    ]
    summary["products"] = {
        ("geant4_summary_csv" if name == "geant4_summary.csv" else Path(name).stem): str(final_dir / name)
        for name in product_names if (staging / name).exists() or name in {"geant4_summary.json", "geant4_summary.csv"}
    }
    _write_json(staging / "geant4_summary.json", summary)
    with (staging / "geant4_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key in ("status", "geant4_mode", "events_imported", "events_transported", "final_primaries", "unsupported_energy_particles", "total_steps", "escaped_particles", "deposited_gev", "runtime_seconds"):
            writer.writerow([key, summary[key]])

    shutil.rmtree(backend_dir, ignore_errors=True)
    if final_dir.exists():
        backup = final_dir.parent / f".{final_dir.name}.previous-{time.time_ns()}"
        final_dir.rename(backup)
        staging.rename(final_dir)
        shutil.rmtree(backup)
    else:
        staging.rename(final_dir)
    return summary


def generate_geant4_products(values: dict[str, dict[str, Any]], *, run_output_dir: Path, input_override: Path | None = None) -> dict[str, Any]:
    """Run H3-W11 using one isolated Geant4 process per physical DIS site."""
    cfg = values.get("geant4", {})
    if (
        input_override is not None
        or str(cfg.get("density_source", "dis_vertex_local")) != "dis_vertex_local"
        or str(cfg.get("mode", "disabled")) == "environment_check"
    ):
        return _generate_single_geant4_products(values, run_output_dir=run_output_dir, input_override=input_override)
    # Preserve the compact single-file contract used by fixtures and legacy
    # one-site runs. Production H3-W10 outputs always carry request IDs and
    # per-request HepMC3 files, which activate the isolated site scheduler.
    event_dir = event_generation_dir(run_output_dir)
    candidate_events = _read_jsonl(event_dir / "event_generation_events_summary.jsonl")
    selected = candidate_events[: int(float(cfg.get("max_events", 2)))]
    if any(
        not event.get("powheg_request_id")
        or not (event_dir / "jobs" / str(event.get("powheg_request_id")) / "events.hepmc3").exists()
        for event in selected
    ):
        return _generate_single_geant4_products(values, run_output_dir=run_output_dir)
    from .geant4_site_batch import generate_site_batch
    return generate_site_batch(values, run_output_dir=run_output_dir)
