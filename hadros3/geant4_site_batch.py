"""Deterministic, bounded H3-W11 orchestration over independent DIS sites."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .geant4_transport import (
    _config_hash, _domain_audit_plot, _domain_violations, _generate_single_geant4_products,
    _plots, _read_jsonl, _sha256, _write_json, _write_jsonl, backend_availability,
)
from .geant4_visualization import write_geant4_visualizations
from .paths import event_generation_dir, geant4_dir


def _seed(base_seed: int, site_key: str) -> int:
    digest = hashlib.sha256(f"H3-W11:{base_seed}:{site_key}".encode()).digest()
    return 1 + int.from_bytes(digest[:8], "big") % 900_000_000


def _plan(cfg: dict[str, Any], run_dir: Path, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dis = {str(row.get("interaction_id")): row for row in _read_jsonl(run_dir / "DIS" / "dis_accepted_interactions.jsonl")}
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for global_index, event in enumerate(events[: int(float(cfg.get("max_events", 2)))]):
        interaction_id = str(event.get("interaction_id") or "")
        request_id = str(event.get("powheg_request_id") or "")
        if not interaction_id or not request_id:
            raise RuntimeError("per-site GEANT4 execution requires interaction_id and powheg_request_id on every H3-W10 event")
        metadata = dis.get(interaction_id)
        if metadata is None:
            raise RuntimeError(f"no DIS interaction metadata found for {interaction_id}")
        density = float(metadata.get("interaction_rho_g_cm3", metadata.get("interaction_point_rho_g_cm3", 0.0)))
        if not math.isfinite(density) or density <= 0:
            raise RuntimeError(f"invalid local DIS density for {interaction_id}: {density}")
        group = groups.setdefault((interaction_id, request_id), {
            "interaction_id": interaction_id, "powheg_request_id": request_id,
            "density_g_cm3": density, "events": [], "global_event_indices": [],
        })
        if not math.isclose(float(group["density_g_cm3"]), density, rel_tol=1e-12):
            raise RuntimeError(f"inconsistent density inside physical site {interaction_id}")
        group["events"].append(event)
        group["global_event_indices"].append(global_index)
    if not groups:
        raise RuntimeError("no H3-W10 events are available for per-site GEANT4 transport")
    jobs = []
    event_dir = event_generation_dir(run_dir)
    for index, group in enumerate(groups.values(), 1):
        input_path = event_dir / "jobs" / group["powheg_request_id"] / "events.hepmc3"
        if not input_path.exists():
            raise FileNotFoundError(f"missing per-request HepMC3 input: {input_path}")
        key = "|".join((group["interaction_id"], group["powheg_request_id"], f"{group['density_g_cm3']:.17g}",
                        str(cfg.get("material")), f"{float(cfg.get('patch_half_size_mm', 10)):.17g}"))
        group.update({
            "site_job_id": f"H3G4SITE-{index:06d}", "site_index": index - 1,
            "site_key": key, "input": str(input_path), "input_sha256": _sha256(input_path),
            "seed": _seed(int(float(cfg.get("random_seed", 59001))), key), "event_count": len(group["events"]),
        })
        jobs.append(group)
    return jobs


def _remap(rows: list[dict[str, Any]], job: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = job["global_event_indices"]
    for row in rows:
        local = int(row["geant4_event_id"])
        if local >= len(mapping):
            raise RuntimeError(f"{job['site_job_id']} returned unplanned event {local}")
        row.update({
            "site_event_index": local, "geant4_event_id": mapping[local],
            "site_job_id": job["site_job_id"], "site_density_g_cm3": job["density_g_cm3"],
        })
    return rows


def _aggregate_import(reports: list[dict[str, Any]], status: str) -> dict[str, Any]:
    ceilings = [float(row["validated_maximum_energy_gev"]) for row in reports if row.get("validated_maximum_energy_gev") is not None]
    return {
        "status": status,
        "events": sum(int(row.get("events", 0)) for row in reports),
        "final_particles": sum(int(row.get("final_particles", 0)) for row in reports),
        "unsupported_energy": sum(int(row.get("unsupported_energy", 0)) for row in reports),
        "unsupported_species": sum(int(row.get("unsupported_species", 0)) for row in reports),
        "maximum_energy_gev": max((float(row.get("maximum_energy_gev", 0)) for row in reports), default=0),
        "validated_maximum_energy_gev": min(ceilings) if ceilings else None,
        "violations": [value for row in reports for value in row.get("violations", [])],
    }


def generate_site_batch(values: dict[str, dict[str, Any]], *, run_output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = dict(values.get("geant4", {}))
    mode = str(cfg.get("mode", "disabled"))
    if mode == "disabled":
        raise ValueError("GEANT4 mode is disabled; choose an explicit run mode and click Run")
    event_dir = event_generation_dir(run_output_dir)
    manifest_path = event_dir / "event_generation_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("stage") != "H3-W10" or manifest.get("generator_frame") != "local_matter_tetrad":
        raise RuntimeError("GEANT4 requires an H3-W10 local_matter_tetrad manifest")
    events = _read_jsonl(event_dir / "event_generation_events_summary.jsonl")
    jobs = _plan(cfg, run_output_dir, events)
    workers = min(int(float(cfg.get("site_workers", 4))), len(jobs))
    if workers < 1:
        raise ValueError("geant4.site_workers must be a positive integer")

    final_dir = geant4_dir(run_output_dir)
    work = final_dir.parent / f".{final_dir.name}.site-work-{os.getpid()}-{time.time_ns()}"
    staging = final_dir.parent / f".{final_dir.name}.staging-{os.getpid()}-{time.time_ns()}"
    work.mkdir(parents=True)
    staging.mkdir(parents=True)
    _write_jsonl(staging / "geant4_site_jobs_plan.jsonl", [{k: v for k, v in job.items() if k != "events"} for job in jobs])

    def run(job: dict[str, Any]) -> dict[str, Any]:
        run_dir = work / job["site_job_id"]
        upstream = event_generation_dir(run_dir)
        upstream.mkdir(parents=True)
        _write_jsonl(upstream / "event_generation_events_summary.jsonl", job["events"])
        local_values = {name: dict(section) for name, section in values.items()}
        local_values["geant4"].update({
            "density_source": "configured_fixture", "density_g_cm3": job["density_g_cm3"],
            "random_seed": job["seed"], "max_events": job["event_count"], "_defer_presentation": True,
        })
        summary = _generate_single_geant4_products(local_values, run_output_dir=run_dir, input_override=Path(job["input"]))
        return {"job": job, "summary": summary, "dir": geant4_dir(run_dir)}

    try:
        completed = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="h3-g4-site") as pool:
            futures = {pool.submit(run, job): job for job in jobs}
            for future in as_completed(futures):
                completed.append(future.result())
        completed.sort(key=lambda item: item["job"]["site_index"])

        output_events: list[dict[str, Any]] = []
        escaped: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        reports, site_rows = [], []
        audits = staging / "site_jobs"
        audits.mkdir()
        for item in completed:
            job, directory, summary = item["job"], item["dir"], item["summary"]
            output_events += _remap(_read_jsonl(directory / "geant4_events_summary.jsonl"), job)
            escaped += _remap(_read_jsonl(directory / "geant4_escaped_particles.jsonl"), job)
            steps += _remap(_read_jsonl(directory / "geant4_steps.jsonl"), job)
            reports.append(json.loads((directory / "geant4_import_report.json").read_text()))
            audit = audits / job["site_job_id"]
            audit.mkdir()
            for name in ("geant4_summary.json", "geant4_manifest.json", "geant4_validation_report.json",
                         "geant4_import_report.json", "geant4_stdout.log", "geant4_stderr.log"):
                if (directory / name).exists():
                    shutil.copy2(directory / name, audit / name)
            site_rows.append({
                "site_job_id": job["site_job_id"], "site_index": job["site_index"],
                "interaction_id": job["interaction_id"], "powheg_request_id": job["powheg_request_id"],
                "density_g_cm3": job["density_g_cm3"], "seed": job["seed"], "input": job["input"],
                "input_sha256": job["input_sha256"], "event_count": job["event_count"],
                "status": summary["status"], "events_transported": summary["events_transported"],
                "total_steps": summary["total_steps"], "recorded_steps": summary["recorded_steps"],
            })

        output_events.sort(key=lambda row: int(row["geant4_event_id"]))
        escaped.sort(key=lambda row: (int(row["geant4_event_id"]), int(row.get("track_id", 0))))
        steps.sort(key=lambda row: (int(row["geant4_event_id"]), int(row.get("event_step_index", 0)), int(row.get("track_id", 0))))
        _write_jsonl(staging / "geant4_events_summary.jsonl", output_events)
        _write_jsonl(staging / "geant4_escaped_particles.jsonl", escaped)
        _write_jsonl(staging / "geant4_steps.jsonl", steps)
        _write_jsonl(staging / "geant4_site_jobs_summary.jsonl", site_rows)

        status = "unsupported_domain" if any(row["status"] == "unsupported_domain" for row in site_rows) else "ok"
        import_report = _aggregate_import(reports, status)
        _write_json(staging / "geant4_import_report.json", import_report)
        violations = _domain_violations(import_report)
        _write_jsonl(staging / "geant4_unsupported_particles.jsonl", violations)
        transported = mode != "import_check" and status == "ok"
        density_for_event = {index: job["density_g_cm3"] for job in jobs for index in job["global_event_indices"]}
        validation = {
            "status": status, "environment_pass": all(item["summary"]["validation"]["environment_pass"] for item in completed),
            "input_contract_pass": all(item["summary"]["validation"]["input_contract_pass"] for item in completed),
            "domain_pass": status == "ok", "site_isolation_pass": len(completed) == len(jobs),
            "density_mapping_pass": all(math.isclose(float(row["site_density_g_cm3"]), float(density_for_event[int(row["geant4_event_id"])]), rel_tol=0, abs_tol=0) for row in output_events),
            "cardinality_pass": (not transported) or len(output_events) == sum(job["event_count"] for job in jobs),
            "escaped_records_pass": all(item["summary"]["validation"]["escaped_records_pass"] for item in completed),
            "recorded_steps_pass": all(item["summary"]["validation"]["recorded_steps_pass"] for item in completed),
            "energy_ledger_pass": all(item["summary"]["validation"]["energy_ledger_pass"] for item in completed),
            "upstream_hash_unchanged": all(_sha256(Path(job["input"])) == job["input_sha256"] for job in jobs),
        }
        if status == "ok" and not all(value for key, value in validation.items() if key.endswith("_pass")):
            raise RuntimeError(f"H3-W11 per-site numerical validation failed: {validation}")
        _write_json(staging / "geant4_validation_report.json", validation)
        availability = backend_availability()
        _write_json(staging / "geant4_environment_manifest.json", availability)
        top_manifest = {
            "stage": "H3-W11", "status": status, "configuration": cfg, "configuration_sha256": _config_hash(cfg),
            "execution_model": "per_site_subprocess", "site_workers": workers, "site_jobs": site_rows,
            "generator_frame": "local_matter_tetrad", "momentum_unit": "GeV", "length_unit": "mm",
            "physics_list": "FTFP_BERT", "physics_domain_policy": "fail_stage",
            "resolved_density_source": "dis_vertex_local", "resolved_density_g_cm3": None,
            "density_applied_to_material": str(cfg.get("material")) == "HADROS3_H_HE",
            "material_density_policy": "one homogeneous local G4Material per physical DIS site",
            "dense_matter_model_limitations": [
                "HADROS3_H_HE is an ideal homogeneous elemental mixture at the sampled mass density",
                "electron degeneracy, plasma screening, collective effects, and an astrophysical equation of state are not modeled by FTFP_BERT",
                "patch-size and production-cut convergence remain required for precision astrophysical interpretation",
            ],
            "event_density_records": [{"event_generation_event_id": event.get("event_generation_event_id"),
                                       "interaction_id": job["interaction_id"], "density_g_cm3": job["density_g_cm3"],
                                       "site_job_id": job["site_job_id"]} for job in jobs for event in job["events"]],
            "upstream_manifest_sha256": _sha256(manifest_path), "dependencies": availability,
        }
        _write_json(staging / "geant4_manifest.json", top_manifest)
        if transported:
            _plots(output_events, escaped, staging)
            sites = write_geant4_visualizations(values, run_output_dir, staging, output_events, escaped, steps,
                                                material=str(cfg.get("material")), density_g_cm3=float(jobs[0]["density_g_cm3"]))
        else:
            sites = []
        if status == "unsupported_domain":
            _domain_audit_plot(violations, staging)

        summary = {
            "status": status, "stage": "H3-W11", "geant4_mode": mode, "execution_model": "per_site_subprocess",
            "site_workers": workers, "site_jobs": len(jobs), "geant4_invoked": transported, "geant4_process_running": False,
            "photon_transport_invoked": False, "spectra_invoked": False,
            "events_imported": import_report["events"], "events_transported": len(output_events),
            "final_primaries": import_report["final_particles"], "unsupported_energy_particles": import_report["unsupported_energy"],
            "unsupported_species": import_report["unsupported_species"], "domain_violations": violations,
            "maximum_input_energy_gev": import_report["maximum_energy_gev"],
            "validated_maximum_energy_gev": import_report["validated_maximum_energy_gev"],
            "total_steps": sum(int(item["summary"]["total_steps"]) for item in completed), "recorded_steps": len(steps),
            "steps_truncated": any(item["summary"]["steps_truncated"] for item in completed), "geant4_sites": len(sites),
            "material": str(cfg.get("material")), "density_applied_to_material": str(cfg.get("material")) == "HADROS3_H_HE",
            "escaped_particles": len(escaped), "deposited_gev": sum(float(row.get("deposited_gev", 0)) for row in output_events),
            "escaped_total_gev": sum(float(row.get("escaped_total_gev", 0)) for row in output_events),
            "validation": validation, "runtime_seconds": time.perf_counter() - started,
            "message": ("Input is outside the validated Geant4 physics domain; no transport was published." if status == "unsupported_domain"
                        else f"GEANT4 completed {len(jobs)} isolated site job(s) and is not currently running."),
        }
        names = ["geant4_summary.json", "geant4_summary.csv", "geant4_manifest.json", "geant4_environment_manifest.json",
                 "geant4_validation_report.json", "geant4_import_report.json", "geant4_events_summary.jsonl",
                 "geant4_escaped_particles.jsonl", "geant4_steps.jsonl", "geant4_sites.json",
                 "geant4_site_jobs_plan.jsonl", "geant4_site_jobs_summary.jsonl", "geant4_unsupported_particles.jsonl",
                 "geant4_energy_balance.png", "geant4_escape_spectrum.png", "geant4_domain_audit.png",
                 "geant4_macro_sites_3d.html", "geant4_event_view.html"]
        summary["products"] = {("geant4_summary_csv" if name.endswith(".csv") else Path(name).stem): str(final_dir / name)
                               for name in names if (staging / name).exists() or name in {"geant4_summary.json", "geant4_summary.csv"}}
        _write_json(staging / "geant4_summary.json", summary)
        with (staging / "geant4_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            for key in ("status", "geant4_mode", "execution_model", "site_jobs", "site_workers", "events_imported",
                        "events_transported", "total_steps", "escaped_particles", "deposited_gev", "runtime_seconds"):
                writer.writerow([key, summary[key]])
        if final_dir.exists():
            backup = final_dir.parent / f".{final_dir.name}.previous-{time.time_ns()}"
            final_dir.rename(backup)
            staging.rename(final_dir)
            shutil.rmtree(backup)
        else:
            staging.rename(final_dir)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
