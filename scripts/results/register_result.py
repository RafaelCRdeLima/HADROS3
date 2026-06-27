#!/usr/bin/env python3
"""Register a HADROS3 run in the central scientific results catalog."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from typing import Any


CATALOG_COLUMNS = [
    "run_id",
    "date",
    "git_commit",
    "software_version",
    "physics_version",
    "pipeline_version",
    "theory_version",
    "stage",
    "case_name",
    "description",
    "spin_a",
    "rho0_g_cm3",
    "E_min_GeV",
    "E_max_GeV",
    "n_source_samples",
    "n_forward_paths",
    "n_dis_interactions",
    "n_observer_candidates",
    "n_kerr_matched",
    "n_powheg_jobs",
    "n_lhe_events",
    "main_output_dir",
    "main_figures",
    "validation_status",
    "paper_candidate",
    "notes",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _nested(payload: dict[str, Any], *keys: str, default: Any = "") -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _first_number(*values: Any) -> Any:
    for value in values:
        if value in (None, ""):
            continue
        return value
    return ""


def _figure_list(run_dir: Path) -> str:
    suffixes = {".png", ".pdf", ".svg", ".html"}
    figures = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if "RunMetadata" in path.parts:
            continue
        figures.append(path.relative_to(run_dir).as_posix())
    return ";".join(figures)


def _load_existing_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: row.get(key, "") for key in CATALOG_COLUMNS} for row in reader]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOG_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CATALOG_COLUMNS})


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_catalog_row(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    provenance = _read_json(run_dir / "RunMetadata" / "hadros3_pipeline_provenance.json")
    config = _read_json(run_dir / "RunMetadata" / "hadros3_config.json")
    values = config.get("hadros3_values", config)
    source = _read_json(run_dir / "UHEsource" / "uhe_neutrino_source_summary.json")
    forward = _read_json(run_dir / "ForwardGeodesics" / "uhe_neutrino_forward_summary.json")
    dis_summary = _read_json(run_dir / "DIS" / "dis_summary.json")
    dis_report = _read_json(run_dir / "DIS" / "dis_diagnostics_report.json")
    observer = _read_json(run_dir / "ObserverBridge" / "observer_bridge_summary.json")
    powheg = _read_json(run_dir / "POWHEG" / "powheg_summary.json")
    powheg_validation = _read_json(run_dir / "POWHEG" / "powheg_validation_report.json")
    release = provenance.get("scientific_release", {})
    source_values = values.get("uhe_neutrino_source", {}) if isinstance(values, dict) else {}
    bh_values = values.get("black_hole", {}) if isinstance(values, dict) else {}
    torus_values = values.get("analytic_torus", {}) if isinstance(values, dict) else {}
    energy = source_values.get("energy_gev", "")
    row = {
        "run_id": args.run_id or run_dir.name,
        "date": args.date or date.today().isoformat(),
        "git_commit": release.get("git_commit") or provenance.get("git_commit") or provenance.get("theory_commit", ""),
        "software_version": release.get("software_version", provenance.get("software_version", "")),
        "physics_version": release.get("physics_version", provenance.get("physics_version", "")),
        "pipeline_version": release.get("pipeline_version", _nested(provenance, "scientific_theory", "theory_pipeline_version")),
        "theory_version": release.get("theory_version", provenance.get("theory_version", "")),
        "stage": args.stage,
        "case_name": args.case_name,
        "description": args.description,
        "spin_a": bh_values.get("spin_a", ""),
        "rho0_g_cm3": torus_values.get("rho0_g_cm3", ""),
        "E_min_GeV": _first_number(source.get("energy_min_gev"), energy),
        "E_max_GeV": _first_number(source.get("energy_max_gev"), energy),
        "n_source_samples": _first_number(source.get("n_samples"), source_values.get("n_samples")),
        "n_forward_paths": forward.get("n_paths", ""),
        "n_dis_interactions": _first_number(dis_report.get("accepted_interactions"), dis_summary.get("n_accepted_interactions")),
        "n_observer_candidates": observer.get("n_candidates_scored", ""),
        "n_kerr_matched": observer.get("kerr_pixel_match_n_matched", ""),
        "n_powheg_jobs": _first_number(powheg.get("n_powheg_jobs"), powheg.get("powheg_jobs_prepared")),
        "n_lhe_events": _first_number(powheg.get("n_lhe_events"), powheg_validation.get("n_lhe_events")),
        "main_output_dir": run_dir.as_posix(),
        "main_figures": args.main_figures or _figure_list(run_dir),
        "validation_status": args.validation_status,
        "paper_candidate": str(bool(args.paper_candidate)).lower(),
        "notes": args.notes,
    }
    return row


def upsert_row(rows: list[dict[str, Any]], row: dict[str, Any]) -> list[dict[str, Any]]:
    run_id = str(row["run_id"])
    filtered = [existing for existing in rows if existing.get("run_id") != run_id]
    filtered.append(row)
    return sorted(filtered, key=lambda item: str(item.get("run_id", "")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Generated output/<run-name>/ directory to register.")
    parser.add_argument("--catalog-dir", default="results/catalog", help="Catalog directory to update.")
    parser.add_argument("--run-id", default="", help="Catalog run id. Defaults to run directory name.")
    parser.add_argument("--date", default="", help="Result date. Defaults to today.")
    parser.add_argument("--stage", default="unclassified", help="Result stage, e.g. validation/observer_bridge.")
    parser.add_argument("--case-name", default="", help="Short physics case name.")
    parser.add_argument("--description", default="", help="One-line result description.")
    parser.add_argument("--validation-status", default="unreviewed", help="validation_status catalog value.")
    parser.add_argument("--paper-candidate", action="store_true", help="Mark as paper candidate.")
    parser.add_argument("--main-figures", default="", help="Semicolon-separated figure list override.")
    parser.add_argument("--notes", default="", help="Short notes for the central catalog.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog_dir = Path(args.catalog_dir)
    csv_path = catalog_dir / "HADROS3_RESULTS_CATALOG.csv"
    json_path = catalog_dir / "HADROS3_RESULTS_CATALOG.json"
    row = build_catalog_row(args)
    rows = upsert_row(_load_existing_csv(csv_path), row)
    _write_csv(csv_path, rows)
    _write_json(json_path, rows)
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
