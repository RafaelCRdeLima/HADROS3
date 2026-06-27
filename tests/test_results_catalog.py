from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.results.register_result import build_catalog_row, upsert_row


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "output" / "RUN_OBS_test_20260626_abcd"
    _write_json(
        run_dir / "RunMetadata" / "hadros3_pipeline_provenance.json",
        {
            "scientific_release": {
                "software_version": "0.9.0",
                "physics_version": "1.1",
                "pipeline_version": "H3-W9b",
                "theory_version": "1.1",
                "git_commit": "abc1234",
            }
        },
    )
    _write_json(
        run_dir / "RunMetadata" / "hadros3_config.json",
        {
            "hadros3_values": {
                "black_hole": {"spin_a": 0.9},
                "analytic_torus": {"rho0_g_cm3": 1.0e10},
                "uhe_neutrino_source": {"energy_gev": "10^{9}", "n_samples": 20},
            }
        },
    )
    _write_json(run_dir / "UHEsource" / "uhe_neutrino_source_summary.json", {"n_samples": 20, "energy_min_gev": 1.0e9, "energy_max_gev": 1.0e9})
    _write_json(run_dir / "ForwardGeodesics" / "uhe_neutrino_forward_summary.json", {"n_paths": 12})
    _write_json(run_dir / "DIS" / "dis_diagnostics_report.json", {"accepted_interactions": 5})
    _write_json(run_dir / "ObserverBridge" / "observer_bridge_summary.json", {"n_candidates_scored": 5, "kerr_pixel_match_n_matched": 4})
    _write_json(run_dir / "POWHEG" / "powheg_summary.json", {"n_powheg_jobs": 1, "n_lhe_events": 2})
    (run_dir / "ObserverBridge").mkdir(parents=True, exist_ok=True)
    (run_dir / "ObserverBridge" / "observer_bridge_camera_overlay.png").write_bytes(b"png")
    return run_dir


def test_build_catalog_row_extracts_pipeline_metadata(tmp_path: Path) -> None:
    run_dir = _fake_run(tmp_path)
    args = type(
        "Args",
        (),
        {
            "run_dir": str(run_dir),
            "run_id": "",
            "date": "2026-06-26",
            "stage": "validation/observer_bridge",
            "case_name": "observer_bridge_smoke",
            "description": "Observer Bridge smoke catalog entry.",
            "main_figures": "",
            "validation_status": "pass",
            "paper_candidate": False,
            "notes": "temporary test run",
        },
    )()

    row = build_catalog_row(args)

    assert row["run_id"] == "RUN_OBS_test_20260626_abcd"
    assert row["git_commit"] == "abc1234"
    assert row["software_version"] == "0.9.0"
    assert row["physics_version"] == "1.1"
    assert row["pipeline_version"] == "H3-W9b"
    assert row["theory_version"] == "1.1"
    assert row["spin_a"] == 0.9
    assert row["rho0_g_cm3"] == 1.0e10
    assert row["n_source_samples"] == 20
    assert row["n_forward_paths"] == 12
    assert row["n_dis_interactions"] == 5
    assert row["n_observer_candidates"] == 5
    assert row["n_kerr_matched"] == 4
    assert row["n_powheg_jobs"] == 1
    assert row["n_lhe_events"] == 2
    assert "ObserverBridge/observer_bridge_camera_overlay.png" in row["main_figures"]


def test_upsert_row_replaces_existing_run_id() -> None:
    rows = [{"run_id": "A", "notes": "old"}, {"run_id": "B", "notes": "keep"}]
    updated = upsert_row(rows, {"run_id": "A", "notes": "new"})

    assert updated == [{"run_id": "A", "notes": "new"}, {"run_id": "B", "notes": "keep"}]


def test_register_result_cli_updates_csv_and_json(tmp_path: Path) -> None:
    run_dir = _fake_run(tmp_path)
    catalog_dir = tmp_path / "catalog"

    subprocess.run(
        [
            sys.executable,
            "scripts/results/register_result.py",
            "--run-dir",
            str(run_dir),
            "--catalog-dir",
            str(catalog_dir),
            "--stage",
            "validation/observer_bridge",
            "--case-name",
            "observer_bridge_smoke",
            "--validation-status",
            "pass",
        ],
        check=True,
    )

    csv_path = catalog_dir / "HADROS3_RESULTS_CATALOG.csv"
    json_path = catalog_dir / "HADROS3_RESULTS_CATALOG.json"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert rows[0]["run_id"] == "RUN_OBS_test_20260626_abcd"
    assert rows[0]["validation_status"] == "pass"
    assert payload[0]["run_id"] == "RUN_OBS_test_20260626_abcd"
