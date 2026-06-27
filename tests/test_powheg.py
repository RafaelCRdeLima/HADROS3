from __future__ import annotations

import hashlib
import csv
import json
from pathlib import Path

import pytest

from hadros3.config import defaults
from hadros3 import powheg as powheg_module
from hadros3.powheg import generate_powheg_products, parse_lhe_particles
from hadros3.provenance import build_provenance


def test_cpp_powheg_driver_fallbacks_match_dashboard_defaults() -> None:
    source = Path("cpp/apps/hadros3_powheg_driver.cpp").read_text(encoding="utf-8")

    assert "int events_per_candidate = 1;" in source
    assert "max_powheg_events" not in source
    assert "int events_per_candidate = 2;" not in source


def test_powheg_particle_plot_labels_use_latex_names() -> None:
    assert powheg_module._particle_latex_name(11) == r"e^{-}"
    assert powheg_module._particle_latex_name(12) == r"\nu_{e}"
    assert powheg_module._particle_latex_name(-2) == r"\bar{u}"
    assert powheg_module._particle_latex_name(-1) == r"\bar{d}"
    assert powheg_module._particle_display_name(12) == "νₑ"
    assert powheg_module._particle_display_name(11) == "e⁻"
    assert powheg_module._particle_display_name(-2) == "ū"
    assert powheg_module._particle_display_name(-1) == "d̄"


def _write_selected_candidates(run_dir: Path, rows: list[dict[str, object]] | None = None) -> Path:
    bridge_dir = run_dir / "ObserverBridge"
    bridge_dir.mkdir(parents=True, exist_ok=True)
    selected_path = bridge_dir / "observer_bridge_selected_candidates.jsonl"
    if rows is None:
        rows = [
            {
                "interaction_id": "int-high",
                "event_id": "evt-high",
                "source_sample_id": "src-high",
                "interaction_E_nu_local_gev": 4.0e9,
                "physics_weight": 0.8,
                "observer_weight": 0.9,
                "final_observation_score": 0.72,
                "selection_policy": "top_n",
                "selected_for_downstream": True,
                "downstream_stage_target": "powheg",
                "selection_rank": 1,
                "selection_reason": "rank<=2",
            },
            {
                "interaction_id": "int-mid",
                "event_id": "evt-mid",
                "source_sample_id": "src-mid",
                "interaction_E_nu_local_gev": 7.0e8,
                "physics_weight": 0.7,
                "observer_weight": 0.6,
                "final_observation_score": 0.42,
                "selection_policy": "top_n",
                "selected_for_downstream": True,
                "downstream_stage_target": "powheg",
                "selection_rank": 2,
                "selection_reason": "rank<=2",
            },
        ]
    selected_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return selected_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _requests(run_dir: Path) -> list[dict[str, object]]:
    path = run_dir / "POWHEG" / "powheg_event_requests.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_powheg_dry_run_generates_ranked_cards_without_lhe_or_observer_modification(tmp_path: Path) -> None:
    selected_path = _write_selected_candidates(tmp_path)
    before = _sha256(selected_path)
    values = defaults()
    values["powheg"].update(
        {
            "events_per_candidate": 3,
            "random_seed": 1000,
            "powheg_seed_mode": "base_plus_candidate_rank",
            "run_mode": "dry_run",
        }
    )

    summary = generate_powheg_products(values, run_output_dir=tmp_path)
    requests = _requests(tmp_path)

    assert _sha256(selected_path) == before
    assert summary["powheg_dry_run_invoked"] is True
    assert summary["powheg_invoked"] is False
    assert summary["pwhg_main_executed"] is False
    assert summary["powheg_lhe_generated"] is False
    assert summary["lhe_parser_invoked"] is False
    assert summary["powheg_lhe_products_generated"] is False
    assert summary["powheg_lhe_message"] == "No LHE available: POWHEG dry run only."
    assert summary["powheg_jobs_prepared"] == 2
    assert summary["powheg_cards_generated"] == 2
    assert summary["powheg_candidate_source"] == "ObserverBridge/observer_bridge_selected_candidates.jsonl"
    assert summary["powheg_n_selected_candidates_input"] == 2
    assert summary["powheg_selection_performed_by"] == "ObserverBridge"
    assert summary["powheg_selection_policy"] == "top_n"
    assert summary["backend_language"] == "C++17"
    assert requests[0]["interaction_id"] == "int-high"
    assert requests[1]["interaction_id"] == "int-mid"
    assert requests[0]["powheg_request_id"] == "H3PWHG-000001"
    assert len({row["powheg_request_id"] for row in requests}) == len(requests)
    assert requests[0]["powheg_seed"] == 1001
    assert requests[1]["powheg_seed"] == 1002
    assert all(row["powheg_status"] == "dry_run_ready" for row in requests)
    assert all(row["powheg_invoked"] is False for row in requests)
    assert all(row["powheg_selection_performed_by"] == "ObserverBridge" for row in requests)

    first_card = tmp_path / requests[0]["powheg_input_path"]
    card = first_card.read_text(encoding="utf-8")
    assert "pwhg_main is NOT executed" in card
    assert "numevts 3" in card
    assert "ebeam1 4.0000000000D+09" in card
    assert "ebeam2 0.938272d0" in card
    assert "lhans1" in card
    assert "lhans2" in card
    assert "Qmax" in card
    assert "iseed 1001" in card
    assert "channel_type 3" in card
    assert "vtype 2" in card
    assert not list((tmp_path / "POWHEG").rglob("*.lhe"))
    assert not (tmp_path / "POWHEG" / "powheg_lhe_particles.jsonl").exists()
    validation = json.loads((tmp_path / "POWHEG" / "powheg_validation_report.json").read_text(encoding="utf-8"))
    assert validation["run_mode"] == "dry_run"
    assert validation["powheg_run_mode"] == "dry_run"
    assert validation["powheg_invoked"] is False
    assert validation["pwhg_main_executed"] is False
    assert validation["powheg_lhe_generated"] is False
    assert validation["lhe_parser_invoked"] is False
    for filename in [
        "powheg_summary.json",
        "powheg_summary.csv",
        "powheg_report.json",
        "powheg_card_preview.png",
        "powheg_energy_distribution.png",
        "powheg_job_summary.png",
    ]:
        assert (tmp_path / "POWHEG" / filename).exists()


def test_powheg_uses_selected_candidates_without_applying_own_ranking(tmp_path: Path) -> None:
    _write_selected_candidates(
        tmp_path,
        [
            {
                "interaction_id": "int-low-first",
                "event_id": "evt-low-first",
                "source_sample_id": "src-low",
                "interaction_E_nu_local_gev": 1.0e8,
                "physics_weight": 0.4,
                "observer_weight": 0.5,
                "final_observation_score": 0.2,
                "selection_policy": "all_candidates",
                "selected_for_downstream": True,
                "downstream_stage_target": "powheg",
                "selection_rank": 1,
                "selection_reason": "all_candidates",
            },
            {
                "interaction_id": "int-high-second",
                "event_id": "evt-high-second",
                "source_sample_id": "src-high",
                "interaction_E_nu_local_gev": 4.0e9,
                "physics_weight": 0.8,
                "observer_weight": 0.9,
                "final_observation_score": 0.72,
                "selection_policy": "all_candidates",
                "selected_for_downstream": True,
                "downstream_stage_target": "powheg",
                "selection_rank": 2,
                "selection_reason": "all_candidates",
            },
        ],
    )
    values = defaults()
    values["powheg"].update({"events_per_candidate": 4, "random_seed": 700})

    summary = generate_powheg_products(values, run_output_dir=tmp_path)
    requests = _requests(tmp_path)
    assert [row["interaction_id"] for row in requests] == ["int-low-first", "int-high-second"]
    assert [row["powheg_seed"] for row in requests] == [701, 702]
    assert summary["powheg_jobs_prepared"] == 2
    assert summary["n_powheg_jobs_requested"] == 2
    assert summary["events_per_candidate_requested"] == 4
    assert summary["powheg_selection_performed_by"] == "ObserverBridge"

    generate_powheg_products(values, run_output_dir=tmp_path)
    repeated = _requests(tmp_path)
    assert [row["powheg_seed"] for row in repeated] == [701, 702]

def test_powheg_fails_without_selected_candidates(tmp_path: Path) -> None:
    values = defaults()
    with pytest.raises(FileNotFoundError, match="ObserverBridge selected candidates not found"):
        generate_powheg_products(values, run_output_dir=tmp_path)


def test_powheg_provenance_marks_dry_run_without_real_powheg_invocation(tmp_path: Path) -> None:
    _write_selected_candidates(tmp_path)
    values = defaults()
    summary = generate_powheg_products(values, run_output_dir=tmp_path)
    provenance = build_provenance(
        root=Path.cwd(),
        values=values,
        products=summary["products"],
        validation={"configuration_valid": True},
        powheg_summary=summary,
    )

    assert provenance["hadros3_stage"] == "H3-W0_to_H3-W9a_powheg_dry_run"
    assert provenance["status"] == "powheg_jobs_prepared_no_lhe"
    assert provenance["disabled_expensive_or_future_stages"]["powheg"] == "active_H3_W9a_dry_run_no_pwhg_main"
    assert provenance["powheg"]["powheg_dry_run_invoked"] is True
    assert provenance["powheg"]["powheg_invoked"] is False
    assert provenance["powheg"]["pwhg_main_executed"] is False
    assert provenance["powheg"]["powheg_lhe_generated"] is False
    assert provenance["powheg"]["lhe_parser_invoked"] is False
    assert provenance["powheg"]["lhe_particles_are_hard_process"] is True
    assert provenance["powheg"]["hadronization_invoked"] is False
    assert provenance["powheg"]["powheg_runtime_self_contained"] is True
    assert provenance["powheg"]["backend_language"] == "C++17"
    assert provenance["powheg"]["pythia_invoked"] is False
    assert provenance["powheg"]["geant4_invoked"] is False
    assert provenance["powheg"]["photon_transport_invoked"] is False


def test_lhe_parser_extracts_particle_kinematics(tmp_path: Path) -> None:
    lhe = tmp_path / "pwgevents.lhe"
    lhe.write_text(
        """<LesHouchesEvents version="1.0">
<event>
4 1 1.0 1.0 1.0 1.0
12 -1 0 0 0 0 0.0 0.0 100.0 100.0 0.0 0.0 9.0
1 -1 0 0 501 0 0.0 0.0 -1.0 1.0 0.0 0.0 9.0
11 1 1 2 0 0 3.0 4.0 30.0 31.0 0.0 0.0 9.0
2 1 1 2 501 0 -3.0 -4.0 69.0 69.5 0.0 0.0 9.0
</event>
</LesHouchesEvents>
""",
        encoding="utf-8",
    )

    particles, events = parse_lhe_particles(lhe, powheg_job_id="H3PWHG-TEST")

    assert len(particles) == 4
    assert len(events) == 1
    assert particles[0]["pdg_id"] == 12
    assert particles[0]["particle_name"] == "nu_e"
    assert particles[0]["particle_display"] == "νₑ"
    assert particles[2]["pdg_id"] == 11
    assert particles[2]["particle_display"] == "e⁻"
    assert particles[2]["status"] == 1
    assert particles[2]["pt_gev"] == 5.0
    assert particles[2]["energy_gev"] == 31.0
    assert events[0]["n_initial_state"] == 2
    assert events[0]["n_final_state"] == 2
    assert events[0]["sum_final_energy_gev"] == 100.5
    assert events[0]["incoming_particles_display"] == ["νₑ", "d"]
    assert events[0]["outgoing_particles_display"] == ["e⁻", "u"]


def test_powheg_real_smoke_fails_clearly_without_local_pwhg_main(tmp_path: Path, monkeypatch) -> None:
    _write_selected_candidates(tmp_path)
    values = defaults()
    values["powheg"].update({"run_mode": "real_smoke", "events_per_candidate": 2})
    monkeypatch.setattr(powheg_module, "POWHEG_BINARY", tmp_path / "missing" / "pwhg_main")

    with pytest.raises(FileNotFoundError, match="Local POWHEG pwhg_main not found"):
        generate_powheg_products(values, run_output_dir=tmp_path)


def test_powheg_real_smoke_executes_local_pwhg_main_and_generates_parseable_lhe(tmp_path: Path, monkeypatch) -> None:
    selected_path = _write_selected_candidates(tmp_path)
    before = _sha256(selected_path)
    fake_pwhg = tmp_path / "external" / "powheg" / "build" / "DIS" / "pwhg_main"
    fake_pwhg.parent.mkdir(parents=True, exist_ok=True)
    fake_pwhg.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
test -f powheg.input
cat > pwgevents.lhe <<'LHE'
<LesHouchesEvents version="1.0">
<header></header>
<init></init>
<event>
4 1 1.0 1.0 1.0 1.0
12 -1 0 0 0 0 0.0 0.0 4.0000000000E+09 4.0000000000E+09 0.0 0.0 9.0
1 -1 0 0 501 0 0.0 0.0 -9.3827200000E-01 9.3827200000E-01 0.0 0.0 9.0
11 1 1 2 0 0 1.0E+03 2.0E+03 3.0E+03 3.8E+03 5.11E-04 0.0 9.0
2 1 1 2 501 0 -1.0E+03 -2.0E+03 3.999996E+09 3.999996E+09 0.0 0.0 9.0
</event>
</LesHouchesEvents>
LHE
""",
        encoding="utf-8",
    )
    fake_pwhg.chmod(0o755)
    monkeypatch.setattr(powheg_module, "POWHEG_BINARY", fake_pwhg)

    values = defaults()
    values["powheg"].update({"run_mode": "real_smoke", "events_per_candidate": 2})
    summary = generate_powheg_products(values, run_output_dir=tmp_path)
    requests = _requests(tmp_path)

    assert _sha256(selected_path) == before
    assert summary["powheg_run_mode"] == "real_smoke"
    assert summary["powheg_dry_run_invoked"] is False
    assert summary["powheg_real_smoke_invoked"] is True
    assert summary["powheg_real_free_invoked"] is False
    assert summary["real_smoke_safety_clamp"] is True
    assert summary["real_free_mode"] is False
    assert summary["powheg_invoked"] is True
    assert summary["pwhg_main_executed"] is True
    assert summary["powheg_lhe_generated"] is True
    assert summary["lhe_parser_invoked"] is True
    assert summary["lhe_particles_are_hard_process"] is True
    assert summary["hadronization_invoked"] is False
    assert summary["n_powheg_jobs"] == 1
    assert summary["powheg_jobs_prepared"] == 1
    assert summary["n_lhe_events"] == 1
    assert summary["n_lhe_particles"] == 4
    assert summary["n_final_state_particles"] == 2
    assert summary["unique_particle_types"] == 4
    assert summary["powheg_lhe_products_generated"] is True
    assert summary["pythia_invoked"] is False
    assert summary["geant4_invoked"] is False
    assert summary["photon_transport_invoked"] is False
    assert requests[0]["interaction_id"] == "int-high"
    assert requests[0]["powheg_status"] == "real_smoke_lhe_generated"
    assert requests[0]["powheg_invoked"] is True
    assert requests[0]["pwhg_main_executed"] is True
    assert requests[0]["powheg_lhe_generated"] is True
    lhe = tmp_path / "POWHEG" / "powheg_lhe" / "H3PWHG-000001" / "pwgevents.lhe"
    log = tmp_path / "POWHEG" / "powheg_run_logs" / "H3PWHG-000001" / "powheg.log"
    validation = json.loads((tmp_path / "POWHEG" / "powheg_validation_report.json").read_text(encoding="utf-8"))
    assert lhe.exists()
    assert log.exists()
    text = lhe.read_text(encoding="utf-8")
    assert "<LesHouchesEvents" in text
    assert "</LesHouchesEvents>" in text
    assert text.count("<event>") == 1
    assert validation["lhe_valid"] is True
    assert validation["run_mode"] == "real_smoke"
    assert validation["powheg_run_mode"] == "real_smoke"
    assert validation["n_lhe_events"] == 1
    assert validation["real_smoke_safety_clamp"] is True
    assert validation["real_free_mode"] is False
    assert validation["lhe_parser_invoked"] is True
    assert validation["n_lhe_particles"] == 4

    particles_path = tmp_path / "POWHEG" / "powheg_lhe_particles.jsonl"
    events_path = tmp_path / "POWHEG" / "powheg_lhe_events_summary.jsonl"
    particle_summary_csv = tmp_path / "POWHEG" / "powheg_lhe_particle_summary.csv"
    particle_summary_json = tmp_path / "POWHEG" / "powheg_lhe_particle_summary.json"
    for path in [
        particles_path,
        events_path,
        particle_summary_csv,
        particle_summary_json,
        tmp_path / "POWHEG" / "powheg_lhe_particle_histogram.png",
        tmp_path / "POWHEG" / "powheg_lhe_energy_spectrum.png",
        tmp_path / "POWHEG" / "powheg_lhe_momentum_spectrum.png",
        tmp_path / "POWHEG" / "powheg_hard_process_event_display.png",
        tmp_path / "POWHEG" / "powheg_hard_process_event_display_view.html",
        tmp_path / "POWHEG" / "powheg_event_summary_table.csv",
        tmp_path / "POWHEG" / "powheg_particle_table.csv",
        tmp_path / "POWHEG" / "powheg_particle_table.html",
        tmp_path / "POWHEG" / "powheg_particle_content_report.json",
        tmp_path / "POWHEG" / "powheg_lhe_event_view.html",
    ]:
        assert path.exists()
    particles = [json.loads(line) for line in particles_path.read_text(encoding="utf-8").splitlines()]
    assert particles[0]["pdg_id"] == 12
    assert particles[0]["particle_name"] == "nu_e"
    assert particles[0]["particle_display"] == "νₑ"
    assert particles[2]["pt_gev"] > 0.0
    particle_summary = json.loads(particle_summary_json.read_text(encoding="utf-8"))
    assert any(row["particle_name"] == "e-" and row["final_state_count"] == 1 for row in particle_summary)
    assert any(row["particle_display"] == "e⁻" and row["final_state_count"] == 1 for row in particle_summary)
    with particle_summary_csv.open(encoding="utf-8", newline="") as handle:
        summary_header = next(csv.reader(handle))
    assert "particle_display" in summary_header
    assert "particle_latex" in summary_header
    assert "initial_state_count" in summary_header
    assert "final_state_count" in summary_header
    assert "mean_pt_gev" in summary_header
    assert "max_pt_gev" in summary_header
    with (tmp_path / "POWHEG" / "powheg_particle_table.csv").open(encoding="utf-8", newline="") as handle:
        particle_header = next(csv.reader(handle))
    for column in [
        "particle_display",
        "particle_name",
        "pdg_id",
        "status",
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
    ]:
        assert column in particle_header
    particle_table_html = (tmp_path / "POWHEG" / "powheg_particle_table.html").read_text(encoding="utf-8")
    assert "POWHEG Particle Table" in particle_table_html
    assert "powheg_particle_table.csv" in particle_table_html
    assert "νₑ" in particle_table_html
    content_report = json.loads((tmp_path / "POWHEG" / "powheg_particle_content_report.json").read_text(encoding="utf-8"))
    assert "why_u_d_c_s_can_appear" in content_report
    assert content_report["pythia_invoked"] is False
    assert summary["powheg_particle_content_report_generated"] is True
    assert summary["powheg_hard_process_event_display_generated"] is True
    assert summary["powheg_hard_process_event_display_view_generated"] is True
    assert summary["powheg_lhe_event_view_generated"] is True
    event_selector_html = (tmp_path / "POWHEG" / "powheg_hard_process_event_display_view.html").read_text(encoding="utf-8")
    assert "Hard process event" in event_selector_html
    assert '<select id="event-select">' in event_selector_html
    assert "νₑ" in event_selector_html
    assert "average_multiplicity" in summary["powheg_physics_summary"]

    provenance = build_provenance(
        root=Path.cwd(),
        values=values,
        products=summary["products"],
        validation={"configuration_valid": True},
        powheg_summary=summary,
    )
    assert provenance["hadros3_stage"] == "H3-W0_to_H3-W9b_powheg_real_smoke"
    assert provenance["status"] == "powheg_real_smoke_lhe_generated"
    assert provenance["disabled_expensive_or_future_stages"]["powheg"] == "active_H3_W9b_real_smoke_local_pwhg_main"
    assert provenance["powheg"]["powheg_invoked"] is True
    assert provenance["powheg"]["powheg_real_free_invoked"] is False
    assert provenance["powheg"]["real_smoke_safety_clamp"] is True
    assert provenance["powheg"]["pwhg_main_executed"] is True
    assert provenance["powheg"]["powheg_lhe_generated"] is True
    assert provenance["powheg"]["n_lhe_events"] == 1
    assert provenance["powheg"]["lhe_parser_invoked"] is True
    assert provenance["powheg"]["lhe_particles_are_hard_process"] is True
    assert provenance["powheg"]["hadronization_invoked"] is False
    assert provenance["powheg"]["n_lhe_particles"] == 4
    assert provenance["powheg"]["pythia_invoked"] is False
    assert provenance["powheg"]["geant4_invoked"] is False
    assert provenance["powheg"]["photon_transport_invoked"] is False


def test_powheg_real_free_runs_configured_jobs_and_aggregates_lhe_products(tmp_path: Path, monkeypatch) -> None:
    _write_selected_candidates(tmp_path)
    fake_pwhg = tmp_path / "external" / "powheg" / "build" / "DIS" / "pwhg_main"
    fake_pwhg.parent.mkdir(parents=True, exist_ok=True)
    fake_pwhg.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
test -f powheg.input
numevts="$(awk '$1 == "numevts" {print $2}' powheg.input)"
cat > pwgevents.lhe <<'LHE_HEAD'
<LesHouchesEvents version="1.0">
<header></header>
<init></init>
LHE_HEAD
for i in $(seq 1 "${numevts}"); do
cat >> pwgevents.lhe <<'LHE_EVENT'
<event>
4 1 1.0 1.0 1.0 1.0
12 -1 0 0 0 0 0.0 0.0 4.0000000000E+09 4.0000000000E+09 0.0 0.0 9.0
1 -1 0 0 501 0 0.0 0.0 -9.3827200000E-01 9.3827200000E-01 0.0 0.0 9.0
11 1 1 2 0 0 1.0E+03 2.0E+03 3.0E+03 3.8E+03 5.11E-04 0.0 9.0
2 1 1 2 501 0 -1.0E+03 -2.0E+03 3.999996E+09 3.999996E+09 0.0 0.0 9.0
</event>
LHE_EVENT
done
cat >> pwgevents.lhe <<'LHE_TAIL'
</LesHouchesEvents>
LHE_TAIL
""",
        encoding="utf-8",
    )
    fake_pwhg.chmod(0o755)
    monkeypatch.setattr(powheg_module, "POWHEG_BINARY", fake_pwhg)

    values = defaults()
    values["powheg"].update({"run_mode": "real_free", "events_per_candidate": 2})
    summary = generate_powheg_products(values, run_output_dir=tmp_path)
    requests = _requests(tmp_path)

    assert summary["powheg_run_mode"] == "real_free"
    assert summary["powheg_real_free_invoked"] is True
    assert summary["powheg_real_smoke_invoked"] is False
    assert summary["real_free_mode"] is True
    assert summary["real_smoke_safety_clamp"] is False
    assert summary["powheg_invoked"] is True
    assert summary["pwhg_main_executed"] is True
    assert summary["powheg_lhe_generated"] is True
    assert summary["events_per_candidate"] == 2
    assert summary["n_powheg_jobs_requested"] == 2
    assert summary["n_powheg_jobs_run"] == 2
    assert summary["events_per_candidate_requested"] == 2
    assert summary["n_lhe_events"] == 4
    assert summary["n_lhe_events_total"] == 4
    assert summary["n_lhe_particles"] == 16
    assert summary["n_final_state_particles"] == 8
    assert summary["powheg_lhe_products_generated"] is True
    assert len(summary["powheg_lhe_paths"]) == 2
    assert all(Path(path).exists() for path in summary["powheg_lhe_paths"])
    assert [row["powheg_status"] for row in requests] == ["real_free_lhe_generated", "real_free_lhe_generated"]
    assert all(row["powheg_invoked"] is True for row in requests)
    assert all(row["n_lhe_events"] == 2 for row in requests)

    events_path = tmp_path / "POWHEG" / "powheg_lhe_events_summary.jsonl"
    particles_path = tmp_path / "POWHEG" / "powheg_lhe_particles.jsonl"
    assert len(events_path.read_text(encoding="utf-8").splitlines()) == 4
    assert len(particles_path.read_text(encoding="utf-8").splitlines()) == 16
    assert (tmp_path / "POWHEG" / "powheg_event_summary_table.csv").exists()
    assert (tmp_path / "POWHEG" / "powheg_particle_table.csv").exists()
    assert (tmp_path / "POWHEG" / "powheg_particle_table.html").exists()
    assert (tmp_path / "POWHEG" / "powheg_lhe_event_view.html").exists()
    assert (tmp_path / "POWHEG" / "powheg_hard_process_event_display.png").exists()
    assert (tmp_path / "POWHEG" / "powheg_hard_process_event_display_view.html").exists()
    validation = json.loads((tmp_path / "POWHEG" / "powheg_validation_report.json").read_text(encoding="utf-8"))
    assert validation["powheg_run_mode"] == "real_free"
    assert validation["n_powheg_jobs_run"] == 2
    assert validation["n_lhe_events_total"] == 4
    assert validation["real_free_mode"] is True
    assert validation["real_smoke_safety_clamp"] is False
    assert validation["pythia_invoked"] is False
    assert validation["geant4_invoked"] is False
    assert validation["photon_transport_invoked"] is False

    provenance = build_provenance(
        root=Path.cwd(),
        values=values,
        products=summary["products"],
        validation={"configuration_valid": True},
        powheg_summary=summary,
    )
    assert provenance["hadros3_stage"] == "H3-W0_to_H3-W9b_powheg_real_free"
    assert provenance["status"] == "powheg_real_free_lhe_generated"
    assert provenance["disabled_expensive_or_future_stages"]["powheg"] == "active_H3_W9b_real_free_local_pwhg_main"
    assert provenance["powheg"]["powheg_real_free_invoked"] is True
    assert provenance["powheg"]["n_powheg_jobs_requested"] == 2
    assert provenance["powheg"]["n_powheg_jobs_run"] == 2
    assert provenance["powheg"]["events_per_candidate_requested"] == 2
    assert provenance["powheg"]["n_lhe_events_total"] == 4
    assert provenance["powheg"]["real_free_mode"] is True
    assert provenance["powheg"]["real_smoke_safety_clamp"] is False
