from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hadros3.config import defaults
from hadros3 import powheg as powheg_module
from hadros3.powheg import generate_powheg_products, parse_lhe_particles
from hadros3.provenance import build_provenance


def _write_ranked_events(run_dir: Path) -> Path:
    bridge_dir = run_dir / "ObserverBridge"
    bridge_dir.mkdir(parents=True, exist_ok=True)
    ranked_path = bridge_dir / "observer_bridge_ranked_events.jsonl"
    rows = [
        {
            "interaction_id": "int-low",
            "event_id": "evt-low",
            "source_sample_id": "src-low",
            "interaction_E_nu_local_gev": 1.0e8,
            "physics_weight": 0.4,
            "observer_weight": 0.5,
            "final_observation_score": 0.2,
        },
        {
            "interaction_id": "int-high",
            "event_id": "evt-high",
            "source_sample_id": "src-high",
            "interaction_E_nu_local_gev": 4.0e9,
            "physics_weight": 0.8,
            "observer_weight": 0.9,
            "final_observation_score": 0.72,
        },
        {
            "interaction_id": "int-mid",
            "event_id": "evt-mid",
            "source_sample_id": "src-mid",
            "interaction_E_nu_local_gev": 7.0e8,
            "physics_weight": 0.7,
            "observer_weight": 0.6,
            "final_observation_score": 0.42,
        },
    ]
    ranked_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return ranked_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _requests(run_dir: Path) -> list[dict[str, object]]:
    path = run_dir / "POWHEG" / "powheg_event_requests.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_powheg_dry_run_generates_ranked_cards_without_lhe_or_observer_modification(tmp_path: Path) -> None:
    ranked_path = _write_ranked_events(tmp_path)
    before = _sha256(ranked_path)
    values = defaults()
    values["powheg"].update(
        {
            "ranking_policy": "top_score",
            "max_powheg_events": 2,
            "events_per_candidate": 3,
            "random_seed": 1000,
            "powheg_seed_mode": "base_plus_candidate_rank",
            "run_mode": "dry_run",
        }
    )

    summary = generate_powheg_products(values, run_output_dir=tmp_path)
    requests = _requests(tmp_path)

    assert _sha256(ranked_path) == before
    assert summary["powheg_dry_run_invoked"] is True
    assert summary["powheg_invoked"] is False
    assert summary["pwhg_main_executed"] is False
    assert summary["powheg_lhe_generated"] is False
    assert summary["lhe_parser_invoked"] is False
    assert summary["powheg_lhe_products_generated"] is False
    assert summary["powheg_lhe_message"] == "No LHE available: POWHEG dry run only."
    assert summary["powheg_jobs_prepared"] == 2
    assert summary["powheg_cards_generated"] == 2
    assert summary["backend_language"] == "C++17"
    assert requests[0]["interaction_id"] == "int-high"
    assert requests[1]["interaction_id"] == "int-mid"
    assert requests[0]["powheg_request_id"] == "H3PWHG-000001"
    assert len({row["powheg_request_id"] for row in requests}) == len(requests)
    assert requests[0]["powheg_seed"] == 1002
    assert requests[1]["powheg_seed"] == 1003
    assert all(row["powheg_status"] == "dry_run_ready" for row in requests)
    assert all(row["powheg_invoked"] is False for row in requests)

    first_card = tmp_path / requests[0]["powheg_input_path"]
    card = first_card.read_text(encoding="utf-8")
    assert "pwhg_main is NOT executed" in card
    assert "numevts 3" in card
    assert "ebeam1 4.0000000000D+09" in card
    assert "ebeam2 0.938272d0" in card
    assert "lhans1" in card
    assert "lhans2" in card
    assert "Qmax" in card
    assert "iseed 1002" in card
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


def test_powheg_ranking_policies_and_seed_reproducibility(tmp_path: Path) -> None:
    _write_ranked_events(tmp_path)
    values = defaults()
    values["powheg"].update({"ranking_policy": "score_threshold", "min_final_observation_score": 0.4, "max_powheg_events": 10, "random_seed": 700})

    generate_powheg_products(values, run_output_dir=tmp_path)
    threshold_requests = _requests(tmp_path)
    assert [row["interaction_id"] for row in threshold_requests] == ["int-high", "int-mid"]
    assert [row["powheg_seed"] for row in threshold_requests] == [702, 703]

    generate_powheg_products(values, run_output_dir=tmp_path)
    repeated = _requests(tmp_path)
    assert [row["powheg_seed"] for row in repeated] == [702, 703]

    values["powheg"].update({"ranking_policy": "all_candidates", "max_powheg_events": 10, "min_final_observation_score": 0.0})
    summary = generate_powheg_products(values, run_output_dir=tmp_path)
    all_requests = _requests(tmp_path)
    assert summary["powheg_jobs_prepared"] == 3
    assert [row["interaction_id"] for row in all_requests] == ["int-high", "int-mid", "int-low"]


def test_powheg_provenance_marks_dry_run_without_real_powheg_invocation(tmp_path: Path) -> None:
    _write_ranked_events(tmp_path)
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
    assert particles[2]["pdg_id"] == 11
    assert particles[2]["status"] == 1
    assert particles[2]["pt_gev"] == 5.0
    assert particles[2]["energy_gev"] == 31.0
    assert events[0]["n_initial_state"] == 2
    assert events[0]["n_final_state"] == 2
    assert events[0]["sum_final_energy_gev"] == 100.5


def test_powheg_real_smoke_fails_clearly_without_local_pwhg_main(tmp_path: Path, monkeypatch) -> None:
    _write_ranked_events(tmp_path)
    values = defaults()
    values["powheg"].update({"run_mode": "real_smoke", "max_powheg_events": 3, "events_per_candidate": 2})
    monkeypatch.setattr(powheg_module, "POWHEG_BINARY", tmp_path / "missing" / "pwhg_main")

    with pytest.raises(FileNotFoundError, match="Local POWHEG pwhg_main not found"):
        generate_powheg_products(values, run_output_dir=tmp_path)


def test_powheg_real_smoke_executes_local_pwhg_main_and_generates_parseable_lhe(tmp_path: Path, monkeypatch) -> None:
    ranked_path = _write_ranked_events(tmp_path)
    before = _sha256(ranked_path)
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
    values["powheg"].update({"run_mode": "real_smoke", "max_powheg_events": 10, "events_per_candidate": 2})
    summary = generate_powheg_products(values, run_output_dir=tmp_path)
    requests = _requests(tmp_path)

    assert _sha256(ranked_path) == before
    assert summary["powheg_run_mode"] == "real_smoke"
    assert summary["powheg_dry_run_invoked"] is False
    assert summary["powheg_real_smoke_invoked"] is True
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
    ]:
        assert path.exists()
    particles = [json.loads(line) for line in particles_path.read_text(encoding="utf-8").splitlines()]
    assert particles[0]["pdg_id"] == 12
    assert particles[0]["particle_name"] == "nu_e"
    assert particles[2]["pt_gev"] > 0.0
    particle_summary = json.loads(particle_summary_json.read_text(encoding="utf-8"))
    assert any(row["particle_name"] == "e-" and row["final_state_count"] == 1 for row in particle_summary)

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
