from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hadros3.config import defaults
from hadros3.powheg import generate_powheg_products
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
    assert provenance["powheg"]["powheg_runtime_self_contained"] is True
    assert provenance["powheg"]["backend_language"] == "C++17"
    assert provenance["powheg"]["pythia_invoked"] is False
    assert provenance["powheg"]["geant4_invoked"] is False
    assert provenance["powheg"]["photon_transport_invoked"] is False
