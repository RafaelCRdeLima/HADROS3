from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from hadros3.config import defaults
from hadros3.observer_bridge import generate_observer_bridge_products
from hadros3.pipeline import render_hadros_web


def _write_dis_inputs(run_dir: Path) -> bytes:
    dis_dir = run_dir / "DIS"
    dis_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "interaction_id": "H3DIS-000001",
            "event_id": "event-inside-low",
            "source_sample_id": 1,
            "interaction_r_rg": 10.0,
            "interaction_theta_rad": math.pi / 2.0,
            "interaction_phi_rad": 0.0,
            "interaction_E_nu_local_gev": 1.0e9,
            "interaction_rho_g_cm3": 1.0e10,
            "interaction_sigma_nuN_cm2": 1.0e-33,
            "source_weight": 1.0,
            "direction_weight": 1.0,
            "interaction_weight": 0.1,
            "final_pre_event_weight": 1.0,
            "expected_interaction_weight": 0.1,
        },
        {
            "interaction_id": "H3DIS-000002",
            "event_id": "event-inside-high",
            "source_sample_id": 2,
            "interaction_r_rg": 12.0,
            "interaction_theta_rad": math.pi / 2.0,
            "interaction_phi_rad": 0.0,
            "interaction_E_nu_local_gev": 1.0e9,
            "interaction_rho_g_cm3": 2.0e10,
            "interaction_sigma_nuN_cm2": 1.0e-33,
            "source_weight": 1.0,
            "direction_weight": 1.0,
            "interaction_weight": 0.1,
            "final_pre_event_weight": 4.0,
            "expected_interaction_weight": 0.4,
        },
        {
            "interaction_id": "H3DIS-000003",
            "event_id": "event-off-fov",
            "source_sample_id": 3,
            "interaction_r_rg": 10.0,
            "interaction_theta_rad": math.pi / 2.0,
            "interaction_phi_rad": math.pi / 2.0,
            "interaction_E_nu_local_gev": 1.0e9,
            "interaction_rho_g_cm3": 1.0e10,
            "interaction_sigma_nuN_cm2": 1.0e-33,
            "source_weight": 1.0,
            "direction_weight": 1.0,
            "interaction_weight": 0.1,
            "final_pre_event_weight": 100.0,
            "expected_interaction_weight": 10.0,
        },
    ]
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    (dis_dir / "dis_accepted_interactions.jsonl").write_bytes(payload)
    return hashlib.sha256(payload).digest()


def test_observer_bridge_scores_all_dis_interactions_without_modifying_dis(tmp_path: Path) -> None:
    values = defaults()
    values["observer_camera"].update(
        {
            "observer_distance_rg": 60.0,
            "inclination_deg": 90.0,
            "azimuth_deg": 0.0,
            "field_of_view_deg": 5.0,
        }
    )
    values["observer_bridge"].update(
        {
            "fov_policy": "hard",
            "max_ranked_events": 10,
            "distance_weight_enabled": False,
            "redshift_weight_enabled": False,
            "line_of_sight_check_enabled": True,
        }
    )
    before_hash = _write_dis_inputs(tmp_path)

    summary = generate_observer_bridge_products(values, run_output_dir=tmp_path)

    after_payload = (tmp_path / "DIS" / "dis_accepted_interactions.jsonl").read_bytes()
    assert hashlib.sha256(after_payload).digest() == before_hash
    assert summary["observer_bridge_invoked"] is True
    assert summary["bridge_mode"] == "scoring_only"
    assert summary["n_interactions_input"] == 3
    assert summary["n_candidates_scored"] == 3
    assert summary["event_generation_invoked"] is False
    assert summary["powheg_invoked"] is False
    assert summary["pythia_invoked"] is False
    assert summary["geant4_invoked"] is False
    assert summary["photon_transport_invoked"] is False
    assert summary["uses_hadros_original_runtime_path"] is False
    assert summary["proxy_physics_risk"] is True
    assert summary["physics_weight_definition"] == "final_pre_event_weight"
    assert summary["final_observation_score_definition"] == "physics_weight * observer_weight"

    bridge_dir = tmp_path / "ObserverBridge"
    for filename in [
        "observer_bridge_candidates.jsonl",
        "observer_bridge_ranked_events.jsonl",
        "observer_bridge_summary.json",
        "observer_bridge_summary.csv",
        "observer_bridge_report.json",
        "observer_bridge_map.png",
        "observer_bridge_score_distribution.png",
        "observer_bridge_weight_breakdown.png",
        "observer_bridge_visibility_map.png",
        "observer_bridge_ranked_events.png",
        "observer_bridge_geometry_3d.html",
    ]:
        assert (bridge_dir / filename).exists()

    candidates = [json.loads(line) for line in (bridge_dir / "observer_bridge_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(candidates) == 3
    assert all(candidate["physics_weight"] >= 0.0 for candidate in candidates)
    assert all(candidate["observer_weight"] >= 0.0 for candidate in candidates)
    assert all(candidate["final_observation_score"] >= 0.0 for candidate in candidates)
    inside = {candidate["event_id"]: candidate for candidate in candidates}
    assert inside["event-inside-low"]["camera_fov_flag"] is True
    assert inside["event-inside-high"]["camera_fov_flag"] is True
    assert inside["event-off-fov"]["camera_fov_flag"] is False
    assert inside["event-off-fov"]["observer_weight"] == 0.0
    assert inside["event-off-fov"]["final_observation_score"] == 0.0

    ranked = [json.loads(line) for line in (bridge_dir / "observer_bridge_ranked_events.jsonl").read_text(encoding="utf-8").splitlines()]
    scores = [row["final_observation_score"] for row in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0]["event_id"] == "event-inside-high"


def test_observer_bridge_provenance_is_scoring_only(tmp_path: Path) -> None:
    values = defaults()
    values["observer_camera"].update({"observer_distance_rg": 60.0, "inclination_deg": 90.0, "azimuth_deg": 0.0})
    _write_dis_inputs(tmp_path)
    bridge_summary = generate_observer_bridge_products(values, run_output_dir=tmp_path)

    render_summary = render_hadros_web(values, root=Path.cwd(), output_dir=tmp_path, observer_bridge_summary=bridge_summary)
    provenance = json.loads(Path(render_summary["products"]["provenance"]).read_text(encoding="utf-8"))

    assert provenance["hadros3_stage"] == "H3-W0_to_H3-W8_observer_bridge_scoring"
    assert provenance["status"] == "observer_bridge_scored_no_event_generation"
    assert provenance["observer_bridge"]["observer_bridge_invoked"] is True
    assert provenance["observer_bridge"]["observer_bridge_active_filter_invoked"] is False
    assert provenance["observer_bridge"]["bridge_mode"] == "scoring_only"
    assert provenance["observer_bridge"]["event_generation_invoked"] is False
    assert provenance["observer_bridge"]["powheg_invoked"] is False
    assert provenance["observer_bridge"]["pythia_invoked"] is False
    assert provenance["observer_bridge"]["geant4_invoked"] is False
    assert provenance["observer_bridge"]["photon_transport_invoked"] is False
    assert provenance["disabled_expensive_or_future_stages"]["observer_bridge_active_filter"] == "active_H3_W8_scoring_only"
