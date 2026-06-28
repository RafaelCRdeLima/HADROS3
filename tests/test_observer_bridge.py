from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from hadros3.config import defaults
from hadros3.observer_bridge import _kerr_pixel_match_candidates, _project_camera, _select_downstream_candidates, _spherical, generate_observer_bridge_products
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
            "kerr_pixel_match_resolution_x": 9,
            "kerr_pixel_match_resolution_y": 5,
            "kerr_pixel_match_tolerance_rg": 5.0,
            "interactive_max_candidates": 3,
            "interactive_max_rays": 2,
            "interactive_ray_stride": 8,
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
        "observer_bridge_selected_candidates.jsonl",
        "observer_bridge_selection_summary.json",
        "observer_bridge_summary.json",
        "observer_bridge_summary.csv",
        "observer_bridge_report.json",
        "observer_bridge_map.png",
        "observer_bridge_score_distribution.png",
        "observer_bridge_weight_breakdown.png",
        "observer_bridge_visibility_map.png",
        "observer_bridge_ranked_events.png",
        "observer_bridge_geometry_3d.html",
        "observer_bridge_camera_view.png",
        "observer_bridge_camera_overlay.png",
        "observer_candidate_kerr_pixel_map.jsonl",
        "observer_bridge_kerr_interactive_view.html",
    ]:
        assert (bridge_dir / filename).exists()

    assert summary["observer_bridge_camera_view_generated"] is True
    assert summary["camera_view_projection_model"] == "geometric_pinhole_proxy"
    assert summary["camera_view_projection_physics_risk"] is True
    assert summary["not_ray_traced"] is True
    assert summary["medium_renderer_used"] is True
    assert summary["density_model_theta_is_hard_cut"] is False
    assert summary["half_opening_angle_interpretation"] == "gaussian_width_not_boundary"
    assert summary["camera_view_candidates_plotted"] == 3
    assert 0 <= summary["camera_view_candidates_inside_fov"] <= 3
    assert summary["camera_view_top_n"] == 5
    assert summary["observer_bridge_camera_overlay_generated"] is True
    assert summary["camera_overlay_background_source"] == "CameraPreview renderer 1024x576"
    assert summary["camera_overlay_resolution_px"] == "1024x576"
    assert summary["candidate_overlay_projection_model"] == "kerr_geodesic_pixel_match"
    assert summary["candidate_overlay_kerr_lensed"] is True
    assert summary["candidate_overlay_not_ray_traced"] is False
    assert summary["candidate_overlay_physics_risk"] is False
    assert summary["candidate_overlay_alignment"] == "camera_preview_pixel_plane"
    assert summary["kerr_geodesic_backend"] == "python_kerr_rk4_diagnostic"
    assert summary["kerr_pixel_match_resolution"] == "9x5"
    assert summary["kerr_pixel_match_n_candidates"] == 3
    assert summary["kerr_pixel_match_n_matched"] >= 1
    assert summary["kerr_pixel_match_n_unmatched"] == 3 - summary["kerr_pixel_match_n_matched"]
    assert summary["camera_overlay_candidates_plotted"] == summary["kerr_pixel_match_n_matched"]
    assert summary["camera_overlay_candidates_inside_fov"] == summary["kerr_pixel_match_n_matched"]
    assert summary["camera_overlay_top_n"] == 5
    assert summary["observer_bridge_kerr_interactive_view_generated"] is True
    assert summary["observer_bridge_selected_candidates_generated"] is True
    assert summary["observer_bridge_selection_summary_generated"] is True
    assert summary["downstream_candidate_selection_enabled"] is True
    assert summary["downstream_selection_policy"] == "top_n"
    assert summary["downstream_n_candidates_ranked"] == 3
    assert summary["downstream_n_candidates_selected"] == 3
    assert summary["downstream_stage_target"] == "powheg"
    assert summary["interactive_view_uses_kerr_ray_matching"] is True
    assert summary["interactive_view_not_final_observed_image"] is True
    assert summary["interactive_view_diagnostic_only"] is True
    assert summary["interactive_max_candidates"] == 3
    assert summary["interactive_max_rays"] == 2
    assert summary["interactive_ray_source"].startswith("observer_candidate_kerr_pixel_map.jsonl")
    assert summary["interactive_rays_displayed"] <= 2

    report = json.loads((bridge_dir / "observer_bridge_report.json").read_text(encoding="utf-8"))
    assert report["observer_bridge_camera_view_generated"] is True
    assert report["camera_view_projection_model"] == "geometric_pinhole_proxy"
    assert report["camera_view_projection_physics_risk"] is True
    assert report["not_ray_traced"] is True
    assert report["medium_renderer_used"] is True
    assert report["observer_bridge_camera_overlay_generated"] is True
    assert report["camera_overlay_resolution_px"] == "1024x576"
    assert report["candidate_overlay_projection_model"] == "kerr_geodesic_pixel_match"
    assert report["candidate_overlay_kerr_lensed"] is True
    assert report["candidate_overlay_not_ray_traced"] is False
    assert report["candidate_overlay_physics_risk"] is False
    assert report["candidate_overlay_alignment"] == "camera_preview_pixel_plane"
    assert report["kerr_pixel_match_n_candidates"] == 3
    assert report["observer_bridge_kerr_interactive_view_generated"] is True
    assert report["interactive_view_uses_kerr_ray_matching"] is True

    pixel_map = [json.loads(line) for line in (bridge_dir / "observer_candidate_kerr_pixel_map.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(pixel_map) == 3
    assert all(row["candidate_overlay_projection_model"] == "kerr_geodesic_pixel_match" for row in pixel_map)
    assert all(row["candidate_overlay_kerr_lensed"] is True for row in pixel_map)
    assert all(row["candidate_overlay_not_ray_traced"] is False for row in pixel_map)
    assert any(row["matched_pixel_found"] is True for row in pixel_map)
    html = (bridge_dir / "observer_bridge_kerr_interactive_view.html").read_text(encoding="utf-8")
    assert "Observer Bridge Kerr Interactive View" in html
    assert "kerr_geodesic_pixel_match" in html
    assert "Drag: rotate | Scroll: zoom | Shift+drag: pan" in html
    assert "mousedown" in html
    assert "mousemove" in html
    assert "mouseup" in html
    assert "wheel" in html
    assert "touchstart" in html
    assert "touchmove" in html

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
    selected = [json.loads(line) for line in (bridge_dir / "observer_bridge_selected_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(selected) == 3
    assert all(row["selected_for_downstream"] is True for row in selected)
    assert all(row["downstream_stage_target"] == "powheg" for row in selected)
    assert [row["selection_rank"] for row in selected] == [1, 2, 3]
    selection_summary = json.loads((bridge_dir / "observer_bridge_selection_summary.json").read_text(encoding="utf-8"))
    assert selection_summary["n_candidates_ranked"] == 3
    assert selection_summary["n_candidates_selected"] == 3
    assert selection_summary["selection_policy"] == "top_n"


def test_observer_bridge_downstream_selection_policies(tmp_path: Path) -> None:
    ranked = [
        {"interaction_id": "a", "final_observation_score": 0.9},
        {"interaction_id": "b", "final_observation_score": 0.5},
        {"interaction_id": "c", "final_observation_score": 0.1},
    ]
    values = defaults()

    values["observer_bridge"].update({"downstream_selection_policy": "all_candidates"})
    all_summary = _select_downstream_candidates(ranked, values, tmp_path)
    all_rows = [json.loads(line) for line in (tmp_path / "observer_bridge_selected_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all_summary["downstream_n_candidates_selected"] == 3
    assert [row["interaction_id"] for row in all_rows] == ["a", "b", "c"]

    values["observer_bridge"].update({"downstream_selection_policy": "top_n", "downstream_top_n_candidates": 2})
    top_summary = _select_downstream_candidates(ranked, values, tmp_path)
    top_rows = [json.loads(line) for line in (tmp_path / "observer_bridge_selected_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert top_summary["downstream_n_candidates_selected"] == 2
    assert [row["interaction_id"] for row in top_rows] == ["a", "b"]
    assert top_rows[0]["selection_reason"] == "rank<=2"

    values["observer_bridge"].update({"downstream_selection_policy": "score_threshold", "downstream_min_final_observation_score": 0.5})
    threshold_summary = _select_downstream_candidates(ranked, values, tmp_path)
    threshold_rows = [json.loads(line) for line in (tmp_path / "observer_bridge_selected_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert threshold_summary["downstream_n_candidates_selected"] == 2
    assert [row["interaction_id"] for row in threshold_rows] == ["a", "b"]
    assert threshold_rows[0]["selection_policy"] == "score_threshold"


def test_observer_bridge_provenance_is_scoring_only(tmp_path: Path) -> None:
    values = defaults()
    values["observer_camera"].update({"observer_distance_rg": 60.0, "inclination_deg": 90.0, "azimuth_deg": 0.0})
    values["observer_bridge"].update({"kerr_pixel_match_resolution_x": 9, "kerr_pixel_match_resolution_y": 5, "kerr_pixel_match_tolerance_rg": 5.0, "interactive_max_rays": 2})
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
    assert provenance["observer_bridge"]["observer_bridge_camera_view_generated"] is True
    assert provenance["observer_bridge"]["camera_view_projection_model"] == "geometric_pinhole_proxy"
    assert provenance["observer_bridge"]["camera_view_projection_physics_risk"] is True
    assert provenance["observer_bridge"]["not_ray_traced"] is True
    assert provenance["observer_bridge"]["medium_renderer_used"] is True
    assert provenance["observer_bridge"]["density_model_theta_is_hard_cut"] is False
    assert provenance["observer_bridge"]["half_opening_angle_interpretation"] == "gaussian_width_not_boundary"
    assert provenance["observer_bridge"]["camera_view_candidates_plotted"] == 3
    assert 0 <= provenance["observer_bridge"]["camera_view_candidates_inside_fov"] <= 3
    assert provenance["observer_bridge"]["camera_view_top_n"] == 5
    assert provenance["observer_bridge"]["observer_bridge_camera_overlay_generated"] is True
    assert provenance["observer_bridge"]["camera_overlay_resolution_px"] == "1024x576"
    assert provenance["observer_bridge"]["candidate_overlay_projection_model"] == "kerr_geodesic_pixel_match"
    assert provenance["observer_bridge"]["candidate_overlay_kerr_lensed"] is True
    assert provenance["observer_bridge"]["candidate_overlay_not_ray_traced"] is False
    assert provenance["observer_bridge"]["candidate_overlay_physics_risk"] is False
    assert provenance["observer_bridge"]["candidate_overlay_alignment"] == "camera_preview_pixel_plane"
    assert provenance["observer_bridge"]["kerr_pixel_match_n_candidates"] == 3
    assert provenance["observer_bridge"]["kerr_pixel_match_n_matched"] >= 1
    assert provenance["observer_bridge"]["camera_overlay_candidates_plotted"] == provenance["observer_bridge"]["kerr_pixel_match_n_matched"]
    assert provenance["observer_bridge"]["camera_overlay_top_n"] == 5
    assert provenance["observer_bridge"]["observer_bridge_kerr_interactive_view_generated"] is True
    assert provenance["observer_bridge"]["interactive_view_uses_kerr_ray_matching"] is True
    assert provenance["observer_bridge"]["downstream_candidate_selection_enabled"] is True
    assert provenance["observer_bridge"]["downstream_selection_policy"] == "top_n"
    assert provenance["observer_bridge"]["downstream_n_candidates_ranked"] == 3
    assert provenance["observer_bridge"]["downstream_n_candidates_selected"] == 3
    assert provenance["observer_bridge"]["downstream_stage_target"] == "powheg"
    assert provenance["observer_bridge"]["interactive_view_not_final_observed_image"] is True
    assert provenance["observer_bridge"]["interactive_view_diagnostic_only"] is True
    assert provenance["observer_bridge"]["interactive_rays_displayed"] <= 2
    assert provenance["disabled_expensive_or_future_stages"]["observer_bridge_active_filter"] == "active_H3_W8_scoring_only"


def test_observer_bridge_camera_overlay_uses_camera_preview_when_available(tmp_path: Path) -> None:
    values = defaults()
    values["observer_camera"].update({"observer_distance_rg": 60.0, "inclination_deg": 90.0, "azimuth_deg": 0.0})
    values["observer_bridge"].update({"kerr_pixel_match_resolution_x": 9, "kerr_pixel_match_resolution_y": 5, "kerr_pixel_match_tolerance_rg": 5.0, "interactive_max_rays": 2})
    _write_dis_inputs(tmp_path)
    camera_dir = tmp_path / "CameraPreview"
    camera_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(camera_dir / "hadros3_camera_preview.png", [[[0.05, 0.05, 0.08], [0.15, 0.15, 0.18]], [[0.08, 0.08, 0.12], [0.2, 0.2, 0.24]]])

    summary = generate_observer_bridge_products(values, run_output_dir=tmp_path)

    assert (tmp_path / "ObserverBridge" / "observer_bridge_camera_overlay.png").exists()
    assert summary["camera_overlay_background_source"] == "CameraPreview renderer 1024x576"
    assert summary["camera_overlay_resolution_px"] == "1024x576"
    assert summary["candidate_overlay_projection_model"] == "kerr_geodesic_pixel_match"
    assert summary["candidate_overlay_not_ray_traced"] is False


def test_kerr_pixel_match_maps_optical_axis_near_center() -> None:
    values = defaults()
    values["observer_camera"].update({"observer_distance_rg": 80.0, "inclination_deg": 90.0, "azimuth_deg": 0.0, "field_of_view_deg": 18.0})
    values["observer_bridge"].update({"kerr_pixel_match_resolution_x": 11, "kerr_pixel_match_resolution_y": 7, "kerr_pixel_match_tolerance_rg": 4.0})
    candidates = [
        {
            "interaction_id": "axis",
            "event_id": "axis",
            "source_sample_id": 1,
            "interaction_r_rg": 12.0,
            "interaction_theta_rad": math.pi / 2.0,
            "interaction_phi_rad": 0.0,
            "final_observation_score": 1.0,
        }
    ]
    rows, metadata = _kerr_pixel_match_candidates(candidates, candidates, values, overlay_width=220, overlay_height=140, top_n=1)

    assert metadata["candidate_overlay_projection_model"] == "kerr_geodesic_pixel_match"
    assert metadata["kerr_pixel_match_n_matched"] == 1
    assert rows[0]["matched_pixel_found"] is True
    assert abs(rows[0]["matched_pixel_x"] - 110.0) < 25.0
    assert abs(rows[0]["matched_pixel_y"] - 70.0) < 25.0


def test_kerr_pixel_match_tolerance_controls_matches() -> None:
    values = defaults()
    values["observer_camera"].update({"observer_distance_rg": 80.0, "inclination_deg": 90.0, "azimuth_deg": 0.0, "field_of_view_deg": 18.0})
    values["observer_bridge"].update({"kerr_pixel_match_resolution_x": 7, "kerr_pixel_match_resolution_y": 5, "kerr_pixel_match_refine_enabled": False})
    candidates = [
        {
            "interaction_id": "off-grid",
            "event_id": "off-grid",
            "source_sample_id": 1,
            "interaction_r_rg": 12.0,
            "interaction_theta_rad": math.pi / 2.0,
            "interaction_phi_rad": 0.08,
            "final_observation_score": 1.0,
        }
    ]
    values["observer_bridge"]["kerr_pixel_match_tolerance_rg"] = 0.001
    low_rows, low_metadata = _kerr_pixel_match_candidates(candidates, candidates, values, overlay_width=140, overlay_height=100, top_n=1)
    values["observer_bridge"]["kerr_pixel_match_tolerance_rg"] = 50.0
    high_rows, high_metadata = _kerr_pixel_match_candidates(candidates, candidates, values, overlay_width=140, overlay_height=100, top_n=1)

    assert low_rows[0]["matched_pixel_found"] is False
    assert high_rows[0]["matched_pixel_found"] is True
    assert low_metadata["kerr_pixel_match_n_matched"] <= high_metadata["kerr_pixel_match_n_matched"]


def test_kerr_pixel_match_does_not_fake_out_of_fov_candidate() -> None:
    values = defaults()
    values["observer_camera"].update({"observer_distance_rg": 80.0, "inclination_deg": 90.0, "azimuth_deg": 0.0, "field_of_view_deg": 12.0})
    values["observer_bridge"].update({"kerr_pixel_match_resolution_x": 9, "kerr_pixel_match_resolution_y": 5, "kerr_pixel_match_tolerance_rg": 5.0})
    candidates = [
        {
            "interaction_id": "behind-camera",
            "event_id": "behind-camera",
            "source_sample_id": 1,
            "interaction_r_rg": 120.0,
            "interaction_theta_rad": math.pi / 2.0,
            "interaction_phi_rad": 0.0,
            "final_observation_score": 1.0,
        }
    ]
    rows, metadata = _kerr_pixel_match_candidates(candidates, candidates, values, overlay_width=180, overlay_height=100, top_n=1)

    assert rows[0]["matched_pixel_found"] is False
    assert rows[0]["matched_pixel_x"] is None
    assert rows[0]["matched_pixel_y"] is None
    assert metadata["kerr_pixel_match_n_unmatched"] == 1


def test_low_spin_long_distance_kerr_match_approximates_geometric_projection() -> None:
    values = defaults()
    values["black_hole"]["spin_a"] = 0.0
    values["observer_camera"].update({"observer_distance_rg": 500.0, "inclination_deg": 90.0, "azimuth_deg": 0.0, "field_of_view_deg": 8.0})
    values["observer_bridge"].update({"kerr_pixel_match_resolution_x": 15, "kerr_pixel_match_resolution_y": 9, "kerr_pixel_match_tolerance_rg": 10.0})
    candidate = {
        "interaction_id": "weak-field",
        "event_id": "weak-field",
        "source_sample_id": 1,
        "interaction_r_rg": 18.0,
        "interaction_theta_rad": math.pi / 2.0,
        "interaction_phi_rad": 0.035,
        "final_observation_score": 1.0,
    }
    rows, _ = _kerr_pixel_match_candidates([candidate], [candidate], values, overlay_width=300, overlay_height=180, top_n=1)
    x_ndc, y_ndc, inside = _project_camera(
        _spherical(candidate["interaction_r_rg"], candidate["interaction_theta_rad"], candidate["interaction_phi_rad"]),
        values,
    )
    expected_x = 0.5 * 300 * (1.0 + x_ndc)
    expected_y = 0.5 * 180 * (1.0 - y_ndc)

    assert inside is True
    assert rows[0]["matched_pixel_found"] is True
    assert abs(rows[0]["matched_pixel_x"] - expected_x) < 45.0
    assert abs(rows[0]["matched_pixel_y"] - expected_y) < 45.0
