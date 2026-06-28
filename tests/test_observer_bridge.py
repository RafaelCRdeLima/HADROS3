from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hadros3.config import defaults
from hadros3.observer_bridge import (
    _camera_basis_diagnostic,
    _camera_plane_to_overlay_image_pixel,
    _camera_preview_local_direction_for_pixel,
    _kerr_pixel_match_candidates,
    _map_kerr_match_to_overlay_pixel,
    _observer_position,
    _project_camera,
    _select_downstream_candidates,
    _spherical,
    generate_observer_bridge_products,
)
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


def _write_camera_preview(run_dir: Path) -> Path:
    camera_dir = run_dir / "CameraPreview"
    camera_dir.mkdir(parents=True, exist_ok=True)
    camera_preview_path = camera_dir / "hadros3_camera_preview.png"
    preview = np.zeros((576, 1024, 3), dtype=float)
    preview[..., 0] = np.linspace(0.02, 0.18, 1024)[None, :]
    preview[..., 1] = np.linspace(0.04, 0.20, 576)[:, None]
    preview[..., 2] = 0.12
    plt.imsave(camera_preview_path, preview)
    return camera_preview_path


def test_camera_overlay_uses_identity_camera_preview_pixel_transform() -> None:
    width = 1024
    height = 576
    matched_pixel_x = 321.25
    matched_pixel_y = 112.5

    image_x, image_y = _camera_plane_to_overlay_image_pixel(matched_pixel_x, matched_pixel_y, width, height)

    assert image_x == matched_pixel_x
    assert image_y == matched_pixel_y
    assert _camera_plane_to_overlay_image_pixel(-10.0, -20.0, width, height) == (0.0, 0.0)
    assert _camera_plane_to_overlay_image_pixel(width + 10.0, height + 20.0, width, height) == (width - 1.0, height - 1.0)


def test_map_kerr_match_to_overlay_pixel_conventions() -> None:
    width = 1024
    height = 576
    x = 120.0
    y = 80.0

    assert _map_kerr_match_to_overlay_pixel(x, y, width, height, "identity") == (x, y)
    assert _map_kerr_match_to_overlay_pixel(x, y, width, height, "flip_y") == (x, height - 1.0 - y)
    assert _map_kerr_match_to_overlay_pixel(x, y, width, height, "flip_x") == (width - 1.0 - x, y)
    assert _map_kerr_match_to_overlay_pixel(x, y, width, height, "flip_x_y") == (width - 1.0 - x, height - 1.0 - y)


def test_kerr_pixel_match_uses_cuda_preview_local_tetrad_basis() -> None:
    values = defaults()
    width = 1024
    height = 576

    center = _camera_preview_local_direction_for_pixel(width / 2 - 0.5, height / 2 - 0.5, width, height, values)
    right = _camera_preview_local_direction_for_pixel(width - 1.0, height / 2 - 0.5, width, height, values)
    top = _camera_preview_local_direction_for_pixel(width / 2 - 0.5, 0.0, width, height, values)
    bottom = _camera_preview_local_direction_for_pixel(width / 2 - 0.5, height - 1.0, width, height, values)
    up_flipped = _camera_preview_local_direction_for_pixel(width / 2 - 0.5, height - 1.0, width, height, values, basis_transform="up_flipped")
    right_flipped = _camera_preview_local_direction_for_pixel(width - 1.0, height / 2 - 0.5, width, height, values, basis_transform="right_flipped")

    assert center == (-1.0, 0.0, 0.0)
    assert top[1] > 0.0
    assert bottom[1] < 0.0
    assert right[2] > 0.0
    assert up_flipped[1] == -bottom[1]
    assert right_flipped[2] == -right[2]


def test_observer_inclination_uses_boyer_lindquist_theta_hemisphere(tmp_path: Path) -> None:
    values = defaults()
    values["observer_camera"]["observer_distance_rg"] = 60.0
    expectations = [(40.0, "positive"), (90.0, "zero"), (100.0, "negative")]
    for inclination, sign in expectations:
        values["observer_camera"]["inclination_deg"] = inclination
        observer = _observer_position(values)
        if sign == "positive":
            assert observer[2] > 0.0
        elif sign == "negative":
            assert observer[2] < 0.0
        else:
            assert abs(observer[2]) < 1.0e-9

    values["observer_camera"]["inclination_deg"] = 66.5
    diagnostic = _camera_basis_diagnostic(values, tmp_path / "camera_basis_diagnostic.json")
    payload = json.loads((tmp_path / "camera_basis_diagnostic.json").read_text(encoding="utf-8"))
    assert payload["inclination_convention"] == "theta_0_north_pi_over_2_equator"
    assert math.isclose(payload["theta_obs_used_by_camera_preview_rad"], math.radians(66.5))
    assert math.isclose(payload["theta_obs_used_by_kerr_pixel_match_rad"], math.radians(66.5))
    assert payload["camera_preview_observer_z_sign"] == "positive"
    assert payload["kerr_pixel_match_observer_z_sign"] == "positive"
    assert payload["camera_preview_observer_position"][2] > 0.0
    assert payload["kerr_pixel_match_observer_position"][2] > 0.0
    assert payload["hemisphere_consistent"] is True
    assert payload["camera_preview_png_top_direction"] == "+e_theta"
    assert payload["camera_preview_png_bottom_direction"] == "-e_theta"
    assert payload["interactive_previous_screen_up_convention"] == "-e_theta"
    assert payload["interactive_screen_up_convention"] == "+e_theta"
    assert payload["interactive_matches_camera_preview"] is True
    assert payload["basis_dot_products"]["previous_up_dot"] < 0.0
    assert payload["basis_dot_products"]["up_dot"] > 0.0
    assert diagnostic["camera_preview_observer_hemisphere"] == "north"
    assert diagnostic["kerr_pixel_match_observer_hemisphere"] == "north"
    assert diagnostic["hemisphere_consistent"] is True
    assert diagnostic["interactive_matches_camera_preview"] is True


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
            "observer_bridge_orientation_diagnostics_enabled": False,
        }
    )
    before_hash = _write_dis_inputs(tmp_path)
    _write_camera_preview(tmp_path)

    summary = generate_observer_bridge_products(values, run_output_dir=tmp_path)

    after_payload = (tmp_path / "DIS" / "dis_accepted_interactions.jsonl").read_bytes()
    assert hashlib.sha256(after_payload).digest() == before_hash
    assert summary["observer_bridge_invoked"] is True
    assert summary["status"] == "ok"
    assert summary["observer_bridge_stage_complete"] is True
    assert summary["observer_bridge_postprocessing_complete"] is True
    assert summary["observer_bridge_required_products_complete"] is True
    assert summary["observer_bridge_partial_state_detected"] is False
    assert summary["required_observer_bridge_products_missing"] == []
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
        "observer_bridge_overlay_background_audit.json",
        "observer_bridge_background_comparison.png",
        "observer_bridge_overlay_hemisphere_diagnostic.png",
        "observer_candidate_kerr_pixel_map.jsonl",
        "observer_bridge_kerr_interactive_view.html",
    ]:
        assert (bridge_dir / filename).exists()
    for filename in [
        "observer_overlay_orientation_markers.png",
        "observer_overlay_orientation_markers.json",
        "observer_overlay_orientation_full_diagnostic.png",
    ]:
        assert not (bridge_dir / filename).exists()
    assert (bridge_dir / "observer_bridge_camera_overlay.png").stat().st_size > 0
    assert (bridge_dir / "observer_bridge_kerr_interactive_view.html").stat().st_size > 0
    assert (bridge_dir / "observer_candidate_kerr_pixel_map.jsonl").exists()
    assert (bridge_dir / "observer_bridge_summary.partial.json").exists()
    assert (bridge_dir / "observer_bridge_report.partial.json").exists()
    final_summary = json.loads((bridge_dir / "observer_bridge_summary.json").read_text(encoding="utf-8"))
    assert final_summary["status"] == "ok"
    assert final_summary["observer_bridge_stage_complete"] is True
    assert final_summary["observer_bridge_postprocessing_complete"] is True
    assert final_summary["observer_bridge_required_products_complete"] is True
    assert final_summary["required_observer_bridge_products_present"] is True
    assert final_summary["required_observer_bridge_products_missing"] == []

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
    assert summary["products"]["observer_bridge_camera_overlay"].endswith("observer_bridge_camera_overlay.png")
    assert summary["products"]["observer_candidate_kerr_pixel_map"].endswith("observer_candidate_kerr_pixel_map.jsonl")
    assert summary["products"]["observer_bridge_kerr_interactive_view"].endswith("observer_bridge_kerr_interactive_view.html")
    assert summary["camera_overlay_background_source"].endswith("CameraPreview/hadros3_camera_preview.png")
    assert summary["camera_overlay_resolution_px"] == "1024x576"
    assert summary["observer_bridge_overlay_background_audit_generated"] is True
    assert summary["observer_bridge_background_comparison_generated"] is True
    assert summary["background_hash_match"] is True
    assert summary["background_transform_applied"] == "none"
    assert summary["candidate_overlay_projection_model"] == "kerr_geodesic_pixel_match"
    assert summary["candidate_overlay_kerr_lensed"] is True
    assert summary["candidate_overlay_not_ray_traced"] is False
    assert summary["candidate_overlay_physics_risk"] is False
    assert summary["candidate_overlay_alignment"] == "camera_preview_pixel_plane"
    assert summary["kerr_pixel_match_coordinate_convention"] == "camera_preview_pixel_grid"
    assert summary["camera_preview_pixel_convention"] == "ppm_top_left_rows"
    assert summary["overlay_image_coordinate_convention"] == "top_left_image"
    assert summary["overlay_image_coordinate_transform"] == "identity_x_y"
    assert summary["matching_ray_basis_transform"] == "cuda_preview_local_tetrad"
    assert summary["inclination_convention"] == "theta_0_north_pi_over_2_equator"
    assert summary["camera_preview_observer_hemisphere"] == "equatorial"
    assert summary["kerr_pixel_match_observer_hemisphere"] == "equatorial"
    assert summary["hemisphere_consistent"] is True
    assert summary["overlay_hemisphere_validated"] is True
    assert summary["observer_bridge_overlay_hemisphere_diagnostic_generated"] is True
    assert summary["overlay_hemisphere_diagnostic_selected_panel"] == "A: theta_obs = inclination_deg"
    assert summary["kerr_pixel_match_basis_validated"] is True
    assert summary["camera_preview_matching_basis_consistent"] is True
    assert summary["camera_basis_diagnostic_generated"] is True
    assert summary["observer_bridge_orientation_diagnostics_enabled"] is False
    assert summary["overlay_orientation_diagnostic_generated"] is False
    assert summary["observer_overlay_orientation_markers_generated"] is False
    assert summary["observer_overlay_orientation_full_diagnostic_generated"] is False
    assert summary["orientation_marker_selected_hypothesis"] is None
    assert summary["orientation_marker_selected_pixel_transform"] is None
    assert summary["orientation_marker_selected_basis_transform"] is None
    assert summary["orientation_marker_mean_pixel_error"] is None
    assert summary["candidate_overlay_pixel_y_convention"] == "image_top_left"
    assert summary["candidate_overlay_y_axis_flipped_for_image"] is False
    assert summary["overlay_orientation_validated"] is True
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
    assert report["kerr_pixel_match_coordinate_convention"] == "camera_preview_pixel_grid"
    assert report["camera_preview_pixel_convention"] == "ppm_top_left_rows"
    assert report["overlay_image_coordinate_convention"] == "top_left_image"
    assert report["overlay_image_coordinate_transform"] == "identity_x_y"
    assert report["matching_ray_basis_transform"] == "cuda_preview_local_tetrad"
    assert report["inclination_convention"] == "theta_0_north_pi_over_2_equator"
    assert report["camera_preview_observer_hemisphere"] == "equatorial"
    assert report["kerr_pixel_match_observer_hemisphere"] == "equatorial"
    assert report["hemisphere_consistent"] is True
    assert report["overlay_hemisphere_validated"] is True
    assert report["observer_bridge_overlay_hemisphere_diagnostic_generated"] is True
    assert report["overlay_hemisphere_diagnostic_selected_panel"] == "A: theta_obs = inclination_deg"
    assert report["kerr_pixel_match_basis_validated"] is True
    assert report["camera_preview_matching_basis_consistent"] is True
    assert report["camera_basis_diagnostic_generated"] is True
    assert report["observer_bridge_orientation_diagnostics_enabled"] is False
    assert report["overlay_orientation_diagnostic_generated"] is False
    assert report["observer_overlay_orientation_markers_generated"] is False
    assert report["observer_overlay_orientation_full_diagnostic_generated"] is False
    assert report["orientation_marker_selected_hypothesis"] is None
    assert report["orientation_marker_mean_pixel_error"] is None
    assert report["candidate_overlay_pixel_y_convention"] == "image_top_left"
    assert report["candidate_overlay_y_axis_flipped_for_image"] is False
    assert report["overlay_orientation_validated"] is True
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
    assert '"screen_up"' in html
    assert "scene.camera.screen_up" in html
    assert '"screen_up_convention": "+e_theta"' in html
    assert '"camera_preview_png_top_direction": "+e_theta"' in html

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


def test_observer_bridge_orientation_diagnostics_can_be_enabled(tmp_path: Path, monkeypatch) -> None:
    values = defaults()
    values["observer_camera"].update({"observer_distance_rg": 60.0, "inclination_deg": 90.0, "azimuth_deg": 0.0})
    values["observer_bridge"].update(
        {
            "candidate_overlay_mapping": "geometric_proxy",
            "observer_bridge_orientation_diagnostics_enabled": True,
            "interactive_max_candidates": 2,
            "interactive_max_rays": 0,
        }
    )
    _write_dis_inputs(tmp_path)

    def fake_basis_diagnostic(candidates, ranked, values, run_output_dir, path, top_n):
        path.write_bytes(b"diagnostic")
        return {
            "overlay_orientation_diagnostic_generated": True,
            "overlay_orientation_diagnostic": str(path),
            "overlay_orientation_diagnostic_selected_panel": "fake",
        }

    def fake_full_diagnostic(candidates, ranked, values, run_output_dir, output_dir, *, top_n):
        markers_png = output_dir / "observer_overlay_orientation_markers.png"
        markers_json = output_dir / "observer_overlay_orientation_markers.json"
        full_png = output_dir / "observer_overlay_orientation_full_diagnostic.png"
        markers_png.write_bytes(b"markers")
        markers_json.write_text(json.dumps({"selected_hypothesis": "identity"}) + "\n", encoding="utf-8")
        full_png.write_bytes(b"full")
        return {
            "observer_overlay_orientation_markers_generated": True,
            "observer_overlay_orientation_markers_json": str(markers_json),
            "observer_overlay_orientation_markers_png": str(markers_png),
            "observer_overlay_orientation_full_diagnostic_generated": True,
            "observer_overlay_orientation_full_diagnostic": str(full_png),
            "orientation_marker_selected_hypothesis": "identity",
            "orientation_marker_selected_pixel_transform": "identity",
            "orientation_marker_selected_basis_transform": "cuda_preview_local_tetrad",
            "orientation_marker_mean_pixel_error": 0.0,
        }

    monkeypatch.setattr("hadros3.observer_bridge._draw_overlay_basis_orientation_diagnostic", fake_basis_diagnostic)
    monkeypatch.setattr("hadros3.observer_bridge._draw_full_orientation_diagnostic", fake_full_diagnostic)

    summary = generate_observer_bridge_products(values, run_output_dir=tmp_path)

    bridge_dir = tmp_path / "ObserverBridge"
    assert summary["observer_bridge_orientation_diagnostics_enabled"] is True
    assert summary["overlay_orientation_diagnostic_generated"] is True
    assert summary["observer_overlay_orientation_markers_generated"] is True
    assert summary["observer_overlay_orientation_full_diagnostic_generated"] is True
    assert summary["observer_bridge_required_products_complete"] is True
    assert (bridge_dir / "observer_bridge_camera_overlay.png").exists()
    assert (bridge_dir / "observer_candidate_kerr_pixel_map.jsonl").exists()
    assert (bridge_dir / "observer_bridge_kerr_interactive_view.html").exists()
    assert (bridge_dir / "observer_overlay_orientation_markers.png").exists()
    assert (bridge_dir / "observer_overlay_orientation_markers.json").exists()
    assert (bridge_dir / "observer_overlay_orientation_full_diagnostic.png").exists()


def test_observer_bridge_provenance_is_scoring_only(tmp_path: Path) -> None:
    values = defaults()
    values["observer_camera"].update({"observer_distance_rg": 60.0, "inclination_deg": 90.0, "azimuth_deg": 0.0})
    values["observer_bridge"].update({"kerr_pixel_match_resolution_x": 9, "kerr_pixel_match_resolution_y": 5, "kerr_pixel_match_tolerance_rg": 5.0, "interactive_max_rays": 2})
    _write_dis_inputs(tmp_path)
    _write_camera_preview(tmp_path)
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
    assert provenance["observer_bridge"]["observer_bridge_overlay_background_audit_generated"] is True
    assert provenance["observer_bridge"]["observer_bridge_background_comparison_generated"] is True
    assert provenance["observer_bridge"]["background_hash_match"] is True
    assert provenance["observer_bridge"]["background_transform_applied"] == "none"
    assert provenance["observer_bridge"]["candidate_overlay_projection_model"] == "kerr_geodesic_pixel_match"
    assert provenance["observer_bridge"]["candidate_overlay_kerr_lensed"] is True
    assert provenance["observer_bridge"]["candidate_overlay_not_ray_traced"] is False
    assert provenance["observer_bridge"]["candidate_overlay_physics_risk"] is False
    assert provenance["observer_bridge"]["candidate_overlay_alignment"] == "camera_preview_pixel_plane"
    assert provenance["observer_bridge"]["kerr_pixel_match_coordinate_convention"] == "camera_preview_pixel_grid"
    assert provenance["observer_bridge"]["camera_preview_pixel_convention"] == "ppm_top_left_rows"
    assert provenance["observer_bridge"]["overlay_image_coordinate_convention"] == "top_left_image"
    assert provenance["observer_bridge"]["overlay_image_coordinate_transform"] == "identity_x_y"
    assert provenance["observer_bridge"]["matching_ray_basis_transform"] == "cuda_preview_local_tetrad"
    assert provenance["observer_bridge"]["inclination_convention"] == "theta_0_north_pi_over_2_equator"
    assert provenance["observer_bridge"]["camera_preview_observer_hemisphere"] == "equatorial"
    assert provenance["observer_bridge"]["kerr_pixel_match_observer_hemisphere"] == "equatorial"
    assert provenance["observer_bridge"]["hemisphere_consistent"] is True
    assert provenance["observer_bridge"]["overlay_hemisphere_validated"] is True
    assert provenance["observer_bridge"]["observer_bridge_overlay_hemisphere_diagnostic_generated"] is True
    assert provenance["observer_bridge"]["overlay_hemisphere_diagnostic_selected_panel"] == "A: theta_obs = inclination_deg"
    assert provenance["observer_bridge"]["kerr_pixel_match_basis_validated"] is True
    assert provenance["observer_bridge"]["camera_preview_matching_basis_consistent"] is True
    assert provenance["observer_bridge"]["camera_basis_diagnostic_generated"] is True
    assert provenance["observer_bridge"]["observer_bridge_orientation_diagnostics_enabled"] is False
    assert provenance["observer_bridge"]["overlay_orientation_diagnostic_generated"] is False
    assert provenance["observer_bridge"]["observer_overlay_orientation_markers_generated"] is False
    assert provenance["observer_bridge"]["observer_overlay_orientation_full_diagnostic_generated"] is False
    assert provenance["observer_bridge"]["orientation_marker_selected_hypothesis"] is None
    assert provenance["observer_bridge"]["orientation_marker_mean_pixel_error"] is None
    assert provenance["observer_bridge"]["candidate_overlay_pixel_y_convention"] == "image_top_left"
    assert provenance["observer_bridge"]["candidate_overlay_y_axis_flipped_for_image"] is False
    assert provenance["observer_bridge"]["overlay_orientation_validated"] is True
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
    camera_preview_path = _write_camera_preview(tmp_path)
    camera_sha = hashlib.sha256(camera_preview_path.read_bytes()).hexdigest()

    summary = generate_observer_bridge_products(values, run_output_dir=tmp_path)

    assert (tmp_path / "ObserverBridge" / "observer_bridge_camera_overlay.png").exists()
    assert (tmp_path / "ObserverBridge" / "observer_bridge_overlay_background_audit.json").exists()
    assert (tmp_path / "ObserverBridge" / "observer_bridge_background_comparison.png").exists()
    assert summary["camera_overlay_background_source"] == str(camera_preview_path)
    assert summary["camera_overlay_resolution_px"] == "1024x576"
    assert summary["camera_preview_path"] == str(camera_preview_path)
    assert summary["overlay_background_source_path"] == str(camera_preview_path)
    assert summary["camera_preview_sha256"] == camera_sha
    assert summary["overlay_background_sha256"] == camera_sha
    assert summary["background_hash_match"] is True
    assert summary["background_transform_applied"] == "none"
    assert summary["background_is_stale"] is False
    assert summary["candidate_overlay_projection_model"] == "kerr_geodesic_pixel_match"
    assert summary["candidate_overlay_not_ray_traced"] is False
    assert summary["overlay_candidate_source"] == "ObserverBridge closest-ray map"


def test_observer_bridge_camera_overlay_prefers_primary_branch_source_when_available(tmp_path: Path) -> None:
    values = defaults()
    values["observer_camera"].update({"observer_distance_rg": 60.0, "inclination_deg": 90.0, "azimuth_deg": 0.0})
    values["observer_bridge"].update({"kerr_pixel_match_resolution_x": 9, "kerr_pixel_match_resolution_y": 5, "kerr_pixel_match_tolerance_rg": 5.0})
    _write_dis_inputs(tmp_path)
    _write_camera_preview(tmp_path)
    branch_dir = tmp_path / "ObserverImageBranches"
    branch_dir.mkdir(parents=True)
    (branch_dir / "observer_image_primary_branches.jsonl").write_text(
        json.dumps(
            {
                "candidate_id": "branch-candidate",
                "candidate_rank": 1,
                "interaction_id": "branch-interaction",
                "event_id": "branch-event",
                "primary_branch_pixel_x": 120.0,
                "primary_branch_pixel_y": 90.0,
                "final_observation_score": 0.75,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = generate_observer_bridge_products(values, run_output_dir=tmp_path)

    assert (tmp_path / "ObserverBridge" / "observer_bridge_camera_overlay.png").exists()
    assert summary["overlay_candidate_source"] == "ObserverImageBranches primary branches"


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
