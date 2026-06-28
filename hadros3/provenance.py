"""Provenance writer for the HADROS3 hadros-web first stage."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .reuse import discover_original_hadros


THEORY_VERSION = "1.1"
THEORY_DOCUMENT = "docs/Theory/HADROS3_Physics_Theory.pdf"
THEORY_SOURCE_DOCUMENT = "docs/Theory/HADROS3_Physics_Theory.tex"
THEORY_COMPATIBLE_HADROS3_COMMIT = "7251499"
THEORY_GENERATION_DATE = "2026-06-26"
THEORY_PIPELINE_VERSION = "H3-W9b"
DEFAULT_SCIENTIFIC_RELEASE = {
    "software_version": "0.9.0",
    "physics_version": "1.1",
    "pipeline_version": THEORY_PIPELINE_VERSION,
    "theory_version": THEORY_VERSION,
    "theory_document": THEORY_DOCUMENT,
}
THEORY_IMPLEMENTATION_STATUS = "implemented_through_H3_W9b_powheg_real_smoke"
THEORY_IMPLEMENTED_STAGES = ["H3-W5", "H3-W6", "H3-W7", "H3-W8", "H3-W9a", "H3-W9b"]
THEORY_PLANNED_STAGES = ["H3-W10", "H3-W11", "H3-W12", "H3-W13"]


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _scientific_release(root: Path, git_commit: str | None) -> dict[str, Any]:
    release = dict(DEFAULT_SCIENTIFIC_RELEASE)
    version_path = root / "VERSION.json"
    try:
        payload = json.loads(version_path.read_text(encoding="utf-8"))
        release.update(payload)
    except Exception:
        pass
    release["git_commit"] = git_commit
    return release


def build_provenance(
    *,
    root: Path,
    values: dict[str, dict[str, Any]],
    products: dict[str, str],
    validation: dict[str, Any],
    camera_preview: dict[str, Any] | None = None,
    source_summary: dict[str, Any] | None = None,
    forward_geodesic_summary: dict[str, Any] | None = None,
    dis_summary: dict[str, Any] | None = None,
    observer_bridge_summary: dict[str, Any] | None = None,
    observer_image_branch_summary: dict[str, Any] | None = None,
    powheg_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created_utc = datetime.now(timezone.utc).isoformat()
    git_commit = _git_commit(root)
    scientific_release = _scientific_release(root, git_commit)
    theory_metadata = {
        "theory_document": scientific_release["theory_document"],
        "theory_source_document": THEORY_SOURCE_DOCUMENT,
        "theory_version": scientific_release["theory_version"],
        "theory_commit": git_commit,
        "theory_compatible_hadros3_commit": THEORY_COMPATIBLE_HADROS3_COMMIT,
        "theory_generation_date": THEORY_GENERATION_DATE,
        "theory_recorded_in_provenance_utc": created_utc,
        "theory_pipeline_version": scientific_release["pipeline_version"],
        "physics_version": scientific_release["physics_version"],
        "software_version": scientific_release["software_version"],
        "theory_implementation_status": THEORY_IMPLEMENTATION_STATUS,
        "theory_implemented_stages": THEORY_IMPLEMENTED_STAGES,
        "theory_planned_stages": THEORY_PLANNED_STAGES,
    }
    source_active = bool(source_summary and source_summary.get("source_sampler_active"))
    forward_active = bool(forward_geodesic_summary and forward_geodesic_summary.get("forward_neutrino_geodesics_invoked"))
    dis_active = bool(dis_summary and dis_summary.get("optical_depth_dis_sampler_invoked"))
    observer_bridge_active = bool(observer_bridge_summary and observer_bridge_summary.get("observer_bridge_invoked"))
    observer_image_branch_active = bool(observer_image_branch_summary and observer_image_branch_summary.get("observer_image_branch_analysis_invoked"))
    powheg_active = bool(
        powheg_summary
        and (
            powheg_summary.get("powheg_dry_run_invoked")
            or powheg_summary.get("powheg_real_smoke_invoked")
            or powheg_summary.get("powheg_real_free_invoked")
            or powheg_summary.get("powheg_invoked")
        )
    )
    powheg_real_smoke_active = bool(powheg_summary and powheg_summary.get("powheg_real_smoke_invoked"))
    powheg_real_free_active = bool(powheg_summary and powheg_summary.get("powheg_real_free_invoked"))
    paint_swatch_disk_diagnostic_mode = bool(camera_preview and camera_preview.get("paint_swatch_disk_diagnostic_mode"))
    paint_swatch_disk_uses_forced_thin_disk = bool(camera_preview and camera_preview.get("paint_swatch_disk_uses_forced_thin_disk"))
    paint_swatch_disk_physical_torus_emission = (
        bool(camera_preview.get("paint_swatch_disk_physical_torus_emission")) if camera_preview and camera_preview.get("paint_swatch_disk_diagnostic_mode") else False
    )
    source_sampler = {
        "source_sampler_active": source_active,
        "source_model": source_summary.get("source_model") if source_summary else values["uhe_neutrino_source"]["source_model"],
        "source_volume_model": source_summary.get("source_volume_model") if source_summary else "coordinate_volume",
        "direction_generator": source_summary.get("direction_generator") if source_summary else values["uhe_neutrino_source"].get("direction_model"),
        "direction_model": source_summary.get("direction_model") if source_summary else values["uhe_neutrino_source"].get("direction_model"),
        "direction_sampling_pdf": source_summary.get("direction_sampling_pdf") if source_summary else None,
        "direction_physical_pdf": source_summary.get("direction_physical_pdf") if source_summary else None,
        "direction_weight": source_summary.get("direction_weight") if source_summary else None,
        "four_momentum_sampled_in_source": False,
        "momentum_generator": source_summary.get("momentum_generator") if source_summary else values["uhe_neutrino_source"].get("momentum_generator"),
        "momentum_is_physical_kerr": source_summary.get("momentum_is_physical_kerr") if source_summary else False,
        "forward_neutrino_geodesics_invoked": forward_active,
        "optical_depth_dis_sampler_invoked": dis_active,
        "observer_bridge_invoked": observer_bridge_active,
        "observer_bridge_active_filter_invoked": False,
        "expensive_event_generation_invoked": False,
        "summary": source_summary,
    }
    dis_sampler = {
        "optical_depth_dis_sampler_invoked": dis_active,
        "dis_model": dis_summary.get("dis_model") if dis_active else values.get("dis_interaction_sampler", {}).get("dis_model"),
        "medium_model": dis_summary.get("medium_model") if dis_active else values.get("dis_interaction_sampler", {}).get("medium_model"),
        "medium_velocity_model": dis_summary.get("medium_velocity_model") if dis_active else values.get("dis_interaction_sampler", {}).get("medium_velocity_model"),
        "medium_velocity_physics_risk": dis_summary.get("medium_velocity_physics_risk") if dis_active else True,
        "density_model": dis_summary.get("density_model") if dis_active else "analytic_torus_density_v1",
        "medium_renderer_used": dis_summary.get("medium_renderer_used") if dis_active else False,
        "density_model_has_hard_radial_cut": dis_summary.get("density_model_has_hard_radial_cut") if dis_active else True,
        "density_model_theta_profile": dis_summary.get("density_model_theta_profile") if dis_active else "gaussian",
        "density_model_theta_is_hard_cut": dis_summary.get("density_model_theta_is_hard_cut") if dis_active else False,
        "half_opening_angle_interpretation": dis_summary.get("half_opening_angle_interpretation") if dis_active else "gaussian_width_not_boundary",
        "sigma_table_path": dis_summary.get("sigma_table_path") if dis_active else None,
        "sigma_table_rows": dis_summary.get("sigma_table_rows") if dis_active else None,
        "sigma_table_energy_min_gev": dis_summary.get("sigma_table_energy_min_gev", dis_summary.get("sigma_energy_min_gev")) if dis_active else None,
        "sigma_table_energy_max_gev": dis_summary.get("sigma_table_energy_max_gev", dis_summary.get("sigma_energy_max_gev")) if dis_active else None,
        "sigma_table_is_compact_builtin_adapter": dis_summary.get("sigma_table_is_compact_builtin_adapter") if dis_active else None,
        "sigma_table_physics_risk": dis_summary.get("sigma_table_physics_risk") if dis_active else None,
        "interaction_sampling_mode": dis_summary.get("interaction_sampling_mode") if dis_active else values.get("dis_interaction_sampler", {}).get("interaction_sampling_mode"),
        "random_seed": dis_summary.get("random_seed") if dis_active else values.get("dis_interaction_sampler", {}).get("random_seed"),
        "n_paths_processed": dis_summary.get("n_paths_processed") if dis_active else 0,
        "n_segments_processed": dis_summary.get("n_segments_processed") if dis_active else 0,
        "n_interactions_accepted": dis_summary.get("n_interactions_accepted") if dis_active else 0,
        "tau_min": dis_summary.get("tau_min") if dis_active else None,
        "tau_mean": dis_summary.get("tau_mean") if dis_active else None,
        "tau_max": dis_summary.get("tau_max") if dis_active else None,
        "observer_bridge_active_filter_invoked": False,
        "expensive_event_generation_invoked": False,
        "dis_backend": dis_summary.get("dis_backend") if dis_active else values.get("dis_interaction_sampler", {}).get("dis_backend"),
        "backend_language": dis_summary.get("backend_language") if dis_active else None,
        "backend_executable": dis_summary.get("backend_executable") if dis_active else None,
        "backend_kind": dis_summary.get("backend_kind") if dis_active else None,
        "backend_version_or_git_commit": dis_summary.get("backend_version_or_git_commit") if dis_active else None,
        "uses_hadros_original_runtime_path": dis_summary.get("uses_hadros_original_runtime_path") if dis_active else None,
        "python_prototype_used": dis_summary.get("python_prototype_used") if dis_active else None,
        "cpp_backend_used": dis_summary.get("cpp_backend_used") if dis_active else None,
        "cuda_backend_used": dis_summary.get("cuda_backend_used") if dis_active else None,
        "powheg_invoked": False,
        "pythia_invoked": False,
        "geant4_invoked": False,
        "summary": dis_summary,
    }
    return {
        "hadros3_stage": ("H3-W0_to_H3-W9b_powheg_real_free" if powheg_real_free_active else ("H3-W0_to_H3-W9b_powheg_real_smoke" if powheg_real_smoke_active else ("H3-W0_to_H3-W9a_powheg_dry_run" if powheg_active else ("H3-W0_to_H3-W8b_observer_image_branch_analysis" if observer_image_branch_active else ("H3-W0_to_H3-W8_observer_bridge_scoring" if observer_bridge_active else ("H3-W0_to_H3-W7_dis_interaction_sampler" if dis_active else ("H3-W0_to_H3-W6_forward_neutrino_geodesics" if forward_active else ("H3-W0_to_H3-W5_hadros_web_uhe_source_shell" if source_active else "H3-W0_to_H3-W4_hadros_web_geometry_shell")))))))),
        "status": ("powheg_real_free_lhe_generated" if powheg_real_free_active else ("powheg_real_smoke_lhe_generated" if powheg_real_smoke_active else ("powheg_jobs_prepared_no_lhe" if powheg_active else ("observer_image_branches_analyzed_no_event_generation" if observer_image_branch_active else ("observer_bridge_scored_no_event_generation" if observer_bridge_active else ("dis_interactions_sampled_no_observer_bridge" if dis_active else ("forward_geodesics_propagated_no_interactions" if forward_active else ("uhe_source_sampled_no_expensive_events" if source_active else "geometry_configured_no_expensive_events")))))))),
        "created_utc": created_utc,
        "hadros3_version": __version__,
        "git_commit": git_commit,
        "theory_document": theory_metadata["theory_document"],
        "theory_version": theory_metadata["theory_version"],
        "theory_commit": theory_metadata["theory_commit"],
        "theory_generation_date": theory_metadata["theory_generation_date"],
        "scientific_release": scientific_release,
        "scientific_theory": theory_metadata,
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": values,
        "reused_hadros_components": discover_original_hadros(),
        "disabled_expensive_or_future_stages": {
            "powheg": ("active_H3_W9b_real_free_local_pwhg_main" if powheg_real_free_active else ("active_H3_W9b_real_smoke_local_pwhg_main" if powheg_real_smoke_active else ("active_H3_W9a_dry_run_no_pwhg_main" if powheg_active else "disabled"))),
            "pythia": "disabled",
            "geant4": "disabled",
            "forward_neutrino_geodesics": "active_H3_W6" if forward_active else "not_invoked",
            "optical_depth_dis_sampler": "active_H3_W7" if dis_active else "not_invoked",
            "observer_bridge_active_filter": "active_H3_W8_scoring_only" if observer_bridge_active else "not_invoked",
            "observer_bridge_event_generation": "disabled",
        },
        "products": products,
        "camera_preview": {
            **(camera_preview or {}),
            "medium_renderer_used": False,
            "camera_preview_ray_traced": bool(camera_preview and camera_preview.get("preview_backend") not in {None, "analytic_geometry_only"}),
        } if camera_preview is not None else None,
        "paint_swatch_disk_diagnostic_mode": paint_swatch_disk_diagnostic_mode,
        "paint_swatch_disk_uses_forced_thin_disk": paint_swatch_disk_uses_forced_thin_disk,
        "paint_swatch_disk_physical_torus_emission": paint_swatch_disk_physical_torus_emission,
        "source_sampler": source_sampler,
        "uhe_neutrino_source": source_sampler,
        "forward_geodesics": {
            "forward_neutrino_geodesics_invoked": forward_active,
            "momentum_generator": forward_geodesic_summary.get("momentum_generator") if forward_active else None,
            "momentum_is_physical_kerr": forward_geodesic_summary.get("momentum_is_physical_kerr") if forward_active else False,
            "direction_model": forward_geodesic_summary.get("direction_model") if forward_active else None,
            "forward_geodesics_consumes_source_direction": True,
            "four_momentum_constructed_from_source_direction": True,
            "four_momentum_sampled_in_source": False,
            "input_source_samples": forward_geodesic_summary.get("input_source_samples") if forward_active else None,
            "forward_backend": forward_geodesic_summary.get("forward_backend") if forward_active else values["forward_geodesics"].get("forward_backend"),
            "backend_language": forward_geodesic_summary.get("backend_language") if forward_active else None,
            "backend_executable": forward_geodesic_summary.get("backend_executable") if forward_active else None,
            "backend_kind": forward_geodesic_summary.get("backend_kind") if forward_active else None,
            "backend_version_or_git_commit": forward_geodesic_summary.get("backend_version_or_git_commit") if forward_active else None,
            "uses_hadros_original_runtime_path": forward_geodesic_summary.get("uses_hadros_original_runtime_path") if forward_active else False,
            "python_prototype_used": forward_geodesic_summary.get("python_prototype_used") if forward_active else None,
            "cpp_backend_used": forward_geodesic_summary.get("cpp_backend_used") if forward_active else None,
            "cuda_backend_used": forward_geodesic_summary.get("cuda_backend_used") if forward_active else None,
            "geodesic_backend": forward_geodesic_summary.get("geodesic_backend") if forward_active else values["forward_geodesics"].get("geodesic_backend"),
            "full_kerr_geodesic": forward_geodesic_summary.get("full_kerr_geodesic") if forward_active else False,
            "theta_phi_evolution": forward_geodesic_summary.get("theta_phi_evolution") if forward_active else False,
            "uses_kerr_metric": forward_geodesic_summary.get("uses_kerr_metric") if forward_active else False,
            "uses_hamiltonian": forward_geodesic_summary.get("uses_hamiltonian") if forward_active else False,
            "uses_zamo_tetrad": forward_geodesic_summary.get("uses_zamo_tetrad") if forward_active else False,
            "uses_christoffel_or_hamiltonian": forward_geodesic_summary.get("uses_christoffel_or_hamiltonian") if forward_active else False,
            "coordinate_radial_preview": forward_geodesic_summary.get("coordinate_radial_preview") if forward_active else False,
            "medium_renderer_used": forward_geodesic_summary.get("medium_renderer_used") if forward_active else False,
            "medium_model": forward_geodesic_summary.get("medium_model") if forward_active else "analytic_torus",
            "density_model": forward_geodesic_summary.get("density_model") if forward_active else "analytic_torus_density_v1",
            "density_model_has_hard_radial_cut": forward_geodesic_summary.get("density_model_has_hard_radial_cut") if forward_active else True,
            "density_model_theta_profile": forward_geodesic_summary.get("density_model_theta_profile") if forward_active else "gaussian",
            "density_model_theta_is_hard_cut": forward_geodesic_summary.get("density_model_theta_is_hard_cut") if forward_active else False,
            "half_opening_angle_interpretation": forward_geodesic_summary.get("half_opening_angle_interpretation") if forward_active else "gaussian_width_not_boundary",
            "n_samples_requested": forward_geodesic_summary.get("n_samples_requested") if forward_active else values["forward_geodesics"].get("n_samples_to_propagate"),
            "n_samples_propagated": forward_geodesic_summary.get("n_samples_propagated") if forward_active else 0,
            "max_steps": forward_geodesic_summary.get("max_steps") if forward_active else values["forward_geodesics"].get("max_steps"),
            "initial_step_rg": forward_geodesic_summary.get("initial_step_rg") if forward_active else values["forward_geodesics"].get("initial_step_rg"),
            "outer_radius_rg": forward_geodesic_summary.get("outer_radius_rg") if forward_active else values["forward_geodesics"].get("outer_radius_rg"),
            "null_invariant_tolerance": forward_geodesic_summary.get("null_invariant_tolerance") if forward_active else values["forward_geodesics"].get("null_invariant_tolerance"),
            "killing_energy_tolerance": forward_geodesic_summary.get("killing_energy_tolerance") if forward_active else values["forward_geodesics"].get("killing_energy_tolerance"),
            "lz_tolerance": forward_geodesic_summary.get("lz_tolerance") if forward_active else values["forward_geodesics"].get("lz_tolerance"),
            "stop_condition_counts": forward_geodesic_summary.get("stop_condition_counts") if forward_active else {},
            "optical_depth_dis_sampler_invoked": dis_active,
            "observer_bridge_active_filter_invoked": False,
            "expensive_event_generation_invoked": False,
            "summary": forward_geodesic_summary,
        },
        "dis_interaction_sampler": dis_sampler,
        "observer_bridge": {
            "observer_bridge_invoked": observer_bridge_active,
            "observer_bridge_active_filter_invoked": False,
            "observer_bridge_backend": observer_bridge_summary.get("observer_bridge_backend") if observer_bridge_active else values.get("observer_bridge", {}).get("observer_bridge_backend"),
            "backend_language": observer_bridge_summary.get("backend_language") if observer_bridge_active else None,
            "backend_executable": observer_bridge_summary.get("backend_executable") if observer_bridge_active else None,
            "bridge_mode": observer_bridge_summary.get("bridge_mode") if observer_bridge_active else values.get("observer_bridge", {}).get("bridge_mode"),
            "secondary_particle_proxy_model": observer_bridge_summary.get("secondary_particle_proxy_model") if observer_bridge_active else values.get("observer_bridge", {}).get("secondary_particle_proxy_model"),
            "escape_proxy_model": observer_bridge_summary.get("escape_proxy_model") if observer_bridge_active else values.get("observer_bridge", {}).get("escape_proxy_model"),
            "visibility_model": observer_bridge_summary.get("visibility_model") if observer_bridge_active else values.get("observer_bridge", {}).get("visibility_model"),
            "redshift_proxy_model": observer_bridge_summary.get("redshift_proxy_model") if observer_bridge_active else values.get("observer_bridge", {}).get("redshift_proxy_model"),
            "line_of_sight_proxy_model": observer_bridge_summary.get("line_of_sight_proxy_model") if observer_bridge_active else values.get("observer_bridge", {}).get("line_of_sight_proxy_model"),
            "fov_policy": observer_bridge_summary.get("fov_policy") if observer_bridge_active else values.get("observer_bridge", {}).get("fov_policy"),
            "physics_weight_definition": observer_bridge_summary.get("physics_weight_definition") if observer_bridge_active else "final_pre_event_weight",
            "observer_weight_definition": observer_bridge_summary.get("observer_weight_definition") if observer_bridge_active else "escape_weight_proxy * visibility_proxy * camera_fov_weight * distance_weight * redshift_weight * line_of_sight_weight",
            "final_observation_score_definition": observer_bridge_summary.get("final_observation_score_definition") if observer_bridge_active else "physics_weight * observer_weight",
            "observer_bridge_camera_view_generated": observer_bridge_summary.get("observer_bridge_camera_view_generated") if observer_bridge_active else False,
            "camera_view_projection_model": observer_bridge_summary.get("camera_view_projection_model") if observer_bridge_active else "geometric_pinhole_proxy",
            "camera_view_projection_physics_risk": observer_bridge_summary.get("camera_view_projection_physics_risk") if observer_bridge_active else True,
            "not_ray_traced": observer_bridge_summary.get("not_ray_traced") if observer_bridge_active else True,
            "observer_bridge_stage_complete": observer_bridge_summary.get("observer_bridge_stage_complete") if observer_bridge_active else False,
            "observer_bridge_postprocessing_complete": observer_bridge_summary.get("observer_bridge_postprocessing_complete") if observer_bridge_active else False,
            "observer_bridge_required_products_complete": observer_bridge_summary.get("observer_bridge_required_products_complete") if observer_bridge_active else False,
            "required_observer_bridge_products_missing": observer_bridge_summary.get("required_observer_bridge_products_missing") if observer_bridge_active else [],
            "observer_bridge_partial_state_detected": observer_bridge_summary.get("observer_bridge_partial_state_detected") if observer_bridge_active else False,
            "observer_bridge_camera_overlay_generated": observer_bridge_summary.get("observer_bridge_camera_overlay_generated") if observer_bridge_active else False,
            "observer_bridge_kerr_interactive_view_generated": observer_bridge_summary.get("observer_bridge_kerr_interactive_view_generated") if observer_bridge_active else False,
            "downstream_candidate_selection_enabled": observer_bridge_summary.get("downstream_candidate_selection_enabled") if observer_bridge_active else False,
            "downstream_selection_policy": observer_bridge_summary.get("downstream_selection_policy") if observer_bridge_active else values.get("observer_bridge", {}).get("downstream_selection_policy"),
            "downstream_n_candidates_ranked": observer_bridge_summary.get("downstream_n_candidates_ranked") if observer_bridge_active else 0,
            "downstream_n_candidates_selected": observer_bridge_summary.get("downstream_n_candidates_selected") if observer_bridge_active else 0,
            "downstream_stage_target": observer_bridge_summary.get("downstream_stage_target") if observer_bridge_active else "powheg",
            "camera_overlay_background_source": observer_bridge_summary.get("camera_overlay_background_source") if observer_bridge_active else None,
            "camera_overlay_resolution_px": observer_bridge_summary.get("camera_overlay_resolution_px") if observer_bridge_active else None,
            "observer_bridge_overlay_background_audit_generated": observer_bridge_summary.get("observer_bridge_overlay_background_audit_generated") if observer_bridge_active else False,
            "observer_bridge_background_comparison_generated": observer_bridge_summary.get("observer_bridge_background_comparison_generated") if observer_bridge_active else False,
            "camera_preview_path": observer_bridge_summary.get("camera_preview_path") if observer_bridge_active else None,
            "camera_preview_exists": observer_bridge_summary.get("camera_preview_exists") if observer_bridge_active else False,
            "camera_preview_sha256": observer_bridge_summary.get("camera_preview_sha256") if observer_bridge_active else None,
            "overlay_background_source_path": observer_bridge_summary.get("overlay_background_source_path") if observer_bridge_active else None,
            "overlay_background_exists": observer_bridge_summary.get("overlay_background_exists") if observer_bridge_active else False,
            "overlay_background_sha256": observer_bridge_summary.get("overlay_background_sha256") if observer_bridge_active else None,
            "background_hash_match": observer_bridge_summary.get("background_hash_match") if observer_bridge_active else False,
            "background_dimensions": observer_bridge_summary.get("background_dimensions") if observer_bridge_active else None,
            "overlay_dimensions": observer_bridge_summary.get("overlay_dimensions") if observer_bridge_active else None,
            "background_transform_applied": observer_bridge_summary.get("background_transform_applied") if observer_bridge_active else None,
            "background_is_stale": observer_bridge_summary.get("background_is_stale") if observer_bridge_active else None,
            "candidate_overlay_projection_model": observer_bridge_summary.get("candidate_overlay_projection_model") if observer_bridge_active else "geometric_pinhole_proxy",
            "candidate_overlay_kerr_lensed": observer_bridge_summary.get("candidate_overlay_kerr_lensed") if observer_bridge_active else False,
            "candidate_overlay_not_ray_traced": observer_bridge_summary.get("candidate_overlay_not_ray_traced") if observer_bridge_active else True,
            "candidate_overlay_physics_risk": observer_bridge_summary.get("candidate_overlay_physics_risk") if observer_bridge_active else True,
            "candidate_overlay_alignment": observer_bridge_summary.get("candidate_overlay_alignment") if observer_bridge_active else "camera_preview_pixel_plane",
            "candidate_overlay_fallback_reason": observer_bridge_summary.get("candidate_overlay_fallback_reason") if observer_bridge_active else None,
            "kerr_pixel_match_coordinate_convention": observer_bridge_summary.get("kerr_pixel_match_coordinate_convention") if observer_bridge_active else "camera_preview_pixel_grid",
            "camera_preview_pixel_convention": observer_bridge_summary.get("camera_preview_pixel_convention") if observer_bridge_active else "ppm_top_left_rows",
            "overlay_image_coordinate_convention": observer_bridge_summary.get("overlay_image_coordinate_convention") if observer_bridge_active else "top_left_image",
            "overlay_image_coordinate_transform": observer_bridge_summary.get("overlay_image_coordinate_transform") if observer_bridge_active else "identity_x_y",
            "matching_ray_basis_transform": observer_bridge_summary.get("matching_ray_basis_transform") if observer_bridge_active else "cuda_preview_local_tetrad",
            "inclination_convention": observer_bridge_summary.get("inclination_convention") if observer_bridge_active else "theta_0_north_pi_over_2_equator",
            "camera_preview_observer_hemisphere": observer_bridge_summary.get("camera_preview_observer_hemisphere") if observer_bridge_active else None,
            "kerr_pixel_match_observer_hemisphere": observer_bridge_summary.get("kerr_pixel_match_observer_hemisphere") if observer_bridge_active else None,
            "hemisphere_consistent": observer_bridge_summary.get("hemisphere_consistent") if observer_bridge_active else False,
            "overlay_hemisphere_validated": observer_bridge_summary.get("overlay_hemisphere_validated") if observer_bridge_active else False,
            "observer_bridge_overlay_hemisphere_diagnostic_generated": observer_bridge_summary.get("observer_bridge_overlay_hemisphere_diagnostic_generated") if observer_bridge_active else False,
            "overlay_hemisphere_diagnostic_selected_panel": observer_bridge_summary.get("overlay_hemisphere_diagnostic_selected_panel") if observer_bridge_active else None,
            "kerr_pixel_match_basis_validated": observer_bridge_summary.get("kerr_pixel_match_basis_validated") if observer_bridge_active else False,
            "camera_preview_matching_basis_consistent": observer_bridge_summary.get("camera_preview_matching_basis_consistent") if observer_bridge_active else False,
            "camera_basis_diagnostic_generated": observer_bridge_summary.get("camera_basis_diagnostic_generated") if observer_bridge_active else False,
            "observer_bridge_orientation_diagnostics_enabled": observer_bridge_summary.get("observer_bridge_orientation_diagnostics_enabled") if observer_bridge_active else False,
            "overlay_orientation_diagnostic_generated": observer_bridge_summary.get("overlay_orientation_diagnostic_generated") if observer_bridge_active else False,
            "overlay_orientation_diagnostic_selected_panel": observer_bridge_summary.get("overlay_orientation_diagnostic_selected_panel") if observer_bridge_active else None,
            "observer_overlay_orientation_markers_generated": observer_bridge_summary.get("observer_overlay_orientation_markers_generated") if observer_bridge_active else False,
            "observer_overlay_orientation_full_diagnostic_generated": observer_bridge_summary.get("observer_overlay_orientation_full_diagnostic_generated") if observer_bridge_active else False,
            "orientation_marker_selected_hypothesis": observer_bridge_summary.get("orientation_marker_selected_hypothesis") if observer_bridge_active else None,
            "orientation_marker_selected_pixel_transform": observer_bridge_summary.get("orientation_marker_selected_pixel_transform") if observer_bridge_active else None,
            "orientation_marker_selected_basis_transform": observer_bridge_summary.get("orientation_marker_selected_basis_transform") if observer_bridge_active else None,
            "orientation_marker_mean_pixel_error": observer_bridge_summary.get("orientation_marker_mean_pixel_error") if observer_bridge_active else None,
            "candidate_overlay_pixel_y_convention": observer_bridge_summary.get("candidate_overlay_pixel_y_convention") if observer_bridge_active else "image_top_left",
            "candidate_overlay_y_axis_flipped_for_image": observer_bridge_summary.get("candidate_overlay_y_axis_flipped_for_image") if observer_bridge_active else False,
            "overlay_orientation_validated": observer_bridge_summary.get("overlay_orientation_validated") if observer_bridge_active else False,
            "camera_overlay_preview_status": observer_bridge_summary.get("camera_overlay_preview_status") if observer_bridge_active else None,
            "camera_overlay_candidates_plotted": observer_bridge_summary.get("camera_overlay_candidates_plotted") if observer_bridge_active else 0,
            "camera_overlay_candidates_inside_fov": observer_bridge_summary.get("camera_overlay_candidates_inside_fov") if observer_bridge_active else 0,
            "camera_overlay_candidates_unmatched": observer_bridge_summary.get("camera_overlay_candidates_unmatched") if observer_bridge_active else 0,
            "camera_overlay_top_n": observer_bridge_summary.get("camera_overlay_top_n") if observer_bridge_active else 0,
            "kerr_geodesic_backend": observer_bridge_summary.get("kerr_geodesic_backend") if observer_bridge_active else None,
            "kerr_pixel_match_resolution": observer_bridge_summary.get("kerr_pixel_match_resolution") if observer_bridge_active else None,
            "kerr_pixel_match_resolution_x": observer_bridge_summary.get("kerr_pixel_match_resolution_x") if observer_bridge_active else None,
            "kerr_pixel_match_resolution_y": observer_bridge_summary.get("kerr_pixel_match_resolution_y") if observer_bridge_active else None,
            "kerr_pixel_match_tolerance_rg": observer_bridge_summary.get("kerr_pixel_match_tolerance_rg") if observer_bridge_active else None,
            "kerr_pixel_match_refine_enabled": observer_bridge_summary.get("kerr_pixel_match_refine_enabled") if observer_bridge_active else None,
            "kerr_pixel_match_n_candidates": observer_bridge_summary.get("kerr_pixel_match_n_candidates") if observer_bridge_active else 0,
            "kerr_pixel_match_n_matched": observer_bridge_summary.get("kerr_pixel_match_n_matched") if observer_bridge_active else 0,
            "kerr_pixel_match_n_unmatched": observer_bridge_summary.get("kerr_pixel_match_n_unmatched") if observer_bridge_active else 0,
            "kerr_pixel_match_mean_closest_approach_rg": observer_bridge_summary.get("kerr_pixel_match_mean_closest_approach_rg") if observer_bridge_active else None,
            "kerr_pixel_match_max_closest_approach_rg": observer_bridge_summary.get("kerr_pixel_match_max_closest_approach_rg") if observer_bridge_active else None,
            "interactive_view_uses_kerr_ray_matching": observer_bridge_summary.get("interactive_view_uses_kerr_ray_matching") if observer_bridge_active else False,
            "interactive_view_not_final_observed_image": observer_bridge_summary.get("interactive_view_not_final_observed_image") if observer_bridge_active else True,
            "interactive_view_diagnostic_only": observer_bridge_summary.get("interactive_view_diagnostic_only") if observer_bridge_active else True,
            "interactive_max_candidates": observer_bridge_summary.get("interactive_max_candidates") if observer_bridge_active else None,
            "interactive_max_rays": observer_bridge_summary.get("interactive_max_rays") if observer_bridge_active else None,
            "interactive_ray_stride": observer_bridge_summary.get("interactive_ray_stride") if observer_bridge_active else None,
            "interactive_ray_source": observer_bridge_summary.get("interactive_ray_source") if observer_bridge_active else None,
            "interactive_candidate_color_mode": observer_bridge_summary.get("interactive_candidate_color_mode") if observer_bridge_active else None,
            "interactive_candidates_displayed": observer_bridge_summary.get("interactive_candidates_displayed") if observer_bridge_active else None,
            "interactive_rays_displayed": observer_bridge_summary.get("interactive_rays_displayed") if observer_bridge_active else None,
            "medium_renderer_used": observer_bridge_summary.get("medium_renderer_used") if observer_bridge_active else False,
            "medium_model": observer_bridge_summary.get("medium_model") if observer_bridge_active else "analytic_torus",
            "density_model": observer_bridge_summary.get("density_model") if observer_bridge_active else "analytic_torus_density_v1",
            "density_model_has_hard_radial_cut": observer_bridge_summary.get("density_model_has_hard_radial_cut") if observer_bridge_active else True,
            "density_model_theta_profile": observer_bridge_summary.get("density_model_theta_profile") if observer_bridge_active else "gaussian",
            "density_model_theta_is_hard_cut": observer_bridge_summary.get("density_model_theta_is_hard_cut") if observer_bridge_active else False,
            "half_opening_angle_interpretation": observer_bridge_summary.get("half_opening_angle_interpretation") if observer_bridge_active else "gaussian_width_not_boundary",
            "camera_view_candidates_plotted": observer_bridge_summary.get("camera_view_candidates_plotted") if observer_bridge_active else 0,
            "camera_view_candidates_inside_fov": observer_bridge_summary.get("camera_view_candidates_inside_fov") if observer_bridge_active else 0,
            "camera_view_top_n": observer_bridge_summary.get("camera_view_top_n") if observer_bridge_active else 0,
            "uses_hadros_original_runtime_path": observer_bridge_summary.get("uses_hadros_original_runtime_path") if observer_bridge_active else False,
            "proxy_physics_risk": observer_bridge_summary.get("proxy_physics_risk") if observer_bridge_active else True,
            "escape_proxy_physics_risk": observer_bridge_summary.get("escape_proxy_physics_risk") if observer_bridge_active else True,
            "visibility_proxy_physics_risk": observer_bridge_summary.get("visibility_proxy_physics_risk") if observer_bridge_active else True,
            "redshift_proxy_physics_risk": observer_bridge_summary.get("redshift_proxy_physics_risk") if observer_bridge_active else True,
            "event_generation_invoked": False,
            "powheg_invoked": False,
            "pythia_invoked": False,
            "geant4_invoked": False,
            "photon_transport_invoked": False,
            "summary": observer_bridge_summary,
        },
        "observer_image_branches": {
            **(observer_image_branch_summary or {}),
            "observer_image_branch_analysis_invoked": observer_image_branch_active,
            "branch_scoring_model": (observer_image_branch_summary or {}).get("branch_scoring_model"),
            "primary_branch_selection_model": (observer_image_branch_summary or {}).get("primary_branch_selection_model"),
            "mean_branches_per_candidate": (observer_image_branch_summary or {}).get("mean_branches_per_candidate"),
            "fraction_multiple_images": (observer_image_branch_summary or {}).get("fraction_multiple_images"),
            "primary_branch_selection_proxy": (observer_image_branch_summary or {}).get("primary_branch_selection_proxy", True),
            "powheg_invoked": powheg_active,
            "pythia_invoked": False,
            "geant4_invoked": False,
            "photon_transport_invoked": False,
            "summary": observer_image_branch_summary,
        },
        "powheg": {
            "powheg_dry_run_invoked": bool(powheg_summary.get("powheg_dry_run_invoked")) if powheg_active else False,
            "powheg_real_smoke_invoked": bool(powheg_summary.get("powheg_real_smoke_invoked")) if powheg_active else False,
            "powheg_real_free_invoked": bool(powheg_summary.get("powheg_real_free_invoked")) if powheg_active else False,
            "powheg_backend": powheg_summary.get("powheg_backend") if powheg_active else values.get("powheg", {}).get("powheg_backend"),
            "powheg_process": powheg_summary.get("powheg_process") if powheg_active else values.get("powheg", {}).get("powheg_process"),
            "powheg_run_mode": powheg_summary.get("powheg_run_mode") if powheg_active else values.get("powheg", {}).get("run_mode"),
            "powheg_candidate_source": powheg_summary.get("powheg_candidate_source") if powheg_active else "ObserverImageBranches/observer_image_primary_branches.jsonl",
            "powheg_n_selected_candidates_input": powheg_summary.get("powheg_n_selected_candidates_input") if powheg_active else 0,
            "powheg_selection_performed_by": powheg_summary.get("powheg_selection_performed_by") if powheg_active else "ObserverImageBranches",
            "powheg_selection_policy": powheg_summary.get("powheg_selection_policy") if powheg_active else values.get("observer_bridge", {}).get("downstream_selection_policy"),
            "events_per_candidate": powheg_summary.get("events_per_candidate") if powheg_active else values.get("powheg", {}).get("events_per_candidate"),
            "random_seed": powheg_summary.get("random_seed") if powheg_active else values.get("powheg", {}).get("random_seed"),
            "powheg_seed_mode": powheg_summary.get("powheg_seed_mode") if powheg_active else values.get("powheg", {}).get("powheg_seed_mode"),
            "powheg_jobs_prepared": powheg_summary.get("powheg_jobs_prepared") if powheg_active else 0,
            "powheg_cards_generated": powheg_summary.get("powheg_cards_generated") if powheg_active else 0,
            "powheg_lhe_generated": bool(powheg_summary.get("powheg_lhe_generated")) if powheg_active else False,
            "powheg_invoked": bool(powheg_summary.get("powheg_invoked")) if powheg_active else False,
            "pwhg_main_executed": bool(powheg_summary.get("pwhg_main_executed")) if powheg_active else False,
            "n_powheg_jobs": powheg_summary.get("n_powheg_jobs") if powheg_active else 0,
            "n_powheg_jobs_requested": powheg_summary.get("n_powheg_jobs_requested") if powheg_active else 0,
            "n_powheg_jobs_run": powheg_summary.get("n_powheg_jobs_run") if powheg_active else 0,
            "events_per_candidate_requested": powheg_summary.get("events_per_candidate_requested") if powheg_active else 0,
            "n_lhe_events": powheg_summary.get("n_lhe_events") if powheg_active else 0,
            "n_lhe_events_total": powheg_summary.get("n_lhe_events_total") if powheg_active else 0,
            "powheg_lhe_path": powheg_summary.get("powheg_lhe_path") if powheg_active else None,
            "powheg_log_path": powheg_summary.get("powheg_log_path") if powheg_active else None,
            "real_free_mode": bool(powheg_summary.get("real_free_mode")) if powheg_active else False,
            "real_smoke_safety_clamp": bool(powheg_summary.get("real_smoke_safety_clamp")) if powheg_active else False,
            "powheg_runtime_self_contained": powheg_summary.get("powheg_runtime_self_contained") if powheg_active else True,
            "backend_language": powheg_summary.get("backend_language") if powheg_active else "C++17",
            "backend_executable": powheg_summary.get("backend_executable") if powheg_active else "bin/hadros3_powheg_driver",
            "lhe_parser_invoked": bool(powheg_summary.get("lhe_parser_invoked")) if powheg_active else False,
            "lhe_particles_are_hard_process": bool(powheg_summary.get("lhe_particles_are_hard_process", True)) if powheg_active else True,
            "n_lhe_particles": powheg_summary.get("n_lhe_particles") if powheg_active else 0,
            "n_final_state_particles": powheg_summary.get("n_final_state_particles") if powheg_active else 0,
            "unique_particle_types": powheg_summary.get("unique_particle_types") if powheg_active else 0,
            "hadronization_invoked": False,
            "pythia_invoked": False,
            "geant4_invoked": False,
            "photon_transport_invoked": False,
            "expensive_event_generation_invoked": False,
            "summary": powheg_summary,
        },
        "validation": validation,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
