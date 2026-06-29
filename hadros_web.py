#!/usr/bin/env python3
"""HADROS3 web/configuration shell.

Use --serve for the H3-W0..H3-W4 web dashboard, or --render/--output-dir to
render the geometry/configuration products and exit.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from hadros3.camera_preview import available_backends, launch_interactive_camera_preview, render_camera_preview
from hadros3.config import deep_update, defaults, load_values, run_output_dir, safe_run_name, schema, validate_values
from hadros3.dis_sampler import generate_dis_interaction_products, generate_gbw_iim_comparison
from hadros3.forward_geodesics import generate_forward_geodesic_products
from hadros3.observer_bridge import generate_observer_bridge_products
from hadros3.observer_image_branches import generate_observer_image_branch_products
from hadros3.paths import camera_preview_dir, clear_dis_outputs, clear_forward_geodesics_outputs, clear_observer_bridge_outputs, clear_observer_image_branches_outputs, clear_powheg_outputs, dashboard_dir, dis_dir, ensure_output_layout, forward_geodesics_dir, geometry_dir, observer_bridge_dir, observer_image_branches_dir, powheg_dir, rel, run_metadata_dir, uhe_source_dir
from hadros3.pipeline import render_hadros_web
from hadros3.powheg import generate_powheg_products
from hadros3.reuse import discover_original_hadros
from hadros3.uhe_source import generate_uhe_source_products


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "presets" / "hadros_web" / "default_config.json"
COMMAND_TIMEOUT_SECONDS = 15 * 60


def load_release_metadata(root: Path = ROOT) -> dict[str, Any]:
    path = root / "VERSION.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def run_dashboard_command(args: list[str], *, root: Path = ROOT, timeout: int = COMMAND_TIMEOUT_SECONDS) -> dict[str, Any]:
    completed = subprocess.run(
        args,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "command": args,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "ok": completed.returncode == 0,
    }


def register_current_run(
    values: dict[str, dict[str, Any]],
    *,
    case_name: str = "",
    stage: str = "",
    description: str = "",
    root: Path = ROOT,
) -> dict[str, Any]:
    run_name = safe_run_name(values.get("run", {}).get("run_name", "HADROS3_run"))
    release = load_release_metadata(root)
    stage_value = (stage or release.get("pipeline_version") or "unclassified").strip()
    case_value = (case_name or run_name).strip()
    run_dir = root / run_output_dir(values)
    csv_path = root / "results" / "catalog" / "HADROS3_RESULTS_CATALOG.csv"
    json_path = root / "results" / "catalog" / "HADROS3_RESULTS_CATALOG.json"
    command = [
        sys.executable,
        "scripts/results/register_result.py",
        "--run-dir",
        str(run_dir),
        "--case-name",
        case_value,
        "--stage",
        stage_value,
        "--description",
        str(description or ""),
    ]
    result = run_dashboard_command(command, root=root)
    row: dict[str, Any] = {}
    if result["stdout"].strip():
        try:
            row = json.loads(result["stdout"])
        except json.JSONDecodeError:
            row = {}
    ok = result["ok"]
    summary = {
        "registered": ok,
        "catalog_csv_updated": csv_path.exists(),
        "catalog_json_updated": json_path.exists(),
        "run_id": row.get("run_id", run_name),
        "message": "Run registered in results catalog." if ok else "Run registration failed.",
        "case_name": case_value,
        "stage": stage_value,
    }
    return {
        "ok": ok,
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "summary": summary,
    }


def write_values(path: Path, values: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dashboard_payload(values: dict[str, dict[str, Any]], config_path: Path | None = None) -> dict[str, Any]:
    output_dir = ROOT / run_output_dir(values)
    camera_dir = camera_preview_dir(output_dir)
    geom_dir = geometry_dir(output_dir)
    metadata_dir = run_metadata_dir(output_dir)
    source_dir = uhe_source_dir(output_dir)
    forward_dir = forward_geodesics_dir(output_dir)
    dis_output_dir = dis_dir(output_dir)
    bridge_dir = observer_bridge_dir(output_dir)
    branch_dir = observer_image_branches_dir(output_dir)
    powheg_output_dir = powheg_dir(output_dir)
    web_dir = dashboard_dir(output_dir)

    camera_preview_path = camera_dir / "hadros3_camera_preview.png"
    camera_summary_path = camera_dir / "hadros3_camera_preview_summary.json"
    interactive_summary_path = camera_dir / "hadros3_camera_preview_interactive_summary.json"
    geometry_preview_path = geom_dir / "hadros3_geometry_preview.png"
    schematic_path = geom_dir / "hadros3_system_schematic.png"
    config_output_path = metadata_dir / "hadros3_config.json"
    provenance_path = metadata_dir / "hadros3_pipeline_provenance.json"
    render_summary_path = metadata_dir / "hadros_web_render_summary.json"
    source_samples_path = source_dir / "uhe_neutrino_source_samples.jsonl"
    source_csv_path = source_dir / "uhe_neutrino_source_summary.csv"
    source_summary_path = source_dir / "uhe_neutrino_source_summary.json"
    source_preview_path = source_dir / "uhe_neutrino_source_preview.png"
    source_uniformity_path = source_dir / "uhe_source_sampling_uniformity.png"
    source_uniformity_report_path = source_dir / "uhe_source_sampling_uniformity_report.json"
    source_direction_uniformity_path = source_dir / "uhe_source_direction_uniformity.png"
    source_direction_uniformity_report_path = source_dir / "uhe_source_direction_uniformity_report.json"
    source_direction_sphere_path = source_dir / "uhe_source_direction_sphere.png"
    forward_paths_path = forward_dir / "uhe_neutrino_forward_paths.jsonl"
    forward_segments_path = forward_dir / "uhe_neutrino_forward_path_segments.jsonl"
    forward_summary_csv_path = forward_dir / "uhe_neutrino_forward_summary.csv"
    forward_summary_path = forward_dir / "uhe_neutrino_forward_summary.json"
    forward_preview_path = forward_dir / "uhe_neutrino_forward_preview.png"
    forward_geometry_3d_path = forward_dir / "uhe_neutrino_forward_geometry_3d.png"
    forward_geometry_3d_json_path = forward_dir / "uhe_neutrino_forward_geometry_3d.json"
    forward_geometry_3d_html_path = forward_dir / "uhe_neutrino_forward_geometry_3d.html"
    strong_diagnostic_png_path = forward_dir / "isotropic_kerr_strong_field_diagnostic.png"
    strong_diagnostic_json_path = forward_dir / "isotropic_kerr_strong_field_diagnostic.json"
    geodesic_validation_path = forward_dir / "geodesic_validation_report.json"
    stop_statistics_path = forward_dir / "stop_condition_statistics.csv"
    diagnostic_report_path = forward_dir / "forward_geodesics_diagnostic_report.md"
    validation_invariants_path = forward_dir / "validation_invariants.png"
    bending_vs_impact_path = forward_dir / "kerr_bending_vs_impact_parameter.png"
    stop_distribution_path = forward_dir / "stop_condition_distribution.png"
    geodesic_density_map_path = forward_dir / "geodesic_density_map.png"
    forward_diagnostics_report_path = forward_dir / "forward_geodesics_diagnostics_report.json"
    dis_path_depths_path = dis_output_dir / "dis_path_optical_depths.jsonl"
    dis_candidates_path = dis_output_dir / "dis_interaction_candidates.jsonl"
    dis_accepted_path = dis_output_dir / "dis_accepted_interactions.jsonl"
    dis_summary_csv_path = dis_output_dir / "dis_summary.csv"
    dis_summary_path = dis_output_dir / "dis_summary.json"
    dis_tau_preview_path = dis_output_dir / "dis_tau_preview.png"
    dis_locations_path = dis_output_dir / "dis_interaction_locations.png"
    dis_locations_3d_html_path = dis_output_dir / "dis_interaction_locations_3d.html"
    dis_report_path = dis_output_dir / "dis_optical_depth_report.json"
    dis_tau_distribution_path = dis_output_dir / "tau_distribution.png"
    dis_probability_distribution_path = dis_output_dir / "interaction_probability_distribution.png"
    dis_optical_depth_map_path = dis_output_dir / "optical_depth_map.png"
    dis_optical_depth_map_3d_html_path = dis_output_dir / "optical_depth_map_3d.html"
    dis_medium_density_map_path = dis_output_dir / "medium_density_map.png"
    dis_interaction_location_distribution_path = dis_output_dir / "interaction_location_distribution.png"
    dis_local_energy_distribution_path = dis_output_dir / "local_energy_distribution.png"
    dis_local_density_distribution_path = dis_output_dir / "local_density_distribution.png"
    dis_sigma_distribution_path = dis_output_dir / "sigma_distribution.png"
    dis_density_energy_sigma_correlation_path = dis_output_dir / "density_energy_sigma_correlation.png"
    dis_diagnostics_report_path = dis_output_dir / "dis_diagnostics_report.json"
    dis_gbw_iim_tau_comparison_path = dis_output_dir / "gbw_vs_iim_tau_comparison.png"
    dis_gbw_iim_probability_comparison_path = dis_output_dir / "gbw_vs_iim_probability_comparison.png"
    dis_gbw_iim_locations_path = dis_output_dir / "gbw_vs_iim_interaction_locations.png"
    dis_gbw_iim_summary_path = dis_output_dir / "gbw_vs_iim_summary.json"
    bridge_candidates_path = bridge_dir / "observer_bridge_candidates.jsonl"
    bridge_ranked_path = bridge_dir / "observer_bridge_ranked_events.jsonl"
    bridge_selected_path = bridge_dir / "observer_bridge_selected_candidates.jsonl"
    bridge_selection_summary_path = bridge_dir / "observer_bridge_selection_summary.json"
    bridge_summary_json_path = bridge_dir / "observer_bridge_summary.json"
    bridge_summary_csv_path = bridge_dir / "observer_bridge_summary.csv"
    bridge_report_path = bridge_dir / "observer_bridge_report.json"
    bridge_map_path = bridge_dir / "observer_bridge_map.png"
    bridge_score_distribution_path = bridge_dir / "observer_bridge_score_distribution.png"
    bridge_weight_breakdown_path = bridge_dir / "observer_bridge_weight_breakdown.png"
    bridge_visibility_map_path = bridge_dir / "observer_bridge_visibility_map.png"
    bridge_ranked_png_path = bridge_dir / "observer_bridge_ranked_events.png"
    bridge_geometry_3d_html_path = bridge_dir / "observer_bridge_geometry_3d.html"
    bridge_camera_view_path = bridge_dir / "observer_bridge_camera_view.png"
    bridge_camera_overlay_path = bridge_dir / "observer_bridge_camera_overlay.png"
    bridge_overlay_background_audit_path = bridge_dir / "observer_bridge_overlay_background_audit.json"
    bridge_background_comparison_path = bridge_dir / "observer_bridge_background_comparison.png"
    bridge_hemisphere_diagnostic_path = bridge_dir / "observer_bridge_overlay_hemisphere_diagnostic.png"
    bridge_candidate_multi_image_audit_path = bridge_dir / "candidate_multi_image_audit.jsonl"
    bridge_candidate_multiple_images_path = bridge_dir / "observer_candidate_multiple_images.png"
    bridge_candidate_multi_image_view_path = bridge_dir / "observer_candidate_multi_image_view.html"
    bridge_multiple_image_statistics_path = bridge_dir / "multiple_image_statistics.json"
    bridge_orientation_markers_path = bridge_dir / "observer_overlay_orientation_markers.png"
    bridge_orientation_markers_json_path = bridge_dir / "observer_overlay_orientation_markers.json"
    bridge_orientation_full_diagnostic_path = bridge_dir / "observer_overlay_orientation_full_diagnostic.png"
    bridge_kerr_pixel_map_path = bridge_dir / "observer_candidate_kerr_pixel_map.jsonl"
    bridge_kerr_interactive_path = bridge_dir / "observer_bridge_kerr_interactive_view.html"
    branch_jsonl_path = branch_dir / "observer_image_branches.jsonl"
    branch_primary_path = branch_dir / "observer_image_primary_branches.jsonl"
    branch_summary_path = branch_dir / "observer_image_branch_summary.json"
    branch_report_path = branch_dir / "observer_image_branch_report.json"
    branch_statistics_json_path = branch_dir / "observer_image_statistics.json"
    branch_score_distribution_path = branch_dir / "observer_branch_score_distribution.png"
    branch_cluster_map_path = branch_dir / "observer_branch_cluster_map.png"
    branch_primary_vs_secondary_path = branch_dir / "observer_branch_primary_vs_secondary.png"
    branch_statistics_csv_path = branch_dir / "observer_branch_statistics.csv"
    branch_view_path = branch_dir / "observer_branch_view.html"
    branch_viewpoint_audit_path = branch_dir / "observer_viewpoint_convention_audit.json"
    branch_viewpoint_diagnostic_path = branch_dir / "observer_viewpoint_convention_diagnostic.png"
    powheg_requests_path = powheg_output_dir / "powheg_event_requests.jsonl"
    powheg_summary_json_path = powheg_output_dir / "powheg_summary.json"
    powheg_summary_csv_path = powheg_output_dir / "powheg_summary.csv"
    powheg_report_path = powheg_output_dir / "powheg_report.json"
    powheg_validation_report_path = powheg_output_dir / "powheg_validation_report.json"
    powheg_lhe_path = powheg_output_dir / "powheg_lhe" / "H3PWHG-000001" / "pwgevents.lhe"
    powheg_log_path = powheg_output_dir / "powheg_run_logs" / "H3PWHG-000001" / "powheg.log"
    powheg_card_preview_path = powheg_output_dir / "powheg_card_preview.png"
    powheg_energy_distribution_path = powheg_output_dir / "powheg_energy_distribution.png"
    powheg_job_summary_path = powheg_output_dir / "powheg_job_summary.png"
    powheg_lhe_particles_path = powheg_output_dir / "powheg_lhe_particles.jsonl"
    powheg_lhe_events_summary_path = powheg_output_dir / "powheg_lhe_events_summary.jsonl"
    powheg_lhe_particle_summary_csv_path = powheg_output_dir / "powheg_lhe_particle_summary.csv"
    powheg_lhe_particle_summary_json_path = powheg_output_dir / "powheg_lhe_particle_summary.json"
    powheg_lhe_particle_histogram_path = powheg_output_dir / "powheg_lhe_particle_histogram.png"
    powheg_lhe_energy_spectrum_path = powheg_output_dir / "powheg_lhe_energy_spectrum.png"
    powheg_lhe_momentum_spectrum_path = powheg_output_dir / "powheg_lhe_momentum_spectrum.png"
    powheg_hard_process_event_display_path = powheg_output_dir / "powheg_hard_process_event_display.png"
    powheg_hard_process_event_display_view_path = powheg_output_dir / "powheg_hard_process_event_display_view.html"
    powheg_event_summary_table_path = powheg_output_dir / "powheg_event_summary_table.csv"
    powheg_particle_table_path = powheg_output_dir / "powheg_particle_table.csv"
    powheg_particle_table_html_path = powheg_output_dir / "powheg_particle_table.html"
    powheg_particle_content_report_path = powheg_output_dir / "powheg_particle_content_report.json"
    powheg_lhe_event_view_path = powheg_output_dir / "powheg_lhe_event_view.html"
    html_path = web_dir / "index.html"

    camera_summary: dict[str, Any] | None = None
    source_summary: dict[str, Any] | None = None
    forward_summary: dict[str, Any] | None = None
    dis_summary: dict[str, Any] | None = None
    observer_bridge_summary: dict[str, Any] | None = None
    observer_image_branch_summary: dict[str, Any] | None = None
    powheg_summary: dict[str, Any] | None = None
    if camera_summary_path.exists():
        try:
            camera_summary = json.loads(camera_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            camera_summary = {"status": "invalid_summary", "message": "Could not parse camera preview summary."}
    if source_summary_path.exists():
        try:
            source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            source_summary = {"status": "invalid_summary", "message": "Could not parse UHE source summary."}
    if forward_summary_path.exists():
        try:
            forward_summary = json.loads(forward_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            forward_summary = {"status": "invalid_summary", "message": "Could not parse forward geodesic summary."}
    if dis_summary_path.exists():
        try:
            dis_summary = json.loads(dis_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            dis_summary = {"status": "invalid_summary", "message": "Could not parse DIS summary."}
    if bridge_summary_json_path.exists():
        try:
            observer_bridge_summary = json.loads(bridge_summary_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            observer_bridge_summary = {"status": "invalid_summary", "message": "Could not parse Observer Bridge summary."}
    if branch_summary_path.exists():
        try:
            observer_image_branch_summary = json.loads(branch_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            observer_image_branch_summary = {"status": "invalid_summary", "message": "Could not parse Observer Image Branch summary."}
    if powheg_summary_json_path.exists():
        try:
            powheg_summary = json.loads(powheg_summary_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            powheg_summary = {"status": "invalid_summary", "message": "Could not parse POWHEG summary."}
    bridge_required_products = {
        "observer_bridge_candidates.jsonl": bridge_candidates_path.exists(),
        "observer_bridge_ranked_events.jsonl": bridge_ranked_path.exists(),
        "observer_bridge_summary.json": bridge_summary_json_path.exists(),
        "observer_bridge_report.json": bridge_report_path.exists(),
        "observer_candidate_kerr_pixel_map.jsonl": bridge_kerr_pixel_map_path.exists(),
        "observer_bridge_camera_overlay.png": bridge_camera_overlay_path.exists(),
        "observer_bridge_kerr_interactive_view.html": bridge_kerr_interactive_path.exists(),
    }
    bridge_required_missing = [name for name, exists in bridge_required_products.items() if not exists]
    bridge_required_complete = not bridge_required_missing
    bridge_summary_complete = bool(
        observer_bridge_summary
        and observer_bridge_summary.get("status") == "ok"
        and observer_bridge_summary.get("observer_bridge_stage_complete", bridge_required_complete) is True
        and observer_bridge_summary.get("observer_bridge_required_products_complete", bridge_required_complete) is True
        and bridge_required_complete
    )
    bridge_partial_state = bool(bridge_summary_json_path.exists() and not bridge_summary_complete)
    if observer_bridge_summary and bridge_partial_state:
        observer_bridge_summary = dict(observer_bridge_summary)
        observer_bridge_summary.update(
            {
                "status": "incomplete",
                "observer_bridge_stage_complete": False,
                "observer_bridge_required_products_complete": False,
                "observer_bridge_partial_state_detected": True,
                "required_observer_bridge_products_missing": bridge_required_missing,
            }
        )
    return {
        "schema": schema(),
        "values": values,
        "config": str(config_path) if config_path is not None else None,
        "release": load_release_metadata(ROOT),
        "camera_backends": available_backends(),
        "camera_summary": camera_summary,
        "source_summary": source_summary,
        "forward_summary": forward_summary,
        "dis_summary": dis_summary,
        "observer_bridge_summary": observer_bridge_summary,
        "powheg_summary": powheg_summary,
        "source_status": {
            "configured_status": values.get("uhe_neutrino_source", {}).get("status"),
            "input_dir": rel(source_dir, output_dir),
            "samples_exists": source_samples_path.exists(),
            "preview_exists": source_preview_path.exists(),
            "summary_exists": source_summary_path.exists(),
            "ready_for_forward_geodesics": source_samples_path.exists(),
        },
        "forward_geodesics_status": {
            "configured_status": values.get("forward_geodesics", {}).get("status"),
            "input_uhe_source_dir": rel(source_dir, output_dir),
            "input_uhe_source_found": source_samples_path.exists(),
            "output_dir": rel(forward_dir, output_dir),
            "paths_exists": forward_paths_path.exists(),
            "path_segments_exists": forward_segments_path.exists(),
            "summary_csv_exists": forward_summary_csv_path.exists(),
            "summary_json_exists": forward_summary_path.exists(),
            "preview_exists": forward_preview_path.exists(),
            "geometry_3d_exists": forward_geometry_3d_path.exists(),
            "geometry_3d_json_exists": forward_geometry_3d_json_path.exists(),
            "geometry_3d_html_exists": forward_geometry_3d_html_path.exists(),
            "isotropic_kerr_strong_field_diagnostic_png_exists": strong_diagnostic_png_path.exists(),
            "isotropic_kerr_strong_field_diagnostic_json_exists": strong_diagnostic_json_path.exists(),
            "geodesic_validation_report_exists": geodesic_validation_path.exists(),
            "stop_condition_statistics_exists": stop_statistics_path.exists(),
            "forward_diagnostic_report_exists": diagnostic_report_path.exists(),
            "validation_invariants_exists": validation_invariants_path.exists(),
            "kerr_bending_vs_impact_parameter_exists": bending_vs_impact_path.exists(),
            "stop_condition_distribution_exists": stop_distribution_path.exists(),
            "geodesic_density_map_exists": geodesic_density_map_path.exists(),
            "forward_geodesics_diagnostics_report_exists": forward_diagnostics_report_path.exists(),
        },
        "dis_interaction_sampler": {
            "configured_status": values.get("dis_interaction_sampler", {}).get("status"),
            "input_uhe_source_found": source_samples_path.exists(),
            "input_forward_geodesics_found": forward_paths_path.exists() and forward_segments_path.exists(),
            "n_paths_available": forward_summary.get("n_paths", 0) if forward_summary else 0,
            "n_segments_available": forward_summary.get("n_segments", 0) if forward_summary else 0,
            "current_run": safe_run_name(values.get("run", {}).get("run_name", "HADROS3_run")),
            "output_dir_found": dis_output_dir.exists(),
            "products": {
                "dis_tau_preview": dis_tau_preview_path.exists(),
                "dis_interaction_locations": dis_locations_path.exists(),
                "dis_interaction_locations_3d_html": dis_locations_3d_html_path.exists(),
                "dis_summary_json": dis_summary_path.exists(),
                "dis_summary": dis_summary_csv_path.exists(),
                "dis_path_optical_depths": dis_path_depths_path.exists(),
                "dis_interaction_candidates": dis_candidates_path.exists(),
                "dis_accepted_interactions": dis_accepted_path.exists(),
                "dis_optical_depth_report": dis_report_path.exists(),
                "tau_distribution": dis_tau_distribution_path.exists(),
                "interaction_probability_distribution": dis_probability_distribution_path.exists(),
                "optical_depth_map": dis_optical_depth_map_path.exists(),
                "optical_depth_map_3d_html": dis_optical_depth_map_3d_html_path.exists(),
                "medium_density_map": dis_medium_density_map_path.exists(),
                "interaction_location_distribution": dis_interaction_location_distribution_path.exists(),
                "local_energy_distribution": dis_local_energy_distribution_path.exists(),
                "local_density_distribution": dis_local_density_distribution_path.exists(),
                "sigma_distribution": dis_sigma_distribution_path.exists(),
                "density_energy_sigma_correlation": dis_density_energy_sigma_correlation_path.exists(),
                "dis_diagnostics_report": dis_diagnostics_report_path.exists(),
                "gbw_vs_iim_tau_comparison": dis_gbw_iim_tau_comparison_path.exists(),
                "gbw_vs_iim_probability_comparison": dis_gbw_iim_probability_comparison_path.exists(),
                "gbw_vs_iim_interaction_locations": dis_gbw_iim_locations_path.exists(),
                "gbw_vs_iim_summary": dis_gbw_iim_summary_path.exists(),
            },
            "summary": dis_summary,
            "links": {
                "dis_tau_preview": rel(dis_tau_preview_path, output_dir),
                "dis_interaction_locations": rel(dis_locations_path, output_dir),
                "dis_interaction_locations_3d_html": rel(dis_locations_3d_html_path, output_dir),
                "dis_summary_json": rel(dis_summary_path, output_dir),
                "dis_summary": rel(dis_summary_csv_path, output_dir),
                "dis_path_optical_depths": rel(dis_path_depths_path, output_dir),
                "dis_interaction_candidates": rel(dis_candidates_path, output_dir),
                "dis_accepted_interactions": rel(dis_accepted_path, output_dir),
                "dis_optical_depth_report": rel(dis_report_path, output_dir),
                "tau_distribution": rel(dis_tau_distribution_path, output_dir),
                "interaction_probability_distribution": rel(dis_probability_distribution_path, output_dir),
                "optical_depth_map": rel(dis_optical_depth_map_path, output_dir),
                "optical_depth_map_3d_html": rel(dis_optical_depth_map_3d_html_path, output_dir),
                "medium_density_map": rel(dis_medium_density_map_path, output_dir),
                "interaction_location_distribution": rel(dis_interaction_location_distribution_path, output_dir),
                "local_energy_distribution": rel(dis_local_energy_distribution_path, output_dir),
                "local_density_distribution": rel(dis_local_density_distribution_path, output_dir),
                "sigma_distribution": rel(dis_sigma_distribution_path, output_dir),
                "density_energy_sigma_correlation": rel(dis_density_energy_sigma_correlation_path, output_dir),
                "dis_diagnostics_report": rel(dis_diagnostics_report_path, output_dir),
                "gbw_vs_iim_tau_comparison": rel(dis_gbw_iim_tau_comparison_path, output_dir),
                "gbw_vs_iim_probability_comparison": rel(dis_gbw_iim_probability_comparison_path, output_dir),
                "gbw_vs_iim_interaction_locations": rel(dis_gbw_iim_locations_path, output_dir),
                "gbw_vs_iim_summary": rel(dis_gbw_iim_summary_path, output_dir),
            },
        },
        "observer_bridge": {
            "configured_status": values.get("observer_bridge", {}).get("status"),
            "input_dis_found": dis_accepted_path.exists(),
            "output_dir": rel(bridge_dir, output_dir),
            "n_interactions_input": observer_bridge_summary.get("n_interactions_input", 0) if observer_bridge_summary else 0,
            "n_candidates_scored": observer_bridge_summary.get("n_candidates_scored", 0) if observer_bridge_summary else 0,
            "required_observer_bridge_products_complete": bridge_required_complete,
            "required_observer_bridge_products_missing": bridge_required_missing,
            "observer_bridge_stage_complete": bridge_summary_complete,
            "observer_bridge_partial_state_detected": bridge_partial_state,
            "products": {
                "observer_bridge_candidates": bridge_candidates_path.exists(),
                "observer_bridge_ranked_events": bridge_ranked_path.exists(),
                "observer_bridge_selected_candidates": bridge_selected_path.exists(),
                "observer_bridge_selection_summary": bridge_selection_summary_path.exists(),
                "observer_bridge_summary_json": bridge_summary_json_path.exists(),
                "observer_bridge_summary": bridge_summary_csv_path.exists(),
                "observer_bridge_report": bridge_report_path.exists(),
                "observer_bridge_map": bridge_map_path.exists(),
                "observer_bridge_score_distribution": bridge_score_distribution_path.exists(),
                "observer_bridge_weight_breakdown": bridge_weight_breakdown_path.exists(),
                "observer_bridge_visibility_map": bridge_visibility_map_path.exists(),
                "observer_bridge_ranked_events_png": bridge_ranked_png_path.exists(),
                "observer_bridge_geometry_3d_html": bridge_geometry_3d_html_path.exists(),
                "observer_bridge_camera_view": bridge_camera_view_path.exists(),
                "observer_bridge_camera_overlay": bridge_camera_overlay_path.exists(),
                "observer_bridge_overlay_background_audit": bridge_overlay_background_audit_path.exists(),
                "observer_bridge_background_comparison": bridge_background_comparison_path.exists(),
                "observer_bridge_overlay_hemisphere_diagnostic": bridge_hemisphere_diagnostic_path.exists(),
                "candidate_multi_image_audit": bridge_candidate_multi_image_audit_path.exists(),
                "observer_candidate_multiple_images": bridge_candidate_multiple_images_path.exists(),
                "observer_candidate_multi_image_view": bridge_candidate_multi_image_view_path.exists(),
                "multiple_image_statistics": bridge_multiple_image_statistics_path.exists(),
                "observer_overlay_orientation_markers": bridge_orientation_markers_path.exists(),
                "observer_overlay_orientation_markers_json": bridge_orientation_markers_json_path.exists(),
                "observer_overlay_orientation_full_diagnostic": bridge_orientation_full_diagnostic_path.exists(),
                "observer_candidate_kerr_pixel_map": bridge_kerr_pixel_map_path.exists(),
                "observer_bridge_kerr_interactive_view": bridge_kerr_interactive_path.exists(),
            },
            "summary": observer_bridge_summary,
            "links": {
                "observer_bridge_candidates": rel(bridge_candidates_path, output_dir),
                "observer_bridge_ranked_events": rel(bridge_ranked_path, output_dir),
                "observer_bridge_selected_candidates": rel(bridge_selected_path, output_dir),
                "observer_bridge_selection_summary": rel(bridge_selection_summary_path, output_dir),
                "observer_bridge_summary_json": rel(bridge_summary_json_path, output_dir),
                "observer_bridge_summary": rel(bridge_summary_csv_path, output_dir),
                "observer_bridge_report": rel(bridge_report_path, output_dir),
                "observer_bridge_map": rel(bridge_map_path, output_dir),
                "observer_bridge_score_distribution": rel(bridge_score_distribution_path, output_dir),
                "observer_bridge_weight_breakdown": rel(bridge_weight_breakdown_path, output_dir),
                "observer_bridge_visibility_map": rel(bridge_visibility_map_path, output_dir),
                "observer_bridge_ranked_events_png": rel(bridge_ranked_png_path, output_dir),
                "observer_bridge_geometry_3d_html": rel(bridge_geometry_3d_html_path, output_dir),
                "observer_bridge_camera_view": rel(bridge_camera_view_path, output_dir),
                "observer_bridge_camera_overlay": rel(bridge_camera_overlay_path, output_dir),
                "observer_bridge_overlay_background_audit": rel(bridge_overlay_background_audit_path, output_dir),
                "observer_bridge_background_comparison": rel(bridge_background_comparison_path, output_dir),
                "observer_bridge_overlay_hemisphere_diagnostic": rel(bridge_hemisphere_diagnostic_path, output_dir),
                "candidate_multi_image_audit": rel(bridge_candidate_multi_image_audit_path, output_dir),
                "observer_candidate_multiple_images": rel(bridge_candidate_multiple_images_path, output_dir),
                "observer_candidate_multi_image_view": rel(bridge_candidate_multi_image_view_path, output_dir),
                "multiple_image_statistics": rel(bridge_multiple_image_statistics_path, output_dir),
                "observer_overlay_orientation_markers": rel(bridge_orientation_markers_path, output_dir),
                "observer_overlay_orientation_markers_json": rel(bridge_orientation_markers_json_path, output_dir),
                "observer_overlay_orientation_full_diagnostic": rel(bridge_orientation_full_diagnostic_path, output_dir),
                "observer_candidate_kerr_pixel_map": rel(bridge_kerr_pixel_map_path, output_dir),
                "observer_bridge_kerr_interactive_view": rel(bridge_kerr_interactive_path, output_dir),
            },
        },
        "observer_image_branches": {
            "configured_status": values.get("observer_image_branches", {}).get("status"),
            "input_selected_candidates_found": bridge_selected_path.exists(),
            "input_kerr_pixel_map_found": bridge_kerr_pixel_map_path.exists(),
            "output_dir": rel(branch_dir, output_dir),
            "n_candidates": observer_image_branch_summary.get("n_candidates", 0) if observer_image_branch_summary else 0,
            "n_branches": observer_image_branch_summary.get("n_branches", 0) if observer_image_branch_summary else 0,
            "mean_branches_per_candidate": observer_image_branch_summary.get("mean_branches_per_candidate", 0) if observer_image_branch_summary else 0,
            "fraction_multiple_images": observer_image_branch_summary.get("fraction_multiple_images", 0) if observer_image_branch_summary else 0,
            "products": {
                "observer_image_branches": branch_jsonl_path.exists(),
                "observer_image_primary_branches": branch_primary_path.exists(),
                "observer_image_branch_summary": branch_summary_path.exists(),
                "observer_image_branch_report": branch_report_path.exists(),
                "observer_image_statistics": branch_statistics_json_path.exists(),
                "observer_branch_score_distribution": branch_score_distribution_path.exists(),
                "observer_branch_cluster_map": branch_cluster_map_path.exists(),
                "observer_branch_primary_vs_secondary": branch_primary_vs_secondary_path.exists(),
                "observer_branch_statistics": branch_statistics_csv_path.exists(),
                "observer_branch_view": branch_view_path.exists(),
                "observer_viewpoint_convention_audit": branch_viewpoint_audit_path.exists(),
                "observer_viewpoint_convention_diagnostic": branch_viewpoint_diagnostic_path.exists(),
            },
            "summary": observer_image_branch_summary,
            "links": {
                "observer_image_branches": rel(branch_jsonl_path, output_dir),
                "observer_image_primary_branches": rel(branch_primary_path, output_dir),
                "observer_image_branch_summary": rel(branch_summary_path, output_dir),
                "observer_image_branch_report": rel(branch_report_path, output_dir),
                "observer_image_statistics": rel(branch_statistics_json_path, output_dir),
                "observer_branch_score_distribution": rel(branch_score_distribution_path, output_dir),
                "observer_branch_cluster_map": rel(branch_cluster_map_path, output_dir),
                "observer_branch_primary_vs_secondary": rel(branch_primary_vs_secondary_path, output_dir),
                "observer_branch_statistics": rel(branch_statistics_csv_path, output_dir),
                "observer_branch_view": rel(branch_view_path, output_dir),
                "observer_viewpoint_convention_audit": rel(branch_viewpoint_audit_path, output_dir),
                "observer_viewpoint_convention_diagnostic": rel(branch_viewpoint_diagnostic_path, output_dir),
            },
        },
        "powheg": {
            "configured_status": values.get("powheg", {}).get("status"),
            "input_observer_image_branches_found": branch_primary_path.exists(),
            "output_dir": rel(powheg_output_dir, output_dir),
            "n_candidates_input": powheg_summary.get("n_candidates_input", 0) if powheg_summary else 0,
            "powheg_jobs_prepared": powheg_summary.get("powheg_jobs_prepared", 0) if powheg_summary else 0,
            "powheg_cards_generated": powheg_summary.get("powheg_cards_generated", 0) if powheg_summary else 0,
            "powheg_lhe_generated": powheg_summary.get("powheg_lhe_generated", False) if powheg_summary else False,
            "n_lhe_events": powheg_summary.get("n_lhe_events", 0) if powheg_summary else 0,
            "products": {
                "powheg_event_requests": powheg_requests_path.exists(),
                "powheg_summary_json": powheg_summary_json_path.exists(),
                "powheg_summary": powheg_summary_csv_path.exists(),
                "powheg_report": powheg_report_path.exists(),
                "powheg_validation_report": powheg_validation_report_path.exists(),
                "powheg_lhe": powheg_lhe_path.exists(),
                "powheg_log": powheg_log_path.exists(),
                "powheg_card_preview": powheg_card_preview_path.exists(),
                "powheg_energy_distribution": powheg_energy_distribution_path.exists(),
                "powheg_job_summary": powheg_job_summary_path.exists(),
                "powheg_lhe_particles": powheg_lhe_particles_path.exists(),
                "powheg_lhe_events_summary": powheg_lhe_events_summary_path.exists(),
                "powheg_lhe_particle_summary_csv": powheg_lhe_particle_summary_csv_path.exists(),
                "powheg_lhe_particle_summary_json": powheg_lhe_particle_summary_json_path.exists(),
                "powheg_lhe_particle_histogram": powheg_lhe_particle_histogram_path.exists(),
                "powheg_lhe_energy_spectrum": powheg_lhe_energy_spectrum_path.exists(),
                "powheg_lhe_momentum_spectrum": powheg_lhe_momentum_spectrum_path.exists(),
                "powheg_hard_process_event_display": powheg_hard_process_event_display_path.exists(),
                "powheg_hard_process_event_display_view": powheg_hard_process_event_display_view_path.exists(),
                "powheg_event_summary_table": powheg_event_summary_table_path.exists(),
                "powheg_particle_table": powheg_particle_table_path.exists(),
                "powheg_particle_table_html": powheg_particle_table_html_path.exists(),
                "powheg_particle_content_report": powheg_particle_content_report_path.exists(),
                "powheg_lhe_event_view": powheg_lhe_event_view_path.exists(),
            },
            "summary": powheg_summary,
            "links": {
                "powheg_event_requests": rel(powheg_requests_path, output_dir),
                "powheg_summary_json": rel(powheg_summary_json_path, output_dir),
                "powheg_summary": rel(powheg_summary_csv_path, output_dir),
                "powheg_report": rel(powheg_report_path, output_dir),
                "powheg_validation_report": rel(powheg_validation_report_path, output_dir),
                "powheg_lhe": rel(powheg_lhe_path, output_dir),
                "powheg_log": rel(powheg_log_path, output_dir),
                "powheg_card_preview": rel(powheg_card_preview_path, output_dir),
                "powheg_energy_distribution": rel(powheg_energy_distribution_path, output_dir),
                "powheg_job_summary": rel(powheg_job_summary_path, output_dir),
                "powheg_lhe_particles": rel(powheg_lhe_particles_path, output_dir),
                "powheg_lhe_events_summary": rel(powheg_lhe_events_summary_path, output_dir),
                "powheg_lhe_particle_summary_csv": rel(powheg_lhe_particle_summary_csv_path, output_dir),
                "powheg_lhe_particle_summary_json": rel(powheg_lhe_particle_summary_json_path, output_dir),
                "powheg_lhe_particle_histogram": rel(powheg_lhe_particle_histogram_path, output_dir),
                "powheg_lhe_energy_spectrum": rel(powheg_lhe_energy_spectrum_path, output_dir),
                "powheg_lhe_momentum_spectrum": rel(powheg_lhe_momentum_spectrum_path, output_dir),
                "powheg_hard_process_event_display": rel(powheg_hard_process_event_display_path, output_dir),
                "powheg_hard_process_event_display_view": rel(powheg_hard_process_event_display_view_path, output_dir),
                "powheg_event_summary_table": rel(powheg_event_summary_table_path, output_dir),
                "powheg_particle_table": rel(powheg_particle_table_path, output_dir),
                "powheg_particle_table_html": rel(powheg_particle_table_html_path, output_dir),
                "powheg_particle_content_report": rel(powheg_particle_content_report_path, output_dir),
                "powheg_lhe_event_view": rel(powheg_lhe_event_view_path, output_dir),
            },
        },
        "pipeline_status": [
            {"stage": "Geometry", "status": "done" if geometry_preview_path.exists() else "pending", "tab": "Camera"},
            {"stage": "Camera", "status": "done" if camera_preview_path.exists() else "pending", "tab": "Camera"},
            {"stage": "UHE Source", "status": "done" if source_samples_path.exists() else "pending", "tab": "UHE Source"},
            {"stage": "Forward Geodesics", "status": "done" if forward_paths_path.exists() and forward_segments_path.exists() else "pending", "tab": "Forward Geodesics"},
            {"stage": "DIS Interaction Sampler", "status": "done" if dis_summary_path.exists() else "pending", "tab": "DIS Interaction Sampler"},
            {"stage": "Observer Bridge", "status": "done" if bridge_summary_complete else "pending", "tab": "Observer Bridge"},
            {"stage": "Observer Image Branches", "status": "done" if branch_summary_path.exists() else "pending", "tab": "Observer Image Branches"},
            {"stage": "POWHEG", "status": "done" if powheg_summary_json_path.exists() else "pending", "tab": "POWHEG"},
            {"stage": "Event Generation", "status": "pending", "tab": "Event Generation"},
            {"stage": "GEANT4", "status": "pending", "tab": "GEANT4"},
            {"stage": "Photon Transport", "status": "pending", "tab": "Photon Transport"},
            {"stage": "Spectra", "status": "pending", "tab": "Spectra"},
        ],
        "outputs": {
            "output_dir": str(output_dir),
            "preview_exists": geometry_preview_path.exists(),
            "schematic_exists": schematic_path.exists(),
            "camera_preview_exists": camera_preview_path.exists(),
            "camera_preview_summary_exists": camera_summary_path.exists(),
            "interactive_camera_summary_exists": interactive_summary_path.exists(),
            "uhe_source_samples_exists": source_samples_path.exists(),
            "uhe_source_summary_exists": source_csv_path.exists(),
            "uhe_source_summary_json_exists": source_summary_path.exists(),
            "uhe_source_preview_exists": source_preview_path.exists(),
            "uhe_source_sampling_uniformity_exists": source_uniformity_path.exists(),
            "uhe_source_sampling_uniformity_report_exists": source_uniformity_report_path.exists(),
            "uhe_source_direction_uniformity_exists": source_direction_uniformity_path.exists(),
            "uhe_source_direction_uniformity_report_exists": source_direction_uniformity_report_path.exists(),
            "uhe_source_direction_sphere_exists": source_direction_sphere_path.exists(),
            "forward_paths_exists": forward_paths_path.exists(),
            "forward_path_segments_exists": forward_segments_path.exists(),
            "forward_summary_exists": forward_summary_csv_path.exists(),
            "forward_summary_json_exists": forward_summary_path.exists(),
            "forward_preview_exists": forward_preview_path.exists(),
            "forward_geometry_3d_exists": forward_geometry_3d_path.exists(),
            "forward_geometry_3d_json_exists": forward_geometry_3d_json_path.exists(),
            "forward_geometry_3d_html_exists": forward_geometry_3d_html_path.exists(),
            "isotropic_kerr_strong_field_diagnostic_png_exists": strong_diagnostic_png_path.exists(),
            "isotropic_kerr_strong_field_diagnostic_json_exists": strong_diagnostic_json_path.exists(),
            "geodesic_validation_report_exists": geodesic_validation_path.exists(),
            "stop_condition_statistics_exists": stop_statistics_path.exists(),
            "forward_diagnostic_report_exists": diagnostic_report_path.exists(),
            "validation_invariants_exists": validation_invariants_path.exists(),
            "kerr_bending_vs_impact_parameter_exists": bending_vs_impact_path.exists(),
            "stop_condition_distribution_exists": stop_distribution_path.exists(),
            "geodesic_density_map_exists": geodesic_density_map_path.exists(),
            "forward_geodesics_diagnostics_report_exists": forward_diagnostics_report_path.exists(),
            "dis_path_optical_depths_exists": dis_path_depths_path.exists(),
            "dis_interaction_candidates_exists": dis_candidates_path.exists(),
            "dis_accepted_interactions_exists": dis_accepted_path.exists(),
            "dis_summary_exists": dis_summary_csv_path.exists(),
            "dis_summary_json_exists": dis_summary_path.exists(),
            "dis_tau_preview_exists": dis_tau_preview_path.exists(),
            "dis_interaction_locations_exists": dis_locations_path.exists(),
            "dis_interaction_locations_3d_html_exists": dis_locations_3d_html_path.exists(),
            "dis_optical_depth_report_exists": dis_report_path.exists(),
            "tau_distribution_exists": dis_tau_distribution_path.exists(),
            "interaction_probability_distribution_exists": dis_probability_distribution_path.exists(),
            "optical_depth_map_exists": dis_optical_depth_map_path.exists(),
            "optical_depth_map_3d_html_exists": dis_optical_depth_map_3d_html_path.exists(),
            "medium_density_map_exists": dis_medium_density_map_path.exists(),
            "interaction_location_distribution_exists": dis_interaction_location_distribution_path.exists(),
            "local_energy_distribution_exists": dis_local_energy_distribution_path.exists(),
            "local_density_distribution_exists": dis_local_density_distribution_path.exists(),
            "sigma_distribution_exists": dis_sigma_distribution_path.exists(),
            "density_energy_sigma_correlation_exists": dis_density_energy_sigma_correlation_path.exists(),
            "dis_diagnostics_report_exists": dis_diagnostics_report_path.exists(),
            "gbw_vs_iim_tau_comparison_exists": dis_gbw_iim_tau_comparison_path.exists(),
            "gbw_vs_iim_probability_comparison_exists": dis_gbw_iim_probability_comparison_path.exists(),
            "gbw_vs_iim_interaction_locations_exists": dis_gbw_iim_locations_path.exists(),
            "gbw_vs_iim_summary_exists": dis_gbw_iim_summary_path.exists(),
            "observer_bridge_candidates_exists": bridge_candidates_path.exists(),
            "observer_bridge_ranked_events_exists": bridge_ranked_path.exists(),
            "observer_bridge_summary_json_exists": bridge_summary_json_path.exists(),
            "observer_bridge_summary_exists": bridge_summary_csv_path.exists(),
            "observer_bridge_report_exists": bridge_report_path.exists(),
            "observer_bridge_map_exists": bridge_map_path.exists(),
            "observer_bridge_score_distribution_exists": bridge_score_distribution_path.exists(),
            "observer_bridge_weight_breakdown_exists": bridge_weight_breakdown_path.exists(),
            "observer_bridge_visibility_map_exists": bridge_visibility_map_path.exists(),
            "observer_bridge_selected_candidates_exists": bridge_selected_path.exists(),
            "observer_bridge_selection_summary_exists": bridge_selection_summary_path.exists(),
            "observer_bridge_ranked_events_png_exists": bridge_ranked_png_path.exists(),
            "observer_bridge_geometry_3d_html_exists": bridge_geometry_3d_html_path.exists(),
            "observer_bridge_camera_view_exists": bridge_camera_view_path.exists(),
            "observer_bridge_camera_overlay_exists": bridge_camera_overlay_path.exists(),
            "observer_bridge_overlay_background_audit_exists": bridge_overlay_background_audit_path.exists(),
            "observer_bridge_background_comparison_exists": bridge_background_comparison_path.exists(),
            "observer_bridge_overlay_hemisphere_diagnostic_exists": bridge_hemisphere_diagnostic_path.exists(),
            "candidate_multi_image_audit_exists": bridge_candidate_multi_image_audit_path.exists(),
            "observer_candidate_multiple_images_exists": bridge_candidate_multiple_images_path.exists(),
            "observer_candidate_multi_image_view_exists": bridge_candidate_multi_image_view_path.exists(),
            "multiple_image_statistics_exists": bridge_multiple_image_statistics_path.exists(),
            "observer_overlay_orientation_markers_exists": bridge_orientation_markers_path.exists(),
            "observer_overlay_orientation_markers_json_exists": bridge_orientation_markers_json_path.exists(),
            "observer_overlay_orientation_full_diagnostic_exists": bridge_orientation_full_diagnostic_path.exists(),
            "observer_candidate_kerr_pixel_map_exists": bridge_kerr_pixel_map_path.exists(),
            "observer_bridge_kerr_interactive_view_exists": bridge_kerr_interactive_path.exists(),
            "observer_image_branches_exists": branch_jsonl_path.exists(),
            "observer_image_primary_branches_exists": branch_primary_path.exists(),
            "observer_image_branch_summary_exists": branch_summary_path.exists(),
            "observer_image_branch_report_exists": branch_report_path.exists(),
            "observer_image_statistics_exists": branch_statistics_json_path.exists(),
            "observer_branch_score_distribution_exists": branch_score_distribution_path.exists(),
            "observer_branch_cluster_map_exists": branch_cluster_map_path.exists(),
            "observer_branch_primary_vs_secondary_exists": branch_primary_vs_secondary_path.exists(),
            "observer_branch_statistics_exists": branch_statistics_csv_path.exists(),
            "observer_branch_view_exists": branch_view_path.exists(),
            "observer_viewpoint_convention_audit_exists": branch_viewpoint_audit_path.exists(),
            "observer_viewpoint_convention_diagnostic_exists": branch_viewpoint_diagnostic_path.exists(),
            "powheg_event_requests_exists": powheg_requests_path.exists(),
            "powheg_summary_json_exists": powheg_summary_json_path.exists(),
            "powheg_summary_exists": powheg_summary_csv_path.exists(),
            "powheg_report_exists": powheg_report_path.exists(),
            "powheg_validation_report_exists": powheg_validation_report_path.exists(),
            "powheg_lhe_exists": powheg_lhe_path.exists(),
            "powheg_log_exists": powheg_log_path.exists(),
            "powheg_card_preview_exists": powheg_card_preview_path.exists(),
            "powheg_energy_distribution_exists": powheg_energy_distribution_path.exists(),
            "powheg_job_summary_exists": powheg_job_summary_path.exists(),
            "powheg_lhe_particles_exists": powheg_lhe_particles_path.exists(),
            "powheg_lhe_events_summary_exists": powheg_lhe_events_summary_path.exists(),
            "powheg_lhe_particle_summary_csv_exists": powheg_lhe_particle_summary_csv_path.exists(),
            "powheg_lhe_particle_summary_json_exists": powheg_lhe_particle_summary_json_path.exists(),
            "powheg_lhe_particle_histogram_exists": powheg_lhe_particle_histogram_path.exists(),
            "powheg_lhe_energy_spectrum_exists": powheg_lhe_energy_spectrum_path.exists(),
            "powheg_lhe_momentum_spectrum_exists": powheg_lhe_momentum_spectrum_path.exists(),
            "powheg_hard_process_event_display_exists": powheg_hard_process_event_display_path.exists(),
            "powheg_hard_process_event_display_view_exists": powheg_hard_process_event_display_view_path.exists(),
            "powheg_event_summary_table_exists": powheg_event_summary_table_path.exists(),
            "powheg_particle_table_exists": powheg_particle_table_path.exists(),
            "powheg_particle_table_html_exists": powheg_particle_table_html_path.exists(),
            "powheg_particle_content_report_exists": powheg_particle_content_report_path.exists(),
            "powheg_lhe_event_view_exists": powheg_lhe_event_view_path.exists(),
            "provenance_exists": provenance_path.exists(),
            "config_exists": config_output_path.exists(),
            "render_summary_exists": render_summary_path.exists(),
            "html_summary_exists": html_path.exists(),
            "paths": {
                "config": rel(config_output_path, output_dir),
                "geometry_preview": rel(geometry_preview_path, output_dir),
                "system_schematic": rel(schematic_path, output_dir),
                "camera_preview": rel(camera_preview_path, output_dir),
                "camera_preview_summary": rel(camera_summary_path, output_dir),
                "interactive_camera_summary": rel(interactive_summary_path, output_dir),
                "uhe_source_samples": rel(source_samples_path, output_dir),
                "uhe_source_summary": rel(source_csv_path, output_dir),
                "uhe_source_summary_json": rel(source_summary_path, output_dir),
                "uhe_source_preview": rel(source_preview_path, output_dir),
                "uhe_source_sampling_uniformity": rel(source_uniformity_path, output_dir),
                "uhe_source_sampling_uniformity_report": rel(source_uniformity_report_path, output_dir),
                "uhe_source_direction_uniformity": rel(source_direction_uniformity_path, output_dir),
                "uhe_source_direction_uniformity_report": rel(source_direction_uniformity_report_path, output_dir),
                "uhe_source_direction_sphere": rel(source_direction_sphere_path, output_dir),
                "forward_paths": rel(forward_paths_path, output_dir),
                "forward_path_segments": rel(forward_segments_path, output_dir),
                "forward_summary": rel(forward_summary_csv_path, output_dir),
                "forward_summary_json": rel(forward_summary_path, output_dir),
                "forward_preview": rel(forward_preview_path, output_dir),
                "forward_geometry_3d": rel(forward_geometry_3d_path, output_dir),
                "forward_geometry_3d_json": rel(forward_geometry_3d_json_path, output_dir),
                "forward_geometry_3d_html": rel(forward_geometry_3d_html_path, output_dir),
                "isotropic_kerr_strong_field_diagnostic_png": rel(strong_diagnostic_png_path, output_dir),
                "isotropic_kerr_strong_field_diagnostic_json": rel(strong_diagnostic_json_path, output_dir),
                "geodesic_validation_report": rel(geodesic_validation_path, output_dir),
                "stop_condition_statistics": rel(stop_statistics_path, output_dir),
                "forward_diagnostic_report": rel(diagnostic_report_path, output_dir),
                "validation_invariants": rel(validation_invariants_path, output_dir),
                "kerr_bending_vs_impact_parameter": rel(bending_vs_impact_path, output_dir),
                "stop_condition_distribution": rel(stop_distribution_path, output_dir),
                "geodesic_density_map": rel(geodesic_density_map_path, output_dir),
                "forward_geodesics_diagnostics_report": rel(forward_diagnostics_report_path, output_dir),
                "dis_path_optical_depths": rel(dis_path_depths_path, output_dir),
                "dis_interaction_candidates": rel(dis_candidates_path, output_dir),
                "dis_accepted_interactions": rel(dis_accepted_path, output_dir),
                "dis_summary": rel(dis_summary_csv_path, output_dir),
                "dis_summary_json": rel(dis_summary_path, output_dir),
                "dis_tau_preview": rel(dis_tau_preview_path, output_dir),
                "dis_interaction_locations": rel(dis_locations_path, output_dir),
                "dis_interaction_locations_3d_html": rel(dis_locations_3d_html_path, output_dir),
                "dis_optical_depth_report": rel(dis_report_path, output_dir),
                "tau_distribution": rel(dis_tau_distribution_path, output_dir),
                "interaction_probability_distribution": rel(dis_probability_distribution_path, output_dir),
                "optical_depth_map": rel(dis_optical_depth_map_path, output_dir),
                "optical_depth_map_3d_html": rel(dis_optical_depth_map_3d_html_path, output_dir),
                "medium_density_map": rel(dis_medium_density_map_path, output_dir),
                "interaction_location_distribution": rel(dis_interaction_location_distribution_path, output_dir),
                "local_energy_distribution": rel(dis_local_energy_distribution_path, output_dir),
                "local_density_distribution": rel(dis_local_density_distribution_path, output_dir),
                "sigma_distribution": rel(dis_sigma_distribution_path, output_dir),
                "density_energy_sigma_correlation": rel(dis_density_energy_sigma_correlation_path, output_dir),
                "dis_diagnostics_report": rel(dis_diagnostics_report_path, output_dir),
                "gbw_vs_iim_tau_comparison": rel(dis_gbw_iim_tau_comparison_path, output_dir),
                "gbw_vs_iim_probability_comparison": rel(dis_gbw_iim_probability_comparison_path, output_dir),
                "gbw_vs_iim_interaction_locations": rel(dis_gbw_iim_locations_path, output_dir),
                "gbw_vs_iim_summary": rel(dis_gbw_iim_summary_path, output_dir),
                "observer_bridge_candidates": rel(bridge_candidates_path, output_dir),
                "observer_bridge_ranked_events": rel(bridge_ranked_path, output_dir),
                "observer_bridge_selected_candidates": rel(bridge_selected_path, output_dir),
                "observer_bridge_selection_summary": rel(bridge_selection_summary_path, output_dir),
                "observer_bridge_summary_json": rel(bridge_summary_json_path, output_dir),
                "observer_bridge_summary": rel(bridge_summary_csv_path, output_dir),
                "observer_bridge_report": rel(bridge_report_path, output_dir),
                "observer_bridge_map": rel(bridge_map_path, output_dir),
                "observer_bridge_score_distribution": rel(bridge_score_distribution_path, output_dir),
                "observer_bridge_weight_breakdown": rel(bridge_weight_breakdown_path, output_dir),
                "observer_bridge_visibility_map": rel(bridge_visibility_map_path, output_dir),
                "observer_bridge_ranked_events_png": rel(bridge_ranked_png_path, output_dir),
                "observer_bridge_geometry_3d_html": rel(bridge_geometry_3d_html_path, output_dir),
                "observer_bridge_camera_view": rel(bridge_camera_view_path, output_dir),
                "observer_bridge_camera_overlay": rel(bridge_camera_overlay_path, output_dir),
                "observer_bridge_overlay_background_audit": rel(bridge_overlay_background_audit_path, output_dir),
                "observer_bridge_background_comparison": rel(bridge_background_comparison_path, output_dir),
                "observer_bridge_overlay_hemisphere_diagnostic": rel(bridge_hemisphere_diagnostic_path, output_dir),
                "candidate_multi_image_audit": rel(bridge_candidate_multi_image_audit_path, output_dir),
                "observer_candidate_multiple_images": rel(bridge_candidate_multiple_images_path, output_dir),
                "observer_candidate_multi_image_view": rel(bridge_candidate_multi_image_view_path, output_dir),
                "multiple_image_statistics": rel(bridge_multiple_image_statistics_path, output_dir),
                "observer_overlay_orientation_markers": rel(bridge_orientation_markers_path, output_dir),
                "observer_overlay_orientation_markers_json": rel(bridge_orientation_markers_json_path, output_dir),
                "observer_overlay_orientation_full_diagnostic": rel(bridge_orientation_full_diagnostic_path, output_dir),
                "observer_candidate_kerr_pixel_map": rel(bridge_kerr_pixel_map_path, output_dir),
                "observer_bridge_kerr_interactive_view": rel(bridge_kerr_interactive_path, output_dir),
                "observer_image_branches": rel(branch_jsonl_path, output_dir),
                "observer_image_primary_branches": rel(branch_primary_path, output_dir),
                "observer_image_branch_summary": rel(branch_summary_path, output_dir),
                "observer_image_branch_report": rel(branch_report_path, output_dir),
                "observer_image_statistics": rel(branch_statistics_json_path, output_dir),
                "observer_branch_score_distribution": rel(branch_score_distribution_path, output_dir),
                "observer_branch_cluster_map": rel(branch_cluster_map_path, output_dir),
                "observer_branch_primary_vs_secondary": rel(branch_primary_vs_secondary_path, output_dir),
                "observer_branch_statistics": rel(branch_statistics_csv_path, output_dir),
                "observer_branch_view": rel(branch_view_path, output_dir),
                "observer_viewpoint_convention_audit": rel(branch_viewpoint_audit_path, output_dir),
                "observer_viewpoint_convention_diagnostic": rel(branch_viewpoint_diagnostic_path, output_dir),
                "powheg_event_requests": rel(powheg_requests_path, output_dir),
                "powheg_summary_json": rel(powheg_summary_json_path, output_dir),
                "powheg_summary": rel(powheg_summary_csv_path, output_dir),
                "powheg_report": rel(powheg_report_path, output_dir),
                "powheg_validation_report": rel(powheg_validation_report_path, output_dir),
                "powheg_lhe": rel(powheg_lhe_path, output_dir),
                "powheg_log": rel(powheg_log_path, output_dir),
                "powheg_card_preview": rel(powheg_card_preview_path, output_dir),
                "powheg_energy_distribution": rel(powheg_energy_distribution_path, output_dir),
                "powheg_job_summary": rel(powheg_job_summary_path, output_dir),
                "powheg_lhe_particles": rel(powheg_lhe_particles_path, output_dir),
                "powheg_lhe_events_summary": rel(powheg_lhe_events_summary_path, output_dir),
                "powheg_lhe_particle_summary_csv": rel(powheg_lhe_particle_summary_csv_path, output_dir),
                "powheg_lhe_particle_summary_json": rel(powheg_lhe_particle_summary_json_path, output_dir),
                "powheg_lhe_particle_histogram": rel(powheg_lhe_particle_histogram_path, output_dir),
                "powheg_lhe_energy_spectrum": rel(powheg_lhe_energy_spectrum_path, output_dir),
                "powheg_lhe_momentum_spectrum": rel(powheg_lhe_momentum_spectrum_path, output_dir),
                "powheg_hard_process_event_display": rel(powheg_hard_process_event_display_path, output_dir),
                "powheg_hard_process_event_display_view": rel(powheg_hard_process_event_display_view_path, output_dir),
                "powheg_event_summary_table": rel(powheg_event_summary_table_path, output_dir),
                "powheg_particle_table": rel(powheg_particle_table_path, output_dir),
                "powheg_particle_table_html": rel(powheg_particle_table_html_path, output_dir),
                "powheg_particle_content_report": rel(powheg_particle_content_report_path, output_dir),
                "powheg_lhe_event_view": rel(powheg_lhe_event_view_path, output_dir),
                "provenance": rel(provenance_path, output_dir),
                "render_summary": rel(render_summary_path, output_dir),
                "html_summary": rel(html_path, output_dir),
            },
        },
    }


def render_html(values: dict[str, dict[str, Any]], config_path: Path) -> str:
    payload = json.dumps(dashboard_payload(values, config_path))
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>HADROS3 hadros-web</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #eef2f6; color: #18202a; }}
    header {{ padding: 18px 24px 16px; background: #000; color: white; display: grid; place-items: center; text-align: center; }}
    .brand {{ display: grid; justify-items: center; gap: 8px; }}
    .brand-logo {{ width: min(1040px, 82vw); height: auto; max-height: 320px; object-fit: contain; display: block; }}
    main {{ max-width: 1680px; margin: 0 auto; padding: 18px; display: grid; grid-template-columns: 240px minmax(420px, 560px) minmax(560px, 1fr); gap: 18px; align-items: start; }}
    nav {{ background: white; border: 1px solid #d6dce5; border-radius: 6px; padding: 10px; position: sticky; top: 14px; }}
    .tab-button {{ display: block; width: 100%; text-align: left; margin: 0 0 6px; border-color: #d6dce5; background: #f8fafc; color: #18202a; }}
    .tab-button.active {{ background: #18202a; border-color: #18202a; color: white; }}
    section {{ border-top: 1px solid #d6dce5; padding: 14px 0; }}
    section h2 {{ font-size: 16px; margin: 0 0 10px; }}
    label {{ display: grid; grid-template-columns: 210px 1fr; gap: 10px; align-items: center; margin: 7px 0; font-size: 14px; }}
    input, select {{ padding: 7px 8px; border: 1px solid #9aa7b6; border-radius: 4px; background: white; min-width: 0; }}
    button {{ margin-right: 8px; padding: 8px 12px; border: 1px solid #18202a; background: #18202a; color: white; border-radius: 4px; }}
    button:disabled {{ opacity: 0.75; cursor: progress; }}
    pre {{ white-space: pre-wrap; background: #101318; color: #f0f4f8; padding: 12px; min-height: 120px; border-radius: 6px; overflow: auto; }}
    .panel {{ background: white; border: 1px solid #d6dce5; border-radius: 6px; padding: 16px; }}
    .panel.observer-image-branches-panel {{ grid-column: 2 / -1; }}
    .run-strip {{ grid-column: 1 / -1; background: white; border: 1px solid #d6dce5; border-radius: 6px; padding: 12px 16px; display: grid; gap: 12px; }}
    .run-main-row {{ display: grid; grid-template-columns: 140px minmax(240px, 420px) 110px 1fr; gap: 10px; align-items: center; }}
    .workflow-actions {{ display: grid; grid-template-columns: repeat(3, minmax(140px, 1fr)); gap: 10px; align-items: end; }}
    .workflow-actions label {{ display: grid; grid-template-columns: 1fr; gap: 5px; margin: 0; }}
    .workflow-buttons {{ grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .workflow-buttons button {{ margin-right: 0; }}
    .workflow-actions-log {{ grid-column: 1 / -1; border: 1px solid #d6dce5; border-radius: 6px; background: #f8fafc; padding: 10px; }}
    .workflow-actions-log summary {{ cursor: pointer; font-weight: 700; }}
    .warning-card {{ border: 1px solid #f59e0b; background: #fffbeb; color: #7c2d12; border-radius: 6px; padding: 12px; }}
    .workflow-actions-log pre {{ min-height: 72px; margin-bottom: 0; }}
    .run-strip label {{ display: contents; }}
    .run-strip input {{ width: 100%; box-sizing: border-box; }}
    .output-folder {{ font-family: ui-monospace, monospace; font-size: 13px; color: #4d5b6b; overflow-wrap: anywhere; }}
    .note {{ color: #4d5b6b; margin-top: 0; }}
    .actions {{ position: sticky; bottom: 0; background: white; border-top: 1px solid #d6dce5; padding-top: 12px; }}
    .geometry-preview-large {{ border: 1px solid #d6dce5; border-radius: 6px; background: #101318; overflow: hidden; min-height: 640px; display: grid; place-items: stretch; }}
    .geometry-preview-large svg {{ width: 100%; height: 100%; min-height: 640px; display: block; background: #101318; }}
    .geometry-preview-empty {{ padding: 28px; color: #cbd5e1; text-align: center; }}
    .context-figure {{ border: 1px solid #d6dce5; border-radius: 6px; background: #101318; min-height: 640px; display: grid; place-items: center; overflow: hidden; }}
    .context-figure img {{ width: 100%; height: 100%; min-height: 640px; object-fit: contain; display: block; background: #101318; }}
    .context-interactive {{ width: 100%; height: min(78vh, 860px); min-height: 640px; border: 0; display: block; background: #f7f8fb; }}
    .context-empty {{ padding: 28px; color: #cbd5e1; text-align: center; }}
    .diagnostic-card-grid {{ display: grid; gap: 12px; }}
    .diagnostic-plot-card {{ margin: 0; border: 1px solid #d6dce5; border-radius: 6px; background: #f8fafc; padding: 10px; }}
    .diagnostic-plot-card img {{ width: 100%; display: block; border: 1px solid #d6dce5; border-radius: 5px; background: white; }}
    .diagnostic-plot-card figcaption {{ margin: 0 0 8px; font-weight: 650; overflow-wrap: anywhere; }}
    .observer-image-branches-layout {{ display: grid; grid-template-columns: minmax(300px, 380px) minmax(0, 1fr); gap: 14px; align-items: start; }}
    .observer-image-branches-main, .observer-image-branches-diagnostics {{ display: grid; gap: 12px; }}
    .observer-image-branches-diagnostics section {{ margin: 0; }}
    .observer-image-branches-diagnostics .diagnostic-grid {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 12px; }}
    .observer-image-branches-diagnostics .context-interactive {{ min-height: 520px; height: min(70vh, 720px); }}
    .ok {{ color: #1f6f46; font-weight: 650; }}
    .pending {{ color: #8a5a0a; font-weight: 650; }}
    .active-panel h2 {{ margin-top: 0; }}
    .backend-table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }}
    .backend-table td, .backend-table th {{ border-top: 1px solid #d6dce5; padding: 6px; text-align: left; vertical-align: top; }}
    .camera-preview-panel {{ border: 1px solid #c8d3df; border-radius: 6px; background: #f8fafc; padding: 12px; margin-top: 14px; }}
    .source-action {{ background: #7c2d12; border-color: #7c2d12; font-weight: 700; }}
    .source-panel {{ border: 1px solid #c8d3df; border-radius: 6px; background: #f8fafc; padding: 12px; margin-top: 14px; }}
    .source-panel img, .output-link-grid img {{ width: 100%; border: 1px solid #d6dce5; border-radius: 5px; background: #101318; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 10px 0; }}
    .summary-item {{ border: 1px solid #d6dce5; border-radius: 5px; padding: 8px; background: white; }}
    .summary-item strong {{ display: block; font-size: 12px; color: #627084; }}
    .output-link-grid {{ display: grid; gap: 10px; }}
    .output-link-grid a {{ display: block; border: 1px solid #d6dce5; border-radius: 5px; padding: 8px; background: #f8fafc; overflow-wrap: anywhere; }}
    .camera-preview-top {{ display: flex; gap: 10px; align-items: center; margin-bottom: 12px; }}
    .camera-preview-button {{ font-weight: 700; background: #0f766e; border-color: #0f766e; }}
    .camera-preview-button.blinking {{ animation: pulse 0.7s ease-in-out 4; }}
    .camera-preview-row {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .camera-preview-field label {{ display: block; margin: 0 0 5px; font-size: 13px; font-weight: 650; }}
    .camera-preview-field select, .camera-preview-field input {{ width: 100%; box-sizing: border-box; }}
    .toggle-row {{ display: flex; grid-template-columns: none; gap: 8px; align-items: flex-start; margin: 0; padding: 8px; border: 1px solid #d6dce5; border-radius: 5px; background: white; }}
    .toggle-main {{ display: grid; gap: 3px; }}
    .toggle-name {{ font-weight: 650; }}
    .toggle-help, .field-help {{ color: #627084; font-size: 12px; }}
    .field-label {{ display: grid; gap: 3px; }}
    .camera-controls-card {{ margin-top: 12px; border-top: 1px solid #d6dce5; padding-top: 12px; }}
    .camera-controls-card h3 {{ margin: 0 0 8px; font-size: 14px; }}
    .camera-controls-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }}
    .camera-control-item {{ display: flex; gap: 8px; align-items: center; font-size: 12px; color: #4d5b6b; }}
    kbd {{ min-width: 70px; text-align: center; border: 1px solid #b8c3d1; border-bottom-width: 2px; border-radius: 4px; padding: 2px 5px; background: white; color: #18202a; font-family: ui-monospace, monospace; }}
    @keyframes pulse {{ 0% {{ transform: scale(1); }} 50% {{ transform: scale(1.04); box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.18); }} 100% {{ transform: scale(1); }} }}
    @media (max-width: 1080px) {{ main {{ grid-template-columns: 1fr; }} nav {{ position: static; }} .panel.observer-image-branches-panel {{ grid-column: auto; }} .observer-image-branches-layout {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<header><div class="brand"><img class="brand-logo" src="/assets/logo/Hadros_logo.png" alt="HADROS logo"></div></header>
<main id="app"></main>
<script>
const state = {payload};
const finalPreviewResolutionsNormal = ["256x144", "512x288", "1024x576", "1920x1080"];
const interactivePreviewResolutionsNormal = ["64x36", "96x54", "128x72", "256x144", "512x288", "1024x576", "1920x1080"];
const previewSkyOptions = [
  ["texture", "ESO Milky Way texture"],
  ["interstellar_coordinate_grid", "Kip Thorne coordinate grid"],
  ["procedural", "Procedural grid"],
];
const previewGeodesicModelOptions = [
  ["kerr_like", "Kerr-like CUDA"],
  ["full_kerr", "Full Kerr CUDA"],
];
const previewNavModeNormal = [
  ["celestial_plus_torus_volume", "físico"],
  ["paint_swatch_disk", "paint_swatch_disk = diagnostic visual test, not physical torus emission"],
];
let previewResolution = state.values.observer_camera.preview_final_resolution || "512x288";
let previewInteractiveResolution = state.values.observer_camera.preview_resolution || "256x144";
let previewSkyMode = "texture";
let previewGeodesicModel = state.values.observer_camera.camera_preview_mode === "full_kerr" ? "full_kerr" : "kerr_like";
let previewNavMode = state.values.observer_camera.preview_nav_mode || "celestial_plus_torus_volume";
let previewCelestialRadiusRs = "40";
let previewPhysicalTorus = true;
let previewOpaqueStructures = false;
let lastCameraMtime = 0;
let previewPollTimer = null;
let sourcePreviewVersion = Date.now();
let forwardPreviewVersion = Date.now();
let disPreviewVersion = Date.now();
let observerBridgePreviewVersion = Date.now();
let powhegPreviewVersion = Date.now();
function outPath(key) {{
  return state.outputs.paths && state.outputs.paths[key] ? state.outputs.paths[key] : key;
}}
function outUrl(key) {{
  return "/output/" + outPath(key);
}}
function setObserverBridgeOutputs(exists) {{
  state.observer_bridge_summary = exists ? state.observer_bridge_summary : null;
  state.outputs.observer_bridge_candidates_exists = exists;
  state.outputs.observer_bridge_ranked_events_exists = exists;
  state.outputs.observer_bridge_selected_candidates_exists = exists;
  state.outputs.observer_bridge_selection_summary_exists = exists;
  state.outputs.observer_bridge_summary_json_exists = exists;
  state.outputs.observer_bridge_summary_exists = exists;
  state.outputs.observer_bridge_report_exists = exists;
  state.outputs.observer_bridge_map_exists = exists;
  state.outputs.observer_bridge_score_distribution_exists = exists;
  state.outputs.observer_bridge_weight_breakdown_exists = exists;
  state.outputs.observer_bridge_visibility_map_exists = exists;
  state.outputs.observer_bridge_ranked_events_png_exists = exists;
  state.outputs.observer_bridge_geometry_3d_html_exists = exists;
  state.outputs.observer_bridge_camera_view_exists = exists;
  state.outputs.observer_bridge_camera_overlay_exists = exists;
  state.outputs.observer_bridge_overlay_background_audit_exists = exists;
  state.outputs.observer_bridge_background_comparison_exists = exists;
  state.outputs.observer_bridge_overlay_hemisphere_diagnostic_exists = exists;
  state.outputs.candidate_multi_image_audit_exists = exists;
  state.outputs.observer_candidate_multiple_images_exists = exists;
  state.outputs.observer_candidate_multi_image_view_exists = exists;
  state.outputs.multiple_image_statistics_exists = exists;
  state.outputs.observer_candidate_kerr_pixel_map_exists = exists;
  state.outputs.observer_bridge_kerr_interactive_view_exists = exists;
}}
function setObserverImageBranchOutputs(exists) {{
  if (!exists) state.observer_image_branches.summary = null;
  state.outputs.observer_image_branches_exists = exists;
  state.outputs.observer_image_primary_branches_exists = exists;
  state.outputs.observer_image_branch_summary_exists = exists;
  state.outputs.observer_image_branch_report_exists = exists;
  state.outputs.observer_image_statistics_exists = exists;
  state.outputs.observer_branch_score_distribution_exists = exists;
  state.outputs.observer_branch_cluster_map_exists = exists;
  state.outputs.observer_branch_primary_vs_secondary_exists = exists;
  state.outputs.observer_branch_statistics_exists = exists;
  state.outputs.observer_branch_view_exists = exists;
  state.outputs.observer_viewpoint_convention_audit_exists = exists;
  state.outputs.observer_viewpoint_convention_diagnostic_exists = exists;
}}
function setPowhegOutputs(exists) {{
  state.powheg_summary = exists ? state.powheg_summary : null;
  state.outputs.powheg_event_requests_exists = exists;
  state.outputs.powheg_summary_json_exists = exists;
  state.outputs.powheg_summary_exists = exists;
  state.outputs.powheg_report_exists = exists;
  state.outputs.powheg_validation_report_exists = exists;
  state.outputs.powheg_lhe_exists = exists;
  state.outputs.powheg_log_exists = exists;
  state.outputs.powheg_card_preview_exists = exists;
  state.outputs.powheg_energy_distribution_exists = exists;
  state.outputs.powheg_job_summary_exists = exists;
  state.outputs.powheg_lhe_particles_exists = exists;
  state.outputs.powheg_lhe_events_summary_exists = exists;
  state.outputs.powheg_lhe_particle_summary_csv_exists = exists;
  state.outputs.powheg_lhe_particle_summary_json_exists = exists;
  state.outputs.powheg_lhe_particle_histogram_exists = exists;
  state.outputs.powheg_lhe_energy_spectrum_exists = exists;
  state.outputs.powheg_lhe_momentum_spectrum_exists = exists;
  state.outputs.powheg_hard_process_event_display_exists = exists;
  state.outputs.powheg_hard_process_event_display_view_exists = exists;
  state.outputs.powheg_event_summary_table_exists = exists;
  state.outputs.powheg_particle_table_exists = exists;
  state.outputs.powheg_particle_table_html_exists = exists;
  state.outputs.powheg_particle_content_report_exists = exists;
  state.outputs.powheg_lhe_event_view_exists = exists;
}}
function inputFor(field, value) {{
  const attrs = [
    `data-section="${{field.section}}"`,
    `data-key="${{field.key}}"`,
    field.min !== undefined ? `min="${{field.min}}"` : "",
    field.step !== undefined ? `step="${{field.step}}"` : "",
  ].filter(Boolean).join(" ");
  if (field.kind === "select") {{
    return `<select ${{attrs}}>` +
      field.options.map(o => `<option value="${{o}}" ${{String(value) === String(o) ? "selected" : ""}}>${{o}}</option>`).join("") +
      `</select>`;
  }}
  if (field.kind === "checkbox") {{
    return `<input type="checkbox" ${{attrs}} ${{value ? "checked" : ""}}>`;
  }}
  return `<input type="${{field.kind === "number" ? "number" : "text"}}" ${{attrs}} value="${{value}}">`;
}}
function coerce(field, raw, checked) {{
  if (field.kind === "checkbox") return checked;
  if (field.kind === "number") return Number(raw);
  return raw;
}}
function collect() {{
  const values = JSON.parse(JSON.stringify(state.values));
  const fields = Object.fromEntries(state.schema.flatMap(tab => tab.fields).map(f => [`${{f.section}}.${{f.key}}`, f]));
  const runName = document.querySelector("#runNameInput");
  if (runName) values.run.run_name = runName.value;
  document.querySelectorAll("[data-section]").forEach(el => {{
    const section = el.dataset.section, key = el.dataset.key;
    values[section][key] = coerce(fields[`${{section}}.${{key}}`], el.value, el.checked);
  }});
  const px = Number(values.observer_camera.pixel_width);
  const py = Number(values.observer_camera.pixel_height);
  if (Number.isFinite(px) && Number.isFinite(py) && px > 0 && py > 0) {{
    values.observer_camera.resolution = `${{Math.round(px)}}x${{Math.round(py)}}`;
  }}
  return values;
}}
function bindNumberInputs() {{
  document.querySelectorAll('input[type="number"]').forEach(input => {{
    input.addEventListener("wheel", event => {{
      if (document.activeElement === input) event.preventDefault();
    }}, {{ passive: false }});
  }});
}}
async function post(path, body) {{
  const res = await fetch(path, {{method: "POST", headers: {{"Content-Type": "application/json"}}, body: JSON.stringify(body)}});
  const text = await res.text();
  document.querySelector("#log").textContent = text;
  if (res.ok && (path === "/api/render" || path === "/api/render-camera-preview")) window.setTimeout(() => window.location.reload(), 500);
  let data = null;
  try {{ data = JSON.parse(text); }} catch (err) {{ data = null; }}
  return {{ok: res.ok, text, data}};
}}
function workflowPayload() {{
  return {{
    values: collect(),
    case_name: document.querySelector("#caseNameInput")?.value || "",
    stage: document.querySelector("#stageInput")?.value || "",
    description: document.querySelector("#descriptionInput")?.value || "",
  }};
}}
function formatWorkflowResult(data) {{
  if (!data) return "No response.";
  const summary = data.summary || {{}};
  const lines = [
    `ok: ${{Boolean(data.ok)}}`,
    `returncode: ${{data.returncode ?? ""}}`,
    "summary:",
    JSON.stringify(summary, null, 2),
  ];
  if (data.stdout) lines.push("\\nstdout:\\n" + data.stdout);
  if (data.stderr) lines.push("\\nstderr:\\n" + data.stderr);
  return lines.join("\\n");
}}
async function workflowPost(path, button) {{
  if (button) button.disabled = true;
  const log = document.querySelector("#workflowActionsOutput");
  try {{
    const res = await fetch(path, {{method: "POST", headers: {{"Content-Type": "application/json"}}, body: JSON.stringify(workflowPayload())}});
    const text = await res.text();
    let data = null;
    try {{ data = JSON.parse(text); }} catch (err) {{ data = {{ok: res.ok, returncode: res.status, stdout: text, stderr: "", summary: {{message: "Non-JSON response"}}}}; }}
    if (log) log.textContent = formatWorkflowResult(data);
    return {{ok: res.ok, data, text}};
  }} finally {{
    if (button) button.disabled = false;
  }}
}}
async function registerRun() {{
  await workflowPost("/api/register-run", document.querySelector("#registerRunButton"));
}}
async function renderProducts() {{
  const button = document.querySelector("#render-button");
  button.disabled = true;
  try {{ await post("/api/render", collect()); }}
  finally {{ button.disabled = false; }}
}}
async function renderCameraPreview() {{
  const button = document.querySelector("#camera-preview-button");
  button.disabled = true;
  try {{ await postCameraPreview("/api/render-camera-preview"); }}
  finally {{ button.disabled = false; }}
}}
async function sampleUheSource() {{
  const button = document.querySelector("#uhe-source-button");
  button.disabled = true;
  try {{
    const values = collect();
    const result = await post("/api/sample-uhe-source", values);
    if (result.ok && result.data && result.data.source) {{
      state.values = values;
      state.values.uhe_neutrino_source.status = "sampled_position_direction_energy_no_forward_kerr_geodesic";
      state.source_summary = result.data.source;
      state.outputs.uhe_source_samples_exists = true;
      state.outputs.uhe_source_summary_exists = true;
      state.outputs.uhe_source_summary_json_exists = true;
      state.outputs.uhe_source_preview_exists = true;
      state.outputs.uhe_source_sampling_uniformity_exists = true;
      state.outputs.uhe_source_sampling_uniformity_report_exists = true;
      state.outputs.uhe_source_direction_uniformity_exists = true;
      state.outputs.uhe_source_direction_uniformity_report_exists = true;
      state.outputs.uhe_source_direction_sphere_exists = true;
      state.source_status = Object.assign({{}}, state.source_status || {{}}, {{
        configured_status: state.values.uhe_neutrino_source.status,
        samples_exists: true,
        preview_exists: true,
        summary_exists: true,
        ready_for_forward_geodesics: true,
      }});
      state.forward_summary = null;
      state.outputs.forward_paths_exists = false;
      state.outputs.forward_path_segments_exists = false;
      state.outputs.forward_summary_exists = false;
      state.outputs.forward_summary_json_exists = false;
      state.outputs.forward_preview_exists = false;
      state.outputs.forward_geometry_3d_exists = false;
      state.outputs.forward_geometry_3d_json_exists = false;
      state.outputs.forward_geometry_3d_html_exists = false;
      state.outputs.isotropic_kerr_strong_field_diagnostic_png_exists = false;
      state.outputs.isotropic_kerr_strong_field_diagnostic_json_exists = false;
      state.outputs.geodesic_validation_report_exists = false;
      state.outputs.stop_condition_statistics_exists = false;
      state.outputs.forward_diagnostic_report_exists = false;
      state.dis_summary = null;
      state.outputs.dis_path_optical_depths_exists = false;
      state.outputs.dis_interaction_candidates_exists = false;
      state.outputs.dis_accepted_interactions_exists = false;
      state.outputs.dis_summary_exists = false;
      state.outputs.dis_summary_json_exists = false;
      state.outputs.dis_tau_preview_exists = false;
      state.outputs.dis_interaction_locations_exists = false;
      state.outputs.dis_interaction_locations_3d_html_exists = false;
      state.outputs.dis_optical_depth_report_exists = false;
      state.outputs.tau_distribution_exists = false;
      state.outputs.interaction_probability_distribution_exists = false;
      state.outputs.optical_depth_map_exists = false;
      state.outputs.optical_depth_map_3d_html_exists = false;
      state.outputs.medium_density_map_exists = false;
      state.outputs.interaction_location_distribution_exists = false;
      state.outputs.local_energy_distribution_exists = false;
      state.outputs.local_density_distribution_exists = false;
      state.outputs.sigma_distribution_exists = false;
      state.outputs.density_energy_sigma_correlation_exists = false;
      state.outputs.dis_diagnostics_report_exists = false;
      state.outputs.gbw_vs_iim_tau_comparison_exists = false;
      state.outputs.gbw_vs_iim_probability_comparison_exists = false;
      state.outputs.gbw_vs_iim_interaction_locations_exists = false;
      state.outputs.gbw_vs_iim_summary_exists = false;
      setObserverBridgeOutputs(false);
      setObserverImageBranchOutputs(false);
      setPowhegOutputs(false);
      state.forward_geodesics_status = Object.assign({{}}, state.forward_geodesics_status || {{}}, {{
        input_uhe_source_found: true,
        paths_exists: false,
        path_segments_exists: false,
        summary_csv_exists: false,
        summary_json_exists: false,
        preview_exists: false,
        geometry_3d_exists: false,
        geometry_3d_json_exists: false,
        geometry_3d_html_exists: false,
        isotropic_kerr_strong_field_diagnostic_png_exists: false,
        isotropic_kerr_strong_field_diagnostic_json_exists: false,
        geodesic_validation_report_exists: false,
        stop_condition_statistics_exists: false,
        forward_diagnostic_report_exists: false,
      }});
      state.dis_interaction_sampler = Object.assign({{}}, state.dis_interaction_sampler || {{}}, {{
        input_uhe_source_found: true,
        input_forward_geodesics_found: false,
        n_paths_available: 0,
        n_segments_available: 0,
        summary: null,
      }});
      state.outputs.provenance_exists = true;
      state.outputs.config_exists = true;
      activeTab = "UHE Source";
      sourcePreviewVersion = Date.now();
      const logText = result.text;
      render();
      const log = document.querySelector("#log");
      if (log) log.textContent = logText;
    }}
  }}
  finally {{ button.disabled = false; }}
}}
async function propagateForwardGeodesics() {{
  const button = document.querySelector("#forward-geodesics-button");
  button.disabled = true;
  try {{
    const values = collect();
    const result = await post("/api/propagate-forward-geodesics", values);
    if (result.ok && result.data && result.data.forward) {{
      state.values = values;
      state.values.forward_geodesics.status = "forward_kerr_geodesics_propagated_no_interactions";
      state.forward_summary = result.data.forward;
      state.outputs.forward_paths_exists = true;
      state.outputs.forward_path_segments_exists = true;
      state.outputs.forward_summary_exists = true;
      state.outputs.forward_summary_json_exists = true;
      state.outputs.forward_preview_exists = true;
      state.outputs.forward_geometry_3d_exists = true;
      state.outputs.forward_geometry_3d_json_exists = true;
      state.outputs.forward_geometry_3d_html_exists = true;
      state.outputs.isotropic_kerr_strong_field_diagnostic_png_exists = true;
      state.outputs.isotropic_kerr_strong_field_diagnostic_json_exists = true;
      state.outputs.geodesic_validation_report_exists = true;
      state.outputs.stop_condition_statistics_exists = true;
      state.outputs.forward_diagnostic_report_exists = true;
      state.dis_summary = null;
      state.outputs.dis_path_optical_depths_exists = false;
      state.outputs.dis_interaction_candidates_exists = false;
      state.outputs.dis_accepted_interactions_exists = false;
      state.outputs.dis_summary_exists = false;
      state.outputs.dis_summary_json_exists = false;
      state.outputs.dis_tau_preview_exists = false;
      state.outputs.dis_interaction_locations_exists = false;
      state.outputs.dis_interaction_locations_3d_html_exists = false;
      state.outputs.dis_optical_depth_report_exists = false;
      state.outputs.tau_distribution_exists = false;
      state.outputs.interaction_probability_distribution_exists = false;
      state.outputs.optical_depth_map_exists = false;
      state.outputs.optical_depth_map_3d_html_exists = false;
      state.outputs.medium_density_map_exists = false;
      state.outputs.interaction_location_distribution_exists = false;
      state.outputs.local_energy_distribution_exists = false;
      state.outputs.local_density_distribution_exists = false;
      state.outputs.sigma_distribution_exists = false;
      state.outputs.density_energy_sigma_correlation_exists = false;
      state.outputs.dis_diagnostics_report_exists = false;
      state.outputs.gbw_vs_iim_tau_comparison_exists = false;
      state.outputs.gbw_vs_iim_probability_comparison_exists = false;
      state.outputs.gbw_vs_iim_interaction_locations_exists = false;
      state.outputs.gbw_vs_iim_summary_exists = false;
      setObserverBridgeOutputs(false);
      setObserverImageBranchOutputs(false);
      setPowhegOutputs(false);
      state.forward_geodesics_status = Object.assign({{}}, state.forward_geodesics_status || {{}}, {{
        configured_status: state.values.forward_geodesics.status,
        input_uhe_source_found: true,
        paths_exists: true,
        path_segments_exists: true,
        summary_csv_exists: true,
        summary_json_exists: true,
        preview_exists: true,
        geometry_3d_exists: true,
        geometry_3d_json_exists: true,
        geometry_3d_html_exists: true,
        isotropic_kerr_strong_field_diagnostic_png_exists: true,
        isotropic_kerr_strong_field_diagnostic_json_exists: true,
        geodesic_validation_report_exists: true,
        stop_condition_statistics_exists: true,
        forward_diagnostic_report_exists: true,
      }});
      state.dis_interaction_sampler = Object.assign({{}}, state.dis_interaction_sampler || {{}}, {{
        input_uhe_source_found: true,
        input_forward_geodesics_found: true,
        n_paths_available: result.data.forward.n_paths,
        n_segments_available: result.data.forward.n_segments,
        summary: null,
      }});
      state.outputs.provenance_exists = true;
      state.outputs.config_exists = true;
      activeTab = "Forward Geodesics";
      forwardPreviewVersion = Date.now();
      const logText = result.text;
      render();
      const log = document.querySelector("#log");
      if (log) log.textContent = logText;
    }}
  }}
  finally {{ button.disabled = false; }}
}}
async function sampleDisInteractions() {{
  const button = document.querySelector("#dis-sampler-button");
  button.disabled = true;
  try {{
    const values = collect();
    const result = await post("/api/sample-dis-interactions", values);
    if (result.ok && result.data && result.data.dis) {{
      state.values = values;
      state.values.dis_interaction_sampler.status = "dis_optical_depth_sampled_no_observer_bridge";
      state.dis_summary = result.data.dis;
      state.outputs.dis_path_optical_depths_exists = true;
      state.outputs.dis_interaction_candidates_exists = true;
      state.outputs.dis_accepted_interactions_exists = true;
      state.outputs.dis_summary_exists = true;
      state.outputs.dis_summary_json_exists = true;
      state.outputs.dis_tau_preview_exists = true;
      state.outputs.dis_interaction_locations_exists = true;
      state.outputs.dis_interaction_locations_3d_html_exists = true;
      state.outputs.dis_optical_depth_report_exists = true;
      state.outputs.tau_distribution_exists = true;
      state.outputs.interaction_probability_distribution_exists = true;
      state.outputs.optical_depth_map_exists = true;
      state.outputs.optical_depth_map_3d_html_exists = true;
      state.outputs.medium_density_map_exists = true;
      state.outputs.interaction_location_distribution_exists = true;
      state.outputs.local_energy_distribution_exists = true;
      state.outputs.local_density_distribution_exists = true;
      state.outputs.sigma_distribution_exists = true;
      state.outputs.density_energy_sigma_correlation_exists = true;
      state.outputs.dis_diagnostics_report_exists = true;
      state.outputs.gbw_vs_iim_tau_comparison_exists = true;
      state.outputs.gbw_vs_iim_probability_comparison_exists = true;
      state.outputs.gbw_vs_iim_interaction_locations_exists = true;
      state.outputs.gbw_vs_iim_summary_exists = true;
      setObserverBridgeOutputs(false);
      setObserverImageBranchOutputs(false);
      setPowhegOutputs(false);
      state.dis_interaction_sampler = Object.assign({{}}, state.dis_interaction_sampler || {{}}, {{
        configured_status: state.values.dis_interaction_sampler.status,
        input_uhe_source_found: true,
        input_forward_geodesics_found: true,
        n_paths_available: result.data.dis.n_paths_processed,
        n_segments_available: result.data.dis.n_segments_processed,
        output_dir_found: true,
        summary: result.data.dis,
      }});
      state.outputs.provenance_exists = true;
      state.outputs.config_exists = true;
      activeTab = "DIS Interaction Sampler";
      disPreviewVersion = Date.now();
      const logText = result.text;
      render();
      const log = document.querySelector("#log");
      if (log) log.textContent = logText;
    }}
  }}
  finally {{ button.disabled = false; }}
}}
async function compareDisModels() {{
  const button = document.querySelector("#dis-compare-button");
  button.disabled = true;
  try {{
    const values = collect();
    const result = await post("/api/compare-dis-models", values);
    if (result.ok && result.data && result.data.comparison) {{
      state.values = values;
      state.outputs.gbw_vs_iim_tau_comparison_exists = true;
      state.outputs.gbw_vs_iim_probability_comparison_exists = true;
      state.outputs.gbw_vs_iim_interaction_locations_exists = true;
      state.outputs.gbw_vs_iim_summary_exists = true;
      state.outputs.dis_diagnostics_report_exists = true;
      activeTab = "DIS Interaction Sampler";
      disPreviewVersion = Date.now();
      const logText = result.text;
      render();
      const log = document.querySelector("#log");
      if (log) log.textContent = logText;
    }}
  }}
  finally {{ button.disabled = false; }}
}}
async function computeObserverBridge() {{
  const button = document.querySelector("#observer-bridge-button");
  button.disabled = true;
  try {{
    const values = collect();
    const result = await post("/api/observer-bridge", values);
    if (result.ok && result.data && result.data.observer_bridge) {{
      state.values = values;
      state.values.observer_bridge.status = "observer_bridge_scored_no_event_generation";
      state.observer_bridge_summary = result.data.observer_bridge;
      setObserverBridgeOutputs(true);
      setObserverImageBranchOutputs(false);
      setPowhegOutputs(false);
      state.observer_bridge = Object.assign({{}}, state.observer_bridge || {{}}, {{
        configured_status: state.values.observer_bridge.status,
        input_dis_found: true,
        n_interactions_input: result.data.observer_bridge.n_interactions_input,
        n_candidates_scored: result.data.observer_bridge.n_candidates_scored,
        summary: result.data.observer_bridge,
      }});
      state.outputs.provenance_exists = true;
      state.outputs.config_exists = true;
      activeTab = "Observer Bridge";
      observerBridgePreviewVersion = Date.now();
      const logText = result.text;
      render();
      const log = document.querySelector("#log");
      if (log) log.textContent = logText;
    }}
  }}
  finally {{ button.disabled = false; }}
}}
async function analyzeObserverImageBranches() {{
  const button = document.querySelector("#observer-image-branches-button");
  button.disabled = true;
  try {{
    const values = collect();
    const result = await post("/api/observer-image-branches", values);
    if (result.ok && result.data && result.data.observer_image_branches) {{
      state.values = values;
      state.values.observer_image_branches.status = "observer_image_branches_analyzed";
      state.observer_image_branches.summary = result.data.observer_image_branches;
      setObserverImageBranchOutputs(true);
      setPowhegOutputs(false);
      state.outputs.provenance_exists = true;
      state.outputs.config_exists = true;
      activeTab = "Observer Image Branches";
      observerBridgePreviewVersion = Date.now();
      const logText = result.text;
      render();
      const log = document.querySelector("#log");
      if (log) log.textContent = logText;
    }}
  }}
  finally {{ button.disabled = false; }}
}}
async function preparePowheg() {{
  const button = document.querySelector("#powheg-button");
  button.disabled = true;
  try {{
    const values = collect();
    const result = await post("/api/powheg", values);
    if (result.ok && result.data && result.data.powheg) {{
      state.values = values;
      state.powheg_summary = result.data.powheg;
      setPowhegOutputs(true);
      state.outputs.powheg_validation_report_exists = Boolean(result.data.powheg.powheg_validation_report_generated);
      state.outputs.powheg_lhe_exists = Boolean(result.data.powheg.powheg_lhe_generated);
      state.outputs.powheg_log_exists = Boolean(result.data.powheg.powheg_lhe_generated);
      state.outputs.powheg_lhe_particles_exists = Boolean(result.data.powheg.powheg_lhe_products_generated);
      state.outputs.powheg_lhe_events_summary_exists = Boolean(result.data.powheg.powheg_lhe_products_generated);
      state.outputs.powheg_lhe_particle_summary_csv_exists = Boolean(result.data.powheg.powheg_lhe_products_generated);
      state.outputs.powheg_lhe_particle_summary_json_exists = Boolean(result.data.powheg.powheg_lhe_products_generated);
      state.outputs.powheg_lhe_particle_histogram_exists = Boolean(result.data.powheg.powheg_lhe_products_generated);
      state.outputs.powheg_lhe_energy_spectrum_exists = Boolean(result.data.powheg.powheg_lhe_products_generated);
      state.outputs.powheg_lhe_momentum_spectrum_exists = Boolean(result.data.powheg.powheg_lhe_products_generated);
      state.outputs.powheg_hard_process_event_display_exists = Boolean(result.data.powheg.powheg_hard_process_event_display_generated);
      state.outputs.powheg_hard_process_event_display_view_exists = Boolean(result.data.powheg.powheg_hard_process_event_display_view_generated);
      state.outputs.powheg_event_summary_table_exists = Boolean(result.data.powheg.powheg_lhe_products_generated);
      state.outputs.powheg_particle_table_exists = Boolean(result.data.powheg.powheg_lhe_products_generated);
      state.outputs.powheg_particle_table_html_exists = Boolean(result.data.powheg.powheg_lhe_products_generated);
      state.outputs.powheg_particle_content_report_exists = Boolean(result.data.powheg.powheg_particle_content_report_generated);
      state.outputs.powheg_lhe_event_view_exists = Boolean(result.data.powheg.powheg_lhe_event_view_generated);
      state.powheg = Object.assign({{}}, state.powheg || {{}}, {{
        input_observer_image_branches_found: true,
        n_candidates_input: result.data.powheg.n_candidates_input,
        powheg_jobs_prepared: result.data.powheg.powheg_jobs_prepared,
        powheg_cards_generated: result.data.powheg.powheg_cards_generated,
        powheg_lhe_generated: Boolean(result.data.powheg.powheg_lhe_generated),
        n_lhe_events: result.data.powheg.n_lhe_events || 0,
        summary: result.data.powheg,
      }});
      state.outputs.provenance_exists = true;
      state.outputs.config_exists = true;
      activeTab = "POWHEG";
      powhegPreviewVersion = Date.now();
      const logText = result.text;
      render();
      const log = document.querySelector("#log");
      if (log) log.textContent = logText;
    }}
  }}
  finally {{ button.disabled = false; }}
}}
async function launchInteractiveCameraPreview() {{
  const button = document.querySelector("#interactive-camera-button");
  button.disabled = true;
  try {{ await post("/api/launch-interactive-camera-preview", collect()); }}
  finally {{ button.disabled = false; }}
}}
function previewOptions() {{
  return {{
    previewResolution,
    previewInteractiveResolution,
    previewSkyMode,
    previewGeodesicModel,
    previewNavMode,
    previewCelestialRadiusRs,
    previewTorusMode: previewPhysicalTorus ? "physical" : "generic",
    previewOpaqueStructures: previewOpaqueStructures ? "1" : "0",
  }};
}}
async function postCameraPreview(path, valuesOverride = null) {{
  const res = await fetch(path, {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({{values: valuesOverride || collect(), previewOptions: previewOptions()}}),
  }});
  document.querySelector("#log").textContent = await res.text();
}}
async function getLastCamera() {{
  const res = await fetch("/api/last-camera", {{cache: "no-store"}});
  if (!res.ok) return {{exists: false, mtime: 0}};
  return await res.json();
}}
function startPreviewCameraPolling(baselineMtime = lastCameraMtime) {{
  lastCameraMtime = baselineMtime || 0;
  if (previewPollTimer) clearInterval(previewPollTimer);
  previewPollTimer = setInterval(async () => {{
    try {{
      const current = await getLastCamera();
      if (current.exists && current.mtime && current.mtime > lastCameraMtime) {{
        await loadSavedCameraPreview(false);
      }}
    }} catch (err) {{
      // Keep polling unobtrusive while the native preview window is open.
    }}
  }}, 1200);
  setTimeout(() => {{
    if (previewPollTimer) {{
      clearInterval(previewPollTimer);
      previewPollTimer = null;
    }}
  }}, 10 * 60 * 1000);
}}
async function launchHadrosCameraPreview() {{
  const button = document.querySelector("#cameraPreview");
  if (button) {{
    button.classList.remove("blinking");
    void button.offsetWidth;
    button.classList.add("blinking");
    setTimeout(() => button.classList.remove("blinking"), 2800);
  }}
  try {{
    const current = await getLastCamera();
    lastCameraMtime = current.mtime || 0;
  }} catch (err) {{
    lastCameraMtime = 0;
  }}
  const previewRMaxRg = Number(previewCelestialRadiusRs || 40) * 2;
  const navDescription = previewNavMode === "paint_swatch_disk"
    ? "paint_swatch_disk = diagnostic visual test, not physical torus emission"
    : previewNavMode;
  document.querySelector("#log").textContent =
    "Launching geodesic camera preview at " + previewResolution + " final / " + previewInteractiveResolution +
    " interactive with " + (previewGeodesicModel === "full_kerr" ? "Full Kerr CUDA" : "Kerr-like CUDA") +
    ", " + navDescription + ", " + previewSkyMode + " sky, " +
    (previewPhysicalTorus ? "physical preset torus/funnel" : "generic torus") + ", " +
    (previewOpaqueStructures ? "opaque structures" : "translucent structures") +
    ", and celestial sphere radius " + previewCelestialRadiusRs + " R_S.\\n\\n" +
    "If no native window appears, run the command shown in the launch summary/log. PREVIEW_R_MAX_RG=" + previewRMaxRg;
  await postCameraPreview("/api/launch-interactive-camera-preview");
  startPreviewCameraPolling(lastCameraMtime);
}}
async function launchCpuCameraPreview() {{
  try {{
    const current = await getLastCamera();
    lastCameraMtime = current.mtime || 0;
  }} catch (err) {{
    lastCameraMtime = 0;
  }}
  const values = collect();
  values.observer_camera.camera_preview_mode = "analytic_geometry_only";
  state.values = values;
  previewGeodesicModel = "kerr_like";
  document.querySelector("#log").textContent =
    "Launching original HADROS CPU/OpenGL geodesic preview. This uses the same camera controls but forces PREVIEW_BACKEND=cpu.\\n" +
    "Use R to rerender, arrows/mouse to orbit, +/- for distance, [] for FOV, S to save, Q to quit.";
  await postCameraPreview("/api/launch-interactive-camera-preview", values);
  startPreviewCameraPolling(lastCameraMtime);
}}
async function loadSavedCameraPreview(manual = true) {{
  const camera = await getLastCamera();
  if (manual) document.querySelector("#log").textContent = JSON.stringify(camera, null, 2);
  if (!camera.exists || !camera.camera) return;
  const values = collect();
  values.black_hole.spin_a = Number(camera.camera.requested_spin ?? camera.camera.spin ?? values.black_hole.spin_a);
  values.observer_camera.observer_distance_rg = Number(camera.camera.observer_distance_rg ?? values.observer_camera.observer_distance_rg);
  values.observer_camera.inclination_deg = Number(camera.camera.inclination_deg ?? values.observer_camera.inclination_deg);
  values.observer_camera.azimuth_deg = Number(camera.camera.azimuth_deg ?? values.observer_camera.azimuth_deg);
  values.observer_camera.field_of_view_deg = Number(camera.camera.fov_deg ?? values.observer_camera.field_of_view_deg);
  values.observer_camera.preview_final_resolution = previewResolution;
  values.observer_camera.preview_resolution = previewInteractiveResolution;
  values.observer_camera.preview_nav_mode = previewNavMode;
  state.values = values;
  lastCameraMtime = camera.mtime || lastCameraMtime;
  render();
  const log = document.querySelector("#log");
  if (log) {{
    log.textContent =
      "Loaded saved interactive camera into the Camera fields.\\n" +
      "observer_distance_rg=" + values.observer_camera.observer_distance_rg + "\\n" +
      "inclination_deg=" + values.observer_camera.inclination_deg + "\\n" +
      "azimuth_deg=" + values.observer_camera.azimuth_deg + "\\n" +
      "FOV=" + values.observer_camera.field_of_view_deg + "\\n" +
      "spin_a=" + values.black_hole.spin_a + "\\n" +
      "source=" + camera.path;
  }}
}}
function renderHadrosCameraPanel() {{
  const options = (items, selected, recommended) => items.map(item => {{
    const value = Array.isArray(item) ? item[0] : item;
    const label = Array.isArray(item) ? item[1] : item;
    const suffix = value === recommended ? " [Recommended]" : "";
    return `<option value="${{value}}" ${{String(selected) === String(value) ? "selected" : ""}}>${{label}}${{suffix}}</option>`;
  }}).join("");
  const controls = [
    ["Left / Right", "azimuth"],
    ["Up / Down", "inclination"],
    ["mouse drag", "azimuth + inclination"],
    ["+ / -", "observer distance"],
    ["[ / ]", "FOV"],
    ["mouse wheel", "FOV"],
    ["A / D", "spin"],
    ["< / >", "integration step"],
    ["R", "rerender"],
    ["S", "save camera to fields"],
    ["Q", "quit"],
  ].map(([key, description]) => `<div class="camera-control-item"><kbd>${{key}}</kbd><span>${{description}}</span></div>`).join("");
  const diagnosticNote = previewNavMode === "paint_swatch_disk"
    ? `<p class="note"><code>paint_swatch_disk</code> = diagnostic visual test, not physical torus emission</p>`
    : "";
  return `<div class="camera-preview-panel">
    <div class="camera-preview-top"><button type="button" id="cameraPreview" class="camera-preview-button">Camera Preview</button><button type="button" id="cpuCameraPreview">CPU/OpenGL Preview</button><button type="button" id="loadCameraPreview">Load Saved Preview</button></div>
    <div class="camera-preview-row">
      <div class="camera-preview-field"><label for="cameraPreviewResolution">Preview resolution</label><select id="cameraPreviewResolution">${{options(finalPreviewResolutionsNormal, previewResolution, "512x288")}}</select></div>
      <div class="camera-preview-field"><label for="cameraPreviewInteractiveResolution">Interactive resolution</label><select id="cameraPreviewInteractiveResolution">${{options(interactivePreviewResolutionsNormal, previewInteractiveResolution, "256x144")}}</select></div>
      <div class="camera-preview-field"><label for="cameraPreviewSky">Sky background</label><select id="cameraPreviewSky">${{options(previewSkyOptions, previewSkyMode, "texture")}}</select></div>
      <div class="camera-preview-field"><label for="cameraPreviewGeodesicModel">Geodesic model</label><select id="cameraPreviewGeodesicModel">${{options(previewGeodesicModelOptions, previewGeodesicModel, "kerr_like")}}</select></div>
      <div class="camera-preview-field"><label for="cameraPreviewNavMode">Preview mode</label><select id="cameraPreviewNavMode">${{options(previewNavModeNormal, previewNavMode, "celestial_plus_torus_volume")}}</select></div>
      <div class="camera-preview-field"><label for="cameraPreviewCelestialRadius">Celestial sphere radius (R_S)</label><input id="cameraPreviewCelestialRadius" type="number" min="5" max="5000" step="1" value="${{previewCelestialRadiusRs}}"></div>
      <label class="toggle-row"><input id="previewPhysicalTorus" type="checkbox" ${{previewPhysicalTorus ? "checked" : ""}}><span class="toggle-main"><span class="toggle-name">Physical preset torus/funnel</span><span class="toggle-help">Use current geometry instead of generic proxy.</span></span></label>
      <label class="toggle-row"><input id="previewOpaqueStructures" type="checkbox" ${{previewOpaqueStructures ? "checked" : ""}}><span class="toggle-main"><span class="toggle-name">Solid opaque structures</span><span class="toggle-help">Display-only solid geometry mode.</span></span></label>
    </div>
    ${{diagnosticNote}}
    <div class="camera-controls-card"><h3>Camera Preview Controls</h3><div class="camera-controls-grid">${{controls}}</div></div>
  </div>`;
}}
function bindHadrosCameraPanel() {{
  const get = id => document.querySelector("#" + id);
  if (!get("cameraPreview")) return;
  get("cameraPreviewResolution").onchange = event => previewResolution = event.target.value;
  get("cameraPreviewInteractiveResolution").onchange = event => previewInteractiveResolution = event.target.value;
  get("cameraPreviewSky").onchange = event => previewSkyMode = event.target.value;
  get("cameraPreviewGeodesicModel").onchange = event => {{
    previewGeodesicModel = event.target.value;
    const values = collect();
    values.observer_camera.camera_preview_mode = previewGeodesicModel === "full_kerr" ? "full_kerr" : "kerr_like_cuda";
    state.values = values;
  }};
  get("cameraPreviewNavMode").onchange = event => {{
    previewNavMode = event.target.value;
    state.values.observer_camera.preview_nav_mode = previewNavMode;
    render();
  }};
  get("cameraPreviewCelestialRadius").oninput = event => previewCelestialRadiusRs = event.target.value || "40";
  get("previewPhysicalTorus").oninput = event => previewPhysicalTorus = event.target.checked;
  get("previewOpaqueStructures").oninput = event => previewOpaqueStructures = event.target.checked;
  get("cameraPreview").onclick = launchHadrosCameraPreview;
  get("cpuCameraPreview").onclick = launchCpuCameraPreview;
  get("loadCameraPreview").onclick = loadSavedCameraPreview;
}}
let activeTab = "Camera";
function safeRunName(name) {{
  const cleaned = String(name || "").trim().replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/^[._-]+|[._-]+$/g, "");
  return cleaned || "HADROS3_run";
}}
function tabLabel(tab) {{
  const aliases = {{"Analytic Torus": "Torus / Medium", "Polar Cone": "Funnel / Cone"}};
  return aliases[tab.tab] || tab.tab;
}}
function orderedTabs() {{
  const order = ["Camera", "Black Hole", "Torus / Medium", "Funnel / Cone", "UHE Source", "Forward Geodesics", "DIS Interaction Sampler", "Observer Bridge", "Observer Image Branches", "POWHEG", "Event Generation", "GEANT4", "Photon Transport", "Spectra", "Outputs", "Provenance"];
  return [...state.schema].sort((a, b) => {{
    const ai = order.indexOf(tabLabel(a));
    const bi = order.indexOf(tabLabel(b));
    return (ai < 0 ? 999 : ai) - (bi < 0 ? 999 : bi);
  }});
}}
function renderFields(tab) {{
  const visibleFields = tab.fields.filter(f => f.visibility !== "INTERNAL");
  return `<section class="active-panel"><h2>${{tabLabel(tab)}}</h2>` +
    visibleFields.map(f => `<label><span>${{f.label}}${{f.visibility === "EXPERT" ? " (Expert)" : ""}}</span>${{inputFor(f, state.values[f.section][f.key])}}</label>`).join("") +
    `</section>`;
}}
function renderBackendTable() {{
  const rows = Object.entries(state.camera_backends).map(([mode, info]) =>
    `<tr><td>${{mode}}</td><td class="${{info.available ? "ok" : "pending"}}">${{info.available ? "available" : "unavailable"}}</td><td>${{info.backend || ""}}</td></tr>`
  ).join("");
  const summary = state.camera_summary;
  const diagnosticSummary = summary && summary.paint_swatch_disk_diagnostic_mode
    ? `<p class="note"><code>paint_swatch_disk</code> = diagnostic visual test, not physical torus emission</p>`
    : "";
  const summaryHtml = summary ? `<p class="note">Camera preview: <strong>${{summary.status}}</strong> / mode <code>${{summary.requested_mode}}</code><br>${{summary.message || ""}}</p>${{diagnosticSummary}}
  <div class="summary-grid">
    <div class="summary-item"><strong>backend_used</strong><code>${{summary.backend_used || "pending"}}</code></div>
    <div class="summary-item"><strong>cuda_used</strong>${{String(summary.cuda_used)}}</div>
    <div class="summary-item"><strong>fallback_used</strong>${{String(summary.fallback_used)}}</div>
    <div class="summary-item"><strong>camera_preview_cuda_self_contained</strong>${{String(summary.camera_preview_cuda_self_contained)}}</div>
    <div class="summary-item"><strong>camera_preview_external_hadros_used</strong>${{String(summary.camera_preview_external_hadros_used)}}</div>
    <div class="summary-item"><strong>paint_swatch_disk_diagnostic_mode</strong>${{String(summary.paint_swatch_disk_diagnostic_mode)}}</div>
    <div class="summary-item"><strong>paint_swatch_disk_uses_forced_thin_disk</strong>${{String(summary.paint_swatch_disk_uses_forced_thin_disk)}}</div>
    <div class="summary-item"><strong>paint_swatch_disk_physical_torus_emission</strong>${{String(summary.paint_swatch_disk_physical_torus_emission)}}</div>
  </div>` : `<p class="note">No camera preview summary yet.</p>`;
  return `${{summaryHtml}}<table class="backend-table"><thead><tr><th>Mode</th><th>Status</th><th>Backend</th></tr></thead><tbody>${{rows}}</tbody></table>`;
}}
function renderSourcePanel() {{
  const summary = state.source_summary;
  const sourceValues = state.values.uhe_neutrino_source;
  const directionModel = sourceValues.direction_model || "isotropic_local";
  const directionOptions = [
    ["isotropic_local", "Isotropic local", true, "Recommended physical model."],
    ["coordinate_radial_outward", "Coordinate radial outward", true, "Diagnostic model."],
    ["jet_axis_future", "Jet axis", false],
    ["cone_emission_future", "Cone emission", false],
    ["custom_future", "Custom (future)", false],
  ].map(([value, label, enabled, help]) => `<label class="toggle-row"><input type="radio" name="directionModelDisplay" value="${{value}}" ${{directionModel === value ? "checked" : ""}} disabled><span class="toggle-main"><span class="toggle-name">${{label}}</span><span class="toggle-help">${{enabled ? help || "Implemented in H3-W5." : "Future model."}}</span></span></label>`).join("");
  const directionStatus = ["coordinate_radial_outward", "isotropic_local"].includes(directionModel) ? "implemented" : "future";
  const directionPanel = `<section><h2>Initial Direction</h2>
    <p class="note">The UHE source samples emission position, energy and direction.<br>The Kerr four-momentum is not sampled here; it is constructed later by Forward Geodesics from position + energy + direction.</p>
    <div class="summary-grid">
      <div class="summary-item"><strong>direction_model</strong>${{directionModel}}</div>
      <div class="summary-item"><strong>direction_opening_angle_deg</strong>${{sourceValues.direction_opening_angle_deg}}</div>
      <div class="summary-item"><strong>direction_seed</strong>${{sourceValues.direction_seed}}</div>
      <div class="summary-item"><strong>direction_sampling_pdf</strong>${{summary && summary.direction_sampling_pdf ? Number(summary.direction_sampling_pdf).toExponential(4) : "pending"}}</div>
      <div class="summary-item"><strong>direction_physical_pdf</strong>${{summary && summary.direction_physical_pdf ? Number(summary.direction_physical_pdf).toExponential(4) : "pending"}}</div>
      <div class="summary-item"><strong>direction_weight</strong>${{summary && summary.direction_weight ? summary.direction_weight : "pending"}}</div>
      <div class="summary-item"><strong>Model status</strong><span class="${{directionStatus === "implemented" ? "ok" : "pending"}}">${{directionStatus}}</span></div>
    </div>
    <div class="camera-controls-card">${{directionOptions}}</div>
  </section>`;
  const sourceLinks = `<div class="output-link-grid">
    ${{state.outputs.uhe_source_samples_exists ? `<a href="${{outUrl("uhe_source_samples")}}" target="_blank">Samples<br><code>${{outPath("uhe_source_samples")}}</code></a>` : ""}}
    ${{state.outputs.uhe_source_summary_exists ? `<a href="${{outUrl("uhe_source_summary")}}" target="_blank">Summary CSV<br><code>${{outPath("uhe_source_summary")}}</code></a>` : ""}}
    ${{state.outputs.uhe_source_summary_json_exists ? `<a href="${{outUrl("uhe_source_summary_json")}}" target="_blank">Summary JSON<br><code>${{outPath("uhe_source_summary_json")}}</code></a>` : ""}}
    ${{state.outputs.uhe_source_sampling_uniformity_exists ? `<a href="${{outUrl("uhe_source_sampling_uniformity")}}" target="_blank">Sampling uniformity PNG<br><code>${{outPath("uhe_source_sampling_uniformity")}}</code></a>` : ""}}
    ${{state.outputs.uhe_source_sampling_uniformity_report_exists ? `<a href="${{outUrl("uhe_source_sampling_uniformity_report")}}" target="_blank">Sampling uniformity JSON<br><code>${{outPath("uhe_source_sampling_uniformity_report")}}</code></a>` : ""}}
    ${{state.outputs.uhe_source_direction_uniformity_exists ? `<a href="${{outUrl("uhe_source_direction_uniformity")}}" target="_blank">Direction uniformity PNG<br><code>${{outPath("uhe_source_direction_uniformity")}}</code></a>` : ""}}
    ${{state.outputs.uhe_source_direction_sphere_exists ? `<a href="${{outUrl("uhe_source_direction_sphere")}}" target="_blank">Direction sphere PNG<br><code>${{outPath("uhe_source_direction_sphere")}}</code></a>` : ""}}
    ${{state.outputs.uhe_source_direction_uniformity_report_exists ? `<a href="${{outUrl("uhe_source_direction_uniformity_report")}}" target="_blank">Direction uniformity JSON<br><code>${{outPath("uhe_source_direction_uniformity_report")}}</code></a>` : ""}}
  </div>`;
  const summaryHtml = summary ? `<div class="summary-grid">
    <div class="summary-item"><strong>Status</strong>${{summary.status}}</div>
    <div class="summary-item"><strong>Samples</strong>${{summary.n_samples}}</div>
    <div class="summary-item"><strong>Energy</strong>${{summary.energy_gev}} GeV</div>
    <div class="summary-item"><strong>Seed</strong>${{summary.random_seed}}</div>
    <div class="summary-item"><strong>Model</strong>${{summary.source_model}}</div>
    <div class="summary-item"><strong>Volume</strong>${{summary.source_volume_model}}</div>
    <div class="summary-item"><strong>source_sampling_pdf</strong>${{Number(summary.source_sampling_pdf).toExponential(4)}}</div>
    <div class="summary-item"><strong>source_physical_pdf</strong>${{Number(summary.source_physical_pdf).toExponential(4)}}</div>
    <div class="summary-item"><strong>source_weight</strong>${{summary.source_weight_mean}}</div>
    <div class="summary-item"><strong>Direction</strong>${{summary.direction_model}}</div>
    <div class="summary-item"><strong>Direction weight</strong>${{summary.direction_weight}}</div>
    <div class="summary-item"><strong>Momentum</strong>${{summary.momentum_generator}}</div>
    <div class="summary-item"><strong>Kerr physical?</strong>${{summary.momentum_is_physical_kerr}}</div>
  </div><p class="note"><strong>Sampler status:</strong> ${{summary.source_status}}</p>${{sourceLinks}}` : `<p class="note">Sampler inactive. Configure this tab and generate source samples through hadros-web.</p>`;
  return `<div class="source-panel">
    ${{directionPanel}}
    <button type="button" id="uhe-source-button" class="source-action">Generate UHE Source Samples</button>
    ${{summaryHtml}}
  </div>`;
}}
function renderForwardPanel() {{
  const summary = state.forward_summary;
  const forwardStatus = state.forward_geodesics_status || {{}};
  const sourceFound = Boolean(forwardStatus.input_uhe_source_found || state.outputs.uhe_source_samples_exists);
  const inputStatus = `<div class="summary-grid">
    <div class="summary-item"><strong>Input UHEsource/</strong><span class="${{sourceFound ? "ok" : "pending"}}">${{sourceFound ? "found" : "missing"}}</span></div>
    <div class="summary-item"><strong>Input samples</strong><code>${{outPath("uhe_source_samples")}}</code></div>
    <div class="summary-item"><strong>Input</strong><code>UHEsource/uhe_neutrino_source_samples.jsonl</code></div>
    <div class="summary-item"><strong>Uses</strong>position + energy + emission_direction</div>
    <div class="summary-item"><strong>Builds</strong>Kerr null four-momentum <code>p_mu</code></div>
    <div class="summary-item"><strong>Propagation</strong>Full Kerr null geodesic propagation</div>
    <div class="summary-item"><strong>Physics backend</strong><code>${{summary ? (summary.forward_backend || "pending") : (state.values.forward_geodesics.forward_backend || "pending")}}</code></div>
  </div>`;
  const forwardLinks = `<div class="output-link-grid">
    ${{state.outputs.forward_geometry_3d_html_exists ? `<a href="${{outUrl("forward_geometry_3d_html")}}" target="_blank">Interactive 3D geometry<br><code>${{outPath("forward_geometry_3d_html")}}</code></a>` : ""}}
    ${{state.outputs.forward_geometry_3d_exists ? `<a href="${{outUrl("forward_geometry_3d")}}" target="_blank">3D geometry PNG<br><code>${{outPath("forward_geometry_3d")}}</code></a>` : ""}}
    ${{state.outputs.forward_geometry_3d_json_exists ? `<a href="${{outUrl("forward_geometry_3d_json")}}" target="_blank">3D geometry JSON<br><code>${{outPath("forward_geometry_3d_json")}}</code></a>` : ""}}
    ${{state.outputs.isotropic_kerr_strong_field_diagnostic_png_exists ? `<a href="${{outUrl("isotropic_kerr_strong_field_diagnostic_png")}}" target="_blank">Strong-field isotropic diagnostic PNG<br><code>${{outPath("isotropic_kerr_strong_field_diagnostic_png")}}</code></a>` : ""}}
    ${{state.outputs.isotropic_kerr_strong_field_diagnostic_json_exists ? `<a href="${{outUrl("isotropic_kerr_strong_field_diagnostic_json")}}" target="_blank">Strong-field isotropic diagnostic JSON<br><code>${{outPath("isotropic_kerr_strong_field_diagnostic_json")}}</code></a>` : ""}}
    ${{state.outputs.forward_preview_exists ? `<a href="${{outUrl("forward_preview")}}" target="_blank">Preview PNG<br><code>${{outPath("forward_preview")}}</code></a>` : ""}}
    ${{state.outputs.forward_summary_json_exists ? `<a href="${{outUrl("forward_summary_json")}}" target="_blank">Summary JSON<br><code>${{outPath("forward_summary_json")}}</code></a>` : ""}}
    ${{state.outputs.forward_summary_exists ? `<a href="${{outUrl("forward_summary")}}" target="_blank">Summary CSV<br><code>${{outPath("forward_summary")}}</code></a>` : ""}}
    ${{state.outputs.forward_paths_exists ? `<a href="${{outUrl("forward_paths")}}" target="_blank">Paths<br><code>${{outPath("forward_paths")}}</code></a>` : ""}}
    ${{state.outputs.forward_path_segments_exists ? `<a href="${{outUrl("forward_path_segments")}}" target="_blank">Segments<br><code>${{outPath("forward_path_segments")}}</code></a>` : ""}}
    ${{state.outputs.geodesic_validation_report_exists ? `<a href="${{outUrl("geodesic_validation_report")}}" target="_blank">Validation<br><code>${{outPath("geodesic_validation_report")}}</code></a>` : ""}}
    ${{state.outputs.stop_condition_statistics_exists ? `<a href="${{outUrl("stop_condition_statistics")}}" target="_blank">Stops<br><code>${{outPath("stop_condition_statistics")}}</code></a>` : ""}}
    ${{state.outputs.forward_diagnostic_report_exists ? `<a href="${{outUrl("forward_diagnostic_report")}}" target="_blank">Diagnostic report<br><code>${{outPath("forward_diagnostic_report")}}</code></a>` : ""}}
  </div>`;
  const diagnosticsReady = state.outputs.validation_invariants_exists || state.outputs.kerr_bending_vs_impact_parameter_exists || state.outputs.stop_condition_distribution_exists || state.outputs.geodesic_density_map_exists || state.outputs.forward_geodesics_diagnostics_report_exists;
  const diagnosticsHtml = diagnosticsReady ? `<section><h2>Diagnostics</h2><div class="output-link-grid">
    ${{state.outputs.validation_invariants_exists ? `<a href="${{outUrl("validation_invariants")}}" target="_blank">Invariant conservation<br><code>${{outPath("validation_invariants")}}</code></a>` : ""}}
    ${{state.outputs.kerr_bending_vs_impact_parameter_exists ? `<a href="${{outUrl("kerr_bending_vs_impact_parameter")}}" target="_blank">Kerr bending vs impact parameter<br><code>${{outPath("kerr_bending_vs_impact_parameter")}}</code></a>` : ""}}
    ${{state.outputs.stop_condition_distribution_exists ? `<a href="${{outUrl("stop_condition_distribution")}}" target="_blank">Stop condition distribution<br><code>${{outPath("stop_condition_distribution")}}</code></a>` : ""}}
    ${{state.outputs.geodesic_density_map_exists ? `<a href="${{outUrl("geodesic_density_map")}}" target="_blank">Geodesic density map<br><code>${{outPath("geodesic_density_map")}}</code></a>` : ""}}
    ${{state.outputs.forward_geodesics_diagnostics_report_exists ? `<a href="${{outUrl("forward_geodesics_diagnostics_report")}}" target="_blank">Diagnostics JSON report<br><code>${{outPath("forward_geodesics_diagnostics_report")}}</code></a>` : ""}}
  </div></section>` : `<section><h2>Diagnostics</h2><p class="note">Diagnostics will be generated automatically with forward geodesic propagation.</p></section>`;
  const stops = summary && summary.stop_condition_counts ? Object.entries(summary.stop_condition_counts).map(([k, v]) => `${{k}}=${{v}}`).join(", ") : "none";
  const summaryHtml = summary ? `${{inputStatus}}<div class="summary-grid">
    <div class="summary-item"><strong>Status</strong>${{summary.status}}</div>
    <div class="summary-item"><strong>Requested paths</strong>${{summary.n_samples_requested}}</div>
    <div class="summary-item"><strong>Propagated paths</strong>${{summary.n_paths}}</div>
    <div class="summary-item"><strong>Available UHE samples</strong>${{summary.n_input_samples}}</div>
    <div class="summary-item"><strong>Trajectories propagated</strong>${{summary.n_samples_propagated}}</div>
    <div class="summary-item"><strong>Paths</strong>${{summary.n_paths}}</div>
    <div class="summary-item"><strong>Segments</strong>${{summary.n_segments}}</div>
    <div class="summary-item"><strong>Backend</strong>${{summary.geodesic_backend}}</div>
    <div class="summary-item"><strong>Delta theta max</strong>${{Number(summary.max_delta_theta_rad || 0).toExponential(4)}}</div>
    <div class="summary-item"><strong>Delta phi max</strong>${{Number(summary.max_delta_phi_rad || 0).toExponential(4)}}</div>
    <div class="summary-item"><strong>Full Kerr?</strong>${{summary.full_kerr_geodesic}}</div>
    <div class="summary-item"><strong>Null max</strong>${{Number(summary.null_norm_max).toExponential(4)}}</div>
    <div class="summary-item"><strong>Killing E error</strong>${{Number(summary.killing_energy_max_error).toExponential(4)}}</div>
    <div class="summary-item"><strong>Lz error</strong>${{Number(summary.lz_max_error).toExponential(4)}}</div>
    <div class="summary-item"><strong>Validation</strong>${{summary.validation_pass}}</div>
    <div class="summary-item"><strong>Momentum</strong>${{summary.momentum_generator}}</div>
    <div class="summary-item"><strong>Kerr physical?</strong>${{summary.momentum_is_physical_kerr}}</div>
    <div class="summary-item"><strong>Backend language</strong>${{summary.backend_language || "pending"}}</div>
    <div class="summary-item"><strong>Backend kind</strong>${{summary.backend_kind || "pending"}}</div>
    <div class="summary-item"><strong>Backend executable</strong><code>${{summary.backend_executable || "pending"}}</code></div>
    <div class="summary-item"><strong>Runtime ../HADROS</strong>${{String(summary.uses_hadros_original_runtime_path)}}</div>
    <div class="summary-item"><strong>Hamiltonian</strong>${{String(summary.uses_hamiltonian)}}</div>
    <div class="summary-item"><strong>ZAMO tetrad</strong>${{String(summary.uses_zamo_tetrad)}}</div>
    <div class="summary-item"><strong>Python prototype used</strong>${{String(summary.python_prototype_used)}}</div>
  </div><p class="note"><strong>Stop conditions:</strong> ${{stops}}</p>${{forwardLinks}}${{diagnosticsHtml}}` : `${{inputStatus}}<p class="note">Forward geodesics inactive. Generate UHE Source samples first, then propagate here.</p>${{diagnosticsHtml}}`;
  return `<div class="source-panel">
    <button type="button" id="forward-geodesics-button" class="source-action">Propagate Forward Geodesics</button>
    ${{summaryHtml}}
  </div>`;
}}
function renderDisPanel() {{
  const summary = state.dis_summary;
  const status = state.dis_interaction_sampler || {{}};
  const disFields = Object.fromEntries(state.schema.flatMap(tab => tab.fields).filter(f => f.section === "dis_interaction_sampler").map(f => [f.key, f]));
  const disValue = key => state.values.dis_interaction_sampler[key];
  const disInput = key => `<label><span>${{disFields[key].label}}</span>${{inputFor(disFields[key], disValue(key))}}</label>`;
  const sourceFound = Boolean(status.input_uhe_source_found || state.outputs.uhe_source_samples_exists);
  const forwardFound = Boolean(status.input_forward_geodesics_found || (state.outputs.forward_paths_exists && state.outputs.forward_path_segments_exists));
  const canRun = sourceFound && forwardFound;
  const pathsAvailable = status.n_paths_available || (state.forward_summary ? state.forward_summary.n_paths : 0) || 0;
  const segmentsAvailable = status.n_segments_available || (state.forward_summary ? state.forward_summary.n_segments : 0) || 0;
  const sigmaEnergyMin = summary ? (summary.sigma_table_energy_min_gev ?? summary.sigma_energy_min_gev) : null;
  const sigmaEnergyMax = summary ? (summary.sigma_table_energy_max_gev ?? summary.sigma_energy_max_gev) : null;
  const inputHtml = `<section><h2>Inputs</h2><div class="summary-grid">
    <div class="summary-item"><strong>UHE Source found</strong><span class="${{sourceFound ? "ok" : "pending"}}">${{sourceFound ? "found" : "missing"}}</span></div>
    <div class="summary-item"><strong>Forward Geodesics found</strong><span class="${{forwardFound ? "ok" : "pending"}}">${{forwardFound ? "found" : "missing"}}</span></div>
    <div class="summary-item"><strong>Number of trajectories</strong>${{pathsAvailable}}</div>
    <div class="summary-item"><strong>Number of path segments</strong>${{segmentsAvailable}}</div>
    <div class="summary-item"><strong>Current run</strong>${{safeRunName(state.values.run.run_name)}}</div>
  </div></section>`;
  const configHtml = `<section><h2>Configuration</h2>
    <div class="camera-controls-card"><h3>Medium</h3>
      ${{disInput("medium_model")}}
      ${{disInput("medium_velocity_model")}}
      ${{disInput("density_floor_g_cm3")}}
    </div>
    <div class="camera-controls-card"><h3>DIS</h3>
      ${{disInput("dis_model")}}
      ${{disInput("interaction_sampling_mode")}}
      ${{disInput("max_interactions")}}
      ${{disInput("random_seed")}}
    </div>
  </section>`;
  const resultsHtml = summary ? `<section><h2>Results</h2><div class="summary-grid">
    <div class="summary-item"><strong>Paths processed</strong>${{summary.n_paths_processed}}</div>
    <div class="summary-item"><strong>Segments processed</strong>${{summary.n_segments_processed}}</div>
    <div class="summary-item"><strong>Tau minimum</strong>${{Number(summary.tau_min || 0).toExponential(4)}}</div>
    <div class="summary-item"><strong>Tau mean</strong>${{Number(summary.tau_mean || 0).toExponential(4)}}</div>
    <div class="summary-item"><strong>Tau maximum</strong>${{Number(summary.tau_max || 0).toExponential(4)}}</div>
    <div class="summary-item"><strong>Accepted interactions</strong>${{summary.n_interactions_accepted}}</div>
    <div class="summary-item"><strong>Acceptance fraction</strong>${{Number(summary.acceptance_fraction || 0).toExponential(4)}}</div>
    <div class="summary-item"><strong>Maximum density</strong>${{Number(summary.max_density_g_cm3 || 0).toExponential(4)}}</div>
    <div class="summary-item"><strong>Maximum sigma</strong>${{Number(summary.max_sigma_cm2 || 0).toExponential(4)}}</div>
    <div class="summary-item"><strong>Maximum d_tau</strong>${{Number(summary.max_d_tau || 0).toExponential(4)}}</div>
    <div class="summary-item"><strong>DIS backend</strong><code>${{summary.dis_backend || "pending"}}</code></div>
    <div class="summary-item"><strong>Backend language</strong>${{summary.backend_language || "pending"}}</div>
    <div class="summary-item"><strong>Backend executable</strong><code>${{summary.backend_executable || "pending"}}</code></div>
    <div class="summary-item"><strong>Backend kind</strong><code>${{summary.backend_kind || "pending"}}</code></div>
    <div class="summary-item"><strong>C++ backend used</strong>${{String(summary.cpp_backend_used)}}</div>
    <div class="summary-item"><strong>Python prototype used</strong>${{String(summary.python_prototype_used)}}</div>
    <div class="summary-item"><strong>uses_hadros_original_runtime_path</strong>${{String(summary.uses_hadros_original_runtime_path)}}</div>
    <div class="summary-item"><strong>sigma_table_path</strong><code>${{summary.sigma_table_path || "pending"}}</code></div>
    <div class="summary-item"><strong>sigma_table_rows</strong>${{summary.sigma_table_rows ?? "pending"}}</div>
    <div class="summary-item"><strong>sigma_table_is_compact_builtin_adapter</strong>${{String(summary.sigma_table_is_compact_builtin_adapter)}}</div>
    <div class="summary-item"><strong>sigma_table_physics_risk</strong>${{String(summary.sigma_table_physics_risk)}}</div>
    <div class="summary-item"><strong>sigma_table_energy_min_gev</strong>${{sigmaEnergyMin !== null && sigmaEnergyMin !== undefined ? Number(sigmaEnergyMin).toExponential(4) : "pending"}}</div>
    <div class="summary-item"><strong>sigma_table_energy_max_gev</strong>${{sigmaEnergyMax !== null && sigmaEnergyMax !== undefined ? Number(sigmaEnergyMax).toExponential(4) : "pending"}}</div>
  </div></section>` : `<section><h2>Results</h2><p class="note">No DIS optical-depth results yet.</p></section>`;
  const outputLinks = `<section><h2>Outputs</h2><div class="output-link-grid">
    ${{state.outputs.dis_tau_preview_exists ? `<a href="${{outUrl("dis_tau_preview")}}" target="_blank">Tau preview PNG<br><code>${{outPath("dis_tau_preview")}}</code></a>` : ""}}
    ${{state.outputs.dis_interaction_locations_exists ? `<a href="${{outUrl("dis_interaction_locations")}}" target="_blank">Interaction locations PNG<br><code>${{outPath("dis_interaction_locations")}}</code></a>` : ""}}
    ${{state.outputs.dis_interaction_locations_3d_html_exists ? `<a href="${{outUrl("dis_interaction_locations_3d_html")}}" target="_blank">Interaction locations HTML<br><code>${{outPath("dis_interaction_locations_3d_html")}}</code></a>` : ""}}
    ${{state.outputs.dis_summary_json_exists ? `<a href="${{outUrl("dis_summary_json")}}" target="_blank">Summary JSON<br><code>${{outPath("dis_summary_json")}}</code></a>` : ""}}
    ${{state.outputs.dis_summary_exists ? `<a href="${{outUrl("dis_summary")}}" target="_blank">Summary CSV<br><code>${{outPath("dis_summary")}}</code></a>` : ""}}
    ${{state.outputs.dis_path_optical_depths_exists ? `<a href="${{outUrl("dis_path_optical_depths")}}" target="_blank">Path optical depths<br><code>${{outPath("dis_path_optical_depths")}}</code></a>` : ""}}
    ${{state.outputs.dis_interaction_candidates_exists ? `<a href="${{outUrl("dis_interaction_candidates")}}" target="_blank">Interaction candidates<br><code>${{outPath("dis_interaction_candidates")}}</code></a>` : ""}}
    ${{state.outputs.dis_accepted_interactions_exists ? `<a href="${{outUrl("dis_accepted_interactions")}}" target="_blank">Accepted interactions<br><code>${{outPath("dis_accepted_interactions")}}</code></a>` : ""}}
    ${{state.outputs.dis_optical_depth_report_exists ? `<a href="${{outUrl("dis_optical_depth_report")}}" target="_blank">Optical-depth report<br><code>${{outPath("dis_optical_depth_report")}}</code></a>` : ""}}
  </div></section>`;
  return `<div class="source-panel">
    ${{inputHtml}}
    ${{configHtml}}
    <section><h2>Run</h2><button type="button" id="dis-sampler-button" class="source-action" ${{canRun ? "" : "disabled"}}>Compute DIS Optical Depth / Sample Interactions</button>
    <button type="button" id="dis-compare-button" class="source-action" ${{canRun ? "" : "disabled"}}>Compare GBW vs IIM</button>
    <p class="note">Runs only H3-W7. Observer Bridge, POWHEG, PYTHIA, GEANT4 and photon transport remain disabled.</p>
    <p class="note">The analytic torus has a hard radial cut and a Gaussian angular profile; the opening angle is a width parameter, not a hard boundary.</p></section>
    ${{resultsHtml}}
    ${{outputLinks}}
  </div>`;
}}
function renderObserverBridgePanel() {{
  const summary = state.observer_bridge_summary;
  const status = state.observer_bridge || {{}};
  const fields = Object.fromEntries(state.schema.flatMap(tab => tab.fields).filter(f => f.section === "observer_bridge").map(f => [f.key, f]));
  const value = key => state.values.observer_bridge[key];
  const input = key => `<label><span class="field-label"><span>${{fields[key].label}}</span>${{fields[key].help ? `<span class="field-help">${{fields[key].help}}</span>` : ""}}</span>${{inputFor(fields[key], value(key))}}</label>`;
  const disFound = Boolean(status.input_dis_found || state.outputs.dis_accepted_interactions_exists);
  const incompleteBridge = Boolean(status.observer_bridge_partial_state_detected || (summary && summary.observer_bridge_stage_complete === false));
  const missingBridgeProducts = status.required_observer_bridge_products_missing || (summary && summary.required_observer_bridge_products_missing) || [];
  const incompleteHtml = incompleteBridge ? `<section class="warning-card"><h2>Observer Bridge output is incomplete</h2><p>Re-run Observer Bridge. The dashboard will not treat this stage as complete until the mandatory overlay products exist.</p><p><strong>Missing required products:</strong> <code>${{missingBridgeProducts.length ? missingBridgeProducts.join(", ") : "unknown"}}</code></p></section>` : "";
  const configHtml = `<section><h2>Configuration</h2>
    ${{input("observer_bridge_backend")}}
    ${{input("bridge_mode")}}
    ${{input("secondary_particle_proxy_model")}}
    ${{input("escape_proxy_model")}}
    ${{input("visibility_model")}}
    ${{input("fov_policy")}}
    ${{input("distance_weight_enabled")}}
    ${{input("redshift_weight_enabled")}}
    ${{input("line_of_sight_check_enabled")}}
    ${{input("max_ranked_events")}}
    ${{input("min_observer_weight")}}
    ${{input("min_final_observation_score")}}
    <section><h2>Downstream Candidate Selection</h2>
      ${{input("downstream_selection_policy")}}
      ${{input("downstream_top_n_candidates")}}
      ${{input("downstream_min_final_observation_score")}}
      <p class="note">Observer Bridge selects the ranked candidates that downstream stages consume. POWHEG uses this selected list directly and does not apply its own ranking policy.</p>
    </section>
    ${{input("candidate_overlay_mapping")}}
    ${{input("kerr_pixel_match_resolution_x")}}
    ${{input("kerr_pixel_match_resolution_y")}}
    ${{input("kerr_pixel_match_tolerance_rg")}}
    ${{input("kerr_pixel_match_refine_enabled")}}
    ${{input("interactive_max_candidates")}}
    ${{input("interactive_max_rays")}}
    ${{input("interactive_ray_stride")}}
    ${{input("interactive_candidate_color_mode")}}
  </section>`;
  const inputHtml = `<section><h2>Inputs</h2><div class="summary-grid">
    <div class="summary-item"><strong>DIS accepted interactions</strong><span class="${{disFound ? "ok" : "pending"}}">${{disFound ? "found" : "missing"}}</span></div>
    <div class="summary-item"><strong>Input file</strong><code>${{outPath("dis_accepted_interactions")}}</code></div>
    <div class="summary-item"><strong>Mode</strong><code>scoring_only</code></div>
    <div class="summary-item"><strong>Runtime ../HADROS</strong>false</div>
  </div></section>`;
  const resultsHtml = summary ? `<section><h2>Results</h2><div class="summary-grid">
    <div class="summary-item"><strong>n_interactions_input</strong>${{summary.n_interactions_input}}</div>
    <div class="summary-item"><strong>n_candidates_scored</strong>${{summary.n_candidates_scored}}</div>
    <div class="summary-item"><strong>n_inside_fov</strong>${{summary.n_inside_fov}}</div>
    <div class="summary-item"><strong>n_visible_proxy</strong>${{summary.n_visible_proxy}}</div>
    <div class="summary-item"><strong>score_min</strong>${{Number(summary.score_min || 0).toExponential(4)}}</div>
    <div class="summary-item"><strong>score_mean</strong>${{Number(summary.score_mean || 0).toExponential(4)}}</div>
    <div class="summary-item"><strong>score_max</strong>${{Number(summary.score_max || 0).toExponential(4)}}</div>
    <div class="summary-item"><strong>top_event_id</strong><code>${{summary.top_event_id || "none"}}</code></div>
    <div class="summary-item"><strong>Ranked candidates</strong>${{summary.downstream_n_candidates_ranked ?? summary.n_candidates_scored ?? 0}}</div>
    <div class="summary-item"><strong>Selected for downstream</strong>${{summary.downstream_n_candidates_selected ?? 0}}</div>
    <div class="summary-item"><strong>Selection policy</strong><code>${{summary.downstream_selection_policy || "pending"}}</code></div>
    <div class="summary-item"><strong>Downstream target</strong><code>${{summary.downstream_stage_target || "powheg"}}</code></div>
    <div class="summary-item"><strong>physics_weight</strong><code>${{summary.physics_weight_definition}}</code></div>
    <div class="summary-item"><strong>observer_weight</strong><code>${{summary.observer_weight_definition}}</code></div>
    <div class="summary-item"><strong>final score</strong><code>${{summary.final_observation_score_definition}}</code></div>
    <div class="summary-item"><strong>camera view plotted</strong>${{summary.camera_view_candidates_plotted || 0}}</div>
    <div class="summary-item"><strong>camera view inside FOV</strong>${{summary.camera_view_candidates_inside_fov || 0}}</div>
    <div class="summary-item"><strong>camera view top N</strong>${{summary.camera_view_top_n || 0}}</div>
    <div class="summary-item"><strong>camera projection</strong><code>${{summary.camera_view_projection_model || "geometric_pinhole_proxy"}}</code></div>
    <div class="summary-item"><strong>overlay background</strong><code>${{summary.camera_overlay_background_source || "pending"}}</code></div>
    <div class="summary-item"><strong>overlay resolution</strong><code>${{summary.camera_overlay_resolution_px || "pending"}}</code></div>
    <div class="summary-item"><strong>overlay plotted</strong>${{summary.camera_overlay_candidates_plotted || 0}}</div>
    <div class="summary-item"><strong>matched candidates</strong>${{summary.kerr_pixel_match_n_matched ?? 0}}</div>
    <div class="summary-item"><strong>unmatched candidates</strong>${{summary.kerr_pixel_match_n_unmatched ?? 0}}</div>
    <div class="summary-item"><strong>match tolerance</strong><code>${{summary.kerr_pixel_match_tolerance_rg ?? "pending"}} rg</code></div>
    <div class="summary-item"><strong>mean closest approach</strong><code>${{summary.kerr_pixel_match_mean_closest_approach_rg == null ? "pending" : Number(summary.kerr_pixel_match_mean_closest_approach_rg).toFixed(3) + " rg"}}</code></div>
    <div class="summary-item"><strong>max closest approach</strong><code>${{summary.kerr_pixel_match_max_closest_approach_rg == null ? "pending" : Number(summary.kerr_pixel_match_max_closest_approach_rg).toFixed(3) + " rg"}}</code></div>
    <div class="summary-item"><strong>overlay projection</strong><code>${{summary.candidate_overlay_projection_model || "geometric_pinhole_proxy"}}</code></div>
    <div class="summary-item"><strong>overlay alignment</strong><code>${{summary.candidate_overlay_alignment || "camera_preview_pixel_plane"}}</code></div>
    <div class="summary-item"><strong>interactive view</strong><code>${{summary.observer_bridge_kerr_interactive_view_generated ? "generated" : "pending"}}</code></div>
    <div class="summary-item"><strong>interactive rays</strong>${{summary.interactive_rays_displayed ?? 0}}</div>
    <div class="summary-item"><strong>interactive diagnostic</strong>${{String(summary.interactive_view_diagnostic_only ?? true)}}</div>
    <div class="summary-item"><strong>proxy_physics_risk</strong>${{String(summary.proxy_physics_risk)}}</div>
  </div></section>` : `<section><h2>Results</h2><p class="note">No Observer Bridge scores yet.</p></section>`;
  const links = `<section><h2>Outputs</h2><div class="output-link-grid">
    ${{state.outputs.observer_bridge_candidates_exists ? `<a href="${{outUrl("observer_bridge_candidates")}}" target="_blank">Candidates JSONL<br><code>${{outPath("observer_bridge_candidates")}}</code></a>` : ""}}
    ${{state.outputs.observer_bridge_ranked_events_exists ? `<a href="${{outUrl("observer_bridge_ranked_events")}}" target="_blank">Ranked events JSONL<br><code>${{outPath("observer_bridge_ranked_events")}}</code></a>` : ""}}
    ${{state.outputs.observer_bridge_selected_candidates_exists ? `<a href="${{outUrl("observer_bridge_selected_candidates")}}" target="_blank">Selected candidates JSONL<br><code>${{outPath("observer_bridge_selected_candidates")}}</code></a>` : ""}}
    ${{state.outputs.observer_bridge_selection_summary_exists ? `<a href="${{outUrl("observer_bridge_selection_summary")}}" target="_blank">Selection summary JSON<br><code>${{outPath("observer_bridge_selection_summary")}}</code></a>` : ""}}
    ${{state.outputs.observer_bridge_summary_json_exists ? `<a href="${{outUrl("observer_bridge_summary_json")}}" target="_blank">Summary JSON<br><code>${{outPath("observer_bridge_summary_json")}}</code></a>` : ""}}
    ${{state.outputs.observer_bridge_summary_exists ? `<a href="${{outUrl("observer_bridge_summary")}}" target="_blank">Summary CSV<br><code>${{outPath("observer_bridge_summary")}}</code></a>` : ""}}
    ${{state.outputs.observer_bridge_report_exists ? `<a href="${{outUrl("observer_bridge_report")}}" target="_blank">Report JSON<br><code>${{outPath("observer_bridge_report")}}</code></a>` : ""}}
  </div></section>`;
  return `<div class="source-panel">
    ${{inputHtml}}
    ${{incompleteHtml}}
    ${{configHtml}}
    <section><h2>Run</h2><button type="button" id="observer-bridge-button" class="source-action" ${{disFound ? "" : "disabled"}}>Compute Observer Bridge Scores</button>
    <p class="note">H3-W8 is scoring-only. Every accepted DIS interaction becomes a candidate; camera/FOV visibility changes observer weights and ranking but does not delete candidates.</p>
    <p class="note">POWHEG, PYTHIA, GEANT4, photon transport and event generation remain disabled.</p></section>
    ${{resultsHtml}}
    ${{links}}
  </div>`;
}}
function renderObserverImageBranchesPanel() {{
  const summary = state.observer_image_branches.summary;
  const status = state.observer_image_branches || {{}};
  const fields = Object.fromEntries(state.schema.flatMap(tab => tab.fields).filter(f => f.section === "observer_image_branches").map(f => [f.key, f]));
  const value = key => state.values.observer_image_branches[key];
  const input = key => `<label><span class="field-label"><span>${{fields[key].label}}</span>${{fields[key].help ? `<span class="field-help">${{fields[key].help}}</span>` : ""}}</span>${{inputFor(fields[key], value(key))}}</label>`;
  const canRun = Boolean(status.input_selected_candidates_found && status.input_kerr_pixel_map_found);
  const resultsHtml = summary ? `<section><h2>Results</h2><div class="summary-grid">
    <div class="summary-item"><strong>Candidate count</strong>${{summary.n_candidates || 0}}</div>
    <div class="summary-item"><strong>Total image branches</strong>${{summary.n_branches || 0}}</div>
    <div class="summary-item"><strong>Mean branches/candidate</strong>${{Number(summary.mean_branches_per_candidate || 0).toFixed(3)}}</div>
    <div class="summary-item"><strong>Fraction with multiple images</strong>${{Number(summary.fraction_multiple_images || 0).toFixed(3)}}</div>
    <div class="summary-item"><strong>Single image candidates</strong>${{summary.n_single_image || 0}}</div>
    <div class="summary-item"><strong>Double image candidates</strong>${{summary.n_double_image || 0}}</div>
    <div class="summary-item"><strong>Maximum branches/candidate</strong>${{summary.maximum_branches_per_candidate || 0}}</div>
    <div class="summary-item"><strong>Branch scoring</strong><code>${{summary.branch_scoring_model || "pending"}}</code></div>
    <div class="summary-item"><strong>Primary branch selection</strong><code>${{summary.primary_branch_selection_model || "pending"}}</code></div>
    <div class="summary-item"><strong>Proxy selection</strong>${{String(summary.primary_branch_selection_proxy ?? true)}}</div>
  </div></section>` : `<section><h2>Results</h2><p class="note">No Observer Image Branch Analysis products yet.</p></section>`;
  const diagnostics = `<section><h2>Diagnostics</h2><div class="diagnostic-grid">
    ${{state.outputs.observer_branch_cluster_map_exists ? `<figure class="diagnostic-plot-card"><figcaption>Branch cluster map</figcaption><a href="${{outUrl("observer_branch_cluster_map")}}" target="_blank"><img src="${{outUrl("observer_branch_cluster_map")}}?v=${{observerBridgePreviewVersion}}" alt="Observer image branch cluster map"></a></figure>` : ""}}
    ${{state.outputs.observer_branch_score_distribution_exists ? `<figure class="diagnostic-plot-card"><figcaption>Branch score distribution</figcaption><a href="${{outUrl("observer_branch_score_distribution")}}" target="_blank"><img src="${{outUrl("observer_branch_score_distribution")}}?v=${{observerBridgePreviewVersion}}" alt="Observer branch score distribution"></a></figure>` : ""}}
    ${{state.outputs.observer_branch_primary_vs_secondary_exists ? `<figure class="diagnostic-plot-card"><figcaption>Branches per candidate</figcaption><a href="${{outUrl("observer_branch_primary_vs_secondary")}}" target="_blank"><img src="${{outUrl("observer_branch_primary_vs_secondary")}}?v=${{observerBridgePreviewVersion}}" alt="Observer primary versus secondary branches"></a></figure>` : ""}}
    ${{state.outputs.observer_viewpoint_convention_diagnostic_exists ? `<figure class="diagnostic-plot-card"><figcaption>Viewpoint convention audit</figcaption><a href="${{outUrl("observer_viewpoint_convention_diagnostic")}}" target="_blank"><img src="${{outUrl("observer_viewpoint_convention_diagnostic")}}?v=${{observerBridgePreviewVersion}}" alt="Observer viewpoint convention diagnostic"></a></figure>` : ""}}
    ${{state.outputs.observer_branch_view_exists ? `<figure class="diagnostic-plot-card"><figcaption>Observer Branch View</figcaption><a href="${{outUrl("observer_branch_view")}}" target="_blank">Open interactive branch viewer</a><iframe class="context-interactive" src="${{outUrl("observer_branch_view")}}?v=${{observerBridgePreviewVersion}}" title="Observer branch viewer"></iframe></figure>` : ""}}
  </div></section>`;
  const inputsHtml = `<section><h2>Inputs</h2><div class="summary-grid">
      <div class="summary-item"><strong>Selected candidates</strong><span class="${{status.input_selected_candidates_found ? "ok" : "pending"}}">${{status.input_selected_candidates_found ? "found" : "missing"}}</span></div>
      <div class="summary-item"><strong>Kerr pixel map</strong><span class="${{status.input_kerr_pixel_map_found ? "ok" : "pending"}}">${{status.input_kerr_pixel_map_found ? "found" : "missing"}}</span></div>
      <div class="summary-item"><strong>Output directory</strong><code>${{status.output_dir || "ObserverImageBranches/"}}</code></div>
    </div></section>
    <section><h2>Configuration</h2>
      ${{input("branch_scoring_model")}}
      ${{input("primary_branch_selection_model")}}
      ${{input("minimum_branch_rays")}}
    </section>
    <section><h2>Run</h2><button type="button" id="observer-image-branches-button" class="source-action" ${{canRun ? "" : "disabled"}}>Analyze Observer Image Branches</button>
      <p class="note">H3-W8b groups Kerr ray matches into observed image branches, scores each branch with an auditable proxy, and writes the primary observed branch consumed by POWHEG.</p>
    </section>`;
  return `<div class="source-panel observer-image-branches-layout">
    <div class="observer-image-branches-main">
      ${{inputsHtml}}
      ${{resultsHtml}}
    </div>
    <div class="observer-image-branches-diagnostics">
      ${{diagnostics}}
    </div>
  </div>`;
}}
function renderPowhegPanel() {{
  const summary = state.powheg_summary;
  const status = state.powheg || {{}};
  const fields = Object.fromEntries(state.schema.flatMap(tab => tab.fields).filter(f => f.section === "powheg").map(f => [f.key, f]));
  const value = key => state.values.powheg[key];
  const input = key => `<label><span class="field-label"><span>${{fields[key].label}}</span>${{fields[key].help ? `<span class="field-help">${{fields[key].help}}</span>` : ""}}</span>${{inputFor(fields[key], value(key))}}</label>`;
  const realSmokeNotice = value("run_mode") === "real_smoke"
    ? `<p class="note"><strong>Real smoke safety mode:</strong> only the first primary image branch is executed, with at most 2 POWHEG events. This intentional limit applies only to real_smoke; dry_run and real_free use the primary branch list prepared by Observer Image Branch Analysis.</p>`
    : "";
  const realFreeNotice = value("run_mode") === "real_free"
    ? `<p class="note"><strong>Real free mode may be computationally expensive.</strong> It will execute pwhg_main for every primary Observer Image Branch and the configured POWHEG events per interaction.</p>`
    : "";
  const bridgeFound = Boolean(status.input_observer_image_branches_found || state.outputs.observer_image_primary_branches_exists);
  const selectedAvailable = summary ? (summary.powheg_n_selected_candidates_input ?? summary.n_candidates_input ?? 0) : 0;
  const eventsPerInteraction = summary ? (summary.events_per_candidate_requested || summary.events_per_candidate || value("events_per_candidate") || 0) : (value("events_per_candidate") || 0);
  const jobsToPrepare = summary ? (summary.powheg_jobs_prepared || summary.n_powheg_jobs || selectedAvailable) : selectedAvailable;
  const inputHtml = `<section><h2>Inputs</h2><div class="summary-grid">
    <div class="summary-item"><strong>Primary Observer Image Branches found</strong><span class="${{bridgeFound ? "ok" : "pending"}}">${{bridgeFound ? "yes" : "no"}}</span></div>
    <div class="summary-item"><strong>Input file</strong><code>${{outPath("observer_image_primary_branches")}}</code></div>
    <div class="summary-item"><strong>Selected candidates available</strong>${{selectedAvailable}}</div>
    <div class="summary-item"><strong>POWHEG jobs to prepare/run</strong>${{jobsToPrepare}}</div>
    <div class="summary-item"><strong>POWHEG Events per Interaction</strong>${{eventsPerInteraction}}</div>
    <div class="summary-item"><strong>Nominal requested LHE events</strong>${{Number(jobsToPrepare || 0) * Number(eventsPerInteraction || 0)}}</div>
    <div class="summary-item"><strong>POWHEG backend</strong><code>local_powheg</code></div>
    <div class="summary-item"><strong>POWHEG process</strong><code>nudis</code></div>
    <div class="summary-item"><strong>run_mode</strong><code>${{value("run_mode")}}</code></div>
  </div></section>`;
  const configHtml = `<section><h2>Configuration</h2>
    <div class="camera-controls-card"><h3>POWHEG driver</h3>
      ${{input("powheg_backend")}}
      ${{input("powheg_process")}}
      ${{input("run_mode")}}
      ${{input("events_per_candidate")}}
      ${{input("random_seed")}}
      ${{input("powheg_seed_mode")}}
      ${{realSmokeNotice}}
      ${{realFreeNotice}}
    </div>
  </section>`;
  const generationSummaryHtml = summary ? `<section><h2>Generation Summary</h2><div class="summary-grid">
    <div class="summary-item"><strong>Primary image branches received</strong>${{summary.powheg_n_selected_candidates_input || summary.n_candidates_input || 0}}</div>
    <div class="summary-item"><strong>POWHEG jobs prepared</strong>${{summary.powheg_jobs_prepared || 0}}</div>
    <div class="summary-item"><strong>Input cards generated</strong>${{summary.powheg_cards_generated || 0}}</div>
    <div class="summary-item"><strong>LHE generated</strong><span class="${{summary.powheg_lhe_generated ? "ok" : "pending"}}">${{summary.powheg_lhe_generated ? "YES" : "NO"}}</span></div>
    <div class="summary-item"><strong>LHE events</strong>${{summary.n_lhe_events || 0}}</div>
    <div class="summary-item"><strong>POWHEG jobs run</strong>${{summary.n_powheg_jobs_run || 0}}</div>
    <div class="summary-item"><strong>POWHEG jobs = selected candidates</strong>${{summary.n_powheg_jobs_requested || summary.powheg_n_selected_candidates_input || summary.powheg_jobs_prepared || 0}}</div>
    <div class="summary-item"><strong>Events per interaction requested</strong>${{summary.events_per_candidate_requested || summary.events_per_candidate || 0}}</div>
    <div class="summary-item"><strong>Selection performed by</strong><code>${{summary.powheg_selection_performed_by || "ObserverBridge"}}</code></div>
    <div class="summary-item"><strong>Selection policy</strong><code>${{summary.powheg_selection_policy || "pending"}}</code></div>
    <div class="summary-item"><strong>Run mode</strong><span class="ok">${{summary.powheg_run_mode || "dry_run"}}</span></div>
    <div class="summary-item"><strong>pwhg_main</strong><span class="${{summary.pwhg_main_executed ? "ok" : "pending"}}">${{summary.pwhg_main_executed ? "executed" : "NOT executed"}}</span></div>
    <div class="summary-item"><strong>powheg_invoked</strong>${{String(summary.powheg_invoked)}}</div>
    <div class="summary-item"><strong>Backend language</strong>${{summary.backend_language || "C++17"}}</div>
    <div class="summary-item"><strong>Backend executable</strong><code>${{summary.backend_executable || "bin/hadros3_powheg_driver"}}</code></div>
    <div class="summary-item"><strong>Runtime self-contained</strong>${{String(summary.powheg_runtime_self_contained)}}</div>
  </div></section>` : `<section><h2>Generation Summary</h2><p class="note">No POWHEG jobs prepared yet.</p></section>`;
  const physics = summary && summary.powheg_physics_summary ? summary.powheg_physics_summary : {{}};
  const physicsSummaryHtml = summary ? `<section><h2>Physics Summary</h2>
    <p class="note">These are POWHEG hard-process/LHE particles. They are not hadronized final-state particles. PYTHIA has not been invoked.</p>
    <div class="summary-grid">
      <div class="summary-item"><strong>n_lhe_events</strong>${{summary.n_lhe_events || 0}}</div>
      <div class="summary-item"><strong>n_lhe_particles</strong>${{summary.n_lhe_particles || 0}}</div>
      <div class="summary-item"><strong>n_final_state_particles</strong>${{summary.n_final_state_particles || 0}}</div>
      <div class="summary-item"><strong>unique_particle_types</strong>${{summary.unique_particle_types || 0}}</div>
      <div class="summary-item"><strong>Incoming particles</strong>${{Array.isArray(physics.incoming_particles) ? physics.incoming_particles.join(", ") : "none"}}</div>
      <div class="summary-item"><strong>Outgoing particles</strong>${{Array.isArray(physics.outgoing_particles) ? physics.outgoing_particles.join(", ") : "none"}}</div>
      <div class="summary-item"><strong>Unique species</strong>${{Array.isArray(physics.unique_particle_species) ? physics.unique_particle_species.length : (summary.unique_particle_types || 0)}}</div>
      <div class="summary-item"><strong>Average event energy</strong>${{Number(physics.average_event_energy_gev || 0).toExponential(4)}} GeV</div>
      <div class="summary-item"><strong>Average event weight</strong>${{Number(physics.average_event_weight || 0).toExponential(4)}}</div>
      <div class="summary-item"><strong>Average pT</strong>${{Number(physics.average_pt_gev || 0).toExponential(4)}} GeV</div>
      <div class="summary-item"><strong>Average multiplicity</strong>${{Number(physics.average_multiplicity || 0).toFixed(2)}}</div>
      <div class="summary-item"><strong>Hadronization</strong>${{String(summary.hadronization_invoked || false)}}</div>
    </div>
    ${{state.outputs.powheg_particle_content_report_exists ? `<p><a href="${{outUrl("powheg_particle_content_report")}}" target="_blank">Open particle content report</a></p>` : ""}}
  </section>` : "";
  const particleRows = summary && Array.isArray(summary.powheg_lhe_particle_summary)
    ? summary.powheg_lhe_particle_summary.slice(0, 12).map(row => `<tr><td>${{row.particle_display || ("PDG " + row.pdg_id)}}</td><td>${{row.pdg_id}}</td><td>${{row.count}}</td><td>${{row.initial_state_count}}</td><td>${{row.final_state_count}}</td><td>${{Number(row.mean_energy_gev || 0).toExponential(4)}}</td><td>${{Number(row.max_energy_gev || 0).toExponential(4)}}</td><td>${{Number(row.mean_pt_gev || 0).toExponential(4)}}</td><td>${{Number(row.max_pt_gev || 0).toExponential(4)}}</td></tr>`).join("")
    : "";
  const particleSummaryTableHtml = summary ? `<section><h2>Particle Summary Table</h2>
    ${{particleRows ? `<table class="backend-table"><thead><tr><th>particle_display</th><th>pdg_id</th><th>count</th><th>initial_state_count</th><th>final_state_count</th><th>mean_energy_gev</th><th>max_energy_gev</th><th>mean_pt_gev</th><th>max_pt_gev</th></tr></thead><tbody>${{particleRows}}</tbody></table>` : `<p class="note">No LHE particles parsed yet.</p>`}}
  </section>` : "";
  const eventDisplayHtml = `<section><h2>Hard Process Event Display</h2>
    ${{state.outputs.powheg_hard_process_event_display_exists ? `<figure class="diagnostic-plot-card"><a href="${{outUrl("powheg_hard_process_event_display")}}" target="_blank"><img src="${{outUrl("powheg_hard_process_event_display")}}?v=${{powhegPreviewVersion}}" alt="POWHEG hard process event display"></a></figure>` : `<p class="note">No LHE event display available yet.</p>`}}
    ${{state.outputs.powheg_hard_process_event_display_view_exists ? `<iframe class="context-interactive" src="${{outUrl("powheg_hard_process_event_display_view")}}?v=${{powhegPreviewVersion}}" title="POWHEG hard process event selector"></iframe>` : ""}}
    ${{state.outputs.powheg_event_summary_table_exists ? `<p><a href="${{outUrl("powheg_event_summary_table")}}" target="_blank">Open Event Summary Table CSV</a></p>` : ""}}
  </section>`;
  const particleTableHtml = summary ? `<section><h2>Particle Table</h2>
    <p class="note">Full particle-level table with particle_display, PDG/status, mothers and four-momentum components.</p>
    ${{state.outputs.powheg_particle_table_html_exists ? `<iframe class="context-interactive" src="${{outUrl("powheg_particle_table_html")}}?v=${{powhegPreviewVersion}}" title="POWHEG particle table with momenta and energies"></iframe>` : `<p class="note">No particle table preview available yet.</p>`}}
    ${{state.outputs.powheg_particle_table_exists ? `<p><a href="${{outUrl("powheg_particle_table")}}" target="_blank">Open full Particle Table CSV</a></p>` : ""}}
  </section>` : "";
  const lheViewerHtml = `<section><h2>LHE Viewer</h2>
    ${{state.outputs.powheg_lhe_event_view_exists ? `<iframe class="context-interactive" src="${{outUrl("powheg_lhe_event_view")}}?v=${{powhegPreviewVersion}}" title="Raw LHE event viewer"></iframe>` : `<p class="note">No raw LHE event available. Dry-run mode does not produce LHE.</p>`}}
  </section>`;
  const lheHtml = summary ? `<section><h2>POWHEG LHE Products</h2>
    <p class="note">${{summary.powheg_lhe_message || "No LHE available: POWHEG dry run only."}}</p>
    <div class="summary-grid">
      <div class="summary-item"><strong>n_lhe_events</strong>${{summary.n_lhe_events || 0}}</div>
      <div class="summary-item"><strong>n_lhe_particles</strong>${{summary.n_lhe_particles || 0}}</div>
      <div class="summary-item"><strong>n_final_state_particles</strong>${{summary.n_final_state_particles || 0}}</div>
      <div class="summary-item"><strong>unique_particle_types</strong>${{summary.unique_particle_types || 0}}</div>
      <div class="summary-item"><strong>lhe_parser_invoked</strong>${{String(summary.lhe_parser_invoked || false)}}</div>
      <div class="summary-item"><strong>hadronization_invoked</strong>${{String(summary.hadronization_invoked || false)}}</div>
    </div>
  </section>` : "";
  const diagnosticsHtml = `<section><h2>Diagnostics</h2><div class="diagnostic-card-grid">
    ${{state.outputs.powheg_energy_distribution_exists ? `<figure class="diagnostic-plot-card"><figcaption>Local neutrino energy submitted to POWHEG</figcaption><a href="${{outUrl("powheg_energy_distribution")}}" target="_blank"><img src="${{outUrl("powheg_energy_distribution")}}?v=${{powhegPreviewVersion}}" alt="Local neutrino energy submitted to POWHEG"></a></figure>` : ""}}
    ${{state.outputs.powheg_lhe_particle_histogram_exists ? `<figure class="diagnostic-plot-card"><figcaption>Particle content by state/category</figcaption><a href="${{outUrl("powheg_lhe_particle_histogram")}}" target="_blank"><img src="${{outUrl("powheg_lhe_particle_histogram")}}?v=${{powhegPreviewVersion}}" alt="POWHEG LHE particle histogram"></a></figure>` : ""}}
    ${{state.outputs.powheg_lhe_energy_spectrum_exists ? `<figure class="diagnostic-plot-card"><figcaption>LHE energy spectrum</figcaption><a href="${{outUrl("powheg_lhe_energy_spectrum")}}" target="_blank"><img src="${{outUrl("powheg_lhe_energy_spectrum")}}?v=${{powhegPreviewVersion}}" alt="POWHEG LHE energy spectrum"></a></figure>` : ""}}
    ${{state.outputs.powheg_lhe_momentum_spectrum_exists ? `<figure class="diagnostic-plot-card"><figcaption>LHE momentum spectra</figcaption><a href="${{outUrl("powheg_lhe_momentum_spectrum")}}" target="_blank"><img src="${{outUrl("powheg_lhe_momentum_spectrum")}}?v=${{powhegPreviewVersion}}" alt="POWHEG LHE momentum spectrum"></a></figure>` : ""}}
    ${{state.outputs.powheg_job_summary_exists ? `<figure class="diagnostic-plot-card"><figcaption>Job summary</figcaption><a href="${{outUrl("powheg_job_summary")}}" target="_blank"><img src="${{outUrl("powheg_job_summary")}}?v=${{powhegPreviewVersion}}" alt="POWHEG job summary"></a></figure>` : ""}}
  </div></section>`;
  const links = `<section><h2>Outputs</h2><div class="output-link-grid">
    ${{state.outputs.powheg_event_requests_exists ? `<a href="${{outUrl("powheg_event_requests")}}" target="_blank">Event requests JSONL<br><code>${{outPath("powheg_event_requests")}}</code></a>` : ""}}
    ${{state.outputs.powheg_summary_json_exists ? `<a href="${{outUrl("powheg_summary_json")}}" target="_blank">Summary JSON<br><code>${{outPath("powheg_summary_json")}}</code></a>` : ""}}
    ${{state.outputs.powheg_summary_exists ? `<a href="${{outUrl("powheg_summary")}}" target="_blank">Summary CSV<br><code>${{outPath("powheg_summary")}}</code></a>` : ""}}
    ${{state.outputs.powheg_report_exists ? `<a href="${{outUrl("powheg_report")}}" target="_blank">Report JSON<br><code>${{outPath("powheg_report")}}</code></a>` : ""}}
    ${{state.outputs.powheg_validation_report_exists ? `<a href="${{outUrl("powheg_validation_report")}}" target="_blank">Validation report<br><code>${{outPath("powheg_validation_report")}}</code></a>` : ""}}
    ${{state.outputs.powheg_lhe_exists ? `<a href="${{outUrl("powheg_lhe")}}" target="_blank">Smoke LHE<br><code>${{outPath("powheg_lhe")}}</code></a>` : ""}}
    ${{state.outputs.powheg_log_exists ? `<a href="${{outUrl("powheg_log")}}" target="_blank">POWHEG log<br><code>${{outPath("powheg_log")}}</code></a>` : ""}}
    ${{state.outputs.powheg_lhe_particles_exists ? `<a href="${{outUrl("powheg_lhe_particles")}}" target="_blank">LHE particles JSONL<br><code>${{outPath("powheg_lhe_particles")}}</code></a>` : ""}}
    ${{state.outputs.powheg_lhe_events_summary_exists ? `<a href="${{outUrl("powheg_lhe_events_summary")}}" target="_blank">LHE events summary JSONL<br><code>${{outPath("powheg_lhe_events_summary")}}</code></a>` : ""}}
    ${{state.outputs.powheg_lhe_particle_summary_csv_exists ? `<a href="${{outUrl("powheg_lhe_particle_summary_csv")}}" target="_blank">LHE particle summary CSV<br><code>${{outPath("powheg_lhe_particle_summary_csv")}}</code></a>` : ""}}
    ${{state.outputs.powheg_lhe_particle_summary_json_exists ? `<a href="${{outUrl("powheg_lhe_particle_summary_json")}}" target="_blank">LHE particle summary JSON<br><code>${{outPath("powheg_lhe_particle_summary_json")}}</code></a>` : ""}}
  </div></section>`;
  return `<div class="source-panel">
    ${{inputHtml}}
    ${{configHtml}}
    <section><h2>Run</h2><button type="button" id="powheg-button" class="source-action" ${{bridgeFound ? "" : "disabled"}}>${{value("run_mode") === "real_smoke" ? "Run POWHEG Real Smoke" : (value("run_mode") === "real_free" ? "Run POWHEG Real Free" : "Prepare POWHEG Jobs")}}</button>
    <p class="note">Dry run prepares requests, deterministic seeds and real POWHEG input cards without executing pwhg_main. Real smoke executes the local pwhg_main for one primary branch and validates a minimal LHE. Real free executes pwhg_main for the configured primary branch and event counts.</p>
    <p class="note">PYTHIA, GEANT4, photon transport and spectra remain disabled.</p></section>
    ${{generationSummaryHtml}}
    ${{physicsSummaryHtml}}
    ${{eventDisplayHtml}}
    ${{particleSummaryTableHtml}}
    ${{particleTableHtml}}
    ${{lheViewerHtml}}
    ${{lheHtml}}
    ${{diagnosticsHtml}}
    ${{links}}
  </div>`;
}}
function renderContextPanel() {{
  const geometryPreviewTabs = new Set(["Camera", "Black Hole", "Torus / Medium", "Funnel / Cone"]);
  if (geometryPreviewTabs.has(activeTab)) {{
    return `<aside class="panel"><h2>Geometry Preview</h2><div class="geometry-preview-large"><svg id="geometrySvg" role="img" aria-label="Dynamic HADROS3 geometry preview"></svg></div></aside>`;
  }}
  if (activeTab === "UHE Source") {{
    const figure = state.outputs.uhe_source_preview_exists
      ? `<img src="${{outUrl("uhe_source_preview")}}?v=${{sourcePreviewVersion}}" alt="UHE source sample preview">`
      : `<div class="context-empty">No UHE source preview generated yet.</div>`;
    const uniformityFigure = state.outputs.uhe_source_sampling_uniformity_exists
      ? `<img src="${{outUrl("uhe_source_sampling_uniformity")}}?v=${{sourcePreviewVersion}}" alt="UHE source sampling uniformity histograms">`
      : `<div class="context-empty">No sampling uniformity diagnostic generated yet.</div>`;
    const directionUniformityFigure = state.outputs.uhe_source_direction_uniformity_exists
      ? `<img src="${{outUrl("uhe_source_direction_uniformity")}}?v=${{sourcePreviewVersion}}" alt="UHE source direction uniformity histograms">`
      : `<div class="context-empty">No direction uniformity diagnostic generated yet.</div>`;
    const directionSphereFigure = state.outputs.uhe_source_direction_sphere_exists
      ? `<img src="${{outUrl("uhe_source_direction_sphere")}}?v=${{sourcePreviewVersion}}" alt="UHE source local direction sphere">`
      : `<div class="context-empty">No local direction sphere generated yet.</div>`;
    return `<aside class="panel"><h2>UHE Source Samples</h2><div class="context-figure">${{figure}}</div><section><h2>Sampling Uniformity</h2><div class="context-figure">${{uniformityFigure}}</div></section><section><h2>Direction Uniformity</h2><div class="context-figure">${{directionUniformityFigure}}</div></section><section><h2>Local Direction Sphere</h2><div class="context-figure">${{directionSphereFigure}}</div></section></aside>`;
  }}
  if (activeTab === "Forward Geodesics") {{
    const figure = state.outputs.forward_geometry_3d_html_exists
      ? `<iframe class="context-interactive" src="${{outUrl("forward_geometry_3d_html")}}?v=${{forwardPreviewVersion}}" title="Interactive forward geodesic 3D geometry"></iframe>`
      : state.outputs.forward_geometry_3d_exists
      ? `<img src="${{outUrl("forward_geometry_3d")}}?v=${{forwardPreviewVersion}}" alt="Forward geodesic 3D geometry">`
      : state.outputs.forward_preview_exists
      ? `<img src="${{outUrl("forward_preview")}}?v=${{forwardPreviewVersion}}" alt="Forward geodesic 2D preview">`
      : `<div class="context-empty">No forward geodesic preview generated yet.</div>`;
    const diagnosticCards = [
      state.outputs.validation_invariants_exists ? `<figure class="diagnostic-plot-card"><figcaption>Invariant conservation</figcaption><a href="${{outUrl("validation_invariants")}}" target="_blank"><img src="${{outUrl("validation_invariants")}}?v=${{forwardPreviewVersion}}" alt="Forward geodesic invariant conservation"></a></figure>` : "",
      state.outputs.kerr_bending_vs_impact_parameter_exists ? `<figure class="diagnostic-plot-card"><figcaption>Kerr bending vs impact parameter</figcaption><a href="${{outUrl("kerr_bending_vs_impact_parameter")}}" target="_blank"><img src="${{outUrl("kerr_bending_vs_impact_parameter")}}?v=${{forwardPreviewVersion}}" alt="Kerr bending vs impact parameter"></a></figure>` : "",
      state.outputs.stop_condition_distribution_exists ? `<figure class="diagnostic-plot-card"><figcaption>Stop condition distribution</figcaption><a href="${{outUrl("stop_condition_distribution")}}" target="_blank"><img src="${{outUrl("stop_condition_distribution")}}?v=${{forwardPreviewVersion}}" alt="Forward geodesic stop condition distribution"></a></figure>` : "",
      state.outputs.geodesic_density_map_exists ? `<figure class="diagnostic-plot-card"><figcaption>Geodesic density map</figcaption><a href="${{outUrl("geodesic_density_map")}}" target="_blank"><img src="${{outUrl("geodesic_density_map")}}?v=${{forwardPreviewVersion}}" alt="Forward geodesic density map"></a></figure>` : "",
    ].filter(Boolean).join("");
    const diagnostics = diagnosticCards ? `<section><h2>Diagnostics</h2><div class="diagnostic-card-grid">${{diagnosticCards}}</div></section>` : "";
    return `<aside class="panel"><h2>Forward Geodesics Geometry</h2><div class="context-figure">${{figure}}</div>${{diagnostics}}</aside>`;
  }}
  if (activeTab === "DIS Interaction Sampler") {{
    const figure = state.outputs.dis_interaction_locations_3d_html_exists
      ? `<iframe class="context-interactive" src="${{outUrl("dis_interaction_locations_3d_html")}}?v=${{disPreviewVersion}}" title="DIS interaction locations"></iframe>`
      : state.outputs.dis_interaction_locations_exists
      ? `<img src="${{outUrl("dis_interaction_locations")}}?v=${{disPreviewVersion}}" alt="DIS interaction locations">`
      : `<div class="context-empty">No DIS interaction map generated yet.</div>`;
    const diagnosticCards = [
      state.outputs.tau_distribution_exists ? `<figure class="diagnostic-plot-card"><figcaption>Tau distribution</figcaption><a href="${{outUrl("tau_distribution")}}" target="_blank"><img src="${{outUrl("tau_distribution")}}?v=${{disPreviewVersion}}" alt="DIS tau distribution"></a></figure>` : "",
      state.outputs.interaction_probability_distribution_exists ? `<figure class="diagnostic-plot-card"><figcaption>Interaction probability distribution</figcaption><a href="${{outUrl("interaction_probability_distribution")}}" target="_blank"><img src="${{outUrl("interaction_probability_distribution")}}?v=${{disPreviewVersion}}" alt="DIS interaction probability distribution"></a></figure>` : "",
      state.outputs.medium_density_map_exists ? `<figure class="diagnostic-plot-card"><figcaption>Medium density map</figcaption><a href="${{outUrl("medium_density_map")}}" target="_blank"><img src="${{outUrl("medium_density_map")}}?v=${{disPreviewVersion}}" alt="DIS medium density map"></a><p class="note">The analytic torus has a hard radial cut and a Gaussian angular profile; the opening angle is a width parameter, not a hard boundary.</p></figure>` : "",
      state.outputs.optical_depth_map_exists ? `<figure class="diagnostic-plot-card"><figcaption>Optical depth map</figcaption><a href="${{outUrl("optical_depth_map")}}" target="_blank"><img src="${{outUrl("optical_depth_map")}}?v=${{disPreviewVersion}}" alt="DIS optical depth map"></a></figure>` : "",
      state.outputs.interaction_location_distribution_exists ? `<figure class="diagnostic-plot-card"><figcaption>Interaction location distribution</figcaption><a href="${{outUrl("interaction_location_distribution")}}" target="_blank"><img src="${{outUrl("interaction_location_distribution")}}?v=${{disPreviewVersion}}" alt="DIS interaction location distribution"></a></figure>` : "",
      state.outputs.local_energy_distribution_exists ? `<figure class="diagnostic-plot-card"><figcaption>Local energy distribution</figcaption><a href="${{outUrl("local_energy_distribution")}}" target="_blank"><img src="${{outUrl("local_energy_distribution")}}?v=${{disPreviewVersion}}" alt="DIS local energy distribution"></a></figure>` : "",
      state.outputs.local_density_distribution_exists ? `<figure class="diagnostic-plot-card"><figcaption>Local density distribution</figcaption><a href="${{outUrl("local_density_distribution")}}" target="_blank"><img src="${{outUrl("local_density_distribution")}}?v=${{disPreviewVersion}}" alt="DIS local density distribution"></a></figure>` : "",
      state.outputs.sigma_distribution_exists ? `<figure class="diagnostic-plot-card"><figcaption>Sigma distribution</figcaption><a href="${{outUrl("sigma_distribution")}}" target="_blank"><img src="${{outUrl("sigma_distribution")}}?v=${{disPreviewVersion}}" alt="DIS sigma distribution"></a></figure>` : "",
      state.outputs.density_energy_sigma_correlation_exists ? `<figure class="diagnostic-plot-card"><figcaption>Density energy sigma correlation</figcaption><a href="${{outUrl("density_energy_sigma_correlation")}}" target="_blank"><img src="${{outUrl("density_energy_sigma_correlation")}}?v=${{disPreviewVersion}}" alt="DIS density energy sigma correlation"></a></figure>` : "",
      state.outputs.gbw_vs_iim_tau_comparison_exists ? `<figure class="diagnostic-plot-card"><figcaption>GBW vs IIM tau</figcaption><a href="${{outUrl("gbw_vs_iim_tau_comparison")}}" target="_blank"><img src="${{outUrl("gbw_vs_iim_tau_comparison")}}?v=${{disPreviewVersion}}" alt="GBW vs IIM tau comparison"></a></figure>` : "",
      state.outputs.gbw_vs_iim_probability_comparison_exists ? `<figure class="diagnostic-plot-card"><figcaption>GBW vs IIM probability</figcaption><a href="${{outUrl("gbw_vs_iim_probability_comparison")}}" target="_blank"><img src="${{outUrl("gbw_vs_iim_probability_comparison")}}?v=${{disPreviewVersion}}" alt="GBW vs IIM probability comparison"></a></figure>` : "",
      state.outputs.gbw_vs_iim_interaction_locations_exists ? `<figure class="diagnostic-plot-card"><figcaption>GBW vs IIM locations</figcaption><a href="${{outUrl("gbw_vs_iim_interaction_locations")}}" target="_blank"><img src="${{outUrl("gbw_vs_iim_interaction_locations")}}?v=${{disPreviewVersion}}" alt="GBW vs IIM interaction locations"></a></figure>` : "",
    ].filter(Boolean).join("");
    const diagnostics = diagnosticCards ? `<section><h2>Diagnostics</h2><div class="diagnostic-card-grid">${{diagnosticCards}}</div></section>` : "";
    return `<aside class="panel"><h2>DIS Interaction Map</h2><div class="context-figure">${{figure}}</div>${{diagnostics}}</aside>`;
  }}
  if (activeTab === "Observer Bridge") {{
    const figure = state.outputs.observer_bridge_kerr_interactive_view_exists
      ? `<a href="${{outUrl("observer_bridge_kerr_interactive_view")}}" target="_blank">Open interactive view</a><iframe class="context-interactive" src="${{outUrl("observer_bridge_kerr_interactive_view")}}?v=${{observerBridgePreviewVersion}}" title="Observer Bridge Kerr Interactive View"></iframe>`
      : state.outputs.observer_bridge_camera_overlay_exists
      ? `<img src="${{outUrl("observer_bridge_camera_overlay")}}?v=${{observerBridgePreviewVersion}}" alt="Observer Bridge camera overlay"><p class="note">Interactive Kerr diagnostic not generated yet; showing the diagnostic overlay fallback.</p>`
      : state.outputs.observer_bridge_camera_view_exists
      ? `<img src="${{outUrl("observer_bridge_camera_view")}}?v=${{observerBridgePreviewVersion}}" alt="Observer Bridge camera view">`
      : `<div class="context-empty">No Observer Bridge diagnostics generated yet.</div>`;
    const diagnosticCards = [
      state.outputs.observer_bridge_camera_overlay_exists ? `<figure class="diagnostic-plot-card"><figcaption>Observer Camera Overlay</figcaption><a href="${{outUrl("observer_bridge_camera_overlay")}}" target="_blank"><img src="${{outUrl("observer_bridge_camera_overlay")}}?v=${{observerBridgePreviewVersion}}" alt="Observer Bridge camera overlay"></a><p class="note">Diagnostic overlay. Candidate image branches may be multiple; use the interactive Kerr view and Observer Image Branches for physical inspection.</p></figure>` : "",
      state.outputs.observer_candidate_multiple_images_exists ? `<figure class="diagnostic-plot-card"><figcaption>Candidate Multi-Image Audit</figcaption><a href="${{outUrl("observer_candidate_multiple_images")}}" target="_blank"><img src="${{outUrl("observer_candidate_multiple_images")}}?v=${{observerBridgePreviewVersion}}" alt="Observer candidate multiple images"></a><p><a href="${{outUrl("observer_candidate_multi_image_view")}}" target="_blank">Open candidate multi-image viewer</a></p></figure>` : "",
      state.outputs.observer_bridge_background_comparison_exists ? `<figure class="diagnostic-plot-card"><figcaption>Overlay background comparison</figcaption><a href="${{outUrl("observer_bridge_background_comparison")}}" target="_blank"><img src="${{outUrl("observer_bridge_background_comparison")}}?v=${{observerBridgePreviewVersion}}" alt="Observer Bridge overlay background comparison"></a></figure>` : "",
      state.outputs.observer_bridge_overlay_hemisphere_diagnostic_exists ? `<figure class="diagnostic-plot-card"><figcaption>Overlay hemisphere diagnostic</figcaption><a href="${{outUrl("observer_bridge_overlay_hemisphere_diagnostic")}}" target="_blank"><img src="${{outUrl("observer_bridge_overlay_hemisphere_diagnostic")}}?v=${{observerBridgePreviewVersion}}" alt="Observer Bridge overlay hemisphere diagnostic"></a></figure>` : "",
      state.outputs.observer_bridge_camera_view_exists ? `<figure class="diagnostic-plot-card"><figcaption>Observer Camera View</figcaption><a href="${{outUrl("observer_bridge_camera_view")}}" target="_blank"><img src="${{outUrl("observer_bridge_camera_view")}}?v=${{observerBridgePreviewVersion}}" alt="Observer Bridge camera view"></a></figure>` : "",
      state.outputs.observer_bridge_map_exists ? `<figure class="diagnostic-plot-card"><figcaption>Candidate map</figcaption><a href="${{outUrl("observer_bridge_map")}}" target="_blank"><img src="${{outUrl("observer_bridge_map")}}?v=${{observerBridgePreviewVersion}}" alt="Observer Bridge map"></a></figure>` : "",
      state.outputs.observer_bridge_score_distribution_exists ? `<figure class="diagnostic-plot-card"><figcaption>Score distribution</figcaption><a href="${{outUrl("observer_bridge_score_distribution")}}" target="_blank"><img src="${{outUrl("observer_bridge_score_distribution")}}?v=${{observerBridgePreviewVersion}}" alt="Observer Bridge score distribution"></a></figure>` : "",
      state.outputs.observer_bridge_weight_breakdown_exists ? `<figure class="diagnostic-plot-card"><figcaption>Weight breakdown</figcaption><a href="${{outUrl("observer_bridge_weight_breakdown")}}" target="_blank"><img src="${{outUrl("observer_bridge_weight_breakdown")}}?v=${{observerBridgePreviewVersion}}" alt="Observer Bridge weight breakdown"></a></figure>` : "",
      state.outputs.observer_bridge_visibility_map_exists ? `<figure class="diagnostic-plot-card"><figcaption>Visibility map</figcaption><a href="${{outUrl("observer_bridge_visibility_map")}}" target="_blank"><img src="${{outUrl("observer_bridge_visibility_map")}}?v=${{observerBridgePreviewVersion}}" alt="Observer Bridge visibility map"></a></figure>` : "",
      state.outputs.observer_bridge_ranked_events_png_exists ? `<figure class="diagnostic-plot-card"><figcaption>Ranked events</figcaption><a href="${{outUrl("observer_bridge_ranked_events_png")}}" target="_blank"><img src="${{outUrl("observer_bridge_ranked_events_png")}}?v=${{observerBridgePreviewVersion}}" alt="Observer Bridge ranked events"></a></figure>` : "",
    ].filter(Boolean).join("");
    const diagnostics = diagnosticCards ? `<section><h2>Diagnostics</h2><div class="diagnostic-card-grid">${{diagnosticCards}}</div></section>` : "";
    return `<aside class="panel"><h2>Observer Bridge Kerr Interactive View</h2><div class="context-figure">${{figure}}</div>${{diagnostics}}</aside>`;
  }}
  if (activeTab === "POWHEG") {{
    const summary = state.powheg_summary || {{}};
    const cardSummary = state.outputs.powheg_event_requests_exists || state.outputs.powheg_card_preview_exists
      ? `<table class="backend-table">
          <tbody>
            <tr><th>Run mode</th><td><code>${{summary.powheg_run_mode || state.values.powheg.run_mode || "dry_run"}}</code></td></tr>
            <tr><th>Selected candidates</th><td>${{summary.powheg_n_selected_candidates_input || summary.n_candidates_input || 0}}</td></tr>
            <tr><th>Jobs prepared</th><td>${{summary.powheg_jobs_prepared || summary.n_powheg_jobs || 0}}</td></tr>
            <tr><th>Events per interaction</th><td>${{summary.events_per_candidate_requested || summary.events_per_candidate || state.values.powheg.events_per_candidate || 0}}</td></tr>
            <tr><th>Selection policy</th><td><code>${{summary.powheg_selection_policy || "Observer Bridge"}}</code></td></tr>
            <tr><th>pwhg_main</th><td>${{summary.pwhg_main_executed ? "executed" : "not executed"}}</td></tr>
            <tr><th>LHE events</th><td>${{summary.n_lhe_events_total || summary.n_lhe_events || 0}}</td></tr>
          </tbody>
        </table>
        ${{state.outputs.powheg_card_preview_exists ? `<p class="note">Raw POWHEG input cards are still written to <code>POWHEG/powheg_input_cards/</code>; open the card preview only when you need the literal card text.</p><a href="${{outUrl("powheg_card_preview")}}" target="_blank">Open raw card preview PNG</a>` : ""}}`
      : `<div class="context-empty">No POWHEG jobs prepared yet.</div>`;
    const diagnosticCards = [
      state.outputs.powheg_energy_distribution_exists ? `<figure class="diagnostic-plot-card"><figcaption>Energy distribution</figcaption><a href="${{outUrl("powheg_energy_distribution")}}" target="_blank"><img src="${{outUrl("powheg_energy_distribution")}}?v=${{powhegPreviewVersion}}" alt="POWHEG energy distribution"></a></figure>` : "",
      state.outputs.powheg_job_summary_exists ? `<figure class="diagnostic-plot-card"><figcaption>Job summary</figcaption><a href="${{outUrl("powheg_job_summary")}}" target="_blank"><img src="${{outUrl("powheg_job_summary")}}?v=${{powhegPreviewVersion}}" alt="POWHEG job summary"></a></figure>` : "",
      state.outputs.powheg_lhe_particle_histogram_exists ? `<figure class="diagnostic-plot-card"><figcaption>LHE particle histogram</figcaption><a href="${{outUrl("powheg_lhe_particle_histogram")}}" target="_blank"><img src="${{outUrl("powheg_lhe_particle_histogram")}}?v=${{powhegPreviewVersion}}" alt="POWHEG LHE particle histogram"></a></figure>` : "",
      state.outputs.powheg_lhe_energy_spectrum_exists ? `<figure class="diagnostic-plot-card"><figcaption>LHE energy spectrum</figcaption><a href="${{outUrl("powheg_lhe_energy_spectrum")}}" target="_blank"><img src="${{outUrl("powheg_lhe_energy_spectrum")}}?v=${{powhegPreviewVersion}}" alt="POWHEG LHE energy spectrum"></a></figure>` : "",
      state.outputs.powheg_lhe_momentum_spectrum_exists ? `<figure class="diagnostic-plot-card"><figcaption>LHE momentum spectrum</figcaption><a href="${{outUrl("powheg_lhe_momentum_spectrum")}}" target="_blank"><img src="${{outUrl("powheg_lhe_momentum_spectrum")}}?v=${{powhegPreviewVersion}}" alt="POWHEG LHE momentum spectrum"></a></figure>` : "",
    ].filter(Boolean).join("");
    const diagnostics = diagnosticCards ? `<section><h2>Diagnostics</h2><div class="diagnostic-card-grid">${{diagnosticCards}}</div></section>` : "";
    return `<aside class="panel"><h2>POWHEG Job Overview</h2><div>${{cardSummary}}</div>${{diagnostics}}</aside>`;
  }}
  return "";
}}
function renderOutputsPanel() {{
  const out = state.outputs;
  const link = (exists, key, label) => exists ? `<a href="${{outUrl(key)}}" target="_blank">${{label}}<br><code>${{outPath(key)}}</code></a>` : `<div class="summary-item"><strong>${{label}}</strong>pending</div>`;
  const group = (title, body) => `<section><h2>${{title}}</h2><div class="output-link-grid">${{body}}</div></section>`;
  return group("RunMetadata/", `
    ${{link(out.config_exists, "config", "Config")}}
    ${{link(out.provenance_exists, "provenance", "Provenance")}}
    ${{link(out.render_summary_exists, "render_summary", "Render summary")}}
  `) + group("Geometry/", `
    ${{link(out.preview_exists, "geometry_preview", "Geometry preview")}}
    ${{link(out.schematic_exists, "system_schematic", "System schematic")}}
  `) + group("CameraPreview/", `
    ${{link(out.camera_preview_exists, "camera_preview", "Camera preview")}}
  `) + group("UHEsource/", `
    ${{link(out.uhe_source_samples_exists, "uhe_source_samples", "UHE source samples")}}
    ${{link(out.uhe_source_summary_exists, "uhe_source_summary", "UHE source summary")}}
    ${{link(out.uhe_source_summary_json_exists, "uhe_source_summary_json", "UHE source summary JSON")}}
    ${{link(out.uhe_source_preview_exists, "uhe_source_preview", "UHE source preview")}}
    ${{link(out.uhe_source_sampling_uniformity_exists, "uhe_source_sampling_uniformity", "UHE source sampling uniformity")}}
    ${{link(out.uhe_source_sampling_uniformity_report_exists, "uhe_source_sampling_uniformity_report", "UHE source sampling uniformity JSON")}}
    ${{link(out.uhe_source_direction_uniformity_exists, "uhe_source_direction_uniformity", "UHE source direction uniformity")}}
    ${{link(out.uhe_source_direction_sphere_exists, "uhe_source_direction_sphere", "UHE source direction sphere")}}
    ${{link(out.uhe_source_direction_uniformity_report_exists, "uhe_source_direction_uniformity_report", "UHE source direction uniformity JSON")}}
    ${{out.uhe_source_preview_exists ? `<img src="${{outUrl("uhe_source_preview")}}" alt="UHE source preview">` : ""}}
    ${{out.uhe_source_sampling_uniformity_exists ? `<img src="${{outUrl("uhe_source_sampling_uniformity")}}" alt="UHE source sampling uniformity">` : ""}}
    ${{out.uhe_source_direction_uniformity_exists ? `<img src="${{outUrl("uhe_source_direction_uniformity")}}" alt="UHE source direction uniformity">` : ""}}
    ${{out.uhe_source_direction_sphere_exists ? `<img src="${{outUrl("uhe_source_direction_sphere")}}" alt="UHE source direction sphere">` : ""}}
  `) + group("ForwardGeodesics/", `
    ${{link(out.forward_geometry_3d_html_exists, "forward_geometry_3d_html", "Interactive forward 3D geometry")}}
    ${{link(out.forward_geometry_3d_exists, "forward_geometry_3d", "Forward 3D geometry")}}
    ${{out.forward_geometry_3d_exists ? `<img src="${{outUrl("forward_geometry_3d")}}" alt="Forward geodesics 3D geometry">` : ""}}
    ${{link(out.forward_geometry_3d_json_exists, "forward_geometry_3d_json", "Forward 3D geometry JSON")}}
    ${{link(out.isotropic_kerr_strong_field_diagnostic_png_exists, "isotropic_kerr_strong_field_diagnostic_png", "Strong-field isotropic diagnostic")}}
    ${{out.isotropic_kerr_strong_field_diagnostic_png_exists ? `<img src="${{outUrl("isotropic_kerr_strong_field_diagnostic_png")}}" alt="Strong-field isotropic Kerr diagnostic">` : ""}}
    ${{link(out.isotropic_kerr_strong_field_diagnostic_json_exists, "isotropic_kerr_strong_field_diagnostic_json", "Strong-field isotropic diagnostic JSON")}}
    ${{link(out.forward_preview_exists, "forward_preview", "Forward preview")}}
    ${{out.forward_preview_exists ? `<img src="${{outUrl("forward_preview")}}" alt="Forward geodesics preview">` : ""}}
    ${{link(out.forward_summary_json_exists, "forward_summary_json", "Forward summary JSON")}}
    ${{link(out.forward_summary_exists, "forward_summary", "Forward summary CSV")}}
    ${{link(out.forward_paths_exists, "forward_paths", "Forward paths")}}
    ${{link(out.forward_path_segments_exists, "forward_path_segments", "Forward segments")}}
    ${{link(out.geodesic_validation_report_exists, "geodesic_validation_report", "Geodesic validation")}}
    ${{link(out.stop_condition_statistics_exists, "stop_condition_statistics", "Stop conditions")}}
    ${{link(out.forward_diagnostic_report_exists, "forward_diagnostic_report", "Forward diagnostic report")}}
    ${{link(out.validation_invariants_exists, "validation_invariants", "Validation invariants")}}
    ${{out.validation_invariants_exists ? `<img src="${{outUrl("validation_invariants")}}" alt="Forward geodesic invariant conservation">` : ""}}
    ${{link(out.kerr_bending_vs_impact_parameter_exists, "kerr_bending_vs_impact_parameter", "Kerr bending vs impact parameter")}}
    ${{out.kerr_bending_vs_impact_parameter_exists ? `<img src="${{outUrl("kerr_bending_vs_impact_parameter")}}" alt="Kerr bending vs impact parameter">` : ""}}
    ${{link(out.stop_condition_distribution_exists, "stop_condition_distribution", "Stop condition distribution")}}
    ${{out.stop_condition_distribution_exists ? `<img src="${{outUrl("stop_condition_distribution")}}" alt="Forward geodesic stop condition distribution">` : ""}}
    ${{link(out.geodesic_density_map_exists, "geodesic_density_map", "Geodesic density map")}}
    ${{out.geodesic_density_map_exists ? `<img src="${{outUrl("geodesic_density_map")}}" alt="Forward geodesic density map">` : ""}}
    ${{link(out.forward_geodesics_diagnostics_report_exists, "forward_geodesics_diagnostics_report", "Forward diagnostics JSON")}}
  `) + group("DIS/", `
    ${{link(out.dis_tau_preview_exists, "dis_tau_preview", "DIS tau preview")}}
    ${{out.dis_tau_preview_exists ? `<img src="${{outUrl("dis_tau_preview")}}" alt="DIS tau preview">` : ""}}
    ${{link(out.dis_interaction_locations_exists, "dis_interaction_locations", "DIS interaction locations")}}
    ${{out.dis_interaction_locations_exists ? `<img src="${{outUrl("dis_interaction_locations")}}" alt="DIS interaction locations">` : ""}}
    ${{link(out.dis_interaction_locations_3d_html_exists, "dis_interaction_locations_3d_html", "DIS interaction locations HTML")}}
    ${{link(out.dis_summary_json_exists, "dis_summary_json", "DIS summary JSON")}}
    ${{link(out.dis_summary_exists, "dis_summary", "DIS summary CSV")}}
    ${{link(out.dis_path_optical_depths_exists, "dis_path_optical_depths", "DIS path optical depths")}}
    ${{link(out.dis_interaction_candidates_exists, "dis_interaction_candidates", "DIS interaction candidates")}}
    ${{link(out.dis_accepted_interactions_exists, "dis_accepted_interactions", "DIS accepted interactions")}}
    ${{link(out.dis_optical_depth_report_exists, "dis_optical_depth_report", "DIS optical-depth report")}}
    ${{link(out.tau_distribution_exists, "tau_distribution", "Tau distribution")}}
    ${{out.tau_distribution_exists ? `<img src="${{outUrl("tau_distribution")}}" alt="DIS tau distribution">` : ""}}
    ${{link(out.interaction_probability_distribution_exists, "interaction_probability_distribution", "Interaction probability distribution")}}
    ${{out.interaction_probability_distribution_exists ? `<img src="${{outUrl("interaction_probability_distribution")}}" alt="DIS interaction probability distribution">` : ""}}
    ${{link(out.optical_depth_map_exists, "optical_depth_map", "Optical depth map")}}
    ${{out.optical_depth_map_exists ? `<img src="${{outUrl("optical_depth_map")}}" alt="DIS optical depth map">` : ""}}
    ${{link(out.optical_depth_map_3d_html_exists, "optical_depth_map_3d_html", "Optical depth map HTML")}}
    ${{link(out.medium_density_map_exists, "medium_density_map", "Medium density map")}}
    ${{out.medium_density_map_exists ? `<img src="${{outUrl("medium_density_map")}}" alt="DIS medium density map">` : ""}}
    ${{link(out.interaction_location_distribution_exists, "interaction_location_distribution", "Interaction location distribution")}}
    ${{out.interaction_location_distribution_exists ? `<img src="${{outUrl("interaction_location_distribution")}}" alt="DIS interaction location distribution">` : ""}}
    ${{link(out.local_energy_distribution_exists, "local_energy_distribution", "Local energy distribution")}}
    ${{out.local_energy_distribution_exists ? `<img src="${{outUrl("local_energy_distribution")}}" alt="DIS local energy distribution">` : ""}}
    ${{link(out.local_density_distribution_exists, "local_density_distribution", "Local density distribution")}}
    ${{out.local_density_distribution_exists ? `<img src="${{outUrl("local_density_distribution")}}" alt="DIS local density distribution">` : ""}}
    ${{link(out.sigma_distribution_exists, "sigma_distribution", "Sigma distribution")}}
    ${{out.sigma_distribution_exists ? `<img src="${{outUrl("sigma_distribution")}}" alt="DIS sigma distribution">` : ""}}
    ${{link(out.density_energy_sigma_correlation_exists, "density_energy_sigma_correlation", "Density energy sigma correlation")}}
    ${{out.density_energy_sigma_correlation_exists ? `<img src="${{outUrl("density_energy_sigma_correlation")}}" alt="DIS density energy sigma correlation">` : ""}}
    ${{link(out.dis_diagnostics_report_exists, "dis_diagnostics_report", "DIS diagnostics JSON")}}
    ${{link(out.gbw_vs_iim_tau_comparison_exists, "gbw_vs_iim_tau_comparison", "GBW vs IIM tau comparison")}}
    ${{out.gbw_vs_iim_tau_comparison_exists ? `<img src="${{outUrl("gbw_vs_iim_tau_comparison")}}" alt="GBW vs IIM tau comparison">` : ""}}
    ${{link(out.gbw_vs_iim_probability_comparison_exists, "gbw_vs_iim_probability_comparison", "GBW vs IIM probability comparison")}}
    ${{out.gbw_vs_iim_probability_comparison_exists ? `<img src="${{outUrl("gbw_vs_iim_probability_comparison")}}" alt="GBW vs IIM probability comparison">` : ""}}
    ${{link(out.gbw_vs_iim_interaction_locations_exists, "gbw_vs_iim_interaction_locations", "GBW vs IIM interaction locations")}}
    ${{out.gbw_vs_iim_interaction_locations_exists ? `<img src="${{outUrl("gbw_vs_iim_interaction_locations")}}" alt="GBW vs IIM interaction locations">` : ""}}
    ${{link(out.gbw_vs_iim_summary_exists, "gbw_vs_iim_summary", "GBW vs IIM summary JSON")}}
  `) + group("ObserverBridge/", `
    ${{link(out.observer_bridge_candidates_exists, "observer_bridge_candidates", "Observer Bridge candidates")}}
    ${{link(out.observer_bridge_ranked_events_exists, "observer_bridge_ranked_events", "Observer Bridge ranked events")}}
    ${{link(out.observer_bridge_selected_candidates_exists, "observer_bridge_selected_candidates", "Observer Bridge selected candidates")}}
    ${{link(out.observer_bridge_selection_summary_exists, "observer_bridge_selection_summary", "Observer Bridge selection summary")}}
    ${{link(out.observer_bridge_summary_json_exists, "observer_bridge_summary_json", "Observer Bridge summary JSON")}}
    ${{link(out.observer_bridge_summary_exists, "observer_bridge_summary", "Observer Bridge summary CSV")}}
    ${{link(out.observer_bridge_report_exists, "observer_bridge_report", "Observer Bridge report")}}
    ${{link(out.observer_bridge_geometry_3d_html_exists, "observer_bridge_geometry_3d_html", "Observer Bridge geometry HTML")}}
    ${{link(out.observer_bridge_camera_overlay_exists, "observer_bridge_camera_overlay", "Observer Bridge camera overlay")}}
    ${{out.observer_bridge_camera_overlay_exists ? `<img src="${{outUrl("observer_bridge_camera_overlay")}}" alt="Observer Bridge camera overlay">` : ""}}
    ${{link(out.observer_bridge_overlay_background_audit_exists, "observer_bridge_overlay_background_audit", "Observer Bridge overlay background audit")}}
    ${{link(out.observer_bridge_background_comparison_exists, "observer_bridge_background_comparison", "Observer Bridge background comparison")}}
    ${{link(out.observer_bridge_overlay_hemisphere_diagnostic_exists, "observer_bridge_overlay_hemisphere_diagnostic", "Observer Bridge overlay hemisphere diagnostic")}}
    ${{link(out.candidate_multi_image_audit_exists, "candidate_multi_image_audit", "Candidate multi-image audit")}}
    ${{link(out.multiple_image_statistics_exists, "multiple_image_statistics", "Multiple image statistics")}}
    ${{link(out.observer_candidate_multiple_images_exists, "observer_candidate_multiple_images", "Observer candidate multiple images")}}
    ${{link(out.observer_candidate_multi_image_view_exists, "observer_candidate_multi_image_view", "Observer candidate multi-image view")}}
    ${{link(out.observer_bridge_kerr_interactive_view_exists, "observer_bridge_kerr_interactive_view", "Observer Bridge Kerr interactive view")}}
    ${{link(out.observer_candidate_kerr_pixel_map_exists, "observer_candidate_kerr_pixel_map", "Observer candidate Kerr pixel map")}}
    ${{link(out.observer_bridge_camera_view_exists, "observer_bridge_camera_view", "Observer Bridge camera view")}}
    ${{out.observer_bridge_camera_view_exists ? `<img src="${{outUrl("observer_bridge_camera_view")}}" alt="Observer Bridge camera view">` : ""}}
    ${{link(out.observer_bridge_map_exists, "observer_bridge_map", "Observer Bridge map")}}
    ${{out.observer_bridge_map_exists ? `<img src="${{outUrl("observer_bridge_map")}}" alt="Observer Bridge map">` : ""}}
    ${{link(out.observer_bridge_score_distribution_exists, "observer_bridge_score_distribution", "Observer Bridge score distribution")}}
    ${{out.observer_bridge_score_distribution_exists ? `<img src="${{outUrl("observer_bridge_score_distribution")}}" alt="Observer Bridge score distribution">` : ""}}
    ${{link(out.observer_bridge_weight_breakdown_exists, "observer_bridge_weight_breakdown", "Observer Bridge weight breakdown")}}
    ${{out.observer_bridge_weight_breakdown_exists ? `<img src="${{outUrl("observer_bridge_weight_breakdown")}}" alt="Observer Bridge weight breakdown">` : ""}}
    ${{link(out.observer_bridge_visibility_map_exists, "observer_bridge_visibility_map", "Observer Bridge visibility map")}}
    ${{out.observer_bridge_visibility_map_exists ? `<img src="${{outUrl("observer_bridge_visibility_map")}}" alt="Observer Bridge visibility map">` : ""}}
    ${{link(out.observer_bridge_ranked_events_png_exists, "observer_bridge_ranked_events_png", "Observer Bridge ranked events PNG")}}
    ${{out.observer_bridge_ranked_events_png_exists ? `<img src="${{outUrl("observer_bridge_ranked_events_png")}}" alt="Observer Bridge ranked events">` : ""}}
  `) + group("ObserverImageBranches/", `
    ${{link(out.observer_image_branches_exists, "observer_image_branches", "Observer image branches")}}
    ${{link(out.observer_image_primary_branches_exists, "observer_image_primary_branches", "Observer image primary branches")}}
    ${{link(out.observer_image_branch_summary_exists, "observer_image_branch_summary", "Observer image branch summary")}}
    ${{link(out.observer_image_branch_report_exists, "observer_image_branch_report", "Observer image branch report")}}
    ${{link(out.observer_image_statistics_exists, "observer_image_statistics", "Observer image statistics")}}
    ${{link(out.observer_branch_statistics_exists, "observer_branch_statistics", "Observer branch statistics CSV")}}
    ${{link(out.observer_branch_view_exists, "observer_branch_view", "Observer branch view")}}
    ${{link(out.observer_viewpoint_convention_audit_exists, "observer_viewpoint_convention_audit", "Observer viewpoint convention audit")}}
    ${{link(out.observer_viewpoint_convention_diagnostic_exists, "observer_viewpoint_convention_diagnostic", "Observer viewpoint convention diagnostic")}}
    ${{link(out.observer_branch_cluster_map_exists, "observer_branch_cluster_map", "Observer branch cluster map")}}
    ${{out.observer_branch_cluster_map_exists ? `<img src="${{outUrl("observer_branch_cluster_map")}}" alt="Observer branch cluster map">` : ""}}
    ${{link(out.observer_branch_score_distribution_exists, "observer_branch_score_distribution", "Observer branch score distribution")}}
    ${{out.observer_branch_score_distribution_exists ? `<img src="${{outUrl("observer_branch_score_distribution")}}" alt="Observer branch score distribution">` : ""}}
    ${{link(out.observer_branch_primary_vs_secondary_exists, "observer_branch_primary_vs_secondary", "Observer branches per candidate")}}
    ${{out.observer_branch_primary_vs_secondary_exists ? `<img src="${{outUrl("observer_branch_primary_vs_secondary")}}" alt="Observer branches per candidate">` : ""}}
  `) + group("POWHEG/", `
    ${{link(out.powheg_event_requests_exists, "powheg_event_requests", "POWHEG event requests")}}
    ${{link(out.powheg_summary_json_exists, "powheg_summary_json", "POWHEG summary JSON")}}
    ${{link(out.powheg_summary_exists, "powheg_summary", "POWHEG summary CSV")}}
    ${{link(out.powheg_report_exists, "powheg_report", "POWHEG report")}}
    ${{link(out.powheg_validation_report_exists, "powheg_validation_report", "POWHEG validation report")}}
    ${{link(out.powheg_lhe_exists, "powheg_lhe", "POWHEG smoke LHE")}}
    ${{link(out.powheg_log_exists, "powheg_log", "POWHEG smoke log")}}
    ${{link(out.powheg_card_preview_exists, "powheg_card_preview", "POWHEG card preview")}}
    ${{link(out.powheg_energy_distribution_exists, "powheg_energy_distribution", "POWHEG energy distribution")}}
    ${{out.powheg_energy_distribution_exists ? `<img src="${{outUrl("powheg_energy_distribution")}}" alt="POWHEG energy distribution">` : ""}}
    ${{link(out.powheg_job_summary_exists, "powheg_job_summary", "POWHEG job summary")}}
    ${{out.powheg_job_summary_exists ? `<img src="${{outUrl("powheg_job_summary")}}" alt="POWHEG job summary">` : ""}}
    ${{link(out.powheg_lhe_particles_exists, "powheg_lhe_particles", "POWHEG LHE particles")}}
    ${{link(out.powheg_lhe_events_summary_exists, "powheg_lhe_events_summary", "POWHEG LHE events summary")}}
    ${{link(out.powheg_lhe_particle_summary_csv_exists, "powheg_lhe_particle_summary_csv", "POWHEG LHE particle summary CSV")}}
    ${{link(out.powheg_lhe_particle_summary_json_exists, "powheg_lhe_particle_summary_json", "POWHEG LHE particle summary JSON")}}
    ${{link(out.powheg_event_summary_table_exists, "powheg_event_summary_table", "POWHEG event summary table")}}
    ${{link(out.powheg_particle_table_exists, "powheg_particle_table", "POWHEG particle table")}}
    ${{link(out.powheg_particle_table_html_exists, "powheg_particle_table_html", "POWHEG particle table preview")}}
    ${{link(out.powheg_particle_content_report_exists, "powheg_particle_content_report", "POWHEG particle content report")}}
    ${{link(out.powheg_lhe_event_view_exists, "powheg_lhe_event_view", "POWHEG LHE event viewer")}}
    ${{link(out.powheg_hard_process_event_display_exists, "powheg_hard_process_event_display", "POWHEG hard process event display")}}
    ${{out.powheg_hard_process_event_display_exists ? `<img src="${{outUrl("powheg_hard_process_event_display")}}" alt="POWHEG hard process event display">` : ""}}
    ${{link(out.powheg_hard_process_event_display_view_exists, "powheg_hard_process_event_display_view", "POWHEG hard process event selector")}}
    ${{link(out.powheg_lhe_particle_histogram_exists, "powheg_lhe_particle_histogram", "POWHEG LHE particle histogram")}}
    ${{out.powheg_lhe_particle_histogram_exists ? `<img src="${{outUrl("powheg_lhe_particle_histogram")}}" alt="POWHEG LHE particle histogram">` : ""}}
    ${{link(out.powheg_lhe_energy_spectrum_exists, "powheg_lhe_energy_spectrum", "POWHEG LHE energy spectrum")}}
    ${{out.powheg_lhe_energy_spectrum_exists ? `<img src="${{outUrl("powheg_lhe_energy_spectrum")}}" alt="POWHEG LHE energy spectrum">` : ""}}
    ${{link(out.powheg_lhe_momentum_spectrum_exists, "powheg_lhe_momentum_spectrum", "POWHEG LHE momentum spectrum")}}
    ${{out.powheg_lhe_momentum_spectrum_exists ? `<img src="${{outUrl("powheg_lhe_momentum_spectrum")}}" alt="POWHEG LHE momentum spectrum">` : ""}}
  `) + group("Dashboard/", `
    ${{link(out.html_summary_exists, "html_summary", "Dashboard HTML")}}
  `);
}}
function fnum(values, section, key, fallback) {{
  const value = Number(values?.[section]?.[key]);
  return Number.isFinite(value) ? value : fallback;
}}
function fmt(value, digits = 2) {{
  if (!Number.isFinite(value)) return "-";
  return value.toFixed(digits).replace(/\\.?0+$/, "");
}}
function drawGeometrySvg() {{
  const svg = document.querySelector("#geometrySvg");
  if (!svg) return;
  const values = collect();
  const spin = Math.max(-0.999, Math.min(0.999, fnum(values, "black_hole", "spin_a", 0.8)));
  const rH = 1 + Math.sqrt(Math.max(0, 1 - spin * spin));
  const torusIn = fnum(values, "analytic_torus", "r_inner_rg", 6) / rH;
  const torusOut = fnum(values, "analytic_torus", "r_outer_rg", 18) / rH;
  const torusPeak = fnum(values, "analytic_torus", "r_peak_rg", 10) / rH;
  const coneDeg = fnum(values, "polar_cone", "opening_angle_deg", 22);
  const cone = coneDeg * Math.PI / 180;
  const coneMin = fnum(values, "polar_cone", "r_min_rg", 2.2) / rH;
  const coneMax = fnum(values, "polar_cone", "r_max_rg", 40) / rH;
  const srcMin = fnum(values, "uhe_neutrino_source", "r_min_rg", 3) / rH;
  const srcMax = fnum(values, "uhe_neutrino_source", "r_max_rg", 12) / rH;
  const obsR = fnum(values, "observer_camera", "observer_distance_rg", 60) / rH;
  const inc = fnum(values, "observer_camera", "inclination_deg", 80) * Math.PI / 180;
  const fov = fnum(values, "observer_camera", "field_of_view_deg", 25) * Math.PI / 180;
  const lim = Math.max(torusOut * 1.35, coneMax * 1.1, obsR * 0.22, 16);
  const vb = [-lim, -lim, 2 * lim, 2 * lim].join(" ");
  svg.setAttribute("viewBox", vb);
  const y = value => -value;
  const p = (x, z) => `${{x}},${{y(z)}}`;
  const circle = (r, extra) => `<circle cx="0" cy="0" r="${{r}}" ${{extra}}/>`;
  const wedgePath = (r1, r2, a1, a2) => {{
    const x1 = r2 * Math.sin(a1), z1 = r2 * Math.cos(a1);
    const x2 = r2 * Math.sin(a2), z2 = r2 * Math.cos(a2);
    const x3 = r1 * Math.sin(a2), z3 = r1 * Math.cos(a2);
    const x4 = r1 * Math.sin(a1), z4 = r1 * Math.cos(a1);
    const large = Math.abs(a2 - a1) > Math.PI ? 1 : 0;
    return `M ${{p(x1,z1)}} A ${{r2}} ${{r2}} 0 ${{large}} 0 ${{p(x2,z2)}} L ${{p(x3,z3)}} A ${{r1}} ${{r1}} 0 ${{large}} 1 ${{p(x4,z4)}} Z`;
  }};
  const conePoly = sign => {{
    const pts = [
      [sign * coneMin * Math.sin(cone), sign * coneMin * Math.cos(cone)],
      [sign * coneMax * Math.sin(cone), sign * coneMax * Math.cos(cone)],
      [-sign * coneMax * Math.sin(cone), sign * coneMax * Math.cos(cone)],
      [-sign * coneMin * Math.sin(cone), sign * coneMin * Math.cos(cone)],
    ].map(([x,z]) => p(x,z)).join(" ");
    return `<polygon points="${{pts}}" fill="#f0c84b" fill-opacity="0.18" stroke="#ffec99" stroke-width="${{0.08 * lim / 25}}"/>`;
  }};
  const obsScale = Math.min(lim * 0.92, obsR);
  const ox = obsScale * Math.sin(inc), oz = obsScale * Math.cos(inc);
  const len = Math.hypot(ox, oz) || 1;
  const dx = -ox / len, dz = -oz / len;
  const nx = -dz, nz = dx;
  const flen = Math.min(lim * 0.48, len * 0.88);
  const cx = ox + dx * flen, cz = oz + dz * flen;
  const hw = Math.tan(0.5 * fov) * flen;
  const lx = cx + nx * hw, lz = cz + nz * hw;
  const rx = cx - nx * hw, rz = cz - nz * hw;
  const gridStep = Math.max(5, Math.round(lim / 5));
  let grid = "";
  for (let g = -Math.ceil(lim / gridStep) * gridStep; g <= lim; g += gridStep) {{
    grid += `<line x1="${{g}}" y1="${{-lim}}" x2="${{g}}" y2="${{lim}}" stroke="#354052" stroke-width="0.035" stroke-dasharray="0.25 0.25"/>`;
    grid += `<line x1="${{-lim}}" y1="${{g}}" x2="${{lim}}" y2="${{g}}" stroke="#354052" stroke-width="0.035" stroke-dasharray="0.25 0.25"/>`;
  }}
  const bipolar = values.polar_cone.draw_mode === "bipolar_funnel";
  svg.innerHTML = `
    <rect x="${{-lim}}" y="${{-lim}}" width="${{2*lim}}" height="${{2*lim}}" fill="#101318"/>
    ${{grid}}
    ${{values.analytic_torus.show_in_preview ? circle(torusOut, 'fill="#2372a3" fill-opacity="0.34" stroke="#95d9ff" stroke-width="0.08"') + circle(torusIn, 'fill="#101318" stroke="#95d9ff" stroke-opacity="0.65" stroke-width="0.05"') + circle(torusPeak, 'fill="none" stroke="#d7f1ff" stroke-width="0.05" stroke-dasharray="0.35 0.25"') : ""}}
    ${{values.polar_cone.enabled ? conePoly(1) + (bipolar ? conePoly(-1) : "") : ""}}
    <path d="${{wedgePath(srcMin, srcMax, -cone, cone)}}" fill="#ff6f59" fill-opacity="0.72" stroke="#ffd1c9" stroke-width="0.06"/>
    ${{circle(1, 'fill="black" stroke="#f4f4f4" stroke-width="0.08"')}}
    ${{circle(2/rH, 'fill="none" stroke="#bbbbbb" stroke-opacity="0.65" stroke-width="0.04" stroke-dasharray="0.2 0.15"')}}
    <line x1="${{ox}}" y1="${{y(oz)}}" x2="${{lx}}" y2="${{y(lz)}}" stroke="#d6ff6b" stroke-width="0.06" stroke-dasharray="0.18 0.16"/>
    <line x1="${{ox}}" y1="${{y(oz)}}" x2="${{rx}}" y2="${{y(rz)}}" stroke="#d6ff6b" stroke-width="0.06" stroke-dasharray="0.18 0.16"/>
    <line x1="${{ox}}" y1="${{y(oz)}}" x2="0" y2="0" stroke="#d6ff6b" stroke-opacity="0.45" stroke-width="0.04" stroke-dasharray="0.25 0.2"/>
    <circle cx="${{ox}}" cy="${{y(oz)}}" r="${{0.32 * lim / 25}}" fill="#d6ff6b" stroke="#0c0f14" stroke-width="0.07"/>
    <text x="0" y="0.33" fill="white" font-size="${{0.8 * lim / 25}}" text-anchor="middle">BH</text>
    <text x="${{-0.96*lim}}" y="${{-0.84*lim}}" fill="#bfeaff" font-size="${{0.8 * lim / 25}}">torus: Rin=${{fmt(torusIn)}} rH, Rpeak=${{fmt(torusPeak)}} rH, Rout=${{fmt(torusOut)}} rH</text>
    <text x="${{0.15*lim}}" y="${{-0.30*lim}}" fill="#ffd5cd" font-size="${{0.68 * lim / 25}}">source ${{fmt(srcMin)}}-${{fmt(srcMax)}} rH</text>
    <text x="${{-0.05*lim}}" y="${{-0.15*lim}}" fill="#ffe680" font-size="${{0.68 * lim / 25}}">cone ${{fmt(coneDeg,1)}} deg</text>
    <text x="${{ox + 0.45}}" y="${{y(oz)}}" fill="#d6ff6b" font-size="${{0.75 * lim / 25}}">observer</text>
    <text x="${{ox + dx*flen*0.36}}" y="${{y(oz + dz*flen*0.36)}}" fill="#d6ff6b" font-size="${{0.68 * lim / 25}}">FOV ${{fmt(fov*180/Math.PI,1)}} deg</text>
    <text x="${{0.97*lim}}" y="${{0.77*lim}}" fill="#e7ebf2" font-size="${{0.75 * lim / 25}}" text-anchor="end">
      <tspan x="${{0.97*lim}}" dy="0">a=${{fmt(spin)}}; rH=${{fmt(rH,3)}} rg</tspan>
      <tspan x="${{0.97*lim}}" dy="${{0.9 * lim / 25}}">camera r=${{fmt(obsR*rH)}} rg = ${{fmt(obsR)}} rH</tspan>
      <tspan x="${{0.97*lim}}" dy="${{0.9 * lim / 25}}">inclination=${{fmt(inc*180/Math.PI,1)}} deg</tspan>
    </text>
    <line x1="${{-0.96*lim}}" y1="${{0.88*lim}}" x2="${{-0.96*lim + gridStep}}" y2="${{0.88*lim}}" stroke="#e7ebf2" stroke-width="0.12"/>
    <text x="${{-0.96*lim + gridStep/2}}" y="${{0.84*lim}}" fill="#e7ebf2" font-size="${{0.68 * lim / 25}}" text-anchor="middle">${{gridStep}} rH</text>
  `;
}}
function render() {{
  const root = document.querySelector("#app");
  const status = state.outputs;
  const tabs = orderedTabs();
  const active = tabs.find(t => tabLabel(t) === activeTab) || tabs[0];
  activeTab = tabLabel(active);
  const runName = state.values.run.run_name || "HADROS3_run";
  const defaultStage = state.release && state.release.pipeline_version ? state.release.pipeline_version : "unclassified";
  const runStrip = `<div class="run-strip">
    <div class="run-main-row"><label><span>Run name</span><input id="runNameInput" type="text" value="${{runName}}"></label><span>Output</span><div class="output-folder">output/${{safeRunName(runName)}}</div></div>
    <div class="workflow-actions">
      <label><span>Case name</span><input id="caseNameInput" type="text" value="${{safeRunName(runName)}}"></label>
      <label><span>Stage</span><input id="stageInput" type="text" value="${{defaultStage}}"></label>
      <label><span>Description</span><input id="descriptionInput" type="text" value=""></label>
      <div class="workflow-buttons">
        <button type="button" id="registerRunButton">Register Run</button>
      </div>
      <details class="workflow-actions-log" open><summary>Workflow Actions Log</summary><pre id="workflowActionsOutput">No workflow action run yet.</pre></details>
    </div>
  </div>`;
  const nav = `<nav>${{tabs.map(tab => `<button class="tab-button ${{tabLabel(tab) === activeTab ? "active" : ""}}" data-tab="${{tabLabel(tab)}}">${{tabLabel(tab)}}</button>`).join("")}}</nav>`;
  const customTabs = new Set(["DIS Interaction Sampler", "Observer Bridge", "Observer Image Branches", "POWHEG"]);
  const genericFields = customTabs.has(activeTab) ? "" : renderFields(active);
  const panelClass = activeTab === "Observer Image Branches" ? "panel observer-image-branches-panel" : "panel";
  root.innerHTML = runStrip + nav + `<div class="${{panelClass}}"><p class="note">Geometry/configuration shell only. Expensive event stages are disabled.</p>${{genericFields}}${{activeTab === "Camera" ? renderHadrosCameraPanel() + renderBackendTable() : ""}}${{activeTab === "UHE Source" ? renderSourcePanel() : ""}}${{activeTab === "Forward Geodesics" ? renderForwardPanel() : ""}}${{activeTab === "DIS Interaction Sampler" ? renderDisPanel() : ""}}${{activeTab === "Observer Bridge" ? renderObserverBridgePanel() : ""}}${{activeTab === "Observer Image Branches" ? renderObserverImageBranchesPanel() : ""}}${{activeTab === "POWHEG" ? renderPowhegPanel() : ""}}${{activeTab === "Outputs" ? renderOutputsPanel() : ""}}` +
    `<pre id="log"></pre></div>` +
    renderContextPanel();
  bindHadrosCameraPanel();
  const uheButton = document.querySelector("#uhe-source-button");
  if (uheButton) uheButton.onclick = sampleUheSource;
  const forwardButton = document.querySelector("#forward-geodesics-button");
  if (forwardButton) forwardButton.onclick = propagateForwardGeodesics;
  const disButton = document.querySelector("#dis-sampler-button");
  if (disButton) disButton.onclick = sampleDisInteractions;
  const disCompareButton = document.querySelector("#dis-compare-button");
  if (disCompareButton) disCompareButton.onclick = compareDisModels;
  const observerBridgeButton = document.querySelector("#observer-bridge-button");
  if (observerBridgeButton) observerBridgeButton.onclick = computeObserverBridge;
  const observerImageBranchesButton = document.querySelector("#observer-image-branches-button");
  if (observerImageBranchesButton) observerImageBranchesButton.onclick = analyzeObserverImageBranches;
  const powhegButton = document.querySelector("#powheg-button");
  if (powhegButton) powhegButton.onclick = preparePowheg;
  const registerRunButton = document.querySelector("#registerRunButton");
  if (registerRunButton) registerRunButton.onclick = registerRun;
  bindNumberInputs();
  drawGeometrySvg();
  document.querySelector("#runNameInput").addEventListener("input", event => {{
    state.values = collect();
    document.querySelector(".output-folder").textContent = "output/" + safeRunName(event.target.value);
    const caseInput = document.querySelector("#caseNameInput");
    if (caseInput && !caseInput.dataset.edited) caseInput.value = safeRunName(event.target.value);
  }});
  const caseInput = document.querySelector("#caseNameInput");
  if (caseInput) caseInput.addEventListener("input", event => event.target.dataset.edited = "1");
  const syncValuesAndGeometry = () => {{
    state.values = collect();
    drawGeometrySvg();
  }};
  document.querySelectorAll("[data-section]").forEach(el => el.addEventListener("input", syncValuesAndGeometry));
  document.querySelectorAll("[data-section]").forEach(el => el.addEventListener("change", syncValuesAndGeometry));
  document.querySelectorAll(".tab-button").forEach(btn => btn.addEventListener("click", () => {{
    state.values = collect();
    activeTab = btn.dataset.tab;
    render();
  }}));
}}
render();
</script>
</main>
</body>
</html>
"""


def state_payload(values: dict[str, dict[str, Any]], config_path: Path | None = None) -> dict[str, Any]:
    return dashboard_payload(values, config_path)


class Handler(BaseHTTPRequestHandler):
    config_path = DEFAULT_CONFIG

    def _send(self, code: int, text: str, content_type: str = "text/plain") -> None:
        payload = text.encode("utf-8")
        self._send_bytes(code, payload, content_type)

    def _send_bytes(self, code: int, payload: bytes, content_type: str = "application/octet-stream") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type if "charset" in content_type else f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_values(self) -> dict[str, dict[str, Any]]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        return deep_update(defaults(), json.loads(self.rfile.read(length) or b"{}"))

    def _read_payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = json.loads(self.rfile.read(length) or b"{}")
        if "values" in raw:
            return {
                "values": deep_update(defaults(), raw.get("values", {})),
                "previewOptions": raw.get("previewOptions", {}),
                "raw": raw,
            }
        return {"values": deep_update(defaults(), raw), "previewOptions": {}, "raw": raw}

    def _output_file(self, values: dict[str, dict[str, Any]]) -> Path | None:
        request_path = urlparse(self.path).path
        if not request_path.startswith("/output/"):
            return None
        relative = request_path.removeprefix("/output/")
        if not relative or ".." in Path(relative).parts:
            return None
        output_dir = ROOT / run_output_dir(values)
        path = output_dir / relative
        if not path.exists() or not path.is_file():
            return None
        return path

    def _asset_file(self) -> Path | None:
        request_path = urlparse(self.path).path
        if not request_path.startswith("/assets/"):
            return None
        relative = request_path.removeprefix("/assets/")
        if not relative or ".." in Path(relative).parts:
            return None
        path = ROOT / "assets" / relative
        if not path.exists() or not path.is_file():
            return None
        return path

    def do_HEAD(self) -> None:  # noqa: N802
        values = load_values(self.config_path)
        if self.path == "/":
            payload = render_html(values, self.config_path).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            return
        if self.path == "/api/state":
            payload = json.dumps(
                state_payload(values, self.config_path),
                indent=2,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            return
        asset = self._asset_file()
        if asset is not None:
            content_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(asset.stat().st_size))
            self.end_headers()
            return
        path = self._output_file(values)
        if path is not None:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        values = load_values(self.config_path)
        if self.path == "/":
            self._send(200, render_html(values, self.config_path), "text/html")
            return
        if self.path == "/api/state":
            self._send(
                200,
                json.dumps(state_payload(values, self.config_path), indent=2),
                "application/json",
            )
            return
        if self.path == "/api/last-camera":
            info = discover_original_hadros()
            local_path = ROOT / "configs" / "cameras" / "last_camera.json"
            raw = info.get("components", {}).get("last_camera_config")
            original_path = Path(raw) if raw else None
            path = local_path if local_path.exists() else original_path
            if path is None or not path.exists():
                self._send(200, json.dumps({"exists": False, "path": str(path) if path else None}, indent=2), "application/json")
                return
            camera = json.loads(path.read_text(encoding="utf-8"))
            self._send(
                200,
                json.dumps({"exists": True, "path": str(path), "mtime": path.stat().st_mtime, "camera": camera}, indent=2),
                "application/json",
            )
            return
        if self.path == "/outputs":
            output_dir = ROOT / run_output_dir(values)
            index = dashboard_dir(output_dir) / "index.html"
            self._send(200, index.read_text(encoding="utf-8") if index.exists() else "No outputs rendered yet.", "text/html")
            return
        asset = self._asset_file()
        if asset is not None:
            content_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
            self._send_bytes(200, asset.read_bytes(), content_type)
            return
        path = self._output_file(values)
        if path is not None:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send_bytes(200, path.read_bytes(), content_type)
            return
        self._send(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        payload = self._read_payload()
        values = payload["values"]
        preview_options = payload["previewOptions"]
        raw = payload.get("raw", {})
        if self.path == "/api/save":
            write_values(self.config_path, values)
            self._send(200, f"wrote {self.config_path}\n")
            return
        if self.path == "/api/register-run":
            summary = register_current_run(
                values,
                case_name=str(raw.get("case_name", "")),
                stage=str(raw.get("stage", "")),
                description=str(raw.get("description", "")),
                root=ROOT,
            )
            self._send(200 if summary["ok"] else 500, json.dumps(summary, indent=2, sort_keys=True) + "\n", "application/json")
            return
        if self.path == "/api/render":
            write_values(self.config_path, values)
            summary = render_hadros_web(values, root=ROOT)
            self._send(200, json.dumps(summary, indent=2, sort_keys=True) + "\n", "application/json")
            return
        if self.path == "/api/render-camera-preview":
            run_dir = ROOT / run_output_dir(values)
            ensure_output_layout(run_dir)
            output_dir = camera_preview_dir(run_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            write_values(self.config_path, values)
            summary = render_camera_preview(values, root=ROOT, output_dir=output_dir, preview_options=preview_options)
            self._send(200, json.dumps(summary, indent=2, sort_keys=True) + "\n", "application/json")
            return
        if self.path == "/api/sample-uhe-source":
            problems = validate_values(values)
            if problems:
                self._send(
                    400,
                    json.dumps({"status": "error", "validation_errors": problems}, indent=2, sort_keys=True) + "\n",
                    "application/json",
                )
                return
            output_dir = ROOT / run_output_dir(values)
            output_dir.mkdir(parents=True, exist_ok=True)
            ensure_output_layout(output_dir)
            values["uhe_neutrino_source"]["status"] = "sampled_position_direction_energy_no_forward_kerr_geodesic"
            source_summary = generate_uhe_source_products(values, output_dir=output_dir)
            clear_forward_geodesics_outputs(output_dir)
            clear_dis_outputs(output_dir)
            clear_observer_bridge_outputs(output_dir)
            clear_observer_image_branches_outputs(output_dir)
            clear_powheg_outputs(output_dir)
            render_summary = render_hadros_web(values, root=ROOT, source_summary=source_summary)
            summary = {"status": "ok", "source": source_summary, "render": render_summary}
            self._send(200, json.dumps(summary, indent=2, sort_keys=True) + "\n", "application/json")
            return
        if self.path == "/api/propagate-forward-geodesics":
            problems = validate_values(values)
            if problems:
                self._send(
                    400,
                    json.dumps({"status": "error", "validation_errors": problems}, indent=2, sort_keys=True) + "\n",
                    "application/json",
                )
                return
            output_dir = ROOT / run_output_dir(values)
            output_dir.mkdir(parents=True, exist_ok=True)
            ensure_output_layout(output_dir)
            values["forward_geodesics"]["status"] = "forward_kerr_geodesics_propagated_no_interactions"
            forward_summary = generate_forward_geodesic_products(values, run_output_dir=output_dir)
            clear_dis_outputs(output_dir)
            clear_observer_bridge_outputs(output_dir)
            clear_observer_image_branches_outputs(output_dir)
            clear_powheg_outputs(output_dir)
            render_summary = render_hadros_web(values, root=ROOT, forward_geodesic_summary=forward_summary)
            summary = {"status": "ok", "forward": forward_summary, "render": render_summary}
            self._send(200, json.dumps(summary, indent=2, sort_keys=True) + "\n", "application/json")
            return
        if self.path == "/api/sample-dis-interactions":
            problems = validate_values(values)
            if problems:
                self._send(
                    400,
                    json.dumps({"status": "error", "validation_errors": problems}, indent=2, sort_keys=True) + "\n",
                    "application/json",
                )
                return
            output_dir = ROOT / run_output_dir(values)
            output_dir.mkdir(parents=True, exist_ok=True)
            ensure_output_layout(output_dir)
            values["dis_interaction_sampler"]["status"] = "dis_optical_depth_sampled_no_observer_bridge"
            dis_summary = generate_dis_interaction_products(values, run_output_dir=output_dir)
            clear_observer_bridge_outputs(output_dir)
            clear_observer_image_branches_outputs(output_dir)
            clear_powheg_outputs(output_dir)
            render_summary = render_hadros_web(values, root=ROOT, dis_summary=dis_summary)
            summary = {"status": "ok", "dis": dis_summary, "render": render_summary}
            self._send(200, json.dumps(summary, indent=2, sort_keys=True) + "\n", "application/json")
            return
        if self.path == "/api/observer-bridge":
            problems = validate_values(values)
            if problems:
                self._send(
                    400,
                    json.dumps({"status": "error", "validation_errors": problems}, indent=2, sort_keys=True) + "\n",
                    "application/json",
                )
                return
            output_dir = ROOT / run_output_dir(values)
            output_dir.mkdir(parents=True, exist_ok=True)
            ensure_output_layout(output_dir)
            values["observer_bridge"]["status"] = "observer_bridge_scored_no_event_generation"
            observer_bridge_summary = generate_observer_bridge_products(values, run_output_dir=output_dir)
            clear_observer_image_branches_outputs(output_dir)
            clear_powheg_outputs(output_dir)
            render_summary = render_hadros_web(values, root=ROOT, observer_bridge_summary=observer_bridge_summary)
            summary = {"status": "ok", "observer_bridge": observer_bridge_summary, "render": render_summary}
            self._send(200, json.dumps(summary, indent=2, sort_keys=True) + "\n", "application/json")
            return
        if self.path == "/api/observer-image-branches":
            problems = validate_values(values)
            if problems:
                self._send(
                    400,
                    json.dumps({"status": "error", "validation_errors": problems}, indent=2, sort_keys=True) + "\n",
                    "application/json",
                )
                return
            output_dir = ROOT / run_output_dir(values)
            output_dir.mkdir(parents=True, exist_ok=True)
            ensure_output_layout(output_dir)
            values["observer_image_branches"]["status"] = "observer_image_branches_analyzed"
            branch_summary = generate_observer_image_branch_products(values, run_output_dir=output_dir)
            clear_powheg_outputs(output_dir)
            render_summary = render_hadros_web(values, root=ROOT, observer_image_branch_summary=branch_summary)
            summary = {"status": "ok", "observer_image_branches": branch_summary, "render": render_summary}
            self._send(200, json.dumps(summary, indent=2, sort_keys=True) + "\n", "application/json")
            return
        if self.path == "/api/powheg":
            problems = validate_values(values)
            if problems:
                self._send(
                    400,
                    json.dumps({"status": "error", "validation_errors": problems}, indent=2, sort_keys=True) + "\n",
                    "application/json",
                )
                return
            output_dir = ROOT / run_output_dir(values)
            output_dir.mkdir(parents=True, exist_ok=True)
            ensure_output_layout(output_dir)
            powheg_summary = generate_powheg_products(values, run_output_dir=output_dir)
            render_summary = render_hadros_web(values, root=ROOT, powheg_summary=powheg_summary)
            summary = {"status": "ok", "powheg": powheg_summary, "render": render_summary}
            self._send(200, json.dumps(summary, indent=2, sort_keys=True) + "\n", "application/json")
            return
        if self.path == "/api/compare-dis-models":
            problems = validate_values(values)
            if problems:
                self._send(
                    400,
                    json.dumps({"status": "error", "validation_errors": problems}, indent=2, sort_keys=True) + "\n",
                    "application/json",
                )
                return
            output_dir = ROOT / run_output_dir(values)
            ensure_output_layout(output_dir)
            comparison = generate_gbw_iim_comparison(values, run_output_dir=output_dir)
            render_summary = render_hadros_web(values, root=ROOT)
            summary = {"status": "ok", "comparison": comparison, "render": render_summary}
            self._send(200, json.dumps(summary, indent=2, sort_keys=True) + "\n", "application/json")
            return
        if self.path == "/api/launch-interactive-camera-preview":
            run_dir = ROOT / run_output_dir(values)
            ensure_output_layout(run_dir)
            output_dir = camera_preview_dir(run_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            write_values(self.config_path, values)
            summary = launch_interactive_camera_preview(values, root=ROOT, output_dir=output_dir, preview_options=preview_options)
            self._send(200, json.dumps(summary, indent=2, sort_keys=True) + "\n", "application/json")
            return
        self._send(404, "not found")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--print-schema", action="store_true")
    parser.add_argument("--write-default-config", type=Path)
    parser.add_argument("--render", action="store_true", help="Render products and exit. This is also the default action.")
    parser.add_argument("--camera-preview-only", action="store_true", help="Render only the HADROS3 camera preview and exit.")
    parser.add_argument("--launch-interactive-camera", action="store_true", help="Launch the original HADROS interactive camera preview and exit.")
    parser.add_argument("--sample-uhe-source", action="store_true", help="Generate H3-W5 UHE source samples through hadros-web orchestration and exit.")
    parser.add_argument("--propagate-forward-geodesics", action="store_true", help="Generate H3-W6 forward neutrino geodesics through hadros-web orchestration and exit.")
    parser.add_argument("--sample-dis-interactions", action="store_true", help="Generate H3-W7 DIS optical-depth interaction samples through hadros-web orchestration and exit.")
    parser.add_argument("--observer-bridge", action="store_true", help="Generate H3-W8 Observer Bridge scoring products through hadros-web orchestration and exit.")
    parser.add_argument("--observer-image-branches", action="store_true", help="Generate H3-W8b Observer Image Branch Analysis products through hadros-web orchestration and exit.")
    parser.add_argument("--powheg", action="store_true", help="Prepare H3-W9a POWHEG dry-run jobs through hadros-web orchestration and exit.")
    parser.add_argument("--powheg-real-smoke", action="store_true", help="Run H3-W9b one-candidate local POWHEG real-smoke mode and exit.")
    parser.add_argument("--powheg-real-free", action="store_true", help="Run local POWHEG for the configured candidate and event counts.")
    parser.add_argument("--serve", action="store_true", help="Serve the web control surface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    args = parser.parse_args()

    if args.print_schema:
        print(json.dumps(schema(), indent=2, sort_keys=True))
        return 0
    if args.write_default_config is not None:
        write_values(args.write_default_config, defaults())
        print(f"wrote {args.write_default_config}")
        return 0
    if args.serve:
        Handler.config_path = args.config
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        print(f"Serving HADROS3 hadros-web at http://{args.host}:{args.port}")
        server.serve_forever()
        return 0

    values = load_values(args.config)
    if not args.config.exists():
        write_values(args.config, values)
    if args.camera_preview_only:
        output_dir = args.output_dir if args.output_dir is not None else ROOT / run_output_dir(values)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        ensure_output_layout(output_dir)
        output_dir = camera_preview_dir(output_dir)
        summary = render_camera_preview(values, root=ROOT, output_dir=output_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.launch_interactive_camera:
        output_dir = args.output_dir if args.output_dir is not None else ROOT / run_output_dir(values)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        ensure_output_layout(output_dir)
        output_dir = camera_preview_dir(output_dir)
        summary = launch_interactive_camera_preview(values, root=ROOT, output_dir=output_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.sample_uhe_source:
        output_dir = args.output_dir if args.output_dir is not None else ROOT / run_output_dir(values)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        ensure_output_layout(output_dir)
        values["uhe_neutrino_source"]["status"] = "sampled_position_direction_energy_no_forward_kerr_geodesic"
        source_summary = generate_uhe_source_products(values, output_dir=output_dir)
        clear_forward_geodesics_outputs(output_dir)
        clear_dis_outputs(output_dir)
        clear_observer_bridge_outputs(output_dir)
        clear_observer_image_branches_outputs(output_dir)
        clear_powheg_outputs(output_dir)
        render_summary = render_hadros_web(values, root=ROOT, output_dir=output_dir, source_summary=source_summary)
        print(json.dumps({"status": "ok", "source": source_summary, "render": render_summary}, indent=2, sort_keys=True))
        return 0
    if args.propagate_forward_geodesics:
        output_dir = args.output_dir if args.output_dir is not None else ROOT / run_output_dir(values)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        ensure_output_layout(output_dir)
        values["forward_geodesics"]["status"] = "forward_kerr_geodesics_propagated_no_interactions"
        forward_summary = generate_forward_geodesic_products(values, run_output_dir=output_dir)
        clear_dis_outputs(output_dir)
        clear_observer_bridge_outputs(output_dir)
        clear_observer_image_branches_outputs(output_dir)
        clear_powheg_outputs(output_dir)
        render_summary = render_hadros_web(values, root=ROOT, output_dir=output_dir, forward_geodesic_summary=forward_summary)
        print(json.dumps({"status": "ok", "forward": forward_summary, "render": render_summary}, indent=2, sort_keys=True))
        return 0
    if args.sample_dis_interactions:
        output_dir = args.output_dir if args.output_dir is not None else ROOT / run_output_dir(values)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        ensure_output_layout(output_dir)
        values["dis_interaction_sampler"]["status"] = "dis_optical_depth_sampled_no_observer_bridge"
        dis_summary = generate_dis_interaction_products(values, run_output_dir=output_dir)
        clear_observer_bridge_outputs(output_dir)
        clear_observer_image_branches_outputs(output_dir)
        clear_powheg_outputs(output_dir)
        render_summary = render_hadros_web(values, root=ROOT, output_dir=output_dir, dis_summary=dis_summary)
        print(json.dumps({"status": "ok", "dis": dis_summary, "render": render_summary}, indent=2, sort_keys=True))
        return 0
    if args.observer_bridge:
        output_dir = args.output_dir if args.output_dir is not None else ROOT / run_output_dir(values)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        ensure_output_layout(output_dir)
        values["observer_bridge"]["status"] = "observer_bridge_scored_no_event_generation"
        observer_bridge_summary = generate_observer_bridge_products(values, run_output_dir=output_dir)
        clear_observer_image_branches_outputs(output_dir)
        clear_powheg_outputs(output_dir)
        render_summary = render_hadros_web(values, root=ROOT, output_dir=output_dir, observer_bridge_summary=observer_bridge_summary)
        print(json.dumps({"status": "ok", "observer_bridge": observer_bridge_summary, "render": render_summary}, indent=2, sort_keys=True))
        return 0
    if args.observer_image_branches:
        output_dir = args.output_dir if args.output_dir is not None else ROOT / run_output_dir(values)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        ensure_output_layout(output_dir)
        values["observer_image_branches"]["status"] = "observer_image_branches_analyzed"
        branch_summary = generate_observer_image_branch_products(values, run_output_dir=output_dir)
        clear_powheg_outputs(output_dir)
        render_summary = render_hadros_web(values, root=ROOT, output_dir=output_dir, observer_image_branch_summary=branch_summary)
        print(json.dumps({"status": "ok", "observer_image_branches": branch_summary, "render": render_summary}, indent=2, sort_keys=True))
        return 0
    if args.powheg:
        output_dir = args.output_dir if args.output_dir is not None else ROOT / run_output_dir(values)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        ensure_output_layout(output_dir)
        powheg_summary = generate_powheg_products(values, run_output_dir=output_dir)
        render_summary = render_hadros_web(values, root=ROOT, output_dir=output_dir, powheg_summary=powheg_summary)
        print(json.dumps({"status": "ok", "powheg": powheg_summary, "render": render_summary}, indent=2, sort_keys=True))
        return 0
    if args.powheg_real_smoke:
        output_dir = args.output_dir if args.output_dir is not None else ROOT / run_output_dir(values)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        ensure_output_layout(output_dir)
        values["powheg"]["run_mode"] = "real_smoke"
        values["powheg"]["events_per_candidate"] = min(2, int(float(values["powheg"].get("events_per_candidate", 2))))
        powheg_summary = generate_powheg_products(values, run_output_dir=output_dir)
        render_summary = render_hadros_web(values, root=ROOT, output_dir=output_dir, powheg_summary=powheg_summary)
        print(json.dumps({"status": "ok", "powheg": powheg_summary, "render": render_summary}, indent=2, sort_keys=True))
        return 0
    if args.powheg_real_free:
        output_dir = args.output_dir if args.output_dir is not None else ROOT / run_output_dir(values)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        ensure_output_layout(output_dir)
        values["powheg"]["run_mode"] = "real_free"
        powheg_summary = generate_powheg_products(values, run_output_dir=output_dir)
        render_summary = render_hadros_web(values, root=ROOT, output_dir=output_dir, powheg_summary=powheg_summary)
        print(json.dumps({"status": "ok", "powheg": powheg_summary, "render": render_summary}, indent=2, sort_keys=True))
        return 0
    summary = render_hadros_web(values, root=ROOT, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
