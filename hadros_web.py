#!/usr/bin/env python3
"""HADROS3 web/configuration shell.

Use --serve for the H3-W0..H3-W4 web dashboard, or --render/--output-dir to
render the geometry/configuration products and exit.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from hadros3.camera_preview import available_backends, launch_interactive_camera_preview, render_camera_preview
from hadros3.config import deep_update, defaults, load_values, run_output_dir, safe_run_name, schema, validate_values
from hadros3.dis_sampler import generate_dis_interaction_products
from hadros3.forward_geodesics import generate_forward_geodesic_products
from hadros3.paths import camera_preview_dir, clear_dis_outputs, clear_forward_geodesics_outputs, dashboard_dir, dis_dir, ensure_output_layout, forward_geodesics_dir, geometry_dir, rel, run_metadata_dir, uhe_source_dir
from hadros3.pipeline import render_hadros_web
from hadros3.reuse import discover_original_hadros
from hadros3.uhe_source import generate_uhe_source_products


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "presets" / "hadros_web" / "default_config.json"


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
    dis_path_depths_path = dis_output_dir / "dis_path_optical_depths.jsonl"
    dis_candidates_path = dis_output_dir / "dis_interaction_candidates.jsonl"
    dis_accepted_path = dis_output_dir / "dis_accepted_interactions.jsonl"
    dis_summary_csv_path = dis_output_dir / "dis_summary.csv"
    dis_summary_path = dis_output_dir / "dis_summary.json"
    dis_tau_preview_path = dis_output_dir / "dis_tau_preview.png"
    dis_locations_path = dis_output_dir / "dis_interaction_locations.png"
    dis_locations_3d_html_path = dis_output_dir / "dis_interaction_locations_3d.html"
    dis_report_path = dis_output_dir / "dis_optical_depth_report.json"
    html_path = web_dir / "index.html"

    camera_summary: dict[str, Any] | None = None
    source_summary: dict[str, Any] | None = None
    forward_summary: dict[str, Any] | None = None
    dis_summary: dict[str, Any] | None = None
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
    return {
        "schema": schema(),
        "values": values,
        "config": str(config_path) if config_path is not None else None,
        "camera_backends": available_backends(),
        "camera_summary": camera_summary,
        "source_summary": source_summary,
        "forward_summary": forward_summary,
        "dis_summary": dis_summary,
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
            },
        },
        "pipeline_status": [
            {"stage": "Geometry", "status": "done" if geometry_preview_path.exists() else "pending", "tab": "Camera"},
            {"stage": "Camera", "status": "done" if camera_preview_path.exists() else "pending", "tab": "Camera"},
            {"stage": "UHE Source", "status": "done" if source_samples_path.exists() else "pending", "tab": "UHE Source"},
            {"stage": "Forward Geodesics", "status": "done" if forward_paths_path.exists() and forward_segments_path.exists() else "pending", "tab": "Forward Geodesics"},
            {"stage": "DIS Interaction Sampler", "status": "done" if dis_summary_path.exists() else "pending", "tab": "DIS Interaction Sampler"},
            {"stage": "Observer Bridge", "status": "pending", "tab": "Observer Bridge"},
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
            "dis_path_optical_depths_exists": dis_path_depths_path.exists(),
            "dis_interaction_candidates_exists": dis_candidates_path.exists(),
            "dis_accepted_interactions_exists": dis_accepted_path.exists(),
            "dis_summary_exists": dis_summary_csv_path.exists(),
            "dis_summary_json_exists": dis_summary_path.exists(),
            "dis_tau_preview_exists": dis_tau_preview_path.exists(),
            "dis_interaction_locations_exists": dis_locations_path.exists(),
            "dis_interaction_locations_3d_html_exists": dis_locations_3d_html_path.exists(),
            "dis_optical_depth_report_exists": dis_report_path.exists(),
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
                "dis_path_optical_depths": rel(dis_path_depths_path, output_dir),
                "dis_interaction_candidates": rel(dis_candidates_path, output_dir),
                "dis_accepted_interactions": rel(dis_accepted_path, output_dir),
                "dis_summary": rel(dis_summary_csv_path, output_dir),
                "dis_summary_json": rel(dis_summary_path, output_dir),
                "dis_tau_preview": rel(dis_tau_preview_path, output_dir),
                "dis_interaction_locations": rel(dis_locations_path, output_dir),
                "dis_interaction_locations_3d_html": rel(dis_locations_3d_html_path, output_dir),
                "dis_optical_depth_report": rel(dis_report_path, output_dir),
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
    .brand-logo {{ width: min(520px, 82vw); height: 160px; object-fit: contain; display: block; }}
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
    .run-strip {{ grid-column: 1 / -1; background: white; border: 1px solid #d6dce5; border-radius: 6px; padding: 12px 16px; display: grid; grid-template-columns: 140px minmax(240px, 420px) 110px 1fr; gap: 10px; align-items: center; }}
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
    .toggle-help {{ color: #627084; font-size: 12px; }}
    .camera-controls-card {{ margin-top: 12px; border-top: 1px solid #d6dce5; padding-top: 12px; }}
    .camera-controls-card h3 {{ margin: 0 0 8px; font-size: 14px; }}
    .camera-controls-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }}
    .camera-control-item {{ display: flex; gap: 8px; align-items: center; font-size: 12px; color: #4d5b6b; }}
    kbd {{ min-width: 70px; text-align: center; border: 1px solid #b8c3d1; border-bottom-width: 2px; border-radius: 4px; padding: 2px 5px; background: white; color: #18202a; font-family: ui-monospace, monospace; }}
    @keyframes pulse {{ 0% {{ transform: scale(1); }} 50% {{ transform: scale(1.04); box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.18); }} 100% {{ transform: scale(1); }} }}
    @media (max-width: 1080px) {{ main {{ grid-template-columns: 1fr; }} nav {{ position: static; }} }}
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
let previewResolution = state.values.observer_camera.resolution || "512x288";
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
function outPath(key) {{
  return state.outputs.paths && state.outputs.paths[key] ? state.outputs.paths[key] : key;
}}
function outUrl(key) {{
  return "/output/" + outPath(key);
}}
function inputFor(field, value) {{
  if (field.kind === "select") {{
    return `<select data-section="${{field.section}}" data-key="${{field.key}}">` +
      field.options.map(o => `<option value="${{o}}" ${{String(value) === String(o) ? "selected" : ""}}>${{o}}</option>`).join("") +
      `</select>`;
  }}
  if (field.kind === "checkbox") {{
    return `<input type="checkbox" data-section="${{field.section}}" data-key="${{field.key}}" ${{value ? "checked" : ""}}>`;
  }}
  return `<input type="${{field.kind === "number" ? "number" : "text"}}" data-section="${{field.section}}" data-key="${{field.key}}" value="${{value}}">`;
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
  const order = ["Camera", "Black Hole", "Torus / Medium", "Funnel / Cone", "UHE Source", "Forward Geodesics", "DIS Interaction Sampler", "Observer Bridge", "Event Generation", "GEANT4", "Photon Transport", "Spectra", "Outputs", "Provenance"];
  return [...state.schema].sort((a, b) => {{
    const ai = order.indexOf(tabLabel(a));
    const bi = order.indexOf(tabLabel(b));
    return (ai < 0 ? 999 : ai) - (bi < 0 ? 999 : bi);
  }});
}}
function renderFields(tab) {{
  return `<section class="active-panel"><h2>${{tabLabel(tab)}}</h2>` +
    tab.fields.map(f => `<label><span>${{f.label}}${{f.visibility === "EXPERT" ? " (Expert)" : ""}}</span>${{inputFor(f, state.values[f.section][f.key])}}</label>`).join("") +
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
  const stops = summary && summary.stop_condition_counts ? Object.entries(summary.stop_condition_counts).map(([k, v]) => `${{k}}=${{v}}`).join(", ") : "none";
  const summaryHtml = summary ? `${{inputStatus}}<div class="summary-grid">
    <div class="summary-item"><strong>Status</strong>${{summary.status}}</div>
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
  </div><p class="note"><strong>Stop conditions:</strong> ${{stops}}</p>${{forwardLinks}}` : `${{inputStatus}}<p class="note">Forward geodesics inactive. Generate UHE Source samples first, then propagate here.</p>`;
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
      ${{disInput("dis_backend")}}
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
    <div class="summary-item"><strong>Python prototype used</strong>${{String(summary.python_prototype_used)}}</div>
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
    ${{state.outputs.dis_tau_preview_exists ? `<img src="${{outUrl("dis_tau_preview")}}?v=${{disPreviewVersion}}" alt="DIS tau preview">` : ""}}
    ${{state.outputs.dis_interaction_locations_exists ? `<img src="${{outUrl("dis_interaction_locations")}}?v=${{disPreviewVersion}}" alt="DIS interaction locations">` : ""}}
  </div></section>`;
  return `<div class="source-panel">
    ${{inputHtml}}
    ${{configHtml}}
    <section><h2>Run</h2><button type="button" id="dis-sampler-button" class="source-action" ${{canRun ? "" : "disabled"}}>Compute DIS Optical Depth / Sample Interactions</button>
    <p class="note">Runs only H3-W7. Observer Bridge, POWHEG, PYTHIA, GEANT4 and photon transport remain disabled.</p></section>
    ${{resultsHtml}}
    ${{outputLinks}}
  </div>`;
}}
function renderContextPanel() {{
  if (activeTab === "Camera") {{
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
    return `<aside class="panel"><h2>Forward Geodesics Geometry</h2><div class="context-figure">${{figure}}</div></aside>`;
  }}
  if (activeTab === "DIS Interaction Sampler") {{
    const figure = state.outputs.dis_interaction_locations_3d_html_exists
      ? `<iframe class="context-interactive" src="${{outUrl("dis_interaction_locations_3d_html")}}?v=${{disPreviewVersion}}" title="DIS interaction locations"></iframe>`
      : state.outputs.dis_interaction_locations_exists
      ? `<img src="${{outUrl("dis_interaction_locations")}}?v=${{disPreviewVersion}}" alt="DIS interaction locations">`
      : `<div class="context-empty">No DIS interaction map generated yet.</div>`;
    return `<aside class="panel"><h2>DIS Interaction Map</h2><div class="context-figure">${{figure}}</div></aside>`;
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
  const runStrip = `<div class="run-strip"><label><span>Run name</span><input id="runNameInput" type="text" value="${{runName}}"></label><span>Output</span><div class="output-folder">output/${{safeRunName(runName)}}</div></div>`;
  const nav = `<nav>${{tabs.map(tab => `<button class="tab-button ${{tabLabel(tab) === activeTab ? "active" : ""}}" data-tab="${{tabLabel(tab)}}">${{tabLabel(tab)}}</button>`).join("")}}</nav>`;
  const genericFields = activeTab === "DIS Interaction Sampler" ? "" : renderFields(active);
  root.innerHTML = runStrip + nav + `<div class="panel"><p class="note">Geometry/configuration shell only. Expensive event stages are disabled.</p>${{genericFields}}${{activeTab === "Camera" ? renderHadrosCameraPanel() + renderBackendTable() : ""}}${{activeTab === "UHE Source" ? renderSourcePanel() : ""}}${{activeTab === "Forward Geodesics" ? renderForwardPanel() : ""}}${{activeTab === "DIS Interaction Sampler" ? renderDisPanel() : ""}}${{activeTab === "Outputs" ? renderOutputsPanel() : ""}}` +
    `<pre id="log"></pre></div>` +
    renderContextPanel();
  bindHadrosCameraPanel();
  const uheButton = document.querySelector("#uhe-source-button");
  if (uheButton) uheButton.onclick = sampleUheSource;
  const forwardButton = document.querySelector("#forward-geodesics-button");
  if (forwardButton) forwardButton.onclick = propagateForwardGeodesics;
  const disButton = document.querySelector("#dis-sampler-button");
  if (disButton) disButton.onclick = sampleDisInteractions;
  bindNumberInputs();
  drawGeometrySvg();
  document.querySelector("#runNameInput").addEventListener("input", event => {{
    state.values = collect();
    document.querySelector(".output-folder").textContent = "output/" + safeRunName(event.target.value);
  }});
  document.querySelectorAll("[data-section]").forEach(el => el.addEventListener("input", drawGeometrySvg));
  document.querySelectorAll("[data-section]").forEach(el => el.addEventListener("change", drawGeometrySvg));
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
            }
        return {"values": deep_update(defaults(), raw), "previewOptions": {}}

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
            raw = info.get("components", {}).get("last_camera_config")
            path = Path(raw) if raw else None
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
        if self.path == "/api/save":
            write_values(self.config_path, values)
            self._send(200, f"wrote {self.config_path}\n")
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
            write_values(self.config_path, values)
            source_summary = generate_uhe_source_products(values, output_dir=output_dir)
            clear_forward_geodesics_outputs(output_dir)
            clear_dis_outputs(output_dir)
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
            write_values(self.config_path, values)
            forward_summary = generate_forward_geodesic_products(values, run_output_dir=output_dir)
            clear_dis_outputs(output_dir)
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
            write_values(self.config_path, values)
            dis_summary = generate_dis_interaction_products(values, run_output_dir=output_dir)
            render_summary = render_hadros_web(values, root=ROOT, dis_summary=dis_summary)
            summary = {"status": "ok", "dis": dis_summary, "render": render_summary}
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
        write_values(args.config, values)
        source_summary = generate_uhe_source_products(values, output_dir=output_dir)
        clear_forward_geodesics_outputs(output_dir)
        clear_dis_outputs(output_dir)
        render_summary = render_hadros_web(values, root=ROOT, output_dir=output_dir, source_summary=source_summary)
        print(json.dumps({"status": "ok", "source": source_summary, "render": render_summary}, indent=2, sort_keys=True))
        return 0
    if args.propagate_forward_geodesics:
        output_dir = args.output_dir if args.output_dir is not None else ROOT / run_output_dir(values)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        ensure_output_layout(output_dir)
        values["forward_geodesics"]["status"] = "forward_kerr_geodesics_propagated_no_interactions"
        write_values(args.config, values)
        forward_summary = generate_forward_geodesic_products(values, run_output_dir=output_dir)
        clear_dis_outputs(output_dir)
        render_summary = render_hadros_web(values, root=ROOT, output_dir=output_dir, forward_geodesic_summary=forward_summary)
        print(json.dumps({"status": "ok", "forward": forward_summary, "render": render_summary}, indent=2, sort_keys=True))
        return 0
    if args.sample_dis_interactions:
        output_dir = args.output_dir if args.output_dir is not None else ROOT / run_output_dir(values)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        ensure_output_layout(output_dir)
        values["dis_interaction_sampler"]["status"] = "dis_optical_depth_sampled_no_observer_bridge"
        write_values(args.config, values)
        dis_summary = generate_dis_interaction_products(values, run_output_dir=output_dir)
        render_summary = render_hadros_web(values, root=ROOT, output_dir=output_dir, dis_summary=dis_summary)
        print(json.dumps({"status": "ok", "dis": dis_summary, "render": render_summary}, indent=2, sort_keys=True))
        return 0
    summary = render_hadros_web(values, root=ROOT, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
