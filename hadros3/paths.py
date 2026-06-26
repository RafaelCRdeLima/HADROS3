"""Output path layout for HADROS3 runs."""

from __future__ import annotations

import shutil
from pathlib import Path


CAMERA_PREVIEW_DIR = "CameraPreview"
GEOMETRY_DIR = "Geometry"
RUN_METADATA_DIR = "RunMetadata"
UHE_SOURCE_DIR = "UHEsource"
FORWARD_GEODESICS_DIR = "ForwardGeodesics"
DIS_DIR = "DIS"
OBSERVER_BRIDGE_DIR = "ObserverBridge"
POWHEG_DIR = "POWHEG"
DASHBOARD_DIR = "Dashboard"


def camera_preview_dir(run_output: Path) -> Path:
    return run_output / CAMERA_PREVIEW_DIR


def geometry_dir(run_output: Path) -> Path:
    return run_output / GEOMETRY_DIR


def run_metadata_dir(run_output: Path) -> Path:
    return run_output / RUN_METADATA_DIR


def uhe_source_dir(run_output: Path) -> Path:
    return run_output / UHE_SOURCE_DIR


def forward_geodesics_dir(run_output: Path) -> Path:
    return run_output / FORWARD_GEODESICS_DIR


def dis_dir(run_output: Path) -> Path:
    return run_output / DIS_DIR


def observer_bridge_dir(run_output: Path) -> Path:
    return run_output / OBSERVER_BRIDGE_DIR


def powheg_dir(run_output: Path) -> Path:
    return run_output / POWHEG_DIR


def clear_forward_geodesics_outputs(run_output: Path) -> None:
    path = forward_geodesics_dir(run_output)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clear_dis_outputs(run_output: Path) -> None:
    path = dis_dir(run_output)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clear_observer_bridge_outputs(run_output: Path) -> None:
    path = observer_bridge_dir(run_output)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clear_powheg_outputs(run_output: Path) -> None:
    path = powheg_dir(run_output)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def dashboard_dir(run_output: Path) -> Path:
    return run_output / DASHBOARD_DIR


def rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def ensure_output_layout(run_output: Path) -> None:
    """Create subfolders and migrate known legacy root-level generated files."""
    folders = {
        CAMERA_PREVIEW_DIR: [
            "hadros3_camera_preview.png",
            "hadros3_camera_preview.ppm",
            "hadros3_camera_preview_summary.json",
            "hadros3_camera_preview_interactive_summary.json",
        ],
        GEOMETRY_DIR: [
            "hadros3_geometry_preview.png",
            "hadros3_system_schematic.png",
        ],
        RUN_METADATA_DIR: [
            "hadros3_config.json",
            "hadros3_pipeline_provenance.json",
            "hadros_web_render_summary.json",
        ],
        UHE_SOURCE_DIR: [
            "uhe_neutrino_source_samples.jsonl",
            "uhe_neutrino_source_summary.csv",
            "uhe_neutrino_source_summary.json",
            "uhe_neutrino_source_preview.png",
            "uhe_source_sampling_uniformity.png",
            "uhe_source_sampling_uniformity_report.json",
            "uhe_source_direction_uniformity.png",
            "uhe_source_direction_uniformity_report.json",
            "uhe_source_direction_sphere.png",
        ],
        FORWARD_GEODESICS_DIR: [
            "uhe_neutrino_forward_paths.jsonl",
            "uhe_neutrino_forward_path_segments.jsonl",
            "uhe_neutrino_forward_summary.csv",
            "uhe_neutrino_forward_summary.json",
            "uhe_neutrino_forward_preview.png",
            "uhe_neutrino_forward_geometry_3d.png",
            "uhe_neutrino_forward_geometry_3d.json",
            "uhe_neutrino_forward_geometry_3d.html",
            "isotropic_kerr_strong_field_diagnostic.png",
            "isotropic_kerr_strong_field_diagnostic.json",
            "geodesic_validation_report.json",
            "stop_condition_statistics.csv",
            "forward_geodesics_diagnostic_report.md",
            "validation_invariants.png",
            "kerr_bending_vs_impact_parameter.png",
            "stop_condition_distribution.png",
            "geodesic_density_map.png",
            "forward_geodesics_diagnostics_report.json",
        ],
        DIS_DIR: [
            "dis_path_optical_depths.jsonl",
            "dis_interaction_candidates.jsonl",
            "dis_accepted_interactions.jsonl",
            "dis_summary.csv",
            "dis_summary.json",
            "dis_tau_preview.png",
            "dis_interaction_locations.png",
            "dis_interaction_locations_3d.html",
            "dis_optical_depth_report.json",
            "backend_validation_report.json",
            "tau_distribution.png",
            "interaction_probability_distribution.png",
            "optical_depth_map.png",
            "optical_depth_map_3d.html",
            "medium_density_map.png",
            "interaction_location_distribution.png",
            "local_energy_distribution.png",
            "local_density_distribution.png",
            "sigma_distribution.png",
            "density_energy_sigma_correlation.png",
            "dis_diagnostics_report.json",
            "gbw_vs_iim_tau_comparison.png",
            "gbw_vs_iim_probability_comparison.png",
            "gbw_vs_iim_interaction_locations.png",
            "gbw_vs_iim_summary.json",
        ],
        OBSERVER_BRIDGE_DIR: [
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
            "observer_bridge_camera_view.png",
            "observer_bridge_camera_overlay.png",
        ],
        POWHEG_DIR: [
            "powheg_event_requests.jsonl",
            "powheg_summary.json",
            "powheg_summary.csv",
            "powheg_report.json",
            "powheg_card_preview.png",
            "powheg_energy_distribution.png",
            "powheg_job_summary.png",
        ],
        DASHBOARD_DIR: [
            "index.html",
        ],
    }
    for folder_name in folders:
        (run_output / folder_name).mkdir(parents=True, exist_ok=True)
    for folder_name, filenames in folders.items():
        target_dir = run_output / folder_name
        for filename in filenames:
            source = run_output / filename
            target = target_dir / filename
            if not source.exists() or not source.is_file():
                continue
            if target.exists():
                source.unlink()
            else:
                source.rename(target)
    legacy_interactive_dir = run_output / "interactive_camera_preview"
    target_interactive_dir = run_output / CAMERA_PREVIEW_DIR / "interactive_camera_preview"
    if legacy_interactive_dir.exists() and legacy_interactive_dir.is_dir() and not target_interactive_dir.exists():
        shutil.move(str(legacy_interactive_dir), str(target_interactive_dir))
    legacy_global_camera_dir = run_output.parent / "camera_preview"
    target_camera_dir = run_output / CAMERA_PREVIEW_DIR
    if legacy_global_camera_dir.exists() and legacy_global_camera_dir.is_dir():
        for source in legacy_global_camera_dir.iterdir():
            if not source.is_file():
                continue
            target = target_camera_dir / source.name
            if target.exists():
                source.unlink()
            else:
                source.rename(target)
        try:
            legacy_global_camera_dir.rmdir()
        except OSError:
            pass
