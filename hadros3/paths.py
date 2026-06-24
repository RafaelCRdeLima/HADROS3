"""Output path layout for HADROS3 runs."""

from __future__ import annotations

import shutil
from pathlib import Path


CAMERA_PREVIEW_DIR = "CameraPreview"
GEOMETRY_DIR = "Geometry"
RUN_METADATA_DIR = "RunMetadata"
UHE_SOURCE_DIR = "UHEsource"
DASHBOARD_DIR = "Dashboard"


def camera_preview_dir(run_output: Path) -> Path:
    return run_output / CAMERA_PREVIEW_DIR


def geometry_dir(run_output: Path) -> Path:
    return run_output / GEOMETRY_DIR


def run_metadata_dir(run_output: Path) -> Path:
    return run_output / RUN_METADATA_DIR


def uhe_source_dir(run_output: Path) -> Path:
    return run_output / UHE_SOURCE_DIR


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
