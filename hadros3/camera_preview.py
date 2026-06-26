"""Integration layer for HADROS3 camera preview backends.

The CUDA preview is ported into HADROS3 as ``bin/hadros3_geodesic_preview_cuda``
when CUDA is available. Legacy CPU/OpenGL preview discovery still records the
neighboring HADROS checkout, but the CUDA camera path does not require it at
runtime.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import warnings
import hashlib
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hadros3_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning)
warnings.filterwarnings("ignore", message="Tight layout not applied.*", category=UserWarning)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .render import kerr_horizon_radius_rg
from .reuse import discover_original_hadros

HADROS3_ROOT = Path(__file__).resolve().parents[1]
HADROS3_CUDA_PREVIEW_BIN = HADROS3_ROOT / "bin" / "hadros3_geodesic_preview_cuda"
PAINT_SWATCH_DISK_LABEL = "paint_swatch_disk = diagnostic visual test, not physical torus emission"
PAINT_SWATCH_DISK_DIAGNOSTIC_CONTRACT = {
    "disk_geometry": "thin_disk",
    "disk_hit_mode": "first_hit",
    "disk_r_in_rg": 9.26,
    "disk_r_out_rg": 18.70,
    "disk_thickness_rg": 0.02,
    "torus_alpha": 0.0,
    "funnel_enabled": False,
}


def _resolution(value: str) -> tuple[int, int]:
    text = str(value).strip().lower()
    if "x" in text:
        left, right = text.split("x", 1)
        return int(left), int(right)
    n = int(text)
    return n, n


def _component_path(name: str) -> Path | None:
    info = discover_original_hadros()
    raw = info.get("components", {}).get(name)
    if not raw:
        return None
    return Path(raw)


def _self_contained_cuda_preview_bin() -> Path:
    return HADROS3_CUDA_PREVIEW_BIN


def _original_hadros_root() -> Path | None:
    info = discover_original_hadros()
    raw = info.get("preferred_root")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path if path.exists() else None


def _space_safe_runtime_dir(output_dir: Path) -> Path:
    digest = hashlib.sha1(str(output_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    return Path("/tmp") / f"hadros3_camera_preview_{digest}"


def _preview_nav_mode(values: dict[str, dict[str, Any]], options: dict[str, Any] | None = None) -> str:
    raw = (options or {}).get("previewNavMode", values.get("observer_camera", {}).get("preview_nav_mode", "celestial_plus_torus_volume"))
    nav_mode = str(raw)
    if nav_mode not in {
        "celestial_plus_torus_volume",
        "detailed",
        "paint_swatch_disk",
        "first_hit_disk_debug",
        "opaque_disk_debug",
        "disk_radius_debug",
        "hit_reason",
        "hit_distance_debug",
        "volume_emissivity_debug",
    }:
        nav_mode = "celestial_plus_torus_volume"
    return nav_mode


def _paint_swatch_disk_metadata(nav_mode: str) -> dict[str, Any]:
    diagnostic = nav_mode == "paint_swatch_disk"
    metadata = {
        "nav_mode": nav_mode,
        "paint_swatch_disk_diagnostic_mode": diagnostic,
        "paint_swatch_disk_uses_forced_thin_disk": diagnostic,
        "paint_swatch_disk_physical_torus_emission": False,
    }
    if diagnostic:
        metadata.update(PAINT_SWATCH_DISK_DIAGNOSTIC_CONTRACT)
        metadata["paint_swatch_disk_label"] = PAINT_SWATCH_DISK_LABEL
    return metadata


def _interactive_make_command(
    values: dict[str, dict[str, Any]], output_dir: Path, preview_options: dict[str, Any] | None = None
) -> tuple[list[str], str, str, Path | None, dict[str, Any]]:
    """Build the same interactive preview command used by HADROS config-web."""
    options = preview_options or {}
    mode = str(values["observer_camera"].get("camera_preview_mode", "kerr_like_cuda"))
    preview_nx, preview_ny = _resolution(
        str(options.get("previewResolution", values["observer_camera"].get("preview_final_resolution", "512x288")))
    )
    interactive_nx, interactive_ny = _resolution(
        str(options.get("previewInteractiveResolution", values["observer_camera"].get("preview_resolution", "256x144")))
    )
    geodesic_model = str(options.get("previewGeodesicModel", "full_kerr" if mode == "full_kerr" else "kerr_like"))
    if geodesic_model not in {"kerr_like", "full_kerr"}:
        geodesic_model = "kerr_like"
    backend = "cpu" if mode == "analytic_geometry_only" else "cuda"
    quality = str(options.get("previewQuality", values["observer_camera"].get("preview_quality", "medium")))
    sky_mode = str(options.get("previewSkyMode", "texture"))
    nav_mode = _preview_nav_mode(values, options)
    preview_metadata = _paint_swatch_disk_metadata(nav_mode)
    preview_r_max_rg = max(
        float(values["polar_cone"]["r_max_rg"]),
        float(values["analytic_torus"]["r_outer_rg"]),
        2.0 * float(options.get("previewCelestialRadiusRs", 40) or 40),
    )
    opaque_structures = "1" if str(options.get("previewOpaqueStructures", "0")).lower() in {"1", "true", "yes", "on"} else "0"
    make_output_dir = _space_safe_runtime_dir(output_dir)
    preview_output_dir = make_output_dir / "interactive_camera_preview"
    torus_sigma = 0.5 * (float(values["analytic_torus"]["r_outer_rg"]) - float(values["analytic_torus"]["r_inner_rg"]))
    disk_r_in = float(values["analytic_torus"]["r_inner_rg"])
    disk_r_out = float(values["analytic_torus"]["r_outer_rg"])
    disk_thickness = max(0.02, 0.02 * float(values["analytic_torus"]["r_peak_rg"]))
    torus_alpha = 0.09
    funnel_enabled = bool(values["polar_cone"]["enabled"])
    if preview_metadata["paint_swatch_disk_diagnostic_mode"]:
        disk_r_in = PAINT_SWATCH_DISK_DIAGNOSTIC_CONTRACT["disk_r_in_rg"]
        disk_r_out = PAINT_SWATCH_DISK_DIAGNOSTIC_CONTRACT["disk_r_out_rg"]
        disk_thickness = PAINT_SWATCH_DISK_DIAGNOSTIC_CONTRACT["disk_thickness_rg"]
        torus_alpha = PAINT_SWATCH_DISK_DIAGNOSTIC_CONTRACT["torus_alpha"]
        funnel_enabled = PAINT_SWATCH_DISK_DIAGNOSTIC_CONTRACT["funnel_enabled"]
    torus_h = float(values["analytic_torus"]["r_peak_rg"]) * math.tan(
        math.radians(float(values["analytic_torus"]["half_opening_angle_deg"]))
    )
    allow_expensive = "0"
    if geodesic_model == "full_kerr":
        preview_nx = min(preview_nx, 512)
        preview_ny = min(preview_ny, 288)
        interactive_nx = min(interactive_nx, 256)
        interactive_ny = min(interactive_ny, 144)
        quality = "medium"

    if backend == "cuda":
        cuda_bin = _self_contained_cuda_preview_bin()
        command = [
            str(cuda_bin),
            "--out",
            str(preview_output_dir / "geodesic_preview_cuda.ppm"),
            "--nx",
            str(preview_nx),
            "--ny",
            str(preview_ny),
            "--interactive-nx",
            str(interactive_nx),
            "--interactive-ny",
            str(interactive_ny),
            "--quality",
            quality,
            "--allow-expensive-preview",
            allow_expensive,
            "--nav-mode",
            nav_mode,
            "--aspect-mode",
            "window",
            "--sky-mode",
            sky_mode,
            "--sky",
            "assets/sky/eso0932a.ppm",
            "--geodesic-model",
            geodesic_model,
            "--spin",
            f"{float(values['black_hole']['spin_a']):.12g}",
            "--spin-convention",
            "thorne",
            "--inclination",
            f"{float(values['observer_camera']['inclination_deg']):.12g}",
            "--fov",
            f"{float(values['observer_camera']['field_of_view_deg']):.12g}",
            "--r-obs",
            f"{float(values['observer_camera']['observer_distance_rg']):.12g}",
            "--r-max",
            f"{preview_r_max_rg:.12g}",
            "--disk-r-in",
            f"{disk_r_in:.12g}",
            "--disk-r-out",
            f"{disk_r_out:.12g}",
            "--disk-thickness",
            f"{disk_thickness:.12g}",
            "--near-clip",
            "1.0",
            "--disk-geometry",
            "thin_disk",
            "--disk-hit-mode",
            "first_hit",
            "--torus-r0",
            f"{float(values['analytic_torus']['r_peak_rg']):.12g}",
            "--torus-sigma-r",
            f"{torus_sigma:.12g}",
            "--torus-h",
            f"{torus_h:.12g}",
            "--torus-alpha",
            f"{torus_alpha:.12g}",
            "--torus-max-alpha-step",
            "0.055",
            "--torus-emissivity-cutoff",
            "1e-8",
            "--funnel",
            "1" if funnel_enabled else "0",
            "--funnel-theta",
            f"{float(values['polar_cone']['opening_angle_deg']):.12g}",
            "--funnel-width",
            "8",
            "--funnel-alpha",
            "0.07",
            "--funnel-brightness",
            "0.85",
            "--opaque-structures",
            opaque_structures,
            "--live",
            "1",
            "--vsync",
            "0",
            "--rot-speed",
            "55",
            "--zoom-speed",
            "18",
            "--fov-speed",
            "35",
        ]
        return command, geodesic_model, backend, None, preview_metadata

    hadros_root = _original_hadros_root()
    command = [
        "make",
        "-C",
        str(hadros_root) if hadros_root is not None else "",
        "geodesic_preview",
        f"OUTPUT_DIR={make_output_dir.as_posix()}",
        f"PLOT_DIR={(make_output_dir / 'plots').as_posix()}",
        f"PREVIEW_OUTPUT_DIR={preview_output_dir.resolve().as_posix()}",
        f"PREVIEW_BACKEND={backend}",
        f"PREVIEW_GEODESIC_MODEL={geodesic_model}",
        "PREVIEW_SPIN_CONVENTION=thorne",
        f"PREVIEW_NAV_MODE={nav_mode}",
        "PREVIEW_LIVE=1",
        "PREVIEW_VSYNC=0",
        "PREVIEW_ASPECT_MODE=window",
        f"PREVIEW_SKY_MODE={sky_mode}",
        "SKY_TEXTURE=assets/sky/eso0932a.ppm",
        f"PREVIEW_R_MAX_RG={preview_r_max_rg:.12g}",
        f"PREVIEW_NX={preview_nx}",
        f"PREVIEW_NY={preview_ny}",
        f"PREVIEW_INTERACTIVE_NX={interactive_nx}",
        f"PREVIEW_INTERACTIVE_NY={interactive_ny}",
        f"PREVIEW_QUALITY={quality}",
        f"PREVIEW_ALLOW_EXPENSIVE={allow_expensive}",
        f"PREVIEW_OPAQUE_STRUCTURES={opaque_structures}",
        "PREVIEW_DISK_GEOMETRY=thin_disk",
        "PREVIEW_DISK_HIT_MODE=first_hit",
        f"PREVIEW_DISK_R_IN_RG={disk_r_in:.12g}",
        f"PREVIEW_DISK_R_OUT_RG={disk_r_out:.12g}",
        f"PREVIEW_DISK_THICKNESS_RG={disk_thickness:.12g}",
        "PREVIEW_NEAR_CLIP_RG=1.0",
        "PREVIEW_TORUS_MAX_ALPHA_STEP=0.055",
        "PREVIEW_TORUS_EMISSIVITY_CUTOFF=1e-8",
        f"ASPIN={float(values['black_hole']['spin_a']):.12g}",
        f"CAM_R_OBS_RG={float(values['observer_camera']['observer_distance_rg']):.12g}",
        f"CAM_THETA_DEG={float(values['observer_camera']['inclination_deg']):.12g}",
        f"CAM_FOV_DEG={float(values['observer_camera']['field_of_view_deg']):.12g}",
        f"PREVIEW_TORUS_R0_RG={float(values['analytic_torus']['r_peak_rg']):.12g}",
        f"PREVIEW_TORUS_SIGMA_R_RG={torus_sigma:.12g}",
        f"PREVIEW_TORUS_H_RG={torus_h:.12g}",
        f"PREVIEW_TORUS_ALPHA={torus_alpha:.12g}",
        "PREVIEW_FUNNEL_ENABLED=1" if funnel_enabled else "PREVIEW_FUNNEL_ENABLED=0",
        f"PREVIEW_FUNNEL_THETA_DEG={float(values['polar_cone']['opening_angle_deg']):.12g}",
        "PREVIEW_FUNNEL_WIDTH_DEG=8",
        "PREVIEW_FUNNEL_ALPHA=0.07",
        "PREVIEW_FUNNEL_BRIGHTNESS=0.85",
        "PREVIEW_ROT_SPEED=55",
        "PREVIEW_ZOOM_SPEED=18",
        "PREVIEW_FOV_SPEED=35",
    ]
    return command, geodesic_model, backend, hadros_root, preview_metadata


def _camera_preview_args(
    values: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    interactive: bool,
    preview_options: dict[str, Any] | None = None,
) -> tuple[list[str], str, Path | None, dict[str, Any]]:
    mode = str(values["observer_camera"].get("camera_preview_mode", "analytic_geometry_only"))
    nx, ny = _resolution(
        str((preview_options or {}).get("previewResolution", values["observer_camera"].get("preview_final_resolution", "512x288")))
    )
    r_max = str(max(float(values["polar_cone"]["r_max_rg"]), float(values["analytic_torus"]["r_outer_rg"]), 80.0))
    torus_h = str(float(values["analytic_torus"]["r_peak_rg"]) * math.tan(math.radians(float(values["analytic_torus"]["half_opening_angle_deg"]))))
    nav_mode = _preview_nav_mode(values, preview_options)
    preview_metadata = _paint_swatch_disk_metadata(nav_mode)
    disk_r_in = float(values["analytic_torus"]["r_inner_rg"])
    disk_r_out = float(values["analytic_torus"]["r_outer_rg"])
    disk_thickness = max(0.02, 0.02 * float(values["analytic_torus"]["r_peak_rg"]))
    torus_alpha = 0.09
    funnel_enabled = bool(values["polar_cone"]["enabled"])
    if preview_metadata["paint_swatch_disk_diagnostic_mode"]:
        disk_r_in = PAINT_SWATCH_DISK_DIAGNOSTIC_CONTRACT["disk_r_in_rg"]
        disk_r_out = PAINT_SWATCH_DISK_DIAGNOSTIC_CONTRACT["disk_r_out_rg"]
        disk_thickness = PAINT_SWATCH_DISK_DIAGNOSTIC_CONTRACT["disk_thickness_rg"]
        torus_alpha = PAINT_SWATCH_DISK_DIAGNOSTIC_CONTRACT["torus_alpha"]
        funnel_enabled = PAINT_SWATCH_DISK_DIAGNOSTIC_CONTRACT["funnel_enabled"]
    if mode in {"kerr_like_cuda", "full_kerr"}:
        cuda_bin = _self_contained_cuda_preview_bin()
        geodesic_model = "full_kerr" if mode == "full_kerr" else "kerr_like"
        command = [
            str(cuda_bin),
            "--nx",
            str(nx),
            "--ny",
            str(ny),
            "--interactive-nx",
            str(min(nx, 256)),
            "--interactive-ny",
            str(min(ny, 144)),
            "--quality",
            str(values["observer_camera"].get("preview_quality", "fast")),
            "--geodesic-model",
            geodesic_model,
            "--spin",
            str(values["black_hole"]["spin_a"]),
            "--spin-convention",
            "thorne",
            "--inclination",
            str(values["observer_camera"]["inclination_deg"]),
            "--azimuth",
            str(values["observer_camera"]["azimuth_deg"]),
            "--fov",
            str(values["observer_camera"]["field_of_view_deg"]),
            "--r-obs",
            str(values["observer_camera"]["observer_distance_rg"]),
            "--r-max",
            r_max,
            "--nav-mode",
            nav_mode,
            "--aspect-mode",
            "window" if interactive else "fixed",
            "--sky-mode",
            "texture",
            "--sky",
            "assets/sky/eso0932a.ppm",
            "--torus-r0",
            str(values["analytic_torus"]["r_peak_rg"]),
            "--torus-sigma-r",
            str(0.5 * (float(values["analytic_torus"]["r_outer_rg"]) - float(values["analytic_torus"]["r_inner_rg"]))),
            "--torus-h",
            torus_h,
            "--torus-alpha",
            f"{torus_alpha:.12g}",
            "--disk-geometry",
            "thin_disk",
            "--disk-hit-mode",
            "first_hit",
            "--disk-r-in",
            f"{disk_r_in:.12g}",
            "--disk-r-out",
            f"{disk_r_out:.12g}",
            "--disk-thickness",
            f"{disk_thickness:.12g}",
            "--funnel",
            "1" if funnel_enabled else "0",
            "--funnel-theta",
            str(values["polar_cone"]["opening_angle_deg"]),
            "--funnel-width",
            "8",
            "--out",
            str(output_path),
        ]
        if interactive:
            command += ["--live", "1", "--vsync", "0", "--rot-speed", "55", "--zoom-speed", "18", "--fov-speed", "35"]
        else:
            command.insert(1, "--headless")
        return command, geodesic_model, cuda_bin, preview_metadata

    cpu_bin = _component_path("geodesic_preview_bin")
    command = [
        str(cpu_bin) if cpu_bin is not None else "",
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--quality",
        str(values["observer_camera"].get("preview_quality", "fast")),
        "--spin",
        str(values["black_hole"]["spin_a"]),
        "--inclination",
        str(values["observer_camera"]["inclination_deg"]),
        "--azimuth",
        str(values["observer_camera"]["azimuth_deg"]),
        "--fov",
        str(values["observer_camera"]["field_of_view_deg"]),
        "--r-obs",
        str(values["observer_camera"]["observer_distance_rg"]),
        "--r-max",
        r_max,
        "--mode",
        "combined",
    ]
    if not interactive:
        command.insert(1, "--headless")
    return command, "legacy_cpu_geodesic_preview", cpu_bin, preview_metadata


def available_backends() -> dict[str, Any]:
    cuda = _self_contained_cuda_preview_bin()
    cpu = _component_path("geodesic_preview_bin")
    cuda_ok = cuda.exists() and cuda.is_file()
    cpu_ok = cpu is not None and cpu.exists() and cpu.is_file()
    return {
        "analytic_geometry_only": {
            "available": True,
            "backend": "hadros3_analytic_camera_placeholder",
            "reason": "Always available; schematic observer-view fallback.",
        },
        "kerr_like_cuda": {
            "available": cuda_ok,
            "backend": str(cuda),
            "reason": "Uses self-contained HADROS3 bin/hadros3_geodesic_preview_cuda with --geodesic-model kerr_like.",
        },
        "full_kerr": {
            "available": cuda_ok,
            "backend": str(cuda),
            "reason": "Uses self-contained HADROS3 bin/hadros3_geodesic_preview_cuda with --geodesic-model full_kerr.",
        },
        "legacy_cpu_geodesic_preview": {
            "available": cpu_ok,
            "backend": str(cpu) if cpu is not None else None,
            "reason": "Detected for provenance; not exposed as a HADROS3 mode yet.",
        },
    }


def _draw_analytic_camera_preview(
    values: dict[str, dict[str, Any]],
    path: Path,
    message: str | None = None,
    preview_options: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx, ny = _resolution(
        str((preview_options or {}).get("previewResolution", values["observer_camera"].get("preview_final_resolution", "512x288")))
    )
    spin = float(values["black_hole"]["spin_a"])
    fov = float(values["observer_camera"]["field_of_view_deg"])
    inc = float(values["observer_camera"]["inclination_deg"])
    horizon = kerr_horizon_radius_rg(spin)

    x = np.linspace(-1.0, 1.0, max(nx, 16))
    y = np.linspace(-1.0, 1.0, max(ny, 16))
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx * xx + yy * yy)
    shadow = np.exp(-((rr / 0.28) ** 8))
    torus = np.exp(-((np.abs(yy + 0.06 * math.cos(math.radians(inc))) - 0.18) / 0.075) ** 2) * np.exp(-((xx / 0.85) ** 6))
    funnel = np.exp(-((np.abs(xx) - 0.18) / 0.08) ** 2) * np.exp(-((yy - 0.18) / 0.55) ** 2)
    sky = 0.22 + 0.22 * np.cos(8.0 * xx) * np.cos(6.0 * yy)

    img = np.zeros((y.size, x.size, 3), dtype=float)
    img[..., 2] = 0.20 + 0.26 * sky + 0.22 * torus
    img[..., 1] = 0.20 + 0.18 * sky + 0.32 * torus + 0.24 * funnel
    img[..., 0] = 0.12 + 0.12 * sky + 0.72 * torus + 0.55 * funnel
    img *= (1.0 - 0.92 * shadow[..., None])
    img = np.clip(img, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(nx / 100.0, ny / 100.0), dpi=100, facecolor="#0b1018")
    ax.imshow(img, origin="lower", extent=(-1, 1, -1, 1))
    ax.add_patch(plt.Circle((0, 0), 0.19 * horizon / 1.6, color="black", ec="white", lw=0.9, alpha=0.96))
    ax.text(-0.96, 0.88, "HADROS3 analytic camera preview", color="white", fontsize=10, ha="left")
    ax.text(-0.96, 0.76, f"a={spin:g}  inc={inc:g} deg  FOV={fov:g} deg", color="#d6e4f0", fontsize=8, ha="left")
    if message and not bool((preview_options or {}).get("suppressMessage", False)):
        ax.text(
            0.0,
            -0.93,
            message,
            color="#ffd166",
            fontsize=8,
            ha="center",
            va="bottom",
            bbox=dict(facecolor="#151b24", edgecolor="#6b5b26", alpha=0.88, pad=4),
        )
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _ppm_to_png(ppm_path: Path, png_path: Path) -> None:
    data = plt.imread(ppm_path)
    plt.imsave(png_path, data)


def render_camera_preview(
    values: dict[str, dict[str, Any]],
    *,
    root: Path,
    output_dir: Path,
    preview_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = str(values["observer_camera"].get("camera_preview_mode", "analytic_geometry_only"))
    png_path = output_dir / "hadros3_camera_preview.png"
    summary_path = output_dir / "hadros3_camera_preview_summary.json"
    ppm_path = output_dir / "hadros3_camera_preview.ppm"
    backends = available_backends()
    reused = discover_original_hadros()
    fallback_used = False
    status = "ok"
    message = ""
    command: list[str] | None = None
    cuda_used = False
    full_kerr_used = False
    backend_used = "hadros3_analytic_camera_placeholder"
    camera_preview_cuda_self_contained = mode in {"kerr_like_cuda", "full_kerr"}
    camera_preview_external_hadros_used = False
    preview_metadata = _paint_swatch_disk_metadata(_preview_nav_mode(values, preview_options))

    if mode == "analytic_geometry_only":
        message = "Analytic geometry-only observer view; no Kerr CUDA backend invoked."
        _draw_analytic_camera_preview(values, png_path, None, preview_options=preview_options)
    else:
        cuda_info = backends.get(mode, {})
        geodesic_model = "full_kerr" if mode == "full_kerr" else "kerr_like"
        command, _, cuda_bin, preview_metadata = _camera_preview_args(values, ppm_path, interactive=False, preview_options=preview_options)
        if not cuda_info.get("available") or cuda_bin is None or not cuda_bin.exists():
            status = "fallback"
            fallback_used = True
            message = "HADROS3 CUDA preview unavailable: bin/hadros3_geodesic_preview_cuda was not found."
            _draw_analytic_camera_preview(values, png_path, message, preview_options=preview_options)
        else:
            try:
                env = os.environ.copy()
                env["HADROS_PREVIEW_OUTPUT_DIR"] = str(output_dir)
                subprocess.run(
                    command,
                    cwd=root,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=45,
                    check=True,
                    env=env,
                )
                _ppm_to_png(ppm_path, png_path)
                backend_used = str(cuda_bin)
                cuda_used = True
                full_kerr_used = mode == "full_kerr"
                message = f"Rendered with self-contained HADROS3 CUDA preview: {geodesic_model}."
            except subprocess.CalledProcessError as exc:
                status = "fallback"
                fallback_used = True
                detail = (exc.stdout or str(exc)).strip().splitlines()
                detail_text = detail[-1] if detail else str(exc)
                message = f"HADROS3 CUDA preview unavailable: {detail_text}"
                _draw_analytic_camera_preview(values, png_path, message, preview_options=preview_options)
            except subprocess.TimeoutExpired as exc:
                status = "fallback"
                fallback_used = True
                message = f"HADROS3 CUDA preview unavailable: timed out after {exc.timeout} seconds."
                _draw_analytic_camera_preview(values, png_path, message, preview_options=preview_options)
            except Exception as exc:
                status = "fallback"
                fallback_used = True
                message = f"HADROS3 CUDA preview unavailable: {exc}"
                _draw_analytic_camera_preview(values, png_path, message, preview_options=preview_options)

    summary = {
        "status": status,
        "requested_mode": mode,
        "backend_used": backend_used,
        "cuda_used": cuda_used,
        "camera_preview_cuda_self_contained": camera_preview_cuda_self_contained,
        "camera_preview_external_hadros_used": camera_preview_external_hadros_used,
        "full_kerr_used": full_kerr_used,
        "fallback_used": fallback_used,
        "message": message,
        "available_backends": backends,
        "reused_hadros_components": reused,
        "command": command,
        "outputs": {
            "png": str(png_path),
            "ppm": str(ppm_path) if ppm_path.exists() else None,
            "summary": str(summary_path),
            "performance_log": str(output_dir / "performance_log.txt") if (output_dir / "performance_log.txt").exists() else None,
        },
    }
    summary.update(preview_metadata)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def launch_interactive_camera_preview(
    values: dict[str, dict[str, Any]],
    *,
    root: Path,
    output_dir: Path,
    preview_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    interactive_dir = output_dir / "interactive_camera_preview"
    interactive_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = _space_safe_runtime_dir(output_dir)
    runtime_preview_dir = runtime_dir / "interactive_camera_preview"
    log_path = interactive_dir / "camera_preview_interactive.log"
    summary_path = output_dir / "hadros3_camera_preview_interactive_summary.json"
    mode = str(values["observer_camera"].get("camera_preview_mode", "analytic_geometry_only"))
    command, geodesic_model, backend, hadros_root, preview_metadata = _interactive_make_command(values, output_dir, preview_options)
    backends = available_backends()
    status = "launched"
    message = (
        "Launched the HADROS3 self-contained CUDA preview."
        if backend == "cuda"
        else "Launched the original HADROS make geodesic_preview target. "
    ) + (
        " Use arrows/mouse to orbit, +/- or mouse wheel to change distance, [] to change FOV, "
        "A/D to change spin, R to render, S to save when supported, and Q/Esc to quit."
    )
    pid = None
    fallback_used = mode == "analytic_geometry_only"
    fallback_reason = "CPU/OpenGL HADROS preview requested by analytic_geometry_only mode." if fallback_used else None
    if backend == "cuda" and not Path(command[0]).exists():
        status = "unavailable"
        fallback_used = True
        fallback_reason = "HADROS3 CUDA preview unavailable: bin/hadros3_geodesic_preview_cuda was not found."
        message = fallback_reason
    elif backend != "cuda" and (hadros_root is None or not (hadros_root / "Makefile").exists()):
        status = "unavailable"
        message = "Original HADROS checkout with Makefile was not found next to HADROS3."
    else:
        env = os.environ.copy()
        env["HADROS_PREVIEW_OUTPUT_DIR"] = str(interactive_dir)
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            proc = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True, env=env)
            pid = proc.pid
    summary = {
        "status": status,
        "requested_mode": mode,
        "geodesic_model": geodesic_model,
        "backend_used": backend,
        "backend_launcher": "hadros3_geodesic_preview_cuda" if backend == "cuda" else "original_hadros_make_geodesic_preview",
        "camera_preview_cuda_self_contained": backend == "cuda",
        "camera_preview_external_hadros_used": backend != "cuda",
        "original_hadros_root": str(hadros_root) if hadros_root is not None else None,
        "cuda_requested": backend == "cuda",
        "full_kerr_requested": mode == "full_kerr",
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "pid": pid,
        "message": message,
        "command": command,
        "preview_options": preview_options or {},
        "log": str(log_path),
        "output_dir": str(runtime_preview_dir),
        "hadros3_log_dir": str(interactive_dir),
        "space_safe_runtime_dir": str(runtime_dir),
        "saved_camera_dir": str((hadros_root / "configs" / "cameras") if hadros_root is not None else output_dir / "configs" / "cameras"),
        "available_backends": backends,
        "reused_hadros_components": discover_original_hadros(),
    }
    summary.update(preview_metadata)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
