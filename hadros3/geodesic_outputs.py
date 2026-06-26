"""Output writers for H3-W6 forward neutrino geodesics."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

from .medium_renderer import MediumRenderer

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hadros3_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    keys = [
        "status",
        "geodesic_backend",
        "n_samples_requested",
        "n_samples_propagated",
        "n_paths",
        "n_segments",
        "direction_model",
        "momentum_generator",
        "momentum_is_physical_kerr",
        "full_kerr_geodesic",
        "theta_phi_evolution",
        "uses_kerr_metric",
        "uses_christoffel_or_hamiltonian",
        "coordinate_radial_preview",
        "max_delta_theta_rad",
        "max_delta_phi_rad",
        "curvature_indicator_max",
        "null_norm_max",
        "killing_energy_max_error",
        "lz_max_error",
        "validation_pass",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        writer.writerow([summary.get(key) for key in keys])


def write_stop_condition_csv(counts: dict[str, int], total: int, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stop_condition", "count", "fraction"])
        for condition in ["outer_escape_radius", "horizon_crossing", "max_steps", "invalid_invariant"]:
            count = int(counts.get(condition, 0))
            writer.writerow([condition, count, count / total if total else 0.0])


def write_diagnostic_report(summary: dict[str, Any], generated_files: list[Path], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stop_counts = summary.get("stop_condition_counts", {})
    lines = [
        "# Forward Geodesics Diagnostic Report",
        "",
        "## Summary",
        "",
        f"- Input source samples: {summary.get('n_input_samples')}",
        f"- Samples requested: {summary.get('n_samples_requested')}",
        f"- Samples propagated: {summary.get('n_samples_propagated')}",
        f"- Paths: {summary.get('n_paths')}",
        f"- Segments: {summary.get('n_segments')}",
        f"- Direction model: {summary.get('direction_model')}",
        f"- Momentum generator: {summary.get('momentum_generator')}",
        f"- Geodesic backend: {summary.get('geodesic_backend')}",
        f"- full_kerr_geodesic: {summary.get('full_kerr_geodesic')}",
        f"- theta_phi_evolution: {summary.get('theta_phi_evolution')}",
        f"- coordinate_radial_preview: {summary.get('coordinate_radial_preview')}",
        "",
        "## Validation",
        "",
        f"- validation_pass: {summary.get('validation_pass')}",
        f"- null_norm_max: {summary.get('null_norm_max')}",
        f"- killing_energy_max_error: {summary.get('killing_energy_max_error')}",
        f"- lz_max_error: {summary.get('lz_max_error')}",
        f"- max_delta_theta_rad: {summary.get('max_delta_theta_rad')}",
        f"- max_delta_phi_rad: {summary.get('max_delta_phi_rad')}",
        "",
        "## Stop Condition Counts",
        "",
    ]
    if stop_counts:
        for condition, count in sorted(stop_counts.items()):
            lines.append(f"- {condition}: {count}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Source Direction Contract",
            "",
            "- Input: UHEsource/uhe_neutrino_source_samples.jsonl",
            "- Uses: position + energy + emission_direction",
            "- Builds: Kerr null four-momentum p_mu",
            f"- forward_geodesics_consumes_source_direction: {summary.get('forward_geodesics_consumes_source_direction')}",
            f"- four_momentum_constructed_from_source_direction: {summary.get('four_momentum_constructed_from_source_direction')}",
            f"- four_momentum_sampled_in_source: {summary.get('four_momentum_sampled_in_source')}",
            "",
            "## Generated Files",
            "",
        ]
    )
    for file_path in generated_files:
        lines.append(f"- {file_path.name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_forward_preview(paths: list[dict[str, Any]], segments: list[dict[str, Any]], output_path: Path, *, outer_radius_rg: float | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 9), facecolor="#101318")
    ax.set_facecolor("#101318")
    by_event: dict[str, list[dict[str, Any]]] = {}
    for segment in segments:
        by_event.setdefault(str(segment["event_id"]), []).append(segment)
    for event_segments in by_event.values():
        event_segments.sort(key=lambda item: int(item["segment_index"]))
        xs: list[float] = []
        zs: list[float] = []
        for segment in event_segments:
            if not xs:
                xs.append(float(segment["r_start_rg"]) * math.sin(float(segment["theta_start_rad"])))
                zs.append(float(segment["r_start_rg"]) * math.cos(float(segment["theta_start_rad"])))
            xs.append(float(segment["r_end_rg"]) * math.sin(float(segment["theta_end_rad"])))
            zs.append(float(segment["r_end_rg"]) * math.cos(float(segment["theta_end_rad"])))
        ax.plot(xs, zs, color="#9de2ff", alpha=0.42, linewidth=0.85)
        if xs:
            ax.scatter([xs[0]], [zs[0]], color="#ff6f59", s=14, alpha=0.82, zorder=4)
            if len(xs) > 1:
                dx = xs[min(4, len(xs) - 1)] - xs[0]
                dz = zs[min(4, len(zs) - 1)] - zs[0]
                ax.arrow(
                    xs[0],
                    zs[0],
                    dx,
                    dz,
                    color="#ffd166",
                    alpha=0.35,
                    width=0.025,
                    head_width=0.65,
                    length_includes_head=True,
                    zorder=3,
                )
    if outer_radius_rg is not None and outer_radius_rg > 0.0:
        ax.add_patch(
            plt.Circle(
                (0.0, 0.0),
                outer_radius_rg,
                fill=False,
                edgecolor="#f8fafc",
                linestyle="--",
                linewidth=1.0,
                alpha=0.62,
            )
        )
        ax.text(
            0.02,
            0.96,
            f"outer stop radius = {outer_radius_rg:g} r_g",
            transform=ax.transAxes,
            color="#f8fafc",
            fontsize=9,
            bbox=dict(facecolor="#151b24", edgecolor="#334155", alpha=0.85, pad=4),
        )
    ax.scatter([0], [0], color="black", edgecolor="white", s=180, zorder=5)
    ax.text(0, 0, "BH", color="white", ha="center", va="center", fontsize=8, zorder=6)
    max_radius = outer_radius_rg if outer_radius_rg is not None else 10.0
    if segments:
        max_radius = max(
            max(abs(float(segment["r_start_rg"])) for segment in segments),
            max(abs(float(segment["r_end_rg"])) for segment in segments),
            max_radius,
            10.0,
        )
    lim = max_radius * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#344052", alpha=0.35, linestyle="--", linewidth=0.5)
    ax.set_xlabel("x / r_g", color="#dce6f2")
    ax.set_ylabel("z / r_g", color="#dce6f2")
    ax.tick_params(colors="#dce6f2")
    ax.set_title("HADROS3 H3-W6 forward neutrino geodesics", color="#f0f4f8")
    ax.text(
        0.02,
        0.02,
        f"emission points: red\nforward paths: blue\nradial direction: gold\npaths={len(paths)} segments={len(segments)}",
        transform=ax.transAxes,
        color="#dce6f2",
        fontsize=9,
        bbox=dict(facecolor="#151b24", edgecolor="#334155", alpha=0.85, pad=4),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _spherical_to_cartesian(r_rg: float, theta_rad: float, phi_rad: float) -> tuple[float, float, float]:
    sin_theta = math.sin(theta_rad)
    return (
        r_rg * sin_theta * math.cos(phi_rad),
        r_rg * sin_theta * math.sin(phi_rad),
        r_rg * math.cos(theta_rad),
    )


def _observer_position(values: dict[str, dict[str, Any]]) -> tuple[float, float, float]:
    camera = values["observer_camera"]
    return _spherical_to_cartesian(
        float(camera["observer_distance_rg"]),
        math.radians(float(camera["inclination_deg"])),
        math.radians(float(camera["azimuth_deg"])),
    )


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm <= 0.0:
        return (0.0, 0.0, 1.0)
    return tuple(component / norm for component in vector)  # type: ignore[return-value]


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    return (a[0] * factor, a[1] * factor, a[2] * factor)


def draw_forward_geometry_3d(
    values: dict[str, dict[str, Any]],
    paths: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    output_path: Path,
    geometry_json_path: Path,
    geometry_html_path: Path | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    output_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_json_path.parent.mkdir(parents=True, exist_ok=True)
    if geometry_html_path is not None:
        geometry_html_path.parent.mkdir(parents=True, exist_ok=True)

    spin_a = float(values["black_hole"]["spin_a"])
    horizon_radius = 1.0 + math.sqrt(1.0 - max(-0.999, min(0.999, spin_a)) ** 2)
    torus = values["analytic_torus"]
    cone = values["polar_cone"]
    camera = values["observer_camera"]
    forward = values["forward_geodesics"]
    r_peak = float(torus["r_peak_rg"])
    r_inner = float(torus["r_inner_rg"])
    r_outer = float(torus["r_outer_rg"])
    cone_opening = math.radians(float(cone["opening_angle_deg"]))
    cone_r_min = float(cone["r_min_rg"])
    cone_r_max = float(cone["r_max_rg"])
    observer = _observer_position(values)
    fov = math.radians(float(camera["field_of_view_deg"]))
    observer_distance = float(camera["observer_distance_rg"])
    frustum_length = max(
        horizon_radius * 4.0,
        min(observer_distance * 0.32, max(float(forward["outer_radius_rg"]), cone_r_max, r_outer) * 0.46),
    )
    view_dir = _unit((-observer[0], -observer[1], -observer[2]))
    up_seed = (0.0, 0.0, 1.0)
    if abs(sum(view_dir[i] * up_seed[i] for i in range(3))) > 0.92:
        up_seed = (0.0, 1.0, 0.0)
    right = _unit(_cross(view_dir, up_seed))
    up = _unit(_cross(right, view_dir))
    half_width = math.tan(0.5 * fov) * frustum_length
    frustum_center = _add(observer, _scale(view_dir, frustum_length))
    frustum_corners = [
        _add(frustum_center, _add(_scale(right, sx * half_width), _scale(up, sy * half_width)))
        for sx, sy in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    ]

    by_event: dict[str, list[dict[str, Any]]] = {}
    for segment in segments:
        by_event.setdefault(str(segment["event_id"]), []).append(segment)

    view_axis = _unit((1.15, -1.35, 0.70))
    screen_x = _unit((view_axis[1], -view_axis[0], 0.0))
    screen_y = _unit(_cross(view_axis, screen_x))

    def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def project(point: tuple[float, float, float]) -> tuple[float, float]:
        return (dot(point, screen_x), dot(point, screen_y))

    def projected_line(points: list[tuple[float, float, float]]) -> tuple[list[float], list[float]]:
        projected = [project(point) for point in points]
        return ([point[0] for point in projected], [point[1] for point in projected])

    fig, ax = plt.subplots(figsize=(11, 9), facecolor="#f7f8fb")
    ax.set_facecolor("#f7f8fb")

    axis_length = max(float(forward["outer_radius_rg"]), cone_r_max, r_outer) * 1.04
    for axis, color, label in [
        ((axis_length, 0.0, 0.0), "#64748b", "x"),
        ((0.0, axis_length, 0.0), "#64748b", "y"),
        ((0.0, 0.0, axis_length), "#334155", "z / spin"),
    ]:
        xs, ys = projected_line([(0.0, 0.0, 0.0), axis])
        ax.plot(xs, ys, color=color, linewidth=0.75, alpha=0.50)
        ax.text(xs[-1], ys[-1], label, fontsize=8, color=color)

    medium_rings = MediumRenderer.proxy_shell_rings(values)
    MediumRenderer.draw_shell_3d_proxy(ax, values, project, color="#64748b", zorder=3)

    for sign in (1.0, -1.0) if cone.get("draw_mode") == "bipolar_funnel" else (1.0,):
        outer_ring = []
        inner_ring = []
        for idx in range(96):
            phi = 2.0 * math.pi * idx / 96
            outer_ring.append((cone_r_max * math.sin(cone_opening) * math.cos(phi), cone_r_max * math.sin(cone_opening) * math.sin(phi), sign * cone_r_max * math.cos(cone_opening)))
            inner_ring.append((cone_r_min * math.sin(cone_opening) * math.cos(phi), cone_r_min * math.sin(cone_opening) * math.sin(phi), sign * cone_r_min * math.cos(cone_opening)))
        outer_projected = [project(point) for point in outer_ring]
        ax.add_patch(Polygon(outer_projected, closed=True, facecolor="#facc15", edgecolor="#a16207", linewidth=0.8, alpha=0.12, zorder=0))
        xs, ys = projected_line(outer_ring + [outer_ring[0]])
        ax.plot(xs, ys, color="#a16207", linewidth=0.9, alpha=0.55, zorder=2)
        xs, ys = projected_line(inner_ring + [inner_ring[0]])
        ax.plot(xs, ys, color="#a16207", linewidth=0.65, alpha=0.35, zorder=2)
        for idx in range(0, 96, 12):
            xs, ys = projected_line([inner_ring[idx], outer_ring[idx]])
            ax.plot(xs, ys, color="#ca8a04", linewidth=0.55, alpha=0.32, zorder=2)

    max_embedded_paths = 256
    default_display_paths = min(5, len(paths))
    by_event_items = sorted(by_event.items(), key=lambda item: str(item[0]))
    if len(by_event_items) > max_embedded_paths:
        selected_indices = sorted(
            {
                round(idx * (len(by_event_items) - 1) / (max_embedded_paths - 1))
                for idx in range(max_embedded_paths)
            }
        )
        display_event_ids_list = [by_event_items[idx][0] for idx in selected_indices]
    else:
        display_event_ids_list = [event_id for event_id, _ in by_event_items]
    display_event_ids = set(display_event_ids_list)
    default_event_ids = set(display_event_ids_list[:default_display_paths])

    source_points: list[tuple[float, float, float]] = []
    for path in paths:
        if path.get("event_id") not in display_event_ids:
            continue
        position = path.get("initial_position", {})
        if position:
            source_points.append(
                _spherical_to_cartesian(
                    float(position["r_rg"]),
                    float(position["theta_rad"]),
                    float(position.get("phi_rad", 0.0)),
                )
            )
    if source_points:
        source_projected = [project(point) for point in source_points]
        ax.scatter([p[0] for p in source_projected], [p[1] for p in source_projected], color="#0ea5e9", s=20, edgecolor="#075985", linewidth=0.25, label="UHE source", zorder=7)

    path_count = 0
    paths_xyz: list[list[tuple[float, float, float]]] = []
    for event_id, event_segments in by_event_items:
        if event_id not in display_event_ids:
            continue
        event_segments.sort(key=lambda item: int(item["segment_index"]))
        xyz: list[tuple[float, float, float]] = []
        for segment in event_segments:
            if not xyz:
                xyz.append(
                    _spherical_to_cartesian(
                        float(segment["r_start_rg"]),
                        float(segment["theta_start_rad"]),
                        float(segment["phi_start_rad"]),
                    )
                )
            xyz.append(
                _spherical_to_cartesian(
                    float(segment["r_end_rg"]),
                    float(segment["theta_end_rad"]),
                    float(segment["phi_end_rad"]),
                )
            )
        if xyz:
            path_count += 1
            paths_xyz.append(xyz)
            if event_id in default_event_ids:
                xs, ys = projected_line(xyz)
                ax.plot(xs, ys, color="#2563eb", alpha=0.30, linewidth=0.8, zorder=5)

    observer_projected = project(observer)
    ax.scatter([observer_projected[0]], [observer_projected[1]], color="#16a34a", s=65, edgecolor="#14532d", linewidth=0.6, label="observer", zorder=8)
    for corner in frustum_corners:
        xs, ys = projected_line([observer, corner])
        ax.plot(xs, ys, color="#64748b", alpha=0.60, linewidth=0.85, zorder=4)
    frustum_projected = [project(corner) for corner in frustum_corners]
    ax.add_patch(Polygon(frustum_projected, closed=True, facecolor="#94a3b8", edgecolor="#475569", linewidth=0.75, alpha=0.15, zorder=1))

    ax.add_patch(Circle(project((0.0, 0.0, 0.0)), horizon_radius, facecolor="#000000", edgecolor="#111827", linewidth=0.8, zorder=9))
    ax.text(project((0.0, 0.0, 0.0))[0], project((0.0, 0.0, 0.0))[1], "BH", color="#ffffff", fontsize=8, ha="center", va="center", zorder=10)

    extents = [horizon_radius, r_outer, cone_r_max, float(forward["outer_radius_rg"]), observer_distance]
    for point in source_points + [observer] + frustum_corners:
        extents.extend(abs(component) for component in point)
    max_extent = max(extents) * 1.05
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("orthographic x / r_g")
    ax.set_ylabel("orthographic y / r_g")
    ax.grid(True, color="#cbd5e1", alpha=0.42, linewidth=0.55)
    ax.set_title("Forward neutrino geodesics with MediumRenderer density contours", pad=16)
    ax.text(
        0.02,
        0.02,
        (
            f"BH horizon, MediumRenderer density contours, polar cone\n"
            f"medium: hard radial shell; Gaussian angular profile, not a hard angular boundary\n"
            f"observer FOV={math.degrees(fov):g} deg, displayed paths={len(default_event_ids)}/{len(paths)}"
        ),
        transform=ax.transAxes,
        fontsize=9,
        color="#1f2937",
        bbox=dict(facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.86, pad=5),
    )
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)

    payload = {
        "status": "ok",
        "coordinate_system": "Cartesian rendering from Boyer-Lindquist-like r, theta, phi in r_g units",
        "black_hole": {"shape": "sphere", "radius_rg": horizon_radius, "spin_a": spin_a, "color": "black"},
        "analytic_torus": {
            "model": torus["model"],
            "r_inner_rg": r_inner,
            "r_outer_rg": r_outer,
            "r_peak_rg": r_peak,
            "half_opening_angle_deg": float(torus["half_opening_angle_deg"]),
            "surface": "MediumRenderer density-level proxy rings",
        },
        "medium_renderer": {
            **MediumRenderer.metadata(),
            "density_level_rings": [
                {
                    "points_xyz_rg": [list(point) for point in ring["points"]],
                    "alpha": ring["alpha"],
                    "density_relative": ring["density_relative"],
                    "hard_radial_cut": ring["hard_radial_cut"],
                    "label": ring["label"],
                }
                for ring in medium_rings
            ],
        },
        "polar_cone": {
            "enabled": bool(cone["enabled"]),
            "draw_mode": cone["draw_mode"],
            "opening_angle_deg": float(cone["opening_angle_deg"]),
            "r_min_rg": cone_r_min,
            "r_max_rg": cone_r_max,
        },
        "uhe_source": {"n_emission_points": len(source_points), "point_color": "blue"},
        "forward_geodesics": {
            "n_paths": len(paths),
            "n_paths_available_for_display": path_count,
            "n_paths_drawn_default": len(default_event_ids),
            "n_segments": len(segments),
            "line_color": "blue",
        },
        "source_points_xyz_rg": [list(point) for point in source_points],
        "paths_xyz_rg": [[list(point) for point in path_points] for path_points in paths_xyz],
        "observer": {
            "position_xyz_rg": list(observer),
            "distance_rg": observer_distance,
            "inclination_deg": float(camera["inclination_deg"]),
            "azimuth_deg": float(camera["azimuth_deg"]),
        },
        "fov_frustum": {
            "field_of_view_deg": float(camera["field_of_view_deg"]),
            "length_rg": frustum_length,
            "corner_xyz_rg": [list(corner) for corner in frustum_corners],
        },
        "products": {
            "geometry_3d_png": str(output_path),
            "geometry_3d_json": str(geometry_json_path),
            "geometry_3d_html": str(geometry_html_path) if geometry_html_path is not None else None,
        },
    }
    write_json(geometry_json_path, payload)
    if geometry_html_path is not None:
        write_forward_geometry_html(payload, geometry_html_path)


def write_forward_geometry_html(payload: dict[str, Any], path: Path) -> None:
    payload_json = json.dumps(payload, sort_keys=True)
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>HADROS3 Forward Geodesics Interactive Geometry</title>
  <style>
    html, body { margin: 0; height: 100%; overflow: hidden; background: #f7f8fb; font-family: system-ui, sans-serif; color: #172033; }
    #wrap { width: 100vw; height: 100vh; display: grid; grid-template-rows: minmax(0, 1fr) auto; }
    #sceneWrap { position: relative; min-height: 0; }
    canvas { width: 100%; height: 100%; display: block; cursor: grab; }
    canvas.dragging { cursor: grabbing; }
    .hud { position: absolute; left: 14px; bottom: 14px; background: rgba(255,255,255,0.88); border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 12px; font-size: 13px; line-height: 1.35; box-shadow: 0 8px 24px rgba(15,23,42,0.08); }
    .hud strong { display: block; margin-bottom: 4px; }
    .toolbar { position: absolute; right: 14px; top: 14px; display: flex; gap: 8px; }
    .path-controls { display: flex; justify-content: flex-end; gap: 8px; align-items: center; background: rgba(255,255,255,0.96); border-top: 1px solid #cbd5e1; padding: 10px 14px; box-shadow: 0 -8px 24px rgba(15,23,42,0.06); }
    .path-controls label { font-size: 13px; color: #334155; }
    .path-controls input { width: 72px; border: 1px solid #94a3b8; border-radius: 5px; padding: 6px 8px; font: inherit; }
    button { border: 1px solid #94a3b8; background: rgba(255,255,255,0.9); color: #172033; border-radius: 5px; padding: 7px 10px; font: inherit; }
  </style>
</head>
<body>
  <div id="wrap">
    <div id="sceneWrap">
      <canvas id="scene"></canvas>
      <div class="toolbar"><button id="reset" type="button">Reset</button></div>
      <div class="hud">
        <strong>Forward Geodesics Geometry</strong>
        Drag to rotate. Wheel to zoom.<br>
        BH, MediumRenderer density contours, polar cones, UHE source, observer FOV and forward neutrino paths.
      </div>
    </div>
    <div class="path-controls">
      <label for="pathLimit">Displayed neutrino paths</label>
      <input id="pathLimit" type="number" min="0" step="1">
      <button id="applyPathLimit" type="button">Apply</button>
    </div>
  </div>
  <script>
    const data = __PAYLOAD__;
    const canvas = document.getElementById("scene");
    const ctx = canvas.getContext("2d");
    let yaw = -0.74;
    let pitch = 0.36;
    let zoom = 1.0;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    const availablePathCount = (data.paths_xyz_rg || []).length;
    let displayedPathLimit = Math.min(
      data.forward_geodesics.n_paths_drawn_default || 5,
      availablePathCount
    );
    const pathLimitInput = document.getElementById("pathLimit");
    pathLimitInput.max = String(availablePathCount);
    pathLimitInput.value = String(displayedPathLimit);

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
      canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function rotatePoint(p) {
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = p[0] * cy - p[1] * sy;
      const y1 = p[0] * sy + p[1] * cy;
      const z1 = p[2];
      return [x1, y1 * cp - z1 * sp, y1 * sp + z1 * cp];
    }

    function extent() {
      const values = [
        data.black_hole.radius_rg,
        data.analytic_torus.r_outer_rg,
        data.polar_cone.r_max_rg,
        data.observer.distance_rg,
        data.fov_frustum.length_rg
      ];
      for (const p of data.source_points_xyz_rg || []) values.push(Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2]));
      for (const p of data.fov_frustum.corner_xyz_rg || []) values.push(Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2]));
      return Math.max(...values) * 1.12;
    }

    function project(p) {
      const r = rotatePoint(p);
      const scale = Math.min(canvas.clientWidth, canvas.clientHeight) / (2.15 * extent()) * zoom;
      return [canvas.clientWidth * 0.5 + r[0] * scale, canvas.clientHeight * 0.5 - r[1] * scale, r[2], scale];
    }

    function line(points, color, width, alpha) {
      if (!points.length) return;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      const p0 = project(points[0]);
      ctx.moveTo(p0[0], p0[1]);
      for (let i = 1; i < points.length; i++) {
        const p = project(points[i]);
        ctx.lineTo(p[0], p[1]);
      }
      ctx.stroke();
      ctx.restore();
    }

    function poly(points, fill, stroke, alpha) {
      if (!points.length) return;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.fillStyle = fill;
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1;
      ctx.beginPath();
      const p0 = project(points[0]);
      ctx.moveTo(p0[0], p0[1]);
      for (let i = 1; i < points.length; i++) {
        const p = project(points[i]);
        ctx.lineTo(p[0], p[1]);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    function coneRing(sign, radius) {
      const out = [];
      const theta = data.polar_cone.opening_angle_deg * Math.PI / 180;
      for (let i = 0; i <= 96; i++) {
        const phi = 2 * Math.PI * i / 96;
        out.push([radius * Math.sin(theta) * Math.cos(phi), radius * Math.sin(theta) * Math.sin(phi), sign * radius * Math.cos(theta)]);
      }
      return out;
    }

    function dotMarker(p, radius, color, stroke) {
      const q = project(p);
      ctx.save();
      ctx.fillStyle = color;
      ctx.strokeStyle = stroke || color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(q[0], q[1], radius, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    function label(p, text, color) {
      const q = project(p);
      ctx.save();
      ctx.fillStyle = color;
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText(text, q[0] + 5, q[1] - 5);
      ctx.restore();
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      ctx.fillStyle = "#f7f8fb";
      ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      const rMax = Math.max(data.polar_cone.r_max_rg, data.analytic_torus.r_outer_rg);
      line([[0,0,0], [rMax,0,0]], "#64748b", 1, 0.5);
      line([[0,0,0], [0,rMax,0]], "#64748b", 1, 0.5);
      line([[0,0,0], [0,0,rMax]], "#334155", 1.2, 0.65);
      label([rMax,0,0], "x", "#64748b");
      label([0,rMax,0], "y", "#64748b");
      label([0,0,rMax], "z / spin", "#334155");

      for (const sign of (data.polar_cone.draw_mode === "bipolar_funnel" ? [1, -1] : [1])) {
        poly(coneRing(sign, data.polar_cone.r_max_rg), "#facc15", "#a16207", 0.13);
        line(coneRing(sign, data.polar_cone.r_min_rg), "#a16207", 1, 0.40);
        for (let i = 0; i < 12; i++) {
          const phi = 2 * Math.PI * i / 12;
          const theta = data.polar_cone.opening_angle_deg * Math.PI / 180;
          const p1 = [data.polar_cone.r_min_rg * Math.sin(theta) * Math.cos(phi), data.polar_cone.r_min_rg * Math.sin(theta) * Math.sin(phi), sign * data.polar_cone.r_min_rg * Math.cos(theta)];
          const p2 = [data.polar_cone.r_max_rg * Math.sin(theta) * Math.cos(phi), data.polar_cone.r_max_rg * Math.sin(theta) * Math.sin(phi), sign * data.polar_cone.r_max_rg * Math.cos(theta)];
          line([p1, p2], "#ca8a04", 0.8, 0.34);
        }
      }

      for (const ring of (data.medium_renderer.density_level_rings || [])) {
        line(ring.points_xyz_rg, ring.hard_radial_cut ? "#475569" : "#64748b", ring.hard_radial_cut ? 1.1 : 0.75, Math.max(0.05, ring.alpha || 0.12));
      }

      const observer = data.observer.position_xyz_rg;
      const corners = data.fov_frustum.corner_xyz_rg;
      poly(corners, "#94a3b8", "#475569", 0.15);
      for (const c of corners) line([observer, c], "#64748b", 1.1, 0.60);

      for (const path of (data.paths_xyz_rg || []).slice(0, displayedPathLimit)) line(path, "#2563eb", 0.9, 0.30);
      for (const p of data.source_points_xyz_rg || []) dotMarker(p, 3.0, "#0ea5e9", "#075985");

      const bh = project([0,0,0]);
      ctx.save();
      ctx.fillStyle = "#000000";
      ctx.strokeStyle = "#111827";
      ctx.beginPath();
      ctx.arc(bh[0], bh[1], Math.max(4, data.black_hole.radius_rg * bh[3]), 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#ffffff";
      ctx.font = "11px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("BH", bh[0], bh[1]);
      ctx.restore();

      dotMarker(observer, 6, "#16a34a", "#14532d");
      label(observer, "observer", "#14532d");
      ctx.fillStyle = "#172033";
      ctx.font = "18px system-ui, sans-serif";
      ctx.fillText("HADROS3 Forward Geodesics Interactive Geometry", 18, 30);
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText(`FOV=${data.fov_frustum.field_of_view_deg} deg, displayed paths=${displayedPathLimit}/${data.forward_geodesics.n_paths}, segments=${data.forward_geodesics.n_segments}, zoom=${zoom.toFixed(2)}`, 18, 50);
      ctx.fillText("MediumRenderer: hard radial shell + Gaussian angular density levels; no hard angular boundary", 18, 70);
    }

    canvas.addEventListener("mousedown", (event) => {
      dragging = true;
      canvas.classList.add("dragging");
      lastX = event.clientX;
      lastY = event.clientY;
    });
    window.addEventListener("mouseup", () => {
      dragging = false;
      canvas.classList.remove("dragging");
    });
    window.addEventListener("mousemove", (event) => {
      if (!dragging) return;
      yaw += (event.clientX - lastX) * 0.008;
      pitch = Math.max(-1.35, Math.min(1.35, pitch + (event.clientY - lastY) * 0.008));
      lastX = event.clientX;
      lastY = event.clientY;
      draw();
    });
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      zoom = Math.max(0.25, Math.min(8.0, zoom * Math.exp(-event.deltaY * 0.001)));
      draw();
    }, { passive: false });
    document.getElementById("reset").onclick = () => {
      yaw = -0.74;
      pitch = 0.36;
      zoom = 1.0;
      draw();
    };
    document.getElementById("applyPathLimit").onclick = () => {
      const requested = Number.parseInt(pathLimitInput.value, 10);
      displayedPathLimit = Math.max(0, Math.min(availablePathCount, Number.isFinite(requested) ? requested : displayedPathLimit));
      pathLimitInput.value = String(displayedPathLimit);
      draw();
    };
    window.addEventListener("resize", resize);
    resize();
  </script>
</body>
</html>
"""
    path.write_text(html.replace("__PAYLOAD__", payload_json), encoding="utf-8")
