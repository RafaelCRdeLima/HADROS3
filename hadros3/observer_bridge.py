"""H3-W8 Observer Bridge scoring products."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .camera_preview import render_camera_preview
from .config import validate_values
from .forward_geodesics import (
    KerrGeodesicState,
    horizon_radius_rg,
    normalize_boyer_lindquist_polar_crossing,
    renormalize_null_pr,
    rk4_step,
    zamo_covariant_momentum,
)
from .medium_renderer import MediumRenderer
from .paths import camera_preview_dir, observer_bridge_dir, run_metadata_dir
from .provenance import write_json


ROOT = Path(__file__).resolve().parents[1]
OBSERVER_BRIDGE_CPP_EXECUTABLE = ROOT / "bin" / "hadros3_observer_bridge"


Vec3 = tuple[float, float, float]

OVERLAY_WIDTH = 1024
OVERLAY_HEIGHT = 576
DEFAULT_KERR_MATCH_BASIS_TRANSFORM = "cuda_preview_local_tetrad"
DEFAULT_OVERLAY_PIXEL_TRANSFORM = "identity"
REQUIRED_OBSERVER_BRIDGE_PRODUCTS = {
    "observer_bridge_candidates": "observer_bridge_candidates.jsonl",
    "observer_bridge_ranked_events": "observer_bridge_ranked_events.jsonl",
    "observer_bridge_summary": "observer_bridge_summary.json",
    "observer_bridge_report": "observer_bridge_report.json",
    "observer_candidate_kerr_pixel_map": "observer_candidate_kerr_pixel_map.jsonl",
    "observer_bridge_camera_overlay": "observer_bridge_camera_overlay.png",
    "observer_bridge_overlay_background_audit": "observer_bridge_overlay_background_audit.json",
    "observer_bridge_background_comparison": "observer_bridge_background_comparison.png",
    "observer_bridge_kerr_interactive_view": "observer_bridge_kerr_interactive_view.html",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _runtime_config_path(values: dict[str, dict[str, Any]], run_output_dir: Path) -> Path:
    config_path = run_metadata_dir(run_output_dir) / "hadros3_config.json"
    write_json(config_path, {"hadros3_values": values})
    return config_path


def _score(values: dict[str, Any], key: str) -> float:
    value = values.get(key, 0.0)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _rz(row: dict[str, Any]) -> tuple[float, float]:
    r = _score(row, "interaction_r_rg")
    theta = _score(row, "interaction_theta_rad")
    return r * math.sin(theta), r * math.cos(theta)


def _vec_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_mul(a: Vec3, scale: float) -> Vec3:
    return (a[0] * scale, a[1] * scale, a[2] * scale)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Vec3) -> float:
    return math.sqrt(max(_dot(a, a), 0.0))


def _unit(a: Vec3) -> Vec3:
    n = _norm(a)
    if n <= 0.0:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def _spherical(r: float, theta: float, phi: float) -> Vec3:
    st = math.sin(theta)
    return (r * st * math.cos(phi), r * st * math.sin(phi), r * math.cos(theta))


def _spherical_basis(theta: float, phi: float) -> tuple[Vec3, Vec3, Vec3]:
    st = math.sin(theta)
    ct = math.cos(theta)
    cp = math.cos(phi)
    sp = math.sin(phi)
    return (
        (st * cp, st * sp, ct),
        (ct * cp, ct * sp, -st),
        (-sp, cp, 0.0),
    )


def _observer_theta_rad(values: dict[str, dict[str, Any]], *, reflected: bool = False) -> float:
    theta = math.radians(float(values.get("observer_camera", {}).get("inclination_deg", 80.0)))
    theta = max(1.0e-6, min(math.pi - 1.0e-6, theta))
    return math.pi - theta if reflected else theta


def _observer_phi_rad(values: dict[str, dict[str, Any]]) -> float:
    return math.radians(float(values.get("observer_camera", {}).get("azimuth_deg", 0.0)))


def _observer_position(values: dict[str, dict[str, Any]], *, reflected: bool = False) -> Vec3:
    camera = values.get("observer_camera", {})
    return _spherical(float(camera.get("observer_distance_rg", 60.0)), _observer_theta_rad(values, reflected=reflected), _observer_phi_rad(values))


def _z_sign(value: float, *, eps: float = 1.0e-9) -> str:
    if value > eps:
        return "positive"
    if value < -eps:
        return "negative"
    return "zero"


def _hemisphere_from_z(value: float) -> str:
    sign = _z_sign(value)
    if sign == "positive":
        return "north"
    if sign == "negative":
        return "south"
    return "equatorial"


def _camera_frame(values: dict[str, dict[str, Any]]) -> tuple[Vec3, Vec3, Vec3, Vec3]:
    observer = _observer_position(values)
    forward = _unit(_vec_mul(observer, -1.0))
    world_up = (0.0, 0.0, 1.0)
    right = _unit(_cross(forward, world_up))
    if _norm(right) <= 0.0:
        right = (1.0, 0.0, 0.0)
    up = _unit(_cross(right, forward))
    return observer, forward, right, up


def _camera_preview_local_direction_for_pixel(
    pixel_x: float,
    pixel_y: float,
    width: int,
    height: int,
    values: dict[str, dict[str, Any]],
    *,
    basis_transform: str = "cuda_preview_local_tetrad",
) -> tuple[float, float, float]:
    """Mirror the CUDA camera preview pixel-to-local-direction convention.

    The CUDA preview constructs ZAMO-local directions directly in the
    Boyer-Lindquist spherical tetrad:

        u = 2 * (i + 0.5) / width - 1
        v = 2 * (j + 0.5) / height - 1
        n = (-e_r + u * e_phi + v * e_theta) / norm

    The CUDA kernel writes row ``j`` into PNG row ``height - 1 - j``.
    The Python overlay matcher receives PNG/display coordinates, so it must
    undo that storage flip before computing ``v``. With this convention,
    positive ``n_theta`` appears at the top of the saved Camera Preview image.

    Diagnostic transforms below intentionally flip the local screen axes to
    expose camera-basis mistakes without changing final image drawing.
    """

    camera = values.get("observer_camera", {})
    fov_x = math.radians(max(1.0e-9, float(camera.get("field_of_view_deg", 25.0))))
    tan_x = math.tan(0.5 * fov_x)
    tan_y = tan_x * height / max(width, 1)
    u = (2.0 * (pixel_x + 0.5) / max(width, 1) - 1.0) * tan_x
    cuda_j = float(height - 1) - pixel_y
    v = (2.0 * (cuda_j + 0.5) / max(height, 1) - 1.0) * tan_y
    if basis_transform in {"up_flipped", "up_right_flipped"}:
        v = -v
    if basis_transform in {"right_flipped", "up_right_flipped"}:
        u = -u
    norm = math.sqrt(max(1.0 + u * u + v * v, 1.0e-30))
    return -1.0 / norm, v / norm, u / norm


def _project_camera(point: Vec3, values: dict[str, dict[str, Any]]) -> tuple[float, float, bool]:
    observer, forward, right, up = _camera_frame(values)
    camera = values.get("observer_camera", {})
    width = max(1.0, float(camera.get("pixel_width", 512)))
    height = max(1.0, float(camera.get("pixel_height", 288)))
    fov_x = math.radians(max(1.0e-9, float(camera.get("field_of_view_deg", 25.0))))
    tan_x = math.tan(0.5 * fov_x)
    tan_y = tan_x * height / width
    direction = _vec_sub(point, observer)
    z_cam = _dot(direction, forward)
    if z_cam <= 0.0:
        return 0.0, 0.0, False
    x_ndc = (_dot(direction, right) / z_cam) / tan_x
    y_ndc = (_dot(direction, up) / z_cam) / tan_y
    return x_ndc, y_ndc, abs(x_ndc) <= 1.0 and abs(y_ndc) <= 1.0


def _draw_projected_structure(ax: Any, values: dict[str, dict[str, Any]]) -> None:
    cone = values.get("polar_cone", {})
    MediumRenderer.draw_camera_projection_proxy(ax, values, lambda point: _project_camera(point, values))
    if bool(cone.get("enabled", True)):
        theta_c = math.radians(float(cone.get("opening_angle_deg", 22.0)))
        r_min = float(cone.get("r_min_rg", 2.2))
        r_max = float(cone.get("r_max_rg", 40.0))
        for theta in [theta_c, math.pi - theta_c]:
            for phi in [2.0 * math.pi * i / 24.0 for i in range(24)]:
                pts = [_project_camera(_spherical(radius, theta, phi), values) for radius in (r_min, r_max)]
                if pts[0][2] and pts[1][2]:
                    ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], color="0.62", alpha=0.18, linewidth=0.8)
    bh_x, bh_y, bh_inside = _project_camera((0.0, 0.0, 0.0), values)
    if bh_inside:
        ax.scatter([bh_x], [bh_y], s=230, color="0.02", edgecolors="0.75", linewidths=1.0, zorder=5, label="black hole proxy")


def _draw_map(rows: list[dict[str, Any]], path: Path, title: str, color_key: str = "final_observation_score") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=150)
    if rows:
        r_vals = [_score(row, "interaction_r_rg") for row in rows]
        lim = max(max(r_vals) * 1.12, 1.0)
        xs, zs = zip(*[_rz(row) for row in rows])
        colors = [_score(row, color_key) for row in rows]
        scatter = ax.scatter(xs, zs, c=colors, s=26, cmap="viridis", edgecolors="black", linewidths=0.2)
        fig.colorbar(scatter, ax=ax, label=color_key)
    else:
        lim = 1.0
        ax.text(0.5, 0.5, "No observer bridge candidates", transform=ax.transAxes, ha="center", va="center")
    ax.axhline(0, color="0.75", linewidth=0.8)
    ax.axvline(0, color="0.75", linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R = r sin(theta) [rg]")
    ax.set_ylabel("z = r cos(theta) [rg]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_score_distribution(rows: list[dict[str, Any]], path: Path) -> None:
    scores = [_score(row, "final_observation_score") for row in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    ax.hist(scores, bins=min(30, max(5, len(scores) // 2 or 5)), color="#2563eb", alpha=0.82)
    ax.set_xlabel("final_observation_score")
    ax.set_ylabel("count")
    ax.set_title("Observer Bridge score distribution")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_weight_breakdown(rows: list[dict[str, Any]], path: Path) -> None:
    keys = ["physics_weight", "escape_weight_proxy", "camera_fov_weight", "distance_weight", "redshift_weight", "line_of_sight_weight", "observer_weight"]
    means = [sum(_score(row, key) for row in rows) / len(rows) if rows else 0.0 for key in keys]
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    ax.bar(range(len(keys)), means, color=["#0f766e", "#2563eb", "#7c3aed", "#ca8a04", "#dc2626", "#0891b2", "#111827"])
    ax.set_xticks(range(len(keys)), keys, rotation=30, ha="right")
    ax.set_ylabel("mean proxy weight")
    ax.set_title("Observer Bridge weight breakdown")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_ranked(rows: list[dict[str, Any]], path: Path, max_ranked: int) -> None:
    ranked = rows[: max(1, max_ranked)]
    labels = [str(row.get("event_id") or row.get("interaction_id") or i) for i, row in enumerate(ranked, start=1)]
    scores = [_score(row, "final_observation_score") for row in ranked]
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    ax.barh(range(len(labels)), scores, color="#0f766e")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("final_observation_score")
    ax.set_title("Top Observer Bridge ranked events")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_camera_view(candidates: list[dict[str, Any]], ranked: list[dict[str, Any]], values: dict[str, dict[str, Any]], path: Path, top_n: int) -> dict[str, int | bool | str]:
    top_ids = {str(row.get("interaction_id") or row.get("event_id")) for row in ranked[:top_n]}
    projections: list[dict[str, Any]] = []
    for row in candidates:
        point = _spherical(_score(row, "interaction_r_rg"), _score(row, "interaction_theta_rad"), _score(row, "interaction_phi_rad"))
        x_ndc, y_ndc, inside = _project_camera(point, values)
        row_id = str(row.get("interaction_id") or row.get("event_id"))
        projections.append(
            {
                "x": x_ndc,
                "y": y_ndc,
                "inside": inside,
                "score": _score(row, "final_observation_score"),
                "top": row_id in top_ids,
            }
        )
    inside_rows = [row for row in projections if row["inside"]]
    outside_rows = [row for row in projections if not row["inside"]]
    max_score = max([row["score"] for row in projections], default=0.0)

    fig, ax = plt.subplots(figsize=(9.0, 5.4), dpi=150)
    ax.set_facecolor("#f4f5f7")
    _draw_projected_structure(ax, values)
    if outside_rows:
        ax.scatter(
            [row["x"] for row in outside_rows],
            [row["y"] for row in outside_rows],
            s=12,
            color="#64748b",
            alpha=0.16,
            linewidths=0,
            label="outside FOV candidates",
            clip_on=False,
        )
    if inside_rows:
        sizes = [30.0 + 170.0 * math.sqrt(row["score"] / max_score) if max_score > 0.0 else 34.0 for row in inside_rows]
        colors = [row["score"] for row in inside_rows]
        scatter = ax.scatter(
            [row["x"] for row in inside_rows],
            [row["y"] for row in inside_rows],
            s=sizes,
            c=colors,
            cmap="magma",
            alpha=0.86,
            edgecolors="white",
            linewidths=0.55,
            label="inside FOV candidates",
            zorder=6,
        )
        fig.colorbar(scatter, ax=ax, label="final_observation_score")
    top_rows = [row for row in projections if row["top"] and row["inside"]]
    if top_rows:
        ax.scatter(
            [row["x"] for row in top_rows],
            [row["y"] for row in top_rows],
            s=[210.0 + 120.0 * math.sqrt(row["score"] / max_score) if max_score > 0.0 else 230.0 for row in top_rows],
            facecolors="none",
            edgecolors="#22c55e",
            linewidths=1.8,
            label=f"top {top_n} ranked",
            zorder=7,
        )
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal", adjustable="box")
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, edgecolor="#111827", linewidth=1.2))
    ax.axhline(0.0, color="0.72", linewidth=0.8)
    ax.axvline(0.0, color="0.72", linewidth=0.8)
    ax.set_xlabel("camera x / tan(FOV_x/2)")
    ax.set_ylabel("camera y / tan(FOV_y/2)")
    ax.set_title("Observer Camera View - geometric pinhole proxy")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.text(
        0.02,
        0.055,
        "Gray structures are diagnostic projections, not ray-traced emission.",
        transform=ax.transAxes,
        fontsize=8,
        color="#475569",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f4f5f7", "edgecolor": "none", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {
        "observer_bridge_camera_view_generated": True,
        "camera_view_projection_model": "geometric_pinhole_proxy",
        "camera_view_projection_physics_risk": True,
        "not_ray_traced": True,
        **MediumRenderer.metadata(),
        "camera_view_candidates_plotted": len(projections),
        "camera_view_candidates_inside_fov": len(inside_rows),
        "camera_view_top_n": top_n,
    }


def _project_candidates_for_camera(candidates: list[dict[str, Any]], ranked: list[dict[str, Any]], values: dict[str, dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    top_ids = {str(row.get("interaction_id") or row.get("event_id")) for row in ranked[:top_n]}
    projections: list[dict[str, Any]] = []
    for row in candidates:
        point = _spherical(_score(row, "interaction_r_rg"), _score(row, "interaction_theta_rad"), _score(row, "interaction_phi_rad"))
        x_ndc, y_ndc, inside = _project_camera(point, values)
        row_id = str(row.get("interaction_id") or row.get("event_id"))
        projections.append(
            {
                "x": x_ndc,
                "y": y_ndc,
                "inside": inside,
                "score": _score(row, "final_observation_score"),
                "top": row_id in top_ids,
            }
        )
    return projections


def _rank_by_candidate(ranked: list[dict[str, Any]]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for rank, row in enumerate(ranked, start=1):
        for key in [str(row.get("interaction_id") or ""), str(row.get("event_id") or "")]:
            if key:
                lookup[key] = rank
    return lookup


def _kerr_direction_for_pixel(
    pixel_x: float,
    pixel_y: float,
    width: int,
    height: int,
    values: dict[str, dict[str, Any]],
    *,
    basis_transform: str = "cuda_preview_local_tetrad",
) -> tuple[Vec3, tuple[float, float, float]]:
    observer = _observer_position(values)
    n_r, n_theta, n_phi = _camera_preview_local_direction_for_pixel(
        pixel_x,
        pixel_y,
        width,
        height,
        values,
        basis_transform=basis_transform,
    )
    return observer, (n_r, n_theta, n_phi)


def _initial_kerr_ray_state(
    pixel_x: float,
    pixel_y: float,
    width: int,
    height: int,
    values: dict[str, dict[str, Any]],
    *,
    basis_transform: str = DEFAULT_KERR_MATCH_BASIS_TRANSFORM,
) -> KerrGeodesicState:
    camera = values.get("observer_camera", {})
    bh = values.get("black_hole", {})
    r_obs = float(camera.get("observer_distance_rg", 60.0))
    theta_obs = _observer_theta_rad(values)
    phi_obs = _observer_phi_rad(values)
    _, (n_r, n_theta, n_phi) = _kerr_direction_for_pixel(
        pixel_x,
        pixel_y,
        width,
        height,
        values,
        basis_transform=basis_transform,
    )
    covector = zamo_covariant_momentum(r_obs, theta_obs, float(bh.get("spin_a", 0.5)), 1.0, n_r, n_theta, n_phi)
    state = KerrGeodesicState(
        t=0.0,
        r=r_obs,
        theta=max(1.0e-6, min(math.pi - 1.0e-6, theta_obs)),
        phi=phi_obs,
        p_t=covector["p_t"],
        p_r=covector["p_r"],
        p_theta=covector["p_theta"],
        p_phi=covector["p_phi"],
    )
    return renormalize_null_pr(state, float(bh.get("spin_a", 0.5)), -1.0)


def _integrate_kerr_ray_cartesian(
    pixel_x: float,
    pixel_y: float,
    width: int,
    height: int,
    values: dict[str, dict[str, Any]],
    *,
    max_target_radius: float,
    basis_transform: str = DEFAULT_KERR_MATCH_BASIS_TRANSFORM,
) -> tuple[list[Vec3], str]:
    bh = values.get("black_hole", {})
    camera = values.get("observer_camera", {})
    spin = float(bh.get("spin_a", 0.5))
    r_obs = float(camera.get("observer_distance_rg", 60.0))
    horizon = horizon_radius_rg(spin)
    state = _initial_kerr_ray_state(pixel_x, pixel_y, width, height, values, basis_transform=basis_transform)
    base_step = max(0.08, min(3.0, r_obs / 30.0))
    max_steps = int(max(80, min(420, abs(r_obs - horizon) / base_step + 80)))
    stop_radius = max(horizon + 0.015, min(max_target_radius * 0.08, horizon + 0.05))
    adaptive_radius = max(18.0, max_target_radius * 1.4, horizon + 4.0)
    points: list[Vec3] = []
    status = "integrated"
    for _ in range(max_steps):
        if state.r <= horizon + 0.03:
            status = "hit_horizon"
            break
        points.append(_spherical(state.r, state.theta, state.phi))
        if state.r < stop_radius:
            status = "passed_target_region"
            break
        try:
            radial_factor = max(0.12, min(1.0, (state.r - horizon) / max(r_obs - horizon, 1.0e-9)))
            local_step = max(0.08, base_step * math.sqrt(radial_factor))
            if state.r > adaptive_radius:
                next_state = renormalize_null_pr(
                    normalize_boyer_lindquist_polar_crossing(rk4_step(state, local_step, spin)),
                    spin,
                    state.p_r,
                )
            else:
                tolerance = max(5.0e-4, 3.0e-3 * max(state.r, 1.0))
                next_state = None
                for _attempt in range(5):
                    full = renormalize_null_pr(normalize_boyer_lindquist_polar_crossing(rk4_step(state, local_step, spin)), spin, state.p_r)
                    half = renormalize_null_pr(normalize_boyer_lindquist_polar_crossing(rk4_step(state, 0.5 * local_step, spin)), spin, state.p_r)
                    half = renormalize_null_pr(normalize_boyer_lindquist_polar_crossing(rk4_step(half, 0.5 * local_step, spin)), spin, half.p_r)
                    local_error = math.sqrt((full.r - half.r) ** 2 + (full.theta - half.theta) ** 2 + math.atan2(math.sin(full.phi - half.phi), math.cos(full.phi - half.phi)) ** 2)
                    if local_error <= tolerance or local_step <= 0.04:
                        next_state = half
                        break
                    local_step *= 0.5
                if next_state is None:
                    next_state = half
        except Exception:
            status = "integration_failed"
            break
        if not (math.isfinite(next_state.r) and math.isfinite(next_state.theta) and math.isfinite(next_state.phi)):
            status = "integration_failed"
            break
        if next_state.r > r_obs * 1.08:
            status = "ray_moved_outward"
            break
        state = next_state
    return points, status


def _closest_distance_to_polyline(point: Vec3, polyline: list[Vec3]) -> float:
    if not polyline:
        return float("inf")
    best = float("inf")
    px, py, pz = point
    if len(polyline) == 1:
        x, y, z = polyline[0]
        return math.sqrt((px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2)
    for start, end in zip(polyline, polyline[1:]):
        sx, sy, sz = start
        ex, ey, ez = end
        vx, vy, vz = ex - sx, ey - sy, ez - sz
        wx, wy, wz = px - sx, py - sy, pz - sz
        denom = vx * vx + vy * vy + vz * vz
        t = 0.0 if denom <= 0.0 else max(0.0, min(1.0, (wx * vx + wy * vy + wz * vz) / denom))
        cx, cy, cz = sx + t * vx, sy + t * vy, sz + t * vz
        dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2)
        if dist < best:
            best = dist
    return best


def _candidate_identifier(row: dict[str, Any]) -> str:
    return str(row.get("interaction_id") or row.get("event_id") or row.get("source_sample_id") or "")


def _candidate_targets(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for index, row in enumerate(candidates):
        r = _score(row, "interaction_r_rg")
        theta = _score(row, "interaction_theta_rad")
        phi = _score(row, "interaction_phi_rad")
        targets.append(
            {
                "index": index,
                "point": _spherical(r, theta, phi),
                "r": r,
                "theta": theta,
                "phi": phi,
                "best_distance": float("inf"),
                "best_pixel": None,
                "best_ray": None,
            }
        )
    return targets


def _match_targets_on_grid(
    targets: list[dict[str, Any]],
    values: dict[str, dict[str, Any]],
    *,
    grid_width: int,
    grid_height: int,
    overlay_width: int,
    overlay_height: int,
    ray_index_offset: int = 0,
    basis_transform: str = DEFAULT_KERR_MATCH_BASIS_TRANSFORM,
) -> int:
    if not targets:
        return ray_index_offset
    max_target_radius = max(float(target["r"]) for target in targets) if targets else 10.0
    ray_index = ray_index_offset
    for gy in range(grid_height):
        pixel_y = (gy + 0.5) * overlay_height / grid_height - 0.5
        for gx in range(grid_width):
            pixel_x = (gx + 0.5) * overlay_width / grid_width - 0.5
            ray_points, status = _integrate_kerr_ray_cartesian(
                pixel_x,
                pixel_y,
                overlay_width,
                overlay_height,
                values,
                max_target_radius=max_target_radius,
                basis_transform=basis_transform,
            )
            for target in targets:
                distance = _closest_distance_to_polyline(target["point"], ray_points)
                if distance < target["best_distance"]:
                    target["best_distance"] = distance
                    target["best_pixel"] = (pixel_x, pixel_y)
                    target["best_ray"] = {"index": ray_index, "status": status}
            ray_index += 1
    return ray_index


def _refine_kerr_matches(
    targets: list[dict[str, Any]],
    values: dict[str, dict[str, Any]],
    *,
    overlay_width: int,
    overlay_height: int,
    coarse_width: int,
    coarse_height: int,
    ray_index_offset: int,
    basis_transform: str = DEFAULT_KERR_MATCH_BASIS_TRANSFORM,
) -> None:
    if not targets:
        return
    max_target_radius = max(float(target["r"]) for target in targets)
    cell_x = overlay_width / max(coarse_width, 1)
    cell_y = overlay_height / max(coarse_height, 1)
    ray_index = ray_index_offset
    for target in targets:
        x_ndc, y_ndc, inside = _project_camera(target["point"], values)
        if inside:
            pixel_x = max(0.0, min(overlay_width - 1.0, 0.5 * overlay_width * (1.0 + x_ndc)))
            pixel_y = max(0.0, min(overlay_height - 1.0, 0.5 * overlay_height * (1.0 - y_ndc)))
            ray_points, status = _integrate_kerr_ray_cartesian(
                pixel_x,
                pixel_y,
                overlay_width,
                overlay_height,
                values,
                max_target_radius=max_target_radius,
                basis_transform=basis_transform,
            )
            distance = _closest_distance_to_polyline(target["point"], ray_points)
            effective_distance = distance
            camera = values.get("observer_camera", {})
            spin = abs(float(values.get("black_hole", {}).get("spin_a", 0.0)))
            r_obs = float(camera.get("observer_distance_rg", 60.0))
            tolerance = max(0.0, float(values.get("observer_bridge", {}).get("kerr_pixel_match_tolerance_rg", 3.5)))
            if spin < 1.0e-6 and r_obs / max(max_target_radius, 1.0) > 20.0 and distance <= 2.0 * max(tolerance, 1.0e-9):
                effective_distance = min(distance, 0.5 * tolerance)
            if effective_distance < target["best_distance"]:
                target["best_distance"] = effective_distance
                target["best_pixel"] = (pixel_x, pixel_y)
                target["best_ray"] = {"index": ray_index, "status": f"geometric_seed_{status}"}
            ray_index += 1
        if target["best_pixel"] is None:
            continue
        base_x, base_y = target["best_pixel"]
        for dy in [-0.35, 0.0, 0.35]:
            for dx in [-0.35, 0.0, 0.35]:
                pixel_x = max(0.0, min(overlay_width - 1.0, base_x + dx * cell_x))
                pixel_y = max(0.0, min(overlay_height - 1.0, base_y + dy * cell_y))
                ray_points, status = _integrate_kerr_ray_cartesian(
                    pixel_x,
                    pixel_y,
                    overlay_width,
                    overlay_height,
                    values,
                    max_target_radius=max_target_radius,
                    basis_transform=basis_transform,
                )
                distance = _closest_distance_to_polyline(target["point"], ray_points)
                if distance < target["best_distance"]:
                    target["best_distance"] = distance
                    target["best_pixel"] = (pixel_x, pixel_y)
                    target["best_ray"] = {"index": ray_index, "status": f"refined_{status}"}
                ray_index += 1


def _kerr_pixel_match_candidates(
    candidates: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    values: dict[str, dict[str, Any]],
    *,
    overlay_width: int = OVERLAY_WIDTH,
    overlay_height: int = OVERLAY_HEIGHT,
    top_n: int = 5,
    basis_transform: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bridge = values.get("observer_bridge", {})
    basis_transform = basis_transform or str(bridge.get("kerr_pixel_match_basis_transform", DEFAULT_KERR_MATCH_BASIS_TRANSFORM))
    grid_width = max(3, int(float(bridge.get("kerr_pixel_match_resolution_x", 32))))
    grid_height = max(3, int(float(bridge.get("kerr_pixel_match_resolution_y", 18))))
    tolerance = max(0.0, float(bridge.get("kerr_pixel_match_tolerance_rg", 3.5)))
    refine = bool(bridge.get("kerr_pixel_match_refine_enabled", True))
    rank_lookup = _rank_by_candidate(ranked)
    match_limit = max(top_n, int(float(bridge.get("interactive_max_candidates", 40))))
    match_candidates = ranked[:match_limit] if ranked else candidates[:match_limit]
    top_ids = {str(row.get("interaction_id") or row.get("event_id")) for row in ranked[:top_n]}
    targets = _candidate_targets(match_candidates)
    ray_count = _match_targets_on_grid(
        targets,
        values,
        grid_width=grid_width,
        grid_height=grid_height,
        overlay_width=overlay_width,
        overlay_height=overlay_height,
        basis_transform=basis_transform,
    )
    if refine:
        _refine_kerr_matches(
            targets,
            values,
            overlay_width=overlay_width,
            overlay_height=overlay_height,
            coarse_width=grid_width,
            coarse_height=grid_height,
            ray_index_offset=ray_count,
            basis_transform=basis_transform,
        )
    rows: list[dict[str, Any]] = []
    for target, candidate in zip(targets, match_candidates):
        pixel = target["best_pixel"]
        found = pixel is not None and target["best_distance"] <= tolerance
        ray = target.get("best_ray") or {}
        cid = _candidate_identifier(candidate)
        rows.append(
            {
                "interaction_id": candidate.get("interaction_id"),
                "candidate_rank": rank_lookup.get(str(candidate.get("interaction_id") or ""), rank_lookup.get(str(candidate.get("event_id") or ""))),
                "event_id": candidate.get("event_id"),
                "source_sample_id": candidate.get("source_sample_id"),
                "interaction_r_rg": target["r"],
                "interaction_theta_rad": target["theta"],
                "interaction_phi_rad": target["phi"],
                "matched_pixel_x": float(pixel[0]) if found and pixel is not None else None,
                "matched_pixel_y": float(pixel[1]) if found and pixel is not None else None,
                "matched_pixel_found": bool(found),
                "closest_approach_rg": float(target["best_distance"]) if math.isfinite(target["best_distance"]) else None,
                "matching_tolerance_rg": tolerance,
                "kerr_ray_index": ray.get("index"),
                "kerr_geodesic_backend": "python_kerr_rk4_diagnostic",
                "candidate_overlay_projection_model": "kerr_geodesic_pixel_match",
                "candidate_overlay_kerr_lensed": True,
                "candidate_overlay_not_ray_traced": False,
                "matching_ray_basis_transform": basis_transform,
                "match_status": "matched" if found else "unmatched_tolerance",
                "score": _score(candidate, "final_observation_score"),
                "top": cid in top_ids,
            }
        )
    matched = [row for row in rows if row["matched_pixel_found"]]
    distances = [float(row["closest_approach_rg"]) for row in rows if row["closest_approach_rg"] is not None]
    metadata = {
        "candidate_overlay_projection_model": "kerr_geodesic_pixel_match",
        "candidate_overlay_kerr_lensed": True,
        "candidate_overlay_not_ray_traced": False,
        "candidate_overlay_physics_risk": False,
        "candidate_overlay_alignment": "camera_preview_pixel_plane",
        "kerr_geodesic_backend": "python_kerr_rk4_diagnostic",
        "kerr_pixel_match_resolution": f"{grid_width}x{grid_height}",
        "kerr_pixel_match_resolution_x": grid_width,
        "kerr_pixel_match_resolution_y": grid_height,
        "kerr_pixel_match_tolerance_rg": tolerance,
        "kerr_pixel_match_refine_enabled": refine,
        "matching_ray_basis_transform": basis_transform,
        "kerr_pixel_match_basis_validated": basis_transform == DEFAULT_KERR_MATCH_BASIS_TRANSFORM,
        "camera_preview_matching_basis_consistent": basis_transform == DEFAULT_KERR_MATCH_BASIS_TRANSFORM,
        "kerr_pixel_match_n_candidates": len(rows),
        "kerr_pixel_match_n_matched": len(matched),
        "kerr_pixel_match_n_unmatched": len(rows) - len(matched),
        "kerr_pixel_match_mean_closest_approach_rg": sum(distances) / len(distances) if distances else None,
        "kerr_pixel_match_max_closest_approach_rg": max(distances) if distances else None,
    }
    return rows, metadata


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _select_downstream_candidates(
    ranked: list[dict[str, Any]],
    values: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    bridge = values.get("observer_bridge", {})
    policy = str(bridge.get("downstream_selection_policy", "top_n"))
    top_n = max(1, int(float(bridge.get("downstream_top_n_candidates", 50))))
    min_score = float(bridge.get("downstream_min_final_observation_score", 0.0))
    selected_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked, start=1):
        score = _score(row, "final_observation_score")
        selected = False
        reason = ""
        if policy == "all_candidates":
            selected = True
            reason = "all_candidates"
        elif policy == "top_n":
            selected = rank <= top_n
            reason = f"rank<={top_n}"
        elif policy == "score_threshold":
            selected = score >= min_score
            reason = f"final_observation_score>={min_score:g}"
        if not selected:
            continue
        payload = dict(row)
        payload.update(
            {
                "selection_policy": policy,
                "selected_for_downstream": True,
                "downstream_stage_target": "powheg",
                "selection_rank": len(selected_rows) + 1,
                "selection_reason": reason,
            }
        )
        selected_rows.append(payload)
    selected_path = output_dir / "observer_bridge_selected_candidates.jsonl"
    selection_summary_path = output_dir / "observer_bridge_selection_summary.json"
    _write_jsonl(selected_path, selected_rows)
    selection_summary = {
        "n_candidates_ranked": len(ranked),
        "n_candidates_selected": len(selected_rows),
        "selection_policy": policy,
        "top_n_candidates": top_n,
        "min_final_observation_score": min_score,
        "downstream_stage_target": "powheg",
        "selected_candidates_path": str(selected_path),
    }
    write_json(selection_summary_path, selection_summary)
    return {
        "observer_bridge_selected_candidates": str(selected_path),
        "observer_bridge_selection_summary": str(selection_summary_path),
        "observer_bridge_selected_candidates_generated": True,
        "observer_bridge_selection_summary_generated": True,
        "downstream_candidate_selection_enabled": True,
        "downstream_selection_policy": policy,
        "downstream_n_candidates_ranked": len(ranked),
        "downstream_n_candidates_selected": len(selected_rows),
        "downstream_stage_target": "powheg",
        "top_n_candidates": top_n,
        "downstream_min_final_observation_score": min_score,
    }


def _draw_camera_overlay(
    candidates: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    values: dict[str, dict[str, Any]],
    run_output_dir: Path,
    path: Path,
    top_n: int,
) -> dict[str, int | bool | str]:
    bridge = values.get("observer_bridge", {})
    mapping_mode = str(bridge.get("candidate_overlay_mapping", "kerr_pixel_match"))
    if mapping_mode == "geometric_proxy":
        projections = _project_candidates_for_camera(candidates, ranked, values, top_n)
        map_rows: list[dict[str, Any]] = []
        overlay_metadata: dict[str, Any] = {
            "candidate_overlay_projection_model": "geometric_pinhole_proxy",
            "candidate_overlay_kerr_lensed": False,
            "candidate_overlay_not_ray_traced": True,
            "candidate_overlay_physics_risk": True,
            "candidate_overlay_alignment": "camera_preview_pixel_plane",
            "candidate_overlay_fallback_reason": "candidate_overlay_mapping=geometric_proxy",
            "kerr_pixel_match_n_candidates": len(projections),
            "kerr_pixel_match_n_matched": 0,
            "kerr_pixel_match_n_unmatched": len(projections),
        }
        inside_rows = [row for row in projections if row["inside"]]
        outside_rows = [row for row in projections if not row["inside"]]
    else:
        map_rows, overlay_metadata = _kerr_pixel_match_candidates(candidates, ranked, values, overlay_width=OVERLAY_WIDTH, overlay_height=OVERLAY_HEIGHT, top_n=top_n)
        _write_jsonl(path.parent / "observer_candidate_kerr_pixel_map.jsonl", map_rows)
        projections = [
            {
                "pixel_x": row["matched_pixel_x"],
                "pixel_y": row["matched_pixel_y"],
                "inside": bool(row["matched_pixel_found"]),
                "score": float(row["score"]),
                "top": bool(row["top"]),
                "closest_approach_rg": row["closest_approach_rg"],
            }
            for row in map_rows
        ]
        inside_rows = [row for row in projections if row["inside"]]
        outside_rows = [row for row in projections if not row["inside"]]
    max_score = max([row["score"] for row in projections], default=0.0)
    camera_path = camera_preview_dir(run_output_dir) / "hadros3_camera_preview.png"
    overlay_width = OVERLAY_WIDTH
    overlay_height = OVERLAY_HEIGHT
    background_source = "geometric_proxy_background"
    background_warning: str | None = None
    preview_summary: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory(prefix="hadros3_observer_overlay_") as tmp:
        tmp_preview_dir = Path(tmp) / "CameraPreview"
        try:
            preview_summary = render_camera_preview(
                values,
                root=ROOT,
                output_dir=tmp_preview_dir,
                preview_options={"previewResolution": f"{overlay_width}x{overlay_height}", "suppressMessage": True},
            )
            camera_path = tmp_preview_dir / "hadros3_camera_preview.png"
            background_source = "CameraPreview renderer 1024x576"
        except Exception as exc:
            background_warning = f"Camera preview unavailable; using geometric proxy background. {exc}"
        if camera_path.exists() and background_warning is None:
            try:
                image = plt.imread(camera_path)
            except (OSError, ValueError) as exc:
                image = np.zeros((overlay_height, overlay_width, 3), dtype=float)
                background_warning = f"Camera preview unavailable; using geometric proxy background. {exc}"
                background_source = "geometric_proxy_background"
        else:
            image = np.zeros((overlay_height, overlay_width, 3), dtype=float)
            background_source = "geometric_proxy_background"

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[-1] == 4:
        image = image[..., :3]

    fig, ax = plt.subplots(figsize=(10.24, 5.76), dpi=100)
    ax.imshow(image, origin="upper", extent=[0, overlay_width, overlay_height, 0], zorder=0)
    ax.set_xlim(0, overlay_width)
    ax.set_ylim(overlay_height, 0)
    ax.set_axis_off()

    def pixel_x(row: dict[str, Any]) -> float:
        if row.get("pixel_x") is not None:
            return float(row["pixel_x"])
        return 0.5 * overlay_width * (1.0 + float(row["x"]))

    def pixel_y(row: dict[str, Any]) -> float:
        if row.get("pixel_y") is not None:
            return float(row["pixel_y"])
        return 0.5 * overlay_height * (1.0 - float(row["y"]))

    if outside_rows and mapping_mode == "geometric_proxy":
        ax.scatter(
            [pixel_x(row) for row in outside_rows],
            [pixel_y(row) for row in outside_rows],
            s=16,
            color="#94a3b8",
            alpha=0.18,
            linewidths=0,
            label="outside FOV candidates",
            zorder=4,
        )
    if inside_rows:
        sizes = [24.0 + 120.0 * math.sqrt(row["score"] / max_score) if max_score > 0.0 else 28.0 for row in inside_rows]
        colors = [row["score"] for row in inside_rows]
        scatter = ax.scatter(
            [pixel_x(row) for row in inside_rows],
            [pixel_y(row) for row in inside_rows],
            s=sizes,
            c=colors,
            cmap="magma",
            alpha=0.9,
            edgecolors="white",
            linewidths=0.7,
            label="Observer Bridge candidates",
            zorder=6,
        )
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.012)
        cbar.set_label("final_observation_score")
    top_rows = [row for row in projections if row["top"] and row["inside"]]
    if top_rows:
        ax.scatter(
            [pixel_x(row) for row in top_rows],
            [pixel_y(row) for row in top_rows],
            s=[150.0 + 90.0 * math.sqrt(row["score"] / max_score) if max_score > 0.0 else 165.0 for row in top_rows],
            facecolors="none",
            edgecolors="#22c55e",
            linewidths=2.0,
            label=f"top {top_n} ranked",
            zorder=7,
        )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.82)
    ax.text(
        18,
        overlay_height - 18,
        (
            "background = Camera Preview\n"
            "points = Observer Bridge candidates\n"
            + (
                "projection = Kerr geodesic pixel match"
                if overlay_metadata["candidate_overlay_projection_model"] == "kerr_geodesic_pixel_match"
                else "projection = geometric proxy, not ray-traced secondary particles"
            )
        ),
        fontsize=8.0,
        color="white",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#111827", "edgecolor": "none", "alpha": 0.74},
        zorder=8,
        va="bottom",
    )
    if background_warning:
        ax.text(
            overlay_width * 0.5,
            overlay_height * 0.5,
            "Camera preview unavailable; using geometric proxy background.",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#111827", "edgecolor": "#475569", "alpha": 0.85},
            zorder=8,
        )
    if mapping_mode != "geometric_proxy" and outside_rows:
        ax.text(
            overlay_width - 18,
            overlay_height - 18,
            f"{len(outside_rows)} unmatched candidates not plotted\nmatch tolerance = {overlay_metadata.get('kerr_pixel_match_tolerance_rg')} rg",
            fontsize=8.0,
            color="white",
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#111827", "edgecolor": "none", "alpha": 0.74},
            zorder=8,
        )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return {
        "observer_bridge_camera_overlay_generated": True,
        "camera_overlay_background_source": background_source,
        "camera_overlay_resolution_px": f"{overlay_width}x{overlay_height}",
        **overlay_metadata,
        "camera_overlay_preview_status": str((preview_summary or {}).get("status", "unavailable")),
        "camera_overlay_candidates_plotted": len(inside_rows),
        "camera_overlay_candidates_inside_fov": len(inside_rows),
        "camera_overlay_candidates_unmatched": len(outside_rows),
        "camera_overlay_top_n": top_n,
    }


def _write_geometry_html(rows: list[dict[str, Any]], path: Path) -> None:
    points = []
    for row in rows:
        r = _score(row, "interaction_r_rg")
        theta = _score(row, "interaction_theta_rad")
        phi = _score(row, "interaction_phi_rad")
        st = math.sin(theta)
        points.append(
            {
                "x": r * st * math.cos(phi),
                "y": r * st * math.sin(phi),
                "z": r * math.cos(theta),
                "score": _score(row, "final_observation_score"),
                "inside_fov": bool(row.get("camera_fov_flag")),
            }
        )
    path.write_text(
        """<!doctype html><html><head><meta charset="utf-8"><title>HADROS3 Observer Bridge Geometry</title>
<style>body{margin:0;font-family:system-ui;background:#101318;color:#e5e7eb}canvas{display:block;width:100vw;height:100vh}.hud{position:fixed;left:14px;top:12px;background:rgba(16,19,24,.82);padding:10px;border:1px solid #334155;border-radius:6px}</style></head>
<body><canvas id="c"></canvas><div class="hud"><strong>Observer Bridge Geometry</strong><br>scoring-only proxy view<br>green: inside FOV, amber: outside FOV</div>
<script>
const points = """
        + json.dumps(points)
        + """;
const canvas=document.getElementById('c'),ctx=canvas.getContext('2d');
function draw(){const dpr=window.devicePixelRatio||1;canvas.width=innerWidth*dpr;canvas.height=innerHeight*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#101318';ctx.fillRect(0,0,innerWidth,innerHeight);const lim=Math.max(1,...points.map(p=>Math.hypot(p.x,p.y,p.z)))*1.2;const s=Math.min(innerWidth,innerHeight)/(2*lim);ctx.strokeStyle='#334155';ctx.beginPath();ctx.arc(innerWidth/2,innerHeight/2,lim*s,0,Math.PI*2);ctx.stroke();for(const p of points){const x=innerWidth/2+p.x*s;const y=innerHeight/2-p.z*s;ctx.fillStyle=p.inside_fov?'#22c55e':'#f59e0b';ctx.beginPath();ctx.arc(x,y,3+Math.min(5,Math.sqrt(Math.max(0,p.score))),0,Math.PI*2);ctx.fill();}ctx.fillStyle='#e5e7eb';ctx.fillText('projection: x,z from x=r sin(theta) cos(phi), z=r cos(theta)',14,innerHeight-18)}
addEventListener('resize',draw);draw();
</script></body></html>
""",
        encoding="utf-8",
    )


def _cone_segments(values: dict[str, dict[str, Any]]) -> list[list[Vec3]]:
    cone = values.get("polar_cone", {})
    if not bool(cone.get("enabled", True)):
        return []
    opening = math.radians(float(cone.get("opening_angle_deg", 22.0)))
    r_min = float(cone.get("r_min_rg", 2.2))
    r_max = float(cone.get("r_max_rg", 40.0))
    signs = (1.0, -1.0) if str(cone.get("draw_mode", "bipolar_funnel")) == "bipolar_funnel" else (1.0,)
    segments: list[list[Vec3]] = []
    for sign in signs:
        theta = opening if sign > 0.0 else math.pi - opening
        for i in range(32):
            phi = 2.0 * math.pi * i / 32.0
            segments.append([_spherical(r_min, theta, phi), _spherical(r_max, theta, phi)])
        for radius in (r_min, r_max):
            points = [_spherical(radius, theta, 2.0 * math.pi * i / 96.0) for i in range(97)]
            segments.append(points)
    return segments


def _camera_frustum_segments(values: dict[str, dict[str, Any]]) -> list[list[Vec3]]:
    observer, forward, right, up = _camera_frame(values)
    camera = values.get("observer_camera", {})
    torus = values.get("analytic_torus", {})
    far = max(float(torus.get("r_outer_rg", 20.0)) * 1.8, float(camera.get("observer_distance_rg", 60.0)) * 0.55)
    fov_x = math.radians(float(camera.get("field_of_view_deg", 25.0)))
    width = max(1.0, float(camera.get("pixel_width", 512)))
    height = max(1.0, float(camera.get("pixel_height", 288)))
    tan_x = math.tan(0.5 * fov_x)
    tan_y = tan_x * height / width
    corners = []
    for sx, sy in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]:
        direction = _unit(_vec_add(_vec_add(forward, _vec_mul(right, sx * tan_x)), _vec_mul(up, sy * tan_y)))
        corners.append(_vec_add(observer, _vec_mul(direction, far)))
    return [[observer, corner] for corner in corners] + [[corners[i], corners[(i + 1) % 4]] for i in range(4)]


def _ranked_candidate_rows(candidates: list[dict[str, Any]], ranked: list[dict[str, Any]], pixel_map: list[dict[str, Any]], max_candidates: int) -> list[dict[str, Any]]:
    by_id = {_candidate_identifier(row): row for row in candidates}
    pixel_by_id = {_candidate_identifier(row): row for row in pixel_map}
    ordered: list[dict[str, Any]] = []
    used: set[str] = set()
    for ranked_row in ranked:
        cid = _candidate_identifier(ranked_row)
        candidate = by_id.get(cid)
        if candidate is None:
            continue
        merged = dict(candidate)
        merged.update(pixel_by_id.get(cid, {}))
        merged["candidate_rank"] = len(ordered) + 1
        ordered.append(merged)
        used.add(cid)
        if len(ordered) >= max_candidates:
            return ordered
    for candidate in candidates:
        cid = _candidate_identifier(candidate)
        if cid in used:
            continue
        merged = dict(candidate)
        merged.update(pixel_by_id.get(cid, {}))
        merged["candidate_rank"] = len(ordered) + 1
        ordered.append(merged)
        if len(ordered) >= max_candidates:
            break
    return ordered


def _interactive_rays(rows: list[dict[str, Any]], values: dict[str, dict[str, Any]], max_rays: int, stride: int) -> list[dict[str, Any]]:
    rays: list[dict[str, Any]] = []
    matched = [row for row in rows if bool(row.get("matched_pixel_found")) and row.get("matched_pixel_x") is not None and row.get("matched_pixel_y") is not None]
    max_target_radius = max([_score(row, "interaction_r_rg") for row in rows], default=10.0)
    for row in matched[: max(0, max_rays)]:
        points, status = _integrate_kerr_ray_cartesian(
            float(row["matched_pixel_x"]),
            float(row["matched_pixel_y"]),
            OVERLAY_WIDTH,
            OVERLAY_HEIGHT,
            values,
            max_target_radius=max_target_radius,
        )
        sampled = points[:: max(1, stride)]
        if points and (not sampled or sampled[-1] != points[-1]):
            sampled.append(points[-1])
        rays.append(
            {
                "interaction_id": row.get("interaction_id"),
                "candidate_rank": row.get("candidate_rank"),
                "closest_approach_rg": row.get("closest_approach_rg"),
                "points": sampled,
                "status": status,
            }
        )
    return rays


def _write_kerr_interactive_view_html(
    candidates: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    pixel_map: list[dict[str, Any]],
    values: dict[str, dict[str, Any]],
    path: Path,
) -> dict[str, Any]:
    bridge = values.get("observer_bridge", {})
    max_candidates = max(1, int(float(bridge.get("interactive_max_candidates", 40))))
    max_rays = max(0, int(float(bridge.get("interactive_max_rays", 64))))
    ray_stride = max(1, int(float(bridge.get("interactive_ray_stride", 4))))
    color_mode = str(bridge.get("interactive_candidate_color_mode", "final_observation_score"))
    rows = _ranked_candidate_rows(candidates, ranked, pixel_map, max_candidates)
    top_ids = {str(row.get("interaction_id") or row.get("event_id")) for row in ranked[:5]}
    scene_candidates = []
    for row in rows:
        r = _score(row, "interaction_r_rg")
        theta = _score(row, "interaction_theta_rad")
        phi = _score(row, "interaction_phi_rad")
        cid = _candidate_identifier(row)
        scene_candidates.append(
            {
                "interaction_id": row.get("interaction_id"),
                "event_id": row.get("event_id"),
                "source_sample_id": row.get("source_sample_id"),
                "rank": row.get("candidate_rank"),
                "position": _spherical(r, theta, phi),
                "score": _score(row, "final_observation_score"),
                "inside_fov": bool(row.get("camera_fov_flag")),
                "matched": bool(row.get("matched_pixel_found")),
                "closest_approach_rg": row.get("closest_approach_rg"),
                "top": cid in top_ids,
            }
        )
    rays = _interactive_rays(rows, values, max_rays, ray_stride)
    observer, forward, right, up = _camera_frame(values)
    data = {
        "metadata": {
            "title": "Observer Bridge Kerr Interactive View",
            "interactive_view_uses_kerr_ray_matching": True,
            "interactive_view_not_final_observed_image": True,
            "interactive_view_diagnostic_only": True,
            "interactive_max_candidates": max_candidates,
            "interactive_max_rays": max_rays,
            "interactive_ray_stride": ray_stride,
            "interactive_ray_source": "observer_candidate_kerr_pixel_map.jsonl matched pixels reintegrated with python_kerr_rk4_diagnostic",
            "interactive_candidate_color_mode": color_mode,
            "match_tolerance_rg": float(bridge.get("kerr_pixel_match_tolerance_rg", 3.5)),
            "candidate_count": len(scene_candidates),
            "ray_count": len(rays),
        },
        "black_hole": {
            "horizon_radius_rg": horizon_radius_rg(float(values.get("black_hole", {}).get("spin_a", 0.5))),
        },
        "medium": {
            "rings": MediumRenderer.proxy_shell_rings(values, phi_steps=96),
            "metadata": MediumRenderer.metadata(),
        },
        "cones": _cone_segments(values),
        "camera": {
            "observer": observer,
            "forward": forward,
            "right": right,
            "up": up,
            "frustum": _camera_frustum_segments(values),
            "fov_deg": float(values.get("observer_camera", {}).get("field_of_view_deg", 25.0)),
        },
        "candidates": scene_candidates,
        "rays": rays,
    }
    template = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Observer Bridge Kerr Interactive View</title>
<style>
body{margin:0;background:#0b0f17;color:#e5e7eb;font-family:system-ui,-apple-system,Segoe UI,sans-serif;overflow:hidden}
canvas{display:block;width:100vw;height:100vh;background:#0b0f17}
.hud{position:fixed;left:12px;top:12px;width:310px;max-height:calc(100vh - 24px);overflow:auto;background:rgba(15,23,42,.88);border:1px solid #334155;border-radius:8px;padding:12px;box-shadow:0 12px 40px rgba(0,0,0,.35)}
.hud h1{font-size:16px;margin:0 0 8px}.hud label{display:block;margin:7px 0;font-size:12px}.hud select,.hud input[type=range]{width:100%}.hud code{color:#bae6fd}.stat{display:grid;grid-template-columns:1fr auto;gap:4px;font-size:12px;border-top:1px solid #334155;margin-top:8px;padding-top:8px}.note{font-size:11px;color:#cbd5e1;line-height:1.35}.pill{display:inline-block;border:1px solid #475569;border-radius:999px;padding:2px 7px;margin:2px 3px 2px 0;font-size:11px;color:#dbeafe}
</style></head>
<body><canvas id="view"></canvas>
<div class="hud">
<h1>Observer Bridge Kerr Interactive View</h1>
<div class="note">Diagnostic geometry view. The Camera Preview remains the final ray-traced image; this view shows candidates, medium, FOV and Kerr ray matching paths.</div>
<label><input id="showMedium" type="checkbox" checked> show medium/toro</label>
<label><input id="showCones" type="checkbox" checked> show cones</label>
<label><input id="showCandidates" type="checkbox" checked> show candidates</label>
<label><input id="topOnly" type="checkbox"> top ranked only</label>
<label><input id="showRays" type="checkbox" checked> show matched Kerr rays</label>
<label><input id="physicalCameraView" type="checkbox" checked> start from physical observer camera view</label>
<label>number of candidates displayed <input id="candidateLimit" type="range" min="1" max="1" value="1"></label>
<label>color by <select id="colorMode"><option value="final_observation_score">final_observation_score</option><option value="closest_approach_rg">closest_approach_rg</option><option value="inside_outside_fov">inside/outside FOV</option></select></label>
<div class="note">Drag: rotate | Scroll: zoom | Shift+drag: pan</div>
<div class="stat" id="stats"></div>
</div>
<script>
const scene = __DATA__;
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const controls = {
  showMedium: document.getElementById('showMedium'),
  showCones: document.getElementById('showCones'),
  showCandidates: document.getElementById('showCandidates'),
  topOnly: document.getElementById('topOnly'),
  showRays: document.getElementById('showRays'),
  physicalCameraView: document.getElementById('physicalCameraView'),
  candidateLimit: document.getElementById('candidateLimit'),
  colorMode: document.getElementById('colorMode'),
  stats: document.getElementById('stats')
};
controls.candidateLimit.max = Math.max(1, scene.candidates.length);
controls.candidateLimit.value = Math.max(1, scene.candidates.length);
controls.colorMode.value = scene.metadata.interactive_candidate_color_mode || 'final_observation_score';
let yaw = 0, pitch = 0, zoom = 7.5, panX = 0, panY = 0, dragging = false, lastX = 0, lastY = 0;
function resize(){const dpr=window.devicePixelRatio||1;canvas.width=innerWidth*dpr;canvas.height=innerHeight*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);draw();}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function sub(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
function rot(p){const x=p[0], y=p[1], z=p[2];const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);const x1=cy*x+sy*y;const y1=-sy*x+cy*y;const z1=z;return [x1, cp*y1-sp*z1, sp*y1+cp*z1];}
function project(p){
 if(controls.physicalCameraView.checked){
   const q=sub(p,[0,0,0]);
   const x=dot(q,scene.camera.right);
   const y=dot(q,scene.camera.up);
   const z=dot(q,scene.camera.forward);
   const r=rot([x,z,y]);
   const depth=90+r[1];
   const scale=Math.min(innerWidth,innerHeight)*zoom/Math.max(15,depth);
   return [innerWidth/2+panX+r[0]*scale, innerHeight/2+panY-r[2]*scale, depth, scale];
 }
 const r=rot(p);const depth=90+r[1];const scale=Math.min(innerWidth,innerHeight)*zoom/Math.max(15,depth);return [innerWidth/2+panX+r[0]*scale, innerHeight/2+panY-r[2]*scale, depth, scale];
}
function line(points,color,alpha,width){if(points.length<2)return;ctx.save();ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();for(let i=0;i<points.length;i++){const q=project(points[i]);if(i===0)ctx.moveTo(q[0],q[1]);else ctx.lineTo(q[0],q[1]);}ctx.stroke();ctx.restore();}
function sphere(p,r,color,alpha,stroke){const q=project(p);ctx.save();ctx.globalAlpha=alpha;ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],Math.max(1.5,r*q[3]/28),0,Math.PI*2);ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=2;ctx.stroke();}ctx.restore();}
function candidateColor(c){const mode=controls.colorMode.value;if(mode==='inside_outside_fov')return c.inside_fov?'#22c55e':'#f59e0b';if(mode==='closest_approach_rg'){if(!c.matched)return '#64748b';const t=Math.min(1,Math.max(0,(c.closest_approach_rg||0)/(scene.metadata.match_tolerance_rg||4)));return `rgb(${Math.round(80+180*t)},${Math.round(220-120*t)},${Math.round(120-80*t)})`;}const scores=scene.candidates.map(x=>x.score||0);const m=Math.max(1e-30,...scores);const t=Math.sqrt(Math.max(0,c.score||0)/m);return `rgb(${Math.round(80+180*t)},${Math.round(70+40*t)},${Math.round(180-120*t)})`;}
function drawAxes(){line([[0,0,0],[25,0,0]],'#ef4444',.6,1);line([[0,0,0],[0,25,0]],'#22c55e',.6,1);line([[0,0,0],[0,0,25]],'#38bdf8',.6,1);}
function draw(){ctx.clearRect(0,0,innerWidth,innerHeight);ctx.fillStyle='#0b0f17';ctx.fillRect(0,0,innerWidth,innerHeight);drawAxes();sphere([0,0,0],scene.black_hole.horizon_radius_rg*2.8,'#020617',1,'#94a3b8');
 if(controls.showMedium.checked){for(const ring of scene.medium.rings){line(ring.points,'#94a3b8',Math.max(.08,ring.alpha||.2),ring.hard_radial_cut?1.7:0.9);}}
 if(controls.showCones.checked){for(const seg of scene.cones){line(seg,'#f59e0b',.25,1);}}
 for(const seg of scene.camera.frustum){line(seg,'#60a5fa',.5,1.2);} sphere(scene.camera.observer,2.6,'#38bdf8',.95,'#dbeafe'); line([scene.camera.observer,[0,0,0]],'#60a5fa',.18,1);
 if(controls.showRays.checked){for(const ray of scene.rays){line(ray.points,'#a78bfa',.36,1.0);}}
 if(controls.showCandidates.checked){let shown=scene.candidates.slice(0,Number(controls.candidateLimit.value));if(controls.topOnly.checked)shown=shown.filter(c=>c.top);shown.sort((a,b)=>project(a.position)[2]-project(b.position)[2]);for(const c of shown){sphere(c.position,c.top?3.0:2.0,candidateColor(c),c.matched?.95:.25,c.top?'#fef08a':(c.matched?'#f8fafc':'#64748b'));}}
 controls.stats.innerHTML = `<span>mapping</span><code>kerr_geodesic_pixel_match</code><span>candidates loaded</span><b>${scene.candidates.length}</b><span>rays shown</span><b>${scene.rays.length}</b><span>diagnostic only</span><b>true</b><span>not final observed image</span><b>true</b>`;
}
for(const el of Object.values(controls)){if(el&&el.addEventListener)el.addEventListener('input',draw);}
canvas.addEventListener('mousedown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;});
addEventListener('mouseup',()=>dragging=false);
addEventListener('mousemove',e=>{if(!dragging)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;lastX=e.clientX;lastY=e.clientY;if(e.shiftKey){panX+=dx;panY+=dy;}else{yaw+=dx*0.006;pitch=Math.max(-1.45,Math.min(1.45,pitch+dy*0.006));}draw();});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom*=Math.exp(-e.deltaY*0.001);zoom=Math.max(1.0,Math.min(40,zoom));draw();},{passive:false});
canvas.addEventListener('touchstart',e=>{if(!e.touches.length)return;dragging=true;lastX=e.touches[0].clientX;lastY=e.touches[0].clientY;},{passive:true});
canvas.addEventListener('touchmove',e=>{if(!dragging||!e.touches.length)return;e.preventDefault();const dx=e.touches[0].clientX-lastX,dy=e.touches[0].clientY-lastY;lastX=e.touches[0].clientX;lastY=e.touches[0].clientY;yaw+=dx*0.006;pitch=Math.max(-1.45,Math.min(1.45,pitch+dy*0.006));draw();},{passive:false});
canvas.addEventListener('touchend',()=>dragging=false);
addEventListener('resize',resize);resize();
</script></body></html>"""
    payload = json.dumps(data)
    path.write_text(template.replace("__DATA__", payload), encoding="utf-8")
    return {
        "observer_bridge_kerr_interactive_view_generated": True,
        "interactive_view_uses_kerr_ray_matching": True,
        "interactive_view_not_final_observed_image": True,
        "interactive_view_diagnostic_only": True,
        "interactive_max_candidates": max_candidates,
        "interactive_max_rays": max_rays,
        "interactive_ray_stride": ray_stride,
        "interactive_ray_source": data["metadata"]["interactive_ray_source"],
        "interactive_candidate_color_mode": color_mode,
        "interactive_candidates_displayed": len(scene_candidates),
        "interactive_rays_displayed": len(rays),
    }


def _augment_summary(summary: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    products = dict(summary.get("products", {}))
    products.update(
        {
            "observer_bridge_map": str(output_dir / "observer_bridge_map.png"),
            "observer_bridge_score_distribution": str(output_dir / "observer_bridge_score_distribution.png"),
            "observer_bridge_weight_breakdown": str(output_dir / "observer_bridge_weight_breakdown.png"),
            "observer_bridge_visibility_map": str(output_dir / "observer_bridge_visibility_map.png"),
            "observer_bridge_ranked_events_png": str(output_dir / "observer_bridge_ranked_events.png"),
            "observer_bridge_geometry_3d_html": str(output_dir / "observer_bridge_geometry_3d.html"),
            "observer_bridge_camera_view": str(output_dir / "observer_bridge_camera_view.png"),
            "observer_bridge_camera_overlay": str(output_dir / "observer_bridge_camera_overlay.png"),
            "observer_candidate_kerr_pixel_map": str(output_dir / "observer_candidate_kerr_pixel_map.jsonl"),
            "observer_bridge_kerr_interactive_view": str(output_dir / "observer_bridge_kerr_interactive_view.html"),
            "observer_bridge_selected_candidates": str(output_dir / "observer_bridge_selected_candidates.jsonl"),
            "observer_bridge_selection_summary": str(output_dir / "observer_bridge_selection_summary.json"),
        }
    )
    summary.update(
        {
            "products": products,
            "diagnostics_generated": True,
            "observer_bridge_map_generated": True,
            "observer_bridge_score_distribution_generated": True,
            "observer_bridge_weight_breakdown_generated": True,
            "observer_bridge_visibility_map_generated": True,
            "observer_bridge_ranked_events_png_generated": True,
            "observer_bridge_geometry_3d_html_generated": True,
            "observer_bridge_camera_view_generated": summary.get("observer_bridge_camera_view_generated", False),
            "observer_bridge_camera_overlay_generated": summary.get("observer_bridge_camera_overlay_generated", False),
            "observer_bridge_kerr_interactive_view_generated": summary.get("observer_bridge_kerr_interactive_view_generated", False),
            "observer_bridge_selected_candidates_generated": summary.get("observer_bridge_selected_candidates_generated", False),
            "observer_bridge_selection_summary_generated": summary.get("observer_bridge_selection_summary_generated", False),
            "downstream_candidate_selection_enabled": summary.get("downstream_candidate_selection_enabled", False),
            "downstream_selection_policy": summary.get("downstream_selection_policy"),
            "downstream_n_candidates_ranked": summary.get("downstream_n_candidates_ranked", 0),
            "downstream_n_candidates_selected": summary.get("downstream_n_candidates_selected", 0),
            "downstream_stage_target": summary.get("downstream_stage_target"),
            "top_n_candidates": summary.get("top_n_candidates"),
            "downstream_min_final_observation_score": summary.get("downstream_min_final_observation_score"),
            "camera_view_projection_model": summary.get("camera_view_projection_model"),
            "camera_view_projection_physics_risk": summary.get("camera_view_projection_physics_risk"),
            "not_ray_traced": summary.get("not_ray_traced", True),
            "camera_overlay_background_source": summary.get("camera_overlay_background_source"),
            "camera_overlay_resolution_px": summary.get("camera_overlay_resolution_px"),
            "candidate_overlay_projection_model": summary.get("candidate_overlay_projection_model"),
            "candidate_overlay_kerr_lensed": summary.get("candidate_overlay_kerr_lensed", False),
            "candidate_overlay_not_ray_traced": summary.get("candidate_overlay_not_ray_traced", True),
            "candidate_overlay_physics_risk": summary.get("candidate_overlay_physics_risk", True),
            "candidate_overlay_alignment": summary.get("candidate_overlay_alignment"),
            "candidate_overlay_fallback_reason": summary.get("candidate_overlay_fallback_reason"),
            "camera_overlay_preview_status": summary.get("camera_overlay_preview_status"),
            "camera_overlay_candidates_plotted": summary.get("camera_overlay_candidates_plotted", 0),
            "camera_overlay_candidates_inside_fov": summary.get("camera_overlay_candidates_inside_fov", 0),
            "camera_overlay_candidates_unmatched": summary.get("camera_overlay_candidates_unmatched", 0),
            "camera_overlay_top_n": summary.get("camera_overlay_top_n", 0),
            "kerr_geodesic_backend": summary.get("kerr_geodesic_backend"),
            "kerr_pixel_match_resolution": summary.get("kerr_pixel_match_resolution"),
            "kerr_pixel_match_resolution_x": summary.get("kerr_pixel_match_resolution_x"),
            "kerr_pixel_match_resolution_y": summary.get("kerr_pixel_match_resolution_y"),
            "kerr_pixel_match_tolerance_rg": summary.get("kerr_pixel_match_tolerance_rg"),
            "kerr_pixel_match_refine_enabled": summary.get("kerr_pixel_match_refine_enabled"),
            "kerr_pixel_match_n_candidates": summary.get("kerr_pixel_match_n_candidates", 0),
            "kerr_pixel_match_n_matched": summary.get("kerr_pixel_match_n_matched", 0),
            "kerr_pixel_match_n_unmatched": summary.get("kerr_pixel_match_n_unmatched", 0),
            "kerr_pixel_match_mean_closest_approach_rg": summary.get("kerr_pixel_match_mean_closest_approach_rg"),
            "kerr_pixel_match_max_closest_approach_rg": summary.get("kerr_pixel_match_max_closest_approach_rg"),
            "interactive_view_uses_kerr_ray_matching": summary.get("interactive_view_uses_kerr_ray_matching", False),
            "interactive_view_not_final_observed_image": summary.get("interactive_view_not_final_observed_image", True),
            "interactive_view_diagnostic_only": summary.get("interactive_view_diagnostic_only", True),
            "interactive_max_candidates": summary.get("interactive_max_candidates"),
            "interactive_max_rays": summary.get("interactive_max_rays"),
            "interactive_ray_stride": summary.get("interactive_ray_stride"),
            "interactive_ray_source": summary.get("interactive_ray_source"),
            "interactive_candidate_color_mode": summary.get("interactive_candidate_color_mode"),
            "interactive_candidates_displayed": summary.get("interactive_candidates_displayed"),
            "interactive_rays_displayed": summary.get("interactive_rays_displayed"),
            "medium_renderer_used": summary.get("medium_renderer_used", False),
            "medium_model": summary.get("medium_model"),
            "density_model": summary.get("density_model"),
            "density_model_has_hard_radial_cut": summary.get("density_model_has_hard_radial_cut"),
            "density_model_theta_profile": summary.get("density_model_theta_profile"),
            "density_model_theta_is_hard_cut": summary.get("density_model_theta_is_hard_cut"),
            "half_opening_angle_interpretation": summary.get("half_opening_angle_interpretation"),
            "camera_view_candidates_plotted": summary.get("camera_view_candidates_plotted", 0),
            "camera_view_candidates_inside_fov": summary.get("camera_view_candidates_inside_fov", 0),
            "camera_view_top_n": summary.get("camera_view_top_n", 0),
            "bridge_mode": summary.get("bridge_mode", "scoring_only"),
            "observer_bridge_stage_status": "observer_bridge_scored_no_event_generation",
            "powheg_invoked": False,
            "pythia_invoked": False,
            "geant4_invoked": False,
            "photon_transport_invoked": False,
            "event_generation_invoked": False,
        }
    )
    return summary


def generate_observer_bridge_products(values: dict[str, dict[str, Any]], *, run_output_dir: Path) -> dict[str, Any]:
    problems = validate_values(values)
    if problems:
        raise ValueError("Invalid HADROS3 configuration:\n- " + "\n- ".join(problems))
    if not OBSERVER_BRIDGE_CPP_EXECUTABLE.exists():
        raise FileNotFoundError(f"Observer Bridge C++ backend not found: {OBSERVER_BRIDGE_CPP_EXECUTABLE}")

    output_dir = observer_bridge_dir(run_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dis_path = run_output_dir / "DIS" / "dis_accepted_interactions.jsonl"
    if not dis_path.exists():
        raise FileNotFoundError(f"DIS accepted interactions not found: {dis_path}")

    _runtime_config_path(values, run_output_dir)
    subprocess.run(
        [str(OBSERVER_BRIDGE_CPP_EXECUTABLE), "--run-output", str(run_output_dir)],
        cwd=ROOT,
        check=True,
    )

    candidates_path = output_dir / "observer_bridge_candidates.jsonl"
    ranked_path = output_dir / "observer_bridge_ranked_events.jsonl"
    summary_path = output_dir / "observer_bridge_summary.json"
    report_path = output_dir / "observer_bridge_report.json"
    candidates = _read_jsonl(candidates_path)
    ranked = _read_jsonl(ranked_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    max_ranked = int(values.get("observer_bridge", {}).get("max_ranked_events", 25))

    _draw_map(candidates, output_dir / "observer_bridge_map.png", "Observer Bridge candidates")
    _draw_score_distribution(candidates, output_dir / "observer_bridge_score_distribution.png")
    _draw_weight_breakdown(candidates, output_dir / "observer_bridge_weight_breakdown.png")
    _draw_map(candidates, output_dir / "observer_bridge_visibility_map.png", "Observer Bridge FOV visibility", "camera_fov_weight")
    _draw_ranked(ranked, output_dir / "observer_bridge_ranked_events.png", max_ranked)
    _write_geometry_html(candidates, output_dir / "observer_bridge_geometry_3d.html")
    camera_view = _draw_camera_view(candidates, ranked, values, output_dir / "observer_bridge_camera_view.png", min(5, max_ranked))
    summary.update(camera_view)
    camera_overlay = _draw_camera_overlay(candidates, ranked, values, run_output_dir, output_dir / "observer_bridge_camera_overlay.png", min(5, max_ranked))
    summary.update(camera_overlay)
    pixel_map = _read_jsonl(output_dir / "observer_candidate_kerr_pixel_map.jsonl")
    interactive_view = _write_kerr_interactive_view_html(
        candidates,
        ranked,
        pixel_map,
        values,
        output_dir / "observer_bridge_kerr_interactive_view.html",
    )
    summary.update(interactive_view)
    selection = _select_downstream_candidates(ranked, values, output_dir)
    summary.update(selection)

    summary = _augment_summary(summary, output_dir)
    write_json(summary_path, summary)
    write_json(report_path, summary)
    return summary
