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
from .medium_renderer import MediumRenderer
from .paths import camera_preview_dir, observer_bridge_dir, run_metadata_dir
from .provenance import write_json


ROOT = Path(__file__).resolve().parents[1]
OBSERVER_BRIDGE_CPP_EXECUTABLE = ROOT / "bin" / "hadros3_observer_bridge"


Vec3 = tuple[float, float, float]


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


def _camera_frame(values: dict[str, dict[str, Any]]) -> tuple[Vec3, Vec3, Vec3, Vec3]:
    camera = values.get("observer_camera", {})
    r_obs = float(camera.get("observer_distance_rg", 60.0))
    inc = math.radians(float(camera.get("inclination_deg", 80.0)))
    azi = math.radians(float(camera.get("azimuth_deg", 0.0)))
    observer = _spherical(r_obs, inc, azi)
    forward = _unit(_vec_mul(observer, -1.0))
    world_up = (0.0, 0.0, 1.0)
    right = _unit(_cross(forward, world_up))
    if _norm(right) <= 0.0:
        right = (1.0, 0.0, 0.0)
    up = _unit(_cross(right, forward))
    return observer, forward, right, up


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


def _draw_camera_overlay(
    candidates: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    values: dict[str, dict[str, Any]],
    run_output_dir: Path,
    path: Path,
    top_n: int,
) -> dict[str, int | bool | str]:
    projections = _project_candidates_for_camera(candidates, ranked, values, top_n)
    inside_rows = [row for row in projections if row["inside"]]
    outside_rows = [row for row in projections if not row["inside"]]
    max_score = max([row["score"] for row in projections], default=0.0)
    camera_path = camera_preview_dir(run_output_dir) / "hadros3_camera_preview.png"
    overlay_width = 1024
    overlay_height = 576
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
        return 0.5 * overlay_width * (1.0 + float(row["x"]))

    def pixel_y(row: dict[str, Any]) -> float:
        return 0.5 * overlay_height * (1.0 - float(row["y"]))

    if outside_rows:
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
        "background = Camera Preview\npoints = Observer Bridge candidates\nprojection = geometric proxy, not ray-traced secondary particles",
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
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return {
        "observer_bridge_camera_overlay_generated": True,
        "camera_overlay_background_source": background_source,
        "camera_overlay_resolution_px": f"{overlay_width}x{overlay_height}",
        "candidate_overlay_projection_model": "geometric_pinhole_proxy",
        "candidate_overlay_not_ray_traced": True,
        "candidate_overlay_physics_risk": True,
        "candidate_overlay_alignment": "camera_preview_pixel_plane",
        "camera_overlay_preview_status": str((preview_summary or {}).get("status", "unavailable")),
        "camera_overlay_candidates_plotted": len(projections),
        "camera_overlay_candidates_inside_fov": len(inside_rows),
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
            "camera_view_projection_model": summary.get("camera_view_projection_model"),
            "camera_view_projection_physics_risk": summary.get("camera_view_projection_physics_risk"),
            "not_ray_traced": summary.get("not_ray_traced", True),
            "camera_overlay_background_source": summary.get("camera_overlay_background_source"),
            "camera_overlay_resolution_px": summary.get("camera_overlay_resolution_px"),
            "candidate_overlay_projection_model": summary.get("candidate_overlay_projection_model"),
            "candidate_overlay_not_ray_traced": summary.get("candidate_overlay_not_ray_traced", True),
            "candidate_overlay_physics_risk": summary.get("candidate_overlay_physics_risk", True),
            "candidate_overlay_alignment": summary.get("candidate_overlay_alignment"),
            "camera_overlay_preview_status": summary.get("camera_overlay_preview_status"),
            "camera_overlay_candidates_plotted": summary.get("camera_overlay_candidates_plotted", 0),
            "camera_overlay_candidates_inside_fov": summary.get("camera_overlay_candidates_inside_fov", 0),
            "camera_overlay_top_n": summary.get("camera_overlay_top_n", 0),
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

    summary = _augment_summary(summary, output_dir)
    write_json(summary_path, summary)
    write_json(report_path, summary)
    return summary
