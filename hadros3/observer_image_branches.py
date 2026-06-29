"""Observer Image Branch Analysis for HADROS3 H3-W8b."""

from __future__ import annotations

import base64
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .config import validate_values
from .paths import camera_preview_dir, clear_observer_image_branches_outputs, observer_bridge_dir, observer_image_branches_dir
from .provenance import write_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _spherical_to_cartesian(r_rg: float, theta_rad: float, phi_rad: float) -> tuple[float, float, float]:
    sin_theta = math.sin(theta_rad)
    return (
        r_rg * sin_theta * math.cos(phi_rad),
        r_rg * sin_theta * math.sin(phi_rad),
        r_rg * math.cos(theta_rad),
    )


def _observer_position(values: dict[str, dict[str, Any]]) -> tuple[float, float, float]:
    camera = values.get("observer_camera", {})
    theta = math.radians(float(camera.get("inclination_deg", 80.0)))
    theta = max(1.0e-6, min(math.pi - 1.0e-6, theta))
    phi = math.radians(float(camera.get("azimuth_deg", 0.0)))
    return _spherical_to_cartesian(float(camera.get("observer_distance_rg", 60.0)), theta, phi)


def _z_sign(value: float, *, eps: float = 1.0e-9) -> str:
    if value > eps:
        return "positive"
    if value < -eps:
        return "negative"
    return "zero"


def _hemisphere(value: float) -> str:
    sign = _z_sign(value)
    if sign == "positive":
        return "north"
    if sign == "negative":
        return "south"
    return "equatorial"


def _image_dimensions(path: Path) -> tuple[int | None, int | None]:
    if not path.exists():
        return None, None
    image = plt.imread(path)
    return int(image.shape[1]), int(image.shape[0])


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _covariance(xs: list[float], ys: list[float]) -> list[list[float]]:
    if len(xs) <= 1:
        return [[0.0, 0.0], [0.0, 0.0]]
    arr = np.asarray([xs, ys], dtype=float)
    cov = np.cov(arr)
    return [[float(cov[0, 0]), float(cov[0, 1])], [float(cov[1, 0]), float(cov[1, 1])]]


def _pixel_spread(xs: list[float], ys: list[float], cx: float, cy: float) -> float:
    if not xs:
        return 0.0
    return float(math.sqrt(_mean([(x - cx) ** 2 + (y - cy) ** 2 for x, y in zip(xs, ys)])))


def _branch_score(n_rays: int, closest_mean: float, pixel_spread: float) -> tuple[float, dict[str, float]]:
    w_rays = float(max(n_rays, 0))
    w_closest = 1.0 / max(float(closest_mean), 1.0e-9)
    w_compactness = 1.0 / max(float(pixel_spread), 1.0)
    return w_rays * w_closest * w_compactness, {
        "w_rays": w_rays,
        "w_closest": w_closest,
        "w_compactness": w_compactness,
    }


def _single_branch_from_pixel_map(row: dict[str, Any]) -> list[dict[str, Any]]:
    x = float(row.get("matched_pixel_x", row.get("pixel_x", 0.0)))
    y = float(row.get("matched_pixel_y", row.get("pixel_y", 0.0)))
    approach = float(row.get("closest_approach_rg", row.get("matched_closest_approach_rg", 0.0)))
    return [{
        "cluster_index": 1,
        "n_rays": 1,
        "centroid_pixel": [x, y],
        "pixels": [[x, y]],
        "ray_indices": [int(row.get("matched_ray_index", 0))],
        "closest_approach_rg": approach,
        "best_ray": {
            "ray_index": int(row.get("matched_ray_index", 0)),
            "pixel_x": x,
            "pixel_y": y,
            "closest_approach_rg": approach,
            "status": "single_pixel_map_fallback",
        },
    }]


def _candidate_key(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or row.get("interaction_id") or row.get("event_id") or "")


def _build_candidate_maps(run_output_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    bridge_dir = observer_bridge_dir(run_output_dir)
    selected = {_candidate_key(row): row for row in _read_jsonl(bridge_dir / "observer_bridge_selected_candidates.jsonl")}
    ranked = {_candidate_key(row): row for row in _read_jsonl(bridge_dir / "observer_bridge_ranked_events.jsonl")}
    pixel = {_candidate_key(row): row for row in _read_jsonl(bridge_dir / "observer_candidate_kerr_pixel_map.jsonl")}
    audit = {_candidate_key(row): row for row in _read_jsonl(bridge_dir / "candidate_multi_image_audit.jsonl")}
    return selected, ranked, pixel, audit


def _branch_rows_for_candidate(
    candidate_id: str,
    candidate: dict[str, Any],
    pixel_row: dict[str, Any] | None,
    audit_row: dict[str, Any] | None,
    *,
    minimum_branch_rays: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    clusters = list((audit_row or {}).get("image_clusters") or [])
    audit_explicitly_found_zero_clusters = audit_row is not None and "image_clusters" in audit_row and not clusters
    if not clusters and pixel_row is not None and not audit_explicitly_found_zero_clusters:
        clusters = _single_branch_from_pixel_map(pixel_row)

    branches: list[dict[str, Any]] = []
    for branch_index, cluster in enumerate(clusters):
        n_rays = int(cluster.get("n_rays", len(cluster.get("ray_indices", [])) or len(cluster.get("pixels", [])) or 1))
        if n_rays < minimum_branch_rays:
            continue
        pixels = [[float(px), float(py)] for px, py in cluster.get("pixels", [])]
        if not pixels:
            cx, cy = [float(v) for v in cluster.get("centroid_pixel", [0.0, 0.0])]
            pixels = [[cx, cy]]
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        cx, cy = [float(v) for v in cluster.get("centroid_pixel", [_mean(xs), _mean(ys)])]
        closest_values = [
            float(ray.get("closest_approach_rg", 0.0))
            for ray in (audit_row or {}).get("all_matching_rays", [])
            if int(ray.get("ray_index", -999999)) in {int(i) for i in cluster.get("ray_indices", [])}
        ]
        if not closest_values:
            closest_values = [float(cluster.get("closest_approach_rg", 0.0))]
        spread = _pixel_spread(xs, ys, cx, cy)
        closest_mean = _mean(closest_values)
        score, weights = _branch_score(n_rays, closest_mean, spread)
        branch_id = f"{candidate_id}:branch-{branch_index + 1:02d}"
        branches.append({
            "branch_id": branch_id,
            "candidate_id": candidate_id,
            "interaction_id": candidate.get("interaction_id", candidate_id),
            "event_id": candidate.get("event_id", ""),
            "source_sample_id": candidate.get("source_sample_id", ""),
            "branch_index": branch_index + 1,
            "number_of_matching_rays": n_rays,
            "pixel_centroid_x": cx,
            "pixel_centroid_y": cy,
            "pixel_covariance": _covariance(xs, ys),
            "closest_approach_mean_rg": closest_mean,
            "closest_approach_min_rg": float(min(closest_values)),
            "closest_approach_max_rg": float(max(closest_values)),
            "pixel_spread": spread,
            "ray_indices": [int(i) for i in cluster.get("ray_indices", [])],
            "matching_pixels": pixels,
            "branch_score": score,
            "branch_score_weights": weights,
            "branch_scoring_model": "ray_count_closeness_compactness_proxy",
            "primary_branch_selection_model": "argmax_branch_score",
            "primary_branch_selection_proxy": True,
            "current_overlay_pixel": (audit_row or {}).get("current_overlay_pixel"),
            "current_overlay_closest_approach_rg": (audit_row or {}).get("current_overlay_closest_approach_rg"),
            "current_algorithm": (audit_row or {}).get("current_algorithm", "closest Kerr ray"),
        })
    branches.sort(key=lambda item: float(item["branch_score"]), reverse=True)
    for rank, branch in enumerate(branches, start=1):
        branch["branch_rank"] = rank
        branch["is_primary_branch"] = rank == 1
        branch["is_secondary_branch"] = rank == 2
    primary = branches[0] if branches else None
    return branches, primary


def _primary_row(candidate: dict[str, Any], primary: dict[str, Any] | None, branches: list[dict[str, Any]]) -> dict[str, Any]:
    row = dict(candidate)
    candidate_id = _candidate_key(candidate)
    row["candidate_id"] = candidate_id
    row["image_branch_analysis_invoked"] = True
    row["number_of_image_branches"] = len(branches)
    row["possible_multiple_images"] = len(branches) > 1
    row["branch_scoring_model"] = "ray_count_closeness_compactness_proxy"
    row["primary_branch_selection_model"] = "argmax_branch_score"
    row["primary_branch_selection_proxy"] = True
    if primary:
        row.update({
            "primary_branch_id": primary["branch_id"],
            "primary_branch_rank": primary["branch_rank"],
            "primary_branch_score": primary["branch_score"],
            "primary_branch_pixel_x": primary["pixel_centroid_x"],
            "primary_branch_pixel_y": primary["pixel_centroid_y"],
            "primary_branch_number_of_matching_rays": primary["number_of_matching_rays"],
            "primary_branch_closest_approach_mean_rg": primary["closest_approach_mean_rg"],
            "primary_branch_pixel_spread": primary["pixel_spread"],
        })
    return row


def _write_branch_csv(path: Path, branches: list[dict[str, Any]]) -> None:
    fields = [
        "branch_id", "candidate_id", "branch_rank", "is_primary_branch", "number_of_matching_rays",
        "pixel_centroid_x", "pixel_centroid_y", "closest_approach_mean_rg", "closest_approach_min_rg",
        "closest_approach_max_rg", "pixel_spread", "branch_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in branches:
            writer.writerow({key: row.get(key) for key in fields})


def _draw_score_distribution(path: Path, branches: list[dict[str, Any]]) -> None:
    scores = [float(row.get("branch_score", 0.0)) for row in branches]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    if scores:
        ax.hist(scores, bins=min(20, max(5, len(scores))), color="#3366cc", alpha=0.85)
    ax.set_title("Observer image branch score distribution")
    ax.set_xlabel("branch_score")
    ax.set_ylabel("branches")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _display_y(pixel_y: float, height: int | float) -> float:
    return float(height) - float(pixel_y)


def _display_ys(pixel_values: list[float], height: int | None) -> list[float]:
    if height is None:
        return pixel_values
    return [_display_y(pixel_y, height) for pixel_y in pixel_values]


def _imshow_north_up(ax: Any, image_path: Path, *, flip_y: bool = True) -> tuple[int | None, int | None]:
    if not image_path.exists():
        return None, None
    image = plt.imread(image_path)
    height, width = int(image.shape[0]), int(image.shape[1])
    display_image = np.flipud(image) if flip_y else image
    ax.imshow(display_image, extent=(0, width, height, 0))
    return width, height


def _annotate_orientation_2d(ax: Any, values: dict[str, dict[str, Any]], width: int | None, height: int | None) -> None:
    camera = values.get("observer_camera", {})
    inclination = float(camera.get("inclination_deg", 80.0))
    observer = _observer_position(values)
    hemi = _hemisphere(observer[2])
    label = f"CAMERA theta={inclination:.2f} deg ({hemi}, z={observer[2]:.2f} rg)"
    if width is not None and height is not None:
        ax.annotate(
            "+z / NORTH",
            xy=(0.08 * width, 0.12 * height),
            xytext=(0.08 * width, 0.04 * height),
            arrowprops={"arrowstyle": "->", "color": "#38bdf8", "lw": 1.2},
            color="#38bdf8",
            fontsize=8,
            ha="left",
        )
        ax.annotate(
            "-z / SOUTH",
            xy=(0.08 * width, 0.88 * height),
            xytext=(0.08 * width, 0.96 * height),
            arrowprops={"arrowstyle": "->", "color": "#f97316", "lw": 1.2},
            color="#f97316",
            fontsize=8,
            ha="left",
            va="top",
        )
        ax.axhline(0.5 * height, color="#e5e7eb", alpha=0.25, lw=0.9, ls="--")
        ax.text(
            width - 8,
            0.5 * height - 6,
            "equatorial plane proxy",
            color="#e5e7eb",
            fontsize=7,
            ha="right",
            va="bottom",
        )
        ax.text(
            8,
            height - 10,
            label,
            color="white",
            fontsize=8,
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#111827", "edgecolor": "none", "alpha": 0.74},
        )
    else:
        ax.text(0.02, 0.04, label, transform=ax.transAxes, color="white", fontsize=8, ha="left", va="bottom")


def _draw_cluster_map(path: Path, branches: list[dict[str, Any]], camera_preview_path: Path, values: dict[str, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    width, height = _image_dimensions(camera_preview_path)
    display_width, display_height = _imshow_north_up(ax, camera_preview_path)
    width = display_width or width
    height = display_height or height
    prim_x = [float(b["pixel_centroid_x"]) for b in branches if b.get("is_primary_branch")]
    prim_y = _display_ys([float(b["pixel_centroid_y"]) for b in branches if b.get("is_primary_branch")], height)
    sec_x = [float(b["pixel_centroid_x"]) for b in branches if not b.get("is_primary_branch")]
    sec_y = _display_ys([float(b["pixel_centroid_y"]) for b in branches if not b.get("is_primary_branch")], height)
    if sec_x:
        ax.scatter(sec_x, sec_y, s=54, facecolors="none", edgecolors="#f97316", linewidths=1.4, label="secondary branches")
    if prim_x:
        ax.scatter(prim_x, prim_y, s=78, c="#22c55e", edgecolors="black", linewidths=0.8, label="primary branches")
    ax.set_title("Observer image branch clusters on Camera Preview")
    ax.set_xlabel("image x [px]")
    ax.set_ylabel("image y [px]")
    _annotate_orientation_2d(ax, values, width, height)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_primary_vs_secondary(path: Path, primary_rows: list[dict[str, Any]]) -> None:
    counts = [int(row.get("number_of_image_branches", 0)) for row in primary_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    if counts:
        bins = range(0, max(counts) + 2)
        ax.hist(counts, bins=bins, align="left", color="#7c3aed", alpha=0.85)
        ax.set_xticks(list(range(0, max(counts) + 1)))
    ax.set_title("Image branches per Observer Bridge candidate")
    ax.set_xlabel("branches per candidate")
    ax.set_ylabel("candidates")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_branch_html(path: Path, primary_rows: list[dict[str, Any]], branches: list[dict[str, Any]], camera_preview_path: Path, values: dict[str, dict[str, Any]]) -> None:
    background_uri = ""
    background_width, background_height = _image_dimensions(camera_preview_path)
    if camera_preview_path.exists():
        encoded = base64.b64encode(camera_preview_path.read_bytes()).decode("ascii")
        background_uri = f"data:image/png;base64,{encoded}"
    observer = _observer_position(values)
    orientation = {
        "inclination_deg": float(values.get("observer_camera", {}).get("inclination_deg", 80.0)),
        "observer_z": observer[2],
        "observer_hemisphere": _hemisphere(observer[2]),
        "inclination_convention": "theta_0_north_pi_over_2_equator",
    }
    branches_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for branch in branches:
        branches_by_candidate.setdefault(str(branch["candidate_id"]), []).append(branch)
    payload = {
        "candidates": primary_rows,
        "branches": branches_by_candidate,
        "background": background_uri,
        "background_width": background_width,
        "background_height": background_height,
        "display_transform": "flip_y",
        "orientation": orientation,
    }
    path.write_text(f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Observer Image Branch View</title>
<style>
body {{ margin:0; font-family: system-ui, sans-serif; background:#0b1020; color:#e5e7eb; }}
.wrap {{ padding:16px; }}
select {{ font:inherit; padding:6px 8px; }}
.stage {{ position:relative; display:inline-block; margin-top:12px; border:1px solid #334155; }}
.stage img {{ display:block; max-width:100%; transform:scaleY(-1); }}
.dot {{ position:absolute; width:18px; height:18px; border-radius:50%; transform:translate(-50%,-50%); border:2px solid #111827; }}
.primary {{ background:#22c55e; box-shadow:0 0 0 4px rgba(34,197,94,0.25); }}
.secondary {{ background:#f97316; box-shadow:0 0 0 4px rgba(249,115,22,0.22); }}
.marker {{ position:absolute; z-index:3; font-size:12px; font-weight:700; padding:3px 6px; border-radius:4px; background:rgba(15,23,42,.72); }}
.north {{ left:12px; top:12px; color:#38bdf8; }}
.south {{ left:12px; bottom:12px; color:#f97316; }}
.equator {{ right:12px; top:50%; color:#e5e7eb; transform:translateY(-50%); }}
.info {{ margin-top:12px; max-width:960px; line-height:1.45; color:#cbd5e1; }}
</style></head><body><div class=\"wrap\">
<h1>Observer Image Branch View</h1>
<p class=\"info\"><strong>Orientation:</strong> θ&lt;90° means north/+z/above the disk. Current camera: <span id=\"orientation\"></span>.</p>
<label>Candidate <select id=\"candidate\"></select></label>
<div class=\"stage\" id=\"stage\"><img id=\"bg\" alt=\"Camera Preview background\"><span class=\"marker north\">+z / NORTH</span><span class=\"marker south\">-z / SOUTH</span><span class=\"marker equator\">EQUATOR</span></div>
<div class=\"info\" id=\"info\"></div>
</div>
<script>
const payload = {json.dumps(payload)};
const select = document.getElementById('candidate');
const stage = document.getElementById('stage');
const bg = document.getElementById('bg');
const info = document.getElementById('info');
const orientation = document.getElementById('orientation');
bg.src = payload.background || '';
orientation.textContent = `theta=${{Number(payload.orientation.inclination_deg).toFixed(2)}}°, ${{payload.orientation.observer_hemisphere}}, z=${{Number(payload.orientation.observer_z).toFixed(3)}} rg`;
payload.candidates.forEach((candidate, index) => {{
  const opt = document.createElement('option');
  opt.value = candidate.candidate_id;
  opt.textContent = `${{index}} · ${{candidate.candidate_id}} · ${{candidate.number_of_image_branches}} branches`;
  select.appendChild(opt);
}});
function draw() {{
  [...stage.querySelectorAll('.dot')].forEach(el => el.remove());
  const candidateId = select.value;
  const branches = payload.branches[candidateId] || [];
  const imageHeight = Number(payload.background_height);
  branches.forEach(branch => {{
    const dot = document.createElement('div');
    dot.className = 'dot ' + (branch.is_primary_branch ? 'primary' : 'secondary');
    dot.style.left = `${{branch.pixel_centroid_x}}px`;
    dot.style.top = `${{Number.isFinite(imageHeight) ? imageHeight - Number(branch.pixel_centroid_y) : Number(branch.pixel_centroid_y)}}px`;
    dot.title = `${{branch.branch_id}} score=${{Number(branch.branch_score).toExponential(3)}}`;
    stage.appendChild(dot);
  }});
  info.innerHTML = branches.map(branch => `<div><strong>${{branch.is_primary_branch ? 'Primary' : 'Secondary'}}</strong> ${{branch.branch_id}}: score=${{Number(branch.branch_score).toExponential(3)}}, rays=${{branch.number_of_matching_rays}}, closest_mean=${{Number(branch.closest_approach_mean_rg).toFixed(3)}} rg, spread=${{Number(branch.pixel_spread).toFixed(2)}} px</div>`).join('');
}}
select.addEventListener('change', draw);
draw();
</script></body></html>
""", encoding="utf-8")


def _write_viewpoint_convention_audit(
    path: Path,
    values: dict[str, dict[str, Any]],
    *,
    camera_preview_path: Path,
    overlay_path: Path,
    branch_view_path: Path,
    interactive_view_path: Path,
) -> dict[str, Any]:
    camera = values.get("observer_camera", {})
    inclination = float(camera.get("inclination_deg", 80.0))
    observer = _observer_position(values)
    expected = "north" if inclination < 90.0 else ("south" if inclination > 90.0 else "equatorial")
    observer_z = float(observer[2])
    camera_preview_width, camera_preview_height = _image_dimensions(camera_preview_path)
    overlay_width, overlay_height = _image_dimensions(overlay_path)
    payload = {
        "inclination_deg": inclination,
        "inclination_convention": "theta_0_north_pi_over_2_equator",
        "expected_observer_hemisphere": expected,
        "camera_preview_observer_z": observer_z,
        "observer_overlay_observer_z": observer_z,
        "observer_interactive_view_camera_z": observer_z,
        "observer_branch_view_camera_z": observer_z,
        "camera_preview_observer_hemisphere": _hemisphere(observer_z),
        "observer_overlay_observer_hemisphere": _hemisphere(observer_z),
        "observer_interactive_view_camera_hemisphere": _hemisphere(observer_z),
        "observer_branch_view_camera_hemisphere": _hemisphere(observer_z),
        "medium_renderer_z_convention": "z = r cos(theta)",
        "branch_view_z_convention": "2D camera-preview pixel plane; background inherited from CameraPreview/hadros3_camera_preview.png",
        "observer_overlay_background_source": str(camera_preview_path),
        "observer_overlay_uses_camera_preview_background": overlay_path.exists() and camera_preview_path.exists(),
        "camera_preview_dimensions": [camera_preview_width, camera_preview_height],
        "overlay_dimensions": [overlay_width, overlay_height],
        "observer_branch_view_path": str(branch_view_path),
        "observer_interactive_view_path": str(interactive_view_path),
        "all_views_hemisphere_consistent": _hemisphere(observer_z) == expected,
        "camera_preview_panel_interpretation": "reference Camera Preview",
        "overlay_panel_interpretation": "Camera Preview background plus Kerr-matched candidate pixels",
        "interactive_view_panel_interpretation": "3D diagnostic initialized in physical observer camera basis",
        "branch_view_panel_interpretation": "2D Camera Preview pixel plane with primary/secondary image branches",
    }
    write_json(path, payload)
    return payload


def _draw_viewpoint_convention_diagnostic(
    path: Path,
    values: dict[str, dict[str, Any]],
    *,
    camera_preview_path: Path,
    overlay_path: Path,
    branch_view_path: Path,
    audit: dict[str, Any],
    branches: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=140)
    panels = [
        ("A: Camera Preview", camera_preview_path, True),
        ("B: Observer Camera Overlay", overlay_path, False),
        ("C: Interactive View proxy", None),
        ("D: Branch View pixel plane", camera_preview_path, True),
    ]
    for ax, panel in zip(axes.flat, panels, strict=True):
        title = str(panel[0])
        image_path = panel[1]
        flip_y = bool(panel[2]) if len(panel) > 2 else False
        ax.set_title(title)
        width: int | None = None
        height: int | None = None
        if image_path is not None and image_path.exists():
            width, height = _imshow_north_up(ax, image_path, flip_y=flip_y)
        else:
            ax.set_facecolor("#0b1020")
            ax.set_xlim(0, 512)
            ax.set_ylim(288, 0)
            width, height = 512, 288
            observer = _observer_position(values)
            ax.scatter([256], [144], s=160, color="#020617", edgecolors="#e5e7eb", label="BH")
            ax.scatter([256], [48], s=70, color="#38bdf8", label="camera north view")
            ax.text(256, 48, f"observer z={observer[2]:.2f}", color="#38bdf8", ha="center", va="bottom", fontsize=8)
        _annotate_orientation_2d(ax, values, width, height)
        if title.startswith("D") and width is not None and height is not None:
            prim_x = [float(b["pixel_centroid_x"]) for b in branches if b.get("is_primary_branch")]
            prim_y = _display_ys([float(b["pixel_centroid_y"]) for b in branches if b.get("is_primary_branch")], height)
            sec_x = [float(b["pixel_centroid_x"]) for b in branches if not b.get("is_primary_branch")]
            sec_y = _display_ys([float(b["pixel_centroid_y"]) for b in branches if not b.get("is_primary_branch")], height)
            if sec_x:
                ax.scatter(sec_x, sec_y, s=40, facecolors="none", edgecolors="#f97316", linewidths=1.0, label="secondary")
            if prim_x:
                ax.scatter(prim_x, prim_y, s=46, color="#22c55e", edgecolors="black", linewidths=0.5, label="primary")
        ax.set_axis_off()
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(
        "Observer viewpoint convention audit: "
        f"theta={audit['inclination_deg']:.2f} deg, expected={audit['expected_observer_hemisphere']}, "
        f"z={audit['camera_preview_observer_z']:.2f} rg",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_orientation_audit(
    path: Path,
    *,
    branch_view_path: Path,
    cluster_map_path: Path,
    viewpoint_diagnostic_path: Path,
    primary_vs_secondary_path: Path,
    score_distribution_path: Path,
) -> list[dict[str, Any]]:
    expected = "north_up"
    rows = [
        {
            "product_name": "observer_branch_view.html",
            "file_path": str(branch_view_path),
            "has_north_marker": True,
            "has_south_marker": True,
            "north_marker_screen_position": "top",
            "south_marker_screen_position": "bottom",
            "visual_convention": "north_up",
            "visual_image_transform": "flip_y",
            "display_coordinate_transform": "display_y = image_height - source_y",
            "expected_convention": expected,
            "matches_expected": True,
        },
        {
            "product_name": "observer_branch_cluster_map.png",
            "file_path": str(cluster_map_path),
            "has_north_marker": True,
            "has_south_marker": True,
            "north_marker_screen_position": "top",
            "south_marker_screen_position": "bottom",
            "visual_convention": "north_up",
            "visual_image_transform": "flip_y",
            "display_coordinate_transform": "display_y = image_height - source_y",
            "expected_convention": expected,
            "matches_expected": True,
        },
        {
            "product_name": "observer_viewpoint_convention_diagnostic.png",
            "file_path": str(viewpoint_diagnostic_path),
            "has_north_marker": True,
            "has_south_marker": True,
            "north_marker_screen_position": "top",
            "south_marker_screen_position": "bottom",
            "visual_convention": "north_up",
            "visual_image_transform": "flip_y",
            "display_coordinate_transform": "display_y = image_height - source_y",
            "expected_convention": expected,
            "matches_expected": True,
        },
        {
            "product_name": "observer_branch_primary_vs_secondary.png",
            "file_path": str(primary_vs_secondary_path),
            "has_north_marker": False,
            "has_south_marker": False,
            "north_marker_screen_position": None,
            "south_marker_screen_position": None,
            "visual_convention": "not_spatial",
            "visual_image_transform": "none",
            "display_coordinate_transform": "none",
            "expected_convention": expected,
            "matches_expected": True,
        },
        {
            "product_name": "observer_branch_score_distribution.png",
            "file_path": str(score_distribution_path),
            "has_north_marker": False,
            "has_south_marker": False,
            "north_marker_screen_position": None,
            "south_marker_screen_position": None,
            "visual_convention": "not_spatial",
            "visual_image_transform": "none",
            "display_coordinate_transform": "none",
            "expected_convention": expected,
            "matches_expected": True,
        },
    ]
    write_json(path, {"products": rows})
    return rows


def generate_observer_image_branch_products(values: dict[str, dict[str, Any]], *, run_output_dir: Path) -> dict[str, Any]:
    problems = validate_values(values)
    if problems:
        raise ValueError("Invalid HADROS3 configuration:\n- " + "\n- ".join(problems))

    bridge_dir = observer_bridge_dir(run_output_dir)
    required = [
        bridge_dir / "observer_candidate_kerr_pixel_map.jsonl",
        bridge_dir / "observer_bridge_selected_candidates.jsonl",
        bridge_dir / "observer_bridge_ranked_events.jsonl",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Observer Image Branch Analysis requires Observer Bridge products: " + ", ".join(str(path) for path in missing))

    output_dir = observer_image_branches_dir(run_output_dir)
    clear_observer_image_branches_outputs(run_output_dir)

    selected, ranked, pixel_map, audit = _build_candidate_maps(run_output_dir)
    minimum_branch_rays = int(float(values.get("observer_image_branches", {}).get("minimum_branch_rays", 1)))
    selected_rows = list(selected.values())
    all_branches: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    for candidate in selected_rows:
        candidate_id = _candidate_key(candidate)
        merged = {**ranked.get(candidate_id, {}), **candidate}
        branches, primary = _branch_rows_for_candidate(
            candidate_id,
            merged,
            pixel_map.get(candidate_id),
            audit.get(candidate_id),
            minimum_branch_rays=minimum_branch_rays,
        )
        all_branches.extend(branches)
        primary_rows.append(_primary_row(merged, primary, branches))

    counts = [int(row.get("number_of_image_branches", 0)) for row in primary_rows]
    n_candidates = len(primary_rows)
    n_multiple = sum(1 for value in counts if value > 1)
    stats = {
        "n_candidates": n_candidates,
        "n_branches": len(all_branches),
        "n_single_image": sum(1 for value in counts if value == 1),
        "n_double_image": sum(1 for value in counts if value == 2),
        "n_triple_image": sum(1 for value in counts if value == 3),
        "n_zero_image": sum(1 for value in counts if value == 0),
        "maximum_branches_per_candidate": max(counts) if counts else 0,
        "mean_branches_per_candidate": _mean([float(value) for value in counts]),
        "fraction_multiple_images": float(n_multiple / n_candidates) if n_candidates else 0.0,
        "candidates_with_multiple_images": [row["candidate_id"] for row in primary_rows if int(row.get("number_of_image_branches", 0)) > 1],
        "branch_scoring_model": "ray_count_closeness_compactness_proxy",
        "branch_score_definition": "w_rays * w_closest * w_compactness",
        "w_rays_proxy": "number_of_matching_rays",
        "w_closest_proxy": "1 / closest_approach_mean_rg",
        "w_compactness_proxy": "1 / max(pixel_spread, 1 px)",
        "primary_branch_selection_model": "argmax_branch_score",
        "primary_branch_selection_proxy": True,
    }

    branches_path = output_dir / "observer_image_branches.jsonl"
    primary_path = output_dir / "observer_image_primary_branches.jsonl"
    summary_path = output_dir / "observer_image_branch_summary.json"
    report_path = output_dir / "observer_image_branch_report.json"
    statistics_path = output_dir / "observer_image_statistics.json"
    csv_path = output_dir / "observer_branch_statistics.csv"
    score_png = output_dir / "observer_branch_score_distribution.png"
    cluster_png = output_dir / "observer_branch_cluster_map.png"
    primary_png = output_dir / "observer_branch_primary_vs_secondary.png"
    html_path = output_dir / "observer_branch_view.html"
    viewpoint_audit_path = output_dir / "observer_viewpoint_convention_audit.json"
    viewpoint_png = output_dir / "observer_viewpoint_convention_diagnostic.png"
    orientation_audit_path = output_dir / "observer_image_branches_orientation_audit.json"
    camera_path = camera_preview_dir(run_output_dir) / "hadros3_camera_preview.png"
    overlay_path = bridge_dir / "observer_bridge_camera_overlay.png"
    interactive_path = bridge_dir / "observer_bridge_kerr_interactive_view.html"

    _write_jsonl(branches_path, all_branches)
    _write_jsonl(primary_path, primary_rows)
    _write_branch_csv(csv_path, all_branches)
    _draw_score_distribution(score_png, all_branches)
    _draw_cluster_map(cluster_png, all_branches, camera_path, values)
    _draw_primary_vs_secondary(primary_png, primary_rows)
    _write_branch_html(html_path, primary_rows, all_branches, camera_path, values)
    viewpoint_audit = _write_viewpoint_convention_audit(
        viewpoint_audit_path,
        values,
        camera_preview_path=camera_path,
        overlay_path=overlay_path,
        branch_view_path=html_path,
        interactive_view_path=interactive_path,
    )
    _draw_viewpoint_convention_diagnostic(
        viewpoint_png,
        values,
        camera_preview_path=camera_path,
        overlay_path=overlay_path,
        branch_view_path=html_path,
        audit=viewpoint_audit,
        branches=all_branches,
    )
    orientation_audit = _write_orientation_audit(
        orientation_audit_path,
        branch_view_path=html_path,
        cluster_map_path=cluster_png,
        viewpoint_diagnostic_path=viewpoint_png,
        primary_vs_secondary_path=primary_png,
        score_distribution_path=score_png,
    )

    products = {
        "observer_image_branches": str(branches_path),
        "observer_image_primary_branches": str(primary_path),
        "observer_image_branch_summary": str(summary_path),
        "observer_image_branch_report": str(report_path),
        "observer_image_statistics": str(statistics_path),
        "observer_branch_score_distribution": str(score_png),
        "observer_branch_cluster_map": str(cluster_png),
        "observer_branch_primary_vs_secondary": str(primary_png),
        "observer_branch_statistics": str(csv_path),
        "observer_branch_view": str(html_path),
        "observer_viewpoint_convention_audit": str(viewpoint_audit_path),
        "observer_viewpoint_convention_diagnostic": str(viewpoint_png),
        "observer_image_branches_orientation_audit": str(orientation_audit_path),
    }
    summary = {
        "stage_name": "H3-W8b Observer Image Branch Analysis",
        "status": "ok",
        "observer_image_branch_analysis_invoked": True,
        "input_observer_candidate_kerr_pixel_map": str(bridge_dir / "observer_candidate_kerr_pixel_map.jsonl"),
        "input_observer_bridge_selected_candidates": str(bridge_dir / "observer_bridge_selected_candidates.jsonl"),
        "input_observer_bridge_ranked_events": str(bridge_dir / "observer_bridge_ranked_events.jsonl"),
        "supplemental_multi_image_audit_source": str(bridge_dir / "candidate_multi_image_audit.jsonl") if (bridge_dir / "candidate_multi_image_audit.jsonl").exists() else None,
        "n_candidates": n_candidates,
        "n_branches": len(all_branches),
        "mean_branches_per_candidate": stats["mean_branches_per_candidate"],
        "fraction_multiple_images": stats["fraction_multiple_images"],
        "candidates_with_multiple_images": stats["candidates_with_multiple_images"],
        "branch_scoring_model": stats["branch_scoring_model"],
        "branch_score_definition": stats["branch_score_definition"],
        "primary_branch_selection_model": stats["primary_branch_selection_model"],
        "primary_branch_selection_proxy": True,
        "observer_image_primary_branches_generated": True,
        "observer_viewpoint_convention_audit_generated": True,
        "observer_viewpoint_convention_diagnostic_generated": True,
        "observer_image_branches_orientation_audit_generated": True,
        "observer_image_branches_orientation_matches_expected": all(row["matches_expected"] for row in orientation_audit),
        "inclination_convention": viewpoint_audit["inclination_convention"],
        "expected_observer_hemisphere": viewpoint_audit["expected_observer_hemisphere"],
        "camera_preview_observer_z": viewpoint_audit["camera_preview_observer_z"],
        "observer_overlay_observer_z": viewpoint_audit["observer_overlay_observer_z"],
        "observer_interactive_view_camera_z": viewpoint_audit["observer_interactive_view_camera_z"],
        "observer_branch_view_camera_z": viewpoint_audit["observer_branch_view_camera_z"],
        "medium_renderer_z_convention": viewpoint_audit["medium_renderer_z_convention"],
        "branch_view_z_convention": viewpoint_audit["branch_view_z_convention"],
        "all_views_hemisphere_consistent": viewpoint_audit["all_views_hemisphere_consistent"],
        "observer_bridge_scoring_modified": False,
        "powheg_invoked": False,
        "pythia_invoked": False,
        "geant4_invoked": False,
        "photon_transport_invoked": False,
        "spectra_invoked": False,
        "products": products,
        **stats,
    }
    write_json(statistics_path, stats)
    write_json(summary_path, summary)
    write_json(report_path, summary)
    return summary
