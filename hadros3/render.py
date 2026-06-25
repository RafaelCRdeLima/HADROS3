"""Static HADROS3 geometry previews for the first hadros-web stage."""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hadros3_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Ellipse, Polygon, Wedge


def _f(values: dict[str, dict[str, Any]], section: str, key: str) -> float:
    return float(values[section][key])


def _resolution(value: str) -> tuple[int, int]:
    text = str(value).lower().strip()
    if "x" not in text:
        n = int(text)
        return n, n
    left, right = text.split("x", 1)
    return int(left), int(right)


def kerr_horizon_radius_rg(spin: float) -> float:
    a = max(-0.999, min(0.999, spin))
    return 1.0 + math.sqrt(1.0 - a * a)


def _fmt(value: float, digits: int = 2) -> str:
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text or "0"


def _angle_arc(ax: Any, radius: float, theta1: float, theta2: float, *, color: str, label: str, label_angle: float) -> None:
    ax.add_patch(Arc((0.0, 0.0), 2 * radius, 2 * radius, theta1=theta1, theta2=theta2, color=color, linewidth=1.0))
    a = math.radians(label_angle)
    ax.text(radius * math.cos(a), radius * math.sin(a), label, color=color, fontsize=8, ha="center", va="center")


def draw_geometry_preview(values: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spin = _f(values, "black_hole", "spin_a")
    horizon = kerr_horizon_radius_rg(spin)
    torus_r_in = _f(values, "analytic_torus", "r_inner_rg")
    torus_r_out = _f(values, "analytic_torus", "r_outer_rg")
    torus_r_peak = _f(values, "analytic_torus", "r_peak_rg")
    cone_angle = math.radians(_f(values, "polar_cone", "opening_angle_deg"))
    cone_r_min = _f(values, "polar_cone", "r_min_rg")
    cone_r_max = _f(values, "polar_cone", "r_max_rg")
    src_r_min = _f(values, "uhe_neutrino_source", "r_min_rg")
    src_r_max = _f(values, "uhe_neutrino_source", "r_max_rg")
    obs_r = _f(values, "observer_camera", "observer_distance_rg")
    inc = math.radians(_f(values, "observer_camera", "inclination_deg"))
    fov = math.radians(_f(values, "observer_camera", "field_of_view_deg"))

    scale_h = max(horizon, 1.0e-6)
    torus_in_h = torus_r_in / scale_h
    torus_out_h = torus_r_out / scale_h
    torus_peak_h = torus_r_peak / scale_h
    cone_min_h = cone_r_min / scale_h
    cone_max_h = cone_r_max / scale_h
    src_min_h = src_r_min / scale_h
    src_max_h = src_r_max / scale_h
    obs_h = obs_r / scale_h

    lim = max(torus_out_h * 1.35, cone_max_h * 1.1, obs_h * 0.22, 16.0)
    fig, ax = plt.subplots(figsize=(11.5, 8.8), facecolor="#101318")
    ax.set_facecolor("#101318")

    if bool(values["analytic_torus"]["show_in_preview"]):
        torus = Wedge(
            (0.0, 0.0),
            torus_out_h,
            0.0,
            360.0,
            width=max(torus_out_h - torus_in_h, 1.0e-6),
            facecolor="#2372a3",
            edgecolor="#95d9ff",
            linewidth=1.4,
            alpha=0.34,
        )
        ax.add_patch(torus)
        ax.add_patch(Circle((0.0, 0.0), torus_peak_h, fill=False, edgecolor="#d7f1ff", linestyle="--", linewidth=1.0))
        ax.text(
            -0.98 * lim,
            0.86 * lim,
            f"torus: Rin={_fmt(torus_in_h)} rH, Rpeak={_fmt(torus_peak_h)} rH, Rout={_fmt(torus_out_h)} rH",
            color="#bfeaff",
            fontsize=9,
            ha="left",
        )

    if bool(values["polar_cone"]["enabled"]):
        for sign in (1, -1) if values["polar_cone"]["draw_mode"] == "bipolar_funnel" else (1,):
            left = np.array([sign * cone_min_h * math.sin(cone_angle), sign * cone_min_h * math.cos(cone_angle)])
            right = np.array([-sign * cone_min_h * math.sin(cone_angle), sign * cone_min_h * math.cos(cone_angle)])
            tip_left = np.array([sign * cone_max_h * math.sin(cone_angle), sign * cone_max_h * math.cos(cone_angle)])
            tip_right = np.array([-sign * cone_max_h * math.sin(cone_angle), sign * cone_max_h * math.cos(cone_angle)])
            poly = Polygon([left, tip_left, tip_right, right], closed=True, facecolor="#f0c84b", edgecolor="#ffec99", alpha=0.18, linewidth=1.1)
            ax.add_patch(poly)
            ax.plot([left[0], tip_left[0]], [left[1], tip_left[1]], color="#ffe680", linewidth=1.2)
            ax.plot([right[0], tip_right[0]], [right[1], tip_right[1]], color="#ffe680", linewidth=1.2)
        _angle_arc(
            ax,
            max(2.4, 0.18 * lim),
            90.0 - math.degrees(cone_angle),
            90.0 + math.degrees(cone_angle),
            color="#ffe680",
            label=f"cone {math.degrees(cone_angle):.1f} deg",
            label_angle=90.0,
        )

    src_width = max(src_max_h - src_min_h, 0.2 / scale_h)
    source = Wedge(
        (0.0, 0.0),
        src_max_h,
        90.0 - math.degrees(cone_angle),
        90.0 + math.degrees(cone_angle),
        width=src_width,
        facecolor="#ff6f59",
        edgecolor="#ffd1c9",
        alpha=0.72,
        linewidth=1.0,
    )
    ax.add_patch(source)

    ax.add_patch(Circle((0.0, 0.0), 1.0, facecolor="black", edgecolor="#f4f4f4", linewidth=1.3, zorder=8))
    ax.add_patch(Circle((0.0, 0.0), 2.0 / scale_h, fill=False, edgecolor="#bbbbbb", linestyle="-.", linewidth=0.8, alpha=0.65))
    obs = np.array([obs_h * math.sin(inc), obs_h * math.cos(inc)])
    obs_draw = obs / max(np.linalg.norm(obs), 1.0) * min(lim * 0.92, obs_h)
    ax.scatter([obs_draw[0]], [obs_draw[1]], s=70, color="#d6ff6b", edgecolor="#0c0f14", zorder=10)
    ax.text(obs_draw[0], obs_draw[1], " observer", color="#e8ff9e", fontsize=10, va="center")

    target = np.array([0.0, 0.0])
    direction = target - obs_draw
    direction /= max(np.linalg.norm(direction), 1.0e-12)
    normal = np.array([-direction[1], direction[0]])
    frustum_len = min(lim * 0.48, np.linalg.norm(obs_draw) * 0.88)
    center = obs_draw + direction * frustum_len
    half_width = math.tan(0.5 * fov) * frustum_len
    left = center + normal * half_width
    right = center - normal * half_width
    ax.plot([obs_draw[0], left[0]], [obs_draw[1], left[1]], color="#d6ff6b", linestyle=":", linewidth=1.0)
    ax.plot([obs_draw[0], right[0]], [obs_draw[1], right[1]], color="#d6ff6b", linestyle=":", linewidth=1.0)
    ax.plot([obs_draw[0], 0.0], [obs_draw[1], 0.0], color="#d6ff6b", linestyle="--", linewidth=0.7, alpha=0.45)
    fov_label_pos = obs_draw + direction * (0.34 * frustum_len)
    ax.text(
        fov_label_pos[0],
        fov_label_pos[1],
        f"FOV {math.degrees(fov):.1f} deg",
        color="#d6ff6b",
        fontsize=8,
        ha="center",
        bbox=dict(facecolor="#101318", edgecolor="#5d742d", alpha=0.85, pad=2),
    )

    ax.text(0.0, 0.0, "BH", color="white", ha="center", va="center", fontsize=11, zorder=11)
    ax.annotate("analytic torus", xy=(0.0, torus_peak_h), xytext=(-0.92 * lim, 0.56 * lim), color="#bfeaff", arrowprops=dict(arrowstyle="->", color="#bfeaff"))
    ax.annotate("polar UHE source", xy=(0.0, src_max_h), xytext=(0.32 * lim, 0.72 * lim), color="#ffd5cd", arrowprops=dict(arrowstyle="->", color="#ffd5cd"))
    ax.annotate(
        f"source {_fmt(src_min_h)}-{_fmt(src_max_h)} rH",
        xy=(0.0, src_max_h),
        xytext=(-0.15 * lim, 0.32 * lim),
        color="#ffd5cd",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#ffd5cd", alpha=0.75),
    )
    ax.annotate("polar funnel", xy=(cone_max_h * math.sin(cone_angle), cone_max_h * math.cos(cone_angle)), xytext=(0.34 * lim, 0.34 * lim), color="#ffe680", arrowprops=dict(arrowstyle="->", color="#ffe680"))

    info = (
        f"a={spin:g}\n"
        f"rH={horizon:.3g} rg\n"
        f"M={_f(values, 'black_hole', 'mass_msun'):g} Msun\n"
        f"camera r={obs_r:g} rg = {_fmt(obs_h)} rH\n"
        f"inclination={math.degrees(inc):g} deg\n"
        f"FOV={math.degrees(fov):g} deg"
    )
    ax.text(0.97 * lim, -0.96 * lim, info, color="#e7ebf2", ha="right", va="bottom", fontsize=9, bbox=dict(facecolor="#1d2430", edgecolor="#596577", alpha=0.9, pad=5))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    scale_len = max(1.0, round(lim / 5.0))
    ax.plot([-0.96 * lim, -0.96 * lim + scale_len], [-0.88 * lim, -0.88 * lim], color="#e7ebf2", linewidth=2.0)
    ax.text(-0.96 * lim + 0.5 * scale_len, -0.86 * lim, f"{_fmt(scale_len, 0)} rH", color="#e7ebf2", fontsize=8, ha="center", va="bottom")

    ax.set_xlabel("x / rH")
    ax.set_ylabel("z / rH")
    ax.grid(color="#354052", linestyle=":", linewidth=0.65, alpha=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_system_schematic(values: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spin = _f(values, "black_hole", "spin_a")
    horizon = kerr_horizon_radius_rg(spin)
    torus_r_peak = _f(values, "analytic_torus", "r_peak_rg")
    torus_r_in = _f(values, "analytic_torus", "r_inner_rg")
    torus_r_out = _f(values, "analytic_torus", "r_outer_rg")
    cone_angle = math.radians(_f(values, "polar_cone", "opening_angle_deg"))
    cone_r_max = _f(values, "polar_cone", "r_max_rg")
    src_r_min = _f(values, "uhe_neutrino_source", "r_min_rg")
    src_r_max = _f(values, "uhe_neutrino_source", "r_max_rg")
    obs_r = _f(values, "observer_camera", "observer_distance_rg")
    inc = math.radians(_f(values, "observer_camera", "inclination_deg"))

    lim = max(1.15 * cone_r_max, 1.35 * torus_r_out, 32.0)
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor="#f5f7fa")
    ax.set_facecolor("#f5f7fa")

    torus_width = max((torus_r_out - torus_r_in) * 0.52, 1.0)
    ax.add_patch(Ellipse((0.0, 0.0), width=2.0 * torus_r_peak, height=2.0 * torus_width, facecolor="#79aeda", edgecolor="#22577a", linewidth=1.3, alpha=0.46))
    ax.add_patch(Ellipse((0.0, 0.0), width=2.0 * torus_r_in, height=max(0.5, 2.0 * torus_width * 0.28), facecolor="#f5f7fa", edgecolor="#22577a", linewidth=0.9, alpha=0.95))

    for sign in (1, -1):
        x = cone_r_max * math.sin(cone_angle)
        y = sign * cone_r_max * math.cos(cone_angle)
        ax.add_patch(Polygon([[0.0, sign * horizon], [x, y], [-x, y]], closed=True, facecolor="#ffd166", edgecolor="#c99a17", alpha=0.34, linewidth=1.2))

    ax.add_patch(Wedge((0.0, 0.0), src_r_max, 90 - math.degrees(cone_angle), 90 + math.degrees(cone_angle), width=max(src_r_max - src_r_min, 0.5), facecolor="#ef476f", alpha=0.78, edgecolor="#7a1730"))
    ax.add_patch(Circle((0.0, 0.0), horizon, facecolor="#111111", edgecolor="#222222", linewidth=1.3, zorder=8))

    obs = np.array([obs_r * math.sin(inc), obs_r * math.cos(inc)])
    obs_draw = obs / max(np.linalg.norm(obs), 1.0) * (lim * 0.9)
    ax.scatter([obs_draw[0]], [obs_draw[1]], s=80, marker="s", color="#2a9d8f", edgecolor="#173f3a", zorder=10)
    ax.plot([obs_draw[0], 0.0], [obs_draw[1], 0.0], color="#2a9d8f", linestyle="--", linewidth=1.0, alpha=0.8)

    ax.text(0.0, -0.75, "Kerr BH", color="white", ha="center", va="center", fontsize=9, zorder=12)
    ax.text(0.0, -torus_width - 2.0, "analytic torus", color="#22577a", ha="center", va="top", fontsize=10)
    ax.text(0.0, cone_r_max * 0.72, "polar cone / funnel", color="#8a690d", ha="center", va="center", fontsize=10)
    ax.text(0.0, src_r_max + 1.5, "UHE neutrino source region", color="#7a1730", ha="center", va="bottom", fontsize=10)
    ax.text(obs_draw[0], obs_draw[1] + 1.2, "observer camera", color="#173f3a", ha="center", va="bottom", fontsize=10)

    y_angle = min(cone_r_max * 0.46, lim * 0.36)
    x_angle = y_angle * math.tan(cone_angle)
    ax.plot([0, x_angle], [0, y_angle], color="#8a690d", linewidth=1.1)
    ax.text(x_angle + 1.0, y_angle * 0.55, f"{math.degrees(cone_angle):.1f} deg", color="#8a690d", fontsize=9)

    summary = (
        f"HADROS3 H3-W0..W4\n"
        f"spin a={spin:g}; horizon={horizon:.2f} rg\n"
        f"torus {torus_r_in:g}-{torus_r_out:g} rg\n"
        f"source {src_r_min:g}-{src_r_max:g} rg\n"
        f"no POWHEG/PYTHIA/GEANT4"
    )
    ax.text(-0.98 * lim, -0.93 * lim, summary, color="#27313f", ha="left", va="bottom", fontsize=9, bbox=dict(facecolor="white", edgecolor="#c8d0db", pad=6))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_html_summary(values: dict[str, dict[str, Any]], products: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def href(key: str) -> str:
        product = products.get(key)
        if not product:
            return ""
        return Path(os.path.relpath(product, path.parent)).as_posix()

    def product_json(key: str) -> dict[str, Any]:
        product = products.get(key)
        if not product:
            return {}
        try:
            return json.loads(Path(product).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    preview = href("geometry_preview")
    schematic = href("system_schematic")
    camera_preview = href("camera_preview")
    config = href("config")
    provenance = href("provenance")
    direction_model = values.get("uhe_neutrino_source", {}).get("direction_model", "unknown")
    direction_seed = values.get("uhe_neutrino_source", {}).get("direction_seed", "unknown")
    forward_backend = values.get("forward_geodesics", {}).get("geodesic_backend", "unknown")
    dis_summary = product_json("dis_summary_json")
    sigma_energy_min = dis_summary.get("sigma_table_energy_min_gev", dis_summary.get("sigma_energy_min_gev", "pending"))
    sigma_energy_max = dis_summary.get("sigma_table_energy_max_gev", dis_summary.get("sigma_energy_max_gev", "pending"))
    sigma_table_section = ""
    if dis_summary:
        sigma_table_section = f"""
  <section>
    <h2>DIS Sigma Table</h2>
    <ul>
      <li><code>sigma_table_path</code>: {dis_summary.get("sigma_table_path", "pending")}</li>
      <li><code>sigma_table_rows</code>: {dis_summary.get("sigma_table_rows", "pending")}</li>
      <li><code>sigma_table_is_compact_builtin_adapter</code>: {dis_summary.get("sigma_table_is_compact_builtin_adapter", "pending")}</li>
      <li><code>sigma_table_physics_risk</code>: {dis_summary.get("sigma_table_physics_risk", "pending")}</li>
      <li><code>sigma_table_energy_min_gev</code>: {sigma_energy_min}</li>
      <li><code>sigma_table_energy_max_gev</code>: {sigma_energy_max}</li>
    </ul>
  </section>
"""
    product_links = "\n".join(
        f'<li><a href="{href(key)}">{key}</a>: <code>{href(key)}</code></li>'
        for key in sorted(products)
        if href(key)
    )
    params = json.dumps(values, indent=2, sort_keys=True)
    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>HADROS3 hadros-web preview</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; color: #18202a; background: #f5f7fa; }}
    header {{ padding: 18px 24px; background: #18202a; color: white; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 20px; }}
    section {{ border-top: 1px solid #d6dce5; padding: 18px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }}
    img {{ width: 100%; background: white; border: 1px solid #d6dce5; border-radius: 6px; }}
    code, pre {{ background: #101318; color: #f0f4f8; border-radius: 6px; }}
    code {{ padding: 2px 5px; }}
    pre {{ padding: 14px; overflow: auto; }}
  </style>
</head>
<body>
<header><h1>HADROS3 hadros-web</h1></header>
<main>
  <section>
    <p>Geometry/configuration shell only. POWHEG, PYTHIA, GEANT4, forward neutrino geodesics, optical-depth DIS, and active observer bridge are disabled.</p>
    <p>Config: <a href="{config}"><code>{config}</code></a> Provenance: <a href="{provenance}"><code>{provenance}</code></a></p>
  </section>
  <section class="grid">
    <figure><img src="{preview}" alt="HADROS3 geometry preview"><figcaption>Geometry preview</figcaption></figure>
    <figure><img src="{schematic}" alt="HADROS3 schematic"><figcaption>System schematic</figcaption></figure>
    <figure><img src="{camera_preview}" alt="HADROS3 camera preview"><figcaption>Camera preview</figcaption></figure>
  </section>
  <section>
    <h2>Initial Direction</h2>
    <p>The UHE source samples emission position, energy and direction.</p>
    <p>The Kerr four-momentum is not sampled here; it is constructed later by Forward Geodesics from position + energy + direction.</p>
    <ul>
      <li><code>direction_model</code>: {direction_model}</li>
      <li><code>direction_seed</code>: {direction_seed}</li>
    </ul>
  </section>
  <section>
    <h2>Forward Geodesics Contract</h2>
    <ul>
      <li>Input: <code>UHEsource/uhe_neutrino_source_samples.jsonl</code></li>
      <li>Uses: position + energy + emission_direction</li>
      <li>Builds: Kerr null four-momentum <code>p_mu</code></li>
      <li>Propagation: Full Kerr null geodesic propagation via <code>{forward_backend}</code></li>
    </ul>
  </section>
  {sigma_table_section}
  <section><h2>Products</h2><ul>{product_links}</ul></section>
  <section><h2>Parameters</h2><pre>{params}</pre></section>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
