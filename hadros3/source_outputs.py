"""Output writers for HADROS3 UHE source samples."""

from __future__ import annotations

import csv
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
from matplotlib.patches import Circle, Polygon, Wedge

from .render import kerr_horizon_radius_rg


def source_summary(records: list[dict[str, Any]], values: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("source sampler produced no records")
    radii = [float(item["position"]["r_rg"]) for item in records]
    theta_deg = [float(item["position"]["theta_deg"]) for item in records]
    phi_deg = [float(item["position"]["phi_deg"]) for item in records]
    weights = [float(item["source_weight"]) for item in records]
    pdfs = [float(item["source_sampling_pdf"]) for item in records]
    return {
        "status": "ok",
        "n_samples": len(records),
        "source_sampler_active": True,
        "source_model": "polar_cone",
        "source_volume_model": "coordinate_volume",
        "energy_model": values["uhe_neutrino_source"]["energy_model"],
        "energy_gev": float(records[0]["E_nu_emit_gev"]),
        "random_seed": int(float(values["uhe_neutrino_source"]["random_seed"])),
        "sampling_mode": values["uhe_neutrino_source"]["sampling_mode"],
        "momentum_generator": records[0]["momentum_generator"],
        "momentum_is_physical_kerr": bool(records[0]["momentum_is_physical_kerr"]),
        "source_status": records[0]["source_status"],
        "r_min_sampled_rg": min(radii),
        "r_max_sampled_rg": max(radii),
        "theta_min_sampled_deg": min(theta_deg),
        "theta_max_sampled_deg": max(theta_deg),
        "phi_min_sampled_deg": min(phi_deg),
        "phi_max_sampled_deg": max(phi_deg),
        "source_sampling_pdf": pdfs[0],
        "source_physical_pdf": records[0]["source_physical_pdf"],
        "source_weight_min": min(weights),
        "source_weight_max": max(weights),
        "source_weight_mean": sum(weights) / len(weights),
    }


def write_source_samples_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_source_summary_csv(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def write_source_summary_json(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def draw_source_preview(records: list[dict[str, Any]], values: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spin = float(values["black_hole"]["spin_a"])
    horizon = kerr_horizon_radius_rg(spin)
    torus_in = float(values["analytic_torus"]["r_inner_rg"]) / horizon
    torus_out = float(values["analytic_torus"]["r_outer_rg"]) / horizon
    cone_angle = math.radians(float(values["polar_cone"]["opening_angle_deg"]))
    cone_max = float(values["polar_cone"]["r_max_rg"]) / horizon
    src_min = float(values["uhe_neutrino_source"]["r_min_rg"]) / horizon
    src_max = float(values["uhe_neutrino_source"]["r_max_rg"]) / horizon
    xs = []
    zs = []
    for record in records:
        r_h = float(record["position"]["r_rg"]) / horizon
        theta = float(record["position"]["theta_rad"])
        phi = float(record["position"]["phi_rad"])
        xs.append(r_h * math.sin(theta) * math.cos(phi))
        zs.append(r_h * math.cos(theta))

    lim = max(torus_out * 1.3, cone_max * 1.08, src_max * 1.35, 12.0)
    fig, ax = plt.subplots(figsize=(10.5, 8.2), facecolor="#101318")
    ax.set_facecolor("#101318")
    ax.add_patch(Wedge((0.0, 0.0), torus_out, 0, 360, width=max(torus_out - torus_in, 1.0e-6), facecolor="#2372a3", edgecolor="#95d9ff", alpha=0.22))
    for sign in (1, -1) if values["polar_cone"]["draw_mode"] == "bipolar_funnel" else (1,):
        points = [
            (sign * src_min * math.sin(cone_angle), sign * src_min * math.cos(cone_angle)),
            (sign * cone_max * math.sin(cone_angle), sign * cone_max * math.cos(cone_angle)),
            (-sign * cone_max * math.sin(cone_angle), sign * cone_max * math.cos(cone_angle)),
            (-sign * src_min * math.sin(cone_angle), sign * src_min * math.cos(cone_angle)),
        ]
        ax.add_patch(Polygon(points, closed=True, facecolor="#f0c84b", edgecolor="#ffec99", alpha=0.12, linewidth=1.0))
    ax.add_patch(Wedge((0.0, 0.0), src_max, 90 - math.degrees(cone_angle), 90 + math.degrees(cone_angle), width=max(src_max - src_min, 1.0e-6), facecolor="#ff6f59", alpha=0.22))
    ax.scatter(xs, zs, s=9, c="#ffdf7e", alpha=0.68, edgecolors="none")
    ax.add_patch(Circle((0.0, 0.0), 1.0, facecolor="black", edgecolor="white", linewidth=1.1, zorder=5))
    ax.text(0.0, 0.0, "BH", color="white", ha="center", va="center", fontsize=9, zorder=6)
    ax.text(-0.96 * lim, 0.88 * lim, f"UHE source samples: {len(records)}", color="#ffdf7e", fontsize=11, ha="left")
    ax.text(-0.96 * lim, 0.82 * lim, "coordinate-volume polar_cone; proxy radial direction", color="#cbd5e1", fontsize=9, ha="left")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / rH")
    ax.set_ylabel("z / rH")
    ax.grid(color="#354052", linestyle=":", linewidth=0.65, alpha=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
