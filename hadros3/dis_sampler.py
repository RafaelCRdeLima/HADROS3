"""HADROS3 H3-W7 optical-depth DIS interaction sampler."""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import statistics
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import validate_values
from .forward_geodesics import kerr_covariant_metric_components
from .medium_model import analytic_torus_density_g_cm3 as shared_analytic_torus_density_g_cm3
from .medium_model import medium_metadata
from .medium_renderer import MediumRenderer
from .paths import DIS_DIR, dis_dir, forward_geodesics_dir, uhe_source_dir


G_CGS = 6.67430e-8
C_CGS = 2.99792458e10
MSUN_G = 1.98847e33
M_BARYON_G = 1.67262192369e-24
SIGMA_TABLE_PATHS = {
    "GBW": Path("data/sigma/sigma_nuN_CC_GBW.dat"),
    "IIM": Path("data/sigma/sigma_nuN_CC_IIM.dat"),
}
ROOT = Path(__file__).resolve().parents[1]
DIS_CPP_EXECUTABLE = ROOT / "bin" / "hadros3_dis_sampler"


@dataclass(frozen=True)
class DISConfig:
    medium_model: str
    medium_velocity_model: str
    density_floor_g_cm3: float
    dis_model: str
    interaction_sampling_mode: str
    max_interactions: int
    random_seed: int
    mass_msun: float
    spin_a: float


class SigmaNuNProvider:
    """Tabulated neutrino-nucleon DIS cross-section provider."""

    def __init__(self, model: str):
        if model not in SIGMA_TABLE_PATHS:
            raise ValueError(f"unsupported DIS model: {model}")
        self.model = model
        self.table_path = SIGMA_TABLE_PATHS[model]
        self.table = self._load_table(self.table_path)
        self.energy_min_gev = self.table[0][0]
        self.energy_max_gev = self.table[-1][0]

    @staticmethod
    def _load_table(path: Path) -> list[tuple[float, float]]:
        if not path.exists():
            raise FileNotFoundError(f"DIS sigma table not found: {path}")
        table: list[tuple[float, float]] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            energy_gev = float(parts[0])
            sigma_cm2 = float(parts[2])
            if energy_gev > 0.0 and sigma_cm2 > 0.0:
                table.append((energy_gev, sigma_cm2))
        if len(table) < 2:
            raise ValueError(f"DIS sigma table must contain at least two valid rows: {path}")
        for previous, current in zip(table, table[1:]):
            if current[0] <= previous[0]:
                raise ValueError(f"DIS sigma table energy grid must be strictly increasing: {path}")
        return table

    def sigma_cm2(self, energy_gev: float) -> float:
        if not math.isfinite(energy_gev) or energy_gev <= 0.0:
            raise ValueError("Interpolation energy must be positive.")
        if energy_gev < self.energy_min_gev or energy_gev > self.energy_max_gev:
            raise ValueError("Requested energy outside sigma table range.")
        energy = energy_gev
        for (e0, s0), (e1, s1) in zip(self.table, self.table[1:]):
            if e0 <= energy <= e1:
                t = (math.log(energy) - math.log(e0)) / (math.log(e1) - math.log(e0))
                sigma = math.exp(math.log(s0) + t * (math.log(s1) - math.log(s0)))
                return sigma
        return self.table[-1][1]


def dis_config_from_values(values: dict[str, dict[str, Any]]) -> DISConfig:
    dis = values["dis_interaction_sampler"]
    return DISConfig(
        medium_model=str(dis["medium_model"]),
        medium_velocity_model=str(dis["medium_velocity_model"]),
        density_floor_g_cm3=float(dis["density_floor_g_cm3"]),
        dis_model=str(dis["dis_model"]),
        interaction_sampling_mode=str(dis["interaction_sampling_mode"]),
        max_interactions=int(float(dis["max_interactions"])),
        random_seed=int(float(dis["random_seed"])),
        mass_msun=float(values["black_hole"]["mass_msun"]),
        spin_a=float(values["black_hole"]["spin_a"]),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def rg_to_cm(mass_msun: float) -> float:
    return G_CGS * mass_msun * MSUN_G / (C_CGS * C_CGS)


def analytic_torus_density_g_cm3(r_rg: float, theta_rad: float, values: dict[str, dict[str, Any]], *, density_floor_g_cm3: float = 0.0) -> float:
    return shared_analytic_torus_density_g_cm3(r_rg, theta_rad, values, density_floor_g_cm3=density_floor_g_cm3)


def zamo_or_static_local_energy_gev(segment: dict[str, Any], spin_a: float, medium_velocity_model: str) -> tuple[float, bool]:
    r = float(segment["r_mid_rg"])
    theta = float(segment["theta_mid_rad"])
    p_t = float(segment["p_t_mid"])
    p_phi = float(segment["p_phi_mid"])
    metric = kerr_covariant_metric_components(r, theta, spin_a)
    if medium_velocity_model == "static" and metric["gtt"] < 0.0:
        u_t = 1.0 / math.sqrt(-metric["gtt"])
        energy = -(p_t * u_t)
        return max(0.0, energy), False
    sigma = metric["sigma"]
    delta = metric["delta"]
    big_a = metric["A"]
    lapse = math.sqrt(max(sigma * delta / big_a, 1.0e-30))
    omega = 2.0 * spin_a * r / big_a
    u_t = 1.0 / lapse
    u_phi = omega / lapse
    energy = -(p_t * u_t + p_phi * u_phi)
    return max(0.0, energy), medium_velocity_model == "static"


def interaction_probability(tau_total: float) -> float:
    return max(0.0, min(1.0, -math.expm1(-max(0.0, tau_total))))


def constant_density_tau(n_baryon_cm3: float, sigma_cm2: float, length_cm: float) -> float:
    return n_baryon_cm3 * sigma_cm2 * length_cm


def _source_by_id(source_samples: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(sample["source_sample_id"]): sample for sample in source_samples}


def _group_segments(segments: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        grouped[str(segment["event_id"])].append(segment)
    for event_segments in grouped.values():
        event_segments.sort(key=lambda item: int(item["segment_index"]))
    return grouped


def _linear_angle(phi0: float, phi1: float, s: float) -> float:
    delta = math.atan2(math.sin(phi1 - phi0), math.cos(phi1 - phi0))
    return phi0 + s * delta


def sample_interaction_point_in_medium(
    segment: dict[str, Any],
    values: dict[str, dict[str, Any]],
    *,
    density_floor_g_cm3: float,
    rng: random.Random,
    max_attempts: int = 32,
) -> dict[str, Any]:
    best = {"rho": 0.0, "r": float(segment["r_mid_rg"]), "theta": float(segment["theta_mid_rad"]), "phi": float(segment["phi_mid_rad"])}
    for attempt in range(1, max_attempts + 1):
        s = rng.random()
        r = float(segment["r_start_rg"]) + s * (float(segment["r_end_rg"]) - float(segment["r_start_rg"]))
        theta = float(segment["theta_start_rad"]) + s * (float(segment["theta_end_rad"]) - float(segment["theta_start_rad"]))
        phi = _linear_angle(float(segment["phi_start_rad"]), float(segment["phi_end_rad"]), s)
        rho = analytic_torus_density_g_cm3(r, theta, values, density_floor_g_cm3=density_floor_g_cm3)
        if rho > float(best["rho"]):
            best = {"rho": rho, "r": r, "theta": theta, "phi": phi}
        if rho > 0.0:
            return {
                "r": r,
                "theta": theta,
                "phi": phi,
                "rho": rho,
                "inside": True,
                "attempts": attempt,
                "method": "rejection_with_midpoint_fallback",
            }
    midpoint_rho = analytic_torus_density_g_cm3(
        float(segment["r_mid_rg"]),
        float(segment["theta_mid_rad"]),
        values,
        density_floor_g_cm3=density_floor_g_cm3,
    )
    if midpoint_rho > 0.0:
        return {
            "r": float(segment["r_mid_rg"]),
            "theta": float(segment["theta_mid_rad"]),
            "phi": float(segment["phi_mid_rad"]),
            "rho": midpoint_rho,
            "inside": True,
            "attempts": max_attempts,
            "method": "rejection_with_midpoint_fallback_midpoint",
        }
    return {
        "r": float(best["r"]),
        "theta": float(best["theta"]),
        "phi": float(best["phi"]),
        "rho": float(best["rho"]),
        "inside": float(best["rho"]) > 0.0,
        "attempts": max_attempts,
        "method": "rejection_with_highest_density_fallback",
    }


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    keys = [
        "status",
        "n_paths_processed",
        "n_segments_processed",
        "tau_min",
        "tau_mean",
        "tau_max",
        "n_interactions_accepted",
        "acceptance_fraction",
        "max_density_g_cm3",
        "max_sigma_cm2",
        "max_d_tau",
        "dis_model",
        "medium_model",
        "medium_velocity_model",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "value"])
        for key in keys:
            writer.writerow([key, summary.get(key)])


def draw_tau_preview(path_records: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taus = [float(record["tau_nuN_total"]) for record in path_records]
    probabilities = [float(record["interaction_probability"]) for record in path_records]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), facecolor="#f8fafc")
    axes[0].hist(taus, bins=min(24, max(6, len(taus))), color="#2563eb", alpha=0.82)
    axes[0].set_title(r"$\tau_{\nu N}$ per forward path")
    axes[0].set_xlabel(r"$\tau_{\nu N}$")
    axes[0].set_ylabel("count")
    axes[1].hist(probabilities, bins=min(24, max(6, len(probabilities))), color="#7c2d12", alpha=0.82)
    axes[1].set_title(r"$P_{\rm int}=1-\exp(-\tau_{\nu N})$")
    axes[1].set_xlabel(r"$P_{\rm int}$")
    axes[1].set_ylabel("count")
    for ax in axes:
        ax.grid(True, color="#cbd5e1", alpha=0.55, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=165)
    plt.close(fig)


def _xyz(r: float, theta: float, phi: float) -> tuple[float, float, float]:
    sin_theta = math.sin(theta)
    return (
        r * sin_theta * math.cos(phi),
        r * sin_theta * math.sin(phi),
        r * math.cos(theta),
    )


def _rz(r: float, theta: float) -> tuple[float, float]:
    return r * math.sin(theta), r * math.cos(theta)


def draw_interaction_locations(accepted: list[dict[str, Any]], segments: list[dict[str, Any]], values: dict[str, dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 7.0), facecolor="#f8fafc")
    config = dis_config_from_values(values)
    torus = values["analytic_torus"]
    r_inner = float(torus["r_inner_rg"])
    r_outer = float(torus["r_outer_rg"])
    limit = max(float(values["forward_geodesics"]["outer_radius_rg"]) * 0.35, r_outer * 1.25, 5.0)
    MediumRenderer.draw_density_map_Rz(
        ax,
        values,
        limit_rg=limit,
        density_floor_g_cm3=config.density_floor_g_cm3,
        n_r=220,
        n_z=260,
        add_colorbar_to=fig,
    )
    for segment in segments[:: max(1, len(segments) // 900)]:
        r0, z0 = _rz(float(segment["r_start_rg"]), float(segment["theta_start_rad"]))
        r1, z1 = _rz(float(segment["r_end_rg"]), float(segment["theta_end_rad"]))
        ax.plot([r0, r1], [z0, z1], color="#93c5fd", alpha=0.18, linewidth=0.55, zorder=3)
    cone = values["polar_cone"]
    if bool(cone["enabled"]):
        opening = math.radians(float(cone["opening_angle_deg"]))
        r_min = float(cone["r_min_rg"])
        r_max = float(cone["r_max_rg"])
        signs = (1.0, -1.0) if str(cone["draw_mode"]) == "bipolar_funnel" else (1.0,)
        for sign in signs:
            polygon = [
                (-r_min * math.sin(opening), sign * r_min * math.cos(opening)),
                (-r_max * math.sin(opening), sign * r_max * math.cos(opening)),
                (r_max * math.sin(opening), sign * r_max * math.cos(opening)),
                (r_min * math.sin(opening), sign * r_min * math.cos(opening)),
            ]
            ax.add_patch(plt.Polygon(polygon, closed=True, facecolor="#60a5fa", edgecolor="#bfdbfe", alpha=0.10, linewidth=1.0, zorder=1))
    ax.add_patch(plt.Circle((0.0, 0.0), 1.0, color="black", alpha=0.95))
    if accepted:
        xs, zs, colors = [], [], []
        for record in accepted:
            r_meridional, z = _rz(float(record["interaction_r_rg"]), float(record["interaction_theta_rad"]))
            xs.append(r_meridional)
            zs.append(z)
            colors.append(float(record.get("interaction_point_rho_g_cm3", record["interaction_rho_g_cm3"])))
        scatter = ax.scatter(xs, zs, c=colors, s=28, cmap="viridis", edgecolors="black", linewidths=0.25, zorder=5)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.82)
        cbar.set_label(r"accepted point $\rho$ [g cm$^{-3}$]")
    ax.set_title(f"DIS interaction locations on MediumRenderer density field\naccepted={len(accepted)}")
    ax.set_xlabel(r"$R=r\sin\theta$ [$r_g$]")
    ax.set_ylabel(r"$z$ [$r_g$]")
    ax.set_xlim(0.0, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#cbd5e1", alpha=0.45, linewidth=0.55)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_interaction_locations_html(accepted: list[dict[str, Any]], values: dict[str, dict[str, Any]], output_path: Path) -> None:
    points = [
        {
            "x": _xyz(float(row["interaction_r_rg"]), float(row["interaction_theta_rad"]), float(row["interaction_phi_rad"]))[0],
            "y": _xyz(float(row["interaction_r_rg"]), float(row["interaction_theta_rad"]), float(row["interaction_phi_rad"]))[1],
            "z": _xyz(float(row["interaction_r_rg"]), float(row["interaction_theta_rad"]), float(row["interaction_phi_rad"]))[2],
            "energy": float(row["interaction_E_nu_local_gev"]),
            "rho": float(row.get("interaction_point_rho_g_cm3", row["interaction_rho_g_cm3"])),
            "inside": bool(row.get("interaction_point_inside_medium", False)),
        }
        for row in accepted
    ]
    payload = json.dumps(points)
    torus = values["analytic_torus"]
    cone = values["polar_cone"]
    geometry = {
        "torus": {
            "rInner": float(torus["r_inner_rg"]),
            "rOuter": float(torus["r_outer_rg"]),
            "rPeak": float(torus["r_peak_rg"]),
            "halfOpeningRad": math.radians(float(torus["half_opening_angle_deg"])),
            "thetaProfile": "gaussian",
            "thetaIsHardCut": False,
            "hardRadialCut": True,
        },
        "cone": {
            "enabled": bool(cone["enabled"]),
            "openingRad": math.radians(float(cone["opening_angle_deg"])),
            "rMin": float(cone["r_min_rg"]),
            "rMax": float(cone["r_max_rg"]),
            "bipolar": str(cone["draw_mode"]) == "bipolar_funnel",
        },
    }
    geometry_payload = json.dumps(geometry)
    output_path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>HADROS3 DIS Interaction Locations</title>
<style>body{{font-family:system-ui,sans-serif;margin:0;background:#f8fafc;color:#172033}}canvas{{display:block;width:100vw;height:82vh;background:#101318;cursor:grab}}.note{{padding:12px 16px}}</style></head>
<body><canvas id="canvas" width="1200" height="760"></canvas><div class="note">DIS interaction locations in the HADROS3 coordinate frame. The analytic torus has a hard radial cut and a Gaussian angular profile; the opening angle is a width parameter, not a hard boundary. Drag to rotate, wheel to zoom. Accepted interactions: {len(points)}</div>
<script>
const points = {payload};
const geometry = {geometry_payload};
const c = document.getElementById("canvas"), ctx = c.getContext("2d");
let yaw = 0.72, pitch = 0.38, zoom = 1.0, dragging = false, lastX = 0, lastY = 0;
function rotate(p) {{
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * p.x - sy * p.y;
  const y1 = sy * p.x + cy * p.y;
  const z1 = p.z;
  return {{x: x1, y: cp * y1 - sp * z1, z: sp * y1 + cp * z1, energy: p.energy, rho: p.rho, inside: p.inside}};
}}
function draw() {{
  ctx.clearRect(0,0,c.width,c.height);
  ctx.fillStyle = "#101318"; ctx.fillRect(0,0,c.width,c.height);
  const rotated = points.map(rotate).sort((a,b) => a.y - b.y);
  const geometryLimit = Math.max(geometry.torus.rOuter, geometry.cone.rMax || 0);
  const pointLimit = points.length ? Math.max(...points.flatMap(p => [Math.abs(p.x), Math.abs(p.y), Math.abs(p.z)])) : 0;
  const lim = Math.max(5, pointLimit, geometryLimit) * 1.25 / zoom;
  const sx = x => c.width * (0.5 + 0.42 * x / lim);
  const sy = z => c.height * (0.5 - 0.42 * z / lim);
  const project = p => {{
    const rp = rotate({{x: p.x, y: p.y, z: p.z, energy: 0}});
    return {{x: sx(rp.x), y: sy(rp.z), depth: rp.y}};
  }};
  const polyline = pts => {{
    if (!pts.length) return;
    const first = project(pts[0]);
    ctx.beginPath(); ctx.moveTo(first.x, first.y);
    for (const p of pts.slice(1)) {{
      const q = project(p);
      ctx.lineTo(q.x, q.y);
    }}
    ctx.stroke();
  }};
  const ring = (radius, z = 0) => Array.from({{length: 145}}, (_, i) => {{
    const a = 2 * Math.PI * i / 144;
    return {{x: radius * Math.cos(a), y: radius * Math.sin(a), z}};
  }});
  const latitudeRing = (radius, theta) => Array.from({{length: 145}}, (_, i) => {{
    const phi = 2 * Math.PI * i / 144;
    const R = radius * Math.sin(theta);
    return {{x: R * Math.cos(phi), y: R * Math.sin(phi), z: radius * Math.cos(theta)}};
  }});
  const meridian = (radius, phi) => Array.from({{length: 97}}, (_, i) => {{
    const theta = Math.PI * i / 96;
    return {{x: radius * Math.sin(theta) * Math.cos(phi), y: radius * Math.sin(theta) * Math.sin(phi), z: radius * Math.cos(theta)}};
  }});
  const coneEdge = (sign, side) => {{
    const a = geometry.cone.openingRad;
    const s = Math.sin(a) * side;
    const c0 = Math.cos(a) * sign;
    return [
      {{x: geometry.cone.rMin * s, y: 0, z: geometry.cone.rMin * c0}},
      {{x: geometry.cone.rMax * s, y: 0, z: geometry.cone.rMax * c0}},
    ];
  }};
  ctx.strokeStyle = "#334155"; ctx.lineWidth = 1;
  for (let i=-4;i<=4;i++) {{
    ctx.beginPath(); ctx.moveTo(sx(-lim), sy(i*lim/4)); ctx.lineTo(sx(lim), sy(i*lim/4)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(sx(i*lim/4), sy(-lim)); ctx.lineTo(sx(i*lim/4), sy(lim)); ctx.stroke();
  }}
  ctx.lineWidth = 2;
  ctx.setLineDash([]);
  ctx.strokeStyle = "rgba(56,189,248,0.82)";
  for (const radius of [geometry.torus.rInner, geometry.torus.rOuter]) {{
    polyline(ring(radius, 0));
    for (let i = 0; i < 6; i++) polyline(meridian(radius, i * Math.PI / 6));
  }}
  ctx.lineWidth = 1.2;
  ctx.setLineDash([5, 5]);
  const sigma = geometry.torus.halfOpeningRad;
  const angularLevels = [
    {{offset: 0, alpha: 0.76, label: "peak"}},
    {{offset: 1, alpha: 0.50, label: "1 sigma"}},
    {{offset: 2, alpha: 0.28, label: "2 sigma"}},
  ];
  for (const level of angularLevels) {{
    const offsets = level.offset === 0 ? [0] : [-level.offset, level.offset];
    ctx.strokeStyle = `rgba(251,191,36,${{level.alpha}})`;
    for (const offset of offsets) {{
      const theta = Math.PI / 2 + offset * sigma;
      if (theta > 0.02 && theta < Math.PI - 0.02) {{
        polyline(latitudeRing(geometry.torus.rInner, theta));
        polyline(latitudeRing(geometry.torus.rPeak, theta));
        polyline(latitudeRing(geometry.torus.rOuter, theta));
      }}
    }}
  }}
  ctx.setLineDash([]);
  if (geometry.cone.enabled) {{
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(96,165,250,0.82)";
    polyline(coneEdge(1, 1)); polyline(coneEdge(1, -1));
    if (geometry.cone.bipolar) {{ polyline(coneEdge(-1, 1)); polyline(coneEdge(-1, -1)); }}
  }}
  ctx.fillStyle = "black"; ctx.beginPath(); ctx.arc(sx(0), sy(0), Math.max(4, 0.55*c.width/lim), 0, 2*Math.PI); ctx.fill();
  const maxRho = Math.max(1e-300, ...points.map(p => p.rho || 0));
  for (const p of rotated) {{
    const depth = 0.55 + 0.45 * Math.max(0, Math.min(1, (p.y + lim) / (2*lim)));
    const rhoLevel = Math.max(0.25, Math.min(1, Math.log10(Math.max(p.rho || 0, 1e-300)) / Math.log10(maxRho)));
    ctx.fillStyle = p.inside ? `rgba(34,197,94,${{depth.toFixed(3)}})` : "rgba(239,68,68,0.92)";
    ctx.beginPath(); ctx.arc(sx(p.x), sy(p.z), 4 + 3 * depth, 0, 2*Math.PI); ctx.fill();
    ctx.strokeStyle = p.inside ? `rgba(240,253,244,${{rhoLevel.toFixed(3)}})` : "rgba(127,29,29,0.95)";
    ctx.stroke();
  }}
  ctx.fillStyle="#e5e7eb"; ctx.font="20px system-ui"; ctx.fillText("HADROS3 DIS Interaction Locations", 18, 32);
  ctx.font="14px system-ui"; ctx.fillText(`yaw=${{yaw.toFixed(2)}} pitch=${{pitch.toFixed(2)}} zoom=${{zoom.toFixed(2)}}`, 18, 54);
  ctx.fillText(`medium: hard radial shell Rin=${{geometry.torus.rInner.toFixed(1)}} Rout=${{geometry.torus.rOuter.toFixed(1)}} rg; angular profile=Gaussian`, 18, 76);
  ctx.fillText(`accepted inside medium=${{points.filter(p => p.inside && p.rho > 0).length}}/${{points.length}}; angular rings are density levels, not boundaries`, 18, 98);
}}
c.addEventListener("mousedown", e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; }});
window.addEventListener("mouseup", () => dragging = false);
window.addEventListener("mousemove", e => {{
  if (!dragging) return;
  yaw += (e.clientX - lastX) * 0.008;
  pitch = Math.max(-1.45, Math.min(1.45, pitch + (e.clientY - lastY) * 0.008));
  lastX = e.clientX; lastY = e.clientY; draw();
}});
c.addEventListener("wheel", e => {{ e.preventDefault(); zoom = Math.max(0.2, Math.min(8, zoom * Math.exp(-e.deltaY * 0.001))); draw(); }}, {{passive:false}});
draw();
</script></body></html>
""",
        encoding="utf-8",
    )


def _finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _stats(values: list[float]) -> dict[str, float]:
    clean = _finite(values)
    if not clean:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0, "max": 0.0}
    return {
        "min": min(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "std": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
        "max": max(clean),
    }


def _hist_bins(values: list[float]) -> int:
    return min(36, max(8, int(math.sqrt(max(1, len(values))))))


def _draw_meridional_geometry(ax: Any, values: dict[str, dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    torus = values["analytic_torus"]
    r_inner = float(torus["r_inner_rg"])
    r_outer = float(torus["r_outer_rg"])
    half_angle = math.radians(float(torus["half_opening_angle_deg"]))
    ax.add_patch(plt.Circle((0.0, 0.0), r_inner, fill=False, edgecolor="#38bdf8", linestyle="--", linewidth=1.0, alpha=0.82, zorder=2))
    ax.add_patch(plt.Circle((0.0, 0.0), r_outer, fill=False, edgecolor="#38bdf8", linestyle="--", linewidth=1.0, alpha=0.82, zorder=2))
    ray_length = r_outer * 1.15
    for sign in (-1.0, 1.0):
        theta = 0.5 * math.pi + sign * half_angle
        for x_sign in (-1.0, 1.0):
            ax.plot(
                [0.0, x_sign * ray_length * math.sin(theta)],
                [0.0, ray_length * math.cos(theta)],
                color="#fde047",
                linestyle=":",
                linewidth=0.8,
                alpha=0.68,
                zorder=2,
            )
    cone = values["polar_cone"]
    if bool(cone["enabled"]):
        opening = math.radians(float(cone["opening_angle_deg"]))
        r_min = float(cone["r_min_rg"])
        r_max = float(cone["r_max_rg"])
        signs = (1.0, -1.0) if str(cone["draw_mode"]) == "bipolar_funnel" else (1.0,)
        for sign in signs:
            polygon = [
                (-r_min * math.sin(opening), sign * r_min * math.cos(opening)),
                (-r_max * math.sin(opening), sign * r_max * math.cos(opening)),
                (r_max * math.sin(opening), sign * r_max * math.cos(opening)),
                (r_min * math.sin(opening), sign * r_min * math.cos(opening)),
            ]
            ax.add_patch(plt.Polygon(polygon, closed=True, facecolor="#60a5fa", edgecolor="#bfdbfe", alpha=0.09, linewidth=0.9, zorder=1))
    ax.add_patch(plt.Circle((0.0, 0.0), 1.0, color="black", alpha=0.95, zorder=4))


def _segment_diagnostics(values: dict[str, dict[str, Any]], segments: list[dict[str, Any]]) -> list[dict[str, float]]:
    config = dis_config_from_values(values)
    provider = SigmaNuNProvider(config.dis_model)
    r_g_cm = rg_to_cm(config.mass_msun)
    rows: list[dict[str, float]] = []
    for segment in segments:
        rho = analytic_torus_density_g_cm3(
            float(segment["r_mid_rg"]),
            float(segment["theta_mid_rad"]),
            values,
            density_floor_g_cm3=config.density_floor_g_cm3,
        )
        n_baryon = rho / M_BARYON_G
        e_local, _ = zamo_or_static_local_energy_gev(segment, config.spin_a, config.medium_velocity_model)
        try:
            sigma = provider.sigma_cm2(e_local)
        except ValueError:
            sigma = 0.0
        d_tau = max(0.0, n_baryon * sigma * float(segment["dl_segment_rg"]) * r_g_cm)
        x0, _, z0 = _xyz(float(segment["r_start_rg"]), float(segment["theta_start_rad"]), float(segment["phi_start_rad"]))
        x1, _, z1 = _xyz(float(segment["r_end_rg"]), float(segment["theta_end_rad"]), float(segment["phi_end_rad"]))
        xm, ym, zm = _xyz(float(segment["r_mid_rg"]), float(segment["theta_mid_rad"]), float(segment["phi_mid_rad"]))
        rows.append(
            {
                "x0": x0,
                "z0": z0,
                "x1": x1,
                "z1": z1,
                "xm": xm,
                "ym": ym,
                "zm": zm,
                "rho_g_cm3": rho,
                "E_nu_local_gev": e_local,
                "sigma_nuN_cm2": sigma,
                "d_tau": d_tau,
            }
        )
    return rows


def _plot_histogram(values: list[float], output_path: Path, *, title: str, xlabel: str, color: str = "#2563eb", logx: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = _finite(values)
    fig, ax = plt.subplots(figsize=(7.4, 4.8), facecolor="#f8fafc")
    if clean:
        plot_values = clean
        if logx:
            plot_values = [value for value in clean if value > 0.0]
            if plot_values:
                ax.set_xscale("log")
        ax.hist(plot_values, bins=_hist_bins(plot_values), color=color, alpha=0.82)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, color="#cbd5e1", alpha=0.55, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=165)
    plt.close(fig)


def draw_optical_depth_map(segment_rows: list[dict[str, float]], values: dict[str, dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(8.0, 7.2), facecolor="#f8fafc")
    _draw_meridional_geometry(ax, values)
    lines = [[(row["x0"], row["z0"]), (row["x1"], row["z1"])] for row in segment_rows]
    d_tau = [row["d_tau"] for row in segment_rows]
    positive = [value for value in d_tau if value > 0.0]
    if lines:
        if positive:
            norm = LogNorm(vmin=max(min(positive), 1.0e-300), vmax=max(positive))
            collection = LineCollection(lines, cmap="magma", norm=norm, linewidths=0.75, alpha=0.86, zorder=3)
            collection.set_array([max(value, norm.vmin) for value in d_tau])
        else:
            collection = LineCollection(lines, colors="#93c5fd", linewidths=0.6, alpha=0.55, zorder=3)
        ax.add_collection(collection)
        if positive:
            cbar = fig.colorbar(collection, ax=ax, shrink=0.82)
            cbar.set_label(r"$d\tau_{\nu N}$ per segment")
    limit = max(float(values["forward_geodesics"]["outer_radius_rg"]) * 0.35, float(values["polar_cone"]["r_max_rg"]) * 0.7, 5.0)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("DIS optical-depth accumulation along geodesics")
    ax.set_xlabel(r"$x$ [$r_g$]")
    ax.set_ylabel(r"$z$ [$r_g$]")
    ax.grid(True, color="#cbd5e1", alpha=0.45, linewidth=0.55)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_optical_depth_map_html(segment_rows: list[dict[str, float]], values: dict[str, dict[str, Any]], output_path: Path) -> None:
    stride = max(1, len(segment_rows) // 2500)
    rows = segment_rows[::stride]
    payload = json.dumps(rows)
    torus = values["analytic_torus"]
    cone = values["polar_cone"]
    geometry = {
        "torus": {
            "rInner": float(torus["r_inner_rg"]),
            "rOuter": float(torus["r_outer_rg"]),
            "rPeak": float(torus["r_peak_rg"]),
            "halfOpeningRad": math.radians(float(torus["half_opening_angle_deg"])),
        },
        "cone": {
            "enabled": bool(cone["enabled"]),
            "openingRad": math.radians(float(cone["opening_angle_deg"])),
            "rMin": float(cone["r_min_rg"]),
            "rMax": float(cone["r_max_rg"]),
            "bipolar": str(cone["draw_mode"]) == "bipolar_funnel",
        },
    }
    output_path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>HADROS3 DIS Optical Depth Map</title>
<style>body{{font-family:system-ui,sans-serif;margin:0;background:#f8fafc;color:#172033}}canvas{{display:block;width:100vw;height:82vh;background:#101318;cursor:grab}}.note{{padding:12px 16px}}</style></head>
<body><canvas id="canvas" width="1200" height="760"></canvas><div class="note">DIS optical depth map. Segment color scales with d_tau. Drag to rotate, wheel to zoom.</div>
<script>
const rows = {payload};
const geometry = {json.dumps(geometry)};
const c = document.getElementById("canvas"), ctx = c.getContext("2d");
let yaw = 0.72, pitch = 0.38, zoom = 1.0, dragging = false, lastX = 0, lastY = 0;
function rotate(p) {{
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * p.x - sy * p.y;
  const y1 = sy * p.x + cy * p.y;
  return {{x: x1, y: cp * y1 - sp * p.z, z: sp * y1 + cp * p.z}};
}}
function color(t) {{
  const u = Math.max(0, Math.min(1, t));
  const r = Math.round(60 + 195*u), g = Math.round(80 + 80*Math.sin(Math.PI*u)), b = Math.round(180 - 150*u);
  return `rgba(${{r}},${{g}},${{b}},0.82)`;
}}
function draw() {{
  ctx.fillStyle="#101318"; ctx.fillRect(0,0,c.width,c.height);
  const maxDtau = Math.max(1e-300, ...rows.map(r => r.d_tau || 0));
  const geometryLimit = Math.max(geometry.torus.rOuter, geometry.cone.rMax || 0);
  const pointLimit = rows.length ? Math.max(...rows.flatMap(r => [Math.abs(r.xm), Math.abs(r.ym), Math.abs(r.zm)])) : 0;
  const lim = Math.max(5, pointLimit, geometryLimit) * 1.25 / zoom;
  const sx = x => c.width * (0.5 + 0.42 * x / lim), sy = z => c.height * (0.5 - 0.42 * z / lim);
  ctx.strokeStyle="#334155"; ctx.lineWidth=1;
  for (let i=-4;i<=4;i++) {{ ctx.beginPath(); ctx.moveTo(sx(-lim), sy(i*lim/4)); ctx.lineTo(sx(lim), sy(i*lim/4)); ctx.stroke(); ctx.beginPath(); ctx.moveTo(sx(i*lim/4), sy(-lim)); ctx.lineTo(sx(i*lim/4), sy(lim)); ctx.stroke(); }}
  const ordered = rows.map(r => {{ const m = rotate({{x:r.xm,y:r.ym,z:r.zm}}); return {{...r, depth:m.y}}; }}).sort((a,b)=>a.depth-b.depth);
  for (const r of ordered) {{
    const a = rotate({{x:r.x0,y:0,z:r.z0}}), b = rotate({{x:r.x1,y:0,z:r.z1}});
    const t = Math.log10(Math.max(r.d_tau, 1e-300)) / Math.log10(maxDtau);
    ctx.strokeStyle = color(Number.isFinite(t) ? t : 0);
    ctx.lineWidth = 1.1;
    ctx.beginPath(); ctx.moveTo(sx(a.x), sy(a.z)); ctx.lineTo(sx(b.x), sy(b.z)); ctx.stroke();
  }}
  ctx.fillStyle="black"; ctx.beginPath(); ctx.arc(sx(0), sy(0), Math.max(4, 0.55*c.width/lim), 0, 2*Math.PI); ctx.fill();
  ctx.fillStyle="#e5e7eb"; ctx.font="20px system-ui"; ctx.fillText("HADROS3 DIS Optical Depth Map", 18, 32);
  ctx.font="14px system-ui"; ctx.fillText(`segments=${{rows.length}} max d_tau=${{maxDtau.toExponential(3)}}`, 18, 54);
}}
c.addEventListener("mousedown", e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; }});
window.addEventListener("mouseup", () => dragging = false);
window.addEventListener("mousemove", e => {{ if (!dragging) return; yaw += (e.clientX-lastX)*0.008; pitch = Math.max(-1.45, Math.min(1.45, pitch+(e.clientY-lastY)*0.008)); lastX=e.clientX; lastY=e.clientY; draw(); }});
c.addEventListener("wheel", e => {{ e.preventDefault(); zoom = Math.max(0.2, Math.min(8, zoom * Math.exp(-e.deltaY * 0.001))); draw(); }}, {{passive:false}});
draw();
</script></body></html>
""",
        encoding="utf-8",
    )


def draw_correlation(segment_rows: list[dict[str, float]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in segment_rows if row["rho_g_cm3"] > 0.0 and row["E_nu_local_gev"] > 0.0 and row["sigma_nuN_cm2"] > 0.0]
    fig, ax = plt.subplots(figsize=(7.4, 5.8), facecolor="#f8fafc")
    if rows:
        stride = max(1, len(rows) // 6000)
        rows = rows[::stride]
        scatter = ax.scatter(
            [row["rho_g_cm3"] for row in rows],
            [row["E_nu_local_gev"] for row in rows],
            c=[row["sigma_nuN_cm2"] for row in rows],
            s=8,
            cmap="plasma",
            alpha=0.62,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.82)
        cbar.set_label(r"$\sigma_{\nu N}$ [cm$^2$]")
    ax.set_title(r"DIS segment correlation: $\rho$, $E_{\rm local}$, $\sigma_{\nu N}$")
    ax.set_xlabel(r"$\rho$ [g cm$^{-3}$]")
    ax.set_ylabel(r"$E_{\nu,\rm local}$ [GeV]")
    ax.grid(True, color="#cbd5e1", alpha=0.45, linewidth=0.55)
    fig.tight_layout()
    fig.savefig(output_path, dpi=165)
    plt.close(fig)


def draw_medium_density_map(
    accepted: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    values: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = dis_config_from_values(values)
    torus = values["analytic_torus"]
    r_outer = float(torus["r_outer_rg"])
    limit = max(r_outer * 1.35, 5.0)

    fig, ax = plt.subplots(figsize=(8.6, 7.8), facecolor="#f8fafc")
    MediumRenderer.draw_density_map_Rz(
        ax,
        values,
        limit_rg=limit,
        density_floor_g_cm3=config.density_floor_g_cm3,
        n_r=280,
        n_z=360,
        add_colorbar_to=fig,
    )
    if segments:
        for segment in segments[:: max(1, len(segments) // 1100)]:
            r0 = float(segment["r_start_rg"])
            th0 = float(segment["theta_start_rad"])
            r1 = float(segment["r_end_rg"])
            th1 = float(segment["theta_end_rad"])
            ax.plot(
                [r0 * math.sin(th0), r1 * math.sin(th1)],
                [r0 * math.cos(th0), r1 * math.cos(th1)],
                color="#93c5fd",
                alpha=0.13,
                linewidth=0.45,
            )
    if accepted:
        scatter = ax.scatter(
            [float(row["interaction_r_rg"]) * math.sin(float(row["interaction_theta_rad"])) for row in accepted],
            [float(row["interaction_r_rg"]) * math.cos(float(row["interaction_theta_rad"])) for row in accepted],
            c=[float(row.get("interaction_point_rho_g_cm3", row["interaction_rho_g_cm3"])) for row in accepted],
            s=30,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.35,
            zorder=5,
        )
        fig.colorbar(scatter, ax=ax, shrink=0.82, label=r"accepted point $\rho$ [g cm$^{-3}$]")
    ax.set_title("DIS medium density map: hard radial shell, Gaussian angular profile")
    ax.set_xlabel(r"$R=r\sin\theta$ [$r_g$]")
    ax.set_ylabel(r"$z=r\cos\theta$ [$r_g$]")
    ax.set_xlim(0.0, limit)
    ax.set_ylim(-limit, limit)
    ax.grid(True, color="#cbd5e1", alpha=0.35, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def generate_dis_diagnostics(values: dict[str, dict[str, Any]], *, run_output_dir: Path, summary: dict[str, Any] | None = None) -> dict[str, Any]:
    output_dir = dis_dir(run_output_dir)
    path_records = read_jsonl(output_dir / "dis_path_optical_depths.jsonl")
    accepted = read_jsonl(output_dir / "dis_accepted_interactions.jsonl")
    segments = read_jsonl(forward_geodesics_dir(run_output_dir) / "uhe_neutrino_forward_path_segments.jsonl")
    segment_rows = _segment_diagnostics(values, segments)

    paths = {
        "tau_distribution": output_dir / "tau_distribution.png",
        "interaction_probability_distribution": output_dir / "interaction_probability_distribution.png",
        "optical_depth_map": output_dir / "optical_depth_map.png",
        "optical_depth_map_3d_html": output_dir / "optical_depth_map_3d.html",
        "medium_density_map": output_dir / "medium_density_map.png",
        "interaction_location_distribution": output_dir / "interaction_location_distribution.png",
        "local_energy_distribution": output_dir / "local_energy_distribution.png",
        "local_density_distribution": output_dir / "local_density_distribution.png",
        "sigma_distribution": output_dir / "sigma_distribution.png",
        "density_energy_sigma_correlation": output_dir / "density_energy_sigma_correlation.png",
        "dis_diagnostics_report": output_dir / "dis_diagnostics_report.json",
    }

    tau_values = [float(record["tau_nuN_total"]) for record in path_records]
    probability_values = [float(record["interaction_probability"]) for record in path_records]
    accepted_energy = [float(record["interaction_E_nu_local_gev"]) for record in accepted]
    accepted_density = [float(record["interaction_rho_g_cm3"]) for record in accepted]
    accepted_sigma = [float(record["interaction_sigma_nuN_cm2"]) for record in accepted]
    segment_sigma = [row["sigma_nuN_cm2"] for row in segment_rows if row["sigma_nuN_cm2"] > 0.0]

    _plot_histogram(tau_values, paths["tau_distribution"], title=r"DIS $\tau_{\nu N}$ distribution", xlabel=r"$\tau_{\nu N,total}$", color="#2563eb", logx=any(value > 0.0 for value in tau_values))
    _plot_histogram(probability_values, paths["interaction_probability_distribution"], title="DIS interaction probability distribution", xlabel=r"$P_{\rm int}=1-\exp(-\tau)$", color="#7c2d12")
    draw_optical_depth_map(segment_rows, values, paths["optical_depth_map"])
    write_optical_depth_map_html(segment_rows, values, paths["optical_depth_map_3d_html"])
    draw_medium_density_map(accepted, segments, values, paths["medium_density_map"])
    draw_interaction_locations(accepted, segments, values, paths["interaction_location_distribution"])
    _plot_histogram(accepted_energy, paths["local_energy_distribution"], title=r"Accepted interaction $E_{\nu,\rm local}$", xlabel=r"$E_{\nu,\rm local}$ [GeV]", color="#059669", logx=True)
    _plot_histogram(accepted_density, paths["local_density_distribution"], title="Accepted interaction density", xlabel=r"$\rho$ [g cm$^{-3}$]", color="#ea580c", logx=True)
    _plot_histogram(segment_sigma, paths["sigma_distribution"], title=r"Segment $\sigma_{\nu N}$ used in optical-depth calculation", xlabel=r"$\sigma_{\nu N}$ [cm$^2$]", color="#9333ea", logx=True)
    draw_correlation(segment_rows, paths["density_energy_sigma_correlation"])

    tau_stats = _stats(tau_values)
    probability_stats = _stats(probability_values)
    energy_stats = _stats(accepted_energy)
    rho_stats = _stats(accepted_density)
    sigma_stats = _stats(segment_sigma)
    interaction_points_total = len(accepted)
    interaction_points_inside = sum(bool(record.get("interaction_point_inside_medium", False)) for record in accepted)
    interaction_points_outside = interaction_points_total - interaction_points_inside
    report = {
        "diagnostics_generated": True,
        "medium_density_map_generated": True,
        "medium_renderer_used": True,
        **medium_metadata(),
        "density_model_has_hard_radial_cut": True,
        "density_model_theta_profile": "gaussian",
        "density_model_theta_is_hard_cut": False,
        "n_paths": len(path_records),
        "n_segments": len(segments),
        "interaction_points_total": interaction_points_total,
        "interaction_points_inside_medium": interaction_points_inside,
        "interaction_points_outside_medium": interaction_points_outside,
        "interaction_points_outside_medium_fraction": interaction_points_outside / interaction_points_total if interaction_points_total else 0.0,
        "tau_statistics": {
            "tau_min": tau_stats["min"],
            "tau_mean": tau_stats["mean"],
            "tau_median": tau_stats["median"],
            "tau_std": tau_stats["std"],
            "tau_max": tau_stats["max"],
        },
        "probability_statistics": {
            "probability_min": probability_stats["min"],
            "probability_mean": probability_stats["mean"],
            "probability_median": probability_stats["median"],
            "probability_std": probability_stats["std"],
            "probability_max": probability_stats["max"],
        },
        "E_local_statistics": {
            "E_local_min": energy_stats["min"],
            "E_local_mean": energy_stats["mean"],
            "E_local_max": energy_stats["max"],
        },
        "rho_statistics": {
            "rho_min": rho_stats["min"],
            "rho_mean": rho_stats["mean"],
            "rho_max": rho_stats["max"],
        },
        "sigma_statistics": {
            "sigma_min": sigma_stats["min"],
            "sigma_mean": sigma_stats["mean"],
            "sigma_max": sigma_stats["max"],
        },
        "accepted_interactions": len(accepted),
        "accepted_fraction": len(accepted) / len(path_records) if path_records else 0.0,
        "GBW_IIM_comparison": None,
        "products": {key: str(path) for key, path in paths.items()},
    }
    existing_comparison = output_dir / "gbw_vs_iim_summary.json"
    if existing_comparison.exists():
        try:
            report["GBW_IIM_comparison"] = json.loads(existing_comparison.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report["GBW_IIM_comparison"] = {"status": "invalid_summary"}
    write_json(paths["dis_diagnostics_report"], report)
    if summary is not None:
        summary.setdefault("products", {}).update({key: str(path) for key, path in paths.items()})
        summary.update(
            {
                "diagnostics_generated": True,
                "medium_density_map_generated": True,
                "medium_renderer_used": True,
                **medium_metadata(),
                "density_model_has_hard_radial_cut": True,
                "density_model_theta_profile": "gaussian",
                "density_model_theta_is_hard_cut": False,
                "interaction_points_total": interaction_points_total,
                "interaction_points_inside_medium": interaction_points_inside,
                "interaction_points_outside_medium": interaction_points_outside,
                "interaction_points_outside_medium_fraction": interaction_points_outside / interaction_points_total if interaction_points_total else 0.0,
                "tau_median": tau_stats["median"],
                "tau_std": tau_stats["std"],
                "E_local_min": energy_stats["min"],
                "E_local_mean": energy_stats["mean"],
                "E_local_max": energy_stats["max"],
                "rho_min": rho_stats["min"],
                "rho_mean": rho_stats["mean"],
                "rho_max": rho_stats["max"],
                "sigma_min": sigma_stats["min"],
                "sigma_mean": sigma_stats["mean"],
                "sigma_max": sigma_stats["max"],
            }
        )
    return report


def generate_gbw_iim_comparison(values: dict[str, dict[str, Any]], *, run_output_dir: Path) -> dict[str, Any]:
    output_dir = dis_dir(run_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_payloads: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="hadros3_dis_model_compare_") as tmp:
        tmp_root = Path(tmp)
        for model in ["GBW", "IIM"]:
            tmp_run = tmp_root / model
            tmp_run.mkdir(parents=True, exist_ok=True)
            shutil.copytree(uhe_source_dir(run_output_dir), uhe_source_dir(tmp_run))
            shutil.copytree(forward_geodesics_dir(run_output_dir), forward_geodesics_dir(tmp_run))
            model_values = json.loads(json.dumps(values))
            model_values["dis_interaction_sampler"]["dis_model"] = model
            summary = generate_dis_interaction_products(model_values, run_output_dir=tmp_run, include_model_comparison=False)
            paths = read_jsonl(dis_dir(tmp_run) / "dis_path_optical_depths.jsonl")
            accepted = read_jsonl(dis_dir(tmp_run) / "dis_accepted_interactions.jsonl")
            diagnostics = json.loads((dis_dir(tmp_run) / "dis_diagnostics_report.json").read_text(encoding="utf-8"))
            model_payloads[model] = {"summary": summary, "paths": paths, "accepted": accepted, "diagnostics": diagnostics}

    gbw = model_payloads["GBW"]
    iim = model_payloads["IIM"]
    tau_path = output_dir / "gbw_vs_iim_tau_comparison.png"
    prob_path = output_dir / "gbw_vs_iim_probability_comparison.png"
    loc_path = output_dir / "gbw_vs_iim_interaction_locations.png"
    summary_path = output_dir / "gbw_vs_iim_summary.json"

    _plot_model_comparison_hist(
        [float(row["tau_nuN_total"]) for row in gbw["paths"]],
        [float(row["tau_nuN_total"]) for row in iim["paths"]],
        tau_path,
        title=r"GBW vs IIM $\tau_{\nu N}$",
        xlabel=r"$\tau_{\nu N,total}$",
        logx=True,
    )
    _plot_model_comparison_hist(
        [float(row["interaction_probability"]) for row in gbw["paths"]],
        [float(row["interaction_probability"]) for row in iim["paths"]],
        prob_path,
        title="GBW vs IIM interaction probability",
        xlabel=r"$P_{\rm int}$",
        logx=False,
    )
    _plot_model_interactions(gbw["accepted"], iim["accepted"], values, loc_path)

    def row(model: str, payload: dict[str, Any]) -> dict[str, float | int]:
        summary = payload["summary"]
        diagnostics = payload["diagnostics"]
        sigma_stats = diagnostics.get("sigma_statistics", {})
        return {
            "tau_mean": float(summary.get("tau_mean", 0.0)),
            "tau_max": float(summary.get("tau_max", 0.0)),
            "accepted_interactions": int(summary.get("n_interactions_accepted", 0)),
            "accepted_fraction": float(summary.get("acceptance_fraction", 0.0)),
            "sigma_mean": float(sigma_stats.get("sigma_mean", 0.0)),
            "sigma_max": float(sigma_stats.get("sigma_max", 0.0)),
        }

    gbw_row = row("GBW", gbw)
    iim_row = row("IIM", iim)
    comparison = {
        "status": "ok",
        "models": {"GBW": gbw_row, "IIM": iim_row},
        "differences": {
            key: float(gbw_row[key]) - float(iim_row[key])
            for key in ["tau_mean", "tau_max", "accepted_interactions", "accepted_fraction", "sigma_mean", "sigma_max"]
        },
        "products": {
            "gbw_vs_iim_tau_comparison": str(tau_path),
            "gbw_vs_iim_probability_comparison": str(prob_path),
            "gbw_vs_iim_interaction_locations": str(loc_path),
            "gbw_vs_iim_summary": str(summary_path),
        },
    }
    write_json(summary_path, comparison)
    report_path = output_dir / "dis_diagnostics_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["GBW_IIM_comparison"] = comparison
        report.setdefault("products", {}).update(comparison["products"])
        write_json(report_path, report)
    return comparison


def _plot_model_comparison_hist(gbw_values: list[float], iim_values: list[float], output_path: Path, *, title: str, xlabel: str, logx: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gbw_clean = _finite(gbw_values)
    iim_clean = _finite(iim_values)
    if logx:
        gbw_clean = [value for value in gbw_clean if value > 0.0]
        iim_clean = [value for value in iim_clean if value > 0.0]
    fig, ax = plt.subplots(figsize=(7.4, 4.8), facecolor="#f8fafc")
    if gbw_clean:
        ax.hist(gbw_clean, bins=_hist_bins(gbw_clean), histtype="step", linewidth=2.0, color="#2563eb", label="GBW")
    if iim_clean:
        ax.hist(iim_clean, bins=_hist_bins(iim_clean), histtype="step", linewidth=2.0, color="#dc2626", label="IIM")
    if logx and any(value > 0.0 for value in gbw_clean + iim_clean):
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(True, color="#cbd5e1", alpha=0.55, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=165)
    plt.close(fig)


def _plot_model_interactions(gbw_accepted: list[dict[str, Any]], iim_accepted: list[dict[str, Any]], values: dict[str, dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), facecolor="#f8fafc", sharex=True, sharey=True)
    for ax, title, records in [(axes[0], "GBW", gbw_accepted), (axes[1], "IIM", iim_accepted)]:
        _draw_meridional_geometry(ax, values)
        if records:
            xs, zs, colors = [], [], []
            for record in records:
                x, _, z = _xyz(float(record["interaction_r_rg"]), float(record["interaction_theta_rad"]), float(record["interaction_phi_rad"]))
                xs.append(x)
                zs.append(z)
                colors.append(float(record["interaction_E_nu_local_gev"]))
            scatter = ax.scatter(xs, zs, c=colors, s=22, cmap="viridis", edgecolors="black", linewidths=0.25, zorder=5)
            fig.colorbar(scatter, ax=ax, shrink=0.76, label=r"$E_{\nu,\rm local}$ [GeV]")
        ax.set_title(f"{title} accepted={len(records)}")
        ax.set_xlabel(r"$x$ [$r_g$]")
        ax.grid(True, color="#cbd5e1", alpha=0.45, linewidth=0.55)
    axes[0].set_ylabel(r"$z$ [$r_g$]")
    limit = max(float(values["forward_geodesics"]["outer_radius_rg"]) * 0.35, float(values["polar_cone"]["r_max_rg"]) * 0.7, 5.0)
    for ax in axes:
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, dpi=165)
    plt.close(fig)


def _generate_dis_interaction_products_python(values: dict[str, dict[str, Any]], *, run_output_dir: Path) -> dict[str, Any]:
    config_problems = validate_values(values)
    if config_problems:
        raise ValueError("Invalid HADROS3 configuration:\n- " + "\n- ".join(config_problems))
    config = dis_config_from_values(values)
    source_samples_path = uhe_source_dir(run_output_dir) / "uhe_neutrino_source_samples.jsonl"
    forward_paths_path = forward_geodesics_dir(run_output_dir) / "uhe_neutrino_forward_paths.jsonl"
    forward_segments_path = forward_geodesics_dir(run_output_dir) / "uhe_neutrino_forward_path_segments.jsonl"
    source_samples = read_jsonl(source_samples_path)
    forward_paths = read_jsonl(forward_paths_path)
    segments = read_jsonl(forward_segments_path)
    source_map = _source_by_id(source_samples)
    grouped_segments = _group_segments(segments)
    provider = SigmaNuNProvider(config.dis_model)
    output_dir = dis_dir(run_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    r_g_cm = rg_to_cm(config.mass_msun)
    rng = random.Random(config.random_seed)
    path_records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    tau_values: list[float] = []
    max_density = 0.0
    max_sigma = 0.0
    max_d_tau = 0.0
    n_oob_sigma = 0
    n_static_fallback = 0
    n_segments_used_total = 0
    interaction_points_inside = 0
    interaction_points_outside = 0
    cdf_normalized = True
    for path in forward_paths:
        event_id = str(path["event_id"])
        source_sample_id = int(path["source_sample_id"])
        event_segments = grouped_segments.get(event_id, [])
        segment_tau_records: list[dict[str, Any]] = []
        tau_total = 0.0
        path_max_density = 0.0
        path_max_sigma = 0.0
        path_max_d_tau = 0.0
        path_oob = False
        for segment in event_segments:
            rho = analytic_torus_density_g_cm3(
                float(segment["r_mid_rg"]),
                float(segment["theta_mid_rad"]),
                values,
                density_floor_g_cm3=config.density_floor_g_cm3,
            )
            n_baryon = rho / M_BARYON_G
            e_local, fallback = zamo_or_static_local_energy_gev(segment, config.spin_a, config.medium_velocity_model)
            try:
                sigma = provider.sigma_cm2(e_local)
                oob = False
            except ValueError:
                sigma = 0.0
                oob = True
            d_tau = n_baryon * sigma * float(segment["dl_segment_rg"]) * r_g_cm
            d_tau = max(0.0, d_tau)
            tau_total += d_tau
            path_oob = path_oob or oob
            n_oob_sigma += 1 if oob else 0
            n_static_fallback += 1 if fallback else 0
            path_max_density = max(path_max_density, rho)
            path_max_sigma = max(path_max_sigma, sigma)
            path_max_d_tau = max(path_max_d_tau, d_tau)
            segment_tau_records.append(
                {
                    "segment": segment,
                    "rho_g_cm3": rho,
                    "n_baryon_cm3": n_baryon,
                    "E_nu_local_gev": e_local,
                    "sigma_nuN_cm2": sigma,
                    "d_tau_nuN": d_tau,
                    "oob_sigma_table": oob,
                }
            )
        probability = interaction_probability(tau_total)
        if tau_total > 0.0:
            cdf_total = sum(float(entry["d_tau_nuN"]) for entry in segment_tau_records)
            cdf_normalized = cdf_normalized and abs(cdf_total / tau_total - 1.0) <= 1.0e-10
        accepted_flag = bool(tau_total > 0.0 and rng.random() < probability and len(accepted) < config.max_interactions)
        source = source_map.get(source_sample_id, {})
        source_weight = float(source.get("source_weight", 1.0))
        direction_weight = float(source.get("direction_weight", 1.0))
        interaction_weight = 1.0
        expected_interaction_weight = source_weight * direction_weight * probability
        path_status = "ok"
        if not event_segments:
            path_status = "no_forward_segments"
        elif path_oob:
            path_status = "oob_sigma_table"
        record = {
            "event_id": event_id,
            "source_sample_id": source_sample_id,
            "tau_nuN_total": tau_total,
            "interaction_probability": probability,
            "n_segments_used": len(event_segments),
            "dis_model": config.dis_model,
            "medium_model": config.medium_model,
            "medium_velocity_model": config.medium_velocity_model,
            "max_rho_g_cm3": path_max_density,
            "max_sigma_cm2": path_max_sigma,
            "max_d_tau": path_max_d_tau,
            "path_status": path_status,
        }
        path_records.append(record)
        tau_values.append(tau_total)
        n_segments_used_total += len(event_segments)
        max_density = max(max_density, path_max_density)
        max_sigma = max(max_sigma, path_max_sigma)
        max_d_tau = max(max_d_tau, path_max_d_tau)
        candidate = {
            **record,
            "interaction_accepted": accepted_flag,
            "interaction_weight": interaction_weight if accepted_flag else 0.0,
            "source_weight": source_weight,
            "direction_weight": direction_weight,
            "expected_interaction_weight": expected_interaction_weight,
        }
        if segment_tau_records and tau_total > 0.0:
            draw = rng.random() * tau_total
            cumulative = 0.0
            chosen = segment_tau_records[-1]
            for entry in segment_tau_records:
                cumulative += float(entry["d_tau_nuN"])
                if draw <= cumulative:
                    chosen = entry
                    break
            segment = chosen["segment"]
            point = sample_interaction_point_in_medium(
                segment,
                values,
                density_floor_g_cm3=config.density_floor_g_cm3,
                rng=rng,
            )
            candidate.update(
                {
                    "candidate_r_rg": point["r"],
                    "candidate_theta_rad": point["theta"],
                    "candidate_phi_rad": point["phi"],
                    "candidate_E_nu_local_gev": chosen["E_nu_local_gev"],
                    "candidate_rho_g_cm3": point["rho"],
                    "candidate_n_baryon_cm3": chosen["n_baryon_cm3"],
                    "candidate_sigma_nuN_cm2": chosen["sigma_nuN_cm2"],
                    "candidate_d_tau_segment": chosen["d_tau_nuN"],
                    "interaction_point_density_checked": True,
                    "interaction_point_rho_g_cm3": point["rho"],
                    "interaction_point_inside_medium": point["inside"],
                    "interaction_point_sampling_method": point["method"],
                    "interaction_point_sampling_attempts": point["attempts"],
                }
            )
            if accepted_flag:
                if bool(point["inside"]):
                    interaction_points_inside += 1
                else:
                    interaction_points_outside += 1
                accepted.append(
                    {
                        "interaction_id": f"H3DIS-{len(accepted):06d}",
                        "event_id": event_id,
                        "source_sample_id": source_sample_id,
                        "interaction_r_rg": point["r"],
                        "interaction_theta_rad": point["theta"],
                        "interaction_phi_rad": point["phi"],
                        "interaction_E_nu_local_gev": chosen["E_nu_local_gev"],
                        "interaction_rho_g_cm3": point["rho"],
                        "interaction_n_baryon_cm3": chosen["n_baryon_cm3"],
                        "interaction_sigma_nuN_cm2": chosen["sigma_nuN_cm2"],
                        "interaction_d_tau_segment": chosen["d_tau_nuN"],
                        "interaction_point_density_checked": True,
                        "interaction_point_rho_g_cm3": point["rho"],
                        "interaction_point_inside_medium": point["inside"],
                        "interaction_point_sampling_method": point["method"],
                        "interaction_point_sampling_attempts": point["attempts"],
                        "tau_nuN_total": tau_total,
                        "interaction_probability": probability,
                        "interaction_weight": interaction_weight,
                        "source_weight": source_weight,
                        "direction_weight": direction_weight,
                        "final_pre_event_weight": source_weight * direction_weight * interaction_weight,
                        "expected_interaction_weight": expected_interaction_weight,
                        "dis_model": config.dis_model,
                        "medium_model": config.medium_model,
                    }
                )
        candidates.append(candidate)
    tau_min = min(tau_values) if tau_values else 0.0
    tau_max = max(tau_values) if tau_values else 0.0
    tau_mean = sum(tau_values) / len(tau_values) if tau_values else 0.0
    summary_path = output_dir / "dis_summary.json"
    summary_csv_path = output_dir / "dis_summary.csv"
    path_depths_path = output_dir / "dis_path_optical_depths.jsonl"
    candidates_path = output_dir / "dis_interaction_candidates.jsonl"
    accepted_path = output_dir / "dis_accepted_interactions.jsonl"
    tau_preview_path = output_dir / "dis_tau_preview.png"
    locations_path = output_dir / "dis_interaction_locations.png"
    locations_html_path = output_dir / "dis_interaction_locations_3d.html"
    report_path = output_dir / "dis_optical_depth_report.json"
    summary = {
        "status": "ok",
        "backend_language": "Python",
        "backend_executable": "hadros3.dis_sampler",
        "backend_version_or_git_commit": "python-prototype",
        "dis_backend": "python_prototype",
        "backend_kind": "python_reference_debug_backend",
        "cpp_backend_used": False,
        "cuda_backend_used": False,
        "python_prototype_used": True,
        "uses_hadros_original_runtime_path": False,
        "optical_depth_dis_sampler_invoked": True,
        "dis_model": config.dis_model,
        "medium_model": config.medium_model,
        "medium_velocity_model": config.medium_velocity_model,
        "medium_velocity_physics_risk": True,
        "density_model": "analytic_torus_density_v1",
        "medium_renderer_used": True,
        **medium_metadata(),
        "sigma_table_path": str(provider.table_path),
        "sigma_table_rows": len(provider.table),
        "sigma_table_is_compact_builtin_adapter": False,
        "sigma_table_physics_risk": False,
        "sigma_table_energy_min_gev": provider.energy_min_gev,
        "sigma_table_energy_max_gev": provider.energy_max_gev,
        "sigma_energy_min_gev": provider.energy_min_gev,
        "sigma_energy_max_gev": provider.energy_max_gev,
        "interaction_sampling_mode": config.interaction_sampling_mode,
        "interaction_point_sampling_method": "rejection_with_midpoint_fallback",
        "interaction_points_total": len(accepted),
        "interaction_points_inside_medium": interaction_points_inside,
        "interaction_points_outside_medium": interaction_points_outside,
        "interaction_points_outside_medium_fraction": interaction_points_outside / len(accepted) if accepted else 0.0,
        "random_seed": config.random_seed,
        "n_paths_processed": len(path_records),
        "n_segments_processed": n_segments_used_total,
        "n_interactions_accepted": len(accepted),
        "acceptance_fraction": len(accepted) / len(path_records) if path_records else 0.0,
        "tau_min": tau_min,
        "tau_mean": tau_mean,
        "tau_max": tau_max,
        "max_density_g_cm3": max_density,
        "max_sigma_cm2": max_sigma,
        "max_d_tau": max_d_tau,
        "density_model_has_hard_radial_cut": True,
        "density_model_theta_profile": "gaussian",
        "density_model_theta_is_hard_cut": False,
        "n_oob_sigma_table_segments": n_oob_sigma,
        "n_static_to_zamo_fallback_segments": n_static_fallback,
        "observer_bridge_active_filter_invoked": False,
        "expensive_event_generation_invoked": False,
        "powheg_invoked": False,
        "pythia_invoked": False,
        "geant4_invoked": False,
        "products": {
            "dis_path_optical_depths": str(path_depths_path),
            "dis_interaction_candidates": str(candidates_path),
            "dis_accepted_interactions": str(accepted_path),
            "dis_summary": str(summary_csv_path),
            "dis_summary_json": str(summary_path),
            "dis_tau_preview": str(tau_preview_path),
            "dis_interaction_locations": str(locations_path),
            "dis_interaction_locations_3d_html": str(locations_html_path),
            "dis_optical_depth_report": str(report_path),
        },
    }
    report = {
        **summary,
        "validations": {
            "rho_non_negative": all(record["max_rho_g_cm3"] >= 0.0 for record in path_records),
            "n_baryon_non_negative": all(
                record.get("candidate_n_baryon_cm3", 0.0) >= 0.0 for record in candidates if "candidate_n_baryon_cm3" in record
            ),
            "sigma_non_negative": all(record["max_sigma_cm2"] >= 0.0 for record in path_records),
            "d_tau_non_negative": all(record["max_d_tau"] >= 0.0 for record in path_records),
            "tau_non_negative": all(record["tau_nuN_total"] >= 0.0 for record in path_records),
            "probability_bounds": all(0.0 <= record["interaction_probability"] <= 1.0 for record in path_records),
            "cdf_normalized": cdf_normalized,
            "observer_bridge_inactive": True,
            "expensive_event_generation_inactive": True,
            "powheg_inactive": True,
            "pythia_inactive": True,
            "geant4_inactive": True,
        },
    }
    write_jsonl(path_depths_path, path_records)
    write_jsonl(candidates_path, candidates)
    write_jsonl(accepted_path, accepted)
    write_json(summary_path, summary)
    write_json(report_path, report)
    _write_summary_csv(summary_csv_path, summary)
    draw_tau_preview(path_records, tau_preview_path)
    draw_interaction_locations(accepted, segments, values, locations_path)
    write_interaction_locations_html(accepted, values, locations_html_path)
    generate_dis_diagnostics(values, run_output_dir=run_output_dir, summary=summary)
    write_json(summary_path, summary)
    _write_summary_csv(summary_csv_path, summary)
    return summary


def _write_runtime_config(values: dict[str, dict[str, Any]], run_output_dir: Path) -> Path:
    metadata_dir = run_output_dir / "RunMetadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / "hadros3_config.json"
    path.write_text(json.dumps({"hadros3_values": values}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _validation_flags(path_records: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "rho_non_negative": all(record["max_rho_g_cm3"] >= 0.0 for record in path_records),
        "n_baryon_non_negative": all(
            record.get("candidate_n_baryon_cm3", 0.0) >= 0.0 for record in candidates if "candidate_n_baryon_cm3" in record
        ),
        "sigma_non_negative": all(record["max_sigma_cm2"] >= 0.0 for record in path_records),
        "d_tau_non_negative": all(record["max_d_tau"] >= 0.0 for record in path_records),
        "tau_non_negative": all(record["tau_nuN_total"] >= 0.0 for record in path_records),
        "probability_bounds": all(0.0 <= record["interaction_probability"] <= 1.0 for record in path_records),
        "cdf_normalized": True,
        "observer_bridge_inactive": True,
        "expensive_event_generation_inactive": True,
        "powheg_inactive": True,
        "pythia_inactive": True,
        "geant4_inactive": True,
    }


def _relative_error(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale


def _compare_backend_summaries(cpp_summary: dict[str, Any], py_summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "n_paths_processed",
        "n_segments_processed",
        "tau_min",
        "tau_mean",
        "tau_max",
        "acceptance_fraction",
        "n_interactions_accepted",
        "max_density_g_cm3",
        "max_sigma_cm2",
        "max_d_tau",
    ]
    tolerances = {
        "n_paths_processed": 0.0,
        "n_segments_processed": 0.0,
        "n_interactions_accepted": max(1.0, 0.25 * float(cpp_summary.get("n_paths_processed", 0.0))),
        "acceptance_fraction": 0.25,
        "tau_min": 5.0e-12,
        "tau_mean": 5.0e-12,
        "tau_max": 5.0e-12,
        "max_density_g_cm3": 5.0e-12,
        "max_sigma_cm2": 5.0e-12,
        "max_d_tau": 5.0e-12,
    }
    metrics: dict[str, Any] = {}
    pass_flag = True
    for key in keys:
        cpp_value = float(cpp_summary.get(key, 0.0))
        py_value = float(py_summary.get(key, 0.0))
        if key in {"n_paths_processed", "n_segments_processed", "n_interactions_accepted"}:
            delta = abs(cpp_value - py_value)
            ok = delta <= tolerances[key]
        else:
            delta = _relative_error(cpp_value, py_value)
            ok = delta <= tolerances[key]
        metrics[key] = {
            "cpp": cpp_summary.get(key),
            "python": py_summary.get(key),
            "difference_or_relative_error": delta,
            "tolerance": tolerances[key],
            "pass": ok,
        }
        pass_flag = pass_flag and ok
    return {
        "status": "ok" if pass_flag else "warning",
        "comparison_pass": pass_flag,
        "tolerance_note": "Tau/density/sigma/d_tau use relative error. Acceptance metrics may differ because Python and C++ RNG streams are independent but seeded.",
        "metrics": metrics,
    }


def _python_reference_summary(values: dict[str, dict[str, Any]], run_output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="hadros3_dis_backend_validation_") as tmp:
        tmp_dir = Path(tmp)
        for dirname in ["UHEsource", "ForwardGeodesics"]:
            shutil.copytree(run_output_dir / dirname, tmp_dir / dirname)
        reference_values = json.loads(json.dumps(values))
        reference_values["dis_interaction_sampler"]["dis_backend"] = "python_prototype"
        return _generate_dis_interaction_products_python(reference_values, run_output_dir=tmp_dir)


def generate_dis_interaction_products_cpp(values: dict[str, dict[str, Any]], *, run_output_dir: Path) -> dict[str, Any]:
    if not DIS_CPP_EXECUTABLE.exists():
        raise FileNotFoundError(f"H3-W7 C++ backend not built: {DIS_CPP_EXECUTABLE}. Run `make cpp` or `make hadros3-dis-sampler`.")
    _write_runtime_config(values, run_output_dir)
    subprocess.run([str(DIS_CPP_EXECUTABLE), "--run-output", str(run_output_dir)], cwd=ROOT, check=True)
    output_dir = dis_dir(run_output_dir)
    summary_path = output_dir / "dis_summary.json"
    report_path = output_dir / "dis_optical_depth_report.json"
    backend_validation_path = output_dir / "backend_validation_report.json"
    path_depths_path = output_dir / "dis_path_optical_depths.jsonl"
    candidates_path = output_dir / "dis_interaction_candidates.jsonl"
    accepted_path = output_dir / "dis_accepted_interactions.jsonl"
    tau_preview_path = output_dir / "dis_tau_preview.png"
    locations_path = output_dir / "dis_interaction_locations.png"
    locations_html_path = output_dir / "dis_interaction_locations_3d.html"
    path_records = read_jsonl(path_depths_path)
    candidates = read_jsonl(candidates_path)
    accepted = read_jsonl(accepted_path)
    segments = read_jsonl(forward_geodesics_dir(run_output_dir) / "uhe_neutrino_forward_path_segments.jsonl")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "backend_language": "C++17",
            "backend_executable": "bin/hadros3_dis_sampler",
            "backend_kind": "ported_hadros_cpp_dis_optical_depth_sampler",
            "backend_version_or_git_commit": "local-build",
            "dis_backend": "cpp_hadros_original_port",
            "cpp_backend_used": True,
            "cuda_backend_used": False,
            "python_prototype_used": False,
            "uses_hadros_original_runtime_path": False,
            "density_model_has_hard_radial_cut": True,
            "density_model_theta_profile": "gaussian",
            "density_model_theta_is_hard_cut": False,
            "products": {
                **summary.get("products", {}),
                "dis_tau_preview": str(tau_preview_path),
                "dis_interaction_locations": str(locations_path),
                "dis_interaction_locations_3d_html": str(locations_html_path),
                "backend_validation_report": str(backend_validation_path),
            },
        }
    )
    report = {**summary, "validations": _validation_flags(path_records, candidates)}
    try:
        py_summary = _python_reference_summary(values, run_output_dir)
        backend_validation = _compare_backend_summaries(summary, py_summary)
        backend_validation.update(
            {
                "cpp_backend": {key: value for key, value in summary.items() if key != "products"},
                "python_backend": {key: value for key, value in py_summary.items() if key != "products"},
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        backend_validation = {
            "status": "error",
            "comparison_pass": False,
            "message": f"Could not run Python reference comparison: {exc}",
        }
    write_json(summary_path, summary)
    write_json(report_path, report)
    write_json(backend_validation_path, backend_validation)
    _write_summary_csv(output_dir / "dis_summary.csv", summary)
    draw_tau_preview(path_records, tau_preview_path)
    draw_interaction_locations(accepted, segments, values, locations_path)
    write_interaction_locations_html(accepted, values, locations_html_path)
    generate_dis_diagnostics(values, run_output_dir=run_output_dir, summary=summary)
    write_json(summary_path, summary)
    _write_summary_csv(output_dir / "dis_summary.csv", summary)
    return summary


def _generate_dis_interaction_products_base(values: dict[str, dict[str, Any]], *, run_output_dir: Path) -> dict[str, Any]:
    config_problems = validate_values(values)
    if config_problems:
        raise ValueError("Invalid HADROS3 configuration:\n- " + "\n- ".join(config_problems))
    backend = str(values.get("dis_interaction_sampler", {}).get("dis_backend", "cpp_hadros_original_port"))
    if backend == "cpp_hadros_original_port":
        return generate_dis_interaction_products_cpp(values, run_output_dir=run_output_dir)
    if backend == "python_prototype":
        return _generate_dis_interaction_products_python(values, run_output_dir=run_output_dir)
    raise ValueError(f"unsupported dis_interaction_sampler.dis_backend: {backend}")


def generate_dis_interaction_products(values: dict[str, dict[str, Any]], *, run_output_dir: Path, include_model_comparison: bool = True) -> dict[str, Any]:
    summary = _generate_dis_interaction_products_base(values, run_output_dir=run_output_dir)
    if include_model_comparison:
        comparison = generate_gbw_iim_comparison(values, run_output_dir=run_output_dir)
        summary["GBW_IIM_comparison"] = comparison
        summary.setdefault("products", {}).update(comparison["products"])
        output_dir = dis_dir(run_output_dir)
        write_json(output_dir / "dis_summary.json", summary)
        _write_summary_csv(output_dir / "dis_summary.csv", summary)
    return summary
