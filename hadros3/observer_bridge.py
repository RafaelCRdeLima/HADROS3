"""H3-W8 Observer Bridge scoring products."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import validate_values
from .paths import observer_bridge_dir, run_metadata_dir
from .provenance import write_json


ROOT = Path(__file__).resolve().parents[1]
OBSERVER_BRIDGE_CPP_EXECUTABLE = ROOT / "bin" / "hadros3_observer_bridge"


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

    summary = _augment_summary(summary, output_dir)
    write_json(summary_path, summary)
    write_json(report_path, summary)
    return summary
