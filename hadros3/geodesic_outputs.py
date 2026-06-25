"""Output writers for H3-W6 forward neutrino geodesics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


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


def draw_forward_preview(paths: list[dict[str, Any]], segments: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#101318")
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
        ax.plot(xs, zs, color="#9de2ff", alpha=0.38, linewidth=0.8)
        if xs:
            ax.scatter([xs[0]], [zs[0]], color="#ff6f59", s=8, alpha=0.65)
    ax.scatter([0], [0], color="black", edgecolor="white", s=180, zorder=5)
    ax.text(0, 0, "BH", color="white", ha="center", va="center", fontsize=8, zorder=6)
    max_radius = 10.0
    if segments:
        max_radius = max(
            max(abs(float(segment["r_start_rg"])) for segment in segments),
            max(abs(float(segment["r_end_rg"])) for segment in segments),
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
        f"paths={len(paths)} segments={len(segments)}",
        transform=ax.transAxes,
        color="#dce6f2",
        fontsize=9,
        bbox=dict(facecolor="#151b24", edgecolor="#334155", alpha=0.85, pad=4),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
