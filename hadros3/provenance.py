"""Provenance writer for the HADROS3 hadros-web first stage."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .reuse import discover_original_hadros


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def build_provenance(
    *,
    root: Path,
    values: dict[str, dict[str, Any]],
    products: dict[str, str],
    validation: dict[str, Any],
    camera_preview: dict[str, Any] | None = None,
    source_summary: dict[str, Any] | None = None,
    forward_geodesic_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_active = bool(source_summary and source_summary.get("source_sampler_active"))
    forward_active = bool(forward_geodesic_summary and forward_geodesic_summary.get("forward_neutrino_geodesics_invoked"))
    return {
        "hadros3_stage": "H3-W0_to_H3-W6_forward_neutrino_geodesics" if forward_active else ("H3-W0_to_H3-W5_hadros_web_uhe_source_shell" if source_active else "H3-W0_to_H3-W4_hadros_web_geometry_shell"),
        "status": "forward_geodesics_propagated_no_interactions" if forward_active else ("uhe_source_sampled_no_expensive_events" if source_active else "geometry_configured_no_expensive_events"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hadros3_version": __version__,
        "git_commit": _git_commit(root),
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": values,
        "reused_hadros_components": discover_original_hadros(),
        "disabled_expensive_or_future_stages": {
            "powheg": "disabled",
            "pythia": "disabled",
            "geant4": "disabled",
            "forward_neutrino_geodesics": "active_H3_W6" if forward_active else "not_invoked",
            "optical_depth_dis_sampler": "not_implemented_in_H3_W6",
            "observer_bridge_active_filter": "placeholder_only",
        },
        "products": products,
        "camera_preview": camera_preview,
        "source_sampler": {
            "source_sampler_active": source_active,
            "source_model": source_summary.get("source_model") if source_summary else values["uhe_neutrino_source"]["source_model"],
            "source_volume_model": source_summary.get("source_volume_model") if source_summary else "coordinate_volume",
            "direction_generator": source_summary.get("direction_generator") if source_summary else values["uhe_neutrino_source"].get("direction_model"),
            "direction_model": source_summary.get("direction_model") if source_summary else values["uhe_neutrino_source"].get("direction_model"),
            "direction_sampling_pdf": source_summary.get("direction_sampling_pdf") if source_summary else None,
            "direction_physical_pdf": source_summary.get("direction_physical_pdf") if source_summary else None,
            "direction_weight": source_summary.get("direction_weight") if source_summary else None,
            "four_momentum_sampled_in_source": False,
            "momentum_generator": source_summary.get("momentum_generator") if source_summary else values["uhe_neutrino_source"].get("momentum_generator"),
            "momentum_is_physical_kerr": source_summary.get("momentum_is_physical_kerr") if source_summary else False,
            "forward_neutrino_geodesics_invoked": forward_active,
            "optical_depth_dis_sampler_invoked": False,
            "observer_bridge_active_filter_invoked": False,
            "expensive_event_generation_invoked": False,
            "summary": source_summary,
        },
        "forward_geodesics": {
            "forward_neutrino_geodesics_invoked": forward_active,
            "momentum_generator": forward_geodesic_summary.get("momentum_generator") if forward_active else None,
            "momentum_is_physical_kerr": forward_geodesic_summary.get("momentum_is_physical_kerr") if forward_active else False,
            "direction_model": forward_geodesic_summary.get("direction_model") if forward_active else None,
            "forward_geodesics_consumes_source_direction": True,
            "four_momentum_constructed_from_source_direction": True,
            "four_momentum_sampled_in_source": False,
            "input_source_samples": forward_geodesic_summary.get("input_source_samples") if forward_active else None,
            "geodesic_backend": forward_geodesic_summary.get("geodesic_backend") if forward_active else values["forward_geodesics"].get("geodesic_backend"),
            "n_samples_requested": forward_geodesic_summary.get("n_samples_requested") if forward_active else values["forward_geodesics"].get("n_samples_to_propagate"),
            "n_samples_propagated": forward_geodesic_summary.get("n_samples_propagated") if forward_active else 0,
            "max_steps": forward_geodesic_summary.get("max_steps") if forward_active else values["forward_geodesics"].get("max_steps"),
            "initial_step_rg": forward_geodesic_summary.get("initial_step_rg") if forward_active else values["forward_geodesics"].get("initial_step_rg"),
            "outer_radius_rg": forward_geodesic_summary.get("outer_radius_rg") if forward_active else values["forward_geodesics"].get("outer_radius_rg"),
            "null_invariant_tolerance": forward_geodesic_summary.get("null_invariant_tolerance") if forward_active else values["forward_geodesics"].get("null_invariant_tolerance"),
            "killing_energy_tolerance": forward_geodesic_summary.get("killing_energy_tolerance") if forward_active else values["forward_geodesics"].get("killing_energy_tolerance"),
            "lz_tolerance": forward_geodesic_summary.get("lz_tolerance") if forward_active else values["forward_geodesics"].get("lz_tolerance"),
            "stop_condition_counts": forward_geodesic_summary.get("stop_condition_counts") if forward_active else {},
            "optical_depth_dis_sampler_invoked": False,
            "observer_bridge_active_filter_invoked": False,
            "expensive_event_generation_invoked": False,
            "summary": forward_geodesic_summary,
        },
        "validation": validation,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
