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
) -> dict[str, Any]:
    source_active = bool(source_summary and source_summary.get("source_sampler_active"))
    return {
        "hadros3_stage": "H3-W0_to_H3-W5_hadros_web_uhe_source_shell" if source_active else "H3-W0_to_H3-W4_hadros_web_geometry_shell",
        "status": "uhe_source_sampled_no_expensive_events" if source_active else "geometry_configured_no_expensive_events",
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
            "forward_neutrino_geodesics": "not_implemented_in_H3_W5",
            "optical_depth_dis_sampler": "not_implemented_in_H3_W5",
            "observer_bridge_active_filter": "placeholder_only",
        },
        "products": products,
        "camera_preview": camera_preview,
        "source_sampler": {
            "source_sampler_active": source_active,
            "source_model": source_summary.get("source_model") if source_summary else values["uhe_neutrino_source"]["source_model"],
            "source_volume_model": source_summary.get("source_volume_model") if source_summary else "coordinate_volume",
            "momentum_generator": source_summary.get("momentum_generator") if source_summary else values["uhe_neutrino_source"].get("momentum_generator"),
            "momentum_is_physical_kerr": source_summary.get("momentum_is_physical_kerr") if source_summary else False,
            "forward_neutrino_geodesics_invoked": False,
            "optical_depth_dis_sampler_invoked": False,
            "observer_bridge_active_filter_invoked": False,
            "expensive_event_generation_invoked": False,
            "summary": source_summary,
        },
        "validation": validation,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
