"""HADROS3 hadros-web render pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .camera_preview import render_camera_preview
from .config import flatten_for_legacy_camera, run_output_dir, validate_values
from .provenance import build_provenance, write_json
from .render import draw_geometry_preview, draw_system_schematic, write_html_summary


def render_hadros_web(
    values: dict[str, dict[str, Any]],
    *,
    root: Path,
    output_dir: Path | None = None,
    source_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    problems = validate_values(values)
    if problems:
        raise ValueError("Invalid HADROS3 configuration:\n- " + "\n- ".join(problems))

    run_output = output_dir if output_dir is not None else root / run_output_dir(values)
    if not run_output.is_absolute():
        run_output = root / run_output
    run_output.mkdir(parents=True, exist_ok=True)
    if source_summary is None:
        existing_source_summary = run_output / "uhe_neutrino_source_summary.json"
        if existing_source_summary.exists():
            try:
                source_summary = json.loads(existing_source_summary.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                source_summary = None

    config_path = run_output / "hadros3_config.json"
    provenance_path = run_output / "hadros3_pipeline_provenance.json"
    preview_path = run_output / "hadros3_geometry_preview.png"
    schematic_path = run_output / "hadros3_system_schematic.png"
    html_path = run_output / "index.html"

    products: dict[str, str] = {}
    camera_preview_summary: dict[str, Any] | None = None
    if source_summary:
        for key, value in source_summary.get("products", {}).items():
            products[key] = str(value)
    if bool(values["outputs"]["write_config"]):
        config_payload = {
            "hadros3_values": values,
            "legacy_hadros_camera_mapping": flatten_for_legacy_camera(values),
        }
        write_json(config_path, config_payload)
        products["config"] = str(config_path)
    if bool(values["outputs"]["write_geometry_preview"]):
        draw_geometry_preview(values, preview_path)
        products["geometry_preview"] = str(preview_path)
    if bool(values["outputs"]["write_schematic"]):
        draw_system_schematic(values, schematic_path)
        products["system_schematic"] = str(schematic_path)
    if bool(values["outputs"].get("write_camera_preview", True)):
        camera_preview_summary = render_camera_preview(values, root=root, output_dir=run_output)
        products["camera_preview"] = str(run_output / "hadros3_camera_preview.png")
        products["camera_preview_summary"] = str(run_output / "hadros3_camera_preview_summary.json")

    validation = {
        "configuration_valid": True,
        "validation_errors": [],
        "expensive_event_generation_invoked": False,
        "forward_neutrino_geodesics_invoked": False,
        "optical_depth_dis_sampler_invoked": False,
        "observer_bridge_active_filter_invoked": False,
        "source_sampler_active": bool(source_summary),
    }
    if bool(values["outputs"]["write_provenance"]):
        provenance = build_provenance(
            root=root,
            values=values,
            products=products,
            validation=validation,
            camera_preview=camera_preview_summary,
            source_summary=source_summary,
        )
        write_json(provenance_path, provenance)
        products["provenance"] = str(provenance_path)
    if bool(values["outputs"]["write_html_summary"]):
        write_html_summary(values, products, html_path)
        products["html_summary"] = str(html_path)

    summary = {
        "status": "ok",
        "output_dir": str(run_output),
        "products": products,
        "validation": validation,
    }
    (run_output / "hadros_web_render_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    products["render_summary"] = str(run_output / "hadros_web_render_summary.json")
    return summary
