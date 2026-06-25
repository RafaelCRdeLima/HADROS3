"""Central HADROS3 configuration schema for hadros_web.py."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


def field(
    section: str,
    key: str,
    label: str,
    default: Any,
    *,
    kind: str = "text",
    options: list[str] | None = None,
    visibility: str = "NORMAL",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "section": section,
        "key": key,
        "label": label,
        "default": default,
        "kind": kind,
        "visibility": visibility,
    }
    if options is not None:
        out["options"] = options
    return out


def schema() -> list[dict[str, Any]]:
    return [
        {
            "tab": "Black Hole",
            "fields": [
                field("black_hole", "metric", "Metric", "kerr", kind="select", options=["kerr"]),
                field("black_hole", "mass_msun", "Mass", 3.0, kind="number"),
                field("black_hole", "spin_a", "Spin a", 0.8, kind="number"),
                field("black_hole", "spin_convention", "Spin convention", "Boyer-Lindquist a/M", visibility="EXPERT"),
            ],
        },
        {
            "tab": "Camera",
            "fields": [
                field("observer_camera", "camera_model", "Camera model", "reused_hadros_kerr_camera", kind="select", options=["reused_hadros_kerr_camera"]),
                field(
                    "observer_camera",
                    "camera_preview_mode",
                    "Camera preview mode",
                    "kerr_like_cuda",
                    kind="select",
                    options=["analytic_geometry_only", "kerr_like_cuda", "full_kerr"],
                ),
                field("observer_camera", "observer_distance_rg", "Observer distance", 60.0, kind="number"),
                field("observer_camera", "inclination_deg", "Inclination", 80.0, kind="number"),
                field("observer_camera", "azimuth_deg", "Azimuth", 0.0, kind="number"),
                field("observer_camera", "field_of_view_deg", "FOV", 25.0, kind="number"),
                field("observer_camera", "resolution", "Resolution", "512x288", kind="select", options=["256x144", "512x288", "1024x576", "1920x1080"]),
                field("observer_camera", "preview_resolution", "Preview resolution", "256x144", kind="select", options=["128x72", "256x144", "512x288", "1024x576", "1920x1080"]),
                field("observer_camera", "preview_backend", "Preview backend", "hadros_geodesic_preview_headless", kind="select", options=["hadros_geodesic_preview_headless"]),
                field("observer_camera", "preview_quality", "Preview quality", "fast", kind="select", options=["fast", "medium", "high"], visibility="EXPERT"),
            ],
        },
        {
            "tab": "Analytic Torus",
            "fields": [
                field("analytic_torus", "model", "Torus model", "analytic_torus", kind="select", options=["analytic_torus"]),
                field("analytic_torus", "r_inner_rg", "Inner radius", 6.0, kind="number"),
                field("analytic_torus", "r_outer_rg", "Outer radius", 18.0, kind="number"),
                field("analytic_torus", "r_peak_rg", "Peak radius", 10.0, kind="number"),
                field("analytic_torus", "half_opening_angle_deg", "Half opening angle", 18.0, kind="number"),
                field("analytic_torus", "density_norm_g_cm3", "Density norm", 1.0e10, kind="number"),
                field("analytic_torus", "show_in_preview", "Show in preview", True, kind="checkbox"),
            ],
        },
        {
            "tab": "Polar Cone",
            "fields": [
                field("polar_cone", "enabled", "Enabled", True, kind="checkbox"),
                field("polar_cone", "opening_angle_deg", "Opening angle", 22.0, kind="number"),
                field("polar_cone", "r_min_rg", "Inner radius", 2.2, kind="number"),
                field("polar_cone", "r_max_rg", "Outer radius", 40.0, kind="number"),
                field("polar_cone", "draw_mode", "Draw mode", "bipolar_funnel", kind="select", options=["bipolar_funnel", "north_only"]),
            ],
        },
        {
            "tab": "UHE Source",
            "fields": [
                field("uhe_neutrino_source", "source_model", "Source model", "polar_cone", kind="select", options=["polar_cone"]),
                field("uhe_neutrino_source", "energy_model", "Energy model", "monoenergetic", kind="select", options=["monoenergetic"]),
                field("uhe_neutrino_source", "energy_gev", "Monoenergetic energy", "10^{9}", kind="text"),
                field("uhe_neutrino_source", "r_min_rg", "Source r min", 3.0, kind="number"),
                field("uhe_neutrino_source", "r_max_rg", "Source r max", 12.0, kind="number"),
                field("uhe_neutrino_source", "theta_min_deg", "Theta min", 0.0, kind="number"),
                field("uhe_neutrino_source", "theta_max_deg", "Theta max", 22.0, kind="number"),
                field("uhe_neutrino_source", "phi_mode", "Phi mode", "uniform_0_2pi", kind="select", options=["uniform_0_2pi"]),
                field("uhe_neutrino_source", "n_samples", "Number of samples", 1000, kind="number"),
                field("uhe_neutrino_source", "random_seed", "Random seed", 12345, kind="number"),
                field(
                    "uhe_neutrino_source",
                    "sampling_mode",
                    "Sampling mode",
                    "uniform_coordinate_volume",
                    kind="select",
                    options=["uniform_coordinate_volume"],
                ),
                field(
                    "uhe_neutrino_source",
                    "momentum_generator",
                    "Initial momentum generator",
                    "ProxyRadialMomentumGenerator",
                    kind="select",
                    options=["ProxyRadialMomentumGenerator"],
                ),
                field(
                    "uhe_neutrino_source",
                    "direction_model",
                    "Direction Model",
                    "coordinate_radial_outward",
                    kind="select",
                    options=[
                        "coordinate_radial_outward",
                        "jet_axis_future",
                        "cone_emission_future",
                        "isotropic_local_future",
                        "custom_future",
                    ],
                ),
                field("uhe_neutrino_source", "direction_opening_angle_deg", "Direction opening angle", 0.0, kind="number"),
                field("uhe_neutrino_source", "direction_seed", "Direction seed", 12345, kind="number"),
                field(
                    "uhe_neutrino_source",
                    "status",
                    "Status",
                    "configured_only_no_sampling",
                    kind="select",
                    options=[
                        "configured_only_no_sampling",
                        "sampled_position_direction_energy_no_forward_kerr_geodesic",
                        "sampled_position_with_proxy_direction_no_forward_kerr_geodesic",
                    ],
                ),
            ],
        },
        {
            "tab": "Forward Geodesics",
            "fields": [
                field(
                    "forward_geodesics",
                    "geodesic_backend",
                    "Geodesic backend",
                    "hadros3_kerr_null_radial",
                    kind="select",
                    options=["hadros3_kerr_null_radial"],
                ),
                field("forward_geodesics", "n_samples_to_propagate", "Neutrinos to propagate", 64, kind="number"),
                field("forward_geodesics", "initial_step_rg", "Initial step", 1.0, kind="number"),
                field("forward_geodesics", "max_steps", "Max steps", 256, kind="number"),
                field("forward_geodesics", "outer_radius_rg", "Outer radius", 80.0, kind="number"),
                field("forward_geodesics", "horizon_tolerance_rg", "Horizon tolerance", 1.0e-3, kind="number"),
                field("forward_geodesics", "null_invariant_tolerance", "Null norm tolerance", 1.0e-6, kind="number"),
                field("forward_geodesics", "killing_energy_tolerance", "Killing energy tolerance", 1.0e-10, kind="number"),
                field("forward_geodesics", "lz_tolerance", "Lz tolerance", 1.0e-10, kind="number"),
                field(
                    "forward_geodesics",
                    "status",
                    "Status",
                    "configured_only_no_forward_geodesics",
                    kind="select",
                    options=["configured_only_no_forward_geodesics", "forward_kerr_geodesics_propagated_no_interactions"],
                ),
            ],
        },
        {
            "tab": "Interaction Sampler",
            "fields": [
                field("interaction_sampler", "mode", "Interaction sampler", "placeholder_disabled", kind="select", options=["placeholder_disabled"]),
                field("interaction_sampler", "planned_model", "Planned model", "optical_depth_DIS", kind="select", options=["optical_depth_DIS"]),
            ],
        },
        {
            "tab": "Observer Bridge",
            "fields": [
                field("observer_bridge", "mode", "Observer bridge", "placeholder_diagnostic_disabled", kind="select", options=["placeholder_diagnostic_disabled"]),
                field("observer_bridge", "planned_model", "Planned model", "distance_to_observer_bundle", kind="select", options=["distance_to_observer_bundle"]),
            ],
        },
        {
            "tab": "Outputs",
            "fields": [
                field("outputs", "write_config", "Write config", True, kind="checkbox"),
                field("outputs", "write_provenance", "Write provenance", True, kind="checkbox"),
                field("outputs", "write_geometry_preview", "Write geometry preview", True, kind="checkbox"),
                field("outputs", "write_schematic", "Write schematic", True, kind="checkbox"),
                field("outputs", "write_camera_preview", "Write camera preview", True, kind="checkbox"),
                field("outputs", "write_html_summary", "Write HTML summary", True, kind="checkbox"),
            ],
        },
        {
            "tab": "Provenance",
            "fields": [
                field("provenance", "record_reused_hadros_components", "Record reused HADROS components", True, kind="checkbox"),
                field("provenance", "record_disabled_stages", "Record disabled expensive stages", True, kind="checkbox"),
                field("provenance", "trust_boundary", "Trust boundary", "H3-W0_W4_geometry_preview_only", kind="select", options=["H3-W0_W4_geometry_preview_only"]),
            ],
        },
    ]


def defaults() -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {"run": {"run_name": "HADROS3_hadros_web_preview"}}
    for tab in schema():
        for item in tab["fields"]:
            values.setdefault(item["section"], {})[item["key"]] = copy.deepcopy(item["default"])
    return values


def safe_run_name(name: Any) -> str:
    text = str(name or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._-")
    return text or "HADROS3_run"


def run_output_dir(values: dict[str, Any]) -> Path:
    return Path("output") / safe_run_name(values.get("run", {}).get("run_name", "HADROS3_run"))


def parse_latex_number(value: Any) -> float:
    """Parse plain/scientific numbers and simple LaTeX powers like 10^{12}."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        raise ValueError("empty numeric expression")
    compact = (
        text.replace(" ", "")
        .replace("\\,", "")
        .replace("\\times", "*")
        .replace("\\cdot", "*")
        .replace("×", "*")
        .replace("{", "")
        .replace("}", "")
    )
    try:
        return float(compact)
    except ValueError:
        pass
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))?\*?10\^([+-]?\d+(?:\.\d*)?)", compact)
    if match:
        coefficient = float(match.group(1)) if match.group(1) not in {None, "", "+", "-"} else 1.0
        if match.group(1) == "-":
            coefficient = -1.0
        return coefficient * (10.0 ** float(match.group(2)))
    raise ValueError(f"unsupported numeric expression: {value!r}")


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for section, section_values in updates.items():
        if isinstance(section_values, dict):
            out.setdefault(section, {})
            out[section].update(section_values)
        else:
            out[section] = section_values
    return out


def load_values(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return defaults()
    return deep_update(defaults(), json.loads(path.read_text(encoding="utf-8")))


def validate_values(values: dict[str, dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    bh = values["black_hole"]
    cam = values["observer_camera"]
    torus = values["analytic_torus"]
    cone = values["polar_cone"]
    source = values["uhe_neutrino_source"]
    forward = values["forward_geodesics"]

    spin = float(bh["spin_a"])
    if not (-0.999 <= spin <= 0.999):
        problems.append("black_hole.spin_a must satisfy -0.999 <= a <= 0.999")
    if float(bh["mass_msun"]) <= 0.0:
        problems.append("black_hole.mass_msun must be positive")
    if float(cam["observer_distance_rg"]) <= 0.0:
        problems.append("observer_camera.observer_distance_rg must be positive")
    if not (0.0 < float(cam["field_of_view_deg"]) < 180.0):
        problems.append("observer_camera.field_of_view_deg must satisfy 0 < FOV < 180")
    if not (0.0 <= float(cam["inclination_deg"]) <= 180.0):
        problems.append("observer_camera.inclination_deg must satisfy 0 <= inclination <= 180")
    if str(cam["camera_preview_mode"]) not in {"analytic_geometry_only", "kerr_like_cuda", "full_kerr"}:
        problems.append("observer_camera.camera_preview_mode is unsupported")
    if float(torus["r_inner_rg"]) <= 0.0 or float(torus["r_outer_rg"]) <= float(torus["r_inner_rg"]):
        problems.append("analytic_torus requires 0 < r_inner_rg < r_outer_rg")
    if not (float(torus["r_inner_rg"]) <= float(torus["r_peak_rg"]) <= float(torus["r_outer_rg"])):
        problems.append("analytic_torus.r_peak_rg must lie inside [r_inner_rg, r_outer_rg]")
    if not (0.0 < float(torus["half_opening_angle_deg"]) < 90.0):
        problems.append("analytic_torus.half_opening_angle_deg must satisfy 0 < angle < 90")
    if float(cone["r_max_rg"]) <= float(cone["r_min_rg"]):
        problems.append("polar_cone requires r_min_rg < r_max_rg")
    if not (0.0 < float(cone["opening_angle_deg"]) < 90.0):
        problems.append("polar_cone.opening_angle_deg must satisfy 0 < angle < 90")
    if float(source["r_max_rg"]) <= float(source["r_min_rg"]):
        problems.append("uhe_neutrino_source requires r_min_rg < r_max_rg")
    if float(source["r_min_rg"]) <= 0.0:
        problems.append("uhe_neutrino_source.r_min_rg must be positive")
    theta_min = float(source["theta_min_deg"])
    theta_max = float(source["theta_max_deg"])
    if not (0.0 <= theta_min < theta_max <= 180.0):
        problems.append("uhe_neutrino_source requires 0 <= theta_min_deg < theta_max_deg <= 180")
    cone_opening = float(cone["opening_angle_deg"])
    if theta_max > cone_opening:
        problems.append("uhe_neutrino_source.theta_max_deg must be <= polar_cone.opening_angle_deg")
    if int(float(source.get("n_samples", 0))) <= 0:
        problems.append("uhe_neutrino_source.n_samples must be positive")
    if str(source.get("source_model")) != "polar_cone":
        problems.append("uhe_neutrino_source.source_model must be polar_cone in H3-W5")
    if str(source.get("energy_model")) != "monoenergetic":
        problems.append("uhe_neutrino_source.energy_model must be monoenergetic in H3-W5")
    if str(source.get("phi_mode")) != "uniform_0_2pi":
        problems.append("uhe_neutrino_source.phi_mode must be uniform_0_2pi in H3-W5")
    if str(source.get("sampling_mode")) != "uniform_coordinate_volume":
        problems.append("uhe_neutrino_source.sampling_mode must be uniform_coordinate_volume in H3-W5")
    if str(source.get("momentum_generator")) != "ProxyRadialMomentumGenerator":
        problems.append("uhe_neutrino_source.momentum_generator must be ProxyRadialMomentumGenerator in H3-W5")
    if str(source.get("direction_model")) != "coordinate_radial_outward":
        problems.append("uhe_neutrino_source.direction_model must be coordinate_radial_outward in H3-W5")
    if float(source.get("direction_opening_angle_deg", 0.0)) < 0.0:
        problems.append("uhe_neutrino_source.direction_opening_angle_deg must be non-negative")
    try:
        int(float(source.get("direction_seed", 0)))
    except (TypeError, ValueError):
        problems.append("uhe_neutrino_source.direction_seed must be an integer")
    try:
        energy_gev = parse_latex_number(source["energy_gev"])
        if energy_gev <= 0.0:
            problems.append("uhe_neutrino_source.energy_gev must be positive")
    except ValueError as exc:
        problems.append(f"uhe_neutrino_source.energy_gev is invalid: {exc}")
    if str(forward.get("geodesic_backend")) != "hadros3_kerr_null_radial":
        problems.append("forward_geodesics.geodesic_backend is unsupported in H3-W6")
    if int(float(forward.get("n_samples_to_propagate", 0))) <= 0:
        problems.append("forward_geodesics.n_samples_to_propagate must be positive")
    if int(float(forward.get("max_steps", 0))) <= 0:
        problems.append("forward_geodesics.max_steps must be positive")
    if float(forward.get("initial_step_rg", 0.0)) <= 0.0:
        problems.append("forward_geodesics.initial_step_rg must be positive")
    if float(forward.get("outer_radius_rg", 0.0)) <= float(source["r_max_rg"]):
        problems.append("forward_geodesics.outer_radius_rg must exceed uhe_neutrino_source.r_max_rg")
    if float(forward.get("horizon_tolerance_rg", 0.0)) < 0.0:
        problems.append("forward_geodesics.horizon_tolerance_rg must be non-negative")
    for key in ["null_invariant_tolerance", "killing_energy_tolerance", "lz_tolerance"]:
        if float(forward.get(key, 0.0)) <= 0.0:
            problems.append(f"forward_geodesics.{key} must be positive")
    return problems


def flatten_for_legacy_camera(values: dict[str, dict[str, Any]]) -> dict[str, Any]:
    camera = values["observer_camera"]
    bh = values["black_hole"]
    torus = values["analytic_torus"]
    return {
        "black_hole_mass_msun": float(bh["mass_msun"]),
        "spin": float(bh["spin_a"]),
        "camera_theta_deg": float(camera["inclination_deg"]),
        "camera_fov_deg": float(camera["field_of_view_deg"]),
        "camera_r_obs_rg": float(camera["observer_distance_rg"]),
        "camera_resolution": str(camera["resolution"]),
        "preview_resolution": str(camera["preview_resolution"]),
        "torus_r0_rg": float(torus["r_peak_rg"]),
        "torus_r_inner_rg": float(torus["r_inner_rg"]),
        "torus_r_outer_rg": float(torus["r_outer_rg"]),
        "torus_half_opening_angle_deg": float(torus["half_opening_angle_deg"]),
    }
