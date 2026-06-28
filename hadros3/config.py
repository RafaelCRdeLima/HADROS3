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
    help_text: str | None = None,
    minimum: int | float | None = None,
    step: int | float | None = None,
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
    if help_text is not None:
        out["help"] = help_text
    if minimum is not None:
        out["min"] = minimum
    if step is not None:
        out["step"] = step
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
                    visibility="INTERNAL",
                ),
                field("observer_camera", "observer_distance_rg", "Observer distance", 60.0, kind="number"),
                field("observer_camera", "inclination_deg", "Inclination", 80.0, kind="number"),
                field("observer_camera", "azimuth_deg", "Azimuth", 0.0, kind="number"),
                field("observer_camera", "field_of_view_deg", "FOV", 25.0, kind="number"),
                field("observer_camera", "pixel_width", "Pixels X", 512, kind="number"),
                field("observer_camera", "pixel_height", "Pixels Y", 288, kind="number"),
                field(
                    "observer_camera",
                    "resolution",
                    "Camera resolution",
                    "512x288",
                    kind="text",
                    visibility="INTERNAL",
                ),
                field(
                    "observer_camera",
                    "preview_final_resolution",
                    "Preview final resolution",
                    "512x288",
                    kind="select",
                    options=["256x144", "512x288", "1024x576", "1920x1080"],
                    visibility="INTERNAL",
                ),
                field(
                    "observer_camera",
                    "preview_resolution",
                    "Preview interactive resolution",
                    "256x144",
                    kind="select",
                    options=["128x72", "256x144", "512x288", "1024x576", "1920x1080"],
                    visibility="INTERNAL",
                ),
                field(
                    "observer_camera",
                    "preview_backend",
                    "Preview backend",
                    "hadros_geodesic_preview_headless",
                    kind="select",
                    options=["hadros_geodesic_preview_headless"],
                    visibility="INTERNAL",
                ),
                field(
                    "observer_camera",
                    "preview_quality",
                    "Preview quality",
                    "fast",
                    kind="select",
                    options=["fast", "medium", "high"],
                    visibility="INTERNAL",
                ),
                field(
                    "observer_camera",
                    "preview_nav_mode",
                    "Preview mode",
                    "celestial_plus_torus_volume",
                    kind="select",
                    options=["celestial_plus_torus_volume", "paint_swatch_disk"],
                    visibility="INTERNAL",
                ),
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
                    "isotropic_local",
                    kind="select",
                    options=[
                        "isotropic_local",
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
                    "forward_backend",
                    "Forward backend",
                    "cpp_hadros_original_port",
                    kind="select",
                    options=["cpp_hadros_original_port", "python_prototype"],
                ),
                field(
                    "forward_geodesics",
                    "geodesic_backend",
                    "Geodesic backend",
                    "cpp_hadros_original_port",
                    kind="select",
                    options=["cpp_hadros_original_port"],
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
            "tab": "DIS Interaction Sampler",
            "fields": [
                field(
                    "dis_interaction_sampler",
                    "medium_model",
                    "Medium model",
                    "analytic_torus",
                    kind="select",
                    options=["analytic_torus", "future_hydrodynamic"],
                ),
                field(
                    "dis_interaction_sampler",
                    "medium_velocity_model",
                    "Medium velocity",
                    "zamo_fallback",
                    kind="select",
                    options=["zamo_fallback", "static", "future_hydro_velocity"],
                ),
                field("dis_interaction_sampler", "density_floor_g_cm3", "Density floor", 0.0, kind="number"),
                field(
                    "dis_interaction_sampler",
                    "dis_backend",
                    "DIS backend",
                    "cpp_hadros_original_port",
                    kind="select",
                    options=["cpp_hadros_original_port", "python_prototype"],
                ),
                field("dis_interaction_sampler", "dis_model", "DIS model", "GBW", kind="select", options=["GBW", "IIM"]),
                field(
                    "dis_interaction_sampler",
                    "interaction_sampling_mode",
                    "Interaction sampling mode",
                    "optical_depth_inverse_cdf",
                    kind="select",
                    options=["optical_depth_inverse_cdf"],
                ),
                field("dis_interaction_sampler", "max_interactions", "Maximum accepted interactions", 1000000, kind="number"),
                field("dis_interaction_sampler", "random_seed", "Random seed", 24680, kind="number"),
                field(
                    "dis_interaction_sampler",
                    "status",
                    "Status",
                    "configured_only_no_dis_sampling",
                    kind="select",
                    options=["configured_only_no_dis_sampling", "dis_optical_depth_sampled_no_observer_bridge"],
                ),
            ],
        },
        {
            "tab": "Observer Bridge",
            "fields": [
                field("observer_bridge", "observer_bridge_backend", "Observer Bridge backend", "cpp_cpu", kind="select", options=["cpp_cpu"]),
                field("observer_bridge", "bridge_mode", "Bridge mode", "scoring_only", kind="select", options=["scoring_only"]),
                field(
                    "observer_bridge",
                    "secondary_particle_proxy_model",
                    "Secondary proxy",
                    "geometric_escape_proxy",
                    kind="select",
                    options=["geometric_escape_proxy"],
                ),
                field("observer_bridge", "escape_proxy_model", "Escape proxy", "geometric_outward_proxy", kind="select", options=["geometric_outward_proxy"]),
                field("observer_bridge", "visibility_model", "Visibility model", "geometric_proxy", kind="select", options=["geometric_proxy"]),
                field("observer_bridge", "redshift_proxy_model", "Redshift proxy", "unity_or_metric_proxy", kind="select", options=["unity_or_metric_proxy"]),
                field("observer_bridge", "line_of_sight_proxy_model", "Line of sight proxy", "geometric_proxy", kind="select", options=["geometric_proxy"]),
                field("observer_bridge", "fov_policy", "FOV policy", "hard", kind="select", options=["hard", "soft"]),
                field("observer_bridge", "distance_weight_enabled", "Distance weight", True, kind="checkbox"),
                field("observer_bridge", "redshift_weight_enabled", "Redshift weight", True, kind="checkbox"),
                field("observer_bridge", "line_of_sight_check_enabled", "Line of sight check", True, kind="checkbox"),
                field("observer_bridge", "max_ranked_events", "Max ranked events", 25, kind="number"),
                field("observer_bridge", "min_observer_weight", "Min observer weight", 0.0, kind="number"),
                field("observer_bridge", "min_final_observation_score", "Min final score", 0.0, kind="number"),
                field(
                    "observer_bridge",
                    "downstream_selection_policy",
                    "Selection policy",
                    "top_n",
                    kind="select",
                    options=["all_candidates", "top_n", "score_threshold"],
                    help_text="Observer Bridge policy for selecting ranked candidates that downstream stages such as POWHEG may consume.",
                ),
                field(
                    "observer_bridge",
                    "downstream_top_n_candidates",
                    "Top N candidates",
                    50,
                    kind="number",
                    help_text="Number of highest-scoring Observer Bridge candidates selected for downstream stages when policy is top_n.",
                    minimum=1,
                    step=1,
                ),
                field(
                    "observer_bridge",
                    "downstream_min_final_observation_score",
                    "Minimum observation score",
                    0.0,
                    kind="number",
                    help_text="Minimum final_observation_score selected for downstream stages when policy is score_threshold.",
                    minimum=0.0,
                ),
                field(
                    "observer_bridge",
                    "candidate_overlay_mapping",
                    "Candidate overlay mapping",
                    "kerr_pixel_match",
                    kind="select",
                    options=["kerr_pixel_match", "geometric_proxy"],
                ),
                field("observer_bridge", "kerr_pixel_match_resolution_x", "Kerr match rays X", 32, kind="number"),
                field("observer_bridge", "kerr_pixel_match_resolution_y", "Kerr match rays Y", 18, kind="number"),
                field("observer_bridge", "kerr_pixel_match_tolerance_rg", "Kerr match tolerance [rg]", 3.5, kind="number"),
                field("observer_bridge", "kerr_pixel_match_refine_enabled", "Kerr match refine", True, kind="checkbox"),
                field("observer_bridge", "candidate_matching_radius_rg", "Candidate matching radius [rg]", 3.5, kind="number", visibility="INTERNAL"),
                field("observer_bridge", "multi_image_audit_resolution_x", "Multi-image audit rays X", 9, kind="number", visibility="INTERNAL"),
                field("observer_bridge", "multi_image_audit_resolution_y", "Multi-image audit rays Y", 5, kind="number", visibility="INTERNAL"),
                field(
                    "observer_bridge",
                    "kerr_pixel_match_basis_transform",
                    "Kerr match ray basis",
                    "cuda_preview_local_tetrad",
                    kind="select",
                    options=["cuda_preview_local_tetrad", "up_flipped", "right_flipped", "up_right_flipped"],
                    visibility="INTERNAL",
                ),
                field(
                    "observer_bridge",
                    "observer_bridge_orientation_diagnostics_enabled",
                    "Orientation diagnostics",
                    False,
                    kind="checkbox",
                    help_text="Generate expensive TOP/BOTTOM/LEFT/RIGHT overlay orientation diagnostics. The main overlay is generated regardless.",
                ),
                field("observer_bridge", "interactive_max_candidates", "Interactive max candidates", 40, kind="number"),
                field("observer_bridge", "interactive_max_rays", "Interactive max rays", 64, kind="number"),
                field("observer_bridge", "interactive_ray_stride", "Interactive ray stride", 4, kind="number"),
                field(
                    "observer_bridge",
                    "interactive_candidate_color_mode",
                    "Interactive color by",
                    "final_observation_score",
                    kind="select",
                    options=["final_observation_score", "closest_approach_rg", "inside_outside_fov"],
                ),
                field(
                    "observer_bridge",
                    "status",
                    "Status",
                    "configured_only_no_observer_bridge",
                    kind="select",
                    options=["configured_only_no_observer_bridge", "observer_bridge_scored_no_event_generation"],
                ),
            ],
        },
        {
            "tab": "POWHEG",
            "fields": [
                field("powheg", "powheg_backend", "POWHEG backend", "local_powheg", kind="select", options=["local_powheg"]),
                field("powheg", "powheg_process", "POWHEG process", "nudis", kind="select", options=["nudis"]),
                field(
                    "powheg",
                    "events_per_candidate",
                    "POWHEG Events per Interaction",
                    1,
                    kind="number",
                    help_text="Number of independent POWHEG Monte Carlo hard-scattering realizations generated for each selected interaction site.",
                    minimum=1,
                    step=1,
                ),
                field("powheg", "random_seed", "Random seed", 12345, kind="number"),
                field("powheg", "powheg_seed_mode", "Seed mode", "base_plus_candidate_rank", kind="select", options=["base_plus_candidate_rank"]),
                field(
                    "powheg",
                    "run_mode",
                    "Run mode",
                    "dry_run",
                    kind="select",
                    options=["dry_run", "real_smoke", "real_free"],
                    help_text=(
                        "dry_run: prepares POWHEG cards only; pwhg_main is not executed. "
                        "real_smoke: safety mode; runs only the top candidate with at most 2 events. "
                        "real_free: production mode; runs pwhg_main for the selected number of interaction candidates and events per interaction."
                    ),
                ),
            ],
        },
        {
            "tab": "Observer Image Branches",
            "fields": [
                field(
                    "observer_image_branches",
                    "branch_scoring_model",
                    "Branch scoring model",
                    "ray_count_closeness_compactness_proxy",
                    kind="select",
                    options=["ray_count_closeness_compactness_proxy"],
                    help_text="Scores each observed image branch with ray-count, closest-approach, and compactness proxies.",
                ),
                field(
                    "observer_image_branches",
                    "primary_branch_selection_model",
                    "Primary branch selection",
                    "argmax_branch_score",
                    kind="select",
                    options=["argmax_branch_score"],
                    help_text="Selects the dominant observed image branch by maximum proxy branch score.",
                ),
                field("observer_image_branches", "minimum_branch_rays", "Minimum rays per branch", 1, kind="number", minimum=1, step=1),
                field(
                    "observer_image_branches",
                    "status",
                    "Status",
                    "configured_only_no_branch_analysis",
                    kind="select",
                    options=["configured_only_no_branch_analysis", "observer_image_branches_analyzed"],
                    visibility="INTERNAL",
                ),
            ],
        },
        {
            "tab": "Event Generation",
            "fields": [
                field("event_generation", "mode", "Event generation", "placeholder_disabled", kind="select", options=["placeholder_disabled"]),
                field("event_generation", "planned_model", "Planned model", "POWHEG_PYTHIA_future", kind="select", options=["POWHEG_PYTHIA_future"]),
            ],
        },
        {
            "tab": "GEANT4",
            "fields": [
                field("geant4", "mode", "GEANT4", "placeholder_disabled", kind="select", options=["placeholder_disabled"]),
                field("geant4", "planned_model", "Planned model", "detector_transport_future", kind="select", options=["detector_transport_future"]),
            ],
        },
        {
            "tab": "Photon Transport",
            "fields": [
                field("photon_transport", "mode", "Photon transport", "placeholder_disabled", kind="select", options=["placeholder_disabled"]),
                field("photon_transport", "planned_model", "Planned model", "radiative_transfer_future", kind="select", options=["radiative_transfer_future"]),
            ],
        },
        {
            "tab": "Spectra",
            "fields": [
                field("spectra", "mode", "Spectra", "placeholder_disabled", kind="select", options=["placeholder_disabled"]),
                field("spectra", "planned_model", "Planned model", "spectral_products_future", kind="select", options=["spectral_products_future"]),
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


def _split_resolution(value: Any) -> tuple[int, int] | None:
    text = str(value or "").strip().lower()
    if "x" not in text:
        return None
    left, right = text.split("x", 1)
    try:
        nx = int(float(left))
        ny = int(float(right))
    except ValueError:
        return None
    if nx <= 0 or ny <= 0:
        return None
    return nx, ny


def _normalize_camera_pixels(values: dict[str, Any], raw: dict[str, Any] | None = None) -> dict[str, Any]:
    camera = values.setdefault("observer_camera", {})
    raw_camera = (raw or {}).get("observer_camera", {}) if isinstance(raw, dict) else {}
    if "pixel_width" not in raw_camera or "pixel_height" not in raw_camera:
        parsed = _split_resolution(camera.get("resolution"))
        if parsed is not None:
            camera["pixel_width"], camera["pixel_height"] = parsed
    try:
        camera["resolution"] = f"{int(float(camera['pixel_width']))}x{int(float(camera['pixel_height']))}"
    except (KeyError, TypeError, ValueError):
        pass
    return values


def load_values(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return _normalize_camera_pixels(defaults())
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _normalize_camera_pixels(deep_update(defaults(), raw), raw)


def validate_values(values: dict[str, dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    bh = values["black_hole"]
    cam = values["observer_camera"]
    torus = values["analytic_torus"]
    cone = values["polar_cone"]
    source = values["uhe_neutrino_source"]
    forward = values["forward_geodesics"]
    dis = values.get("dis_interaction_sampler", {})
    bridge = values.get("observer_bridge", {})
    branches = values.get("observer_image_branches", {})
    powheg = values.get("powheg", {})

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
    try:
        if int(float(cam.get("pixel_width", 0))) <= 0:
            problems.append("observer_camera.pixel_width must be positive")
        if int(float(cam.get("pixel_height", 0))) <= 0:
            problems.append("observer_camera.pixel_height must be positive")
    except (TypeError, ValueError):
        problems.append("observer_camera pixel dimensions must be numeric")
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
    if str(source.get("direction_model")) not in {"coordinate_radial_outward", "isotropic_local"}:
        problems.append("uhe_neutrino_source.direction_model must be coordinate_radial_outward or isotropic_local in H3-W5")
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
    if str(forward.get("geodesic_backend")) != "cpp_hadros_original_port":
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
    if str(dis.get("medium_model", "analytic_torus")) != "analytic_torus":
        problems.append("dis_interaction_sampler.medium_model must be analytic_torus in H3-W7")
    if str(dis.get("medium_velocity_model", "zamo_fallback")) not in {"zamo_fallback", "static"}:
        problems.append("dis_interaction_sampler.medium_velocity_model must be zamo_fallback or static in H3-W7")
    if float(dis.get("density_floor_g_cm3", 0.0)) < 0.0:
        problems.append("dis_interaction_sampler.density_floor_g_cm3 must be non-negative")
    if str(dis.get("dis_backend", "cpp_hadros_original_port")) not in {"cpp_hadros_original_port", "python_prototype"}:
        problems.append("dis_interaction_sampler.dis_backend must be cpp_hadros_original_port or python_prototype")
    if str(dis.get("dis_model", "GBW")) not in {"GBW", "IIM"}:
        problems.append("dis_interaction_sampler.dis_model must be GBW or IIM")
    if str(dis.get("interaction_sampling_mode", "optical_depth_inverse_cdf")) != "optical_depth_inverse_cdf":
        problems.append("dis_interaction_sampler.interaction_sampling_mode must be optical_depth_inverse_cdf")
    if int(float(dis.get("max_interactions", 0))) <= 0:
        problems.append("dis_interaction_sampler.max_interactions must be positive")
    try:
        int(float(dis.get("random_seed", 0)))
    except (TypeError, ValueError):
        problems.append("dis_interaction_sampler.random_seed must be an integer")
    if str(bridge.get("observer_bridge_backend", "cpp_cpu")) != "cpp_cpu":
        problems.append("observer_bridge.observer_bridge_backend must be cpp_cpu in H3-W8")
    if str(bridge.get("bridge_mode", "scoring_only")) != "scoring_only":
        problems.append("observer_bridge.bridge_mode must be scoring_only in H3-W8")
    if str(bridge.get("secondary_particle_proxy_model", "geometric_escape_proxy")) != "geometric_escape_proxy":
        problems.append("observer_bridge.secondary_particle_proxy_model must be geometric_escape_proxy in H3-W8")
    if str(bridge.get("escape_proxy_model", "geometric_outward_proxy")) != "geometric_outward_proxy":
        problems.append("observer_bridge.escape_proxy_model must be geometric_outward_proxy in H3-W8")
    if str(bridge.get("visibility_model", "geometric_proxy")) != "geometric_proxy":
        problems.append("observer_bridge.visibility_model must be geometric_proxy in H3-W8")
    if str(bridge.get("redshift_proxy_model", "unity_or_metric_proxy")) != "unity_or_metric_proxy":
        problems.append("observer_bridge.redshift_proxy_model must be unity_or_metric_proxy in H3-W8")
    if str(bridge.get("line_of_sight_proxy_model", "geometric_proxy")) != "geometric_proxy":
        problems.append("observer_bridge.line_of_sight_proxy_model must be geometric_proxy in H3-W8")
    if str(bridge.get("fov_policy", "hard")) not in {"hard", "soft"}:
        problems.append("observer_bridge.fov_policy must be hard or soft")
    if int(float(bridge.get("max_ranked_events", 0))) <= 0:
        problems.append("observer_bridge.max_ranked_events must be positive")
    if float(bridge.get("min_observer_weight", 0.0)) < 0.0:
        problems.append("observer_bridge.min_observer_weight must be non-negative")
    if float(bridge.get("min_final_observation_score", 0.0)) < 0.0:
        problems.append("observer_bridge.min_final_observation_score must be non-negative")
    if str(bridge.get("downstream_selection_policy", "top_n")) not in {"all_candidates", "top_n", "score_threshold"}:
        problems.append("observer_bridge.downstream_selection_policy must be all_candidates, top_n, or score_threshold")
    if int(float(bridge.get("downstream_top_n_candidates", 50))) <= 0:
        problems.append("observer_bridge.downstream_top_n_candidates must be positive")
    if float(bridge.get("downstream_min_final_observation_score", 0.0)) < 0.0:
        problems.append("observer_bridge.downstream_min_final_observation_score must be non-negative")
    if str(bridge.get("candidate_overlay_mapping", "kerr_pixel_match")) not in {"kerr_pixel_match", "geometric_proxy"}:
        problems.append("observer_bridge.candidate_overlay_mapping must be kerr_pixel_match or geometric_proxy")
    if int(float(bridge.get("kerr_pixel_match_resolution_x", 32))) <= 0:
        problems.append("observer_bridge.kerr_pixel_match_resolution_x must be positive")
    if int(float(bridge.get("kerr_pixel_match_resolution_y", 18))) <= 0:
        problems.append("observer_bridge.kerr_pixel_match_resolution_y must be positive")
    if float(bridge.get("kerr_pixel_match_tolerance_rg", 3.5)) < 0.0:
        problems.append("observer_bridge.kerr_pixel_match_tolerance_rg must be non-negative")
    if float(bridge.get("candidate_matching_radius_rg", 3.5)) < 0.0:
        problems.append("observer_bridge.candidate_matching_radius_rg must be non-negative")
    if int(float(bridge.get("multi_image_audit_resolution_x", 48))) <= 0:
        problems.append("observer_bridge.multi_image_audit_resolution_x must be positive")
    if int(float(bridge.get("multi_image_audit_resolution_y", 27))) <= 0:
        problems.append("observer_bridge.multi_image_audit_resolution_y must be positive")
    if str(bridge.get("kerr_pixel_match_basis_transform", "cuda_preview_local_tetrad")) not in {"cuda_preview_local_tetrad", "up_flipped", "right_flipped", "up_right_flipped"}:
        problems.append("observer_bridge.kerr_pixel_match_basis_transform is unsupported")
    if int(float(bridge.get("interactive_max_candidates", 40))) <= 0:
        problems.append("observer_bridge.interactive_max_candidates must be positive")
    if int(float(bridge.get("interactive_max_rays", 64))) < 0:
        problems.append("observer_bridge.interactive_max_rays must be non-negative")
    if int(float(bridge.get("interactive_ray_stride", 4))) <= 0:
        problems.append("observer_bridge.interactive_ray_stride must be positive")
    if str(bridge.get("interactive_candidate_color_mode", "final_observation_score")) not in {"final_observation_score", "closest_approach_rg", "inside_outside_fov"}:
        problems.append("observer_bridge.interactive_candidate_color_mode is unsupported")
    if str(branches.get("branch_scoring_model", "ray_count_closeness_compactness_proxy")) != "ray_count_closeness_compactness_proxy":
        problems.append("observer_image_branches.branch_scoring_model must be ray_count_closeness_compactness_proxy in H3-W8b")
    if str(branches.get("primary_branch_selection_model", "argmax_branch_score")) != "argmax_branch_score":
        problems.append("observer_image_branches.primary_branch_selection_model must be argmax_branch_score in H3-W8b")
    if int(float(branches.get("minimum_branch_rays", 1))) <= 0:
        problems.append("observer_image_branches.minimum_branch_rays must be positive")
    if str(powheg.get("powheg_backend", "local_powheg")) != "local_powheg":
        problems.append("powheg.powheg_backend must be local_powheg in H3-W9a")
    if str(powheg.get("powheg_process", "nudis")) != "nudis":
        problems.append("powheg.powheg_process must be nudis in H3-W9a")
    if "max_powheg_events" in powheg and int(float(powheg.get("max_powheg_events", 50))) <= 0:
        problems.append("powheg.max_powheg_events must be positive")
    if int(float(powheg.get("events_per_candidate", 1))) <= 0:
        problems.append("powheg.events_per_candidate must be positive")
    try:
        int(float(powheg.get("random_seed", 0)))
    except (TypeError, ValueError):
        problems.append("powheg.random_seed must be an integer")
    if str(powheg.get("powheg_seed_mode", "base_plus_candidate_rank")) != "base_plus_candidate_rank":
        problems.append("powheg.powheg_seed_mode must be base_plus_candidate_rank in H3-W9a")
    if "min_final_observation_score" in powheg and float(powheg.get("min_final_observation_score", 0.0)) < 0.0:
        problems.append("powheg.min_final_observation_score must be non-negative")
    if str(powheg.get("run_mode", "dry_run")) not in {"dry_run", "real_smoke", "real_free"}:
        problems.append("powheg.run_mode must be dry_run, real_smoke, or real_free")
    return problems


def flatten_for_legacy_camera(values: dict[str, dict[str, Any]]) -> dict[str, Any]:
    camera = values["observer_camera"]
    bh = values["black_hole"]
    torus = values["analytic_torus"]
    camera_resolution = f"{int(float(camera.get('pixel_width', 512)))}x{int(float(camera.get('pixel_height', 288)))}"
    return {
        "black_hole_mass_msun": float(bh["mass_msun"]),
        "spin": float(bh["spin_a"]),
        "camera_theta_deg": float(camera["inclination_deg"]),
        "camera_fov_deg": float(camera["field_of_view_deg"]),
        "camera_r_obs_rg": float(camera["observer_distance_rg"]),
        "camera_resolution": camera_resolution,
        "preview_resolution": str(camera["preview_resolution"]),
        "torus_r0_rg": float(torus["r_peak_rg"]),
        "torus_r_inner_rg": float(torus["r_inner_rg"]),
        "torus_r_outer_rg": float(torus["r_outer_rg"]),
        "torus_half_opening_angle_deg": float(torus["half_opening_angle_deg"]),
    }
