from __future__ import annotations

import json
from pathlib import Path

from hadros3.config import defaults, parse_latex_number, schema, validate_values
from hadros3.pipeline import render_hadros_web
from hadros_web import dashboard_payload, render_html


def test_schema_exposes_hadros3_first_stage_controls() -> None:
    keys = {(field["section"], field["key"]) for tab in schema() for field in tab["fields"]}
    expected = {
        ("black_hole", "mass_msun"),
        ("black_hole", "spin_a"),
        ("observer_camera", "observer_distance_rg"),
        ("observer_camera", "camera_preview_mode"),
        ("observer_camera", "inclination_deg"),
        ("observer_camera", "field_of_view_deg"),
        ("analytic_torus", "r_inner_rg"),
        ("polar_cone", "opening_angle_deg"),
        ("uhe_neutrino_source", "energy_gev"),
        ("uhe_neutrino_source", "direction_model"),
        ("uhe_neutrino_source", "direction_opening_angle_deg"),
        ("uhe_neutrino_source", "direction_seed"),
        ("forward_geodesics", "geodesic_backend"),
        ("forward_geodesics", "n_samples_to_propagate"),
        ("interaction_sampler", "mode"),
        ("observer_bridge", "mode"),
        ("provenance", "trust_boundary"),
    }
    assert expected <= keys


def test_default_config_is_valid() -> None:
    assert validate_values(defaults()) == []


def test_versioned_hadros_web_preset_is_valid() -> None:
    values = json.loads(Path("presets/hadros_web/default_config.json").read_text(encoding="utf-8"))
    assert validate_values(values) == []


def test_uhe_energy_accepts_latex_power_notation() -> None:
    assert parse_latex_number("10^{12}") == 1.0e12
    assert parse_latex_number("3\\times10^{12}") == 3.0e12
    values = defaults()
    values["uhe_neutrino_source"]["energy_gev"] = "10^{12}"
    assert validate_values(values) == []


def test_uhe_source_theta_range_must_be_ordered() -> None:
    values = defaults()
    values["uhe_neutrino_source"]["theta_min_deg"] = 10.0
    values["uhe_neutrino_source"]["theta_max_deg"] = 9.0
    problems = validate_values(values)
    assert any("theta_min_deg < theta_max_deg" in problem for problem in problems)


def test_uhe_source_theta_max_must_fit_inside_funnel() -> None:
    values = defaults()
    values["polar_cone"]["opening_angle_deg"] = 20.0
    values["uhe_neutrino_source"]["theta_min_deg"] = 0.0
    values["uhe_neutrino_source"]["theta_max_deg"] = 25.0
    problems = validate_values(values)
    assert "uhe_neutrino_source.theta_max_deg must be <= polar_cone.opening_angle_deg" in problems


def test_render_hadros_web_writes_first_stage_products(tmp_path: Path) -> None:
    values = defaults()
    summary = render_hadros_web(values, root=Path.cwd(), output_dir=tmp_path)
    assert summary["validation"]["expensive_event_generation_invoked"] is False
    for key in [
        "config",
        "geometry_preview",
        "system_schematic",
        "camera_preview",
        "camera_preview_summary",
        "provenance",
        "html_summary",
        "render_summary",
    ]:
        assert key in summary["products"]
        assert Path(summary["products"][key]).exists()

    provenance = json.loads(Path(summary["products"]["provenance"]).read_text(encoding="utf-8"))
    assert provenance["status"] == "geometry_configured_no_expensive_events"
    assert provenance["disabled_expensive_or_future_stages"]["powheg"] == "disabled"
    assert provenance["camera_preview"]["requested_mode"] in {"analytic_geometry_only", "kerr_like_cuda", "full_kerr"}


def test_forward_geodesics_dashboard_integration_is_separate_from_uhe_source() -> None:
    values = defaults()
    payload = dashboard_payload(values, Path("presets/hadros_web/default_config.json"))
    html = render_html(values, Path("presets/hadros_web/default_config.json"))

    assert payload["source_status"]["input_dir"] == "UHEsource"
    assert payload["forward_geodesics_status"]["input_uhe_source_dir"] == "UHEsource"
    assert payload["forward_geodesics_status"]["output_dir"] == "ForwardGeodesics"
    assert payload["outputs"]["paths"]["forward_preview"] == "ForwardGeodesics/uhe_neutrino_forward_preview.png"
    assert payload["outputs"]["paths"]["forward_summary_json"] == "ForwardGeodesics/uhe_neutrino_forward_summary.json"

    assert "Forward Geodesics" in html
    assert "Propagate Forward Geodesics" in html
    assert "Initial Direction" in html
    assert "Coordinate radial outward" in html
    assert "Jet axis" in html
    assert "Cone emission" in html
    assert "The UHE source samples emission position, energy and direction." in html
    assert "The Kerr four-momentum is not sampled here" in html
    assert "Uses</strong>position + energy + emission_direction" in html
    assert "Builds</strong>Kerr null four-momentum" in html
    assert "Input UHEsource/" in html
    assert "ForwardGeodesics/" in html
    assert "uhe_neutrino_forward_preview.png" in html
    assert "uhe_neutrino_forward_summary.json" in html
    assert "uhe_neutrino_forward_summary.csv" in html
    assert "uhe_neutrino_forward_paths.jsonl" in html
    assert "uhe_neutrino_forward_path_segments.jsonl" in html
    assert "geodesic_validation_report.json" in html
    assert "stop_condition_statistics.csv" in html
    assert "uhe_neutrino_source_preview.png" in html
