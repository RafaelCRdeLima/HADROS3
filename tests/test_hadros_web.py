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
        ("dis_interaction_sampler", "dis_model"),
        ("dis_interaction_sampler", "dis_backend"),
        ("dis_interaction_sampler", "medium_model"),
        ("dis_interaction_sampler", "medium_velocity_model"),
        ("observer_bridge", "observer_bridge_backend"),
        ("observer_bridge", "bridge_mode"),
        ("powheg", "powheg_backend"),
        ("powheg", "ranking_policy"),
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


def test_geometry_preview_context_is_available_on_geometry_tabs() -> None:
    html = render_html(defaults(), Path("presets/hadros_web/default_config.json"))

    assert 'const geometryPreviewTabs = new Set(["Camera", "Black Hole", "Torus / Medium", "Funnel / Cone"]);' in html
    assert "geometryPreviewTabs.has(activeTab)" in html
    assert "DIS Interaction Map" in html
    assert "Observer Camera Overlay" in html
    assert "Forward Geodesics Geometry" in html
    assert "UHE Source Samples" in html


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
    assert provenance["camera_preview"]["medium_renderer_used"] is False
    assert provenance["theory_document"] == "docs/Theory/HADROS3_Physics_Theory.pdf"
    assert provenance["theory_version"] == "1.0"
    assert provenance["theory_commit"]
    assert provenance["theory_generation_date"]
    assert provenance["scientific_theory"]["theory_pipeline_version"] == "H3-W9a"
    version_payload = json.loads(Path("VERSION.json").read_text(encoding="utf-8"))
    for key in ["software_version", "physics_version", "pipeline_version", "theory_version"]:
        assert key in version_payload
        assert key in provenance["scientific_release"]
    assert provenance["scientific_release"]["software_version"] == version_payload["software_version"]
    assert provenance["scientific_release"]["physics_version"] == version_payload["physics_version"]
    assert provenance["scientific_release"]["pipeline_version"] == version_payload["pipeline_version"]
    assert provenance["scientific_release"]["theory_version"] == version_payload["theory_version"]
    assert provenance["scientific_release"]["theory_document"] == version_payload["theory_document"]
    assert provenance["scientific_release"]["git_commit"]


def test_forward_geodesics_dashboard_integration_is_separate_from_uhe_source(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("hadros_web.ROOT", tmp_path)
    values = defaults()
    payload = dashboard_payload(values, Path("presets/hadros_web/default_config.json"))
    html = render_html(values, Path("presets/hadros_web/default_config.json"))

    assert payload["source_status"]["input_dir"] == "UHEsource"
    assert payload["forward_geodesics_status"]["input_uhe_source_dir"] == "UHEsource"
    assert payload["forward_geodesics_status"]["output_dir"] == "ForwardGeodesics"
    assert payload["outputs"]["paths"]["forward_preview"] == "ForwardGeodesics/uhe_neutrino_forward_preview.png"
    assert payload["outputs"]["paths"]["forward_geometry_3d"] == "ForwardGeodesics/uhe_neutrino_forward_geometry_3d.png"
    assert payload["outputs"]["paths"]["forward_geometry_3d_json"] == "ForwardGeodesics/uhe_neutrino_forward_geometry_3d.json"
    assert payload["outputs"]["paths"]["forward_geometry_3d_html"] == "ForwardGeodesics/uhe_neutrino_forward_geometry_3d.html"
    assert payload["outputs"]["paths"]["forward_summary_json"] == "ForwardGeodesics/uhe_neutrino_forward_summary.json"
    assert payload["dis_interaction_sampler"]["input_uhe_source_found"] is False
    assert payload["dis_interaction_sampler"]["input_forward_geodesics_found"] is False
    assert payload["outputs"]["paths"]["dis_tau_preview"] == "DIS/dis_tau_preview.png"
    assert payload["outputs"]["paths"]["dis_interaction_locations"] == "DIS/dis_interaction_locations.png"
    assert payload["outputs"]["paths"]["tau_distribution"] == "DIS/tau_distribution.png"
    assert payload["outputs"]["paths"]["interaction_probability_distribution"] == "DIS/interaction_probability_distribution.png"
    assert payload["outputs"]["paths"]["optical_depth_map"] == "DIS/optical_depth_map.png"
    assert payload["outputs"]["paths"]["optical_depth_map_3d_html"] == "DIS/optical_depth_map_3d.html"
    assert payload["outputs"]["paths"]["medium_density_map"] == "DIS/medium_density_map.png"
    assert payload["outputs"]["paths"]["interaction_location_distribution"] == "DIS/interaction_location_distribution.png"
    assert payload["outputs"]["paths"]["local_energy_distribution"] == "DIS/local_energy_distribution.png"
    assert payload["outputs"]["paths"]["local_density_distribution"] == "DIS/local_density_distribution.png"
    assert payload["outputs"]["paths"]["sigma_distribution"] == "DIS/sigma_distribution.png"
    assert payload["outputs"]["paths"]["density_energy_sigma_correlation"] == "DIS/density_energy_sigma_correlation.png"
    assert payload["outputs"]["paths"]["dis_diagnostics_report"] == "DIS/dis_diagnostics_report.json"
    assert payload["outputs"]["paths"]["gbw_vs_iim_summary"] == "DIS/gbw_vs_iim_summary.json"
    assert payload["outputs"]["paths"]["observer_bridge_candidates"] == "ObserverBridge/observer_bridge_candidates.jsonl"
    assert payload["outputs"]["paths"]["observer_bridge_ranked_events"] == "ObserverBridge/observer_bridge_ranked_events.jsonl"
    assert payload["outputs"]["paths"]["observer_bridge_summary_json"] == "ObserverBridge/observer_bridge_summary.json"
    assert payload["outputs"]["paths"]["observer_bridge_summary"] == "ObserverBridge/observer_bridge_summary.csv"
    assert payload["outputs"]["paths"]["observer_bridge_report"] == "ObserverBridge/observer_bridge_report.json"
    assert payload["outputs"]["paths"]["observer_bridge_map"] == "ObserverBridge/observer_bridge_map.png"
    assert payload["outputs"]["paths"]["observer_bridge_score_distribution"] == "ObserverBridge/observer_bridge_score_distribution.png"
    assert payload["outputs"]["paths"]["observer_bridge_weight_breakdown"] == "ObserverBridge/observer_bridge_weight_breakdown.png"
    assert payload["outputs"]["paths"]["observer_bridge_visibility_map"] == "ObserverBridge/observer_bridge_visibility_map.png"
    assert payload["outputs"]["paths"]["observer_bridge_ranked_events_png"] == "ObserverBridge/observer_bridge_ranked_events.png"
    assert payload["outputs"]["paths"]["observer_bridge_geometry_3d_html"] == "ObserverBridge/observer_bridge_geometry_3d.html"
    assert payload["outputs"]["paths"]["observer_bridge_camera_view"] == "ObserverBridge/observer_bridge_camera_view.png"
    assert payload["outputs"]["paths"]["observer_bridge_camera_overlay"] == "ObserverBridge/observer_bridge_camera_overlay.png"
    assert payload["outputs"]["paths"]["powheg_event_requests"] == "POWHEG/powheg_event_requests.jsonl"
    assert payload["outputs"]["paths"]["powheg_summary_json"] == "POWHEG/powheg_summary.json"
    assert payload["outputs"]["paths"]["powheg_summary"] == "POWHEG/powheg_summary.csv"
    assert payload["outputs"]["paths"]["powheg_report"] == "POWHEG/powheg_report.json"
    assert payload["outputs"]["paths"]["powheg_card_preview"] == "POWHEG/powheg_card_preview.png"
    assert payload["outputs"]["paths"]["powheg_energy_distribution"] == "POWHEG/powheg_energy_distribution.png"
    assert payload["outputs"]["paths"]["powheg_job_summary"] == "POWHEG/powheg_job_summary.png"
    assert payload["outputs"]["paths"]["dis_summary_json"] == "DIS/dis_summary.json"
    assert payload["outputs"]["paths"]["uhe_source_sampling_uniformity"] == "UHEsource/uhe_source_sampling_uniformity.png"
    assert payload["outputs"]["paths"]["uhe_source_sampling_uniformity_report"] == "UHEsource/uhe_source_sampling_uniformity_report.json"
    assert payload["outputs"]["paths"]["uhe_source_direction_uniformity"] == "UHEsource/uhe_source_direction_uniformity.png"
    assert payload["outputs"]["paths"]["uhe_source_direction_uniformity_report"] == "UHEsource/uhe_source_direction_uniformity_report.json"
    assert payload["outputs"]["paths"]["uhe_source_direction_sphere"] == "UHEsource/uhe_source_direction_sphere.png"
    assert payload["values"]["forward_geodesics"]["geodesic_backend"] == "cpp_hadros_original_port"

    assert "Forward Geodesics" in html
    assert "DIS Interaction Sampler" in html
    assert "Propagate Forward Geodesics" in html
    assert "Requested paths" in html
    assert "Propagated paths" in html
    assert "Available UHE samples" in html
    assert "Compute DIS Optical Depth / Sample Interactions" in html
    assert "Compute Observer Bridge Scores" in html
    assert "Prepare POWHEG Jobs" in html
    assert "Dry Run" in html
    assert "pwhg_main NOT executed" in html
    assert "cpp_hadros_original_port" in html
    assert "Full Kerr null geodesic propagation" in html
    assert "Initial Direction" in html
    assert "Isotropic local" in html
    assert "Coordinate radial outward" in html
    assert "Jet axis" in html
    assert "Cone emission" in html
    assert "Recommended physical model" in html
    assert "The UHE source samples emission position, energy and direction." in html
    assert "The Kerr four-momentum is not sampled here" in html
    assert "Uses</strong>position + energy + emission_direction" in html
    assert "Builds</strong>Kerr null four-momentum" in html
    assert "Input UHEsource/" in html
    assert "ForwardGeodesics/" in html
    assert "uhe_neutrino_forward_geometry_3d.html" in html
    assert "uhe_neutrino_forward_geometry_3d.png" in html
    assert "uhe_neutrino_forward_geometry_3d.json" in html
    assert "uhe_neutrino_forward_preview.png" in html
    assert "uhe_neutrino_forward_summary.json" in html
    assert "uhe_neutrino_forward_summary.csv" in html
    assert "uhe_neutrino_forward_paths.jsonl" in html
    assert "uhe_neutrino_forward_path_segments.jsonl" in html
    assert "uhe_source_sampling_uniformity.png" in html
    assert "uhe_source_sampling_uniformity_report.json" in html
    assert "uhe_source_direction_uniformity.png" in html
    assert "uhe_source_direction_uniformity_report.json" in html
    assert "uhe_source_direction_sphere.png" in html
    assert "geodesic_validation_report.json" in html
    assert "stop_condition_statistics.csv" in html
    assert "uhe_neutrino_source_preview.png" in html
    assert "dis_tau_preview.png" in html
    assert "dis_interaction_locations.png" in html
    assert "tau_distribution.png" in html
    assert "interaction_probability_distribution.png" in html
    assert "optical_depth_map.png" in html
    assert "medium_density_map.png" in html
    assert "The analytic torus has a hard radial cut and a Gaussian angular profile; the opening angle is a width parameter, not a hard boundary." in html
    assert "interaction_location_distribution.png" in html
    assert "local_energy_distribution.png" in html
    assert "local_density_distribution.png" in html
    assert "sigma_distribution.png" in html
    assert "density_energy_sigma_correlation.png" in html
    assert "dis_diagnostics_report.json" in html
    assert "Compare GBW vs IIM" in html
    assert "sigma_table_path" in html
    assert "sigma_table_rows" in html
    assert "sigma_table_is_compact_builtin_adapter" in html
    assert "sigma_table_physics_risk" in html
    assert "sigma_table_energy_min_gev" in html
    assert "sigma_table_energy_max_gev" in html
    assert "dis_path_optical_depths.jsonl" in html
    assert "dis_interaction_candidates.jsonl" in html
    assert "dis_accepted_interactions.jsonl" in html
    assert "dis_optical_depth_report.json" in html
    assert "ObserverBridge/" in html
    assert "observer_bridge_candidates.jsonl" in html
    assert "observer_bridge_ranked_events.jsonl" in html
    assert "observer_bridge_summary.json" in html
    assert "observer_bridge_summary.csv" in html
    assert "observer_bridge_report.json" in html
    assert "observer_bridge_map.png" in html
    assert "observer_bridge_score_distribution.png" in html
    assert "observer_bridge_weight_breakdown.png" in html
    assert "observer_bridge_visibility_map.png" in html
    assert "observer_bridge_ranked_events.png" in html
    assert "observer_bridge_geometry_3d.html" in html
    assert "observer_bridge_camera_view.png" in html
    assert "observer_bridge_camera_overlay.png" in html
    assert "POWHEG/" in html
    assert "powheg_event_requests.jsonl" in html
    assert "powheg_summary.json" in html
    assert "powheg_summary.csv" in html
    assert "powheg_report.json" in html
    assert "powheg_card_preview.png" in html
    assert "powheg_energy_distribution.png" in html
    assert "powheg_job_summary.png" in html
    assert "Observer Camera Overlay" in html
    assert "overlay resolution" in html
    assert "camera_preview_pixel_plane" in html
    assert "Observer Camera View" in html
