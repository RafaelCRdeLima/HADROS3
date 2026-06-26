from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hadros3.config import defaults
from hadros3.dis_sampler import SigmaNuNProvider, analytic_torus_density_g_cm3, constant_density_tau, generate_dis_interaction_products, generate_gbw_iim_comparison, interaction_probability
from hadros3.forward_geodesics import generate_forward_geodesic_products
from hadros3.medium_renderer import MediumRenderer
from hadros3.uhe_source import generate_uhe_source_products


def _values() -> dict:
    values = defaults()
    values["black_hole"]["spin_a"] = 0.9
    values["analytic_torus"].update(
        {
            "r_inner_rg": 2.5,
            "r_outer_rg": 28.0,
            "r_peak_rg": 8.0,
            "half_opening_angle_deg": 70.0,
            "density_norm_g_cm3": 1.0e13,
        }
    )
    values["polar_cone"]["opening_angle_deg"] = 20.0
    values["uhe_neutrino_source"].update(
        {
            "direction_model": "isotropic_local",
            "energy_gev": "10^{9}",
            "r_min_rg": 3.0,
            "r_max_rg": 4.5,
            "theta_min_deg": 1.0,
            "theta_max_deg": 12.0,
            "n_samples": 8,
            "random_seed": 1122,
            "direction_seed": 3344,
        }
    )
    values["forward_geodesics"].update(
        {
            "n_samples_to_propagate": 5,
            "initial_step_rg": 1.0,
            "max_steps": 32,
            "outer_radius_rg": 30.0,
        }
    )
    values["dis_interaction_sampler"].update(
        {
            "dis_backend": "python_prototype",
            "dis_model": "GBW",
            "medium_model": "analytic_torus",
            "medium_velocity_model": "zamo_fallback",
            "density_floor_g_cm3": 0.0,
            "random_seed": 24680,
        }
    )
    return values


def test_constant_density_tau_matches_analytic_case() -> None:
    n_baryon = 2.0e34
    sigma = 1.0e-33
    length = 5.0e6
    assert constant_density_tau(n_baryon, sigma, length) == n_baryon * sigma * length
    assert interaction_probability(0.0) == 0.0
    assert 0.0 < interaction_probability(0.5) < 1.0


def test_sigma_provider_reads_original_hadros_tables() -> None:
    gbw = SigmaNuNProvider("GBW")
    iim = SigmaNuNProvider("IIM")
    assert str(gbw.table_path) == "data/sigma/sigma_nuN_CC_GBW.dat"
    assert str(iim.table_path) == "data/sigma/sigma_nuN_CC_IIM.dat"
    assert len(gbw.table) == 300
    assert len(iim.table) == 300
    assert gbw.energy_min_gev == 1.0e3
    assert gbw.energy_max_gev == 1.0e14
    assert math.isclose(gbw.sigma_cm2(1.0e3), 2.6022e-35)
    assert math.isclose(iim.sigma_cm2(1.0e3), 9.45275e-38)


def test_medium_renderer_density_matches_dis_sampler_python_model() -> None:
    values = _values()
    torus = values["analytic_torus"]
    r_inner = float(torus["r_inner_rg"])
    r_outer = float(torus["r_outer_rg"])
    r_peak = float(torus["r_peak_rg"])
    theta_width = math.radians(float(torus["half_opening_angle_deg"]))

    for radius in [r_inner, r_peak, r_outer, 0.5 * (r_inner + r_outer)]:
        for theta in [0.5 * math.pi, 0.5 * math.pi + theta_width, 0.5 * math.pi + 2.0 * theta_width]:
            assert MediumRenderer.density(radius, theta, values) == analytic_torus_density_g_cm3(radius, theta, values)

    assert MediumRenderer.density(r_inner - 1.0e-4, 0.5 * math.pi, values) == 0.0
    assert MediumRenderer.density(r_outer + 1.0e-4, 0.5 * math.pi, values) == 0.0
    assert MediumRenderer.density(r_peak, 0.5 * math.pi + 2.0 * theta_width, values) > 0.0
    assert MediumRenderer.metadata()["density_model_theta_is_hard_cut"] is False


def test_dis_sampler_consumes_source_and_forward_outputs(tmp_path: Path) -> None:
    values = _values()
    generate_uhe_source_products(values, output_dir=tmp_path)
    generate_forward_geodesic_products(values, run_output_dir=tmp_path)
    summary = generate_dis_interaction_products(values, run_output_dir=tmp_path)

    assert summary["optical_depth_dis_sampler_invoked"] is True
    assert summary["observer_bridge_active_filter_invoked"] is False
    assert summary["expensive_event_generation_invoked"] is False
    assert summary["powheg_invoked"] is False
    assert summary["pythia_invoked"] is False
    assert summary["geant4_invoked"] is False
    assert summary["dis_backend"] == "python_prototype"
    assert summary["backend_language"] == "Python"
    assert summary["python_prototype_used"] is True
    assert summary["cpp_backend_used"] is False
    assert summary["uses_hadros_original_runtime_path"] is False
    assert summary["dis_model"] == "GBW"
    assert summary["medium_model"] == "analytic_torus"
    assert summary["medium_velocity_model"] == "zamo_fallback"
    assert summary["medium_velocity_physics_risk"] is True
    assert summary["sigma_table_path"] == "data/sigma/sigma_nuN_CC_GBW.dat"
    assert summary["sigma_table_rows"] == 300
    assert summary["sigma_table_is_compact_builtin_adapter"] is False
    assert summary["sigma_table_physics_risk"] is False
    assert summary["sigma_table_energy_min_gev"] == 1.0e3
    assert summary["sigma_table_energy_max_gev"] == 1.0e14
    assert summary["n_paths_processed"] == 5
    assert summary["n_segments_processed"] > 0
    assert summary["tau_min"] >= 0.0
    assert summary["tau_mean"] >= 0.0
    assert summary["tau_max"] >= summary["tau_min"]
    assert summary["max_density_g_cm3"] >= 0.0
    assert summary["max_sigma_cm2"] >= 0.0
    assert summary["max_d_tau"] >= 0.0

    dis_dir = tmp_path / "DIS"
    for filename in [
        "dis_path_optical_depths.jsonl",
        "dis_interaction_candidates.jsonl",
        "dis_accepted_interactions.jsonl",
        "dis_summary.csv",
        "dis_summary.json",
        "dis_tau_preview.png",
        "dis_interaction_locations.png",
        "dis_interaction_locations_3d.html",
        "dis_optical_depth_report.json",
        "tau_distribution.png",
        "interaction_probability_distribution.png",
        "optical_depth_map.png",
        "optical_depth_map_3d.html",
        "medium_density_map.png",
        "interaction_location_distribution.png",
        "local_energy_distribution.png",
        "local_density_distribution.png",
        "sigma_distribution.png",
        "density_energy_sigma_correlation.png",
        "dis_diagnostics_report.json",
    ]:
        assert (dis_dir / filename).exists()

    paths = [json.loads(line) for line in (dis_dir / "dis_path_optical_depths.jsonl").read_text(encoding="utf-8").splitlines()]
    assert paths
    for path in paths:
        assert path["tau_nuN_total"] >= 0.0
        assert 0.0 <= path["interaction_probability"] <= 1.0
        assert path["max_rho_g_cm3"] >= 0.0
        assert path["max_sigma_cm2"] >= 0.0
        assert path["max_d_tau"] >= 0.0

    report = json.loads((dis_dir / "dis_optical_depth_report.json").read_text(encoding="utf-8"))
    assert report["validations"]["rho_non_negative"] is True
    assert report["validations"]["sigma_non_negative"] is True
    assert report["validations"]["d_tau_non_negative"] is True
    assert report["validations"]["tau_non_negative"] is True
    assert report["validations"]["probability_bounds"] is True
    assert report["interaction_points_outside_medium"] == 0
    assert report["density_model_theta_is_hard_cut"] is False
    accepted = [json.loads(line) for line in (dis_dir / "dis_accepted_interactions.jsonl").read_text(encoding="utf-8").splitlines()]
    for interaction in accepted:
        assert interaction["interaction_point_density_checked"] is True
        assert interaction["interaction_point_inside_medium"] is True
        assert interaction["interaction_point_rho_g_cm3"] > 0.0
        assert interaction["interaction_rho_g_cm3"] > 0.0
        assert interaction["interaction_point_sampling_attempts"] >= 1
    locations_html = (dis_dir / "dis_interaction_locations_3d.html").read_text(encoding="utf-8")
    assert "hard radial shell" in locations_html
    assert "angular profile=Gaussian" in locations_html
    assert "angular rings are density levels, not boundaries" in locations_html
    assert "rho: p.rho, inside: p.inside" in locations_html
    assert "torusHalfHeight" not in locations_html
    diagnostics = json.loads((dis_dir / "dis_diagnostics_report.json").read_text(encoding="utf-8"))
    assert diagnostics["diagnostics_generated"] is True
    assert diagnostics["medium_density_map_generated"] is True
    assert diagnostics["medium_renderer_used"] is True
    assert diagnostics["density_model_has_hard_radial_cut"] is True
    assert diagnostics["density_model_theta_profile"] == "gaussian"
    assert diagnostics["density_model_theta_is_hard_cut"] is False
    assert diagnostics["half_opening_angle_interpretation"] == "gaussian_width_not_boundary"
    assert diagnostics["interaction_points_outside_medium"] == 0
    assert diagnostics["n_paths"] == 5
    assert diagnostics["n_segments"] > 0
    assert diagnostics["tau_statistics"]["tau_max"] >= diagnostics["tau_statistics"]["tau_min"]
    assert diagnostics["probability_statistics"]["probability_max"] <= 1.0


def test_dis_sampler_seed_is_reproducible(tmp_path: Path) -> None:
    values = _values()
    generate_uhe_source_products(values, output_dir=tmp_path)
    generate_forward_geodesic_products(values, run_output_dir=tmp_path)
    generate_dis_interaction_products(values, run_output_dir=tmp_path)
    first = (tmp_path / "DIS" / "dis_interaction_candidates.jsonl").read_text(encoding="utf-8")
    generate_dis_interaction_products(values, run_output_dir=tmp_path)
    second = (tmp_path / "DIS" / "dis_interaction_candidates.jsonl").read_text(encoding="utf-8")
    assert first == second


def test_cpp_dis_sampler_backend_matches_python_contract(tmp_path: Path) -> None:
    if not Path("bin/hadros3_dis_sampler").exists():
        pytest.skip("H3-W7 C++ DIS sampler is not built")
    values = _values()
    values["dis_interaction_sampler"]["dis_backend"] = "cpp_hadros_original_port"
    generate_uhe_source_products(values, output_dir=tmp_path)
    generate_forward_geodesic_products(values, run_output_dir=tmp_path)
    summary = generate_dis_interaction_products(values, run_output_dir=tmp_path)

    assert summary["dis_backend"] == "cpp_hadros_original_port"
    assert summary["backend_language"] == "C++17"
    assert summary["backend_executable"] == "bin/hadros3_dis_sampler"
    assert summary["backend_kind"] == "ported_hadros_cpp_dis_optical_depth_sampler"
    assert summary["cpp_backend_used"] is True
    assert summary["python_prototype_used"] is False
    assert summary["cuda_backend_used"] is False
    assert summary["uses_hadros_original_runtime_path"] is False
    assert summary["sigma_table_path"] == "data/sigma/sigma_nuN_CC_GBW.dat"
    assert summary["sigma_table_rows"] == 300
    assert summary["n_paths_processed"] == 5
    assert summary["n_segments_processed"] > 0

    dis_dir = tmp_path / "DIS"
    validation = json.loads((dis_dir / "backend_validation_report.json").read_text(encoding="utf-8"))
    assert validation["comparison_pass"] is True
    report = json.loads((dis_dir / "dis_optical_depth_report.json").read_text(encoding="utf-8"))
    assert report["validations"]["rho_non_negative"] is True
    assert report["validations"]["n_baryon_non_negative"] is True
    assert report["validations"]["sigma_non_negative"] is True
    assert report["validations"]["d_tau_non_negative"] is True
    assert report["validations"]["tau_non_negative"] is True
    assert report["validations"]["probability_bounds"] is True
    assert report["validations"]["cdf_normalized"] is True
    assert report["interaction_points_outside_medium"] == 0
    assert report["density_model_theta_is_hard_cut"] is False
    accepted = [json.loads(line) for line in (dis_dir / "dis_accepted_interactions.jsonl").read_text(encoding="utf-8").splitlines()]
    for interaction in accepted:
        assert interaction["interaction_point_density_checked"] is True
        assert interaction["interaction_point_inside_medium"] is True
        assert interaction["interaction_point_rho_g_cm3"] > 0.0
        assert interaction["interaction_rho_g_cm3"] > 0.0
    diagnostics = json.loads((dis_dir / "dis_diagnostics_report.json").read_text(encoding="utf-8"))
    assert diagnostics["diagnostics_generated"] is True
    assert diagnostics["medium_density_map_generated"] is True
    assert diagnostics["interaction_points_outside_medium"] == 0


def test_gbw_iim_comparison_generates_diagnostics(tmp_path: Path) -> None:
    if not Path("bin/hadros3_dis_sampler").exists():
        pytest.skip("H3-W7 C++ DIS sampler is not built")
    values = _values()
    values["dis_interaction_sampler"]["dis_backend"] = "cpp_hadros_original_port"
    generate_uhe_source_products(values, output_dir=tmp_path)
    generate_forward_geodesic_products(values, run_output_dir=tmp_path)
    generate_dis_interaction_products(values, run_output_dir=tmp_path)
    comparison = generate_gbw_iim_comparison(values, run_output_dir=tmp_path)

    assert comparison["status"] == "ok"
    assert "GBW" in comparison["models"]
    assert "IIM" in comparison["models"]
    dis_dir = tmp_path / "DIS"
    for filename in [
        "gbw_vs_iim_tau_comparison.png",
        "gbw_vs_iim_probability_comparison.png",
        "gbw_vs_iim_interaction_locations.png",
        "gbw_vs_iim_summary.json",
    ]:
        assert (dis_dir / filename).exists()
    diagnostics = json.loads((dis_dir / "dis_diagnostics_report.json").read_text(encoding="utf-8"))
    assert diagnostics["GBW_IIM_comparison"]["status"] == "ok"


def test_interaction_probability_bounds() -> None:
    for tau in [0.0, 1.0e-12, 0.1, 10.0, 1.0e6]:
        probability = interaction_probability(tau)
        assert math.isfinite(probability)
        assert 0.0 <= probability <= 1.0
