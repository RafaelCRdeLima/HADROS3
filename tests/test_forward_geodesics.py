from __future__ import annotations

import json
import math
from pathlib import Path

from hadros3.config import defaults
from hadros3.forward_geodesics import generate_forward_geodesic_products
from hadros3.uhe_source import generate_uhe_source_products


def _values() -> dict:
    values = defaults()
    values["polar_cone"]["opening_angle_deg"] = 20.0
    values["uhe_neutrino_source"].update(
        {
            "direction_model": "coordinate_radial_outward",
            "energy_gev": "10^{10}",
            "r_min_rg": 3.0,
            "r_max_rg": 5.0,
            "theta_min_deg": 1.0,
            "theta_max_deg": 8.0,
            "n_samples": 8,
            "random_seed": 1122,
        }
    )
    values["forward_geodesics"].update(
        {
            "n_samples_to_propagate": 5,
            "initial_step_rg": 2.0,
            "max_steps": 20,
            "outer_radius_rg": 30.0,
            "null_invariant_tolerance": 1.0e-6,
            "killing_energy_tolerance": 1.0e-10,
            "lz_tolerance": 1.0e-10,
        }
    )
    return values


def test_forward_geodesics_consume_h3_w5_source_and_write_outputs(tmp_path: Path) -> None:
    values = _values()
    generate_uhe_source_products(values, output_dir=tmp_path)
    summary = generate_forward_geodesic_products(values, run_output_dir=tmp_path)

    assert summary["forward_neutrino_geodesics_invoked"] is True
    assert summary["momentum_generator"] == "KerrNullMomentumGenerator"
    assert summary["momentum_is_physical_kerr"] is True
    assert summary["forward_backend"] == "cpp_hadros_original_port"
    assert summary["geodesic_backend"] == "cpp_hadros_original_port"
    assert summary["backend_kind"] == "ported_hadros_kerr_engine"
    assert summary["uses_hadros_original_runtime_path"] is False
    assert summary["uses_hamiltonian"] is True
    assert summary["uses_zamo_tetrad"] is True
    assert summary["full_kerr_geodesic"] is True
    assert summary["theta_phi_evolution"] is True
    assert summary["uses_kerr_metric"] is True
    assert summary["uses_christoffel_or_hamiltonian"] is True
    assert summary["coordinate_radial_preview"] is False
    assert summary["direction_generator"] == "CoordinateRadialOutwardDirectionGenerator"
    assert summary["direction_model"] == "coordinate_radial_outward"
    assert summary["n_samples_requested"] == 5
    assert summary["n_samples_propagated"] == 5
    assert summary["n_paths"] == 5
    assert summary["n_segments"] > 0
    assert summary["validation_pass"] is True
    assert summary["max_delta_phi_rad"] > 0.0
    assert summary["max_delta_theta_rad"] > 0.0
    assert summary["optical_depth_dis_sampler_invoked"] is False
    assert summary["observer_bridge_active_filter_invoked"] is False
    assert summary["expensive_event_generation_invoked"] is False

    forward_dir = tmp_path / "ForwardGeodesics"
    for filename in [
        "uhe_neutrino_forward_paths.jsonl",
        "uhe_neutrino_forward_path_segments.jsonl",
        "uhe_neutrino_forward_summary.csv",
        "uhe_neutrino_forward_summary.json",
        "uhe_neutrino_forward_preview.png",
        "uhe_neutrino_forward_geometry_3d.png",
        "uhe_neutrino_forward_geometry_3d.json",
        "uhe_neutrino_forward_geometry_3d.html",
        "geodesic_validation_report.json",
        "stop_condition_statistics.csv",
        "forward_geodesics_diagnostic_report.md",
    ]:
        assert (forward_dir / filename).exists()
    assert summary["products"]["forward_geometry_3d"].endswith("uhe_neutrino_forward_geometry_3d.png")
    assert summary["products"]["forward_geometry_3d_json"].endswith("uhe_neutrino_forward_geometry_3d.json")
    assert summary["products"]["forward_geometry_3d_html"].endswith("uhe_neutrino_forward_geometry_3d.html")

    first_path = json.loads((forward_dir / "uhe_neutrino_forward_paths.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first_path["direction_model"] == "coordinate_radial_outward"
    assert first_path["emission_direction"]["direction_local_components"]["dr"] == 1.0
    momentum = first_path["initial_momentum"]["four_momentum"]
    assert momentum["p_r"] > 0.0
    assert momentum["p_theta"] == 0.0
    assert abs(momentum["p_phi"]) < 1.0e-7


def test_forward_geodesics_respects_updated_sample_count_in_same_run(tmp_path: Path) -> None:
    values = _values()
    values["uhe_neutrino_source"]["n_samples"] = 20
    generate_uhe_source_products(values, output_dir=tmp_path)

    values["forward_geodesics"]["n_samples_to_propagate"] = 5
    summary_5 = generate_forward_geodesic_products(values, run_output_dir=tmp_path)
    config_5 = json.loads((tmp_path / "RunMetadata" / "hadros3_config.json").read_text(encoding="utf-8"))
    assert config_5["hadros3_values"]["forward_geodesics"]["n_samples_to_propagate"] == 5
    assert summary_5["n_input_samples"] == 20
    assert summary_5["n_samples_requested"] == 5
    assert summary_5["n_paths"] == 5

    values["forward_geodesics"]["n_samples_to_propagate"] = 12
    summary_12 = generate_forward_geodesic_products(values, run_output_dir=tmp_path)
    config_12 = json.loads((tmp_path / "RunMetadata" / "hadros3_config.json").read_text(encoding="utf-8"))
    assert config_12["hadros3_values"]["forward_geodesics"]["n_samples_to_propagate"] == 12
    assert summary_12["n_input_samples"] == 20
    assert summary_12["n_samples_requested"] == 12
    assert summary_12["n_paths"] == 12


def test_forward_geodesic_segments_are_finite_and_auditable(tmp_path: Path) -> None:
    values = _values()
    generate_uhe_source_products(values, output_dir=tmp_path)
    generate_forward_geodesic_products(values, run_output_dir=tmp_path)
    segment_path = tmp_path / "ForwardGeodesics" / "uhe_neutrino_forward_path_segments.jsonl"
    segments = [json.loads(line) for line in segment_path.read_text(encoding="utf-8").splitlines()]
    assert segments
    for segment in segments:
        for key in [
            "r_start_rg",
            "theta_start_rad",
            "phi_start_rad",
            "r_end_rg",
            "theta_end_rad",
            "phi_end_rad",
            "r_mid_rg",
            "theta_mid_rad",
            "phi_mid_rad",
            "p_t_mid",
            "p_r_mid",
            "p_theta_mid",
            "p_phi_mid",
            "dl_segment_rg",
            "E_nu_local_gev_mid",
        ]:
            assert math.isfinite(segment[key])
        assert segment["dl_segment_rg"] > 0.0
        assert segment["E_nu_local_gev_mid"] > 0.0
        assert segment["geodesic_status"] in {"propagated_forward_no_interaction", "invalid_invariant"}
    assert any(abs(segment["theta_end_rad"] - segment["theta_start_rad"]) > 0.0 for segment in segments)
    assert any(abs(segment["phi_end_rad"] - segment["phi_start_rad"]) > 0.0 for segment in segments)
    assert any(abs(segment["p_theta_mid"]) > 0.0 for segment in segments)


def test_forward_geodesic_validation_report_and_stop_statistics(tmp_path: Path) -> None:
    values = _values()
    generate_uhe_source_products(values, output_dir=tmp_path)
    summary = generate_forward_geodesic_products(values, run_output_dir=tmp_path)
    validation = json.loads((tmp_path / "ForwardGeodesics" / "geodesic_validation_report.json").read_text(encoding="utf-8"))
    assert validation["null_norm_max"] <= values["forward_geodesics"]["null_invariant_tolerance"]
    assert validation["killing_energy_max_error"] <= values["forward_geodesics"]["killing_energy_tolerance"]
    assert validation["lz_max_error"] <= values["forward_geodesics"]["lz_tolerance"]
    assert validation["validation_pass"] is True
    summary_path = tmp_path / "ForwardGeodesics" / "uhe_neutrino_forward_summary.json"
    summary_json = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_json["direction_model"] == "coordinate_radial_outward"
    assert summary_json["geodesic_backend"] == "cpp_hadros_original_port"
    assert summary_json["coordinate_radial_preview"] is False
    stop_csv = (tmp_path / "ForwardGeodesics" / "stop_condition_statistics.csv").read_text(encoding="utf-8")
    assert "stop_condition,count,fraction" in stop_csv
    assert "outer_escape_radius" in stop_csv
    assert sum(summary["stop_condition_counts"].values()) == summary["n_paths"]


def test_isotropic_local_source_produces_nonradial_kerr_momenta_and_bending(tmp_path: Path) -> None:
    values = defaults()
    values["black_hole"]["spin_a"] = 0.999
    values["polar_cone"]["opening_angle_deg"] = 20.0
    values["uhe_neutrino_source"].update(
        {
            "direction_model": "isotropic_local",
            "energy_gev": "10^{9}",
            "r_min_rg": 1.45,
            "r_max_rg": 2.2,
            "theta_min_deg": 1.0,
            "theta_max_deg": 18.0,
            "n_samples": 12,
            "random_seed": 1122,
            "direction_seed": 3344,
        }
    )
    values["forward_geodesics"].update(
        {
            "n_samples_to_propagate": 8,
            "initial_step_rg": 0.35,
            "max_steps": 120,
            "outer_radius_rg": 25.0,
        }
    )
    generate_uhe_source_products(values, output_dir=tmp_path)
    summary = generate_forward_geodesic_products(values, run_output_dir=tmp_path)
    assert summary["direction_model"] == "isotropic_local"
    assert summary["max_delta_phi_rad"] > 0.1
    assert summary["max_delta_theta_rad"] > 0.01
    assert summary["strong_field_diagnostic"]["classification"] == "PASS: isotropic local directions produce Kerr bending"

    first_path = json.loads((tmp_path / "ForwardGeodesics" / "uhe_neutrino_forward_paths.jsonl").read_text(encoding="utf-8").splitlines()[0])
    momentum = first_path["initial_momentum"]["four_momentum"]
    assert momentum["basis"] == "Boyer-Lindquist_covariant_from_ZAMO_orthonormal"
    assert first_path["initial_momentum"]["local_tetrad"] == "ZAMO"
    assert abs(momentum["p_theta"]) > 0.0
    assert abs(momentum["p_phi"]) > 0.0
