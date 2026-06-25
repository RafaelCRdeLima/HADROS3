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
    assert summary["n_samples_requested"] == 5
    assert summary["n_samples_propagated"] == 5
    assert summary["n_paths"] == 5
    assert summary["n_segments"] > 0
    assert summary["validation_pass"] is True
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
        "geodesic_validation_report.json",
        "stop_condition_statistics.csv",
    ]:
        assert (forward_dir / filename).exists()


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


def test_forward_geodesic_validation_report_and_stop_statistics(tmp_path: Path) -> None:
    values = _values()
    generate_uhe_source_products(values, output_dir=tmp_path)
    summary = generate_forward_geodesic_products(values, run_output_dir=tmp_path)
    validation = json.loads((tmp_path / "ForwardGeodesics" / "geodesic_validation_report.json").read_text(encoding="utf-8"))
    assert validation["null_norm_max"] <= values["forward_geodesics"]["null_invariant_tolerance"]
    assert validation["killing_energy_max_error"] <= values["forward_geodesics"]["killing_energy_tolerance"]
    assert validation["lz_max_error"] <= values["forward_geodesics"]["lz_tolerance"]
    assert validation["validation_pass"] is True
    stop_csv = (tmp_path / "ForwardGeodesics" / "stop_condition_statistics.csv").read_text(encoding="utf-8")
    assert "stop_condition,count,fraction" in stop_csv
    assert "outer_escape_radius" in stop_csv
    assert sum(summary["stop_condition_counts"].values()) == summary["n_paths"]
