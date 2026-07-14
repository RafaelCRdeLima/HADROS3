from __future__ import annotations

import os
from pathlib import Path

import pytest

from hadros3.config import defaults
from hadros3.dis_sampler import generate_dis_interaction_products
from hadros3.event_generation import BACKEND as EVENT_BACKEND, generate_event_generation_products
from hadros3.forward_geodesics import generate_forward_geodesic_products
from hadros3.observer_bridge import generate_observer_bridge_products
from hadros3.observer_image_branches import generate_observer_image_branch_products
from hadros3.powheg import POWHEG_BINARY, generate_powheg_products
from hadros3.uhe_source import generate_uhe_source_products


@pytest.mark.skipif(os.environ.get("HADROS3_RUN_E2E_NLO_TEST") != "1", reason="opt-in source-to-LHE NLO campaign test")
def test_deterministic_small_pipeline_from_source_to_nlo_lhe(tmp_path: Path) -> None:
    if not POWHEG_BINARY.exists():
        pytest.skip("local POWHEG pwhg_main is not built")
    if not EVENT_BACKEND.exists():
        pytest.skip("local PYTHIA 8/HepMC3 event backend is not built")
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
        {"n_samples_to_propagate": 5, "initial_step_rg": 1.0, "max_steps": 32, "outer_radius_rg": 30.0}
    )
    values["dis_interaction_sampler"].update(
        {
            "dis_backend": "cpp_hadros_original_port",
            "density_floor_g_cm3": 0.0,
            "random_seed": 24680,
            "max_interactions": 1,
        }
    )
    values["observer_bridge"].update(
        {
            "fov_policy": "soft",
            "downstream_selection_policy": "top_n",
            "downstream_top_n_candidates": 1,
            "kerr_pixel_match_resolution_x": 5,
            "kerr_pixel_match_resolution_y": 3,
            "multi_image_audit_resolution_x": 5,
            "multi_image_audit_resolution_y": 3,
            "interactive_max_candidates": 1,
            "interactive_max_rays": 1,
            "observer_bridge_orientation_diagnostics_enabled": False,
        }
    )
    values["powheg"].update(
        {"run_mode": "real_smoke", "perturbative_order": "NLO", "events_per_candidate": 1, "random_seed": 34100}
    )

    source = generate_uhe_source_products(values, output_dir=tmp_path)
    forward = generate_forward_geodesic_products(values, run_output_dir=tmp_path)
    dis = generate_dis_interaction_products(values, run_output_dir=tmp_path, include_model_comparison=False)
    observer = generate_observer_bridge_products(values, run_output_dir=tmp_path)
    branches = generate_observer_image_branch_products(values, run_output_dir=tmp_path)
    powheg = generate_powheg_products(values, run_output_dir=tmp_path)
    values["event_generation"]["mode"] = "real_smoke"
    event_generation = generate_event_generation_products(values, run_output_dir=tmp_path)

    assert source["n_samples"] == 8
    assert forward["n_paths"] == 5
    assert dis["n_interactions_accepted"] == 1
    assert observer["n_candidates_scored"] == 1
    assert branches["n_candidates"] == 1
    assert powheg["powheg_lhe_generated"] is True
    assert powheg["perturbative_order"] == "NLO"
    assert powheg["n_lhe_events"] >= 1
    assert powheg["powheg_physics_summary"]["four_momentum_residual_relative_max"] <= 5.0e-8
    assert powheg["powheg_physics_summary"]["four_momentum_conservation_pass"] is True
    assert event_generation["pythia_invoked"] is True
    assert event_generation["n_events_generated"] >= 1
    assert event_generation["n_final_hadrons"] > 0
    assert event_generation["n_final_partons"] == 0
    assert event_generation["validation"]["matching_scale_pass"] is True
    assert event_generation["validation"]["four_momentum_residual_relative_max"] <= 5.0e-8
