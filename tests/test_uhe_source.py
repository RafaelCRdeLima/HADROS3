from __future__ import annotations

import json
import math
from pathlib import Path

from hadros3.config import defaults
from hadros3.pipeline import render_hadros_web
from hadros3.source_models import sample_polar_cone
from hadros3.uhe_source import generate_uhe_source_products


def _source_values() -> dict:
    values = defaults()
    values["uhe_neutrino_source"].update(
        {
            "direction_model": "isotropic_local",
            "energy_gev": "10^{12}",
            "r_min_rg": 3.0,
            "r_max_rg": 9.0,
            "theta_min_deg": 2.0,
            "theta_max_deg": 20.0,
            "n_samples": 64,
            "random_seed": 9876,
        }
    )
    return values


def test_polar_cone_sampler_respects_domains_and_weights() -> None:
    values = _source_values()
    records = sample_polar_cone(values)
    assert len(records) == 64
    for record in records:
        position = record["position"]
        assert 3.0 <= position["r_rg"] <= 9.0
        assert 2.0 <= position["theta_deg"] <= 20.0
        assert 0.0 <= position["phi_rad"] < 2.0 * math.pi
        assert record["x_emit_r"] == position["r_rg"]
        assert record["x_emit_theta"] == position["theta_rad"]
        assert record["x_emit_phi"] == position["phi_rad"]
        assert record["E_nu_emit_gev"] == 1.0e12
        assert record["source_physical_pdf"] > 0.0
        assert record["source_sampling_pdf"] > 0.0
        assert record["source_physical_pdf"] == record["source_sampling_pdf"]
        assert math.isfinite(record["source_weight"])
        assert record["source_weight"] == 1.0
        assert record["direction_generator"] == "IsotropicLocalDirectionGenerator"
        assert record["direction_model"] == "isotropic_local"
        direction = record["direction_local_components"]
        assert direction["basis"] == "ZAMO_orthonormal"
        norm = math.sqrt(direction["n_r"] ** 2 + direction["n_theta"] ** 2 + direction["n_phi"] ** 2)
        assert abs(norm - 1.0) < 1.0e-12
        assert record["direction_sampling_pdf"] == 1.0 / (4.0 * math.pi)
        assert record["direction_physical_pdf"] == 1.0 / (4.0 * math.pi)
        assert record["direction_weight"] == 1.0
        assert record["emission_direction"]["direction_model"] == "isotropic_local"
        for key in [
            "x_emit_r",
            "x_emit_theta",
            "x_emit_phi",
            "E_nu_emit_gev",
            "direction_model",
            "emission_direction",
            "direction_local_components",
            "direction_sampling_pdf",
            "direction_physical_pdf",
            "direction_weight",
            "source_sampling_pdf",
            "source_physical_pdf",
            "source_weight",
        ]:
            assert key in record
        assert record["momentum_generator"] == "ProxyRadialMomentumGenerator"
        assert record["momentum_is_physical_kerr"] is False
        assert record["initial_momentum"]["four_momentum"] is None


def test_polar_cone_sampler_seed_is_reproducible() -> None:
    values = _source_values()
    assert sample_polar_cone(values) == sample_polar_cone(values)
    values["uhe_neutrino_source"]["random_seed"] = 9877
    assert sample_polar_cone(values) != sample_polar_cone(_source_values())


def test_generate_uhe_source_products_and_provenance(tmp_path: Path) -> None:
    values = _source_values()
    summary = generate_uhe_source_products(values, output_dir=tmp_path)
    assert summary["source_sampler_active"] is True
    assert summary["source_model"] == "polar_cone"
    assert summary["source_volume_model"] == "coordinate_volume"
    assert summary["direction_generator"] == "IsotropicLocalDirectionGenerator"
    assert summary["direction_model"] == "isotropic_local"
    assert summary["direction_sampling_pdf"] == 1.0 / (4.0 * math.pi)
    assert summary["direction_physical_pdf"] == 1.0 / (4.0 * math.pi)
    assert summary["direction_weight"] == 1.0
    assert summary["momentum_generator"] == "ProxyRadialMomentumGenerator"
    assert summary["momentum_is_physical_kerr"] is False

    for filename in [
        "uhe_neutrino_source_samples.jsonl",
        "uhe_neutrino_source_summary.csv",
        "uhe_neutrino_source_summary.json",
        "uhe_neutrino_source_preview.png",
    ]:
        assert (tmp_path / "UHEsource" / filename).exists()

    render_summary = render_hadros_web(values, root=Path.cwd(), output_dir=tmp_path, source_summary=summary)
    provenance = json.loads(Path(render_summary["products"]["provenance"]).read_text(encoding="utf-8"))
    assert provenance["source_sampler"]["source_sampler_active"] is True
    assert provenance["source_sampler"]["source_model"] == "polar_cone"
    assert provenance["source_sampler"]["direction_generator"] == "IsotropicLocalDirectionGenerator"
    assert provenance["source_sampler"]["direction_model"] == "isotropic_local"
    assert provenance["source_sampler"]["direction_sampling_pdf"] == 1.0 / (4.0 * math.pi)
    assert provenance["source_sampler"]["direction_physical_pdf"] == 1.0 / (4.0 * math.pi)
    assert provenance["source_sampler"]["direction_weight"] == 1.0
    assert provenance["source_sampler"]["four_momentum_sampled_in_source"] is False
    assert provenance["source_sampler"]["momentum_is_physical_kerr"] is False
    assert provenance["source_sampler"]["forward_neutrino_geodesics_invoked"] is False
    assert provenance["source_sampler"]["optical_depth_dis_sampler_invoked"] is False
    assert provenance["source_sampler"]["observer_bridge_active_filter_invoked"] is False
    assert provenance["source_sampler"]["expensive_event_generation_invoked"] is False
