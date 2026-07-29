from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import pytest
import jsonschema

from hadros3.config import defaults, validate_values
from hadros3.geant4_transport import backend_availability, generate_geant4_products, geant4_environment
from hadros3.geant4_visualization import write_geant4_visualizations
from hadros_web import render_html


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "geant4"


@pytest.fixture(scope="session")
def geant4_binary() -> Path:
    executable = ROOT / "bin" / "hadros3_geant4_transport"
    if not executable.exists():
        # Geant4 comes from conda-forge only: on a checkout without it, skip
        # instead of failing the whole suite. `make doctor` reports the gap.
        result = subprocess.run(["make", "bin/hadros3_geant4_transport"], cwd=ROOT, check=False, capture_output=True, text=True)
        if result.returncode != 0 or not executable.exists():
            pytest.skip("H3-W11 Geant4 backend is not available (run 'make setup' with micromamba/conda)")
    return executable


def run_backend(executable: Path, input_path: Path, output: Path, *args: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
    command = [str(executable), "--input", str(input_path), "--output-dir", str(output), *args]
    result = subprocess.run(command, cwd=ROOT, env=geant4_environment(), text=True, capture_output=True, check=False)
    assert result.returncode == expected, result.stdout + "\n" + result.stderr
    return result


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_gamma_events(path: Path, n: int, energy_gev: float = 1.0) -> None:
    lines = ["HepMC::Version 3.03.01", "HepMC::Asciiv3-START_EVENT_LISTING"]
    for event in range(n):
        lines += [f"E {event} 0 1", "U GEV MM", f"P 1 0 22 0 0 {energy_gev:.17g} {energy_gev:.17g} 0 1"]
    lines.append("HepMC::Asciiv3-END_EVENT_LISTING")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_particle_events(path: Path, n: int, pdg: int, momentum_gev: float, mass_gev: float) -> None:
    energy = math.sqrt(momentum_gev**2 + mass_gev**2)
    lines = ["HepMC::Version 3.03.01", "HepMC::Asciiv3-START_EVENT_LISTING"]
    for event in range(n):
        lines += [f"E {event} 0 1", "U GEV MM", f"P 1 0 {pdg} 0 0 {momentum_gev:.17g} {energy:.17g} {mass_gev:.17g} 1"]
    lines.append("HepMC::Asciiv3-END_EVENT_LISTING")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_environment_is_pinned_and_all_datasets_exist(geant4_binary: Path) -> None:
    availability = backend_availability()
    assert availability["available"] is True
    assert availability["geant4_version"] == "11.4.2"
    assert availability["hepmc3_version"] == "3.3.1"
    assert availability["backend_sha256"] == hashlib.sha256(geant4_binary.read_bytes()).hexdigest()
    assert availability["datasets"]
    assert all(entry["available"] for entry in availability["datasets"].values())


def test_import_contract_has_exact_cardinality_and_units(geant4_binary: Path, tmp_path: Path) -> None:
    run_backend(geant4_binary, FIXTURES / "six_muons_vacuum.hepmc3", tmp_path, "--mode", "import_check")
    report = json.loads((tmp_path / "geant4_import_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert report["events"] == 1
    assert report["final_particles"] == 6
    assert report["momentum_unit"] == "GeV"
    assert report["length_unit"] == "mm"
    assert report["generator_frame"] == "local_matter_tetrad"
    assert report["violations"] == []


def test_uhe_domain_guard_refuses_before_transport(geant4_binary: Path, tmp_path: Path) -> None:
    result = run_backend(
        geant4_binary, FIXTURES / "unsupported_uhe_pion.hepmc3", tmp_path,
        "--mode", "material_smoke", "--max-energy-gev", "100000", expected=3,
    )
    report = json.loads((tmp_path / "geant4_import_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "unsupported_domain"
    assert report["unsupported_energy"] == 1
    assert report["maximum_energy_gev"] == 1.0e8
    assert not (tmp_path / "geant4_backend_summary.json").exists()
    assert "domain guard refused" in result.stderr


def test_orchestrator_publishes_visible_uhe_domain_audit(geant4_binary: Path, tmp_path: Path) -> None:
    values = defaults()
    values["geant4"]["mode"] = "real_free"
    summary = generate_geant4_products(
        values,
        run_output_dir=tmp_path,
        input_override=FIXTURES / "unsupported_uhe_pion.hepmc3",
    )
    assert summary["status"] == "unsupported_domain"
    assert summary["geant4_invoked"] is False
    assert summary["events_transported"] == 0
    assert summary["domain_violations"] == [{
        "event_id": 0,
        "particle_id": 1,
        "pdg_id": 211,
        "energy_gev": 1.0e8,
        "validated_maximum_energy_gev": 1.0e5,
        "violation": "unsupported_energy event=0 particle=1 pdg=211 energy_gev=100000000.000000 maximum_gev=100000.000000",
    }]
    assert (tmp_path / "GEANT4" / "geant4_domain_audit.png").stat().st_size > 0
    viewer = (tmp_path / "GEANT4" / "geant4_event_view.html").read_text(encoding="utf-8")
    assert "transport not started" in viewer
    assert "geant4_unsupported_particles.jsonl" in viewer


def test_vacuum_axes_crossings_and_energy_ledger(geant4_binary: Path, tmp_path: Path) -> None:
    run_backend(
        geant4_binary, FIXTURES / "six_muons_vacuum.hepmc3", tmp_path,
        "--mode", "vacuum_smoke", "--half-size-mm", "10", "--world-margin-mm", "10", "--seed", "59001",
    )
    summary = json.loads((tmp_path / "geant4_backend_summary.json").read_text(encoding="utf-8"))
    escaped = read_jsonl(tmp_path / "geant4_escaped_particles_raw.jsonl")
    steps = read_jsonl(tmp_path / "geant4_steps_raw.jsonl")
    assert summary["events_transported"] == 1
    assert summary["escaped_particles"] == 6
    assert summary["recorded_steps"] == 6
    assert summary["steps_truncated"] is False
    assert summary["deposited_gev"] < 1.0e-20
    assert summary["max_abs_normalized_unexplained_residual"] < 1.0e-6
    positions = {tuple(round(value, 9) for value in row["position_local_mm"]) for row in escaped}
    assert positions == {(10.0, 0.0, 0.0), (-10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, -10.0, 0.0), (0.0, 0.0, 10.0), (0.0, 0.0, -10.0)}
    for row in escaped:
        x = row["position_local_mm"]
        p = row["momentum_local_gev"]
        assert sum(a * b for a, b in zip(x, p)) > 0.0
        assert row["parent_track_id"] == 0
        assert row["geant4_statistical_weight"] == 1.0
        assert row["creator_process"] == "primary"
        assert sum(a * b for a, b in zip(row["position_local_mm"], row["boundary_normal_local"])) == pytest.approx(10.0)
    assert {row["process_name"] for row in steps} == {"Transportation"}
    assert all(row["is_boundary"] and not row["is_interaction"] for row in steps)
    assert {tuple(row["pre_position_local_mm"]) for row in steps} == {(0, 0, 0)}
    assert {tuple(row["post_position_local_mm"]) for row in steps} == positions


def test_serial_seed_is_byte_reproducible(geant4_binary: Path, tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    args = ("--mode", "vacuum_smoke", "--seed", "59001")
    run_backend(geant4_binary, FIXTURES / "six_muons_vacuum.hepmc3", first, *args)
    run_backend(geant4_binary, FIXTURES / "six_muons_vacuum.hepmc3", second, *args)
    for name in ("geant4_backend_summary.json", "geant4_events_raw.jsonl", "geant4_escaped_particles_raw.jsonl", "geant4_steps_raw.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_linked_macro_and_local_geant4_viewers_use_exact_dis_site(tmp_path: Path) -> None:
    values = defaults()
    dis_dir = tmp_path / "DIS"
    dis_dir.mkdir()
    interaction = {
        "interaction_id": "H3DIS-TEST", "interaction_r_rg": 10.0,
        "interaction_theta_rad": math.pi / 2.0, "interaction_phi_rad": math.pi / 2.0,
        "interaction_rho_g_cm3": 2.5,
    }
    (dis_dir / "dis_accepted_interactions.jsonl").write_text(json.dumps(interaction) + "\n", encoding="utf-8")
    output = tmp_path / "GEANT4"
    output.mkdir()
    events = [{"geant4_event_id": 0, "interaction_id": "H3DIS-TEST", "deposited_gev": 0.2}]
    steps = [{
        "geant4_event_id": 0, "track_id": 1, "parent_track_id": 0, "pdg_id": 211,
        "pre_position_local_mm": [0.0, 0.0, 0.0], "post_position_local_mm": [1.0, 2.0, 3.0],
        "is_interaction": True, "process_name": "pi+Inelastic",
    }]
    sites = write_geant4_visualizations(
        values, tmp_path, output, events, [], steps, material="HADROS3_H_HE", density_g_cm3=2.5,
    )
    assert sites[0]["global_position_available"] is True
    assert sites[0]["position_xyz_rg"] == pytest.approx([0.0, 10.0, 0.0], abs=1e-12)
    assert sites[0]["interaction_steps"] == 1
    assert sites[0]["local_view"] == "geant4_event_view.html?event=0"
    macro = (output / "geant4_macro_sites_3d.html").read_text(encoding="utf-8")
    local = (output / "geant4_event_view.html").read_text(encoding="utf-8")
    assert "window.open(q.s.local_view" in macro
    assert "BH + analytic torus + polar funnels" in macro
    assert "pi+Inelastic" in local
    assert "Two-scale view" in local
    payload = json.loads((output / "geant4_sites.json").read_text(encoding="utf-8"))
    assert payload["global_length_unit"] == "r_g"
    assert payload["local_length_unit"] == "mm"


def test_orchestrator_publishes_recorded_steps_and_both_scales(geant4_binary: Path, tmp_path: Path) -> None:
    values = defaults()
    values["geant4"].update({"mode": "vacuum_smoke", "max_events": 1, "max_recorded_steps": 100})
    summary = generate_geant4_products(
        values, run_output_dir=tmp_path, input_override=FIXTURES / "six_muons_vacuum.hepmc3",
    )
    output = tmp_path / "GEANT4"
    assert summary["status"] == "ok"
    assert summary["recorded_steps"] == 6
    assert summary["geant4_sites"] == 1
    assert summary["validation"]["recorded_steps_pass"] is True
    assert len(read_jsonl(output / "geant4_steps.jsonl")) == 6
    assert (output / "geant4_sites.json").exists()
    assert (output / "geant4_macro_sites_3d.html").exists()
    assert "Local GEANT4 volume" in (output / "geant4_event_view.html").read_text(encoding="utf-8")
    assert summary["products"]["geant4_macro_sites_3d"].endswith("geant4_macro_sites_3d.html")


def test_gamma_attenuation_is_exponential_in_lead(geant4_binary: Path, tmp_path: Path) -> None:
    input_path = tmp_path / "gamma.hepmc3"
    n_events = 500
    write_gamma_events(input_path, n_events)
    inferred_mu: list[float] = []
    survivals: list[float] = []
    for half_size in (2.0, 5.0, 10.0):
        output = tmp_path / f"lead_{half_size:g}"
        run_backend(
            geant4_binary, input_path, output, "--mode", "material_smoke", "--material", "G4_Pb",
            "--half-size-mm", str(half_size), "--world-margin-mm", "1", "--max-events", str(n_events), "--seed", "59001",
        )
        rows = read_jsonl(output / "geant4_escaped_particles_raw.jsonl")
        event_rows = read_jsonl(output / "geant4_events_raw.jsonl")
        assert all(abs(row["unexplained_residual_gev"]) < 1e-12 for row in event_rows)
        assert all(row["raw_energy_balance_gev"] == pytest.approx(row["inferred_medium_rest_mass_and_binding_exchange_gev"], abs=1e-12) for row in event_rows)
        uncollided = sum(row["pdg_id"] == 22 and row["parent_track_id"] == 0 and row["energy_local_gev"] > 0.999999 for row in rows)
        survival = uncollided / n_events
        survivals.append(survival)
        inferred_mu.append(-math.log(survival) / half_size)
    assert survivals[0] > survivals[1] > survivals[2]
    assert max(inferred_mu) / min(inferred_mu) < 1.20
    assert 0.10 < sum(inferred_mu) / len(inferred_mu) < 0.16  # inverse mm for this Geant4/Pb/1-GeV fixture


def test_muon_stopping_power_scales_with_water_thickness(geant4_binary: Path, tmp_path: Path) -> None:
    input_path = tmp_path / "muons.hepmc3"
    write_particle_events(input_path, 500, 13, 1.0, 0.1056583755)
    deposits = []
    for half_size in (1.0, 2.0):
        output = tmp_path / f"water_{half_size:g}"
        run_backend(
            geant4_binary, input_path, output, "--mode", "material_smoke", "--material", "G4_WATER",
            "--half-size-mm", str(half_size), "--world-margin-mm", "1", "--max-events", "500", "--seed", "59001",
        )
        summary = json.loads((output / "geant4_backend_summary.json").read_text(encoding="utf-8"))
        deposits.append(summary["deposited_gev"] / 500.0)
    assert 1.9 < deposits[1] / deposits[0] < 2.1
    assert 1.5e-4 < deposits[0] < 2.0e-4


def test_hadronic_survival_has_consistent_interaction_length(geant4_binary: Path, tmp_path: Path) -> None:
    input_path = tmp_path / "protons.hepmc3"
    write_particle_events(input_path, 500, 2212, 10.0, 0.93827208816)
    inferred_mu = []
    survivals = []
    for half_size in (10.0, 30.0, 50.0):
        output = tmp_path / f"proton_lead_{half_size:g}"
        run_backend(
            geant4_binary, input_path, output, "--mode", "material_smoke", "--material", "G4_Pb",
            "--half-size-mm", str(half_size), "--world-margin-mm", "1", "--max-events", "500", "--seed", "59001",
        )
        rows = read_jsonl(output / "geant4_escaped_particles_raw.jsonl")
        primary_survivors = sum(row["parent_track_id"] == 0 and row["pdg_id"] == 2212 for row in rows)
        survival = primary_survivors / 500.0
        survivals.append(survival)
        inferred_mu.append(-math.log(survival) / half_size)
    assert survivals[0] > survivals[1] > survivals[2]
    assert max(inferred_mu) / min(inferred_mu) < 1.30
    assert 0.004 < sum(inferred_mu) / len(inferred_mu) < 0.007


def test_neutrino_is_transport_only_pass_through(geant4_binary: Path, tmp_path: Path) -> None:
    input_path = tmp_path / "neutrino.hepmc3"
    write_particle_events(input_path, 1, 12, 10.0, 0.0)
    run_backend(
        geant4_binary, input_path, tmp_path / "out", "--mode", "material_smoke", "--material", "G4_Pb",
        "--half-size-mm", "10", "--world-margin-mm", "1", "--max-events", "1",
    )
    summary = json.loads((tmp_path / "out" / "geant4_backend_summary.json").read_text(encoding="utf-8"))
    rows = read_jsonl(tmp_path / "out" / "geant4_escaped_particles_raw.jsonl")
    assert summary["deposited_gev"] == 0.0
    assert len(rows) == 1
    assert rows[0]["pdg_id"] == 12
    assert rows[0]["energy_local_gev"] == pytest.approx(10.0, rel=1e-12)


def test_charged_pion_decay_preserves_energy_and_lineage(geant4_binary: Path, tmp_path: Path) -> None:
    input_path = tmp_path / "pion_rest.hepmc3"
    write_particle_events(input_path, 1, 211, 0.0, 0.13957039)
    run_backend(
        geant4_binary, input_path, tmp_path / "out", "--mode", "vacuum_smoke",
        "--half-size-mm", "100", "--world-margin-mm", "10", "--max-events", "1", "--seed", "59001",
    )
    summary = json.loads((tmp_path / "out" / "geant4_backend_summary.json").read_text(encoding="utf-8"))
    rows = read_jsonl(tmp_path / "out" / "geant4_escaped_particles_raw.jsonl")
    assert {row["pdg_id"] for row in rows} == {-13, 14}
    assert all(row["parent_track_id"] == 1 and row["creator_process"] == "Decay" for row in rows)
    assert sum(row["energy_local_gev"] for row in rows) == pytest.approx(0.13957039, rel=1e-12)
    assert summary["max_abs_normalized_unexplained_residual"] < 1e-12


def test_orchestrator_preserves_upstream_ids_and_signed_weights(geant4_binary: Path, tmp_path: Path) -> None:
    event_dir = tmp_path / "EventGeneration"
    event_dir.mkdir()
    (event_dir / "event_generation_events_summary.jsonl").write_text(
        json.dumps({"event_generation_event_id": "fixture:1", "powheg_request_id": "P1", "lhe_event_index": 1, "xwgtup": -2.5, "physics_weight": 3.0, "observer_weight": 0.25}) + "\n",
        encoding="utf-8",
    )
    values = defaults()
    values["geant4"]["mode"] = "vacuum_smoke"
    summary = generate_geant4_products(values, run_output_dir=tmp_path, input_override=FIXTURES / "six_muons_vacuum.hepmc3")
    assert summary["status"] == "ok"
    assert summary["geant4_invoked"] is True
    rows = read_jsonl(tmp_path / "GEANT4" / "geant4_escaped_particles.jsonl")
    assert rows
    assert {row["event_generation_event_id"] for row in rows} == {"fixture:1"}
    assert {row["xwgtup"] for row in rows} == {-2.5}
    assert {row["physics_weight"] for row in rows} == {3.0}
    assert {row["observer_weight"] for row in rows} == {0.25}
    assert {row["geant4_statistical_weight"] for row in rows} == {1.0}


def test_failed_run_preserves_last_atomic_product(geant4_binary: Path, tmp_path: Path) -> None:
    values = defaults()
    values["geant4"]["mode"] = "vacuum_smoke"
    generate_geant4_products(values, run_output_dir=tmp_path, input_override=FIXTURES / "six_muons_vacuum.hepmc3")
    summary_path = tmp_path / "GEANT4" / "geant4_summary.json"
    previous = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    corrupt = tmp_path / "corrupt.hepmc3"
    corrupt.write_text("not HepMC3\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        generate_geant4_products(values, run_output_dir=tmp_path, input_override=corrupt)
    assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == previous
    assert not list(tmp_path.glob(".GEANT4.staging-*"))


def test_config_blocks_unvalidated_multithreading_and_optical_physics() -> None:
    values = defaults()
    assert validate_values(values) == []
    contract = json.loads((ROOT / "schemas" / "geant4_transport_contract.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(values["geant4"], contract)
    values["geant4"]["threads"] = 2
    assert "geant4.threads must remain 1 until MT equivalence is validated" in validate_values(values)
    values["geant4"]["threads"] = 1
    values["geant4"]["optical_physics_enabled"] = True
    assert "geant4 optical physics is not supported in H3-W11 v1" in validate_values(values)
    values["geant4"]["optical_physics_enabled"] = False
    values["geant4"]["validated_maximum_energy_gev"] = 100001
    assert "geant4.validated_maximum_energy_gev cannot exceed the validated H3-W11 v1 ceiling of 100000 GeV" in validate_values(values)


def test_environment_check_does_not_require_h3_w10_input(geant4_binary: Path, tmp_path: Path) -> None:
    values = defaults()
    values["geant4"]["mode"] = "environment_check"
    summary = generate_geant4_products(values, run_output_dir=tmp_path)
    assert summary["status"] == "ok"
    assert summary["geant4_invoked"] is False
    assert (tmp_path / "GEANT4" / "geant4_environment_manifest.json").exists()


def test_official_run_resolves_density_from_originating_dis_vertex(geant4_binary: Path, tmp_path: Path) -> None:
    event_dir = tmp_path / "EventGeneration"
    dis_dir = tmp_path / "DIS"
    event_dir.mkdir()
    dis_dir.mkdir()
    (event_dir / "event_generation_events.hepmc3").write_bytes((FIXTURES / "six_muons_vacuum.hepmc3").read_bytes())
    (event_dir / "event_generation_manifest.json").write_text(json.dumps({"stage": "H3-W10", "generator_frame": "local_matter_tetrad", "momentum_unit": "GeV", "length_unit": "mm"}), encoding="utf-8")
    (event_dir / "event_generation_summary.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    (event_dir / "event_generation_events_summary.jsonl").write_text(json.dumps({"event_generation_event_id": "fixture:1", "interaction_id": "H3DIS-fixture"}) + "\n", encoding="utf-8")
    (dis_dir / "dis_accepted_interactions.jsonl").write_text(json.dumps({"interaction_id": "H3DIS-fixture", "interaction_rho_g_cm3": 12.5}) + "\n", encoding="utf-8")
    values = defaults()
    values["geant4"]["mode"] = "import_check"
    summary = generate_geant4_products(values, run_output_dir=tmp_path)
    manifest = json.loads((tmp_path / "GEANT4" / "geant4_manifest.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert manifest["resolved_density_source"] == "dis_vertex_local"
    assert manifest["resolved_density_g_cm3"] == 12.5
    assert manifest["event_density_records"] == [{"event_generation_event_id": "fixture:1", "interaction_id": "H3DIS-fixture", "density_g_cm3": 12.5}]


def test_per_site_scheduler_isolates_distinct_dis_densities(geant4_binary: Path, tmp_path: Path) -> None:
    event_dir = tmp_path / "EventGeneration"
    dis_dir = tmp_path / "DIS"
    event_dir.mkdir()
    dis_dir.mkdir()
    (event_dir / "event_generation_manifest.json").write_text(json.dumps({
        "stage": "H3-W10", "generator_frame": "local_matter_tetrad", "momentum_unit": "GeV", "length_unit": "mm",
    }), encoding="utf-8")
    rows = []
    dis_rows = []
    for index, density in enumerate((1.25, 7.5), 1):
        request_id = f"P{index}"
        interaction_id = f"D{index}"
        job = event_dir / "jobs" / request_id
        job.mkdir(parents=True)
        (job / "events.hepmc3").write_bytes((FIXTURES / "six_muons_vacuum.hepmc3").read_bytes())
        rows.append({"event_generation_event_id": f"{request_id}:1", "powheg_request_id": request_id,
                     "interaction_id": interaction_id, "lhe_event_index": 1})
        dis_rows.append({"interaction_id": interaction_id, "interaction_rho_g_cm3": density})
    (event_dir / "event_generation_events_summary.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (dis_dir / "dis_accepted_interactions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in dis_rows), encoding="utf-8")
    values = defaults()
    values["geant4"].update({"mode": "import_check", "max_events": 2, "site_workers": 2})
    summary = generate_geant4_products(values, run_output_dir=tmp_path)
    sites = read_jsonl(tmp_path / "GEANT4" / "geant4_site_jobs_summary.jsonl")
    validation = json.loads((tmp_path / "GEANT4" / "geant4_validation_report.json").read_text())
    assert summary["execution_model"] == "per_site_subprocess"
    assert summary["site_jobs"] == 2
    assert [row["density_g_cm3"] for row in sites] == [1.25, 7.5]
    assert len({row["seed"] for row in sites}) == 2
    assert validation["site_isolation_pass"] is True
    assert validation["upstream_hash_unchanged"] is True


def test_web_requires_explicit_geant4_action_and_separates_process_state() -> None:
    html = render_html(defaults(), ROOT / "presets" / "hadros_web" / "default_config.json")
    assert "H3-W11 GEANT4 Local Material Transport" in html
    assert 'post("/api/geant4", {values, action: "run_geant4"})' in html
    assert "GEANT4 requires an explicit Run action" in (ROOT / "hadros_web.py").read_text(encoding="utf-8")
    assert "completed — not currently running" in html
    assert "input outside validated physics domain — no transport started" in html
    assert "The run did produce an audit result" in html
    assert "Input-domain audit plot" in html
    assert "opening" not in html[html.index("async function runGeant4") : html.index("async function launchInteractiveCameraPreview")]


def test_current_h3_w10_sample_is_classified_by_domain_guard_if_available(geant4_binary: Path, tmp_path: Path) -> None:
    current = ROOT / "output" / "HADROS3_hadros_web_preview" / "EventGeneration" / "event_generation_events.hepmc3"
    if not current.exists():
        pytest.skip("current H3-W10 sample is not present")
    command = [str(geant4_binary), "--input", str(current), "--output-dir", str(tmp_path), "--mode", "import_check", "--max-events", "2"]
    result = subprocess.run(command, cwd=ROOT, env=geant4_environment(), text=True, capture_output=True, check=False)
    assert result.returncode in {0, 3}, result.stdout + "\n" + result.stderr
    report = json.loads((tmp_path / "geant4_import_report.json").read_text(encoding="utf-8"))
    above_ceiling = report["maximum_energy_gev"] > report["validated_maximum_energy_gev"]
    assert (report["unsupported_energy"] > 0) is above_ceiling
    assert (result.returncode == 3) is above_ceiling
