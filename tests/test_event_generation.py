from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from hadros3.config import defaults, validate_values
from hadros3.event_generation import BACKEND, _canonicalize_lhe_beam_frame, _merge_hepmc, _run_backend, backend_availability, generate_event_generation_products, pdg_symbol
from hadros3.paths import EVENT_GENERATION_DIR, clear_powheg_outputs, event_generation_dir
from hadros3.powheg import generate_powheg_products
from hadros3.provenance import build_provenance


def test_hepmc_merge_has_one_run_header_contiguous_events_and_global_ids(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "geant4" / "six_muons_vacuum.hepmc3"
    merged = tmp_path / "merged.hepmc3"
    _merge_hepmc([source, source], merged)
    lines = merged.read_text(encoding="utf-8").splitlines()
    assert lines.count("W Weight") <= 1
    assert [line.split()[1] for line in lines if line.startswith("E ")] == ["0", "1"]
    first_end = max(index for index, line in enumerate(lines) if line.startswith("P ") and index < lines.index(next(line for line in lines if line.startswith("E 1 "))))
    assert lines[first_end + 1].startswith("E 1 ")


def _fixture_run(run_dir: Path, *, duplicate: bool = False, beam_energy: float = 1.0e8) -> Path:
    powheg_dir = run_dir / "POWHEG"
    lhe_dir = powheg_dir / "powheg_lhe" / "H3PWHG-000001"
    lhe_dir.mkdir(parents=True)
    target_parton_energy = 0.01
    transverse = math.sqrt(beam_energy * target_parton_energy)
    outgoing_pz = 0.5 * (beam_energy - target_parton_energy)
    outgoing_energy = math.sqrt(transverse**2 + outgoing_pz**2)
    event = f"""<event>
4 10001 -2.5 10.0 0.007297 0.118
12 -1 0 0 0 0 0 0 {beam_energy:.17e} {beam_energy:.17e} 0 0 9
1 -1 0 0 501 0 0 0 -0.01 0.01 0 0 9
11 1 1 2 0 0 {transverse:.17e} 0 {outgoing_pz:.17e} {outgoing_energy:.17e} 0 0 9
2 1 1 2 501 0 {-transverse:.17e} 0 {outgoing_pz:.17e} {outgoing_energy:.17e} 0 0 9
</event>"""
    text = f"""<LesHouchesEvents version="3.0">
<init>
12 2212 {beam_energy:.17e} 0.938272 -1 -1 -1 -1 -4 1
2.5 0.1 1.0 10001
</init>
{event}
{event if duplicate else ''}
</LesHouchesEvents>
"""
    lhe = lhe_dir / "pwgevents.lhe"
    lhe.write_text(text, encoding="utf-8")
    request = {
        "powheg_request_id": "H3PWHG-000001",
        "powheg_lhe_generated": True,
        "powheg_lhe_path": "POWHEG/powheg_lhe/H3PWHG-000001/pwgevents.lhe",
        "interaction_id": "H3DIS-TEST",
        "source_sample_id": "H3SRC-TEST",
        "primary_branch_id": "branch-test",
        "physics_weight": 0.25,
        "observer_weight": 0.5,
        "final_observation_score": 0.125,
    }
    (powheg_dir / "powheg_event_requests.jsonl").write_text(json.dumps(request) + "\n", encoding="utf-8")
    (powheg_dir / "powheg_summary.json").write_text(
        json.dumps({"status": "ok", "powheg_run_mode": "real_smoke", "powheg_lhe_generated": True, "n_lhe_events": 2 if duplicate else 1}) + "\n",
        encoding="utf-8",
    )
    return lhe


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_event_generation_defaults_and_paths_are_active_contract() -> None:
    values = defaults()
    cfg = values["event_generation"]
    assert cfg["mode"] == "disabled"
    assert cfg["backend"] == "pythia8"
    assert cfg["isr_enabled"] is False
    assert cfg["fsr_enabled"] is True
    assert cfg["mpi_enabled"] is False
    assert EVENT_GENERATION_DIR == "EventGeneration"
    assert not validate_values(values)


def test_checked_in_event_generation_schema_covers_official_records() -> None:
    schema = json.loads(Path("schemas/event_generation_contract.schema.json").read_text(encoding="utf-8"))
    assert set(schema["$defs"]) == {"job", "event", "particle", "manifest"}
    assert {"xwgtup", "physics_weight", "observer_weight", "final_observation_score"} <= set(schema["$defs"]["event"]["properties"])
    assert schema["$defs"]["manifest"]["properties"]["stage"]["const"] == "H3-W10"


def test_pdg_ids_are_rendered_as_particle_symbols() -> None:
    assert pdg_symbol(22) == r"$\gamma$"
    assert pdg_symbol(211) == r"$\pi^{+}$"
    assert pdg_symbol(-211) == r"$\pi^{-}$"
    assert pdg_symbol(11) == r"$e^{-}$"
    assert pdg_symbol(-2212) == r"$\bar{p}$"


def test_backend_availability_reports_pinned_toolchain() -> None:
    state = backend_availability()
    assert state["available"] is BACKEND.exists()
    if state["available"]:
        assert state["pythia_version"] == "8.312"
        assert state["hepmc3_version"] == "3.03.01"


def test_parton_check_is_bijective_preserves_weights_and_upstream(tmp_path: Path) -> None:
    lhe = _fixture_run(tmp_path, duplicate=True)
    before = _hash(lhe)
    values = defaults()
    values["event_generation"].update({"mode": "parton_check", "max_events_per_request": 2})

    summary = generate_event_generation_products(values, run_output_dir=tmp_path)
    events = [json.loads(line) for line in (event_generation_dir(tmp_path) / "event_generation_events_summary.jsonl").read_text().splitlines()]
    particles = [json.loads(line) for line in (event_generation_dir(tmp_path) / "event_generation_final_particles.jsonl").read_text().splitlines()]

    assert _hash(lhe) == before
    assert summary["n_events_generated"] == 2
    assert len({row["event_generation_event_id"] for row in events}) == 2
    assert [row["xwgtup"] for row in events] == [-2.5, -2.5]
    assert summary["weight_statistics"]["event_weight_mean"] == -2.5
    assert summary["weight_statistics"]["event_cross_section_estimator_pb"] == -2.5
    assert summary["weight_statistics"]["raw_weight_sum_is_cross_section"] is False
    assert summary["hadronization_invoked"] is False
    assert summary["pythia_invoked"] is False
    assert summary["validation"]["four_momentum_conservation_pass"] is True
    assert events[0]["physics_weight"] == 0.25
    assert events[0]["observer_weight"] == 0.5
    assert events[0]["final_observation_score"] == 0.125
    assert events[0]["xwgtup"] == -2.5
    assert any(row["color1"] == 501 for row in particles)
    assert all("color2" in row for row in particles)
    schema = json.loads(Path("schemas/event_generation_contract.schema.json").read_text(encoding="utf-8"))
    manifest = json.loads((event_generation_dir(tmp_path) / "event_generation_manifest.json").read_text(encoding="utf-8"))
    Draft202012Validator({**schema, "$ref": "#/$defs/manifest"}).validate(manifest)


def test_dry_run_validates_real_lhe_without_invoking_pythia(tmp_path: Path) -> None:
    _fixture_run(tmp_path)
    values = defaults()
    values["event_generation"]["mode"] = "dry_run"
    summary = generate_event_generation_products(values, run_output_dir=tmp_path)
    assert summary["status"] == "ok"
    assert summary["n_events_generated"] == 0
    assert summary["pythia_invoked"] is False
    assert summary["expensive_event_generation_invoked"] is False
    assert (event_generation_dir(tmp_path) / "event_generation_manifest.json").exists()


def test_truncated_lhe_is_rejected_even_in_dry_run(tmp_path: Path) -> None:
    lhe = _fixture_run(tmp_path)
    lhe.write_text("<LesHouchesEvents><event>\n", encoding="utf-8")
    values = defaults()
    values["event_generation"]["mode"] = "dry_run"
    with pytest.raises(ValueError, match="invalid or truncated LHE"):
        generate_event_generation_products(values, run_output_dir=tmp_path)


def test_nonfinite_lhe_is_rejected(tmp_path: Path) -> None:
    lhe = _fixture_run(tmp_path)
    lhe.write_text(lhe.read_text(encoding="utf-8").replace("-2.5 10.0", "nan 10.0"), encoding="utf-8")
    values = defaults()
    values["event_generation"]["mode"] = "parton_check"
    with pytest.raises(ValueError, match="non-finite LHE event field"):
        generate_event_generation_products(values, run_output_dir=tmp_path)


def test_lhe_frame_canonicalization_is_a_common_lorentz_boost(tmp_path: Path) -> None:
    source = _fixture_run(tmp_path)
    text = source.read_text(encoding="utf-8").replace(
        "12 -1 0 0 0 0 0 0 1.00000000000000000e+08 1.00000000000000000e+08",
        "12 -1 0 0 0 0 0 0 1.01000000000000000e+08 1.01000000000000000e+08",
    )
    source.write_text(text, encoding="utf-8")
    target = tmp_path / "canonical.lhe"
    report = _canonicalize_lhe_beam_frame(source, target)
    assert report["max_input_beam_mismatch_relative"] == pytest.approx(0.01)
    assert report["target_mass_gev"] == pytest.approx(0.938272)
    incoming = next(line.split() for line in target.read_text(encoding="utf-8").splitlines() if line.strip().startswith("12 -1"))
    assert float(incoming[8]) == pytest.approx(1.0e8, rel=1e-15)
    assert float(incoming[9]) == pytest.approx(1.0e8, rel=1e-15)


@pytest.mark.skipif(not BACKEND.exists(), reason="PYTHIA backend is not built")
def test_low_energy_backend_keeps_powheg_target_at_rest(tmp_path: Path) -> None:
    lhe = _fixture_run(tmp_path, beam_energy=1.0e4)
    job_dir = tmp_path / "pythia-low-energy"
    job_dir.mkdir()
    cfg = defaults()["event_generation"]
    cfg["write_hepmc3"] = False
    summary = _run_backend(
        lhe,
        {"powheg_request_id": "H3PWHG-000001"},
        job_dir,
        cfg,
        "real_free",
        148004,
        1,
    )
    event = json.loads((job_dir / "events_summary.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert summary["four_momentum_residual_relative_max"] <= 5e-8
    assert event["target_beam_energy_gev"] == pytest.approx(0.938272, rel=1e-12)
    assert abs(event["target_beam_pz_gev"]) <= 1e-9
    assert event["target_rest_frame_residual"] <= 1e-9


def test_rerunning_powheg_invalidates_event_generation(tmp_path: Path) -> None:
    directory = event_generation_dir(tmp_path)
    directory.mkdir(parents=True)
    (directory / "stale.json").write_text("{}")
    clear_powheg_outputs(tmp_path)
    assert directory.exists()
    assert not (directory / "stale.json").exists()


def test_provenance_promotes_successful_event_generation_to_h3_w10(tmp_path: Path) -> None:
    summary = {"status": "ok", "pythia_invoked": True, "event_generation_mode": "real_smoke"}
    provenance = build_provenance(
        root=tmp_path,
        values=defaults(),
        products={},
        validation={},
        event_generation_summary=summary,
    )
    assert provenance["hadros3_stage"] == "H3-W0_to_H3-W10_event_generation"
    assert provenance["status"] == "event_generation_complete"
    assert provenance["disabled_expensive_or_future_stages"]["pythia"] == "active_H3_W10"
    # The scientific reference tracks the latest implemented pipeline even
    # when this isolated provenance fixture stops at H3-W10.
    assert provenance["scientific_theory"]["theory_pipeline_version"] == "H3-W11"


@pytest.mark.skipif(os.environ.get("HADROS3_RUN_REAL_EVENT_GENERATION_TEST") != "1", reason="opt-in PYTHIA 8/HepMC3 integration test")
@pytest.mark.parametrize("perturbative_order", ["LO", "NLO"])
def test_real_event_generation_smoke_passes_numerical_contract(tmp_path: Path, perturbative_order: str) -> None:
    branch_dir = tmp_path / "ObserverImageBranches"
    branch_dir.mkdir()
    candidate = {
        "interaction_id": "H3DIS-EG-SMOKE", "event_id": "H3SRC-EG-SMOKE", "source_sample_id": "H3SRC-EG-SMOKE",
        "interaction_E_nu_local_gev": 1.0e8, "physics_weight": 0.5, "observer_weight": 0.5,
        "final_observation_score": 0.25, "selected_for_downstream": True, "selection_rank": 1,
        "selection_policy": "top_n", "primary_branch_id": "H3DIS-EG-SMOKE:branch-01",
    }
    (branch_dir / "observer_image_primary_branches.jsonl").write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    values = defaults()
    values["powheg"].update({"run_mode": "real_smoke", "perturbative_order": perturbative_order, "events_per_candidate": 1, "random_seed": 68100})
    generate_powheg_products(values, run_output_dir=tmp_path)
    values["event_generation"]["mode"] = "real_smoke"
    summary = generate_event_generation_products(values, run_output_dir=tmp_path)
    canonical_before = (event_generation_dir(tmp_path) / "event_generation_events_summary.jsonl").read_bytes()
    hepmc_before = (event_generation_dir(tmp_path) / "event_generation_events.hepmc3").read_bytes()
    assert summary["pythia_invoked"] is True
    assert summary["hadronization_invoked"] is True
    assert summary["n_events_generated"] == 1
    assert summary["n_final_hadrons"] > 0
    assert summary["n_final_partons"] == 0
    assert summary["validation"]["four_momentum_residual_relative_max"] <= 5e-8
    assert summary["validation"]["onshell_residual_relative_max"] <= 1e-8
    assert (event_generation_dir(tmp_path) / "event_generation_events.hepmc3").exists()
    assert summary["validation"]["matching_scale_pass"] is True
    assert summary["runtime_seconds"] > 0
    assert summary["events_per_second"] > 0
    assert summary["hepmc3_bytes"] > 0
    settings = event_generation_dir(tmp_path) / "jobs" / "H3PWHG-000001" / "pythia.cmnd"
    assert "PartonLevel:ISR = off" in settings.read_text(encoding="utf-8")
    assert "POWHEG:veto = 1" in settings.read_text(encoding="utf-8")
    repeated = generate_event_generation_products(values, run_output_dir=tmp_path)
    assert repeated["n_events_generated"] == summary["n_events_generated"]
    assert (event_generation_dir(tmp_path) / "event_generation_events_summary.jsonl").read_bytes() == canonical_before
    assert (event_generation_dir(tmp_path) / "event_generation_events.hepmc3").read_bytes() == hepmc_before
    content = json.loads((event_generation_dir(tmp_path) / "event_generation_particle_content.json").read_text(encoding="utf-8"))
    assert content["counts_by_particle_class"]["baryon"] > 0
    assert content["counts_by_particle_class"]["meson"] > 0
    assert content["symbols_by_pdg_id"]["22"] == r"$\gamma$"

    if perturbative_order == "LO":
        values["event_generation"]["decays_enabled"] = False
        no_decays = generate_event_generation_products(values, run_output_dir=tmp_path)
        assert no_decays["decays_invoked"] is False
        assert no_decays["n_final_hadrons"] > 0
        assert no_decays["validation"]["four_momentum_conservation_pass"] is True
    else:
        values["event_generation"]["random_seed"] += 1
        changed_seed = generate_event_generation_products(values, run_output_dir=tmp_path)
        assert changed_seed["n_events_generated"] == summary["n_events_generated"]
        assert (event_generation_dir(tmp_path) / "event_generation_events_summary.jsonl").read_bytes() != canonical_before
