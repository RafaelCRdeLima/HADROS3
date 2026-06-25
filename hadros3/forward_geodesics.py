"""HADROS3 H3-W6 forward neutrino geodesic layer."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import validate_values
from .geodesic_outputs import (
    draw_forward_preview,
    write_json,
    write_jsonl,
    write_stop_condition_csv,
    write_summary_csv,
)
from .geodesic_validation import validate_forward_products
from .paths import FORWARD_GEODESICS_DIR, forward_geodesics_dir, uhe_source_dir
from .source_models import KerrNullMomentumGenerator


@dataclass(frozen=True)
class ForwardGeodesicConfig:
    geodesic_backend: str
    n_samples_to_propagate: int
    initial_step_rg: float
    max_steps: int
    outer_radius_rg: float
    horizon_tolerance_rg: float
    null_invariant_tolerance: float
    killing_energy_tolerance: float
    lz_tolerance: float
    spin_a: float


def config_from_values(values: dict[str, dict[str, Any]]) -> ForwardGeodesicConfig:
    forward = values["forward_geodesics"]
    return ForwardGeodesicConfig(
        geodesic_backend=str(forward["geodesic_backend"]),
        n_samples_to_propagate=int(float(forward["n_samples_to_propagate"])),
        initial_step_rg=float(forward["initial_step_rg"]),
        max_steps=int(float(forward["max_steps"])),
        outer_radius_rg=float(forward["outer_radius_rg"]),
        horizon_tolerance_rg=float(forward["horizon_tolerance_rg"]),
        null_invariant_tolerance=float(forward["null_invariant_tolerance"]),
        killing_energy_tolerance=float(forward["killing_energy_tolerance"]),
        lz_tolerance=float(forward["lz_tolerance"]),
        spin_a=float(values["black_hole"]["spin_a"]),
    )


def load_source_samples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"UHE source samples not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def horizon_radius_rg(spin_a: float) -> float:
    a = max(-0.999, min(0.999, spin_a))
    return 1.0 + math.sqrt(1.0 - a * a)


def kerr_inverse_metric_components(r_rg: float, theta_rad: float, spin_a: float) -> dict[str, float]:
    a = spin_a
    sin_theta = math.sin(theta_rad)
    sin2 = max(sin_theta * sin_theta, 1.0e-12)
    cos_theta = math.cos(theta_rad)
    sigma = r_rg * r_rg + a * a * cos_theta * cos_theta
    delta = r_rg * r_rg - 2.0 * r_rg + a * a
    if delta <= 0.0 or sigma <= 0.0:
        raise ValueError("Kerr inverse metric requested inside or at the horizon")
    return {
        "gtt": -(((r_rg * r_rg + a * a) ** 2 - a * a * delta * sin2) / (sigma * delta)),
        "grr": delta / sigma,
        "gthth": 1.0 / sigma,
        "gphph": (delta - a * a * sin2) / (sigma * delta * sin2),
        "gtphi": -2.0 * a * r_rg / (sigma * delta),
    }


def null_norm_covariant(p_t: float, p_r: float, p_theta: float, p_phi: float, r_rg: float, theta_rad: float, spin_a: float) -> float:
    metric = kerr_inverse_metric_components(r_rg, theta_rad, spin_a)
    return (
        metric["gtt"] * p_t * p_t
        + metric["grr"] * p_r * p_r
        + metric["gthth"] * p_theta * p_theta
        + metric["gphph"] * p_phi * p_phi
        + 2.0 * metric["gtphi"] * p_t * p_phi
    )


def emission_direction_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    direction = sample.get("emission_direction")
    if direction is None:
        legacy_momentum = sample.get("initial_momentum") or {}
        legacy_direction_model = legacy_momentum.get("direction_model")
        if legacy_direction_model == "outward_coordinate_radial_proxy":
            direction = {
                "direction_generator": "CoordinateRadialOutwardDirectionGenerator",
                "direction_model": "coordinate_radial_outward",
                "direction_local_components": {
                    "basis": "Boyer-Lindquist_coordinate_direction",
                    "dr": 1.0,
                    "dtheta": 0.0,
                    "dphi": 0.0,
                },
                "direction_sampling_pdf": 1.0,
                "direction_physical_pdf": 1.0,
                "direction_weight": 1.0,
            }
        else:
            direction = {
                "direction_generator": sample.get("direction_generator"),
                "direction_model": sample.get("direction_model"),
                "direction_local_components": sample.get("direction_local_components"),
                "direction_sampling_pdf": sample.get("direction_sampling_pdf"),
                "direction_physical_pdf": sample.get("direction_physical_pdf"),
                "direction_weight": sample.get("direction_weight"),
            }
    if direction.get("direction_model") != "coordinate_radial_outward":
        raise ValueError(f"unsupported source emission direction: {direction.get('direction_model')}")
    components = direction.get("direction_local_components") or {}
    if components.get("basis") != "Boyer-Lindquist_coordinate_direction":
        raise ValueError(f"unsupported source emission direction basis: {components.get('basis')}")
    if not (
        float(components.get("dr", 0.0)) > 0.0
        and float(components.get("dtheta", math.inf)) == 0.0
        and float(components.get("dphi", math.inf)) == 0.0
    ):
        raise ValueError("H3-W6 currently supports only coordinate radial outward emission directions")
    return direction


def kerr_null_momentum(position: dict[str, float], direction: dict[str, Any], energy_gev: float, spin_a: float) -> dict[str, Any]:
    r_rg = float(position["r_rg"])
    theta_rad = float(position["theta_rad"])
    metric = kerr_inverse_metric_components(r_rg, theta_rad, spin_a)
    components = direction["direction_local_components"]
    p_t = -float(energy_gev)
    p_phi = 0.0
    p_theta = 0.0
    radial_sign = 1.0 if float(components["dr"]) >= 0.0 else -1.0
    p_r = math.sqrt(max(0.0, -metric["gtt"] * p_t * p_t / metric["grr"]))
    p_r *= radial_sign
    raw_null_norm = null_norm_covariant(p_t, p_r, p_theta, p_phi, r_rg, theta_rad, spin_a)
    null_norm = raw_null_norm / max(energy_gev * energy_gev, 1.0)
    return {
        "generator": KerrNullMomentumGenerator.name,
        "momentum_is_physical_kerr": True,
        "direction_generator": direction["direction_generator"],
        "direction_model": direction["direction_model"],
        "direction_local_components": direction["direction_local_components"],
        "four_momentum": {
            "basis": "Boyer-Lindquist_covariant",
            "p_t": p_t,
            "p_r": p_r,
            "p_theta": p_theta,
            "p_phi": p_phi,
        },
        "energy_gev": energy_gev,
        "killing_energy_gev": -p_t,
        "lz": p_phi,
        "null_norm": null_norm,
        "raw_null_norm": raw_null_norm,
        "status": "physical_kerr_null_momentum_for_H3_W6_forward_geodesics",
    }


def _sample_subset(samples: list[dict[str, Any]], requested: int) -> list[dict[str, Any]]:
    if requested <= 0:
        return []
    return samples[: min(requested, len(samples))]


def propagate_one(sample: dict[str, Any], config: ForwardGeodesicConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    position = sample["position"]
    energy_gev = float(sample["E_nu_emit_gev"])
    r = float(position["r_rg"])
    theta = float(position["theta_rad"])
    phi = float(position["phi_rad"])
    event_id = str(sample["event_id"])
    source_sample_id = int(sample["source_sample_id"])
    emission_direction = emission_direction_from_sample(sample)
    initial_momentum = kerr_null_momentum(position, emission_direction, energy_gev, config.spin_a)
    p = initial_momentum["four_momentum"]
    p_t = float(p["p_t"])
    p_theta = float(p["p_theta"])
    p_phi = float(p["p_phi"])
    killing_energy0 = -p_t
    lz0 = p_phi
    r_h = horizon_radius_rg(config.spin_a)
    segments: list[dict[str, Any]] = []
    null_errors: list[float] = []
    energy_errors: list[float] = []
    lz_errors: list[float] = []
    stop_condition = "max_steps"
    status = "propagated_forward_no_interaction"
    for segment_index in range(config.max_steps):
        if r <= r_h + config.horizon_tolerance_rg:
            stop_condition = "horizon_crossing"
            break
        if r >= config.outer_radius_rg:
            stop_condition = "outer_escape_radius"
            break
        try:
            momentum = kerr_null_momentum({"r_rg": r, "theta_rad": theta}, emission_direction, energy_gev, config.spin_a)
        except ValueError:
            stop_condition = "horizon_crossing"
            break
        p_mid = momentum["four_momentum"]
        p_r = float(p_mid["p_r"])
        null_norm = float(momentum["null_norm"])
        energy_error = abs(float(momentum["killing_energy_gev"]) - killing_energy0) / max(abs(killing_energy0), 1.0)
        lz_error = abs(float(momentum["lz"]) - lz0)
        null_errors.append(abs(null_norm))
        energy_errors.append(energy_error)
        lz_errors.append(lz_error)
        if abs(null_norm) > config.null_invariant_tolerance:
            stop_condition = "invalid_invariant"
            status = "invalid_invariant"
            break
        step = min(config.initial_step_rg, max(config.outer_radius_rg - r, 0.0))
        if step <= 0.0:
            stop_condition = "outer_escape_radius"
            break
        r_next = r + step
        theta_next = theta
        phi_next = phi
        r_mid = 0.5 * (r + r_next)
        theta_mid = theta
        phi_mid = phi
        segments.append(
            {
                "event_id": event_id,
                "source_sample_id": source_sample_id,
                "segment_index": segment_index,
                "r_start_rg": r,
                "theta_start_rad": theta,
                "phi_start_rad": phi,
                "r_end_rg": r_next,
                "theta_end_rad": theta_next,
                "phi_end_rad": phi_next,
                "r_mid_rg": r_mid,
                "theta_mid_rad": theta_mid,
                "phi_mid_rad": phi_mid,
                "p_t_mid": float(p_mid["p_t"]),
                "p_r_mid": p_r,
                "p_theta_mid": float(p_mid["p_theta"]),
                "p_phi_mid": float(p_mid["p_phi"]),
                "dl_segment_rg": step,
                "E_nu_local_gev_mid": max(energy_gev / math.sqrt(max(1.0 - 2.0 / max(r_mid, 2.000001), 1.0e-8)), energy_gev),
                "geodesic_status": status,
            }
        )
        r, theta, phi = r_next, theta_next, phi_next
    else:
        stop_condition = "max_steps"
    if not segments and stop_condition not in {"horizon_crossing", "outer_escape_radius"}:
        status = "no_segments"
    validation_pass = (
        bool(segments)
        and (max(null_errors) if null_errors else math.inf) <= config.null_invariant_tolerance
        and (max(energy_errors) if energy_errors else math.inf) <= config.killing_energy_tolerance
        and (max(lz_errors) if lz_errors else math.inf) <= config.lz_tolerance
    )
    path_record = {
        "event_id": event_id,
        "source_sample_id": source_sample_id,
        "geodesic_backend": config.geodesic_backend,
        "momentum_generator": KerrNullMomentumGenerator.name,
        "momentum_is_physical_kerr": True,
        "direction_generator": emission_direction["direction_generator"],
        "direction_model": emission_direction["direction_model"],
        "emission_direction": emission_direction,
        "initial_position": position,
        "initial_momentum": initial_momentum,
        "n_segments": len(segments),
        "stop_condition": stop_condition,
        "geodesic_status": status,
        "null_norm_max_abs": max(null_errors) if null_errors else math.inf,
        "killing_energy_max_error": max(energy_errors) if energy_errors else math.inf,
        "lz_max_error": max(lz_errors) if lz_errors else math.inf,
        "validation_pass": validation_pass,
        "optical_depth_dis_sampler_invoked": False,
        "observer_bridge_active_filter_invoked": False,
        "expensive_event_generation_invoked": False,
    }
    return path_record, segments


def generate_forward_geodesic_products(values: dict[str, dict[str, Any]], *, run_output_dir: Path) -> dict[str, Any]:
    config_problems = validate_values(values)
    if config_problems:
        raise ValueError("Invalid HADROS3 configuration:\n- " + "\n- ".join(config_problems))
    config = config_from_values(values)
    source_path = uhe_source_dir(run_output_dir) / "uhe_neutrino_source_samples.jsonl"
    output_dir = forward_geodesics_dir(run_output_dir)
    if output_dir.name != FORWARD_GEODESICS_DIR:
        output_dir = output_dir / FORWARD_GEODESICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = _sample_subset(load_source_samples(source_path), config.n_samples_to_propagate)
    paths: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    for sample in samples:
        path_record, path_segments = propagate_one(sample, config)
        paths.append(path_record)
        segments.extend(path_segments)
    validation_errors, validation_report, stop_counts = validate_forward_products(
        paths,
        segments,
        expected_paths=len(samples),
        null_tolerance=config.null_invariant_tolerance,
        killing_energy_tolerance=config.killing_energy_tolerance,
        lz_tolerance=config.lz_tolerance,
    )
    paths_path = output_dir / "uhe_neutrino_forward_paths.jsonl"
    segments_path = output_dir / "uhe_neutrino_forward_path_segments.jsonl"
    summary_csv_path = output_dir / "uhe_neutrino_forward_summary.csv"
    summary_json_path = output_dir / "uhe_neutrino_forward_summary.json"
    preview_path = output_dir / "uhe_neutrino_forward_preview.png"
    validation_path = output_dir / "geodesic_validation_report.json"
    stop_path = output_dir / "stop_condition_statistics.csv"
    summary = {
        "status": "ok" if not validation_errors else "validation_failed",
        "forward_neutrino_geodesics_invoked": True,
        "momentum_generator": KerrNullMomentumGenerator.name,
        "momentum_is_physical_kerr": True,
        "direction_generator": samples[0].get("direction_generator") if samples else None,
        "direction_model": samples[0].get("direction_model") if samples else None,
        "input_source_samples": str(source_path),
        "geodesic_backend": config.geodesic_backend,
        "n_samples_requested": config.n_samples_to_propagate,
        "n_samples_propagated": len(samples),
        "n_paths": len(paths),
        "n_segments": len(segments),
        "max_steps": config.max_steps,
        "initial_step_rg": config.initial_step_rg,
        "outer_radius_rg": config.outer_radius_rg,
        "horizon_tolerance_rg": config.horizon_tolerance_rg,
        "null_invariant_tolerance": config.null_invariant_tolerance,
        "killing_energy_tolerance": config.killing_energy_tolerance,
        "lz_tolerance": config.lz_tolerance,
        "stop_condition_counts": stop_counts,
        "validation_errors": validation_errors,
        **validation_report,
        "optical_depth_dis_sampler_invoked": False,
        "observer_bridge_active_filter_invoked": False,
        "expensive_event_generation_invoked": False,
        "products": {
            "forward_paths": str(paths_path),
            "forward_path_segments": str(segments_path),
            "forward_summary": str(summary_csv_path),
            "forward_summary_json": str(summary_json_path),
            "forward_preview": str(preview_path),
            "geodesic_validation_report": str(validation_path),
            "stop_condition_statistics": str(stop_path),
        },
    }
    write_jsonl(paths, paths_path)
    write_jsonl(segments, segments_path)
    write_summary_csv(summary, summary_csv_path)
    write_json(summary_json_path, summary)
    write_json(validation_path, validation_report)
    write_stop_condition_csv(stop_counts, len(paths), stop_path)
    draw_forward_preview(paths, segments, preview_path)
    return summary
