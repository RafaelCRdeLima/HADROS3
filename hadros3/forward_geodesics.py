"""HADROS3 H3-W6 forward neutrino geodesic layer."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import validate_values
from .geodesic_outputs import (
    draw_forward_geometry_3d,
    draw_forward_preview,
    write_diagnostic_report,
    write_json,
    write_jsonl,
    write_stop_condition_csv,
    write_summary_csv,
)
from .geodesic_validation import validate_forward_products
from .paths import FORWARD_GEODESICS_DIR, forward_geodesics_dir, uhe_source_dir
from .source_models import IsotropicLocalDirectionGenerator, KerrNullMomentumGenerator


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


@dataclass(frozen=True)
class KerrGeodesicState:
    t: float
    r: float
    theta: float
    phi: float
    p_t: float
    p_r: float
    p_theta: float
    p_phi: float


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


def kerr_covariant_metric_components(r_rg: float, theta_rad: float, spin_a: float) -> dict[str, float]:
    a = spin_a
    sin_theta = math.sin(theta_rad)
    sin2 = max(sin_theta * sin_theta, 1.0e-12)
    cos_theta = math.cos(theta_rad)
    sigma = r_rg * r_rg + a * a * cos_theta * cos_theta
    delta = r_rg * r_rg - 2.0 * r_rg + a * a
    if delta <= 0.0 or sigma <= 0.0:
        raise ValueError("Kerr metric requested inside or at the horizon")
    big_a = (r_rg * r_rg + a * a) ** 2 - a * a * delta * sin2
    return {
        "gtt": -(1.0 - 2.0 * r_rg / sigma),
        "grr": sigma / delta,
        "gthth": sigma,
        "gphph": big_a * sin2 / sigma,
        "gtphi": -2.0 * a * r_rg * sin2 / sigma,
        "sigma": sigma,
        "delta": delta,
        "A": big_a,
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


def covector_from_contravariant(p_t: float, p_r: float, p_theta: float, p_phi: float, r_rg: float, theta_rad: float, spin_a: float) -> dict[str, float]:
    metric = kerr_covariant_metric_components(r_rg, theta_rad, spin_a)
    return {
        "p_t": metric["gtt"] * p_t + metric["gtphi"] * p_phi,
        "p_r": metric["grr"] * p_r,
        "p_theta": metric["gthth"] * p_theta,
        "p_phi": metric["gtphi"] * p_t + metric["gphph"] * p_phi,
    }


def zamo_covariant_momentum(
    r_rg: float,
    theta_rad: float,
    spin_a: float,
    energy_gev: float,
    n_r: float,
    n_theta: float,
    n_phi: float,
) -> dict[str, float]:
    metric = kerr_covariant_metric_components(r_rg, theta_rad, spin_a)
    sigma = metric["sigma"]
    delta = metric["delta"]
    big_a = metric["A"]
    sin_theta = max(math.sin(theta_rad), 1.0e-12)
    lapse = math.sqrt(max(sigma * delta / big_a, 1.0e-30))
    omega = 2.0 * spin_a * r_rg / big_a
    e_t = (1.0 / lapse, 0.0, 0.0, omega / lapse)
    e_r = (0.0, math.sqrt(delta / sigma), 0.0, 0.0)
    e_theta = (0.0, 0.0, 1.0 / math.sqrt(sigma), 0.0)
    e_phi = (0.0, 0.0, 0.0, math.sqrt(sigma / big_a) / sin_theta)
    p_contra = tuple(
        energy_gev * (e_t[i] + n_r * e_r[i] + n_theta * e_theta[i] + n_phi * e_phi[i])
        for i in range(4)
    )
    return covector_from_contravariant(p_contra[0], p_contra[1], p_contra[2], p_contra[3], r_rg, theta_rad, spin_a)


def _metric_component(metric: dict[str, float], mu: int, nu: int) -> float:
    if mu == 0 and nu == 0:
        return metric["gtt"]
    if mu == 1 and nu == 1:
        return metric["grr"]
    if mu == 2 and nu == 2:
        return metric["gthth"]
    if mu == 3 and nu == 3:
        return metric["gphph"]
    if {mu, nu} == {0, 3}:
        return metric["gtphi"]
    return 0.0


def kerr_inverse_metric_derivatives(r_rg: float, theta_rad: float, spin_a: float) -> tuple[list[list[float]], list[list[float]]]:
    dr = max(1.0e-5, abs(r_rg) * 1.0e-5)
    dtheta = 1.0e-5
    r_h = horizon_radius_rg(spin_a)
    r_minus = max(r_h + 1.0e-5, r_rg - dr)
    r_plus = r_rg + dr
    theta_minus = max(1.0e-6, theta_rad - dtheta)
    theta_plus = min(math.pi - 1.0e-6, theta_rad + dtheta)
    metric_r_minus = kerr_inverse_metric_components(r_minus, theta_rad, spin_a)
    metric_r_plus = kerr_inverse_metric_components(r_plus, theta_rad, spin_a)
    metric_theta_minus = kerr_inverse_metric_components(r_rg, theta_minus, spin_a)
    metric_theta_plus = kerr_inverse_metric_components(r_rg, theta_plus, spin_a)
    dgdr = [[0.0 for _ in range(4)] for _ in range(4)]
    dgdtheta = [[0.0 for _ in range(4)] for _ in range(4)]
    for mu in range(4):
        for nu in range(4):
            dgdr[mu][nu] = (_metric_component(metric_r_plus, mu, nu) - _metric_component(metric_r_minus, mu, nu)) / (r_plus - r_minus)
            dgdtheta[mu][nu] = (_metric_component(metric_theta_plus, mu, nu) - _metric_component(metric_theta_minus, mu, nu)) / (theta_plus - theta_minus)
    return dgdr, dgdtheta


def hamiltonian_null_norm(state: KerrGeodesicState, spin_a: float) -> float:
    return null_norm_covariant(state.p_t, state.p_r, state.p_theta, state.p_phi, state.r, state.theta, spin_a)


def kerr_geodesic_rhs(state: KerrGeodesicState, spin_a: float) -> KerrGeodesicState:
    metric = kerr_inverse_metric_components(state.r, state.theta, spin_a)
    covector = [state.p_t, state.p_r, state.p_theta, state.p_phi]
    dx = [0.0, 0.0, 0.0, 0.0]
    for mu in range(4):
        for nu in range(4):
            dx[mu] += _metric_component(metric, mu, nu) * covector[nu]
    dgdr, dgdtheta = kerr_inverse_metric_derivatives(state.r, state.theta, spin_a)
    dp_r = 0.0
    dp_theta = 0.0
    for mu in range(4):
        for nu in range(4):
            product = covector[mu] * covector[nu]
            dp_r -= 0.5 * dgdr[mu][nu] * product
            dp_theta -= 0.5 * dgdtheta[mu][nu] * product
    return KerrGeodesicState(
        t=dx[0],
        r=dx[1],
        theta=dx[2],
        phi=dx[3],
        p_t=0.0,
        p_r=dp_r,
        p_theta=dp_theta,
        p_phi=0.0,
    )


def _state_add(state: KerrGeodesicState, *terms: tuple[float, KerrGeodesicState]) -> KerrGeodesicState:
    values = {
        "t": state.t,
        "r": state.r,
        "theta": state.theta,
        "phi": state.phi,
        "p_t": state.p_t,
        "p_r": state.p_r,
        "p_theta": state.p_theta,
        "p_phi": state.p_phi,
    }
    for scale, term in terms:
        values["t"] += scale * term.t
        values["r"] += scale * term.r
        values["theta"] += scale * term.theta
        values["phi"] += scale * term.phi
        values["p_t"] += scale * term.p_t
        values["p_r"] += scale * term.p_r
        values["p_theta"] += scale * term.p_theta
        values["p_phi"] += scale * term.p_phi
    return KerrGeodesicState(**values)


def rk4_step(state: KerrGeodesicState, h: float, spin_a: float) -> KerrGeodesicState:
    k1 = kerr_geodesic_rhs(state, spin_a)
    k2 = kerr_geodesic_rhs(_state_add(state, (0.5 * h, k1)), spin_a)
    k3 = kerr_geodesic_rhs(_state_add(state, (0.5 * h, k2)), spin_a)
    k4 = kerr_geodesic_rhs(_state_add(state, (h, k3)), spin_a)
    return _state_add(state, (h / 6.0, k1), (h / 3.0, k2), (h / 3.0, k3), (h / 6.0, k4))


def renormalize_null_pr(state: KerrGeodesicState, spin_a: float, preferred_sign: float) -> KerrGeodesicState:
    if not (state.r > horizon_radius_rg(spin_a) and 1.0e-6 < state.theta < math.pi - 1.0e-6):
        return state
    metric = kerr_inverse_metric_components(state.r, state.theta, spin_a)
    rest = (
        metric["gtt"] * state.p_t * state.p_t
        + metric["gthth"] * state.p_theta * state.p_theta
        + metric["gphph"] * state.p_phi * state.p_phi
        + 2.0 * metric["gtphi"] * state.p_t * state.p_phi
    )
    value = -rest / metric["grr"]
    if value < 0.0:
        return state
    sign = 1.0 if (state.p_r if state.p_r != 0.0 else preferred_sign) >= 0.0 else -1.0
    return KerrGeodesicState(
        t=state.t,
        r=state.r,
        theta=state.theta,
        phi=state.phi,
        p_t=state.p_t,
        p_r=sign * math.sqrt(value),
        p_theta=state.p_theta,
        p_phi=state.p_phi,
    )


def normalize_boyer_lindquist_polar_crossing(state: KerrGeodesicState) -> KerrGeodesicState:
    theta = state.theta
    phi = state.phi
    p_theta = state.p_theta
    while theta < 0.0 or theta > math.pi:
        if theta < 0.0:
            theta = -theta
            phi += math.pi
            p_theta = -p_theta
        elif theta > math.pi:
            theta = 2.0 * math.pi - theta
            phi += math.pi
            p_theta = -p_theta
    eps = 1.0e-6
    if theta < eps:
        theta = eps
    elif theta > math.pi - eps:
        theta = math.pi - eps
    return KerrGeodesicState(
        t=state.t,
        r=state.r,
        theta=theta,
        phi=phi,
        p_t=state.p_t,
        p_r=state.p_r,
        p_theta=p_theta,
        p_phi=state.p_phi,
    )


def coordinate_path_distance(a: KerrGeodesicState, b: KerrGeodesicState) -> float:
    r_mid = max(0.5 * (a.r + b.r), 1.0e-6)
    theta_mid = 0.5 * (a.theta + b.theta)
    dphi = math.atan2(math.sin(b.phi - a.phi), math.cos(b.phi - a.phi))
    return math.sqrt((b.r - a.r) ** 2 + (r_mid * (b.theta - a.theta)) ** 2 + (r_mid * max(math.sin(theta_mid), 1.0e-6) * dphi) ** 2)


def physical_momentum_from_state(state: KerrGeodesicState, energy_gev: float) -> dict[str, float]:
    return {
        "p_t": state.p_t * energy_gev,
        "p_r": state.p_r * energy_gev,
        "p_theta": state.p_theta * energy_gev,
        "p_phi": state.p_phi * energy_gev,
    }


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
    if direction.get("direction_model") not in {"coordinate_radial_outward", "isotropic_local"}:
        raise ValueError(f"unsupported source emission direction: {direction.get('direction_model')}")
    components = direction.get("direction_local_components") or {}
    if components.get("basis") not in {"Boyer-Lindquist_coordinate_direction", "ZAMO_orthonormal"}:
        raise ValueError(f"unsupported source emission direction basis: {components.get('basis')}")
    if direction.get("direction_model") == "coordinate_radial_outward":
        if not (
            float(components.get("dr", 0.0)) > 0.0
            and float(components.get("dtheta", math.inf)) == 0.0
            and float(components.get("dphi", math.inf)) == 0.0
        ):
            raise ValueError("coordinate radial outward direction must have dr>0, dtheta=0, dphi=0")
    if direction.get("direction_model") == "isotropic_local":
        norm = math.sqrt(
            float(components.get("n_r", math.inf)) ** 2
            + float(components.get("n_theta", math.inf)) ** 2
            + float(components.get("n_phi", math.inf)) ** 2
        )
        if components.get("basis") != "ZAMO_orthonormal" or abs(norm - 1.0) > 1.0e-10:
            raise ValueError("isotropic_local direction must be a unit vector in ZAMO_orthonormal basis")
    return direction


def kerr_null_momentum(position: dict[str, float], direction: dict[str, Any], energy_gev: float, spin_a: float) -> dict[str, Any]:
    r_rg = float(position["r_rg"])
    theta_rad = float(position["theta_rad"])
    components = direction["direction_local_components"]
    if direction["direction_model"] == "isotropic_local":
        covector = zamo_covariant_momentum(
            r_rg,
            theta_rad,
            spin_a,
            float(energy_gev),
            float(components["n_r"]),
            float(components["n_theta"]),
            float(components["n_phi"]),
        )
        p_t = covector["p_t"]
        p_r = covector["p_r"]
        p_theta = covector["p_theta"]
        p_phi = covector["p_phi"]
        momentum_basis = "Boyer-Lindquist_covariant_from_ZAMO_orthonormal"
    else:
        metric = kerr_inverse_metric_components(r_rg, theta_rad, spin_a)
        p_t = -float(energy_gev)
        p_phi = 0.0
        p_theta = 0.0
        radial_sign = 1.0 if float(components["dr"]) >= 0.0 else -1.0
        p_r = math.sqrt(max(0.0, -metric["gtt"] * p_t * p_t / metric["grr"]))
        p_r *= radial_sign
        momentum_basis = "Boyer-Lindquist_covariant"
    raw_null_norm = null_norm_covariant(p_t, p_r, p_theta, p_phi, r_rg, theta_rad, spin_a)
    null_norm = raw_null_norm / max(energy_gev * energy_gev, 1.0)
    return {
        "generator": KerrNullMomentumGenerator.name,
        "momentum_is_physical_kerr": True,
        "direction_generator": direction["direction_generator"],
        "direction_model": direction["direction_model"],
        "direction_local_components": direction["direction_local_components"],
        "four_momentum": {
            "basis": momentum_basis,
            "p_t": p_t,
            "p_r": p_r,
            "p_theta": p_theta,
            "p_phi": p_phi,
        },
        "energy_gev": energy_gev,
        "killing_energy_gev": -p_t,
        "lz": p_phi,
        "local_tetrad": "ZAMO" if direction["direction_model"] == "isotropic_local" else None,
        "null_norm": null_norm,
        "raw_null_norm": raw_null_norm,
        "status": "physical_kerr_null_momentum_for_H3_W6_forward_geodesics",
    }


def _sample_subset(samples: list[dict[str, Any]], requested: int) -> list[dict[str, Any]]:
    if requested <= 0:
        return []
    return samples[: min(requested, len(samples))]


def _synthetic_strong_field_samples(n_samples: int, spin_a: float, energy_gev: float) -> list[dict[str, Any]]:
    r_h = horizon_radius_rg(spin_a)
    direction_generator = IsotropicLocalDirectionGenerator(seed=91991)
    samples: list[dict[str, Any]] = []
    for sample_id in range(n_samples):
        phi = 2.0 * math.pi * sample_id / max(n_samples, 1)
        theta = 0.5 * math.pi + 0.08 * math.sin(3.0 * phi)
        r = r_h + 0.34 + 0.10 * ((sample_id % 5) / 4.0)
        position = {
            "t": 0.0,
            "r_rg": r,
            "theta_rad": theta,
            "theta_deg": math.degrees(theta),
            "phi_rad": phi,
            "phi_deg": math.degrees(phi),
        }
        direction = direction_generator.sample(position, energy_gev, sample_id)
        samples.append(
            {
                "source_sample_id": sample_id,
                "event_id": f"H3STRONG-{sample_id:06d}",
                "position": position,
                "x_emit_r": r,
                "x_emit_theta": theta,
                "x_emit_phi": phi,
                "E_nu_emit_gev": energy_gev,
                "emission_direction": direction,
                "direction_generator": direction["direction_generator"],
                "direction_model": direction["direction_model"],
                "direction_local_components": direction["direction_local_components"],
                "direction_sampling_pdf": direction["direction_sampling_pdf"],
                "direction_physical_pdf": direction["direction_physical_pdf"],
                "direction_weight": direction["direction_weight"],
            }
        )
    return samples


def propagate_one(sample: dict[str, Any], config: ForwardGeodesicConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    position = sample["position"]
    energy_gev = float(sample["E_nu_emit_gev"])
    event_id = str(sample["event_id"])
    source_sample_id = int(sample["source_sample_id"])
    emission_direction = emission_direction_from_sample(sample)
    initial_momentum = kerr_null_momentum(position, emission_direction, energy_gev, config.spin_a)
    p = initial_momentum["four_momentum"]
    state = KerrGeodesicState(
        t=0.0,
        r=float(position["r_rg"]),
        theta=float(position["theta_rad"]),
        phi=float(position["phi_rad"]),
        p_t=float(p["p_t"]) / energy_gev,
        p_r=float(p["p_r"]) / energy_gev,
        p_theta=float(p["p_theta"]) / energy_gev,
        p_phi=float(p["p_phi"]) / energy_gev,
    )
    p_t = state.p_t
    p_phi = state.p_phi
    killing_energy0 = -p_t
    lz0 = p_phi
    r_h = horizon_radius_rg(config.spin_a)
    segments: list[dict[str, Any]] = []
    null_errors: list[float] = []
    energy_errors: list[float] = []
    lz_errors: list[float] = []
    stop_condition = "max_steps"
    status = "propagated_forward_no_interaction"
    preferred_pr_sign = 1.0 if state.p_r >= 0.0 else -1.0
    initial_theta = state.theta
    initial_phi = state.phi
    max_delta_theta = 0.0
    max_delta_phi = 0.0
    max_curvature_indicator = 0.0
    for segment_index in range(config.max_steps):
        if state.r <= r_h + config.horizon_tolerance_rg:
            stop_condition = "horizon_crossing"
            break
        if state.r >= config.outer_radius_rg:
            stop_condition = "outer_escape_radius"
            break
        start = state
        integrated_distance = 0.0
        substeps = 0
        previous_rhs = kerr_geodesic_rhs(state, config.spin_a)
        while integrated_distance < config.initial_step_rg and substeps < 256:
            if state.r <= r_h + config.horizon_tolerance_rg:
                stop_condition = "horizon_crossing"
                break
            if state.r >= config.outer_radius_rg:
                stop_condition = "outer_escape_radius"
                break
            rhs = kerr_geodesic_rhs(state, config.spin_a)
            coordinate_speed = math.sqrt(
                rhs.r * rhs.r
                + (state.r * rhs.theta) ** 2
                + (state.r * max(math.sin(state.theta), 1.0e-6) * rhs.phi) ** 2
            )
            if coordinate_speed <= 0.0 or not math.isfinite(coordinate_speed):
                stop_condition = "invalid_invariant"
                status = "invalid_invariant"
                break
            target_step = max(config.initial_step_rg / 4.0, 1.0e-3)
            polar_cap = 0.025 if min(state.theta, math.pi - state.theta) < 0.05 else 0.2
            horizon_cap = 0.025 if state.r < r_h + 0.75 else 0.2
            h = min(0.2, polar_cap, horizon_cap, max(1.0e-5, target_step / coordinate_speed))
            next_state = None
            null_norm_next = math.inf
            for _attempt in range(10):
                try:
                    candidate = rk4_step(state, h, config.spin_a)
                    candidate = normalize_boyer_lindquist_polar_crossing(candidate)
                    candidate = renormalize_null_pr(candidate, config.spin_a, preferred_pr_sign)
                    if not (math.isfinite(candidate.r) and math.isfinite(candidate.theta) and math.isfinite(candidate.phi)):
                        raise ValueError("non-finite geodesic state")
                    if not (1.0e-6 < candidate.theta < math.pi - 1.0e-6):
                        raise ValueError("geodesic reached Boyer-Lindquist polar coordinate singularity")
                    candidate_null_norm = hamiltonian_null_norm(candidate, config.spin_a)
                    if abs(candidate_null_norm) <= config.null_invariant_tolerance:
                        next_state = candidate
                        null_norm_next = candidate_null_norm
                        break
                except ValueError:
                    pass
                h *= 0.5
                if h < 1.0e-8:
                    break
            if next_state is None or abs(null_norm_next) > config.null_invariant_tolerance:
                stop_condition = "invalid_invariant"
                status = "invalid_invariant"
                break
            integrated_distance += coordinate_path_distance(state, next_state)
            max_curvature_indicator = max(
                max_curvature_indicator,
                abs(rhs.theta - previous_rhs.theta),
                abs(rhs.phi - previous_rhs.phi),
            )
            previous_rhs = rhs
            state = next_state
            substeps += 1
        if stop_condition in {"horizon_crossing", "outer_escape_radius"} and coordinate_path_distance(start, state) <= 0.0:
            break
        if status == "invalid_invariant":
            break
        if coordinate_path_distance(start, state) <= 0.0:
            stop_condition = "max_steps"
            break
        physical_p = physical_momentum_from_state(state, energy_gev)
        null_norm = hamiltonian_null_norm(state, config.spin_a)
        energy_error = abs((-state.p_t) - killing_energy0) / max(abs(killing_energy0), 1.0)
        lz_error = abs(state.p_phi - lz0)
        null_errors.append(abs(null_norm))
        energy_errors.append(energy_error)
        lz_errors.append(lz_error)
        if abs(null_norm) > config.null_invariant_tolerance:
            stop_condition = "invalid_invariant"
            status = "invalid_invariant"
            break
        delta_phi_total = math.atan2(math.sin(state.phi - initial_phi), math.cos(state.phi - initial_phi))
        max_delta_theta = max(max_delta_theta, abs(state.theta - initial_theta))
        max_delta_phi = max(max_delta_phi, abs(delta_phi_total))
        r_mid = 0.5 * (start.r + state.r)
        theta_mid = 0.5 * (start.theta + state.theta)
        phi_mid = start.phi + 0.5 * math.atan2(math.sin(state.phi - start.phi), math.cos(state.phi - start.phi))
        segments.append(
            {
                "event_id": event_id,
                "source_sample_id": source_sample_id,
                "segment_index": segment_index,
                "r_start_rg": start.r,
                "theta_start_rad": start.theta,
                "phi_start_rad": start.phi,
                "r_end_rg": state.r,
                "theta_end_rad": state.theta,
                "phi_end_rad": state.phi,
                "r_mid_rg": r_mid,
                "theta_mid_rad": theta_mid,
                "phi_mid_rad": phi_mid,
                "p_t_mid": physical_p["p_t"],
                "p_r_mid": physical_p["p_r"],
                "p_theta_mid": physical_p["p_theta"],
                "p_phi_mid": physical_p["p_phi"],
                "dl_segment_rg": coordinate_path_distance(start, state),
                "E_nu_local_gev_mid": max(energy_gev / math.sqrt(max(1.0 - 2.0 / max(r_mid, 2.000001), 1.0e-8)), energy_gev),
                "geodesic_status": status,
                "full_kerr_geodesic": True,
                "theta_phi_evolution": True,
                "uses_kerr_metric": True,
                "uses_christoffel_or_hamiltonian": True,
            }
        )
    else:
        stop_condition = "max_steps"
    if not segments and stop_condition not in {"horizon_crossing", "outer_escape_radius"}:
        status = "no_segments"
    validation_pass = (
        bool(segments)
        and stop_condition != "invalid_invariant"
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
        "max_delta_theta_rad": max_delta_theta,
        "max_delta_phi_rad": max_delta_phi,
        "curvature_indicator_max": max_curvature_indicator,
        "validation_pass": validation_pass,
        "full_kerr_geodesic": True,
        "theta_phi_evolution": True,
        "uses_kerr_metric": True,
        "uses_christoffel_or_hamiltonian": True,
        "coordinate_radial_preview": False,
        "optical_depth_dis_sampler_invoked": False,
        "observer_bridge_active_filter_invoked": False,
        "expensive_event_generation_invoked": False,
    }
    return path_record, segments


def generate_strong_field_diagnostic(values: dict[str, dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    spin_a = 0.999
    diagnostic_config = ForwardGeodesicConfig(
        geodesic_backend="full_kerr_geodesic",
        n_samples_to_propagate=48,
        initial_step_rg=0.45,
        max_steps=180,
        outer_radius_rg=35.0,
        horizon_tolerance_rg=1.0e-4,
        null_invariant_tolerance=1.0e-6,
        killing_energy_tolerance=1.0e-10,
        lz_tolerance=1.0e-10,
        spin_a=spin_a,
    )
    samples = _synthetic_strong_field_samples(diagnostic_config.n_samples_to_propagate, spin_a, 1.0e9)
    paths: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    for sample in samples:
        path_record, path_segments = propagate_one(sample, diagnostic_config)
        paths.append(path_record)
        segments.extend(path_segments)
    validation_errors, validation_report, stop_counts = validate_forward_products(
        paths,
        segments,
        expected_paths=len(samples),
        null_tolerance=diagnostic_config.null_invariant_tolerance,
        killing_energy_tolerance=diagnostic_config.killing_energy_tolerance,
        lz_tolerance=diagnostic_config.lz_tolerance,
    )
    min_radius = min(
        [float(segment["r_start_rg"]) for segment in segments] + [float(segment["r_end_rg"]) for segment in segments],
        default=math.inf,
    )
    max_delta_phi = max((float(path.get("max_delta_phi_rad", 0.0)) for path in paths), default=0.0)
    max_delta_theta = max((float(path.get("max_delta_theta_rad", 0.0)) for path in paths), default=0.0)
    png_path = output_dir / "isotropic_kerr_strong_field_diagnostic.png"
    json_path = output_dir / "isotropic_kerr_strong_field_diagnostic.json"
    diagnostic_values = {
        **values,
        "black_hole": {**values["black_hole"], "spin_a": spin_a},
        "forward_geodesics": {
            **values["forward_geodesics"],
            "geodesic_backend": "full_kerr_geodesic",
            "outer_radius_rg": diagnostic_config.outer_radius_rg,
        },
        "observer_camera": {
            **values["observer_camera"],
            "observer_distance_rg": 55.0,
            "inclination_deg": 68.0,
            "azimuth_deg": -35.0,
            "field_of_view_deg": 70.0,
        },
        "polar_cone": {**values["polar_cone"], "opening_angle_deg": 45.0, "r_min_rg": 1.05, "r_max_rg": 45.0},
    }
    draw_forward_geometry_3d(diagnostic_values, paths, segments, png_path, json_path)
    payload = {
        "status": "ok" if not validation_errors else "validation_failed",
        "diagnostic": "isotropic_kerr_strong_field",
        "spin_a": spin_a,
        "source_model": "synthetic_near_horizon_isotropic_local_zamo",
        "direction_model": "isotropic_local",
        "n_geodesics": len(paths),
        "n_escape": int(stop_counts.get("outer_escape_radius", 0)),
        "n_horizon_crossing": int(stop_counts.get("horizon_crossing", 0)),
        "n_max_steps": int(stop_counts.get("max_steps", 0)),
        "n_invalid_invariant": int(stop_counts.get("invalid_invariant", 0)),
        "max_delta_phi_rad": max_delta_phi,
        "max_delta_theta_rad": max_delta_theta,
        "min_radius_reached": min_radius,
        "null_norm_max": validation_report["null_norm_max"],
        "killing_energy_max_error": validation_report["killing_energy_max_error"],
        "lz_max_error": validation_report["lz_max_error"],
        "validation_pass": validation_report["validation_pass"],
        "stop_condition_counts": stop_counts,
        "visual_pass": max_delta_phi > 0.5 and max_delta_theta > 0.05 and (stop_counts.get("horizon_crossing", 0) > 0),
        "classification": (
            "PASS: isotropic local directions produce Kerr bending"
            if max_delta_phi > 0.5 and max_delta_theta > 0.05 and (stop_counts.get("horizon_crossing", 0) > 0)
            else "FAIL: isotropic local directions are not producing Kerr bending"
        ),
        "products": {
            "isotropic_kerr_strong_field_diagnostic_png": str(png_path),
            "isotropic_kerr_strong_field_diagnostic_json": str(json_path),
        },
    }
    write_json(json_path, payload)
    return payload


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
    source_samples = load_source_samples(source_path)
    samples = _sample_subset(source_samples, config.n_samples_to_propagate)
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
    max_delta_theta = max((float(path.get("max_delta_theta_rad", 0.0)) for path in paths), default=0.0)
    max_delta_phi = max((float(path.get("max_delta_phi_rad", 0.0)) for path in paths), default=0.0)
    curvature_indicator_max = max((float(path.get("curvature_indicator_max", 0.0)) for path in paths), default=0.0)
    paths_path = output_dir / "uhe_neutrino_forward_paths.jsonl"
    segments_path = output_dir / "uhe_neutrino_forward_path_segments.jsonl"
    summary_csv_path = output_dir / "uhe_neutrino_forward_summary.csv"
    summary_json_path = output_dir / "uhe_neutrino_forward_summary.json"
    preview_path = output_dir / "uhe_neutrino_forward_preview.png"
    geometry_3d_path = output_dir / "uhe_neutrino_forward_geometry_3d.png"
    geometry_3d_json_path = output_dir / "uhe_neutrino_forward_geometry_3d.json"
    geometry_3d_html_path = output_dir / "uhe_neutrino_forward_geometry_3d.html"
    strong_diagnostic_png_path = output_dir / "isotropic_kerr_strong_field_diagnostic.png"
    strong_diagnostic_json_path = output_dir / "isotropic_kerr_strong_field_diagnostic.json"
    validation_path = output_dir / "geodesic_validation_report.json"
    stop_path = output_dir / "stop_condition_statistics.csv"
    diagnostic_path = output_dir / "forward_geodesics_diagnostic_report.md"
    summary = {
        "status": "ok" if not validation_errors else "validation_failed",
        "forward_neutrino_geodesics_invoked": True,
        "momentum_generator": KerrNullMomentumGenerator.name,
        "momentum_is_physical_kerr": True,
        "direction_generator": samples[0].get("direction_generator") if samples else None,
        "direction_model": samples[0].get("direction_model") if samples else None,
        "input_source_samples": str(source_path),
        "n_input_samples": len(source_samples),
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
        "max_delta_theta_rad": max_delta_theta,
        "max_delta_phi_rad": max_delta_phi,
        "curvature_indicator_max": curvature_indicator_max,
        "full_kerr_geodesic": True,
        "theta_phi_evolution": True,
        "uses_kerr_metric": True,
        "uses_christoffel_or_hamiltonian": True,
        "coordinate_radial_preview": False,
        "strong_field_diagnostic": None,
        "validation_errors": validation_errors,
        **validation_report,
        "optical_depth_dis_sampler_invoked": False,
        "observer_bridge_active_filter_invoked": False,
        "expensive_event_generation_invoked": False,
        "forward_geodesics_consumes_source_direction": True,
        "four_momentum_constructed_from_source_direction": True,
        "four_momentum_sampled_in_source": False,
        "products": {
            "forward_paths": str(paths_path),
            "forward_path_segments": str(segments_path),
            "forward_summary": str(summary_csv_path),
            "forward_summary_json": str(summary_json_path),
            "forward_preview": str(preview_path),
            "forward_geometry_3d": str(geometry_3d_path),
            "forward_geometry_3d_json": str(geometry_3d_json_path),
            "forward_geometry_3d_html": str(geometry_3d_html_path),
            "isotropic_kerr_strong_field_diagnostic_png": str(strong_diagnostic_png_path),
            "isotropic_kerr_strong_field_diagnostic_json": str(strong_diagnostic_json_path),
            "geodesic_validation_report": str(validation_path),
            "stop_condition_statistics": str(stop_path),
            "diagnostic_report": str(diagnostic_path),
        },
    }
    write_jsonl(paths, paths_path)
    write_jsonl(segments, segments_path)
    write_summary_csv(summary, summary_csv_path)
    write_json(summary_json_path, summary)
    write_json(validation_path, validation_report)
    write_stop_condition_csv(stop_counts, len(paths), stop_path)
    generated_files = [
        paths_path,
        segments_path,
        summary_csv_path,
        summary_json_path,
        preview_path,
        geometry_3d_path,
        geometry_3d_json_path,
        geometry_3d_html_path,
        strong_diagnostic_png_path,
        strong_diagnostic_json_path,
        validation_path,
        stop_path,
        diagnostic_path,
    ]
    draw_forward_preview(paths, segments, preview_path, outer_radius_rg=config.outer_radius_rg)
    draw_forward_geometry_3d(values, paths, segments, geometry_3d_path, geometry_3d_json_path, geometry_3d_html_path)
    strong_diagnostic = generate_strong_field_diagnostic(values, output_dir)
    summary["strong_field_diagnostic"] = strong_diagnostic
    write_json(summary_json_path, summary)
    write_diagnostic_report(summary, generated_files, diagnostic_path)
    return summary
