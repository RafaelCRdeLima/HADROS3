"""Validation helpers for H3-W6 forward neutrino geodesics."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


ALLOWED_STOP_CONDITIONS = {"horizon_crossing", "outer_escape_radius", "max_steps", "invalid_invariant"}


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_forward_products(
    paths: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    *,
    expected_paths: int,
    null_tolerance: float,
    killing_energy_tolerance: float,
    lz_tolerance: float,
) -> tuple[list[str], dict[str, Any], dict[str, int]]:
    problems: list[str] = []
    if len(paths) != expected_paths:
        problems.append("forward geodesic path count does not match requested sample count")
    if paths and not segments:
        problems.append("forward geodesic propagation produced no path segments")

    counts = Counter(str(path.get("stop_condition", "")) for path in paths)
    for condition in counts:
        if condition not in ALLOWED_STOP_CONDITIONS:
            problems.append(f"unsupported stop_condition: {condition}")

    null_errors = [abs(float(path.get("null_norm_max_abs", math.inf))) for path in paths]
    energy_errors = [abs(float(path.get("killing_energy_max_error", math.inf))) for path in paths]
    lz_errors = [abs(float(path.get("lz_max_error", math.inf))) for path in paths]
    invalid_count = 0
    for path in paths:
        if not path.get("validation_pass", False):
            invalid_count += 1
    for index, segment in enumerate(segments):
        required = [
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
        ]
        for key in required:
            if not finite_number(segment.get(key)):
                problems.append(f"segment {index} has non-finite {key}")
        if finite_number(segment.get("dl_segment_rg")) and float(segment["dl_segment_rg"]) <= 0.0:
            problems.append(f"segment {index} has non-positive dl_segment_rg")
        if finite_number(segment.get("E_nu_local_gev_mid")) and float(segment["E_nu_local_gev_mid"]) <= 0.0:
            problems.append(f"segment {index} has non-positive E_nu_local_gev_mid")

    null_max = max(null_errors) if null_errors else math.inf
    energy_max = max(energy_errors) if energy_errors else math.inf
    lz_max = max(lz_errors) if lz_errors else math.inf
    if null_max > null_tolerance:
        problems.append("null norm tolerance exceeded")
    if energy_max > killing_energy_tolerance:
        problems.append("Killing energy tolerance exceeded")
    if lz_max > lz_tolerance:
        problems.append("Lz tolerance exceeded")

    report = {
        "null_norm_max": null_max,
        "null_norm_mean": sum(null_errors) / len(null_errors) if null_errors else math.inf,
        "killing_energy_max_error": energy_max,
        "killing_energy_mean_error": sum(energy_errors) / len(energy_errors) if energy_errors else math.inf,
        "lz_max_error": lz_max,
        "lz_mean_error": sum(lz_errors) / len(lz_errors) if lz_errors else math.inf,
        "n_invalid": invalid_count,
        "validation_pass": not problems,
    }
    return problems, report, dict(counts)
