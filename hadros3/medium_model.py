"""Shared analytic medium model for HADROS3 diagnostics."""

from __future__ import annotations

import math
from typing import Any


DENSITY_MODEL = "analytic_torus_density_v1"
MEDIUM_MODEL = "analytic_torus"
THETA_PROFILE = "gaussian"


def medium_metadata() -> dict[str, Any]:
    return {
        "medium_model": MEDIUM_MODEL,
        "density_model": DENSITY_MODEL,
        "density_model_has_hard_radial_cut": True,
        "density_model_theta_profile": THETA_PROFILE,
        "density_model_theta_is_hard_cut": False,
        "half_opening_angle_interpretation": "gaussian_width_not_boundary",
    }


def analytic_torus_density_g_cm3(
    r_rg: float,
    theta_rad: float,
    values: dict[str, dict[str, Any]],
    *,
    density_floor_g_cm3: float = 0.0,
) -> float:
    torus = values["analytic_torus"]
    r_inner = float(torus["r_inner_rg"])
    r_outer = float(torus["r_outer_rg"])
    r_peak = float(torus["r_peak_rg"])
    half_angle = math.radians(float(torus["half_opening_angle_deg"]))
    density_norm = float(torus["density_norm_g_cm3"])
    if r_rg < r_inner or r_rg > r_outer:
        return 0.0
    theta_width = max(half_angle, 1.0e-6)
    radial_width = max(0.5 * (r_outer - r_inner), 1.0e-6)
    radial_profile = math.exp(-0.5 * ((r_rg - r_peak) / radial_width) ** 2)
    theta_profile = math.exp(-0.5 * ((theta_rad - 0.5 * math.pi) / theta_width) ** 2)
    rho = density_norm * radial_profile * theta_profile
    if rho <= 0.0:
        return 0.0
    return max(rho, density_floor_g_cm3)
