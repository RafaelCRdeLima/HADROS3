"""Shared fixed-target kinematics for HADROS3 POWHEG cards."""

from __future__ import annotations

import math


NUCLEON_MASS_GEV = 0.938272
POWHEG_QMAX_CAP_GEV = 1.0e5


def qmax_for_energy_gev(energy_gev: float) -> float:
    """Return the fixed-target high-energy limit used by the nudis card."""

    return min(
        math.sqrt(2.0 * NUCLEON_MASS_GEV * max(float(energy_gev), 0.0)),
        POWHEG_QMAX_CAP_GEV,
    )


def fortran_double(value: float, precision: int = 10) -> str:
    return f"{value:.{precision}E}".replace("E", "D")
