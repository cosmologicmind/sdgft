"""Neutrino mixing angles from 6-cone geometry.

Replaces the ad-hoc formulas with geometrically motivated derivations
based on tribimaximal mixing (TBM) and the 24-cell topology.

Key results:
    theta_12: TBM * (1 - delta_g) = arctan(1/sqrt2) * 23/24 = 33.80 deg
              (old ad-hoc: 29.0 deg, observed: 33.41 +/- 0.75 deg)
              Improvement: 5.87 sigma -> 0.51 sigma

    theta_23: 45*(1 + Delta/sqrt6) = 48.83 deg
              (observed: 49.0 +/- 1.4 deg, 0.12 sigma)

    theta_13: arcsin(Delta/sqrt2) = 8.47 deg
              (observed: 8.54 +/- 0.15 deg, 0.46 sigma)

Physical reasoning:

1. TBM base: The 6-cone system (3 axes, 6 cones) with S_3 permutation
   symmetry naturally produces tribimaximal mixing:
   theta_12_TBM = arctan(1/sqrt2) = 35.26 deg
   theta_23_TBM = 45 deg (maximal)
   theta_13_TBM = 0 deg

2. Delta-corrections from the topological defect:
   - theta_12: Reduced by delta_g = 1/24 (single-vertex sampling).
     The observer occupies 1 of 24 equivalent 24-cell vertices,
     reducing the effective mixing to (N-1)/N = 23/24 of TBM.
   - theta_23: Enhanced by Delta/sqrt(6) (Fibonacci-defect projected
     onto the 3D flavor subspace with sqrt(6) normalization from
     6 cones).
   - theta_13: Generated entirely by Delta (zero in TBM).
     The projection Delta/sqrt(2) onto the 1-3 subspace gives
     the reactor angle.
"""

from __future__ import annotations

import math

from ..constants import DELTA_F, DELTA_G_F, PHI
from ..registry import Observable, REGISTRY


# ── TBM base values ───────────────────────────────────────────────

THETA_12_TBM: float = math.degrees(math.atan(1.0 / math.sqrt(2.0)))
"""TBM solar angle: arctan(1/sqrt2) = 35.26 deg."""

THETA_23_TBM: float = 45.0
"""TBM atmospheric angle: 45 deg (maximal mixing)."""

THETA_13_TBM: float = 0.0
"""TBM reactor angle: 0 deg (no mixing)."""


# ── 4D solid angle calculations ───────────────────────────────────

THETA_MAX_RAD: float = math.pi / 6.0
"""Cone half-opening angle: 30 deg = pi/6."""

N_VERTICES: int = 24
"""Number of vertices of the 24-cell."""


def solid_angle_4d_cone(alpha: float) -> float:
    """Solid angle of a cone with half-angle alpha on S^3.

    Omega_4(alpha) = 2*pi*(alpha - sin(alpha)*cos(alpha))

    This is the integral of sin^2(theta) from 0 to alpha
    on the 3-sphere.

    Args:
        alpha: Half-opening angle in radians.

    Returns:
        Solid angle (area on unit S^3).
    """
    return 2.0 * math.pi * (alpha - math.sin(alpha) * math.cos(alpha))


OMEGA_CONE: float = solid_angle_4d_cone(THETA_MAX_RAD)
"""Solid angle of a single 30-degree cone on S^3."""

OMEGA_TOTAL: float = 2.0 * math.pi ** 2
"""Total surface area of S^3."""

CONE_FRACTION: float = OMEGA_CONE / OMEGA_TOTAL
"""Fraction of S^3 covered by one cone."""


# ── Geometric correction functions ────────────────────────────────

def theta_12_geo(delta_g: float = DELTA_G_F) -> float:
    """Solar mixing angle: TBM corrected by single-vertex sampling.

    theta_12 = arctan(1/sqrt2) * (1 - delta_g)
             = arctan(1/sqrt2) * (N-1)/N   where N=24

    The observer occupies 1 of 24 equivalent vertices of the 24-cell,
    reducing the effective solar mixing from TBM by the fraction 1/N.

    Args:
        delta_g: Elementary lattice tension (1/24).

    Returns:
        Predicted theta_12 in degrees.
    """
    return THETA_12_TBM * (1.0 - delta_g)


def theta_23_geo(delta: float = DELTA_F) -> float:
    """Atmospheric mixing angle: TBM with Fibonacci-defect correction.

    theta_23 = 45 * (1 + Delta/sqrt(6))

    The Fibonacci defect Delta = 5/24 breaks the mu-tau symmetry.
    The factor sqrt(6) is the normalization from projecting the
    5-dimensional defect onto the 6-cone system sqrt(N_cones).

    Args:
        delta: Fibonacci-lattice conflict (5/24).

    Returns:
        Predicted theta_23 in degrees.
    """
    return 45.0 * (1.0 + delta / math.sqrt(6.0))


def theta_13_geo(delta: float = DELTA_F) -> float:
    """Reactor mixing angle: generated entirely by the defect.

    theta_13 = arcsin(Delta/sqrt(2))

    Zero in TBM (no 1-3 mixing). The defect Delta projected onto
    the 2D subspace spanned by mass eigenstates 1 and 3 yields
    an amplitude Delta/sqrt(2).

    Args:
        delta: Fibonacci-lattice conflict (5/24).

    Returns:
        Predicted theta_13 in degrees.
    """
    return math.degrees(math.asin(delta / math.sqrt(2.0)))


def compute_overlap_matrix(
    alpha: float = THETA_MAX_RAD,
    delta: float = DELTA_F,
) -> list[list[float]]:
    """Compute the 3x3 overlap matrix for the 6-cone system.

    Returns:
        3x3 matrix where entry [i][j] is the fractional overlap
        between cone pair i and cone pair j.
    """
    # Adjacent cones (at 90 deg): overlap scales as (delta/alpha)^2
    f_adj = (delta / alpha) ** 2 / 2.0
    return [
        [1.0, f_adj, f_adj],
        [f_adj, 1.0, f_adj],
        [f_adj, f_adj, 1.0],
    ]


# ── Module-level computed values ──────────────────────────────────

THETA_12_GEO: float = theta_12_geo()
"""Solar mixing angle from cone geometry: 33.80 deg."""

THETA_23_GEO: float = theta_23_geo()
"""Atmospheric mixing angle from cone geometry: 48.83 deg."""

THETA_13_GEO: float = theta_13_geo()
"""Reactor mixing angle from cone geometry: 8.47 deg."""

OVERLAP_MATRIX: list[list[float]] = compute_overlap_matrix()
"""3x3 cone overlap matrix."""


# ── Registry ──────────────────────────────────────────────────────

def register_all(registry=REGISTRY) -> None:
    """Register geometric mixing angle observables."""
    registry.register(Observable(
        name="exp_theta_12",
        symbol="theta_12_geo",
        formula="arctan(1/sqrt2) * (1 - 1/24)",
        predicted=THETA_12_GEO,
        observed=33.41,
        observed_uncertainty=0.75,
        unit="degrees",
        level=6,
        d_star_variant="none",
        dependencies=("delta_g",),
    ))
    registry.register(Observable(
        name="exp_theta_23",
        symbol="theta_23_geo",
        formula="45*(1 + Delta/sqrt6)",
        predicted=THETA_23_GEO,
        observed=49.0,
        observed_uncertainty=1.4,
        unit="degrees",
        level=6,
        d_star_variant="none",
        dependencies=("delta",),
    ))
    registry.register(Observable(
        name="exp_theta_13",
        symbol="theta_13_geo",
        formula="arcsin(Delta/sqrt2)",
        predicted=THETA_13_GEO,
        observed=8.54,
        observed_uncertainty=0.15,
        unit="degrees",
        level=6,
        d_star_variant="none",
        dependencies=("delta",),
    ))
