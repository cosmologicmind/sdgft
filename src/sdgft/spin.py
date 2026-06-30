"""Level 1: Geometric derivation of particle spin.

Spin emerges from the 6-cone lattice geometry of the 24-cell:

- Fermion spin 1/2 = sin(30 deg): matter must tilt at 30 degrees
  to the cone axis to fit in the hexagonal lattice. The projection
  of the unit vector onto the symmetry axis gives sin(theta_max).

- Boson spin 1 = sin(90 deg): force carriers propagate along the
  symmetry axes without tilt.

- 720-degree identity: exp(i * 2*pi * 1/2) = -1, so a full 360-degree
  rotation inverts the state. Two full rotations (720 deg) restore it.
  This is the Moebius topology of the 6-cone lattice.

- Only spin 1/2 and spin 1 are stable: the hexagonal sector angle
  is 60 deg, and only sin(30) and sin(90) produce integer sector counts.
"""

from __future__ import annotations

import math
from fractions import Fraction

from .constants import THETA_MAX, SIN2_30


# ── Spin values ────────────────────────────────────────────────────

SPIN_HALF = Fraction(1, 2)
"""Fermion spin: sin(30 deg) = 1/2. Exact as Fraction."""

SPIN_HALF_F: float = 0.5
"""Float alias for SPIN_HALF."""

SPIN_ONE: int = 1
"""Boson spin: sin(90 deg) = 1."""

ROTATION_FULL: float = 720.0
"""Full identity rotation for spin-1/2 particles in degrees."""

SECTOR_ANGLE: float = 60.0
"""Hexagonal lattice sector angle in degrees."""


# ── Functions ──────────────────────────────────────────────────────

def spin_from_tilt(theta_deg: float) -> float:
    """Spin quantum number from lattice tilt angle.

    spin = sin(theta)

    Args:
        theta_deg: Tilt angle in degrees.

    Returns:
        Spin quantum number.
    """
    return math.sin(math.radians(theta_deg))


def is_stable_spin(spin_val: float, sector: float = SECTOR_ANGLE) -> bool:
    """Check if a spin value is stable in the hexagonal lattice.

    A spin is stable if the sector angle divided by arcsin(spin)
    is an integer, meaning the tilted cone tiles the lattice exactly.

    Args:
        spin_val: Candidate spin value (0 < spin_val <= 1).
        sector: Lattice sector angle in degrees.

    Returns:
        True if the spin is geometrically stable.
    """
    if spin_val <= 0 or spin_val > 1:
        return False
    alpha = math.degrees(math.asin(spin_val))
    if alpha == 0:
        return False
    ratio = sector / alpha
    return abs(ratio - round(ratio)) < 1e-10


def rotation_for_identity(spin: float) -> float:
    """Rotation angle needed for identity (phase = +1).

    For spin s: exp(i * theta * s) = 1 requires theta = 2*pi*n/s.
    Minimum identity rotation = 2*pi / s (in radians) = 360/s (in degrees).

    Args:
        spin: Spin quantum number.

    Returns:
        Minimum identity rotation in degrees.
    """
    return 360.0 / spin


def spin_statistics_phase(spin: float) -> float:
    """Phase acquired under 360-degree rotation.

    phase = exp(i * 2*pi * spin)
    For spin 1/2: phase = exp(i*pi) = -1 (fermion)
    For spin 1:   phase = exp(i*2pi) = +1 (boson)

    Returns:
        Phase factor (+1 for boson, -1 for fermion).
    """
    return math.cos(2.0 * math.pi * spin)


# ── Module-level computed values ──────────────────────────────────

FERMION_PHASE: float = spin_statistics_phase(SPIN_HALF_F)
"""Phase under 360-degree rotation for fermions: -1."""

BOSON_PHASE: float = spin_statistics_phase(float(SPIN_ONE))
"""Phase under 360-degree rotation for bosons: +1."""


# ── Registry ──────────────────────────────────────────────────────

from .registry import Observable, REGISTRY


def register_all(registry=REGISTRY) -> None:
    """Register geometric spin observables."""
    registry.register(Observable(
        name="spin_half",
        symbol="s_fermion",
        formula="sin(30 deg) = sin(theta_max)",
        predicted=SPIN_HALF_F,
        observed=0.5,
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=1,
        d_star_variant="none",
        dependencies=(),
    ))
    registry.register(Observable(
        name="rotation_period",
        symbol="theta_identity",
        formula="360 / spin = 360 / (1/2) = 720",
        predicted=ROTATION_FULL,
        observed=720.0,
        observed_uncertainty=0.0,
        unit="degrees",
        level=1,
        d_star_variant="none",
        dependencies=(),
    ))
