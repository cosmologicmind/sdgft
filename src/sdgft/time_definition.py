"""Level 2: Geometric definition of the second (time scaling).

Time in SDGFT is defined through interactions on the boundary
(surface) of the effective D*-dimensional spacetime:

    surface_dim = D* - 1    (interaction surface dimension)
    time_exponent = 1 / (D* - 1)  (time scaling exponent)

Physical interpretation:
    - The effective spacetime has dimension D* ~ 2.79 (not exactly 3).
    - Interactions propagate on the (D*-1)-dimensional surface.
    - The time exponent 1/(D*-1) governs how time intervals scale
      with spatial extent: dt ~ dr^{1/(D*-1)}.
    - At D* = 2 (UV fixed point): exponent = 1   (direct, linear coupling)
    - At D* ~ 2.79 (today):      exponent ~ 0.557 (fractional scaling)
    - At D* = 3 (exact 3D):      exponent = 0.5  (diffusive scaling)

The surface dimension D* - 1 coincides with the f(R) index 2n - 1
(since n = D*/2), providing an independent cross-check.
"""

from __future__ import annotations

import math
from fractions import Fraction

from .constants import DELTA, DELTA_G, DELTA_F, DELTA_G_F
from .dimension import (
    D_STAR_TREE, D_STAR_TREE_F, D_STAR_FP,
    TWO_N_MINUS_1_TREE, TWO_N_MINUS_1_TREE_F, TWO_N_MINUS_1_FP,
)
from .registry import Observable, REGISTRY


# ── Surface dimension (D* - 1) ───────────────────────────────────

SURFACE_DIM_TREE = Fraction(43, 24)
"""Surface dimension (tree): D*_tree - 1 = 67/24 - 1 = 43/24."""

SURFACE_DIM_TREE_F: float = float(SURFACE_DIM_TREE)
"""Float alias: 1.79166..."""

SURFACE_DIM_FP: float = D_STAR_FP - 1.0
"""Surface dimension (fixed-point): D*_fp - 1 ~ 1.79676."""

# Cross-check: surface_dim = 2n - 1
assert SURFACE_DIM_TREE == TWO_N_MINUS_1_TREE, (
    f"Surface dim (tree) must equal 2n-1: {SURFACE_DIM_TREE} != {TWO_N_MINUS_1_TREE}"
)
assert abs(SURFACE_DIM_FP - TWO_N_MINUS_1_FP) < 1e-14, (
    f"Surface dim (fp) must equal 2n-1: {SURFACE_DIM_FP} != {TWO_N_MINUS_1_FP}"
)

# Cross-check: tree value matches D*_tree - 1
assert SURFACE_DIM_TREE == D_STAR_TREE - 1, (
    f"Surface dim (tree) must be D*-1: {SURFACE_DIM_TREE} != {D_STAR_TREE - 1}"
)


# ── Time exponent 1/(D* - 1) ────────────────────────────────────

TIME_EXPONENT_TREE = Fraction(24, 43)
"""Time exponent (tree): 1/(D*_tree - 1) = 24/43."""

TIME_EXPONENT_TREE_F: float = float(TIME_EXPONENT_TREE)
"""Float alias: 0.55814..."""

TIME_EXPONENT_FP: float = 1.0 / SURFACE_DIM_FP
"""Time exponent (fixed-point): 1/(D*_fp - 1) ~ 0.55656."""


# ── Limiting cases ───────────────────────────────────────────────

TIME_EXPONENT_UV: float = 1.0
"""UV limit (D* = 2): exponent = 1/(2-1) = 1."""

TIME_EXPONENT_3D: float = 0.5
"""Exact 3D limit (D* = 3): exponent = 1/(3-1) = 0.5."""


# ── Functions ────────────────────────────────────────────────────

def surface_dimension(d_star: float) -> float:
    """Surface dimension for a given effective dimension.

    Args:
        d_star: Effective spacetime dimension.

    Returns:
        Surface dimension D* - 1.
    """
    return d_star - 1.0


def time_exponent(d_star: float) -> float:
    """Time scaling exponent for a given effective dimension.

    dt ~ dr^{1/(D*-1)}

    Args:
        d_star: Effective spacetime dimension (must be > 1).

    Returns:
        Time scaling exponent 1/(D* - 1).

    Raises:
        ValueError: If d_star <= 1.
    """
    if d_star <= 1.0:
        raise ValueError(f"D* must be > 1, got {d_star}")
    return 1.0 / (d_star - 1.0)


def time_ratio(d_star_1: float, d_star_2: float) -> float:
    """Ratio of time exponents between two scales.

    Useful for comparing UV vs IR time scaling:
    ratio = exponent(D*_1) / exponent(D*_2) = (D*_2 - 1) / (D*_1 - 1)

    Args:
        d_star_1: First effective dimension.
        d_star_2: Second effective dimension.

    Returns:
        Ratio of time exponents.
    """
    return (d_star_2 - 1.0) / (d_star_1 - 1.0)


# ── Registry ─────────────────────────────────────────────────────

def register_all(registry=REGISTRY) -> None:
    """Register time-definition observables."""
    registry.register(Observable(
        name="surface_dimension",
        symbol="D*-1",
        formula="D*_tree - 1 = 67/24 - 1 = 43/24",
        predicted=SURFACE_DIM_TREE_F,
        observed=SURFACE_DIM_TREE_F,
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=2,
        d_star_variant="tree",
        dependencies=("d_star_tree",),
    ))
    registry.register(Observable(
        name="time_exponent",
        symbol="1/(D*-1)",
        formula="1/(D*_tree - 1) = 24/43",
        predicted=TIME_EXPONENT_TREE_F,
        observed=TIME_EXPONENT_TREE_F,
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=2,
        d_star_variant="tree",
        dependencies=("d_star_tree",),
    ))
