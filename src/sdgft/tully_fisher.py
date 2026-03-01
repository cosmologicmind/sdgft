"""Level 6: Tully-Fisher relation and galactic rotation curves.

The Tully-Fisher slope and galactic transition scale emerge from the
running gravitational coupling G(k) evaluated at galactic scales.

Key results:
    - b_TF = D* + 1 = 67/24 + 1 = 91/24 ~ 3.79
      Observed: 3.98 +/- 0.10
    - epsilon_gal = 0.16 +/- 0.05 (galactic IR modification)
    - r_kpc ~ 1.82 kpc (transition scale from Newtonian to modified)

Source: ch05_cosmology.tex
"""

from __future__ import annotations

import math
from fractions import Fraction

from .constants import DELTA, DELTA_F, DELTA_G, DELTA_G_F
from .dimension import D_STAR_TREE, D_STAR_TREE_F, D_STAR_FP
from .physical_constants import G_N, R_P, R_H, KPC_M
from .registry import Observable, REGISTRY


# ── Tully-Fisher slope ───────────────────────────────────────────

B_TF_TREE = Fraction(91, 24)
"""Tully-Fisher slope (tree): D*_tree + 1 = 67/24 + 1 = 91/24."""

B_TF_TREE_F: float = float(B_TF_TREE)
"""Float alias: 3.79166..."""

B_TF_FP: float = D_STAR_FP + 1.0
"""Tully-Fisher slope (fixed-point): D*_fp + 1 ~ 3.797."""

# Verify
assert B_TF_TREE == D_STAR_TREE + 1, "B_TF must equal D* + 1"


# ── epsilon_gal (galactic IR modification) ───────────────────────

EPSILON_GAL: float = 0.16
"""Galactic IR regime modification strength.

v^2(r) = G_N * M(r)/r * [1 + epsilon_gal * ln(r/r_ref)]

Observationally: 0.16 +/- 0.05 from rotation curve fits.
This is the same parameter as in quantum_gravity.py, included
here for the self-contained Tully-Fisher context.
"""


# ── Galactic transition scale ────────────────────────────────────

def transition_radius_kpc(
    d_star: float = D_STAR_TREE_F,
    delta_g: float = DELTA_G_F,
    r_p: float = R_P,
    r_h: float = R_H,
) -> float:
    """Galactic transition scale in kpc.

    r_trans = r_H * (r_P / r_H)^(D* * delta_g)

    This is the scale where the running G(k) transitions from
    Newtonian to the modified (logarithmic) regime. The exponent
    uses delta_g = 1/24 (elementary lattice tension).

    Args:
        d_star: Effective dimension.
        delta_g: Elementary lattice tension (1/24).
        r_p: Planck length in m.
        r_h: Hubble radius in m.

    Returns:
        Transition radius in kpc.
    """
    exponent = d_star * delta_g
    r_m = r_h * (r_p / r_h) ** exponent
    return r_m / KPC_M


R_TRANS_KPC: float = transition_radius_kpc()
"""Galactic transition scale ~ 1 kpc."""


# ── Rotation curve model ─────────────────────────────────────────

def v_circular_squared(
    r: float,
    m_enclosed: float,
    g_n: float = G_N,
    epsilon: float = EPSILON_GAL,
    r_ref: float = 1.0,
) -> float:
    """Circular velocity squared including running-G correction.

    v^2(r) = G_N * M(r) / r * [1 + epsilon * ln(r / r_ref)]

    Args:
        r: Radial distance (same units as r_ref).
        m_enclosed: Enclosed mass in kg.
        g_n: Newton's constant.
        epsilon: IR modification strength.
        r_ref: Reference scale.

    Returns:
        v^2 in (m/s)^2.
    """
    return g_n * m_enclosed / r * (1.0 + epsilon * math.log(r / r_ref))


# ── Registry ─────────────────────────────────────────────────────

def register_all(registry=REGISTRY) -> None:
    """Register Tully-Fisher observables."""
    registry.register(Observable(
        name="b_tully_fisher",
        symbol="b_TF",
        formula="D*_tree + 1 = 91/24",
        predicted=B_TF_TREE_F,
        observed=3.85,
        observed_uncertainty=0.09,
        unit="dimensionless",
        level=6,
        d_star_variant="tree",
        dependencies=("d_star_tree",),
    ))  # Lelli, McGaugh, Schombert (2019) SPARC data. Old: McGaugh 2012 = 3.98.
    registry.register(Observable(
        name="epsilon_gal",
        symbol="epsilon_gal",
        formula="IR modification from running G",
        predicted=EPSILON_GAL,
        observed=0.16,
        observed_uncertainty=0.05,
        unit="dimensionless",
        level=6,
        d_star_variant="none",
        dependencies=(),
    ))
