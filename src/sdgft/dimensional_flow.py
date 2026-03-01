"""Dimensional flow: the running of D*(r) with scale.

The effective dimension of spacetime runs from D*_UV = 2 at the Planck
scale to D*_IR = 67/24 at cosmological scales, in analogy with the
running of coupling constants in QFT.

Key relations (monograph ch02 Sec. 5):
    beta(D*)    = (Delta / D*) * (D*_IR - D*)            beta-function
    D*(N)       = D*_IR - (D*_IR - D*_start) * exp(-Delta/D*_IR * N)
    D*_start    = 2 + 1/24 = 49/24                      UV + 1 lattice quantum
    N_e         = (D*/Delta) * ln[(D*-2-delta)/(Delta*delta)] = 59.95

Dark energy RG flow (monograph ch03):
    gamma_DE    = delta_g^2 / D*                         anomalous dimension
    dOmega_DE/dlnr = -gamma_DE * Omega_DE                RG equation
    Omega_DE(r) = 3/4 * (r / r_P)^{-delta_g^2/D*}       solution
"""

import math
from fractions import Fraction

from .constants import DELTA, DELTA_F, DELTA_G, DELTA_G_F
from .dimension import D_STAR_TREE, D_STAR_TREE_F, D_STAR_FP
from .registry import Observable, REGISTRY


# ── Physical constants ────────────────────────────────────────────

R_PLANCK: float = 1.616255e-35
"""Planck length [m]."""

R_HUBBLE: float = 8.8e26
"""Hubble radius [m] (approximate)."""

R_HUBBLE_OVER_R_PLANCK: float = R_HUBBLE / R_PLANCK
"""Hierarchy ratio r_H / r_P ~ 5.5e61."""


# ── UV and start values ──────────────────────────────────────────

D_STAR_UV: float = 2.0
"""D* at the Planck scale (UV boundary condition from WDW equation)."""

D_STAR_START = Fraction(49, 24)
"""D*_start = 2 + 1/24 = 49/24 (Planck + 1 lattice quantum)."""

D_STAR_START_F: float = float(D_STAR_START)


# ── Beta-function for dimensional flow ────────────────────────────

def beta_dim(d_star: float, d_star_ir: float, delta: float) -> float:
    """Beta-function controlling the dimensional flow.

    dD*/dN = (Delta / D*) * (D*_IR - D*)

    (Monograph ch02 eq:beta_dim.)

    Newton cooling law: rate proportional to distance from fixed point.
    Two zeros: D*=0 (trivial, unstable) and D*=D*_IR (non-trivial, stable).

    Args:
        d_star:    Current D*.
        d_star_ir: IR fixed-point D*.
        delta:     Fundamental lattice scale Delta.

    Returns:
        dD*/dN (rate of change per e-fold).
    """
    return (delta / d_star) * (d_star_ir - d_star)


def d_star_efold(n_efold: float, d_star_ir: float, d_star_start: float,
                 delta: float) -> float:
    """D* as function of e-fold number during inflation.

    D*(N) = D*_IR - (D*_IR - D*_start) * exp(-Delta/D*_IR * N)

    (Monograph ch02 eq:dim_flow_solution.)

    At N=0:  D* = D*_start
    At N→∞:  D* → D*_IR  (IR fixed point)

    Args:
        n_efold:     Number of e-folds since start of flow.
        d_star_ir:   IR fixed-point D*.
        d_star_start: Starting value of D*.
        delta:       Fundamental lattice scale Delta.

    Returns:
        D* at N e-folds.
    """
    return d_star_ir - (d_star_ir - d_star_start) * math.exp(
        -delta / d_star_ir * n_efold
    )


# ── D* at inflation milestones ───────────────────────────────────

D_STAR_N0: float = d_star_efold(0.0, D_STAR_TREE_F, D_STAR_START_F, DELTA_F)
"""D* at N=0 (start of inflation): = D*_start ~ 2.042."""

D_STAR_N30: float = d_star_efold(30.0, D_STAR_TREE_F, D_STAR_START_F, DELTA_F)
"""D* at N=30 (mid-inflation): ~ 2.69."""

D_STAR_N60: float = d_star_efold(60.0, D_STAR_TREE_F, D_STAR_START_F, DELTA_F)
"""D* at N=60 (end of inflation): ~ 2.79 (converged to IR)."""


# ── Scale-dependent effective dimension D*(r) ─────────────────────

def d_star_of_r(
    r: float,
    d_star_ir: float = D_STAR_TREE_F,
    delta: float = DELTA_F,
    r_p: float = R_PLANCK,
) -> float:
    """Effective dimension as a function of physical scale.

    D*(r) = D*_IR · (r / r_P)^{-Delta^2}

    (Besemer 2026, Eq. 7.)

    The exponent Delta^2 = (5/24)^2 = 25/576 ~ 0.0434 governs the
    slow power-law running of the spectral dimension from UV to IR.
    At r >> r_P the dimension asymptotes to D*_IR.

    Note: this power-law parametrisation is valid for r >> r_P.
    At the Planck scale the full beta-function (beta_dim) must be used.

    Args:
        r:         Physical scale [m].
        d_star_ir: IR fixed-point D*.
        delta:     Fundamental lattice scale Delta = 5/24.
        r_p:       Planck length [m].

    Returns:
        Effective dimension at scale r.
    """
    exponent = -(delta ** 2)
    return d_star_ir * (r / r_p) ** exponent


D_STAR_GALACTIC: float = d_star_of_r(1.8e3 * 3.0857e19)  # 1.8 kpc in metres
"""D* at galactic scale (~1.8 kpc) from power-law running."""


# ── Dark energy RG flow ──────────────────────────────────────────

GAMMA_DE_TREE = DELTA_G ** 2 / D_STAR_TREE
"""Anomalous dimension: gamma_DE = delta_g^2/D* = (1/24)^2 / (67/24) = 1/1608."""

GAMMA_DE_TREE_F: float = float(GAMMA_DE_TREE)
"""Float alias: ~ 0.000622."""


def omega_de_rg(r: float, d_star: float, delta_g: float) -> float:
    """Dark energy density from RG flow.

    Omega_DE(r) = 3/4 * (r / r_P)^{-delta_g^2 / D*}

    UV boundary: Omega_DE(r_P) = cos^2(30°) = 3/4.
    IR integration to r_H gives Omega_DE ~ 0.687.

    (Monograph ch03 eq:ODE_integral.)

    Args:
        r:       Physical scale [m].
        d_star:  Effective dimension D*.
        delta_g: Lattice tension delta_g.

    Returns:
        Omega_DE at scale r.
    """
    exponent = -(delta_g ** 2) / d_star
    return 0.75 * (r / R_PLANCK) ** exponent


# Omega_DE at the Hubble scale via RG flow
OMEGA_DE_RG: float = omega_de_rg(R_HUBBLE, D_STAR_TREE_F, DELTA_G_F)
"""Omega_DE(r_H) from RG integration ~ 0.687.

The exact-fraction value from flatness closure is 1589/2304 ~ 0.690.
The 0.4% difference indicates the magnitude of higher-order corrections.
(Monograph ch03 eq:ODE_result.)
"""


# ── Registry ──────────────────────────────────────────────────────

def register_all(registry=REGISTRY) -> None:
    """Register dimensional flow observables."""
    registry.register(Observable(
        name="gamma_de",
        symbol="gamma_DE",
        formula="delta_g^2 / D* = 1/1608",
        predicted=GAMMA_DE_TREE_F,
        observed=0.001,
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=6,
        d_star_variant="tree",
        dependencies=("delta_g", "d_star_tree"),
        is_upper_limit=True,
    ))
    registry.register(Observable(
        name="omega_de_rg",
        symbol="Omega_DE^RG",
        formula="3/4 * (r_H/r_P)^{-delta_g^2/D*}",
        predicted=OMEGA_DE_RG,
        observed=0.6847,
        observed_uncertainty=0.0073,
        unit="dimensionless",
        level=6,
        d_star_variant="tree",
        dependencies=("delta_g", "d_star_tree"),
    ))
    registry.register(Observable(
        name="d_star_galactic",
        symbol="D*_gal",
        formula="D*_IR * (r_gal / r_P)^{-Delta^2}",
        predicted=D_STAR_GALACTIC,
        observed=float("nan"),
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=6,
        d_star_variant="tree",
        dependencies=("d_star_tree", "delta"),
        is_diagnostic=True,
    ))  # Eq.7: power-law running; CDT / spectral-dimension measurement pending.
