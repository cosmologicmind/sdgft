"""Level 3: Modified gravity parameters (Horndeski).

From the f(R) = R^n action with n = D*/2, the Bellini-Sawicki
Horndeski parameters follow algebraically:

    alpha_M = (n - 1) / (2n - 1)     Planck mass run-rate
    alpha_B = -alpha_M / 2            Braiding
    alpha_T = 0                       Tensor speed (GW170817)
    alpha_K = 0                       Kineticity

Growth of structure and gravitational lensing:
    gamma    ~ 0.41                   Growth index (LCDM: 0.55; SDGFT: 0.41, MCMC)
    eta_slip = [1+2B(k/aH)^2] /      Gravitational slip Phi/Psi (scale-dependent)
               [1+4B(k/aH)^2]        with B = (n-1); GR limit: eta = 1
"""

from fractions import Fraction

from .constants import DELTA_G_F
from .dimension import (
    N_TREE, N_TREE_F, N_FP,
    TWO_N_MINUS_1_TREE,
    D_STAR_TREE, D_STAR_FP,
)
from .registry import Observable, REGISTRY


# ── Pure functions ────────────────────────────────────────────────

def alpha_m(n: float) -> float:
    """Planck mass run-rate: alpha_M = (n - 1) / (2n - 1).

    For n = 67/48 (tree): alpha_M = 19/86 ~ 0.2209.
    For n ~ 1.398 (fp):   alpha_M ~ 0.2215.
    """
    return (n - 1.0) / (2.0 * n - 1.0)


def alpha_b(n: float) -> float:
    """Braiding coefficient: alpha_B = -alpha_M / 2."""
    return -alpha_m(n) / 2.0


# ── Tree-level constants (exact fractions) ────────────────────────

ALPHA_M_TREE = Fraction(19, 86)
"""alpha_M (tree) = (67/48 - 1) / (67/24 - 1) = (19/48) / (43/24) = 19/86."""

ALPHA_M_TREE_F: float = float(ALPHA_M_TREE)

ALPHA_B_TREE = -ALPHA_M_TREE / 2
"""alpha_B (tree) = -19/172."""

ALPHA_B_TREE_F: float = float(ALPHA_B_TREE)

ALPHA_T: int = 0
"""Tensor speed excess: alpha_T = 0 (GW170817 constraint: |c_T - c| < 10^{-15})."""

ALPHA_K: int = 0
"""Kineticity: alpha_K = 0 (no K-essence term in f(R) gravity)."""

# Verify the fraction
assert ALPHA_M_TREE == (N_TREE - 1) / (2 * N_TREE - 1), (
    f"alpha_M fraction check failed: {(N_TREE - 1) / (2 * N_TREE - 1)}"
)

# ── Fixed-point variants ──────────────────────────────────────────

ALPHA_M_FP: float = alpha_m(N_FP)
"""alpha_M (fixed-point) ~ 0.2215."""

ALPHA_B_FP: float = alpha_b(N_FP)
"""alpha_B (fixed-point) ~ -0.1108."""


# ── Growth index ──────────────────────────────────────────────────

def growth_index_analytic(n: float) -> float:
    """Analytic approximation for the growth index (monograph ch03 eq:growth_index).

    gamma_analytic = (6 - 3n) / (6n - 7)

    WARNING: This formula from the monograph evaluates to 29/22 ~ 1.32
    for n = 67/48, which is NOT the correct reduced growth index ~ 0.41.
    The value gamma ~ 0.41 comes from MCMC + hi_class Boltzmann code
    (monograph ch07, Sec. 7.2).  The analytic formula likely applies
    to a different parameterization.  Use GAMMA_GROWTH for predictions.

    Args:
        n: f(R) exponent (D*/2).

    Returns:
        Analytic approximation (unreliable for n = 67/48).
    """
    return (6.0 - 3.0 * n) / (6.0 * n - 7.0)


# Primary prediction: MCMC numerical result from ch07
GAMMA_GROWTH: float = 0.41
"""Growth index (MCMC/hi_class): gamma ~ 0.41.

The growth rate f(z) = Omega_m(z)^gamma.  GR prediction: gamma ~ 0.55.
SDGFT predicts reduced growth (gamma ~ 0.41) due to the modified
Poisson equation in f(R) = R^{67/48} gravity.  This is a key
falsification target for Euclid/DESI.

Source: monograph ch07 numerical Boltzmann + MCMC analysis.
"""

# For reference: the broken analytic value
_GAMMA_ANALYTIC_TREE = Fraction(29, 22)
"""Analytic formula gives 29/22 ~ 1.32 — does NOT match MCMC.
Kept for documentation; not used in predictions."""


# ── Gravitational slip (scale-dependent) ──────────────────────────

def grav_slip(n: float, k_over_aH: float = 10.0) -> float:
    """Gravitational slip in the quasi-static approximation.

    eta_slip = Phi/Psi = (1 + 2*B*x^2) / (1 + 4*B*x^2)

    where B = (n - 1) and x = k/(aH).  This follows from the
    modified Poisson equation in f(R) = R^n gravity (monograph ch03
    eq:slip).

    Limits:
        - GR (n = 1):      eta = 1  (no slip)
        - Super-horizon (x → 0): eta → 1
        - Deep sub-horizon (x → ∞): eta → 1/2

    Default scale k/(aH) = 10 is representative for Euclid/DESI
    weak lensing surveys.

    Args:
        n:          f(R) exponent (D*/2).
        k_over_aH:  Dimensionless wavenumber k/(aH).

    Returns:
        Gravitational slip Phi/Psi.
    """
    b = n - 1.0
    x2 = k_over_aH ** 2
    return (1.0 + 2.0 * b * x2) / (1.0 + 4.0 * b * x2)


# Representative values at survey scales
ETA_SLIP_SUBHORIZON = Fraction(1, 2)
"""Deep sub-horizon limit of gravitational slip: eta → 1/2."""

ETA_SLIP_SURVEY: float = grav_slip(N_TREE_F, k_over_aH=10.0)
"""Gravitational slip at k/(aH) = 10: ~ 0.503 (survey-relevant scale)."""

ETA_SLIP_HORIZON: float = grav_slip(N_TREE_F, k_over_aH=1.0)
"""Gravitational slip at k/(aH) = 1: ~ 0.694 (horizon scale)."""


# ── Registry ──────────────────────────────────────────────────────

def register_all(registry=REGISTRY) -> None:
    """Register gravity observables."""
    registry.register(Observable(
        name="alpha_m",
        symbol="alpha_M",
        formula="(n - 1) / (2n - 1), n = 67/48",
        predicted=ALPHA_M_TREE_F,
        observed=0.36,
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=3,
        d_star_variant="tree",
        dependencies=("n_tree",),
        is_upper_limit=True,
    ))
    registry.register(Observable(
        name="growth_index",
        symbol="gamma",
        formula="MCMC (hi_class Boltzmann code, ch07)",
        predicted=GAMMA_GROWTH,
        observed=float("nan"),
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=3,
        d_star_variant="both",
        dependencies=("n_tree",),
    ))  # obs=0.55 is LCDM/GR, not a model-independent measurement. Pending: Euclid RSD.
    registry.register(Observable(
        name="eta_slip",
        symbol="eta_slip",
        formula="[1+2(n-1)(k/aH)^2]/[1+4(n-1)(k/aH)^2], k/aH=10",
        predicted=ETA_SLIP_SURVEY,
        observed=float("nan"),
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=3,
        d_star_variant="tree",
        dependencies=("n_tree",),
    ))  # obs=1.0 is GR default, not a measurement. Pending: Euclid WL ~2029.
