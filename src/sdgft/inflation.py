"""Level 4: Inflationary observables.

Inflation in SDGFT is not driven by an ad-hoc inflaton field but by
dimensional flow from D_UV = 2 (UV fixed point) to D*_IR (IR attractor).

Key observables:
    N_e      = e-folds of inflation
    n_s      = primordial spectral index
    r        = tensor-to-scalar ratio
    beta_iso = isocurvature amplitude

Slow-roll parameters (f(R) = R^n, Jordan frame, Hwang 2001):
    epsilon  = (2n-1)(n-2)^2 / [N_e(2n-1) + n]^2
    eta_sr   = (2n-1)(2n^2-7n+4) / [N_e(2n-1)+n]^2  - (2n-1)/[N_e(2n-1)+n]
"""

from fractions import Fraction
import math

from .constants import DELTA, DELTA_F, DELTA_G, DELTA_G_F
from .dimension import (
    D_STAR_TREE_F, D_STAR_FP,
    N_TREE_F, TWO_N_MINUS_1_TREE_F,
)
from .registry import Observable, REGISTRY


# ── Pure functions ────────────────────────────────────────────────

def e_folds(d_star: float, delta: float, delta_g: float) -> float:
    """Number of inflationary e-folds.

    N_e = (D* / Delta) * ln[(D* - 2 - delta) / (Delta * delta)]

    The UV starting dimension is D_UV = 2. The argument of the
    logarithm measures the gap between the IR attractor D* and
    the UV start, normalized by the product of the two fundamental
    lattice scales.
    """
    numerator = d_star - 2.0 - delta_g
    denominator = delta * delta_g
    return (d_star / delta) * math.log(numerator / denominator)


def spectral_index(n: float, n_e: float) -> float:
    """Primordial spectral index for f(R) = R^n inflation.

    n_s = 1 - 2(2n - 1) / [N_e * (2n - 1) + n]

    Derived from the Mukhanov-Sasaki equation in R^n gravity.
    """
    two_n_m1 = 2.0 * n - 1.0
    denom = n_e * two_n_m1 + n
    return 1.0 - 2.0 * two_n_m1 / denom


def tensor_to_scalar(n: float, n_e: float) -> float:
    """Tensor-to-scalar ratio for f(R) = R^n inflation.

    r = 48 * (2n - 1)^2 / [N_e * (2n - 1) + n]^2
    """
    two_n_m1 = 2.0 * n - 1.0
    denom = n_e * two_n_m1 + n
    return 48.0 * two_n_m1 ** 2 / denom ** 2


def slow_roll_epsilon(n: float, n_e: float) -> float:
    """First slow-roll parameter for f(R) = R^n inflation.

    epsilon = (2n-1)(n-2)^2 / [N_e(2n-1) + n]^2

    This is the Hubble slow-roll parameter in the Jordan frame
    (Hwang 2001, monograph ch03 eq:epsilon_fR).

    For n = 67/48, N_e ~ 60: epsilon ~ 5.5e-5.

    Note: the f(R) consistency relation is NOT r = 16*epsilon.
    Instead r = [48(2n-1)/(n-2)^2] * epsilon.
    """
    two_n_m1 = 2.0 * n - 1.0
    denom = n_e * two_n_m1 + n
    return two_n_m1 * (n - 2.0) ** 2 / denom ** 2


def slow_roll_eta(n: float, n_e: float) -> float:
    """Second slow-roll parameter for f(R) = R^n inflation.

    eta = (2n-1)(2n^2 - 7n + 4) / [N_e(2n-1)+n]^2
          - (2n-1) / [N_e(2n-1)+n]

    (Hwang 2001, monograph ch03 eq:eta_fR).
    For n = 67/48, N_e ~ 60: eta ~ -0.0167.
    """
    two_n_m1 = 2.0 * n - 1.0
    denom = n_e * two_n_m1 + n
    term1 = two_n_m1 * (2.0 * n ** 2 - 7.0 * n + 4.0) / denom ** 2
    term2 = two_n_m1 / denom
    return term1 - term2


# ── Constants ─────────────────────────────────────────────────────

BETA_ISO = Fraction(1, 36)
"""Isocurvature amplitude: (1/6)^2 = 1/36.

Six-cone geometry has 6 independent channels; the observer samples
1/6 of the amplitude, yielding (1/6)^2 in power.
"""

BETA_ISO_F: float = float(BETA_ISO)

# E-folds: typically use D*_fp for the IR attractor
N_EFOLDS_FP: float = e_folds(D_STAR_FP, DELTA_F, DELTA_G_F)
"""E-folds using D*_fp ~ 59.95."""

N_EFOLDS_TREE: float = e_folds(D_STAR_TREE_F, DELTA_F, DELTA_G_F)
"""E-folds using D*_tree ~ 59.75."""

# Spectral index: n from tree (67/48), N_e from fp
N_S: float = spectral_index(N_TREE_F, N_EFOLDS_FP)
"""Spectral index n_s ~ 0.967 (uses n_tree, N_e from fp)."""

# Tensor-to-scalar ratio
R_TENSOR: float = tensor_to_scalar(N_TREE_F, N_EFOLDS_FP)
"""Tensor-to-scalar ratio r ~ 0.013."""

# Slow-roll parameters
EPSILON_SR: float = slow_roll_epsilon(N_TREE_F, N_EFOLDS_FP)
"""First slow-roll parameter epsilon ~ 5.5e-5."""

ETA_SR: float = slow_roll_eta(N_TREE_F, N_EFOLDS_FP)
"""Second slow-roll parameter eta ~ -0.017."""


# ── Registry ──────────────────────────────────────────────────────

def register_all(registry=REGISTRY) -> None:
    """Register inflationary observables."""
    registry.register(Observable(
        name="n_efolds",
        symbol="N_e",
        formula="(D*/Delta) * ln[(D* - 2 - delta) / (Delta * delta)]",
        predicted=N_EFOLDS_FP,
        observed=60.0,
        observed_uncertainty=10.0,
        unit="dimensionless",
        level=4,
        d_star_variant="fp",
        dependencies=("d_star_fp", "delta", "delta_g"),
    ))
    registry.register(Observable(
        name="n_s",
        symbol="n_s",
        formula="1 - 2(2n-1) / [N_e(2n-1) + n]",
        predicted=N_S,
        observed=0.9649,
        observed_uncertainty=0.0042,
        unit="dimensionless",
        level=4,
        d_star_variant="both",
        dependencies=("n_tree", "n_efolds"),
    ))
    registry.register(Observable(
        name="r_tensor",
        symbol="r",
        formula="48(2n-1)^2 / [N_e(2n-1) + n]^2",
        predicted=R_TENSOR,
        observed=0.036,
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=4,
        d_star_variant="both",
        dependencies=("n_tree", "n_efolds"),
        is_upper_limit=True,
    ))
    registry.register(Observable(
        name="beta_iso",
        symbol="beta_iso",
        formula="(1/6)^2 = 1/36",
        predicted=BETA_ISO_F,
        observed=0.038,
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=4,
        d_star_variant="none",
        dependencies=(),
        is_upper_limit=True,
    ))
    registry.register(Observable(
        name="epsilon_sr",
        symbol="epsilon",
        formula="(2n-1)(n-2)^2 / [N_e(2n-1)+n]^2",
        predicted=EPSILON_SR,
        observed=0.002,
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=4,
        d_star_variant="both",
        dependencies=("n_tree", "n_efolds"),
        is_upper_limit=True,
    ))
    registry.register(Observable(
        name="eta_sr",
        symbol="eta_sr",
        formula="(2n-1)(2n^2-7n+4)/[N_e(2n-1)+n]^2 - (2n-1)/[N_e(2n-1)+n]",
        predicted=ETA_SR,
        observed=0.1,
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=4,
        d_star_variant="both",
        dependencies=("n_tree", "n_efolds"),
        is_upper_limit=True,
    ))
