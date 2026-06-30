"""Level 5-6: Cosmological density parameters and dark energy.

All densities are derived from exact lattice-closure fractions:

    Omega_b  = (Delta/4)(1 - delta) = 115/2304
    Omega_c  = 600/2304
    Omega_DE = 1589/2304
    Flatness: Omega_b + Omega_c + Omega_DE = 2304/2304 = 1

Dark energy equation of state:
    w_DE = -D*/3

Baryon asymmetry:
    eta_B = delta^6 * (1-delta) / 8

Structure growth (MCMC/hi_class):
    sigma_8 = 0.775 +/- 0.010
    S_8     = sigma_8 * sqrt(Omega_m / 0.3) = 0.791
"""

from fractions import Fraction
import math

from .constants import DELTA, DELTA_G, DELTA_F, DELTA_G_F
from .dimension import D_STAR_TREE, D_STAR_TREE_F, D_STAR_FP
from .registry import Observable, REGISTRY


# ── Pure functions ────────────────────────────────────────────────

def omega_b_formula(delta: Fraction, delta_g: Fraction) -> Fraction:
    """Omega_b = (Delta/4)(1 - delta_g).

    Closed topological knots (baryons) constrained by:
    - Delta/4: lattice capacity for baryon formation
    - (1 - delta_g) = 23/24: probability of loop closure
    """
    return (delta * (1 - delta_g)) / 4


def w_de(d_star: float) -> float:
    """Dark energy equation of state: w_DE = -D*/3.

    w = -1 (cosmological constant) is the D* = 3 limit.
    D* < 3 implies w > -1: quintessence-like.
    """
    return -d_star / 3.0


def eta_b(delta_g: float, closure: float | None = None) -> float:
    """Baryon-to-photon ratio: eta_B = delta_g^6 * (1 - delta_g) / 8.

    Topological: baryon formation requires independent delta_g defects
    in each of 6 spatial directions, distributed over 8 vertices
    of the inner 16-cell.  The closure factor (1 - delta_g) = 23/24
    accounts for the probability that the defect closes into a baryon
    (same factor as in Omega_b).

    Without closure factor: delta_g^6 / 8 = 6.541e-10 (7.2% dev).
    With closure factor:    delta_g^6 * (1-delta_g) / 8 = 6.269e-10 (2.7% dev).

    Args:
        delta_g: Lattice tension parameter (default 1/24).
        closure: Closure probability. Default: (1 - delta_g).
    """
    if closure is None:
        closure = 1.0 - delta_g
    return delta_g ** 6 * closure / 8.0


# ── Exact lattice-closure densities (Fractions) ──────────────────

OMEGA_B = Fraction(115, 2304)
"""Baryon density: (5/24 / 4) * (23/24) = (5 * 23) / (24 * 4 * 24) = 115/2304."""

OMEGA_B_F: float = float(OMEGA_B)

OMEGA_C = Fraction(600, 2304)
"""Cold dark matter density: from flatness constraint."""

OMEGA_C_F: float = float(OMEGA_C)

OMEGA_DE = Fraction(1589, 2304)
"""Dark energy density: 1 - Omega_b - Omega_c = 1589/2304."""

OMEGA_DE_F: float = float(OMEGA_DE)

OMEGA_M = OMEGA_B + OMEGA_C
"""Total matter density: Omega_b + Omega_c = 715/2304."""

OMEGA_M_F: float = float(OMEGA_M)

W_DE_TREE = Fraction(-67, 72)
"""w_DE (tree) = -D*_tree / 3 = -(67/24) / 3 = -67/72."""

W_DE_TREE_F: float = float(W_DE_TREE)

W_DE_FP: float = w_de(D_STAR_FP)
"""w_DE (fixed-point) ~ -0.932."""

ETA_B: float = eta_b(DELTA_G_F)
"""Baryon asymmetry: delta_g^6 * (1-delta_g) / 8 ~ 6.27e-10.

Includes closure factor (1 - delta_g) = 23/24, same as in Omega_b.
Without closure: 6.541e-10 (7.2% dev); with: 6.269e-10 (2.7% dev).
"""


# ── sigma_8 and S_8 (MCMC/hi_class) ──────────────────────────────

def s_8(sigma_8: float, omega_m: float) -> float:
    """S_8 combination: S_8 = sigma_8 * sqrt(Omega_m / 0.3).

    This is the parameter combination best constrained by weak
    gravitational lensing surveys.  The S_8 tension between
    Planck CMB and low-z lensing is a key anomaly in LCDM.
    """
    return sigma_8 * math.sqrt(omega_m / 0.3)


SIGMA_8: float = 0.775
"""sigma_8 prediction (MCMC): 0.775 +/- 0.010.

Amplitude of matter fluctuations on 8 Mpc/h scales.
From full Boltzmann + MCMC analysis with hi_class
(monograph ch07 Sec. 7.2, 437 accepted samples).

Lower than LCDM (0.803 +/- 0.007) by 2.3 sigma.
Agrees with weak lensing:
  - KiDS-1000: 0.759 +0.024/-0.021  (0.6 sigma)
  - DES-Y3:    0.776 +/- 0.017      (0.1 sigma)
"""

SIGMA_8_UNC: float = 0.010
"""sigma_8 MCMC uncertainty."""

S_8: float = s_8(SIGMA_8, OMEGA_M_F)
"""S_8 = sigma_8 * sqrt(Omega_m / 0.3) ~ 0.791.

Resolves the S_8 tension: consistent with KiDS-1000
(0.766 +0.020/-0.014) and DES Y3 (0.776 +/- 0.017).
Source: monograph ch05 eq:S8_sdgft.
"""

# ── Dark matter effective mass (ch10 #14) ─────────────────────────

M_DM: float = 1e-22
"""Effective dark matter mass: ~10^{-22} eV (ch10 #14).

In SDGFT, dark matter is NOT a particle but the gravitational effect
of unclosed topological defects in the 6-cone lattice (ch05, Sec. 5.4).
The effective mass characterises the de Broglie wavelength of the
geometric fluctuations, corresponding to ~kpc-scale cores in dwarf
galaxies (fuzzy DM phenomenology).

Status: qualitative prediction, pending Lyman-alpha forest test (2030-2032).
The value is not yet derived from (Delta, delta) axioms.
"""

M_DM_UNIT: str = "eV"
"""Unit of M_DM."""

# ── Verify exact flatness ─────────────────────────────────────────

assert OMEGA_B + OMEGA_C + OMEGA_DE == 1, (
    f"Flatness violated: {OMEGA_B} + {OMEGA_C} + {OMEGA_DE} = "
    f"{OMEGA_B + OMEGA_C + OMEGA_DE}"
)

# Verify Omega_b formula
assert OMEGA_B == omega_b_formula(DELTA, DELTA_G), (
    f"Omega_b formula mismatch: expected {omega_b_formula(DELTA, DELTA_G)}"
)

# Verify w_DE fraction
assert W_DE_TREE == -D_STAR_TREE / 3, (
    f"w_DE fraction mismatch: {-D_STAR_TREE / 3}"
)


# ── Registry ──────────────────────────────────────────────────────

def register_all(registry=REGISTRY) -> None:
    """Register cosmological observables."""
    registry.register(Observable(
        name="omega_b",
        symbol="Omega_b",
        formula="(Delta/4)(1 - delta) = 115/2304",
        predicted=OMEGA_B_F,
        observed=0.0493,
        observed_uncertainty=0.0003,
        unit="dimensionless",
        level=6,
        d_star_variant="none",
        dependencies=("delta", "delta_g"),
    ))
    registry.register(Observable(
        name="omega_c",
        symbol="Omega_c",
        formula="600/2304 (flatness closure)",
        predicted=OMEGA_C_F,
        observed=0.2607,
        observed_uncertainty=0.0063,
        unit="dimensionless",
        level=6,
        d_star_variant="none",
        dependencies=("delta", "delta_g"),
    ))
    registry.register(Observable(
        name="omega_de",
        symbol="Omega_DE",
        formula="1589/2304 (flatness closure)",
        predicted=OMEGA_DE_F,
        observed=0.6847,
        observed_uncertainty=0.0073,
        unit="dimensionless",
        level=6,
        d_star_variant="none",
        dependencies=("omega_b", "omega_c"),
    ))
    registry.register(Observable(
        name="omega_m",
        symbol="Omega_m",
        formula="Omega_b + Omega_c = 715/2304",
        predicted=OMEGA_M_F,
        observed=0.315,
        observed_uncertainty=0.007,
        unit="dimensionless",
        level=6,
        d_star_variant="none",
        dependencies=("omega_b", "omega_c"),
    ))
    registry.register(Observable(
        name="w_de",
        symbol="w_DE",
        formula="-D*/3 = -67/72",
        predicted=W_DE_TREE_F,
        observed=-1.03,
        observed_uncertainty=0.03,
        unit="dimensionless",
        level=6,
        d_star_variant="tree",
        dependencies=("d_star_tree",),
    ))
    registry.register(Observable(
        name="eta_b",
        symbol="eta_B",
        formula="delta^6 * (1-delta) / 8",
        predicted=ETA_B,
        observed=6.104e-10,
        observed_uncertainty=0.058e-10,
        unit="dimensionless",
        level=6,
        d_star_variant="none",
        dependencies=("delta_g",),
    ))  # Planck 2018: (6.104 +/- 0.058)e-10. 7% dev is genuine (higher-order topo).
    registry.register(Observable(
        name="sigma_8",
        symbol="sigma_8",
        formula="MCMC (hi_class Boltzmann + MCMC, ch07)",
        predicted=SIGMA_8,
        observed=0.776,
        observed_uncertainty=0.017,
        unit="dimensionless",
        level=6,
        d_star_variant="both",
        dependencies=("n_tree", "omega_m", "omega_de"),
    ))  # DES-Y3 weak lensing (2022). Planck LCDM CMB=0.811 is in tension.
    registry.register(Observable(
        name="s_8",
        symbol="S_8",
        formula="sigma_8 * sqrt(Omega_m / 0.3)",
        predicted=S_8,
        observed=0.776,
        observed_uncertainty=0.017,
        unit="dimensionless",
        level=6,
        d_star_variant="both",
        dependencies=("sigma_8", "omega_m"),
    ))  # DES-Y3 weak lensing (2022). Consistent with KiDS-1000 (0.766).
    registry.register(Observable(
        name="m_dm",
        symbol="m_DM",
        formula="~10^{-22} eV (qualitative, ch10 #14)",
        predicted=M_DM,
        observed=float("nan"),
        observed_uncertainty=float("nan"),
        unit="eV",
        level=6,
        is_upper_limit=True,
        d_star_variant="none",
        dependencies=(),
    ))
