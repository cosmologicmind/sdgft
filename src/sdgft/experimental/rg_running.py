"""SM 1-loop RG running of gauge couplings with SDGFT boundary conditions.

Provides two key results:

1. Analytic verification that gamma_EW = sin^2(theta_W)(M_Z) - 1/9 = 0.120
   is arithmetic, not a dynamical computation.

2. Full SM 1-loop running of (alpha_1, alpha_2, alpha_3) showing:
   - The actual SM trajectory from M_Z to M_Pl
   - That sin^2(theta_W)(M_Pl) ~ 0.47 in the SM (not 1/9)
   - Where the GUT unification scale lies (alpha_1 = alpha_2)
   - SDGFT requires new physics above M_GUT to reach sin^2(theta_W) = 1/9

SDGFT boundary condition (ch04, Eq.4.3):
    sin^2(theta_W)(M_Pl) = (theta_max / 90)^2 = (30/90)^2 = 1/9
"""

from __future__ import annotations

import math

from ..constants import DELTA_F
from ..registry import Observable, REGISTRY


# ── Physical constants ────────────────────────────────────────────

M_Z: float = 91.1876
"""Z boson mass in GeV."""

M_PL: float = 1.2209e19
"""Planck mass in GeV."""

T_PL: float = math.log(M_PL / M_Z)
"""t_Pl = ln(M_Pl / M_Z) ~ 39.47. RG 'time' from M_Z to M_Pl."""

N_F: int = 3
"""Number of fermion generations (SM)."""

# Observed values at M_Z (MSbar, PDG 2022)
ALPHA_EM_INV_MZ: float = 127.952
"""alpha_em^{-1}(M_Z) in MSbar scheme."""

SIN2_THETA_W_MZ: float = 0.23122
"""sin^2(theta_W)(M_Z) in MSbar scheme."""

ALPHA_S_MZ: float = 0.1179
"""alpha_s(M_Z) in MSbar scheme."""

# SDGFT boundary condition
SIN2_THETA_W_PLANCK: float = 1.0 / 9.0
"""sin^2(theta_W)(M_Pl) = 1/9 (SDGFT geometric prediction)."""


# ── SM 1-loop beta coefficients (GUT normalization) ──────────────

B1: float = 41.0 / 10.0
"""1-loop coefficient for U(1)_Y (GUT-normalized)."""

B2: float = -19.0 / 6.0
"""1-loop coefficient for SU(2)_L."""

B3: float = -7.0
"""1-loop coefficient for SU(3)_c."""


# ── Coupling conversion formulas ─────────────────────────────────

def couplings_from_observables(
    alpha_em_inv: float,
    sin2_theta_w: float,
    alpha_s: float,
) -> tuple[float, float, float]:
    """Derive (1/alpha_1, 1/alpha_2, 1/alpha_3) from observables.

    Relations (GUT normalization, alpha_1_GUT = (5/3)*alpha_1_SM):
        sin^2(theta_W) = (3/5)*alpha_1 / ((3/5)*alpha_1 + alpha_2)
        1/alpha_em = (5/3)/alpha_1 + 1/alpha_2
        alpha_3 = alpha_s
    """
    sw = sin2_theta_w
    # 1/alpha_1 = (3/5) * alpha_em_inv * (1 - sw)
    # 1/alpha_2 = alpha_em_inv * sw
    # (These follow from the two equations above)
    inv_a1 = (3.0 / 5.0) * alpha_em_inv * (1.0 - sw)
    inv_a2 = alpha_em_inv * sw
    inv_a3 = 1.0 / alpha_s
    return inv_a1, inv_a2, inv_a3


def sin2_from_inv_couplings(inv_a1: float, inv_a2: float) -> float:
    """sin^2(theta_W) from inverse gauge couplings.

    sw = (3/5)*inv_a2 / ((3/5)*inv_a2 + inv_a1)

    Derived from sw = alpha_1_SM / (alpha_1_SM + alpha_2) with
    alpha_1_SM = (3/5)*alpha_1_GUT, multiplied through by inv_a1*inv_a2.
    """
    return (3.0 / 5.0) * inv_a2 / ((3.0 / 5.0) * inv_a2 + inv_a1)


def alpha_em_inv_from_couplings(inv_a1: float, inv_a2: float) -> float:
    """alpha_em^{-1} from inverse gauge couplings.

    1/alpha_em = 1/alpha_1_SM + 1/alpha_2
              = (5/3)*inv_a1_GUT + inv_a2

    Verified: (5/3)*59.02 + 29.58 = 127.95 = alpha_em^{-1}(M_Z).
    """
    return (5.0 / 3.0) * inv_a1 + inv_a2


# ── Analytic 1-loop running ──────────────────────────────────────

def run_inverse_couplings(
    inv_a1_0: float,
    inv_a2_0: float,
    inv_a3_0: float,
    delta_t: float,
) -> tuple[float, float, float]:
    """Analytic 1-loop running: 1/alpha_i(t) = 1/alpha_i(0) - b_i*t/(2*pi).

    This is EXACT for 1-loop and does not require numerical integration.

    Args:
        inv_a1_0, inv_a2_0, inv_a3_0: Inverse couplings at starting scale.
        delta_t: RG 'time' = ln(mu_final/mu_initial).

    Returns:
        Tuple of inverse couplings at final scale.
    """
    TWO_PI = 2.0 * math.pi
    return (
        inv_a1_0 - B1 * delta_t / TWO_PI,
        inv_a2_0 - B2 * delta_t / TWO_PI,
        inv_a3_0 - B3 * delta_t / TWO_PI,
    )


def run_to_scale(
    t: float,
    inv_a1_mz: float | None = None,
    inv_a2_mz: float | None = None,
    inv_a3_mz: float | None = None,
) -> dict[str, float]:
    """Run gauge couplings from M_Z to scale mu = M_Z * exp(t).

    Uses default observed values at M_Z if not specified.

    Returns:
        Dictionary with all derived quantities at scale mu.
    """
    if inv_a1_mz is None or inv_a2_mz is None or inv_a3_mz is None:
        ia1, ia2, ia3 = couplings_from_observables(
            ALPHA_EM_INV_MZ, SIN2_THETA_W_MZ, ALPHA_S_MZ
        )
        inv_a1_mz = inv_a1_mz or ia1
        inv_a2_mz = inv_a2_mz or ia2
        inv_a3_mz = inv_a3_mz or ia3

    ia1, ia2, ia3 = run_inverse_couplings(inv_a1_mz, inv_a2_mz, inv_a3_mz, t)
    sw = sin2_from_inv_couplings(ia1, ia2)
    aem_inv = alpha_em_inv_from_couplings(ia1, ia2)

    return {
        "inv_alpha_1": ia1,
        "inv_alpha_2": ia2,
        "inv_alpha_3": ia3,
        "sin2_theta_w": sw,
        "alpha_em_inv": aem_inv,
        "alpha_s": 1.0 / ia3 if ia3 > 0 else float("inf"),
        "scale_gev": M_Z * math.exp(t),
    }


def find_unification_scale() -> tuple[float, float]:
    """Find the GUT scale where alpha_1 = alpha_2.

    At unification: 1/alpha_1 = 1/alpha_2
    => inv_a1_0 - b1*t/(2pi) = inv_a2_0 - b2*t/(2pi)
    => t_GUT = 2*pi*(inv_a1_0 - inv_a2_0) / (b1 - b2)
    """
    ia1, ia2, ia3 = couplings_from_observables(
        ALPHA_EM_INV_MZ, SIN2_THETA_W_MZ, ALPHA_S_MZ
    )
    t_gut = 2.0 * math.pi * (ia1 - ia2) / (B1 - B2)
    m_gut = M_Z * math.exp(t_gut)
    return t_gut, m_gut


def rg_trajectory(n_points: int = 200) -> list[dict[str, float]]:
    """Compute the full SM RG trajectory from M_Z to M_Pl.

    Returns a list of dictionaries, each containing coupling values
    at a given scale.
    """
    points = []
    for i in range(n_points + 1):
        t = T_PL * i / n_points
        points.append(run_to_scale(t))
    return points


# ── Module-level computed values ──────────────────────────────────

# Couplings at M_Z
_IA1_MZ, _IA2_MZ, _IA3_MZ = couplings_from_observables(
    ALPHA_EM_INV_MZ, SIN2_THETA_W_MZ, ALPHA_S_MZ
)

# Run to Planck scale
_AT_PLANCK = run_to_scale(T_PL)
SIN2_THETA_W_SM_PLANCK: float = _AT_PLANCK["sin2_theta_w"]
"""sin^2(theta_W) at M_Pl from SM 1-loop running (~0.47).
    This does NOT agree with the SDGFT prediction of 1/9.
    The discrepancy indicates SDGFT requires new physics above M_GUT."""

# gamma_EW: the arithmetic definition (not from RG)
GAMMA_EW_COMPUTED: float = SIN2_THETA_W_MZ - SIN2_THETA_W_PLANCK
"""gamma_EW defined as sin^2(theta_W)(M_Z) - sin^2(theta_W)(M_Pl).
    Using SDGFT's M_Pl value (1/9): gamma_EW = 0.231 - 0.111 = 0.120.
    Using SM running to M_Pl: gamma_EW = 0.231 - 0.47 = -0.24 (inconsistent!)."""

GAMMA_EW_SDGFT: float = SIN2_THETA_W_MZ - SIN2_THETA_W_PLANCK
"""Same as GAMMA_EW_COMPUTED. Included for naming clarity."""

# What SDGFT predicts: gamma_EW = sin2(M_Z) - 1/9
GAMMA_EW_ARITHMETIC: float = SIN2_THETA_W_MZ - 1.0 / 9.0
"""gamma_EW = sin^2(theta_W)(M_Z) - 1/9 = 0.120.
    This is the value used in the base SDGFT framework.
    It is an arithmetic identity, not a dynamical RG result."""

# sin^2(theta_W) using SDGFT RG (= 1/9 + gamma_EW_arithmetic = observed)
SIN2_THETA_W_RG: float = 1.0 / 9.0 + GAMMA_EW_ARITHMETIC
"""sin^2(theta_W)(M_Z) = 1/9 + gamma_EW. This trivially equals the observed value
    because gamma_EW is defined as the difference."""

# GUT unification
T_GUT, M_GUT = find_unification_scale()
"""GUT scale where alpha_1 = alpha_2."""

# alpha_s at M_Z from running (just the observed value since we start there)
ALPHA_S_RG: float = ALPHA_S_MZ
"""alpha_s(M_Z). In this module we start from observed M_Z values,
    so this is trivially the input value."""

# The ratio alpha_1/alpha_2 at M_Pl (SDGFT)
# If sin^2(theta_W)(M_Pl) = 1/9, then:
# alpha_1/alpha_2 = (5/3) * sw/(1-sw) = (5/3) * (1/9)/(8/9) = 5/24 = Delta!
ALPHA_RATIO_SDGFT: float = (5.0 / 3.0) * (1.0 / 9.0) / (8.0 / 9.0)
"""alpha_1_GUT/alpha_2 at M_Pl IF sin^2(theta_W) = 1/9.
    Equals 5/24 = Delta. This is a remarkable coincidence (or not)."""


# ── Registry ──────────────────────────────────────────────────────

def register_all(registry=REGISTRY) -> None:
    """Register RG running observables."""
    registry.register(Observable(
        name="exp_sin2_theta_w",
        symbol="sin^2(theta_W)_RG",
        formula="1/9 + (sin^2(theta_W)_obs - 1/9)",
        predicted=SIN2_THETA_W_RG,
        observed=0.23122,
        observed_uncertainty=0.00003,
        unit="dimensionless",
        level=5,
        d_star_variant="none",
        dependencies=(),
        is_diagnostic=True,
    ))  # Tautological: obs - 1/9 + 1/9 = obs. No predictive content.
    registry.register(Observable(
        name="exp_gamma_ew",
        symbol="gamma_EW",
        formula="sin^2(theta_W)(M_Z) - 1/9",
        predicted=GAMMA_EW_ARITHMETIC,
        observed=0.120,
        observed_uncertainty=0.001,
        unit="dimensionless",
        level=5,
        d_star_variant="none",
        dependencies=(),
    ))
    registry.register(Observable(
        name="exp_sin2_theta_w_sm_planck",
        symbol="sin^2(theta_W)(M_Pl)_SM",
        formula="SM 1-loop running from M_Z to M_Pl",
        predicted=SIN2_THETA_W_SM_PLANCK,
        observed=SIN2_THETA_W_PLANCK,
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=5,
        d_star_variant="none",
        dependencies=(),
        is_diagnostic=True,
    ))  # Diagnostic: shows SM running doesn't reach 1/9 => BSM above M_GUT needed.
