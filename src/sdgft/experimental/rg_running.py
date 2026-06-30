"""SM 1-loop and 2-loop RG running of gauge couplings with SDGFT boundary conditions.

Provides three key results:

1. Analytic verification that gamma_EW = sin^2(theta_W)(M_Z) - 1/9 = 0.120
   is arithmetic, not a dynamical computation.

2. Full SM 1-loop and 2-loop running of (alpha_1, alpha_2, alpha_3) showing:
   - The actual SM trajectory from M_Z to M_Pl
   - That sin^2(theta_W)(M_Pl) ~ 0.47 in the SM (not 1/9)
   - Where the GUT unification scale lies (alpha_1 = alpha_2)
   - SDGFT requires new physics above M_GUT to reach sin^2(theta_W) = 1/9

SDGFT boundary condition (ch04, Eq.4.3):
    sin^2(theta_W)(M_Pl) = (theta_max / 90)^2 = (30/90)^2 = 1/9
"""

from __future__ import annotations

import math
from typing import Tuple, List

from ..constants import DELTA_F
from ..registry import Observable, REGISTRY


# ── Physical constants ────────────────────────────────────────────

M_Z: float = 91.1876
MZ: float = M_Z
"""Z boson mass in GeV."""

M_PL: float = 1.2209e19
"""Planck mass in GeV."""

MT: float = 173.1
"""Top quark threshold in GeV."""

T_PL: float = math.log(M_PL / M_Z)
"""t_Pl = ln(M_Pl / M_Z) ~ 39.47. RG 'time' from M_Z to M_Pl."""

N_F: int = 3
"""Number of fermion generations (SM)."""

# Observed values at M_Z (MSbar, PDG 2022)
ALPHA_EM_INV_MZ: float = 127.952
"""alpha_em^{-1}(M_Z) in MSbar scheme."""

SIN2_THETA_W_MZ: float = 0.23122
"""sin^2(theta_W)(M_Z) in MSbar scheme."""

# Fundamental SDGFT Gauge Constants
ALPHA_S_MZ_EXACT: float = math.sqrt(2) / 12.0  # Exact topology value ~0.117851
ALPHA_S_MZ: float = ALPHA_S_MZ_EXACT
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
"""1-loop coefficient for SU(3)_c (above MT)."""

# 1-loop beta coefficients b_i (GUT normalization for g_1)
B_N_F5: List[float] = [4.1, -19.0/6.0, -23.0/3.0]
B_N_F6: List[float] = [4.1, -19.0/6.0, -7.0]

# 2-loop beta matrices b_ij
B_IJ_N_F5: List[List[float]] = [
    [199.0/50.0, 27.0/10.0,  44.0/5.0],
    [9.0/10.0,   35.0/6.0,   12.0],
    [11.0/10.0,  9.0/2.0,   -116.0/3.0]
]

B_IJ_N_F6: List[List[float]] = [
    [199.0/50.0, 27.0/10.0,  44.0/5.0],
    [9.0/10.0,   35.0/6.0,   12.0],
    [11.0/10.0,  9.0/2.0,   -26.0]
]


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
    inv_a1 = (3.0 / 5.0) * alpha_em_inv * (1.0 - sw)
    inv_a2 = alpha_em_inv * sw
    inv_a3 = 1.0 / alpha_s
    return inv_a1, inv_a2, inv_a3


def sin2_from_inv_couplings(inv_a1: float, inv_a2: float) -> float:
    """sin^2(theta_W) from inverse gauge couplings.

    sw = (3/5)*inv_a2 / ((3/5)*inv_a2 + inv_a1)
    """
    return (3.0 / 5.0) * inv_a2 / ((3.0 / 5.0) * inv_a2 + inv_a1)


def alpha_em_inv_from_couplings(inv_a1: float, inv_a2: float) -> float:
    """alpha_em^{-1} from inverse gauge couplings.

    1/alpha_em = (5/3)*inv_a1 + inv_a2
    """
    return (5.0 / 3.0) * inv_a1 + inv_a2


# ── Beta Coefficients and RGE Derivatives ────────────────────────

def get_beta_coefficients(mu: float) -> Tuple[List[float], List[List[float]]]:
    """Returns the 1-loop and 2-loop beta coefficients based on the energy scale (thresholds)."""
    if mu < MT:
        return B_N_F5, B_IJ_N_F5
    return B_N_F6, B_IJ_N_F6


def rge_derivative_2loop(t: float, alpha_inv: List[float], mu_0: float) -> List[float]:
    """Calculates d(alpha_i^-1)/dt at 2-loop order where t = ln(mu/mu_0)."""
    mu = mu_0 * math.exp(t)
    b_i, b_ij = get_beta_coefficients(mu)
    
    # Recover alpha values from their inverses
    alphas = [1.0 / a_inv for a_inv in alpha_inv]
    
    derivatives = []
    for i in range(3):
        # 1-loop contribution (with correct negative sign for d(alpha^-1)/dt)
        term_1loop = -b_i[i] / (2.0 * math.pi)
        
        # 2-loop contribution sum_j (b_ij * alpha_j) (with correct negative sign for d(alpha^-1)/dt)
        term_2loop = 0.0
        for j in range(3):
            term_2loop += b_ij[i][j] * alphas[j]
        term_2loop /= (8.0 * math.pi**2)
        
        derivatives.append(term_1loop - term_2loop)
        
    return derivatives


# ── RGE Solvers ──────────────────────────────────────────────────

def run_inverse_couplings(
    inv_a1_0: float,
    inv_a2_0: float,
    inv_a3_0: float,
    delta_t: float,
    mu_0: float = M_Z,
) -> tuple[float, float, float]:
    """Analytic 1-loop running including the top quark mass threshold at M_T = 173.1 GeV."""
    mu_final = mu_0 * math.exp(delta_t)
    TWO_PI = 2.0 * math.pi
    
    ia1 = inv_a1_0 - B1 * delta_t / TWO_PI
    ia2 = inv_a2_0 - B2 * delta_t / TWO_PI
    
    # SU(3)_c has flavor threshold at M_T = 173.1 GeV
    b3_5 = -23.0 / 3.0  # N_f = 5
    b3_6 = -7.0         # N_f = 6
    
    if mu_0 >= MT:
        ia3 = inv_a3_0 - b3_6 * delta_t / TWO_PI
    elif mu_final <= MT:
        ia3 = inv_a3_0 - b3_5 * delta_t / TWO_PI
    else:
        t_to_mt = math.log(MT / mu_0)
        t_from_mt = delta_t - t_to_mt
        ia3_mt = inv_a3_0 - b3_5 * t_to_mt / TWO_PI
        ia3 = ia3_mt - b3_6 * t_from_mt / TWO_PI
        
    return ia1, ia2, ia3


def run_2loop_rge(
    mu_target: float,
    alpha_inv_mz: List[float],
    steps: int = 1000
) -> List[float]:
    """Integrates the 2-loop SM RGEs from M_Z to the target scale using RK4 integration."""
    t_end = math.log(mu_target / MZ)
    dt = t_end / steps
    
    current_alpha_inv = list(alpha_inv_mz)
    t = 0.0
    
    for _ in range(steps):
        # RK4 Steps
        k1 = rge_derivative_2loop(t, current_alpha_inv, MZ)
        
        state_k2 = [current_alpha_inv[i] + 0.5 * dt * k1[i] for i in range(3)]
        k2 = rge_derivative_2loop(t + 0.5 * dt, state_k2, MZ)
        
        state_k3 = [current_alpha_inv[i] + 0.5 * dt * k2[i] for i in range(3)]
        k3 = rge_derivative_2loop(t + 0.5 * dt, state_k3, MZ)
        
        state_k4 = [current_alpha_inv[i] + dt * k3[i] for i in range(3)]
        k4 = rge_derivative_2loop(t + dt, state_k4, MZ)
        
        for i in range(3):
            current_alpha_inv[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            
        t += dt
        
    return current_alpha_inv


def run_to_scale(
    t: float,
    inv_a1_mz: float | None = None,
    inv_a2_mz: float | None = None,
    inv_a3_mz: float | None = None,
    loop_order: int = 1,
) -> dict[str, float]:
    """Run gauge couplings from M_Z to scale mu = M_Z * exp(t).

    Uses default observed values at M_Z if not specified.
    """
    if inv_a1_mz is None or inv_a2_mz is None or inv_a3_mz is None:
        ia1, ia2, ia3 = couplings_from_observables(
            ALPHA_EM_INV_MZ, SIN2_THETA_W_MZ, ALPHA_S_MZ
        )
        inv_a1_mz = inv_a1_mz or ia1
        inv_a2_mz = inv_a2_mz or ia2
        inv_a3_mz = inv_a3_mz or ia3

    mu_target = M_Z * math.exp(t)
    
    if loop_order == 2:
        res = run_2loop_rge(mu_target, [inv_a1_mz, inv_a2_mz, inv_a3_mz])
        ia1, ia2, ia3 = res[0], res[1], res[2]
    else:
        ia1, ia2, ia3 = run_inverse_couplings(inv_a1_mz, inv_a2_mz, inv_a3_mz, t, M_Z)

    sw = sin2_from_inv_couplings(ia1, ia2)
    aem_inv = alpha_em_inv_from_couplings(ia1, ia2)

    return {
        "inv_alpha_1": ia1,
        "inv_alpha_2": ia2,
        "inv_alpha_3": ia3,
        "sin2_theta_w": sw,
        "alpha_em_inv": aem_inv,
        "alpha_s": 1.0 / ia3 if ia3 > 0 else float("inf"),
        "scale_gev": mu_target,
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


def rg_trajectory(n_points: int = 200, loop_order: int = 1) -> list[dict[str, float]]:
    """Compute the full SM RG trajectory from M_Z to M_Pl.

    Returns a list of dictionaries, each containing coupling values
    at a given scale.
    """
    points = []
    for i in range(n_points + 1):
        t = T_PL * i / n_points
        points.append(run_to_scale(t, loop_order=loop_order))
    return points


# ── Module-level computed values ──────────────────────────────────

# Couplings at M_Z
_IA1_MZ, _IA2_MZ, _IA3_MZ = couplings_from_observables(
    ALPHA_EM_INV_MZ, SIN2_THETA_W_MZ, ALPHA_S_MZ
)

# Run to Planck scale (using default 1-loop for backwards consistency)
_AT_PLANCK = run_to_scale(T_PL)
SIN2_THETA_W_SM_PLANCK: float = _AT_PLANCK["sin2_theta_w"]

# gamma_EW: the arithmetic definition (not from RG)
GAMMA_EW_COMPUTED: float = SIN2_THETA_W_MZ - SIN2_THETA_W_PLANCK
GAMMA_EW_SDGFT: float = SIN2_THETA_W_MZ - SIN2_THETA_W_PLANCK

# What SDGFT predicts: gamma_EW = sin2(M_Z) - 1/9
GAMMA_EW_ARITHMETIC: float = SIN2_THETA_W_MZ - 1.0 / 9.0

# sin^2(theta_W) using SDGFT RG (= 1/9 + gamma_EW_arithmetic = observed)
SIN2_THETA_W_RG: float = 1.0 / 9.0 + GAMMA_EW_ARITHMETIC

# GUT unification
T_GUT, M_GUT = find_unification_scale()

# alpha_s at M_Z from running (just the observed value since we start there)
ALPHA_S_RG: float = ALPHA_S_MZ

# The ratio alpha_1/alpha_2 at M_Pl (SDGFT)
ALPHA_RATIO_SDGFT: float = (5.0 / 3.0) * (1.0 / 9.0) / (8.0 / 9.0)


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
    ))
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
    ))
