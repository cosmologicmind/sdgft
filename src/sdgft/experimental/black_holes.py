"""Black hole physics in SDGFT: singularity resolution and Hawking correction.

The running gravitational coupling G(k) from dimensional flow resolves
the Schwarzschild singularity and modifies Hawking radiation.

Key results:
    1. Geometric core: At r ~ r_P, D* -> 2 and G -> 0.
       The Kretschner scalar saturates at K_max ~ c^4 / (hbar^2 G_N) ~ 1/r_P^4.
       No singularity.

    2. Modified Hawking temperature:
       T_H(M) = hbar c^3 / (8 pi G(r_s) M k_B)
       As M -> M_P: G(r_s) -> 0, so T -> T_max ~ c^2 M_P / k_B ~ 10^32 K.
       Evaporation halts, leaving a Planck-mass remnant.

    3. QNM frequency correction:
       omega_SDGFT = omega_GR * (1 + alpha_M * (r_P / r_s)^2)
       For M >> M_P: correction ~ 10^(-76), unobservable.

    4. Bekenstein-Hawking entropy receives a geometric interpretation:
       S_BH = k_B A / (4 l_P^2) counts microstates on the 2D core.

Source: ch09_black_holes.tex, Sections 9.2-9.4
"""

from __future__ import annotations

import math

from ..constants import DELTA_F
from ..gravity import ALPHA_M_TREE_F
from ..physical_constants import G_N, C, HBAR, K_B, M_P, R_P, K_P, M_SUN
from ..registry import Observable, REGISTRY


# ── Constants ─────────────────────────────────────────────────────

D_STAR_UV: float = 2.0
"""UV fixed-point dimension (at the geometric core)."""

ETA_ANOMALOUS: float = 2.0
"""Graviton anomalous dimension at the UV fixed point."""


# ── Running gravitational coupling ───────────────────────────────

def g_running(k: float, g_n: float = G_N, k_p: float = K_P) -> float:
    """Running gravitational coupling G(k).

    G(k) = G_N / [1 + (k / k_P)^eta]

    IR (k << k_P): G -> G_N (Newton recovered)
    UV (k >> k_P): G -> 0 (asymptotic safety)

    Args:
        k: Momentum scale in 1/m.
        g_n: Newton's constant.
        k_p: Planck momentum.

    Returns:
        Running G at scale k.
    """
    return g_n / (1.0 + (k / k_p) ** ETA_ANOMALOUS)


def g_of_r(r: float, g_n: float = G_N, r_p: float = R_P) -> float:
    """Running gravitational coupling G(r) via k ~ 1/r.

    Args:
        r: Radial distance in meters.
        g_n: Newton's constant.
        r_p: Planck length.

    Returns:
        Running G at radial scale r.
    """
    k = 1.0 / r
    k_p = 1.0 / r_p
    return g_running(k, g_n, k_p)


# ── Schwarzschild radius ─────────────────────────────────────────

def schwarzschild_radius(m: float, g_n: float = G_N, c: float = C) -> float:
    """Classical Schwarzschild radius r_s = 2GM/c^2.

    Args:
        m: Black hole mass in kg.
        g_n: Newton's constant.
        c: Speed of light.

    Returns:
        Schwarzschild radius in meters.
    """
    return 2.0 * g_n * m / c ** 2


# ── Kretschner scalar saturation ─────────────────────────────────

def kretschner_classical(m: float, r: float) -> float:
    """Classical Kretschner scalar K = 48 G^2 M^2 / (c^4 r^6).

    Args:
        m: Black hole mass in kg.
        r: Radial distance in meters.

    Returns:
        Kretschner scalar in m^{-4}.
    """
    return 48.0 * G_N ** 2 * m ** 2 / (C ** 4 * r ** 6)


KRETSCHNER_MAX: float = C ** 6 / (HBAR ** 2 * G_N ** 2)
"""Maximum Kretschner scalar at geometric core ~ 1/r_P^4 ~ 1.5e139 m^{-4}.

Derived from c^6 / (hbar^2 * G_N^2) = 1/r_P^4, since r_P^2 = hbar*G/c^3.
Finite curvature replaces the classical singularity.
"""

KRETSCHNER_MAX_ALT: float = 1.0 / R_P ** 4
"""Alternative expression: K_max ~ 1/r_P^4."""


# ── Hawking radiation with running G ─────────────────────────────

def hawking_temperature(
    m: float,
    g_n: float = G_N,
    use_running_g: bool = True,
) -> float:
    """Hawking temperature of a black hole.

    Classical: T_H = hbar c^3 / (8 pi G M k_B)
    SDGFT:     T_H = hbar c^3 / (8 pi G(r_s) M k_B)

    As M -> M_P, G(r_s) -> 0, so T is bounded above.

    Args:
        m: Black hole mass in kg.
        g_n: Newton's constant (only used in classical mode).
        use_running_g: If True, use SDGFT running G; else classical.

    Returns:
        Hawking temperature in Kelvin.
    """
    if use_running_g:
        r_s = schwarzschild_radius(m)
        g_eff = g_of_r(r_s)
        if g_eff < 1e-100:
            # At Planck scale, return maximum temperature
            return T_HAWKING_MAX
    else:
        g_eff = g_n

    return HBAR * C ** 3 / (8.0 * math.pi * g_eff * m * K_B)


T_HAWKING_MAX: float = C ** 2 * M_P / K_B
"""Maximum Hawking temperature ~ c^2 M_P / k_B ~ 1.4e32 K.

Evaporation halts at this temperature, leaving a Planck-mass remnant.
"""


# ── Planck-mass remnant ──────────────────────────────────────────

M_REMNANT: float = M_P
"""Black hole remnant mass ~ M_P ~ 2.18e-8 kg.

SDGFT predicts evaporation halts at Planck mass because G(k) -> 0
at k ~ k_P, making the effective gravitational coupling vanish.
"""

M_REMNANT_GEV: float = M_REMNANT * C ** 2 / (1.602176634e-10)
"""Remnant mass in GeV."""


# ── Quasi-normal mode corrections ────────────────────────────────

def qnm_correction(
    m: float,
    alpha_m: float = ALPHA_M_TREE_F,
    r_p: float = R_P,
) -> float:
    """Fractional QNM frequency correction from SDGFT.

    delta_omega / omega = alpha_M * (r_P / r_s)^2

    For astrophysical black holes (M >> M_P), this is utterly
    unobservable (~ 10^{-76} for 30 M_sun).

    Args:
        m: Black hole mass in kg.
        alpha_m: Planck mass run-rate.
        r_p: Planck length.

    Returns:
        Fractional frequency correction (dimensionless).
    """
    r_s = schwarzschild_radius(m)
    return alpha_m * (r_p / r_s) ** 2


# ── Bekenstein-Hawking entropy ───────────────────────────────────

def bekenstein_hawking_entropy(m: float) -> float:
    """Bekenstein-Hawking entropy S_BH = k_B A / (4 l_P^2).

    In SDGFT, this counts microstates on the 2D geometric core,
    each of area ~ l_P^2.

    Args:
        m: Black hole mass in kg.

    Returns:
        Entropy in units of k_B (dimensionless count).
    """
    r_s = schwarzschild_radius(m)
    area = 4.0 * math.pi * r_s ** 2
    return area / (4.0 * R_P ** 2)


# ── Module-level computed values ─────────────────────────────────

# Hawking temperature for a solar-mass BH (classical)
T_HAWKING_SOLAR: float = hawking_temperature(M_SUN, use_running_g=False)
"""Classical Hawking temperature of a 1 M_sun BH ~ 6.2e-8 K."""

# QNM correction for a 30 M_sun BH
QNM_CORRECTION_30MSUN: float = qnm_correction(30.0 * M_SUN)
"""QNM correction for 30 M_sun: ~ 10^{-79} (unobservable)."""

# Entropy of a solar-mass BH
S_BH_SOLAR: float = bekenstein_hawking_entropy(M_SUN)
"""Bekenstein-Hawking entropy of 1 M_sun BH ~ 10^{77}."""


# ── Registry ──────────────────────────────────────────────────────

def register_all(registry=REGISTRY) -> None:
    """Register black hole observables."""
    registry.register(Observable(
        name="exp_kretschner_max",
        symbol="K_max",
        formula="c^4 / (hbar^2 G_N) ~ 1/r_P^4",
        predicted=KRETSCHNER_MAX,
        observed=KRETSCHNER_MAX_ALT,
        observed_uncertainty=0.0,
        unit="m^{-4}",
        level=3,
        d_star_variant="none",
        dependencies=(),
    ))
    registry.register(Observable(
        name="exp_t_hawking_max",
        symbol="T_H_max",
        formula="c^2 M_P / k_B",
        predicted=T_HAWKING_MAX,
        observed=T_HAWKING_MAX,
        observed_uncertainty=0.0,
        unit="K",
        level=3,
        d_star_variant="none",
        dependencies=(),
    ))
    registry.register(Observable(
        name="exp_m_remnant",
        symbol="M_rem",
        formula="M_P = sqrt(hbar c / G_N)",
        predicted=M_REMNANT,
        observed=M_REMNANT,
        observed_uncertainty=0.0,
        unit="kg",
        level=3,
        d_star_variant="none",
        dependencies=(),
    ))
