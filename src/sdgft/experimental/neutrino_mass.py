"""Neutrino mass sum prediction from SDGFT.

SDGFT predicts a total neutrino mass sum from the 6-cone geometry:

    sum(m_nu) = delta_g * m_e * (v / M_Pl)^(1/3)

where the factors encode:
    - delta_g = 1/24: single-vertex sampling (mass suppression)
    - m_e: electron mass as the base lepton scale
    - (v / M_Pl)^(1/3): seesaw suppression from the ratio of
      electroweak VEV to Planck mass, cube-rooted for 3 generations

This gives sum(m_nu) ~ 0.058 eV, at the lower edge of the
normal hierarchy minimum (0.06 eV) and well below the current
cosmological upper bound (< 0.12 eV from Planck + BAO).

This is one of SDGFT's 5 "Scientific Bet" predictions from ch10,
testable by KATRIN and cosmological surveys by ~2028.

Source: ch10_outlook.tex, Prediction #7
"""

from __future__ import annotations

import math

from ..constants import DELTA_G_F
from ..physical_constants import M_PL_GEV
from ..registry import Observable, REGISTRY


# ── Physical constants (particle physics) ─────────────────────────

M_E_GEV: float = 0.000511
"""Electron mass in GeV."""

V_HIGGS_GEV: float = 246.22
"""Higgs vacuum expectation value in GeV."""


# ── Neutrino mass sum formula ────────────────────────────────────

def neutrino_mass_sum(
    delta_g: float = DELTA_G_F,
    m_e: float = M_E_GEV,
    v: float = V_HIGGS_GEV,
    m_pl: float = M_PL_GEV,
) -> float:
    """Total neutrino mass sum in eV.

    sum(m_nu) = delta_g * m_e * (v / M_Pl)^(1/3)

    Physical reasoning:
    - delta_g = 1/24 is the single-vertex mass suppression
    - (v / M_Pl)^(1/3) is the seesaw-like suppression,
      cube-rooted because the mass is shared across 3 generations

    Args:
        delta_g: Elementary lattice tension (1/24).
        m_e: Electron mass in GeV.
        v: Higgs VEV in GeV.
        m_pl: Planck mass in GeV.

    Returns:
        sum(m_nu) in eV.
    """
    # Compute in GeV, convert to eV at the end
    sum_gev = delta_g * m_e * (v / m_pl) ** (1.0 / 3.0)
    return sum_gev * 1e9  # GeV -> eV


# ── Individual mass estimates (normal hierarchy) ─────────────────

def mass_splitting_normal() -> tuple[float, float, float]:
    """Estimate individual neutrino masses assuming normal hierarchy.

    Uses observed mass-squared differences:
        Delta m^2_21 = 7.53e-5 eV^2  (solar)
        Delta m^2_32 = 2.453e-3 eV^2 (atmospheric)

    With m_1 as the lightest mass, solves:
        m_1^2 + m_2^2 + m_3^2 and sum constraint.

    Returns:
        Tuple of (m_1, m_2, m_3) in eV.
    """
    dm21_sq = 7.53e-5  # eV^2
    dm32_sq = 2.453e-3  # eV^2

    sigma = neutrino_mass_sum()

    # Iterative solve: m_1 + sqrt(m_1^2 + dm21_sq) + sqrt(m_1^2 + dm21_sq + dm32_sq) = sigma
    m1 = 0.0
    for _ in range(100):
        m2 = math.sqrt(m1 ** 2 + dm21_sq)
        m3 = math.sqrt(m1 ** 2 + dm21_sq + dm32_sq)
        total = m1 + m2 + m3
        if abs(total - sigma) < 1e-10:
            break
        # Newton step
        if total > sigma:
            m1 = max(0.0, m1 - 0.001)
        else:
            m1 += 0.001

    m2 = math.sqrt(m1 ** 2 + dm21_sq)
    m3 = math.sqrt(m1 ** 2 + dm21_sq + dm32_sq)
    return m1, m2, m3


# ── Module-level computed values ─────────────────────────────────

SUM_M_NU: float = neutrino_mass_sum()
"""Total neutrino mass sum ~ 0.058 eV."""

M_NU_1, M_NU_2, M_NU_3 = mass_splitting_normal()
"""Individual neutrino mass estimates (normal hierarchy)."""


# ── Registry ──────────────────────────────────────────────────────

def register_all(registry=REGISTRY) -> None:
    """Register neutrino mass observables."""
    registry.register(Observable(
        name="exp_sum_m_nu",
        symbol="sum(m_nu)",
        formula="delta_g * m_e * (v/M_Pl)^(1/3)",
        predicted=SUM_M_NU,
        observed=0.12,
        observed_uncertainty=0.0,
        unit="eV",
        level=6,
        d_star_variant="none",
        dependencies=("delta_g",),
        is_upper_limit=True,
    ))
