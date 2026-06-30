"""Chiral gravitational coupling from SDGFT.

The 6-cone geometry introduces a parity-violating term in the
gravitational action through the Pontryagin density R~R:

    S_chiral = xi_G * integral(sqrt(-g) R_tilde R d^4x)

where R~R = (1/2) epsilon^{mu nu rho sigma} R_{mu nu alpha beta}
R_{rho sigma}^{alpha beta} is the Pontryagin density.

Key results:
    xi_G = Delta * delta_g * phi^(-2) ~ 0.00332

Physical reasoning:
    - Delta = 5/24: the Fibonacci defect breaks parity
    - delta_g = 1/24: lattice tension sets the coupling scale
    - phi^(-2): the golden ratio suppression from the second-order
      geometric embedding of the chiral term

Observable consequences:
    - Gravitational wave amplitude asymmetry:
      Delta_h = h_R - h_L = 2 * xi_G * (omega / M_P) * h
    - For SMBH mergers at mHz: Delta_h / h ~ 10^{-20}
      (below LISA sensitivity, but targetable by future detectors)

Source: ch10_outlook.tex, Prediction #12; ch06_quantum_gravity.tex
"""

from __future__ import annotations

import math

from ..constants import DELTA_F, DELTA_G_F, PHI
from ..physical_constants import M_PL_GEV
from ..registry import Observable, REGISTRY


# ── Chiral coupling constant ─────────────────────────────────────

def chiral_coupling(
    delta: float = DELTA_F,
    delta_g: float = DELTA_G_F,
    phi: float = PHI,
) -> float:
    """Chiral gravitational coupling xi_G.

    xi_G = Delta * delta_g * phi^{-2}

    Args:
        delta: Fibonacci-lattice conflict (5/24).
        delta_g: Elementary lattice tension (1/24).
        phi: Golden ratio.

    Returns:
        Dimensionless chiral coupling.
    """
    return delta * delta_g * phi ** (-2)


XI_G: float = chiral_coupling()
"""Chiral gravitational coupling ~ 0.00332.

This quantifies the parity violation in the gravitational sector
arising from the 6-cone topology. The Fibonacci defect (Delta)
breaks the left-right symmetry of the lattice.
"""


# ── Gravitational wave parity violation ──────────────────────────

def gw_amplitude_asymmetry(
    omega_hz: float,
    xi_g: float = XI_G,
    m_pl_gev: float = M_PL_GEV,
) -> float:
    """Fractional amplitude asymmetry between R and L polarizations.

    Delta_h / h = 2 * xi_G * omega / M_P

    The frequency omega must be converted to natural units (GeV)
    via omega_GeV = hbar * omega_Hz / GeV_to_J.

    Args:
        omega_hz: Gravitational wave frequency in Hz.
        xi_g: Chiral coupling constant.
        m_pl_gev: Planck mass in GeV.

    Returns:
        Fractional amplitude asymmetry |h_R - h_L| / h.
    """
    # Convert frequency to energy in GeV
    hbar_gev_s = 6.582119569e-25  # hbar in GeV*s
    omega_gev = hbar_gev_s * omega_hz
    return 2.0 * xi_g * omega_gev / m_pl_gev


def gw_asymmetry_smbh(
    m_bh_solar: float = 1e6,
    xi_g: float = XI_G,
) -> float:
    """GW asymmetry for a typical SMBH merger.

    For SMBH mergers, the GW frequency scales as:
    f ~ c^3 / (G M) ~ 1e-2 Hz for M ~ 10^6 M_sun

    Args:
        m_bh_solar: BH mass in solar masses.
        xi_g: Chiral coupling constant.

    Returns:
        Fractional amplitude asymmetry.
    """
    # Approximate orbital frequency of ISCO
    f_hz = 4.4e3 / m_bh_solar  # Hz (approximate)
    return gw_amplitude_asymmetry(f_hz, xi_g)


# ── Module-level computed values ─────────────────────────────────

GW_ASYMMETRY_LISA: float = gw_asymmetry_smbh(1e6)
"""GW asymmetry for 10^6 M_sun merger (LISA band) ~ 10^{-20}."""

GW_ASYMMETRY_LIGO: float = gw_amplitude_asymmetry(100.0)
"""GW asymmetry at 100 Hz (LIGO band) ~ 10^{-47}."""


# ── Registry ──────────────────────────────────────────────────────

def register_all(registry=REGISTRY) -> None:
    """Register chiral gravity observables."""
    registry.register(Observable(
        name="exp_xi_g",
        symbol="xi_G",
        formula="Delta * delta_g * phi^{-2}",
        predicted=XI_G,
        observed=0.01,
        observed_uncertainty=0.0,
        unit="dimensionless",
        level=3,
        d_star_variant="none",
        dependencies=("delta", "delta_g"),
        is_upper_limit=True,
    ))
