"""Fundamental physical constants in SI units.

Centralizes external (non-SDGFT) constants used by multiple modules:
quantum_gravity, black_holes, neutrino_mass, tully_fisher.

Values from CODATA 2018 recommended values.
"""

import math

# ── Fundamental constants ──────────────────────────────────────────

G_N: float = 6.67430e-11
"""Newtonian gravitational constant in m^3 / (kg * s^2)."""

C: float = 2.99792458e8
"""Speed of light in m/s."""

HBAR: float = 1.054571817e-34
"""Reduced Planck constant in J * s."""

K_B: float = 1.380649e-23
"""Boltzmann constant in J / K."""

# ── Planck units ───────────────────────────────────────────────────

M_P: float = math.sqrt(HBAR * C / G_N)
"""Planck mass in kg: sqrt(hbar * c / G_N) ~ 2.176e-8 kg."""

R_P: float = math.sqrt(HBAR * G_N / C ** 3)
"""Planck length in m: sqrt(hbar * G_N / c^3) ~ 1.616e-35 m."""

T_P: float = math.sqrt(HBAR * G_N / C ** 5)
"""Planck time in s: sqrt(hbar * G_N / c^5) ~ 5.39e-44 s."""

E_P: float = M_P * C ** 2
"""Planck energy in J: M_P * c^2 ~ 1.956e9 J."""

K_P: float = 1.0 / R_P
"""Planck momentum scale in 1/m."""

M_PL_GEV: float = 1.2209e19
"""Planck mass in GeV (commonly used in particle physics)."""

# ── Astrophysical scales ──────────────────────────────────────────

R_H: float = 4.4e26
"""Hubble radius in m (c / H_0 for H_0 ~ 67.4 km/s/Mpc)."""

KPC_M: float = 3.0857e19
"""1 kiloparsec in meters."""

M_SUN: float = 1.989e30
"""Solar mass in kg."""

# ── Unit conversions ──────────────────────────────────────────────

GEV_TO_KG: float = 1.78266192e-27
"""1 GeV/c^2 in kg."""

GEV_TO_J: float = 1.602176634e-10
"""1 GeV in Joules."""
