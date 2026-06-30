"""Level 6: Modified Dispersion Relation & Lorentz Violation.

Implements the modified photon dispersion relation first-order scale:
    E^2 = p^2*c^2 * (1 + xi * E / E_Planck)

Where xi is the Lorentz-violating parameter:
    xi = 1 / sqrt(D*) = sqrt(24 / 67) ~ 0.599
"""

import math
from fractions import Fraction
from typing import Callable

from ..constants import DELTA_G_F
from ..physical_constants import M_PL_GEV
from ..registry import Observable, REGISTRY

# Starre geometrische Konstanten
D_STAR_IR = Fraction(67, 24)
XI_LIV = math.sqrt(1.0 / float(D_STAR_IR))  # sqrt(24/67) ~ 0.599238

# Kosmologische Dichteparameter
OMEGA_B_CANONICAL = Fraction(115, 2304)   # ~0.049913
OMEGA_C_CANONICAL = Fraction(600, 2304)   # ~0.260417
ALPHA_M_0 = Fraction(19, 86)              # ~0.220930
MPC_KM = 3.08567758e19                    # 1 Mpc in km


def get_xi_liv() -> float:
    """Liefert den starr fixierten Lorentz-Verletzungs-Parameter xi."""
    return XI_LIV


def standard_simpson_integrate(func: Callable[[float], float], a: float, b: float, n: int = 2000) -> float:
    """Numerischer Simpson-Integrator für Umgebungen ohne scipy."""
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    integration_sum = func(a) + func(b)
    
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            integration_sum += 2.0 * func(x)
        else:
            integration_sum += 4.0 * func(x)
            
    return integration_sum * (h / 3.0)


def compute_time_delay(
    z: float,
    E_gev: float,
    h0: float = 67.36,
    use_horndeski: bool = True,
    running_alpha: bool = True,
) -> float:
    r"""
    Berechnet die kosmische Laufzeitverzögerung (in Sekunden) für ein Photon.
    
    Delta t = xi * (E / E_Planck) * \int_0^z (1+z') / H(z') dz'
    """
    omega_b = float(OMEGA_B_CANONICAL)
    omega_c = float(OMEGA_C_CANONICAL)
    omega_m = omega_b + omega_c
    
    # Strahlungsdichte und Neutrinos
    h = h0 / 100.0
    omega_gamma = 2.4728e-5 / h**2
    n_eff = 3.044
    neutrino_factor = n_eff * (7.0 / 8.0) * (4.0 / 11.0)**(4.0 / 3.0)
    omega_nu = omega_gamma * neutrino_factor
    omega_r = omega_gamma + omega_nu
    omega_de = 1.0 - omega_m - omega_r

    def integrand(zp: float) -> float:
        scale_m = omega_m * (1.0 + zp)**3
        scale_r = omega_r * (1.0 + zp)**4
        w_de = -67.0 / 72.0
        scale_de = omega_de * (1.0 + zp)**(3.0 * (1.0 + w_de))
        
        e_squared = scale_m + scale_r + scale_de
        
        if use_horndeski:
            if running_alpha:
                current_alpha = float(ALPHA_M_0) * (scale_de / (scale_de + scale_m + scale_r))
            else:
                current_alpha = float(ALPHA_M_0)
            
            # G_eff / G
            numerator = (1.0 + current_alpha) * (2.0 + current_alpha)
            denominator = 2.0 * (1.0 - current_alpha / 3.0)
            g_eff = numerator / denominator
            h_squared = e_squared * g_eff
        else:
            h_squared = e_squared
            
        h_zp = math.sqrt(h_squared)
        return (1.0 + zp) / h_zp

    try:
        from scipy.integrate import quad
        val, _ = quad(integrand, 0.0, z)
    except ImportError:
        val = standard_simpson_integrate(integrand, 0.0, z)

    # H0 in s^-1
    h0_s = h0 / MPC_KM
    
    return XI_LIV * (E_gev / M_PL_GEV) * (val / h0_s)


def check_fermi_lat_limit(z: float, E_gev: float, delta_t_limit: float = 0.82) -> bool:
    """Prüft, ob die berechnete Verzögerung unter der Fermi-LAT Schranke liegt."""
    delay = compute_time_delay(z, E_gev)
    return delay < delta_t_limit


# ── Registry ──────────────────────────────────────────────────────

def register_all(registry=REGISTRY) -> None:
    """Registriert LIV-Observablen."""
    registry.register(Observable(
        name="xi_liv",
        symbol="xi",
        formula="1 / sqrt(D*) = sqrt(24/67)",
        predicted=XI_LIV,
        observed=1.0,
        observed_uncertainty=0.2,
        unit="dimensionless",
        level=6,
        d_star_variant="none",
        dependencies=(),
        is_upper_limit=True,
    ))
